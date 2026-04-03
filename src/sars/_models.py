"""SAR model definitions and fitting routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class SARFit:
    """Result of fitting a single SAR model.

    Attributes
    ----------
    model : str
        Short model identifier matching R sars function suffix (e.g. 'power').
    params : dict[str, float]
        Fitted parameter values keyed by parameter name.
    r_squared : float
        Coefficient of determination in arithmetic space.
    aic : float
        Akaike Information Criterion (normal log-likelihood convention).
    aicc : float
        AIC corrected for small sample size.
    bic : float
        Bayesian Information Criterion.
    n : int
        Number of observations used in fit.
    converged : bool
        Whether the NLS solver converged.
    data : pd.DataFrame
        Original data (columns: area, species).
    """

    model: str
    params: dict
    r_squared: float
    aic: float
    aicc: float
    bic: float
    n: int
    converged: bool
    data: pd.DataFrame

    def predict(self, area: float | np.ndarray) -> np.ndarray:
        """Predict species richness for given area value(s)."""
        area = np.asarray(area, dtype=float)
        if self.model == "power":
            return self.params["c"] * area ** self.params["z"]
        raise NotImplementedError(f"predict not yet implemented for '{self.model}'")

    def __repr__(self) -> str:
        p = "  ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        return (
            f"SARFit(model='{self.model}', {p}, "
            f"R²={self.r_squared:.4f}, AICc={self.aicc:.2f})"
        )


def _compute_ic(k: int, n: int, rss: float) -> tuple[float, float, float]:
    """Compute AIC, AICc, and BIC from residual sum of squares.

    Uses the normal log-likelihood convention consistent with R sars:
        logL = -n/2 * log(2*pi*rss/n) - n/2

    Parameters
    ----------
    k : int
        Number of estimated parameters (including sigma).
    n : int
        Number of observations.
    rss : float
        Residual sum of squares.

    Returns
    -------
    tuple[float, float, float]
        (AIC, AICc, BIC)
    """
    # k already includes sigma in R sars convention
    log_lik = -n / 2.0 * np.log(2.0 * np.pi * rss / n) - n / 2.0
    aic = -2.0 * log_lik + 2.0 * k
    # AICc correction
    if n - k - 1 > 0:
        aicc = aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    else:
        aicc = np.inf
    bic = -2.0 * log_lik + k * np.log(n)
    return aic, aicc, bic


def sar_power(data: pd.DataFrame, grid_start: bool = True) -> SARFit:
    """Fit the power law SAR model: S = c * A^z

    The Arrhenius (1921) power model is the most widely used SAR model.
    Fitted using nonlinear least squares in arithmetic space with a grid
    of starting values to avoid local minima.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' (float, km²) and 'species' (int).
    grid_start : bool
        If True, use a grid of starting values. Recommended for all use.

    Returns
    -------
    SARFit
        Fitted model with params {'c': ..., 'z': ...}.

    References
    ----------
    Arrhenius O (1921) Species and area. Journal of Ecology 9:95-99.
    """
    area = np.asarray(data["area"], dtype=float)
    species = np.asarray(data["species"], dtype=float)
    n = len(area)

    def residuals(p: np.ndarray) -> np.ndarray:
        c, z = p
        return species - c * area**z

    best_cost = np.inf
    best_result = None

    if grid_start:
        # Grid of starting values — log-space OLS gives a good anchor
        log_a = np.log(area)
        log_s = np.log(np.maximum(species, 1e-10))
        slope, intercept = np.polyfit(log_a, log_s, 1)
        c_ols = np.exp(intercept)
        z_ols = slope

        # Build grid around OLS estimate + wide exploration
        c_starts = np.array([c_ols * f for f in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]])
        z_starts = np.array([z_ols * f for f in [0.1, 0.5, 1.0, 1.5, 2.0]])
        c_starts = np.append(c_starts, [1.0, 10.0, 50.0, 100.0])
        z_starts = np.append(z_starts, [0.1, 0.2, 0.3, 0.5, 0.8])
    else:
        c_starts = np.array([1.0, 10.0, 50.0])
        z_starts = np.array([0.1, 0.3, 0.5])

    for c0 in c_starts:
        for z0 in z_starts:
            if c0 <= 0:
                continue
            try:
                result = least_squares(
                    residuals,
                    x0=[c0, z0],
                    bounds=([1e-10, -5.0], [1e6, 5.0]),
                    method="trf",
                    max_nfev=2000,
                )
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
            except Exception:
                continue

    if best_result is None or not best_result.success:
        return SARFit(
            model="power",
            params={"c": np.nan, "z": np.nan},
            r_squared=np.nan,
            aic=np.nan,
            aicc=np.nan,
            bic=np.nan,
            n=n,
            converged=False,
            data=data,
        )

    c_fit, z_fit = best_result.x
    rss = 2.0 * best_result.cost  # least_squares minimises 0.5 * sum(r^2)
    ss_tot = np.sum((species - np.mean(species)) ** 2)
    r_squared = 1.0 - rss / ss_tot

    # k = number of model params + 1 for sigma (R sars convention)
    k = 3  # c, z, sigma
    aic, aicc, bic = _compute_ic(k, n, rss)

    return SARFit(
        model="power",
        params={"c": c_fit, "z": z_fit},
        r_squared=r_squared,
        aic=aic,
        aicc=aicc,
        bic=bic,
        n=n,
        converged=True,
        data=data,
    )
