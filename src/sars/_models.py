"""SAR model definitions and fitting routines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SARFit dataclass
# ---------------------------------------------------------------------------

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
        spec = _MODEL_REGISTRY.get(self.model)
        if spec is None:
            raise NotImplementedError(
                f"predict not yet implemented for '{self.model}'"
            )
        return spec.func(area, *[self.params[k] for k in spec.param_names])

    def __repr__(self) -> str:
        p = "  ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        return (
            f"SARFit(model='{self.model}', {p}, "
            f"R²={self.r_squared:.4f}, AICc={self.aicc:.2f})"
        )


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------

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
    log_lik = -n / 2.0 * np.log(2.0 * np.pi * rss / n) - n / 2.0
    aic = -2.0 * log_lik + 2.0 * k
    if n - k - 1 > 0:
        aicc = aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    else:
        aicc = np.inf
    bic = -2.0 * log_lik + k * np.log(n)
    return aic, aicc, bic


# ---------------------------------------------------------------------------
# Model specification registry
# ---------------------------------------------------------------------------

class _SARFunc(Protocol):
    """Signature for SAR model functions: f(area, *params) -> predicted S."""

    def __call__(self, area: np.ndarray, /, *params: float) -> np.ndarray: ...


@dataclass
class _ModelSpec:
    """Internal specification for a SAR model."""

    name: str
    func: _SARFunc
    param_names: list[str]
    bounds_lower: list[float]
    bounds_upper: list[float]
    start_grid: list[list[float]]  # one list of starting values per param


_MODEL_REGISTRY: dict[str, _ModelSpec] = {}


def _register(spec: _ModelSpec) -> _ModelSpec:
    _MODEL_REGISTRY[spec.name] = spec
    return spec


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_data(data: pd.DataFrame) -> None:
    """Validate a SAR input DataFrame.

    Checks that *data* has the required ``area`` and ``species`` columns,
    contains at least two rows, and that all values are positive and finite.

    Parameters
    ----------
    data : pd.DataFrame

    Raises
    ------
    TypeError
        If *data* is not a DataFrame.
    KeyError
        If required columns are missing.
    ValueError
        If data is empty, contains non-positive values, or non-finite values.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame, got {type(data).__name__}"
        )

    missing = [c for c in ("area", "species") if c not in data.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"DataFrame has columns: {list(data.columns)}"
        )

    if len(data) < 2:
        raise ValueError(
            f"Need at least 2 observations, got {len(data)}"
        )

    area = data["area"]
    species = data["species"]

    if not np.isfinite(area.to_numpy(dtype=float)).all():
        raise ValueError("'area' contains NaN or infinite values")
    if not np.isfinite(species.to_numpy(dtype=float)).all():
        raise ValueError("'species' contains NaN or infinite values")
    if (area.to_numpy(dtype=float) <= 0).any():
        raise ValueError("'area' must contain only positive values")
    if (species.to_numpy(dtype=float) <= 0).any():
        raise ValueError("'species' must contain only positive values")


# ---------------------------------------------------------------------------
# Generic NLS fitter
# ---------------------------------------------------------------------------

def _fit_nls(
    name: str,
    data: pd.DataFrame,
    start_from: dict[str, float] | None = None,
) -> SARFit:
    """Fit a model from the registry using multi-start NLS.

    Parameters
    ----------
    name : str
        Model name (key in _MODEL_REGISTRY).
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    start_from : dict[str, float], optional
        If provided, use these parameter values as the single starting point
        instead of the full grid search. Used by bootstrap to avoid the
        expensive grid search on each resample.

    Returns
    -------
    SARFit
    """
    _validate_data(data)
    spec = _MODEL_REGISTRY[name]
    area = np.asarray(data["area"], dtype=float)
    species = np.asarray(data["species"], dtype=float)
    n = len(area)

    def residuals(p: np.ndarray) -> np.ndarray:
        pred = spec.func(area, *p)
        res = species - pred
        # Replace any NaN/Inf with a large penalty
        return np.where(np.isfinite(res), res, 1e10)

    best_cost = np.inf
    best_result = None

    if start_from is not None:
        expected = set(spec.param_names)
        provided = set(start_from)
        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if extra:
                parts.append(f"unexpected {extra}")
            raise ValueError(
                f"start_from keys don't match model '{name}' "
                f"params {spec.param_names}: {', '.join(parts)}"
            )
        starts = [[start_from[k] for k in spec.param_names]]
    else:
        starts = product(*spec.start_grid)

    # Early-termination: stop after this many consecutive successful solver
    # calls that don't improve the best cost.  For large grids this avoids
    # hundreds of redundant solver calls once the optimum has stabilised.
    # The limit is generous enough to avoid premature stopping.
    n_params = len(spec.param_names)
    stale_limit = 50 * n_params  # 100 for 2-p, 150 for 3-p, 200 for 4-p

    n_starts = 0
    n_errors = 0
    stale_runs = 0
    for start in starts:
        n_starts += 1
        try:
            result = least_squares(
                residuals,
                x0=list(start),
                bounds=(spec.bounds_lower, spec.bounds_upper),
                method="trf",
                max_nfev=5000,
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                stale_runs = 0
            else:
                stale_runs += 1
        except Exception:
            n_errors += 1
            logger.debug(
                "Model '%s' NLS failed for start %s", name, list(start),
                exc_info=True,
            )

        if stale_runs >= stale_limit and best_result is not None:
            logger.debug(
                "Model '%s': early stop after %d starts "
                "(%d without improvement)",
                name, n_starts, stale_limit,
            )
            break

        if n_starts % 100 == 0:
            logger.debug(
                "Model '%s': %d starts evaluated (best cost=%.4g)",
                name, n_starts, best_cost,
            )

    if n_errors:
        logger.info(
            "Model '%s': %d/%d starting points raised exceptions",
            name, n_errors, n_starts,
        )

    if best_result is None:
        return _failed_fit(name, spec.param_names, n, data)

    rss = 2.0 * best_result.cost
    ss_tot = np.sum((species - np.mean(species)) ** 2)
    r_squared = 1.0 - rss / ss_tot

    k = len(spec.param_names) + 1  # +1 for sigma
    aic, aicc, bic = _compute_ic(k, n, rss)
    params = dict(zip(spec.param_names, best_result.x))

    return SARFit(
        model=name,
        params=params,
        r_squared=r_squared,
        aic=aic,
        aicc=aicc,
        bic=bic,
        n=n,
        converged=True,
        data=data,
    )


def _failed_fit(
    name: str, param_names: list[str], n: int, data: pd.DataFrame
) -> SARFit:
    return SARFit(
        model=name,
        params={k: np.nan for k in param_names},
        r_squared=np.nan,
        aic=np.nan,
        aicc=np.nan,
        bic=np.nan,
        n=n,
        converged=False,
        data=data,
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# --- Non-asymptotic models (8) ---

# 1. power: S = c * A^z
_register(_ModelSpec(
    name="power",
    func=lambda a, c, z: c * a**z,
    param_names=["c", "z"],
    bounds_lower=[1e-10, -5.0],
    bounds_upper=[1e6, 5.0],
    start_grid=[
        [1.0, 5.0, 10.0, 30.0, 50.0, 100.0],
        [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    ],
))

# 2. powerR: S = c * A^z + d
_register(_ModelSpec(
    name="powerR",
    func=lambda a, f, c, z: f + c * a**z,
    param_names=["f", "c", "z"],
    bounds_lower=[-1e6, 1e-10, -5.0],
    bounds_upper=[1e6, 1e6, 5.0],
    start_grid=[
        [-100.0, -10.0, 0.0, 10.0],
        [1.0, 10.0, 50.0, 100.0],
        [0.05, 0.15, 0.3, 0.5, 1.0],
    ],
))

# 3. loga: S = c + z * log(A)
_register(_ModelSpec(
    name="loga",
    func=lambda a, c, z: c + z * np.log(a),
    param_names=["c", "z"],
    bounds_lower=[-1e6, -1e6],
    bounds_upper=[1e6, 1e6],
    start_grid=[
        [-10.0, 0.0, 0.3, 1.0, 10.0, 50.0],
        [1.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0],
    ],
))

# 4. linear: S = c + m * A  (OLS)
_register(_ModelSpec(
    name="linear",
    func=lambda a, c, m: c + m * a,
    param_names=["c", "m"],
    bounds_lower=[-1e6, -1e6],
    bounds_upper=[1e6, 1e6],
    start_grid=[
        [0.0, 10.0, 50.0, 70.0, 100.0],
        [0.01, 0.1, 0.2, 0.5, 1.0],
    ],
))

# 5. epm1: log(S) = log(c) + z1*log(A) + z2*(log(A))^2
#    => S = c * A^z1 * exp(z2 * (log(A))^2)
#    But R fits in log-space: we fit in arithmetic space for consistency
#    with R sars output (R² and AIC are computed in arithmetic space).
def _epm1_func(a, c, z, d):
    log_a = np.log(a)
    return c * a**z * np.exp(d * log_a**2)

_register(_ModelSpec(
    name="epm1",
    func=_epm1_func,
    param_names=["c", "z", "d"],
    bounds_lower=[1e-10, -10.0, -5.0],
    bounds_upper=[1e6, 10.0, 5.0],
    start_grid=[
        [0.01, 0.1, 1.0, 10.0],
        [0.5, 1.0, 2.0, 3.5, 5.0],
        [-0.5, -0.1, 0.0, 0.1, 0.17, 0.5],
    ],
))

# 6. epm2: S = c * A^(z1 * A^z2)
def _epm2_func(a, c, z1, z2):
    return c * a ** (z1 * a**z2)

_register(_ModelSpec(
    name="epm2",
    func=_epm2_func,
    param_names=["c", "z1", "z2"],
    bounds_lower=[1e-10, -10.0, -10.0],
    bounds_upper=[1e6, 10.0, 10.0],
    start_grid=[
        [1.0, 10.0, 36.0, 50.0, 100.0],
        [0.05, 0.1, 0.27, 0.5, 1.0],
        [-1.0, -0.5, 0.0, 0.5, 1.0],
    ],
))

# 7. p1: S = c * A^z * exp(-d * A)  (Persistence function 1)
def _p1_func(a, c, z, d):
    return c * a**z * np.exp(-d * a)

_register(_ModelSpec(
    name="p1",
    func=_p1_func,
    param_names=["c", "z", "d"],
    bounds_lower=[1e-10, -5.0, 0.0],
    bounds_upper=[1e6, 5.0, 10.0],
    start_grid=[
        [1.0, 5.0, 8.6, 20.0, 50.0],
        [0.1, 0.3, 0.5, 0.67, 1.0],
        [0.0001, 0.001, 0.002, 0.01, 0.1],
    ],
))

# 8. p2: S = c * A^z * exp(-d / A)  (Persistence function 2)
def _p2_func(a, c, z, d):
    return c * a**z * np.exp(-d / a)

_register(_ModelSpec(
    name="p2",
    func=_p2_func,
    param_names=["c", "z", "d"],
    bounds_lower=[1e-10, -5.0, 0.0],
    bounds_upper=[1e6, 5.0, 1e4],
    start_grid=[
        [10.0, 50.0, 100.0, 195.0, 500.0],
        [0.001, 0.01, 0.05, 0.1, 0.5],
        [1.0, 5.0, 10.0, 25.0, 50.0],
    ],
))

# --- Asymptotic convex models (5) ---

# 9. koba: S = c * log(1 + A/z)
_register(_ModelSpec(
    name="koba",
    func=lambda a, c, z: c * np.log(1.0 + a / z),
    param_names=["c", "z"],
    bounds_lower=[1e-10, 1e-10],
    bounds_upper=[1e6, 1e6],
    start_grid=[
        [5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        [0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 50.0],
    ],
))

# 10. monod: S = d * A / (c + A)
_register(_ModelSpec(
    name="monod",
    func=lambda a, d, c: d * a / (c + a),
    param_names=["d", "c"],
    bounds_lower=[1e-10, 1e-10],
    bounds_upper=[1e6, 1e6],
    start_grid=[
        [100.0, 150.0, 200.0, 222.0, 300.0, 500.0],
        [5.0, 10.0, 20.0, 47.0, 100.0, 200.0],
    ],
))

# 11. negexpo: S = d * (1 - exp(-z * A))
_register(_ModelSpec(
    name="negexpo",
    func=lambda a, d, z: d * (1.0 - np.exp(-z * a)),
    param_names=["d", "z"],
    bounds_lower=[1e-10, 1e-10],
    bounds_upper=[1e6, 100.0],
    start_grid=[
        [100.0, 150.0, 200.0, 208.0, 300.0, 500.0],
        [0.001, 0.005, 0.01, 0.014, 0.05, 0.1],
    ],
))

# 12. asymp: S = d - c * exp(-z * A)
_register(_ModelSpec(
    name="asymp",
    func=lambda a, d, c, z: d - c * np.exp(-z * a),
    param_names=["d", "c", "z"],
    bounds_lower=[1e-10, 1e-10, 1e-10],
    bounds_upper=[1e6, 1e6, 100.0],
    start_grid=[
        [150.0, 200.0, 209.0, 300.0],
        [100.0, 150.0, 185.0, 300.0],
        [0.001, 0.005, 0.01, 0.989, 1.0],
    ],
))

# 13. ratio: S = (c + z * A) / (1 + d * A)
_register(_ModelSpec(
    name="ratio",
    func=lambda a, c, z, d: (c + z * a) / (1.0 + d * a),
    param_names=["c", "z", "d"],
    bounds_lower=[-1e6, 1e-10, 1e-10],
    bounds_upper=[1e6, 1e6, 100.0],
    start_grid=[
        [-10.0, 0.0, 10.0, 20.0, 50.0],
        [0.5, 1.0, 3.5, 5.0, 10.0],
        [0.001, 0.005, 0.015, 0.05, 0.1],
    ],
))

# --- Asymptotic sigmoid models (7) ---

# 14. mmf: S = d / (1 + c * A^(-z))
_register(_ModelSpec(
    name="mmf",
    func=lambda a, d, c, z: d / (1.0 + c * a**(-z)),
    param_names=["d", "c", "z"],
    bounds_lower=[1e-10, 1e-10, 1e-10],
    bounds_upper=[1e6, 1e6, 10.0],
    start_grid=[
        [100.0, 150.0, 223.0, 300.0, 500.0],
        [10.0, 30.0, 45.0, 80.0, 150.0],
        [0.1, 0.5, 0.99, 1.5, 2.0],
    ],
))

# 15. gompertz: S = d * exp(-exp(-z * (A - c)))
_register(_ModelSpec(
    name="gompertz",
    func=lambda a, d, z, c: d * np.exp(-np.exp(-z * (a - c))),
    param_names=["d", "z", "c"],
    bounds_lower=[1e-10, 1e-10, -1e6],
    bounds_upper=[1e6, 100.0, 1e6],
    start_grid=[
        [150.0, 200.0, 211.0, 300.0, 500.0],
        [0.001, 0.005, 0.01, 0.017, 0.05, 0.1],
        [0.0, 10.0, 38.0, 50.0, 100.0],
    ],
))

# 16. weibull3: S = d * (1 - exp(-c * A^z))
def _weibull3_func(a, d, c, z):
    return d * (1.0 - np.exp(-c * a**z))

_register(_ModelSpec(
    name="weibull3",
    func=_weibull3_func,
    param_names=["d", "c", "z"],
    bounds_lower=[1e-10, 1e-10, 1e-10],
    bounds_upper=[1e6, 100.0, 10.0],
    start_grid=[
        [150.0, 200.0, 208.0, 300.0, 500.0],
        [0.001, 0.01, 0.029, 0.1, 0.5],
        [0.1, 0.5, 0.83, 1.0, 1.5],
    ],
))

# 17. weibull4: S = d * (1 - exp(-c * A^z))^f
def _weibull4_func(a, d, c, z, f):
    inner = 1.0 - np.exp(-c * a**z)
    return d * np.sign(inner) * np.abs(inner)**f

_register(_ModelSpec(
    name="weibull4",
    func=_weibull4_func,
    param_names=["d", "c", "z", "f"],
    bounds_lower=[1e-10, 0.0, 1e-10, 1e-10],
    bounds_upper=[1e6, 100.0, 10.0, 100.0],
    start_grid=[
        [150.0, 200.0, 210.0, 300.0],
        [0.0, 0.001, 0.01, 0.1, 0.5],
        [0.5, 1.0, 3.0, 6.4, 8.0],
        [0.01, 0.05, 0.089, 0.5, 1.0, 2.0],
    ],
))

# 18. chapman: S = d * (1 - exp(-z * A))^c
def _chapman_func(a, d, z, c):
    inner = 1.0 - np.exp(-z * a)
    return d * np.sign(inner) * np.abs(inner)**c

_register(_ModelSpec(
    name="chapman",
    func=_chapman_func,
    param_names=["d", "z", "c"],
    bounds_lower=[1e-10, 1e-10, 1e-10],
    bounds_upper=[1e6, 100.0, 100.0],
    start_grid=[
        [150.0, 200.0, 208.0, 300.0, 500.0],
        [0.001, 0.005, 0.01, 0.05, 0.1],
        [0.1, 0.5, 0.68, 1.0, 2.0],
    ],
))

# 19. betap: S = d * (1 - (1 + (A/c)^z)^(-f))
def _betap_func(a, d, c, z, f):
    return d * (1.0 - (1.0 + (a / c)**z)**(-f))

_register(_ModelSpec(
    name="betap",
    func=_betap_func,
    param_names=["d", "c", "z", "f"],
    bounds_lower=[1e-10, 1e-10, 1e-10, 1e-10],
    bounds_upper=[1e6, 1e6, 10.0, 100.0],
    start_grid=[
        [150.0, 200.0, 208.0, 300.0],
        [50.0, 100.0, 378.0, 500.0, 1000.0],
        [0.1, 0.5, 0.9, 1.5, 2.0],
        [0.5, 1.0, 2.0, 5.2, 10.0],
    ],
))

# 20. heleg: S = c / (f + A^(-z))
#     R sars params: c, f, z (all positive). Asymptote = c / f.
#     R reference: c=4.95781, f=0.022238, z=0.988229
def _heleg_func(a, c, f, z):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = c / (f + a**(-z))
    return np.where(np.isfinite(result), result, 0.0)

_register(_ModelSpec(
    name="heleg",
    func=_heleg_func,
    param_names=["c", "f", "z"],
    bounds_lower=[1e-10, 1e-10, 1e-10],
    bounds_upper=[1e6, 1e6, 1e6],
    start_grid=[
        [1.0, 5.0, 10.0, 50.0, 100.0],
        [0.001, 0.01, 0.022, 0.05, 0.1],
        [0.1, 0.5, 0.99, 2.0, 5.0],
    ],
))


# ---------------------------------------------------------------------------
# Public API — one function per model
# ---------------------------------------------------------------------------

def sar_power(data: pd.DataFrame) -> SARFit:
    """Fit the power law SAR model: S = c * A^z

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit

    References
    ----------
    Arrhenius O (1921) Species and area. Journal of Ecology 9:95-99.
    """
    return _fit_nls("power", data)


def sar_powerR(data: pd.DataFrame) -> SARFit:  # noqa: N802
    """Fit the power-R SAR model: S = f + c * A^z

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("powerR", data)


def sar_loga(data: pd.DataFrame) -> SARFit:
    """Fit the logarithmic SAR model: S = c + z * log(A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit

    References
    ----------
    Gleason HA (1922) On the relation between species and area.
    Ecology 3:158-162.
    """
    return _fit_nls("loga", data)


def sar_linear(data: pd.DataFrame) -> SARFit:
    """Fit the linear SAR model: S = c + m * A

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("linear", data)


def sar_epm1(data: pd.DataFrame) -> SARFit:
    """Fit the extended power model 1: S = c * A^z * exp(d * (log A)^2)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("epm1", data)


def sar_epm2(data: pd.DataFrame) -> SARFit:
    """Fit the extended power model 2: S = c * A^(z1 * A^z2)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("epm2", data)


def sar_p1(data: pd.DataFrame) -> SARFit:
    """Fit persistence function 1: S = c * A^z * exp(-d * A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("p1", data)


def sar_p2(data: pd.DataFrame) -> SARFit:
    """Fit persistence function 2: S = c * A^z * exp(-d / A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("p2", data)


def sar_koba(data: pd.DataFrame) -> SARFit:
    """Fit the Kobayashi logarithmic model: S = c * log(1 + A/z)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("koba", data)


def sar_monod(data: pd.DataFrame) -> SARFit:
    """Fit the Monod model: S = d * A / (c + A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("monod", data)


def sar_negexpo(data: pd.DataFrame) -> SARFit:
    """Fit the negative exponential model: S = d * (1 - exp(-z * A))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("negexpo", data)


def sar_asymp(data: pd.DataFrame) -> SARFit:
    """Fit the asymptotic model: S = d - c * exp(-z * A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("asymp", data)


def sar_ratio(data: pd.DataFrame) -> SARFit:
    """Fit the rational function model: S = (c + z * A) / (1 + d * A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("ratio", data)


def sar_mmf(data: pd.DataFrame) -> SARFit:
    """Fit the Morgan-Mercer-Flodin model: S = d / (1 + c * A^(-z))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("mmf", data)


def sar_gompertz(data: pd.DataFrame) -> SARFit:
    """Fit the Gompertz model: S = d * exp(-exp(-z * (A - c)))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("gompertz", data)


def sar_weibull3(data: pd.DataFrame) -> SARFit:
    """Fit the 3-parameter Weibull model: S = d * (1 - exp(-c * A^z))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("weibull3", data)


def sar_weibull4(data: pd.DataFrame) -> SARFit:
    """Fit the 4-parameter Weibull model: S = d * (1 - exp(-c * A^z))^f

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("weibull4", data)


def sar_chapman(data: pd.DataFrame) -> SARFit:
    """Fit the Chapman-Richards model: S = d * (1 - exp(-z * A))^c

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("chapman", data)


def sar_betap(data: pd.DataFrame) -> SARFit:
    """Fit the Beta-P model: S = d * (1 - (1 + (A/c)^z)^(-f))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("betap", data)


def sar_heleg(data: pd.DataFrame) -> SARFit:
    """Fit the Heleg model: S = c / (f + A^(-z))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("heleg", data)
