"""Multi-model inference, averaging, and bootstrap CI."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sars._models import (
    _MODEL_REGISTRY,
    SARFit,
    _fit_nls,
)

# All 20 model names in canonical order
ALL_MODEL_NAMES: list[str] = [
    "power", "powerR", "epm1", "epm2", "p1", "p2", "loga", "koba",
    "mmf", "monod", "negexpo", "chapman", "weibull3", "asymp",
    "ratio", "gompertz", "weibull4", "betap", "heleg", "linear",
]

# Static shape classification for each model.
#
# NOTE: This is a simplification. The R sars package classifies shape
# dynamically based on fitted parameter values — a model can exhibit
# convex, sigmoid, or other curvature depending on the data. Here we
# assign each model its *typical* shape class. In particular, several
# asymptotic models listed as "convex" (koba, monod, negexpo, asymp,
# ratio) can produce sigmoid curves for certain parameter combinations,
# and some sigmoid models can appear convex. Users needing the dynamic
# classification from R should inspect the fitted curve directly.
_MODEL_SHAPE: dict[str, str] = {
    "power": "convex", "powerR": "convex", "loga": "convex",
    "linear": "linear", "epm1": "convex", "epm2": "convex",
    "p1": "convex", "p2": "convex",
    "koba": "convex", "monod": "convex", "negexpo": "convex",
    "asymp": "convex", "ratio": "convex",
    "mmf": "sigmoid", "gompertz": "sigmoid", "weibull3": "sigmoid",
    "weibull4": "sigmoid", "chapman": "sigmoid", "betap": "sigmoid",
    "heleg": "sigmoid",
}

# Models that have an asymptote parameter (d)
_ASYMPTOTE_MODELS: set[str] = {
    "monod", "negexpo", "asymp", "ratio", "mmf", "gompertz",
    "weibull3", "weibull4", "chapman", "betap", "heleg",
}


def _get_asymptote(fit: SARFit) -> float | None:
    """Extract the asymptote value from a fitted model, if applicable."""
    if fit.model not in _ASYMPTOTE_MODELS:
        return None
    # For ratio model, asymptote is z/d
    if fit.model == "ratio":
        d = fit.params.get("d", 0.0)
        z = fit.params.get("z", 0.0)
        return z / d if d != 0 else np.inf
    # For heleg, asymptote is c/f (as A -> inf, A^(-z) -> 0)
    if fit.model == "heleg":
        f = fit.params.get("f", 0.0)
        c = fit.params.get("c", 0.0)
        return c / f if f != 0 else np.inf
    # For all others, the asymptote is 'd'
    return fit.params.get("d", np.nan)


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def aicc(k: int, n: int, rss: float) -> float:
    """Compute AICc from residual sum of squares.

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
    float
        AICc value.
    """
    log_lik = -n / 2.0 * np.log(2.0 * np.pi * rss / n) - n / 2.0
    aic = -2.0 * log_lik + 2.0 * k
    if n - k - 1 > 0:
        return aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    return np.inf


def akaike_weights(ic_values: np.ndarray) -> np.ndarray:
    """Compute Akaike weights from a vector of information criterion values.

    The weighting formula ``w_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta))``
    applies identically to AIC, AICc, and BIC (Burnham & Anderson 2002).

    Parameters
    ----------
    ic_values : array-like
        Information criterion values (AIC, AICc, or BIC) for each candidate
        model.

    Returns
    -------
    np.ndarray
        Model weights (sum to 1). Non-finite inputs produce 0 weight.
    """
    vals = np.asarray(ic_values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.zeros_like(vals)
    min_ic = np.min(vals[finite])
    delta = vals - min_ic
    # Non-finite entries get zero weight
    raw = np.where(finite, np.exp(-0.5 * delta), 0.0)
    total = np.sum(raw)
    if total == 0:
        return np.zeros_like(vals)
    return raw / total


# ---------------------------------------------------------------------------
# MultiSARFit
# ---------------------------------------------------------------------------

@dataclass
class MultiSARFit:
    """Result of fitting multiple SAR models to one dataset.

    Attributes
    ----------
    fits : list[SARFit]
        Individual model fits (converged only).
    data : pd.DataFrame
        Original data.
    summary : pd.DataFrame
        Summary table with columns: model, R2, AIC, AICc, BIC, delta_AICc,
        weight, shape, asymptote. Sorted by AICc ascending. The ``shape``
        column is a static classification based on each model's typical
        curvature (see ``_MODEL_SHAPE``), not a dynamic assessment of the
        fitted curve.
    """

    fits: list[SARFit]
    data: pd.DataFrame
    summary: pd.DataFrame = field(repr=False)

    def __repr__(self) -> str:
        n = len(self.fits)
        best = self.summary.iloc[0]["model"] if len(self.summary) > 0 else "?"
        return f"MultiSARFit({n} models, best='{best}')"


def sar_multi(
    data: pd.DataFrame,
    models: str | list[str] = "all",
) -> MultiSARFit:
    """Fit multiple SAR models to a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    models : str or list[str]
        Model names to fit. Use "all" for all 20 models.

    Returns
    -------
    MultiSARFit
        Container with all converged fits and a summary table.
    """
    if models == "all":
        model_names = ALL_MODEL_NAMES
    else:
        model_names = list(models)

    fits: list[SARFit] = []
    for name in model_names:
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name!r}")
        try:
            fit = _fit_nls(name, data)
        except ValueError:
            continue
        if fit.converged:
            fits.append(fit)

    summary = _build_summary(fits)
    return MultiSARFit(fits=fits, data=data, summary=summary)


def _build_summary(fits: list[SARFit]) -> pd.DataFrame:
    """Build summary DataFrame from a list of SARFit objects."""
    if not fits:
        return pd.DataFrame(
            columns=["model", "R2", "AIC", "AICc", "BIC",
                     "delta_AICc", "weight", "shape", "asymptote"]
        )

    rows = []
    aicc_vals = []
    for fit in fits:
        asym = _get_asymptote(fit)
        rows.append({
            "model": fit.model,
            "R2": fit.r_squared,
            "AIC": fit.aic,
            "AICc": fit.aicc,
            "BIC": fit.bic,
            "shape": _MODEL_SHAPE.get(fit.model, "unknown"),
            "asymptote": asym,
        })
        aicc_vals.append(fit.aicc)

    df = pd.DataFrame(rows)
    weights = akaike_weights(np.array(aicc_vals))
    min_aicc = np.nanmin(aicc_vals)
    df["delta_AICc"] = df["AICc"] - min_aicc
    df["weight"] = weights
    df = df.sort_values("AICc").reset_index(drop=True)
    return df[["model", "R2", "AIC", "AICc", "BIC",
               "delta_AICc", "weight", "shape", "asymptote"]]


# ---------------------------------------------------------------------------
# AveragedSAR
# ---------------------------------------------------------------------------

@dataclass
class AveragedSAR:
    """Model-averaged species-area relationship.

    Attributes
    ----------
    multi : MultiSARFit
        The underlying multi-model fit.
    ic : str
        Information criterion used for weighting ("AICc", "AIC", or "BIC").
    weights : dict[str, float]
        Model weights keyed by model name.
    """

    multi: MultiSARFit
    ic: str
    weights: dict[str, float]

    def predict(self, area: float | np.ndarray) -> np.ndarray:
        """Weighted-average prediction across models.

        Parameters
        ----------
        area : float or array-like
            Area value(s) to predict for.

        Returns
        -------
        np.ndarray
            Model-averaged predicted species richness.
        """
        area = np.asarray(area, dtype=float)
        pred = np.zeros_like(area)
        for fit in self.multi.fits:
            w = self.weights.get(fit.model, 0.0)
            if w > 0:
                pred += w * fit.predict(area)
        return pred

    def __repr__(self) -> str:
        n = len(self.weights)
        return f"AveragedSAR({n} models, ic='{self.ic}')"


_VALID_ICS = {"AIC", "AICc", "BIC"}


def sar_average(
    data: pd.DataFrame,
    models: str | list[str] = "all",
    ic: str = "AICc",
) -> AveragedSAR:
    """Compute model-averaged SAR predictions using information-theoretic weights.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    models : str or list[str]
        Model names to fit. Use "all" for all 20 models.
    ic : str
        Information criterion for weighting: ``"AICc"`` (default), ``"AIC"``,
        or ``"BIC"``. AICc may be infinite when the sample size is small
        relative to the number of parameters; in that case ``"AIC"`` or
        ``"BIC"`` can be used instead.

    Returns
    -------
    AveragedSAR
        Model-averaged SAR with weighted predictions.
    """
    if ic not in _VALID_ICS:
        raise ValueError(
            f"Unsupported IC: {ic!r}. Choose from {sorted(_VALID_ICS)}."
        )

    multi = sar_multi(data, models=models)
    weights_arr = akaike_weights(multi.summary[ic].to_numpy(dtype=float))
    weights = dict(zip(multi.summary["model"], weights_arr))
    return AveragedSAR(multi=multi, ic=ic, weights=weights)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

@dataclass
class BootstrappedCI:
    """Bootstrap confidence intervals for model-averaged predictions.

    Attributes
    ----------
    area_grid : np.ndarray
        Area values at which predictions were made.
    mean : np.ndarray
        Mean prediction across bootstrap replicates.
    lower : np.ndarray
        Lower confidence bound.
    upper : np.ndarray
        Upper confidence bound.
    conf : float
        Confidence level (e.g. 0.95).
    n_boot : int
        Number of bootstrap replicates completed.
    convergence_counts : list[int]
        Number of models that converged in each successful replicate.
        Length equals ``n_boot``. Empty when no replicates succeeded.
    n_models_attempted : int
        Number of models attempted per replicate (i.e. the number that
        converged on the original data).
    """

    area_grid: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    conf: float
    n_boot: int
    convergence_counts: list[int] = field(default_factory=list, repr=False)
    n_models_attempted: int = 0

    def __repr__(self) -> str:
        parts = [
            f"n_boot={self.n_boot}",
            f"conf={self.conf}",
            f"points={len(self.area_grid)}",
            f"n_models_attempted={self.n_models_attempted}",
        ]
        if self.convergence_counts:
            rate = np.mean(self.convergence_counts) / self.n_models_attempted
            parts.append(f"mean_convergence_rate={rate:.0%}")
        return f"BootstrappedCI({', '.join(parts)})"


def _sar_multi_fast(
    data: pd.DataFrame,
    warm_starts: dict[str, dict[str, float]],
) -> MultiSARFit:
    """Fit multiple SAR models using warm-start parameters (no grid search).

    Used internally by bootstrap to avoid the expensive full grid search
    on each resample. Each model is fitted from a single starting point
    derived from the original data fit.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    warm_starts : dict[str, dict[str, float]]
        Map of model name -> parameter dict from the original fit.

    Returns
    -------
    MultiSARFit
    """
    fits: list[SARFit] = []
    for name, params in warm_starts.items():
        fit = _fit_nls(name, data, start_from=params)
        if fit.converged:
            fits.append(fit)

    summary = _build_summary(fits)
    return MultiSARFit(fits=fits, data=data, summary=summary)


def bootstrap_ci(
    data: pd.DataFrame,
    models: str | list[str] = "all",
    n_boot: int = 1000,
    conf: float = 0.95,
    area_grid: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    method: str = "fast",
    ic: str = "AICc",
) -> BootstrappedCI:
    """Compute bootstrap confidence intervals for model-averaged predictions.

    Resamples rows of the data with replacement, fits all models, computes
    model-averaged predictions at each area value, then returns percentile-based
    confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    models : str or list[str]
        Model names to use. "all" for all 20 models.
    n_boot : int
        Number of bootstrap replicates.
    conf : float
        Confidence level (default 0.95).
    area_grid : np.ndarray, optional
        Area values at which to predict. If None, uses 100 points spanning
        the range of the data.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    method : str
        Bootstrap fitting strategy. ``"fast"`` (default) fits all models once
        on the original data with full grid search, then uses those converged
        parameters as warm starts for each resample (~2000x faster).
        ``"full"`` runs the complete grid search on every resample (slow but
        maximally robust).
    ic : str
        Information criterion for weighting: ``"AICc"`` (default), ``"AIC"``,
        or ``"BIC"``.

    Returns
    -------
    BootstrappedCI
    """
    if method not in ("fast", "full"):
        raise ValueError(f"method must be 'fast' or 'full', got {method!r}")
    if ic not in _VALID_ICS:
        raise ValueError(
            f"Unsupported IC: {ic!r}. Choose from {sorted(_VALID_ICS)}."
        )

    if rng is None:
        rng = np.random.default_rng()

    if area_grid is None:
        a_min = data["area"].min()
        a_max = data["area"].max()
        area_grid = np.linspace(a_min, a_max, 100)

    # For "fast" mode, do one full grid-search fit to get warm-start params
    warm_starts: dict[str, dict[str, float]] | None = None
    if method == "fast":
        base_multi = sar_multi(data, models=models)
        warm_starts = {
            fit.model: dict(fit.params) for fit in base_multi.fits
        }
        if not warm_starts:
            nan_arr = np.full_like(area_grid, np.nan)
            return BootstrappedCI(
                area_grid=area_grid, mean=nan_arr, lower=nan_arr,
                upper=nan_arr, conf=conf, n_boot=0,
            )

    if warm_starts is not None:
        n_attempted = len(warm_starts)
    else:
        n_attempted = len(ALL_MODEL_NAMES if models == "all" else list(models))
    n = len(data)
    alpha = 1.0 - conf
    preds = []
    convergence_counts: list[int] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_data = data.iloc[idx].reset_index(drop=True)
        try:
            if method == "fast":
                multi = _sar_multi_fast(boot_data, warm_starts)
                n_converged = len(multi.fits)
                if not multi.fits:
                    continue
                w_arr = akaike_weights(
                    multi.summary[ic].to_numpy(dtype=float)
                )
                weights = dict(zip(multi.summary["model"], w_arr))
                avg = AveragedSAR(multi=multi, ic=ic, weights=weights)
            else:
                avg = sar_average(boot_data, models=models, ic=ic)
                n_converged = len(avg.multi.fits)
            pred = avg.predict(area_grid)
            if np.all(np.isfinite(pred)):
                preds.append(pred)
                convergence_counts.append(n_converged)
        except Exception:
            continue

    if not preds:
        nan_arr = np.full_like(area_grid, np.nan)
        return BootstrappedCI(
            area_grid=area_grid, mean=nan_arr, lower=nan_arr,
            upper=nan_arr, conf=conf, n_boot=0,
            convergence_counts=[], n_models_attempted=n_attempted,
        )

    # Warn if convergence rate across resamples was poor
    if n_attempted > 0:
        if convergence_counts:
            mean_converged = np.mean(convergence_counts)
            rate = mean_converged / n_attempted
            if rate < 0.5:
                warnings.warn(
                    f"Bootstrap convergence rate is low: on average "
                    f"{mean_converged:.1f}/{n_attempted} models converged "
                    f"per resample ({rate:.0%}). CI estimates may be "
                    f"unreliable.",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"No bootstrap resamples produced valid predictions "
                f"({n_attempted} models attempted per resample). "
                f"CI estimates are not available.",
                stacklevel=2,
            )

    preds_arr = np.array(preds)
    return BootstrappedCI(
        area_grid=area_grid,
        mean=np.mean(preds_arr, axis=0),
        lower=np.percentile(preds_arr, 100 * alpha / 2, axis=0),
        upper=np.percentile(preds_arr, 100 * (1 - alpha / 2), axis=0),
        conf=conf,
        n_boot=len(preds),
        convergence_counts=convergence_counts,
        n_models_attempted=n_attempted,
    )
