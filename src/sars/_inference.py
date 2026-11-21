"""Multi-model inference, averaging, and bootstrap CI."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sars._checks import CheckOptions
from sars._models import (
    _MODEL_REGISTRY,
    SARFit,
    _area_grid,
    _fit_nls,
    _validate_data,
)

# All 20 model names in canonical order
ALL_MODEL_NAMES: list[str] = [
    "power",
    "powerR",
    "epm1",
    "epm2",
    "p1",
    "p2",
    "loga",
    "koba",
    "mmf",
    "monod",
    "negexpo",
    "chapman",
    "weibull3",
    "asymp",
    "ratio",
    "gompertz",
    "weibull4",
    "betap",
    "heleg",
    "linear",
]

# Each model's general shape, as classified by Matthews et al. (2019, Table 1)
# with the corrections R sars has made since: "convex/sigmoid" where the
# parameters decide which. It describes the model, not the fitted curve; a
# convex model's fit can still look nearly linear over the observed range.
_MODEL_SHAPE: dict[str, str] = {
    "power": "convex",
    "powerR": "convex",
    "loga": "convex",
    "linear": "linear",
    "epm1": "convex/sigmoid",
    "epm2": "sigmoid",
    "p1": "convex/sigmoid",
    "p2": "sigmoid",
    "koba": "convex",
    "monod": "convex",
    "negexpo": "convex",
    "asymp": "convex/sigmoid",
    "ratio": "convex",
    "mmf": "convex/sigmoid",
    "gompertz": "sigmoid",
    "weibull3": "sigmoid",
    "weibull4": "sigmoid",
    "chapman": "sigmoid",
    "betap": "sigmoid",
    "heleg": "convex/sigmoid",
}

# Models that have an asymptote parameter (d)
_ASYMPTOTE_MODELS: set[str] = {
    "monod",
    "negexpo",
    "asymp",
    "ratio",
    "mmf",
    "gompertz",
    "weibull3",
    "weibull4",
    "chapman",
    "betap",
    "heleg",
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
        Model weights (sum to 1). A non-finite input gets 0 weight. If no
        input is finite the weights are undefined and every entry is NaN,
        so a downstream average fails loudly instead of predicting zero.
    """
    vals = np.asarray(ic_values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.full_like(vals, np.nan)
    delta = vals - np.min(vals[finite])
    # The best model contributes exp(0) = 1, so the total is never zero.
    raw = np.where(finite, np.exp(-0.5 * delta), 0.0)
    return raw / np.sum(raw)


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
        weight, shape, asymptote, observed_shape. Sorted by AICc ascending.
        ``shape`` is the model's general shape as classified by Matthews et al.
        (2019, Table 1): "convex", "sigmoid", "convex/sigmoid" (the parameters
        decide) or "linear". It describes the model, not the fitted curve.
        ``observed_shape`` is the shape the fitted curve takes over the
        observed areas: "convex", "sigmoid", "linear" or "none" (see
        ``SARFit.observed_shape``).
    excluded : dict[str, str]
        Models that converged but were left out because they failed a
        residual check that was asked for, keyed by model name, with the
        check failed and its numbers as the value. Empty when no checks were
        asked for. A model that fails several checks is listed under the
        first, in the order normality, homoscedasticity, negative fitted
        values.
    """

    fits: list[SARFit]
    data: pd.DataFrame
    summary: pd.DataFrame = field(repr=False)
    excluded: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        n = len(self.fits)
        best = self.summary.iloc[0]["model"] if len(self.summary) > 0 else "?"
        return f"MultiSARFit({n} models, best='{best}')"


def _exclude_failing(
    fits: list[SARFit],
    checks: CheckOptions,
) -> tuple[list[SARFit], dict[str, str]]:
    """Split fits into those that pass the residual checks asked for and the
    reasons the others fail."""
    if not checks.requested:
        return fits, {}
    kept: list[SARFit] = []
    excluded: dict[str, str] = {}
    for fit in fits:
        reason = fit._run_checks(checks).failure(
            checks.alpha_normality, checks.alpha_homoscedasticity, checks.neg_check
        )
        if reason is None:
            kept.append(fit)
        else:
            excluded[fit.model] = reason
    return kept, excluded


def _sar_multi(
    data: pd.DataFrame,
    models: str | list[str],
    checks: CheckOptions,
    stacklevel: int,
) -> MultiSARFit:
    """``sar_multi``, with the exclusion warning aimed ``stacklevel`` frames up
    so it points at the user's call whichever public function made it."""
    # Validate once up front so bad input raises here. The per-model
    # except below is only for errors raised while fitting.
    _validate_data(data)

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

    fits, excluded = _exclude_failing(fits, checks)
    if excluded:
        listed = "; ".join(f"{model}: {reason}" for model, reason in excluded.items())
        warnings.warn(
            f"{len(excluded)} of {len(excluded) + len(fits)} fitted models failed a residual "
            f"check and were left out: {listed}",
            stacklevel=stacklevel + 1,
        )

    summary = _build_summary(fits)
    return MultiSARFit(fits=fits, data=data, summary=summary, excluded=excluded)


def sar_multi(
    data: pd.DataFrame,
    models: str | list[str] = "all",
    normality: str | None = None,
    homoscedasticity: str | None = None,
    cor_method: str = "spearman",
    neg_check: bool = False,
    alpha_normality: float = 0.05,
    alpha_homoscedasticity: float = 0.05,
) -> MultiSARFit:
    """Fit multiple SAR models to a dataset.

    By default every model that converges is kept. Ask for residual checks
    (``normality``, ``homoscedasticity``, ``neg_check``) to leave out the
    models that fail them, as R sars's ``sar_average`` does: the summary's
    weights are then shared among the models that pass, and a warning lists
    the ones left out.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    models : str or list[str]
        Model names to fit. Use "all" for all 20 models.
    normality : str or None
        Leave out models whose residuals fail this normality test:
        ``"shapiro"`` or ``"kolmogorov"`` (see ``SARFit.residual_checks``).
        None (default) for no test.
    homoscedasticity : str or None
        Leave out models whose squared residuals are correlated with the
        fitted values (``"cor.fitted"``) or with area (``"cor.area"``).
        None (default) for no test.
    cor_method : str
        Correlation for the homoscedasticity test: ``"spearman"`` (default),
        ``"pearson"`` or ``"kendall"``.
    neg_check : bool
        Leave out models that predict negative richness at an observed area.
        Whenever any check is asked for, a model whose fitted values aren't
        all finite is left out too.
    alpha_normality : float
        A model fails the normality test when its p value is below this.
    alpha_homoscedasticity : float
        A model fails the homoscedasticity test when its p value is below this.

    Returns
    -------
    MultiSARFit
        Container with the converged fits that passed the checks asked for, a
        summary table, and the models left out in ``excluded``.

    Warns
    -----
    UserWarning
        If any model fails a check that was asked for.
    """
    checks = CheckOptions(
        normality=normality,
        homoscedasticity=homoscedasticity,
        cor_method=cor_method,
        neg_check=neg_check,
        alpha_normality=alpha_normality,
        alpha_homoscedasticity=alpha_homoscedasticity,
    )
    return _sar_multi(data, models, checks, stacklevel=2)


_SUMMARY_COLUMNS = [
    "model",
    "R2",
    "AIC",
    "AICc",
    "BIC",
    "delta_AICc",
    "weight",
    "shape",
    "asymptote",
    "observed_shape",
]


def _build_summary(fits: list[SARFit], shapes: bool = True) -> pd.DataFrame:
    """Build summary DataFrame from a list of SARFit objects. ``shapes=False``
    leaves ``observed_shape`` empty: the bootstrap builds a summary per
    resample and reads only its weights."""
    if not fits:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)

    rows = []
    aicc_vals = []
    for fit in fits:
        asym = _get_asymptote(fit)
        rows.append(
            {
                "model": fit.model,
                "R2": fit.r_squared,
                "AIC": fit.aic,
                "AICc": fit.aicc,
                "BIC": fit.bic,
                "shape": _MODEL_SHAPE[fit.model],
                "observed_shape": fit.observed_shape() if shapes else None,
                "asymptote": asym,
            }
        )
        aicc_vals.append(fit.aicc)

    df = pd.DataFrame(rows)
    weights = akaike_weights(np.array(aicc_vals))
    min_aicc = np.nanmin(aicc_vals)
    df["delta_AICc"] = df["AICc"] - min_aicc
    df["weight"] = weights
    df = df.sort_values("AICc").reset_index(drop=True)
    return df[_SUMMARY_COLUMNS]


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

    def __post_init__(self) -> None:
        # An average of no models, like one with undefined weights or none on
        # the models fitted, would silently predict zero everywhere.
        if not self.multi.fits:
            raise ValueError(
                "There is nothing to average: no model converged on the data "
                "(or no model was given). Try other models, or check the data."
            )
        # akaike_weights returns all-NaN when no IC value is finite.
        if not all(np.isfinite(w) for w in self.weights.values()):
            raise ValueError(
                f"Model weights are undefined: no model has a finite {self.ic}. "
                f"For small samples AICc is infinite when n - k - 1 <= 0; try "
                f'ic="AIC" or ic="BIC", or more data points.'
            )
        if not sum(self.weights.get(fit.model, 0.0) for fit in self.multi.fits) > 0:
            raise ValueError(
                "The weights give no weight to any of the fitted models "
                f"({', '.join(fit.model for fit in self.multi.fits)}), so the average "
                "would be zero everywhere."
            )

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


# A bound above this many times the largest observed richness has run away.
_RUNAWAY_FACTOR = 10.0


def _drop_reason(exc: Exception) -> str:
    """A short key for a resample's error: its type and the first line of its
    message, cut short, so failures of one kind are counted together."""
    first = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {first[:60]}" if first else type(exc).__name__


_VALID_ICS = {"AIC", "AICc", "BIC"}


def _require_unexcluded(multi: MultiSARFit) -> None:
    """Raise if the residual checks left nothing to average. An average of no
    models would predict zero richness everywhere."""
    if multi.fits or not multi.excluded:
        return
    listed = "; ".join(f"{model}: {reason}" for model, reason in multi.excluded.items())
    raise ValueError(
        f"Every fitted model failed a residual check, so there is nothing to average "
        f"({listed}). Relax the checks or their alpha levels, or try other models."
    )


def sar_average(
    data: pd.DataFrame,
    models: str | list[str] = "all",
    ic: str = "AICc",
    normality: str | None = None,
    homoscedasticity: str | None = None,
    cor_method: str = "spearman",
    neg_check: bool = False,
    alpha_normality: float = 0.05,
    alpha_homoscedasticity: float = 0.05,
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
    normality, homoscedasticity : str or None
        Residual checks that leave failing models out, as in ``sar_multi``:
        a normality test (``"shapiro"`` or ``"kolmogorov"``) and a
        homoscedasticity test (``"cor.fitted"`` or ``"cor.area"``). None
        (default) for no test.
    cor_method : str
        Correlation for the homoscedasticity test: ``"spearman"`` (default),
        ``"pearson"`` or ``"kendall"``.
    neg_check : bool
        Leave out models that predict negative richness at an observed area.
    alpha_normality, alpha_homoscedasticity : float
        A model fails a test when its p value is below its alpha (0.05).

    Returns
    -------
    AveragedSAR
        Model-averaged SAR with weighted predictions. Models left out by a
        residual check are listed in its ``multi.excluded``.

    Raises
    ------
    ValueError
        If ``ic`` is unsupported, if no model converged (or none was given),
        if every fitted model failed the residual checks asked for, or if no
        fitted model has a finite ``ic`` value, so the weights are undefined.

    Warns
    -----
    UserWarning
        If any model fails a check that was asked for.
    """
    if ic not in _VALID_ICS:
        raise ValueError(f"Unsupported IC: {ic!r}. Choose from {sorted(_VALID_ICS)}.")

    checks = CheckOptions(
        normality=normality,
        homoscedasticity=homoscedasticity,
        cor_method=cor_method,
        neg_check=neg_check,
        alpha_normality=alpha_normality,
        alpha_homoscedasticity=alpha_homoscedasticity,
    )
    multi = _sar_multi(data, models, checks, stacklevel=2)
    _require_unexcluded(multi)
    weights_arr = akaike_weights(multi.summary[ic].to_numpy(dtype=float))
    weights = dict(zip(multi.summary["model"], weights_arr, strict=True))
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
        Mean prediction across bootstrap replicates. A few resamples whose
        prediction runs away can dominate it; for a central curve use the
        averaged fit's own prediction, or ``median`` as the interval's centre.
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
    median : np.ndarray
        Median prediction across bootstrap replicates: the centre of the
        percentile interval, and unlike ``mean`` always inside it.
    dropped : dict[str, int]
        Resamples left out of the interval, counted by reason: no model
        converged, the averaged prediction wasn't finite at some area, or the
        error a fit raised. ``n_boot`` plus their total is the number of
        resamples drawn.
    """

    area_grid: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    conf: float
    n_boot: int
    convergence_counts: list[int] = field(default_factory=list, repr=False)
    n_models_attempted: int = 0
    median: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    dropped: dict[str, int] = field(default_factory=dict, repr=False)

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

    summary = _build_summary(fits, shapes=False)
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
    normality: str | None = None,
    homoscedasticity: str | None = None,
    cor_method: str = "spearman",
    neg_check: bool = False,
    alpha_normality: float = 0.05,
    alpha_homoscedasticity: float = 0.05,
) -> BootstrappedCI:
    """Compute bootstrap confidence intervals for model-averaged predictions.

    Resamples rows of the data with replacement, fits all models, computes
    model-averaged predictions at each area value, then returns percentile-based
    confidence intervals.

    Residual checks (``normality``, ``homoscedasticity``, ``neg_check``) are
    applied once, to the fits on the original data, and fix the set of models:
    every resample averages the models that passed, without testing them
    again. That keeps the interval about the same average that
    ``sar_average`` with the same checks reports. Testing each resample would
    average a different set of models each time, and a resample's repeated
    rows produce tied residuals that the tests weren't designed for.

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
        Area values at which to predict. If None, about 100 points spanning
        the range of the data, half evenly spaced and half geometrically, so
        the small areas are covered when the areas span orders of magnitude.
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
    normality, homoscedasticity : str or None
        Residual checks that leave failing models out, as in ``sar_multi``:
        a normality test (``"shapiro"`` or ``"kolmogorov"``) and a
        homoscedasticity test (``"cor.fitted"`` or ``"cor.area"``). None
        (default) for no test.
    cor_method : str
        Correlation for the homoscedasticity test: ``"spearman"`` (default),
        ``"pearson"`` or ``"kendall"``.
    neg_check : bool
        Leave out models that predict negative richness at an observed area.
    alpha_normality, alpha_homoscedasticity : float
        A model fails a test when its p value is below its alpha (0.05).

    Returns
    -------
    BootstrappedCI

    Raises
    ------
    ValueError
        If ``method`` or ``ic`` is unsupported, or if every fitted model
        failed the residual checks asked for.

    Warns
    -----
    UserWarning
        If any model fails a check that was asked for; if resamples were
        dropped (``BootstrappedCI.dropped`` says why); if the mean is pulled
        outside the central 95% of the resample predictions; or if the upper
        bound runs away, exceeding ten times the largest observed richness.
    """
    if method not in ("fast", "full"):
        raise ValueError(f"method must be 'fast' or 'full', got {method!r}")
    if ic not in _VALID_ICS:
        raise ValueError(f"Unsupported IC: {ic!r}. Choose from {sorted(_VALID_ICS)}.")
    checks = CheckOptions(
        normality=normality,
        homoscedasticity=homoscedasticity,
        cor_method=cor_method,
        neg_check=neg_check,
        alpha_normality=alpha_normality,
        alpha_homoscedasticity=alpha_homoscedasticity,
    )

    _validate_data(data)
    if not isinstance(models, str) and not list(models):
        raise ValueError("No models given: there is nothing to average.")
    if rng is None:
        rng = np.random.default_rng()

    if area_grid is None:
        area_grid = _area_grid(float(data["area"].min()), float(data["area"].max()), 100)
    area_grid = np.asarray(area_grid, dtype=float)

    # For "fast" mode, do one full grid-search fit to get warm-start params.
    # Residual checks need that fit too, whatever the method: the models that
    # pass on the original data are the ones every resample averages.
    warm_starts: dict[str, dict[str, float]] | None = None
    if method == "fast" or checks.requested:
        base_multi = _sar_multi(data, models, checks, stacklevel=2)
        if base_multi.excluded:
            _require_unexcluded(base_multi)
            # Only the models that failed a check go. One that didn't converge
            # on the original data stays in the set a "full" resample tries,
            # as it does when no checks are asked for.
            asked = ALL_MODEL_NAMES if models == "all" else list(models)
            models = [name for name in asked if name not in base_multi.excluded]
        if method == "fast":
            warm_starts = {fit.model: dict(fit.params) for fit in base_multi.fits}
    if method == "fast":
        if not warm_starts:
            warnings.warn(
                "No model converged on the original data, so there is nothing to "
                "resample; the interval is NaN.",
                stacklevel=2,
            )
            nan_arr = np.full_like(area_grid, np.nan)
            return BootstrappedCI(
                area_grid=area_grid,
                mean=nan_arr,
                median=nan_arr.copy(),
                lower=nan_arr.copy(),
                upper=nan_arr.copy(),
                conf=conf,
                n_boot=0,
                n_models_attempted=len(ALL_MODEL_NAMES if models == "all" else list(models)),
            )

    if warm_starts is not None:
        n_attempted = len(warm_starts)
    else:
        n_attempted = len(ALL_MODEL_NAMES if models == "all" else list(models))
    n = len(data)
    alpha = 1.0 - conf
    preds = []
    convergence_counts: list[int] = []
    dropped: Counter[str] = Counter()

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_data = data.iloc[idx].reset_index(drop=True)
        # A resample that fails or overflows is dropped and counted, so numpy's
        # warnings about it, from the fits or the prediction, say nothing more.
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if method == "fast":
                    assert warm_starts is not None  # set above for the fast method
                    multi = _sar_multi_fast(boot_data, warm_starts)
                    n_converged = len(multi.fits)
                    if not multi.fits:
                        dropped["no model converged"] += 1
                        continue
                    w_arr = akaike_weights(multi.summary[ic].to_numpy(dtype=float))
                    weights = dict(zip(multi.summary["model"], w_arr, strict=True))
                    avg = AveragedSAR(multi=multi, ic=ic, weights=weights)
                else:
                    multi = sar_multi(boot_data, models=models)
                    n_converged = len(multi.fits)
                    if not multi.fits:
                        dropped["no model converged"] += 1
                        continue
                    avg = AveragedSAR(
                        multi=multi,
                        ic=ic,
                        weights=dict(
                            zip(
                                multi.summary["model"],
                                akaike_weights(multi.summary[ic].to_numpy(dtype=float)),
                                strict=True,
                            )
                        ),
                    )
                pred = avg.predict(area_grid)
        except Exception as exc:
            dropped[_drop_reason(exc)] += 1
            continue
        if np.all(np.isfinite(pred)):
            preds.append(pred)
            convergence_counts.append(n_converged)
        else:
            dropped["prediction not finite"] += 1

    if dropped:
        listed = dropped.most_common(3)
        reasons = "; ".join(f"{count} {reason}" for reason, count in listed)
        if len(dropped) > len(listed):
            reasons += f"; {len(dropped) - len(listed)} other reasons"
        outcome = (
            f"the interval is from the other {len(preds)}"
            if preds
            else "there is no interval (NaN), and BootstrappedCI.dropped has every reason"
        )
        warnings.warn(
            f"{sum(dropped.values())} of {n_boot} bootstrap resamples were dropped "
            f"({reasons}); {outcome}.",
            stacklevel=2,
        )
    if not preds:
        nan_arr = np.full_like(area_grid, np.nan)
        return BootstrappedCI(
            area_grid=area_grid,
            mean=nan_arr,
            median=nan_arr.copy(),
            lower=nan_arr.copy(),
            upper=nan_arr.copy(),
            conf=conf,
            n_boot=0,
            convergence_counts=[],
            n_models_attempted=n_attempted,
            dropped=dict(dropped),
        )

    # Warn if convergence rate across resamples was poor
    if n_attempted > 0:
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

    preds_arr = np.array(preds)
    mean = np.mean(preds_arr, axis=0)
    lower, median, upper = np.percentile(
        preds_arr, [100 * alpha / 2, 50, 100 * (1 - alpha / 2)], axis=0
    )
    # A runaway check that doesn't depend on ``conf``: a mean outside the central
    # 95% of the resample predictions is pulled there by a few extreme ones.
    lo95, hi95 = np.percentile(preds_arr, [2.5, 97.5], axis=0)
    slack = 1e-9 * np.maximum(np.abs(hi95 - lo95), np.abs(median))
    outside = int(np.sum((mean < lo95 - slack) | (mean > hi95 + slack)))
    if outside:
        warnings.warn(
            f"The bootstrap mean is outside the central 95% of the resample "
            f"predictions at {outside} of {len(area_grid)} areas: a few resamples' "
            f"predictions run away there. Use the averaged fit's own prediction as the "
            f"central curve, or the median as the interval's centre.",
            stacklevel=2,
        )
    # Where more than a tail's share of resamples run away, the bound itself
    # does: typically among areas few observations lie near, which resamples
    # often leave out, so the curve is extrapolated there.
    # A floor of one species keeps data of all-small counts from flagging everything.
    ceiling = _RUNAWAY_FACTOR * max(float(np.max(data["species"])), 1.0)
    runaway = upper > ceiling
    if runaway.any():
        where = area_grid[runaway]
        warnings.warn(
            f"The interval's upper bound exceeds {_RUNAWAY_FACTOR:g} times the largest "
            f"observed richness at {int(runaway.sum())} of {len(area_grid)} areas "
            f"(from {where.min():.4g} to {where.max():.4g}): there more than "
            f"{alpha / 2:.1%} of the resamples' predictions run away, usually because "
            f"few observations lie near those areas and resamples that leave them out "
            f"extrapolate. Read the interval as unbounded there.",
            stacklevel=2,
        )
    return BootstrappedCI(
        area_grid=area_grid,
        mean=mean,
        median=median,
        lower=lower,
        upper=upper,
        conf=conf,
        n_boot=len(preds),
        convergence_counts=convergence_counts,
        n_models_attempted=n_attempted,
        dropped=dict(dropped),
    )
