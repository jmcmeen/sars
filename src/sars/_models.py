"""SAR model definitions and fitting routines."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product

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
            raise NotImplementedError(f"predict not yet implemented for '{self.model}'")
        return spec.func(area, *[self.params[k] for k in spec.param_names])

    def __repr__(self) -> str:
        p = "  ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        return f"SARFit(model='{self.model}', {p}, R²={self.r_squared:.4f}, AICc={self.aicc:.2f})"


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


# A SAR model function, f(area, p1, p2, ...) -> predicted S. Each model takes a
# fixed number of parameters, named by its spec's ``param_names``, which a
# Callable type can't express for every arity at once.
_SARFunc = Callable[..., np.ndarray]


@dataclass
class _ModelSpec:
    """Internal specification for a SAR model."""

    name: str
    func: _SARFunc
    param_names: list[str]
    # Search limits on the rescaled parameters (see _RESCALE), not on the ones
    # reported: a fit in other units can report values beyond them.
    bounds_lower: list[float]
    bounds_upper: list[float]
    # One list of starting values per param, on the rescaled data the fit runs on
    # (see _RESCALE): area over its geometric mean, species over its maximum.
    start_grid: list[list[float]]
    # Parameters the solver searches on the natural-log scale, for ones whose
    # optimum can sit many orders of magnitude from the start values (weibull4's
    # c reaches 1e-15). Their bounds and start values are still given, and their
    # fitted values reported, on the natural scale; a lower bound of 0 becomes
    # exp(_LOG_FLOOR).
    log_params: tuple[str, ...] = ()


# Smallest log value a log-scale parameter can take (exp(-700) is about 1e-304).
_LOG_FLOOR = -700.0


def _to_search_scale(values: Sequence[float] | np.ndarray, log_idx: list[int]) -> list[float]:
    out = [float(v) for v in values]
    for i in log_idx:
        out[i] = max(float(np.log(out[i])) if out[i] > 0 else -np.inf, _LOG_FLOOR)
    return out


def _to_natural_scale(values: Sequence[float] | np.ndarray, log_idx: list[int]) -> np.ndarray:
    out = np.array(values, dtype=float)
    out[log_idx] = np.exp(out[log_idx])
    return out


_MODEL_REGISTRY: dict[str, _ModelSpec] = {}


def _register(spec: _ModelSpec) -> _ModelSpec:
    if spec.name not in _RESCALE:
        raise ValueError(f"model {spec.name!r} needs a rescaling entry in _RESCALE")
    _MODEL_REGISTRY[spec.name] = spec
    return spec


# ---------------------------------------------------------------------------
# Rescaling
# ---------------------------------------------------------------------------

# Each model is fitted on rescaled data, A' = A / a and S' = S / s, where a is
# the geometric mean of the areas and s the largest richness, and its
# parameters converted back. The start grids and bounds are then in units
# near 1 whatever the data's units, and a model whose form survives a change
# of units gives the same fit (the same R2, and parameters converted exactly)
# in km2 or m2, for tens of species or thousands. Each entry maps rescaled
# parameters p' to the data's units, and back, in the spec's parameter order.
# epm1 and epm2 change form with the area units (their exponent depends on A
# itself), so only their richness is rescaled.
_Rescale = Callable[[np.ndarray, float, float], list[float]]
_RESCALE: dict[str, tuple[_Rescale, _Rescale]] = {
    # S = c A^z: c = s c' a^-z
    "power": (
        lambda p, a, s: [s * p[0] * a ** -p[1], p[1]],
        lambda p, a, s: [p[0] * a ** p[1] / s, p[1]],
    ),
    # S = f + c A^z: f = s f', c = s c' a^-z
    "powerR": (
        lambda p, a, s: [s * p[0], s * p[1] * a ** -p[2], p[2]],
        lambda p, a, s: [p[0] / s, p[1] * a ** p[2] / s, p[2]],
    ),
    # S = c + z log A: z = s z', c = s (c' - z' log a)
    "loga": (
        lambda p, a, s: [s * (p[0] - p[1] * np.log(a)), s * p[1]],
        lambda p, a, s: [p[0] / s + p[1] / s * np.log(a), p[1] / s],
    ),
    # S = c + m A: c = s c', m = s m' / a
    "linear": (
        lambda p, a, s: [s * p[0], s * p[1] / a],
        lambda p, a, s: [p[0] / s, p[1] * a / s],
    ),
    # epm1 and epm2: c = s c' (richness only)
    "epm1": (
        lambda p, a, s: [s * p[0], p[1], p[2]],
        lambda p, a, s: [p[0] / s, p[1], p[2]],
    ),
    "epm2": (
        lambda p, a, s: [s * p[0], p[1], p[2]],
        lambda p, a, s: [p[0] / s, p[1], p[2]],
    ),
    # S = c A^z exp(-d A): c = s c' a^-z, d = d' / a
    "p1": (
        lambda p, a, s: [s * p[0] * a ** -p[1], p[1], p[2] / a],
        lambda p, a, s: [p[0] * a ** p[1] / s, p[1], p[2] * a],
    ),
    # S = c A^z exp(-d / A): c = s c' a^-z, d = d' a
    "p2": (
        lambda p, a, s: [s * p[0] * a ** -p[1], p[1], p[2] * a],
        lambda p, a, s: [p[0] * a ** p[1] / s, p[1], p[2] / a],
    ),
    # S = c log(1 + A / z): c = s c', z = z' a
    "koba": (
        lambda p, a, s: [s * p[0], p[1] * a],
        lambda p, a, s: [p[0] / s, p[1] / a],
    ),
    # S = d A / (c + A): d = s d', c = c' a
    "monod": (
        lambda p, a, s: [s * p[0], p[1] * a],
        lambda p, a, s: [p[0] / s, p[1] / a],
    ),
    # S = d (1 - exp(-z A)): d = s d', z = z' / a
    "negexpo": (
        lambda p, a, s: [s * p[0], p[1] / a],
        lambda p, a, s: [p[0] / s, p[1] * a],
    ),
    # S = d - c exp(-z A): d = s d', c = s c', z = z' / a
    "asymp": (
        lambda p, a, s: [s * p[0], s * p[1], p[2] / a],
        lambda p, a, s: [p[0] / s, p[1] / s, p[2] * a],
    ),
    # S = (c + z A) / (1 + d A): c = s c', z = s z' / a, d = d' / a
    "ratio": (
        lambda p, a, s: [s * p[0], s * p[1] / a, p[2] / a],
        lambda p, a, s: [p[0] / s, p[1] * a / s, p[2] * a],
    ),
    # S = d / (1 + c A^-z): d = s d', c = c' a^z
    "mmf": (
        lambda p, a, s: [s * p[0], p[1] * a ** p[2], p[2]],
        lambda p, a, s: [p[0] / s, p[1] * a ** -p[2], p[2]],
    ),
    # S = d exp(-exp(-z (A - c))): d = s d', z = z' / a, c = c' a
    "gompertz": (
        lambda p, a, s: [s * p[0], p[1] / a, p[2] * a],
        lambda p, a, s: [p[0] / s, p[1] * a, p[2] / a],
    ),
    # S = d (1 - exp(-c A^z)): d = s d', c = c' a^-z
    "weibull3": (
        lambda p, a, s: [s * p[0], p[1] * a ** -p[2], p[2]],
        lambda p, a, s: [p[0] / s, p[1] * a ** p[2], p[2]],
    ),
    # S = d (1 - exp(-c A^z))^f: d = s d', c = c' a^-z
    "weibull4": (
        lambda p, a, s: [s * p[0], p[1] * a ** -p[2], p[2], p[3]],
        lambda p, a, s: [p[0] / s, p[1] * a ** p[2], p[2], p[3]],
    ),
    # S = d (1 - exp(-z A))^c: d = s d', z = z' / a
    "chapman": (
        lambda p, a, s: [s * p[0], p[1] / a, p[2]],
        lambda p, a, s: [p[0] / s, p[1] * a, p[2]],
    ),
    # S = d (1 - (1 + (A / c)^z)^-f): d = s d', c = c' a
    "betap": (
        lambda p, a, s: [s * p[0], p[1] * a, p[2], p[3]],
        lambda p, a, s: [p[0] / s, p[1] / a, p[2], p[3]],
    ),
    # S = c / (f + A^-z): c = s c' a^-z, f = f' a^-z
    "heleg": (
        lambda p, a, s: [s * p[0] * a ** -p[2], p[1] * a ** -p[2], p[2]],
        lambda p, a, s: [p[0] * a ** p[2] / s, p[1] * a ** p[2], p[2]],
    ),
}
# Models whose area isn't rescaled (see above).
_AREA_UNSCALED = {"epm1", "epm2"}


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
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")

    missing = [c for c in ("area", "species") if c not in data.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. DataFrame has columns: {list(data.columns)}"
        )

    if len(data) < 2:
        raise ValueError(f"Need at least 2 observations, got {len(data)}")

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

    # The fit runs on rescaled data (see _RESCALE).
    a_scale = 1.0 if name in _AREA_UNSCALED else float(np.exp(np.mean(np.log(area))))
    s_scale = float(np.max(species))
    to_natural, to_scaled = _RESCALE[name]
    area_s, species_s = area / a_scale, species / s_scale

    log_idx = [spec.param_names.index(p) for p in spec.log_params]
    lower = _to_search_scale(spec.bounds_lower, log_idx)
    upper = _to_search_scale(spec.bounds_upper, log_idx)

    def residuals(p: np.ndarray) -> np.ndarray:
        pred = spec.func(area_s, *(_to_natural_scale(p, log_idx) if log_idx else p))
        res = species_s - pred
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
        bad = [k for k in spec.log_params if not start_from[k] > 0]
        if bad:
            raise ValueError(
                f"start_from values for {bad} must be positive: model '{name}' "
                f"searches them on a log scale"
            )
        given = np.array([start_from[k] for k in spec.param_names], dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            warm = np.array(to_scaled(given, a_scale, s_scale), dtype=float)
        if np.all(np.isfinite(warm)) and all(warm[i] > 0 for i in log_idx):
            starts: Iterable[Sequence[float]] = [warm.tolist()]
        else:  # it doesn't survive rescaling (overflow, or a log parameter at 0)
            logger.debug("Model '%s': warm start %s can't be rescaled; using the grid", name, given)
            starts = product(*spec.start_grid)
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
    n_unsuccessful = 0
    stale_runs = 0
    for start in starts:
        n_starts += 1
        try:
            # The search visits parameters where the model overflows; residuals()
            # turns the non-finite values into a penalty, so numpy's warnings about
            # them carry nothing and would flood an interactive session.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                result = least_squares(
                    residuals,
                    x0=np.clip(_to_search_scale(start, log_idx), lower, upper),
                    bounds=(lower, upper),
                    method="trf",
                    max_nfev=5000,
                )
            if not result.success:
                # Stopped on max_nfev or similar: not a converged solution,
                # so it may neither win on cost nor count as a stale run.
                n_unsuccessful += 1
                continue
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                stale_runs = 0
            else:
                stale_runs += 1
        except Exception:
            n_errors += 1
            logger.debug(
                "Model '%s' NLS failed for start %s",
                name,
                list(start),
                exc_info=True,
            )

        if stale_runs >= stale_limit and best_result is not None:
            logger.debug(
                "Model '%s': early stop after %d starts (%d without improvement)",
                name,
                n_starts,
                stale_limit,
            )
            break

        if n_starts % 100 == 0:
            logger.debug(
                "Model '%s': %d starts evaluated (best cost=%.4g)",
                name,
                n_starts,
                best_cost,
            )

    if n_errors:
        logger.info(
            "Model '%s': %d/%d starting points raised exceptions",
            name,
            n_errors,
            n_starts,
        )
    if n_unsuccessful:
        logger.info(
            "Model '%s': %d/%d starting points stopped without converging",
            name,
            n_unsuccessful,
            n_starts,
        )

    if best_result is None:
        return _failed_fit(name, spec.param_names, n, data)

    if start_from is None:
        # The default tolerances stop early along flat ridges of the likelihood,
        # leaving parameters loose in the third digit; one tight solve from the best
        # grid start pins them down. A warm start (the bootstrap's) skips it: its
        # resamples need the fit, not the last digits.
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                polished = least_squares(
                    residuals,
                    x0=best_result.x,
                    bounds=(lower, upper),
                    method="trf",
                    max_nfev=5000,
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14,
                )
            if polished.success and polished.cost <= best_result.cost:
                best_result = polished
        except Exception:
            logger.debug("Model '%s': polishing the best fit failed", name, exc_info=True)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = np.array(
            to_natural(_to_natural_scale(best_result.x, log_idx), a_scale, s_scale), dtype=float
        )
        # RSS in the data's units, from the converted parameters: it is what's
        # reported, and it checks the conversion.
        rss = float(np.sum((species - spec.func(area, *values)) ** 2))
    if not (np.all(np.isfinite(values)) and np.isfinite(rss)):
        logger.info("Model '%s': the fit's parameters overflow in the data's units", name)
        return _failed_fit(name, spec.param_names, n, data)
    params = dict(zip(spec.param_names, (float(v) for v in values), strict=True))
    ss_tot = np.sum((species - np.mean(species)) ** 2)
    # R² is undefined when every island has the same richness.
    r_squared = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan

    k = len(spec.param_names) + 1  # +1 for sigma
    aic, aicc, bic = _compute_ic(k, n, rss)

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


def _failed_fit(name: str, param_names: list[str], n: int, data: pd.DataFrame) -> SARFit:
    return SARFit(
        model=name,
        params=dict.fromkeys(param_names, np.nan),
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
_register(
    _ModelSpec(
        name="power",
        func=lambda a, c, z: c * a**z,
        param_names=["c", "z"],
        bounds_lower=[1e-10, -5.0],
        bounds_upper=[1e6, 5.0],
        start_grid=[
            [0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
            [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
        ],
    )
)

# 2. powerR: S = c * A^z + d
_register(
    _ModelSpec(
        name="powerR",
        func=lambda a, f, c, z: f + c * a**z,
        param_names=["f", "c", "z"],
        bounds_lower=[-1e6, 1e-10, -5.0],
        bounds_upper=[1e6, 1e6, 5.0],
        start_grid=[
            [-1.0, -0.3, 0.0, 0.1],
            [0.1, 0.3, 1.0, 3.0],
            [0.05, 0.15, 0.3, 0.5, 1.0],
        ],
    )
)

# 3. loga: S = c + z * log(A)
_register(
    _ModelSpec(
        name="loga",
        func=lambda a, c, z: c + z * np.log(a),
        param_names=["c", "z"],
        bounds_lower=[-1e6, -1e6],
        bounds_upper=[1e6, 1e6],
        start_grid=[
            [-1.0, -0.3, 0.0, 0.3, 1.0],
            [0.01, 0.03, 0.1, 0.3, 1.0],
        ],
    )
)

# 4. linear: S = c + m * A  (OLS)
_register(
    _ModelSpec(
        name="linear",
        func=lambda a, c, m: c + m * a,
        param_names=["c", "m"],
        bounds_lower=[-1e6, -1e6],
        bounds_upper=[1e6, 1e6],
        start_grid=[
            [0.0, 0.1, 0.3, 0.5, 1.0],
            [0.001, 0.01, 0.03, 0.1, 1.0],
        ],
    )
)


# 5. epm1: S = c * A^(z * A^-d)  (Extended power model 1, Tjørve 2009)
def _epm1_func(a, c, z, d):
    return c * a ** (z * a ** (-d))


_register(
    _ModelSpec(
        name="epm1",
        func=_epm1_func,
        param_names=["c", "z", "d"],
        bounds_lower=[0.0, -50.0, -50.0],
        bounds_upper=[np.inf, 50.0, 50.0],
        start_grid=[
            [1e-08, 1e-05, 0.001, 0.03, 0.3],
            [0.05, 0.3, 1.0, 3.0, 8.0],
            [-1.0, -0.3, 0.0, 0.1, 0.5, 1.0],
        ],
        # c spans dozens of orders of magnitude with the area units (1e-49 in m²).
        log_params=("c",),
    )
)


# 6. epm2: S = c * A^(z - d/A)  (Extended power model 2, Tjørve 2009). z lies in
#    [0, 1], as R sars bounds it; d is unbounded because it scales with the area
#    units (d/A).
def _epm2_func(a, c, z, d):
    return c * a ** (z - d / a)


_register(
    _ModelSpec(
        name="epm2",
        func=_epm2_func,
        param_names=["c", "z", "d"],
        bounds_lower=[0.0, 0.0, -np.inf],
        bounds_upper=[np.inf, 1.0, np.inf],
        start_grid=[
            [0.002, 0.02, 0.06, 0.15, 0.3],
            [0.05, 0.25, 0.5, 0.9],
            [-10.0, -1.0, 0.0, 0.5, 1.0, 10.0],
        ],
    )
)


# 7. p1: S = c * A^z * exp(-d * A)  (Persistence function 1)
def _p1_func(a, c, z, d):
    return c * a**z * np.exp(-d * a)


_register(
    _ModelSpec(
        name="p1",
        func=_p1_func,
        param_names=["c", "z", "d"],
        bounds_lower=[1e-10, -5.0, 0.0],
        bounds_upper=[1e6, 5.0, 10.0],
        start_grid=[
            [0.05, 0.1, 0.3, 0.5, 1.0],
            [0.1, 0.3, 0.5, 0.7, 1.0],
            [0.001, 0.01, 0.03, 0.1, 0.3],
        ],
    )
)


# 8. p2: S = c * A^z * exp(-d / A)  (Persistence function 2)
def _p2_func(a, c, z, d):
    return c * a**z * np.exp(-d / a)


_register(
    _ModelSpec(
        name="p2",
        func=_p2_func,
        param_names=["c", "z", "d"],
        bounds_lower=[1e-10, -5.0, -np.inf],
        bounds_upper=[1e6, 5.0, np.inf],
        start_grid=[
            [0.1, 0.3, 0.5, 1.0, 2.0],
            [0.001, 0.01, 0.05, 0.1, 0.5],
            [0.03, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)

# --- koba (convex, no asymptote), then the asymptotic convex models (4) ---

# 9. koba: S = c * log(1 + A/z)
_register(
    _ModelSpec(
        name="koba",
        func=lambda a, c, z: c * np.log(1.0 + a / z),
        param_names=["c", "z"],
        bounds_lower=[1e-10, 1e-10],
        bounds_upper=[1e6, 1e6],
        start_grid=[
            [0.01, 0.03, 0.1, 0.2, 0.3, 0.5],
            [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)

# 10. monod: S = d * A / (c + A)
_register(
    _ModelSpec(
        name="monod",
        func=lambda a, d, c: d * a / (c + a),
        param_names=["d", "c"],
        bounds_lower=[1e-10, 1e-10],
        bounds_upper=[1e6, 1e6],
        start_grid=[
            [0.3, 0.5, 0.8, 1.0, 1.5, 3.0],
            [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        ],
    )
)

# 11. negexpo: S = d * (1 - exp(-z * A))
_register(
    _ModelSpec(
        name="negexpo",
        func=lambda a, d, z: d * (1.0 - np.exp(-z * a)),
        param_names=["d", "z"],
        bounds_lower=[1e-10, 1e-10],
        bounds_upper=[1e6, 100.0],
        start_grid=[
            [0.3, 0.5, 0.8, 1.0, 1.5, 3.0],
            [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)

# 12. asymp: S = d - c * exp(-z * A)
_register(
    _ModelSpec(
        name="asymp",
        func=lambda a, d, c, z: d - c * np.exp(-z * a),
        param_names=["d", "c", "z"],
        bounds_lower=[1e-10, 1e-10, 1e-10],
        bounds_upper=[1e6, 1e6, 100.0],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0],
            [0.3, 0.6, 1.0, 2.0],
            [0.003, 0.03, 0.1, 0.3, 1.0],
        ],
    )
)

# 13. ratio: S = (c + z * A) / (1 + d * A)
_register(
    _ModelSpec(
        name="ratio",
        func=lambda a, c, z, d: (c + z * a) / (1.0 + d * a),
        param_names=["c", "z", "d"],
        bounds_lower=[-1e6, 1e-10, 1e-10],
        bounds_upper=[1e6, 1e6, 100.0],
        start_grid=[
            [-0.1, 0.0, 0.03, 0.1, 0.3],
            [0.03, 0.1, 0.3, 1.0, 3.0],
            [0.03, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)

# --- Asymptotic sigmoid models (7) ---

# 14. mmf: S = d / (1 + c * A^(-z))
_register(
    _ModelSpec(
        name="mmf",
        func=lambda a, d, c, z: d / (1.0 + c * a ** (-z)),
        param_names=["d", "c", "z"],
        bounds_lower=[1e-10, 1e-10, 1e-10],
        bounds_upper=[np.inf, np.inf, 10.0],
        start_grid=[
            [0.3, 0.5, 1.0, 2.0, 5.0],
            [0.1, 0.5, 1.0, 3.0, 10.0],
            [0.1, 0.5, 1.0, 1.5, 2.0],
        ],
        # On small samples d and c grow together toward a power-law limit.
        log_params=("d", "c"),
    )
)

# 15. gompertz: S = d * exp(-exp(-z * (A - c)))
_register(
    _ModelSpec(
        name="gompertz",
        func=lambda a, d, z, c: d * np.exp(-np.exp(-z * (a - c))),
        param_names=["d", "z", "c"],
        bounds_lower=[1e-10, 1e-10, -1e6],
        bounds_upper=[np.inf, 100.0, 1e6],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0, 5.0],
            [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            [0.0, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)


# 16. weibull3: S = d * (1 - exp(-c * A^z))
def _weibull3_func(a, d, c, z):
    return d * (1.0 - np.exp(-c * a**z))


_register(
    _ModelSpec(
        name="weibull3",
        func=_weibull3_func,
        param_names=["d", "c", "z"],
        bounds_lower=[1e-10, 0.0, 1e-10],
        bounds_upper=[np.inf, 100.0, 10.0],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0, 5.0],
            [0.01, 0.1, 0.3, 1.0, 3.0],
            [0.1, 0.5, 1.0, 1.5, 2.0],
        ],
        # On small samples d runs off toward a power-law limit, with c shrinking.
        log_params=("d", "c"),
    )
)


# 17. weibull4: S = d * (1 - exp(-c * A^z))^f
def _weibull4_func(a, d, c, z, f):
    inner = 1.0 - np.exp(-c * a**z)
    return d * np.sign(inner) * np.abs(inner) ** f


_register(
    _ModelSpec(
        name="weibull4",
        func=_weibull4_func,
        param_names=["d", "c", "z", "f"],
        bounds_lower=[1e-10, 0.0, 1e-10, 1e-10],
        bounds_upper=[np.inf, 100.0, 10.0, 100.0],
        start_grid=[
            [0.5, 1.0, 2.0],
            [1e-08, 1e-05, 0.001, 0.03, 0.3, 1.0],
            [0.5, 1.0, 3.0, 6.0, 9.0],
            [0.05, 0.1, 0.5, 1.0, 2.0],
        ],
        log_params=("c",),
    )
)


# 18. chapman: S = d * (1 - exp(-z * A))^c
def _chapman_func(a, d, z, c):
    inner = 1.0 - np.exp(-z * a)
    return d * np.sign(inner) * np.abs(inner) ** c


_register(
    _ModelSpec(
        name="chapman",
        func=_chapman_func,
        param_names=["d", "z", "c"],
        bounds_lower=[1e-10, 1e-10, 1e-10],
        bounds_upper=[1e6, 100.0, 100.0],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0, 5.0],
            [0.03, 0.1, 0.3, 1.0, 3.0],
            [0.1, 0.3, 0.5, 1.0, 2.0],
        ],
    )
)


# 19. betap: S = d * (1 - (1 + (A/c)^z)^(-f))
def _betap_func(a, d, c, z, f):
    return d * (1.0 - (1.0 + (a / c) ** z) ** (-f))


_register(
    _ModelSpec(
        name="betap",
        func=_betap_func,
        param_names=["d", "c", "z", "f"],
        bounds_lower=[1e-10, 1e-10, 1e-10, 1e-10],
        bounds_upper=[1e6, 1e6, 10.0, 100.0],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0],
            [0.3, 1.0, 3.0, 10.0, 30.0],
            [0.1, 0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 2.0, 5.0, 10.0],
        ],
    )
)


# 20. heleg: S = c / (f + A^(-z))
#     Parameters c, f, z (all positive). Asymptote = c / f.
def _heleg_func(a, c, f, z):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = c / (f + a ** (-z))
    return np.where(np.isfinite(result), result, 0.0)


_register(
    _ModelSpec(
        name="heleg",
        func=_heleg_func,
        param_names=["c", "f", "z"],
        bounds_lower=[1e-10, 1e-10, 1e-10],
        bounds_upper=[1e6, 1e6, 1e6],
        start_grid=[
            [0.03, 0.1, 0.3, 1.0, 3.0],
            [0.03, 0.1, 0.3, 1.0, 3.0],
            [0.1, 0.5, 1.0, 2.0, 5.0],
        ],
    )
)


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
    """Fit the extended power model 1: S = c * A^(z * A^-d)

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
    """Fit the extended power model 2: S = c * A^(z - d/A)

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

    ``d`` may be negative, as in R ``sars``; the curve then rises steeply toward
    the smallest areas, so predictions below the data's range can be very large.

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

    R ``sars`` writes the same curve as ``d - c * z^A``, so its ``z`` is
    ``exp(-z)`` here. ``z`` is kept positive, which keeps the asymptote; R can
    also reach ``z > 1`` in its form, a curve that grows without bound.

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
