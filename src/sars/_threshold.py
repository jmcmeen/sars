"""Threshold / piecewise SAR models.

Implements the one-threshold piecewise regression approach described in
Matthews & Rigal (2021), Frontiers of Biogeography 13(1), e49404.

Three models are supported:
- ContOne: continuous two-slope (breakpoint where slope changes)
- ZslopeOne: left-horizontal + right slope (small island effect)
- Linear: simple linear baseline (no breakpoint)

Breakpoint selection is a two-level process: within each piecewise model
class (ContOne, ZslopeOne), a grid search over the natural log of area (or
over area itself with ``log_area=False``) selects the breakpoint that
minimises RSS; then across model classes (including the
no-breakpoint Linear baseline), AICc ranks the best candidate from each
class to choose the overall winner.

Meaningful threshold detection generally requires at least 7 observations
(n >= 7). With fewer data points the grid search may find zero valid
breakpoint candidates, leaving only the Linear baseline.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sars._models import _compute_ic, _validate_data

# ---------------------------------------------------------------------------
# ThresholdFit dataclass
# ---------------------------------------------------------------------------


@dataclass
class ThresholdFit:
    """Result of fitting threshold / piecewise SAR models.

    Attributes
    ----------
    best_model : str
        Name of the best model by AICc ("ContOne", "ZslopeOne", or "Linear").
    best_breakpoint : float | None
        Breakpoint on the fitted scale: the natural log of area when ``log_area``
        is True (the default), area itself otherwise. None for the linear model.
        ``breakpoint_area`` is the same point in area units.
    best_segments : list[dict]
        Segment parameters as list of dicts with keys 'intercept', 'slope',
        'x_min', 'x_max', on the fitted scale.
    summary : pd.DataFrame
        Comparison table with columns: model, breakpoint (fitted scale),
        breakpoint_area (area units), AIC, AICc, BIC, R2.
    all_fits : dict[str, dict]
        Detailed fit results keyed by model name.
    data : pd.DataFrame
        Original data (columns: area, species).
    log_area : bool
        Whether the models were fitted on the natural log of area (True) or on
        area itself (False).
    """

    best_model: str
    best_breakpoint: float | None
    best_segments: list[dict]
    summary: pd.DataFrame = field(repr=False)
    all_fits: dict[str, dict] = field(repr=False)
    data: pd.DataFrame = field(repr=False)
    log_area: bool = True

    def predict(self, area: float | np.ndarray) -> np.ndarray:
        """Predict species richness for given area value(s).

        Parameters
        ----------
        area : float or array-like
            Area value(s) in original (non-log) space.

        Returns
        -------
        np.ndarray
            Predicted species richness. Outside the fitted area range the first
            segment (below) or the last (above) is extended, so a falling line
            can predict fewer than zero species far below the range.

        Raises
        ------
        ValueError
            If the fit is on ln(area) and an area is zero or negative.
        """
        area = np.asarray(area, dtype=float)
        if self.log_area and np.any(area <= 0):
            raise ValueError("Areas must be positive: this fit is on ln(area).")
        x = np.log(area) if self.log_area else area
        return _predict_segments(x, self.best_segments)

    @property
    def breakpoint_area(self) -> float | None:
        """The best model's breakpoint in area units (None for the linear model)."""
        return _to_area(self.best_breakpoint, self.log_area)

    def __repr__(self) -> str:
        if self.best_breakpoint is None:
            return f"ThresholdFit(best='{self.best_model}')"
        if not self.log_area:  # the fitted scale is area: one number says it all
            return f"ThresholdFit(best='{self.best_model}', breakpoint_area={self.breakpoint_area:.4g})"
        return (
            f"ThresholdFit(best='{self.best_model}', "
            f"breakpoint={self.best_breakpoint:.3f} (ln area), "
            f"breakpoint_area={self.breakpoint_area:.4g})"
        )


def _to_area(x: float | None, log_area: bool) -> float | None:
    """A value on the fitted scale back in area units."""
    if x is None:
        return None
    return float(np.exp(x)) if log_area else float(x)


def _predict_segments(x: np.ndarray, segments: list[dict]) -> np.ndarray:
    """Predict from piecewise linear segments, extending the end segments.

    ``segments`` are in order along x, as the fitters build them. Each segment
    applies from its ``x_min`` on, so the first one's line continues below the
    fitted range and the last one's above it, as the SAR models' ``predict``
    extrapolates their curves. A NaN input gives NaN.
    """
    first = segments[0]
    pred = first["intercept"] + first["slope"] * x
    for seg in segments[1:]:
        pred = np.where(x >= seg["x_min"], seg["intercept"] + seg["slope"] * x, pred)
    return np.asarray(pred, dtype=float)


# ---------------------------------------------------------------------------
# Individual piecewise model fits
# ---------------------------------------------------------------------------


def _fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit simple linear model y = c + m*x (no breakpoint).

    Parameters: c (intercept), m (slope) + sigma = 3 for AICc.
    """
    n = len(x)
    # OLS via numpy
    x_mat = np.column_stack([np.ones(n), x])
    coeffs, rss_arr, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    c, m = coeffs
    pred = x_mat @ coeffs
    rss = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0

    k = 3  # intercept + slope + sigma
    aic, aicc, bic = _compute_ic(k, n, rss)

    segments = [
        {
            "intercept": float(c),
            "slope": float(m),
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
        }
    ]

    return {
        "model": "Linear",
        "breakpoint": None,
        "coeffs": {"c": float(c), "m": float(m)},
        "segments": segments,
        "rss": rss,
        "r2": r2,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "k": k,
        "n": n,
    }


def _fit_cont_one(x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Fit continuous one-threshold model.

    y = c1 + z1*x + z_delta * (x - T) * I(x > T)

    Left segment:  y = c1 + z1*x
    Right segment: y = c1 + z1*T + (z1 + z_delta)*(x - T)
                     = (c1 + z_delta*T) + (z1 + z_delta)*x  ... wait, no:
                     = c1 + z1*x + z_delta*(x - T)

    Parameters: c1, z1, z_delta + breakpoint + sigma = 5 for AICc.
    """
    n = len(x)
    # Design matrix: [1, x, (x - T) * I(x > T)]
    right_part = np.where(x > threshold, x - threshold, 0.0)
    x_mat = np.column_stack([np.ones(n), x, right_part])
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    c1, z1, z_delta = coeffs

    pred = x_mat @ coeffs
    rss = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0

    k = 5  # c1 + z1 + z_delta + breakpoint + sigma
    aic, aicc, bic = _compute_ic(k, n, rss)

    segments = [
        {
            "intercept": float(c1),
            "slope": float(z1),
            "x_min": float(np.min(x)),
            "x_max": float(threshold),
        },
        {
            "intercept": float(c1 - z_delta * threshold),
            "slope": float(z1 + z_delta),
            "x_min": float(threshold),
            "x_max": float(np.max(x)),
        },
    ]

    return {
        "model": "ContOne",
        "breakpoint": float(threshold),
        "coeffs": {
            "c1": float(c1),
            "z1": float(z1),
            "z_delta": float(z_delta),
        },
        "segments": segments,
        "rss": rss,
        "r2": r2,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "k": k,
        "n": n,
    }


def _fit_zslope_one(x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Fit left-horizontal one-threshold model (small island effect).

    y = c1 + z2 * (x - T) * I(x > T)

    Left segment:  y = c1  (flat)
    Right segment: y = c1 + z2*(x - T)

    Parameters: c1, z2 + breakpoint + sigma = 4 for AICc.
    """
    n = len(x)
    # Design matrix: [1, (x - T) * I(x > T)]
    right_part = np.where(x > threshold, x - threshold, 0.0)
    x_mat = np.column_stack([np.ones(n), right_part])
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    c1, z2 = coeffs

    pred = x_mat @ coeffs
    rss = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0

    k = 4  # c1 + z2 + breakpoint + sigma
    aic, aicc, bic = _compute_ic(k, n, rss)

    segments = [
        {
            "intercept": float(c1),
            "slope": 0.0,
            "x_min": float(np.min(x)),
            "x_max": float(threshold),
        },
        {
            "intercept": float(c1 - z2 * threshold),
            "slope": float(z2),
            "x_min": float(threshold),
            "x_max": float(np.max(x)),
        },
    ]

    return {
        "model": "ZslopeOne",
        "breakpoint": float(threshold),
        "coeffs": {"c1": float(c1), "z2": float(z2)},
        "segments": segments,
        "rss": rss,
        "r2": r2,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "k": k,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Grid search for optimal breakpoint
# ---------------------------------------------------------------------------


def _grid_search_breakpoint(
    x: np.ndarray,
    y: np.ndarray,
    fit_func: Callable[[np.ndarray, np.ndarray, float], dict],
    interval: float | None,
    min_points: int = 3,
) -> dict | None:
    """Search for the breakpoint that minimises RSS.

    Parameters
    ----------
    x : np.ndarray
        Predictor values on the fitted scale (log-area, or area).
    y : np.ndarray
        Response values (species).
    fit_func : callable
        One of _fit_cont_one or _fit_zslope_one.
    interval : float or None
        Grid spacing for breakpoint candidates, on the same scale as ``x``.
        None: a thousandth of the window the candidates can lie in.
    min_points : int
        Minimum number of data points required on each side.

    Returns
    -------
    dict or None
        Best fit result dict, or None if no valid breakpoint found.
    """
    x_sorted = np.sort(x)
    # Ensure at least min_points on each side
    x_min_bp = x_sorted[min_points - 1]
    x_max_bp = x_sorted[-(min_points)]

    if x_min_bp >= x_max_bp:
        return None
    if interval is None:
        interval = (x_max_bp - x_min_bp) / 1000
        if not interval > 0:  # a window too narrow for float steps: no breakpoint
            return None

    candidates = np.arange(x_min_bp, x_max_bp + interval / 2, interval)
    if len(candidates) == 0:
        return None

    best_fit = None
    best_rss = np.inf

    for t in candidates:
        # Ensure at least min_points on each side
        n_left = np.sum(x <= t)
        n_right = np.sum(x > t)
        if n_left < min_points or n_right < min_points:
            continue
        try:
            result = fit_func(x, y, t)
            if result["rss"] < best_rss:
                best_rss = result["rss"]
                best_fit = result
        except Exception:
            continue

    return best_fit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sar_threshold(
    data: pd.DataFrame,
    models: list[str] | None = None,
    interval: float | None = None,
    log_area: bool = True,
) -> ThresholdFit:
    """Fit threshold / piecewise SAR models.

    Fits up to three piecewise models to the data and selects the best
    by AICc:

    - **ContOne**: continuous two-slope model with one breakpoint
    - **ZslopeOne**: left-horizontal + right slope (small island effect)
    - **Linear**: simple linear model (no breakpoint, baseline)

    By default the models are fitted on the natural log of area, as R sars
    does (``logAreaT = TRUE``); ``log_area=False`` fits them on area itself.
    The breakpoint is reported on the fitted scale (``best_breakpoint`` and
    the ``breakpoint`` column of ``summary``) and in area units
    (``breakpoint_area``).

    Breakpoint selection uses a two-level procedure. Within each piecewise
    model class (ContOne, ZslopeOne), a grid search over the fitted scale
    selects the breakpoint that minimises residual sum of squares (RSS). Then,
    across model classes — including the no-breakpoint Linear baseline —
    the best candidate from each class is ranked by AICc to choose the
    overall best model. Because the models differ in parameter count
    (Linear: k=3, ZslopeOne: k=4, ContOne: k=5), AICc penalises
    complexity appropriately at the cross-model level.

    Meaningful threshold detection requires at least 7 observations. The
    breakpoint grid search needs a minimum of 3 data points on each side
    of a candidate breakpoint (``min_points=3``), so with fewer than 7
    observations the piecewise models may find zero valid candidates and
    only the Linear baseline will be returned. A ``UserWarning`` is
    emitted when ``n < 7``.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    models : list[str], optional
        Which models to fit. Default is all three:
        ["ContOne", "ZslopeOne", "Linear"].
    interval : float, optional
        Grid spacing for the breakpoint search, on the fitted scale. Default:
        0.1 on the natural-log scale, and a thousandth of the breakpoint window (from
        the third-smallest to the third-largest area) when ``log_area`` is
        False. Island areas are right-skewed, so on the area scale even that
        grid is coarse among the small islands, where a threshold would be;
        the log scale is the default for that reason.
    log_area : bool
        Fit on the natural log of area (True, the default) or on area itself.

    Returns
    -------
    ThresholdFit
        Result with best model, breakpoint, segments, and comparison table.

    Warns
    -----
    UserWarning
        If the dataset has fewer than 7 observations.

    References
    ----------
    Matthews TJ & Rigal F (2021). Thresholds and the species-area
    relationship. Frontiers of Biogeography 13(1), e49404.
    """
    if models is None:
        models = ["ContOne", "ZslopeOne", "Linear"]

    valid_models = {"ContOne", "ZslopeOne", "Linear"}
    for m in models:
        if m not in valid_models:
            raise ValueError(f"Unknown threshold model: {m!r}. Choose from {sorted(valid_models)}")

    _validate_data(data)

    area = np.asarray(data["area"], dtype=float)
    species = np.asarray(data["species"], dtype=float)
    n = len(area)

    if n < 7:
        warnings.warn(
            f"Dataset has {n} observations. Threshold detection generally "
            f"requires n >= 7 (at least 3 points on each side of a "
            f"breakpoint). Piecewise models may not be estimable and only "
            f"the Linear baseline may be returned.",
            UserWarning,
            stacklevel=2,
        )

    if interval is not None and not (np.isfinite(interval) and interval > 0):
        raise ValueError(f"interval must be positive and finite, got {interval!r}")

    x = np.log(area) if log_area else area
    y = species
    if interval is None and log_area:
        interval = 0.1  # on the area scale the grid search sizes it from its window

    all_fits: dict[str, dict] = {}

    if "Linear" in models:
        all_fits["Linear"] = _fit_linear(x, y)

    if "ContOne" in models:
        result = _grid_search_breakpoint(x, y, _fit_cont_one, interval)
        if result is not None:
            all_fits["ContOne"] = result

    if "ZslopeOne" in models:
        result = _grid_search_breakpoint(x, y, _fit_zslope_one, interval)
        if result is not None:
            all_fits["ZslopeOne"] = result

    if not all_fits:
        raise RuntimeError("No threshold models could be fitted to the data.")

    # Build summary table
    rows = []
    for name, fit in all_fits.items():
        rows.append(
            {
                "model": name,
                "breakpoint": fit["breakpoint"],
                "breakpoint_area": _to_area(fit["breakpoint"], log_area),
                "AIC": fit["aic"],
                "AICc": fit["aicc"],
                "BIC": fit["bic"],
                "R2": fit["r2"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)

    # Select best by AICc
    best_name = summary.iloc[0]["model"]
    best = all_fits[best_name]

    return ThresholdFit(
        best_model=best_name,
        best_breakpoint=best["breakpoint"],
        best_segments=best["segments"],
        summary=summary,
        all_fits=all_fits,
        data=data,
        log_area=log_area,
    )
