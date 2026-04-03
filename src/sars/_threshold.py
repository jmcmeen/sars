"""Threshold / piecewise SAR models.

Implements the one-threshold piecewise regression approach described in
Matthews & Rigal (2021), Frontiers of Biogeography 13(1), e49404.

Three models are supported:
- ContOne: continuous two-slope (breakpoint where slope changes)
- ZslopeOne: left-horizontal + right slope (small island effect)
- Linear: simple linear baseline (no breakpoint)

Breakpoints are found by grid search over log(area), selecting by AICc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sars._models import _compute_ic

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
        Breakpoint in log(area) space, or None for the linear model.
    best_segments : list[dict]
        Segment parameters as list of dicts with keys 'intercept', 'slope',
        'x_min', 'x_max' (in log-area space).
    summary : pd.DataFrame
        Comparison table with columns: model, breakpoint, AIC, AICc, BIC, R2.
    all_fits : dict[str, dict]
        Detailed fit results keyed by model name.
    data : pd.DataFrame
        Original data (columns: area, species).
    log_area : bool
        Whether area was log-transformed (always True currently).
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
            Predicted species richness.
        """
        area = np.asarray(area, dtype=float)
        x = np.log(area) if self.log_area else area
        return _predict_segments(x, self.best_segments)

    def __repr__(self) -> str:
        bp = (f", breakpoint={self.best_breakpoint:.3f}"
              if self.best_breakpoint is not None else "")
        return f"ThresholdFit(best='{self.best_model}'{bp})"


def _predict_segments(
    x: np.ndarray, segments: list[dict]
) -> np.ndarray:
    """Predict from piecewise linear segments."""
    pred = np.zeros_like(x, dtype=float)
    for seg in segments:
        mask = (x >= seg["x_min"]) & (x <= seg["x_max"])
        pred[mask] = seg["intercept"] + seg["slope"] * x[mask]
    return pred


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

    segments = [{
        "intercept": float(c),
        "slope": float(m),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
    }]

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


def _fit_cont_one(
    x: np.ndarray, y: np.ndarray, threshold: float
) -> dict:
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
            "c1": float(c1), "z1": float(z1), "z_delta": float(z_delta),
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


def _fit_zslope_one(
    x: np.ndarray, y: np.ndarray, threshold: float
) -> dict:
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
    fit_func: callable,
    interval: float = 0.1,
    min_points: int = 3,
) -> dict | None:
    """Search for the breakpoint that minimises RSS.

    Parameters
    ----------
    x : np.ndarray
        Predictor values (log-area).
    y : np.ndarray
        Response values (species).
    fit_func : callable
        One of _fit_cont_one or _fit_zslope_one.
    interval : float
        Grid spacing for breakpoint candidates.
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
    interval: float = 0.1,
) -> ThresholdFit:
    """Fit threshold / piecewise SAR models.

    Fits up to three piecewise models to the data and selects the best
    by AICc:

    - **ContOne**: continuous two-slope model with one breakpoint
    - **ZslopeOne**: left-horizontal + right slope (small island effect)
    - **Linear**: simple linear model (no breakpoint, baseline)

    Area is log-transformed before fitting (consistent with R sars default).
    Breakpoints are found by grid search over log(area).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    models : list[str], optional
        Which models to fit. Default is all three:
        ["ContOne", "ZslopeOne", "Linear"].
    interval : float
        Grid spacing for breakpoint search in log(area) space.
        Default 0.1.

    Returns
    -------
    ThresholdFit
        Result with best model, breakpoint, segments, and comparison table.

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
            raise ValueError(
                f"Unknown threshold model: {m!r}. "
                f"Choose from {sorted(valid_models)}"
            )

    area = np.asarray(data["area"], dtype=float)
    species = np.asarray(data["species"], dtype=float)
    x = np.log(area)
    y = species

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
        rows.append({
            "model": name,
            "breakpoint": fit["breakpoint"],
            "AIC": fit["aic"],
            "AICc": fit["aicc"],
            "BIC": fit["bic"],
            "R2": fit["r2"],
        })
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
        log_area=True,
    )
