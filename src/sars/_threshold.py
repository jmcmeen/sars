"""Threshold / piecewise SAR models.

Implements the one-threshold piecewise regression approach described in
Matthews & Rigal (2021), Frontiers of Biogeography 13(1), e49404.

The models are R sars's:
- ContOne: continuous two-slope (breakpoint where slope changes)
- ZslopeOne: left-horizontal + right slope (small island effect)
- DiscOne: discontinuous, a separate line either side of the breakpoint
- ContTwo: continuous, three joined segments
- ZslopeTwo: left-horizontal, then two joined sloping segments
- DiscTwo: discontinuous, a separate line in each of three segments
- Linear: simple linear baseline (no breakpoint)

Breakpoint selection is a two-level process: within each piecewise model
class, a search selects the breakpoint (or pair) that minimises RSS; then
across model classes (including the no-breakpoint Linear baseline), AICc
ranks the best candidate from each class to choose the overall winner. The
continuous and left-horizontal models search a grid over the natural log of
area (or over area itself with ``log_area=False``); the discontinuous models
put their breakpoints at observed areas, as in R.

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
        Name of the best model by AICc: one of the names in ``sar_threshold``'s
        ``models``.
    best_breakpoint : float | None
        Breakpoint on the fitted scale: the natural log of area when ``log_area``
        is True (the default), area itself otherwise. None for the linear model;
        the first breakpoint for a two-threshold model. ``breakpoint_area`` is
        the same point in area units.
    best_segments : list[dict]
        Segment parameters as list of dicts with keys 'intercept', 'slope',
        'x_min', 'x_max', on the fitted scale, in order along it. Each segment
        after the first applies above its ``x_min``.
    summary : pd.DataFrame
        Comparison table with columns: model, breakpoint (fitted scale),
        breakpoint_area (area units), breakpoint2 and breakpoint2_area (the
        second breakpoint of a two-threshold model, NaN otherwise), AIC, AICc,
        BIC, R2.
    all_fits : dict[str, dict]
        Detailed fit results keyed by model name.
    data : pd.DataFrame
        Original data (columns: area, species).
    log_area : bool
        Whether the models were fitted on the natural log of area (True) or on
        area itself (False).
    best_breakpoints : tuple[float, ...]
        All of the best model's breakpoints on the fitted scale, in order: none
        for the linear model, two for a two-threshold model.
        ``breakpoints_area`` gives them in area units.
    """

    best_model: str
    best_breakpoint: float | None
    best_segments: list[dict]
    summary: pd.DataFrame = field(repr=False)
    all_fits: dict[str, dict] = field(repr=False)
    data: pd.DataFrame = field(repr=False)
    log_area: bool = True
    best_breakpoints: tuple[float, ...] = ()

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
        areas = self.breakpoints_area
        return areas[0] if areas else None

    @property
    def breakpoints_area(self) -> tuple[float, ...]:
        """All of the best model's breakpoints in area units. A discontinuous
        model's are observed areas, as given in ``data``."""
        stored = self.all_fits.get(self.best_model, {}).get("breakpoints_area")
        if stored is not None:
            return tuple(stored)
        return tuple(float(np.exp(t)) if self.log_area else float(t) for t in self.best_breakpoints)

    def __post_init__(self) -> None:
        if self.best_breakpoint is not None and not self.best_breakpoints:
            self.best_breakpoints = (self.best_breakpoint,)

    def __repr__(self) -> str:
        if self.best_breakpoint is None:
            return f"ThresholdFit(best='{self.best_model}')"
        if len(self.best_breakpoints) > 1:
            in_area = ", ".join(f"{t:.4g}" for t in self.breakpoints_area)
            if not self.log_area:
                return f"ThresholdFit(best='{self.best_model}', breakpoints_area=({in_area}))"
            on_log = ", ".join(f"{t:.3f}" for t in self.best_breakpoints)
            return (
                f"ThresholdFit(best='{self.best_model}', "
                f"breakpoints=({on_log}) (ln area), breakpoints_area=({in_area}))"
            )
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
    after the first applies above its ``x_min`` (a point at a breakpoint
    belongs to the segment on its left, as in the discontinuous fits; the
    continuous segments meet there), so the first one's line continues below
    the fitted range and the last one's above it, as the SAR models' ``predict``
    extrapolates their curves. A NaN input gives NaN.
    """
    first = segments[0]
    pred = first["intercept"] + first["slope"] * x
    for seg in segments[1:]:
        pred = np.where(x > seg["x_min"], seg["intercept"] + seg["slope"] * x, pred)
    return np.asarray(pred, dtype=float)


# ---------------------------------------------------------------------------
# Individual piecewise model fits
# ---------------------------------------------------------------------------

# Parameters counted for AIC, AICc and BIC, as R sars counts them: each
# coefficient, each breakpoint, and sigma.
_K = {
    "ContOne": 5,
    "ZslopeOne": 4,
    "DiscOne": 6,
    "ContTwo": 7,
    "ZslopeTwo": 6,
    "DiscTwo": 9,
    "Linear": 3,
}
MODELS = tuple(_K)
_DEFAULT_MODELS = ("ContOne", "ZslopeOne", "Linear")


def _result(
    model: str,
    y: np.ndarray,
    pred: np.ndarray,
    breakpoints: tuple[float, ...],
    coeffs: dict[str, float],
    segments: list[dict],
) -> dict:
    """A fit's result dict: its coefficients, segments, RSS, R2 and information criteria."""
    n = len(y)
    rss = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
    k = _K[model]
    aic, aicc, bic = _compute_ic(k, n, rss)
    return {
        "model": model,
        "breakpoint": breakpoints[0] if breakpoints else None,
        "breakpoints": breakpoints,
        "coeffs": coeffs,
        "segments": segments,
        "rss": rss,
        "r2": r2,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "k": k,
        "n": n,
    }


def _fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit simple linear model y = c + m*x (no breakpoint).

    Parameters: c (intercept), m (slope) + sigma = 3 for AICc.
    """
    x_mat = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    c, m = (float(v) for v in coeffs)
    segments = [{"intercept": c, "slope": m, "x_min": float(np.min(x)), "x_max": float(np.max(x))}]
    return _result("Linear", y, x_mat @ coeffs, (), {"c": c, "m": m}, segments)


def _hinge(x: np.ndarray, threshold: float) -> np.ndarray:
    """(x - T) * I(x > T): zero up to the breakpoint, then rising with slope 1."""
    return np.where(x > threshold, x - threshold, 0.0)


def _fit_hinged(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, ...],
    sloped_first: bool,
    names: tuple[str, ...],
) -> dict:
    """Fit a continuous piecewise line: y = c + z*x + sum_i d_i * (x - T_i) * I(x > T_i).

    ``sloped_first`` False drops the ``z*x`` term, so the first segment is flat
    (the left-horizontal models). Each breakpoint adds ``d_i`` to the slope.
    """
    columns = [np.ones(len(x))] + ([x] if sloped_first else [])
    columns += [_hinge(x, t) for t in thresholds]
    x_mat = np.column_stack(columns)
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)

    intercept = float(coeffs[0])
    slope = float(coeffs[1]) if sloped_first else 0.0
    deltas = coeffs[2:] if sloped_first else coeffs[1:]
    edges = [float(np.min(x)), *(float(t) for t in thresholds), float(np.max(x))]
    segments = []
    for j in range(len(thresholds) + 1):
        if j > 0:  # the line bends at T_j: the slope gains d_j, keeping the line continuous
            slope += float(deltas[j - 1])
            intercept -= float(deltas[j - 1]) * float(thresholds[j - 1])
        segments.append(
            {"intercept": intercept, "slope": slope, "x_min": edges[j], "x_max": edges[j + 1]}
        )
    return _result(
        model,
        y,
        x_mat @ coeffs,
        tuple(float(t) for t in thresholds),
        {name: float(v) for name, v in zip(names, coeffs, strict=True)},
        segments,
    )


def _fit_cont_one(x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Continuous one-threshold model: y = c1 + z1*x + z_delta*(x - T)*I(x > T).

    Parameters: c1, z1, z_delta + breakpoint + sigma = 5 for AICc.
    """
    return _fit_hinged("ContOne", x, y, (threshold,), True, ("c1", "z1", "z_delta"))


def _fit_zslope_one(x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Left-horizontal one-threshold model (small island effect): y = c1 + z2*(x - T)*I(x > T).

    Parameters: c1, z2 + breakpoint + sigma = 4 for AICc.
    """
    return _fit_hinged("ZslopeOne", x, y, (threshold,), False, ("c1", "z2"))


def _fit_cont_two(x: np.ndarray, y: np.ndarray, t1: float, t2: float) -> dict:
    """Continuous two-threshold model:
    y = c1 + z1*x + z_delta1*(x - T1)*I(x > T1) + z_delta2*(x - T2)*I(x > T2).

    Parameters: c1, z1, z_delta1, z_delta2 + 2 breakpoints + sigma = 7 for AICc.
    """
    return _fit_hinged("ContTwo", x, y, (t1, t2), True, ("c1", "z1", "z_delta1", "z_delta2"))


def _fit_zslope_two(x: np.ndarray, y: np.ndarray, t1: float, t2: float) -> dict:
    """Left-horizontal two-threshold model:
    y = c1 + z2*(x - T1)*I(x > T1) + z_delta*(x - T2)*I(x > T2).

    Parameters: c1, z2, z_delta + 2 breakpoints + sigma = 6 for AICc.
    """
    return _fit_hinged("ZslopeTwo", x, y, (t1, t2), False, ("c1", "z2", "z_delta"))


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares intercept and slope; a flat line through the mean when x has one value."""
    if len(np.unique(x)) < 2:
        return float(np.mean(y)), 0.0
    coeffs, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _fit_disc(model: str, x: np.ndarray, y: np.ndarray, thresholds: tuple[float, ...]) -> dict:
    """Fit a separate line to each segment; a point at a breakpoint is in the segment to its left.

    The coefficients are each segment's intercept and slope: c1, z1, c2, z2, ...
    A segment holding a single area gets a flat line through its mean, as R's
    rank-deficient ``lm`` predicts.
    """
    cuts = [-np.inf, *thresholds, np.inf]
    edges = [float(np.min(x)), *(float(t) for t in thresholds), float(np.max(x))]
    pred = np.empty(len(x))
    coeffs: dict[str, float] = {}
    segments = []
    for j in range(len(cuts) - 1):
        inside = (x > cuts[j]) & (x <= cuts[j + 1])
        intercept, slope = _line(x[inside], y[inside])
        pred[inside] = intercept + slope * x[inside]
        coeffs[f"c{j + 1}"], coeffs[f"z{j + 1}"] = intercept, slope
        segments.append(
            {"intercept": intercept, "slope": slope, "x_min": edges[j], "x_max": edges[j + 1]}
        )
    return _result(model, y, pred, tuple(float(t) for t in thresholds), coeffs, segments)


def _fit_disc_one(x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Discontinuous one-threshold model: a separate line either side of T.

    Parameters: 2 intercepts, 2 slopes + breakpoint + sigma = 6 for AICc.
    """
    return _fit_disc("DiscOne", x, y, (threshold,))


def _fit_disc_two(x: np.ndarray, y: np.ndarray, t1: float, t2: float) -> dict:
    """Discontinuous two-threshold model: a separate line in each of three segments.

    Parameters: 3 intercepts, 3 slopes + 2 breakpoints + sigma = 9 for AICc.
    """
    return _fit_disc("DiscTwo", x, y, (t1, t2))


# ---------------------------------------------------------------------------
# Breakpoint searches
# ---------------------------------------------------------------------------


# Each candidate is a least-squares fit; past this the search would seem to hang.
_MAX_CANDIDATES = 100_000
# The two-threshold searches score the second breakpoints for each first one
# together; past this many pairs they too would seem to hang.
_MAX_PAIRS = 10_000_000
# Most (observation, candidate) cells a ContTwo/ZslopeTwo block scores at once
# (8 MB per temporary array), so memory stays bounded however large the data.
_BLOCK = 1_000_000


def _breakpoint_grid(x: np.ndarray, interval: float | None, min_points: int) -> np.ndarray | None:
    """Grid of candidate breakpoints for the continuous models, on the scale of ``x``.

    It runs from the ``min_points``-th smallest value to the ``min_points``-th
    largest in steps of ``interval`` (None: a thousandth of that window), as R
    sars's ``nisl`` window does. None when the window is empty.
    """
    x_sorted = np.sort(x)
    x_min_bp = x_sorted[min_points - 1]
    x_max_bp = x_sorted[-(min_points)]

    if x_min_bp >= x_max_bp:
        return None
    if interval is None:
        interval = (x_max_bp - x_min_bp) / 1000
        if not interval > 0:  # a window too narrow for float steps: no breakpoint
            return None

    n_candidates = (x_max_bp - x_min_bp) / interval + 1
    if n_candidates > _MAX_CANDIDATES:
        raise ValueError(
            f"interval={interval!r} gives {n_candidates:.3g} breakpoint candidates "
            f"(at most {_MAX_CANDIDATES:,}); use a larger interval."
        )
    candidates = np.arange(x_min_bp, x_max_bp + interval / 2, interval)
    return candidates if len(candidates) else None


def _check_pairs(n_pairs: float, hint: str) -> None:
    if n_pairs > _MAX_PAIRS:
        raise ValueError(
            f"The two-threshold search would score {n_pairs:.3g} pairs of breakpoints "
            f"(at most {_MAX_PAIRS:,}); {hint}."
        )


def _grid_search_breakpoint(
    x: np.ndarray,
    y: np.ndarray,
    fit_func: Callable[[np.ndarray, np.ndarray, float], dict],
    interval: float | None,
    min_points: int = 3,
) -> dict | None:
    """Search the grid for the one breakpoint that minimises RSS.

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
    candidates = _breakpoint_grid(x, interval, min_points)
    if candidates is None:
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


def _hinge_rss(x: np.ndarray, q: np.ndarray, r_y: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """RSS of the T1 fit (orthonormal basis ``q``, residuals ``r_y``) plus a hinge at each ``t2``."""
    hinges = np.maximum(x[:, None] - t2[None, :], 0.0)
    r_h = hinges - q @ (q.T @ hinges)
    norms = np.sum(r_h**2, axis=0)
    # A hinge the T1 fit can already make adds nothing. The cut is far below
    # where its remainder stops being resolvable: rounding leaves about 1e-16
    # of the hinge in r_h, so at 1e-10 of it the gain is still good to ~1e-5.
    usable = norms > 1e-20 * np.sum(hinges**2, axis=0)
    gain = np.zeros(len(t2))
    gain[usable] = (r_y @ r_h[:, usable]) ** 2 / norms[usable]
    return r_y @ r_y - gain


def _grid_search_two_breakpoints(
    x: np.ndarray,
    y: np.ndarray,
    fit_func: Callable[[np.ndarray, np.ndarray, float, float], dict],
    sloped_first: bool,
    interval: float | None,
    min_points: int = 3,
) -> dict | None:
    """Search the grid for the pair of breakpoints T1 < T2 that minimises RSS.

    The first segment (up to T1) and the last (above T2) hold at least
    ``min_points`` values each; the middle one may hold any number, as in R
    sars. For each T1, every T2 is scored at once: the RSS falls from that of
    the fit without the T2 term by (r_y . r_h)^2 / (r_h . r_h), where r_y and
    r_h are y and the T2 hinge with the T1 fit's columns projected out. The
    winning pair is then refitted with ``fit_func``.
    """
    grid = _breakpoint_grid(x, interval, min_points)
    if grid is None:
        return None
    x_sorted = np.sort(x)
    n_left = np.searchsorted(x_sorted, grid, side="right")
    first_ok = n_left >= min_points
    last_ok = len(x) - n_left >= min_points
    m = len(grid)
    _check_pairs(m * (m - 1) / 2, "use a larger interval")

    best_rss, best_pair = np.inf, None
    for i in np.nonzero(first_ok)[0]:
        js = np.nonzero(last_ok[i + 1 :])[0] + i + 1
        if not len(js):
            continue
        columns = [np.ones(len(x))] + ([x] if sloped_first else []) + [_hinge(x, grid[i])]
        u, s, _ = np.linalg.svd(np.column_stack(columns), full_matrices=False)
        q = u[:, s > s[0] * 1e-10]  # an orthonormal basis for the T1 fit's columns
        r_y = y - q @ (q.T @ y)
        rss = np.concatenate(
            [
                _hinge_rss(x, q, r_y, grid[block])
                for block in np.array_split(js, -(-len(js) * len(x) // _BLOCK))
            ]
        )
        j = int(np.argmin(rss))
        if rss[j] < best_rss:
            best_rss, best_pair = rss[j], (grid[i], grid[js[j]])

    if best_pair is None:
        return None
    return fit_func(x, y, *best_pair)


class _Runs:
    """Least-squares lines through runs [start, stop) of the data sorted by x.

    Scores come from running sums of 1, x, y, x^2, xy and y^2 (centred on the
    data's means), so any run costs O(1). Those sums lose precision for a run
    whose spread in x is tiny next to its distance from the mean of x (a
    cluster of areas far from the rest, say); such runs are fitted directly.
    """

    # Below this ratio of a run's spread in x to its sum of squares about the
    # mean, the running-sum spread has lost too many digits to trust.
    _RELIABLE = 1e-8

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        order = np.argsort(x, kind="stable")
        self.xs, self.ys = x[order], y[order]
        xc, yc = self.xs - self.xs.mean(), self.ys - self.ys.mean()
        terms = np.column_stack([np.ones(len(xc)), xc, yc, xc**2, xc * yc, yc**2])
        self.sums = np.vstack([np.zeros(6), np.cumsum(terms, axis=0)])
        self._exact: dict[tuple[int, int], float] = {}

    def rss(self, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
        """RSS of the line through each run. A run with one distinct x
        contributes its spread about the mean, as a flat line does."""
        d = self.sums[stop] - self.sums[start]
        count, sx, sy, sxx, sxy, syy = (d[..., k] for k in range(6))
        with np.errstate(divide="ignore", invalid="ignore"):
            cxx = sxx - sx**2 / count
            cxy = sxy - sx * sy / count
            cyy = syy - sy**2 / count
            reliable = cxx > self._RELIABLE * sxx
            explained = np.where(reliable, cxy**2 / np.where(reliable, cxx, 1.0), 0.0)
        rss = np.maximum(cyy - explained, 0.0)
        for k in np.nonzero(~reliable & (count > 1))[0]:
            rss[k] = self._fit(int(start[k]), int(stop[k]))
        return rss

    def _fit(self, start: int, stop: int) -> float:
        key = (start, stop)
        if key not in self._exact:
            xs, ys = self.xs[start:stop], self.ys[start:stop]
            intercept, slope = _line(xs - xs.mean(), ys)
            self._exact[key] = float(np.sum((ys - intercept - slope * (xs - xs.mean())) ** 2))
        return self._exact[key]


def _disc_candidates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, _Runs]:
    """The distinct values of ``x`` (the discontinuous models' candidate
    breakpoints), how many points lie at or below each, and the runs to score.
    """
    runs = _Runs(x, y)
    values = np.unique(runs.xs)
    n_left = np.searchsorted(runs.xs, values, side="right")
    return values, n_left, runs


def _search_disc_one(x: np.ndarray, y: np.ndarray, min_points: int = 3) -> dict | None:
    """The observed area, with at least ``min_points`` values on each side, that
    minimises the discontinuous one-threshold model's RSS."""
    values, n_left, runs = _disc_candidates(x, y)
    n = len(x)
    ok = (n_left >= min_points) & (n - n_left >= min_points)
    if not ok.any():
        return None
    cut = n_left[ok]
    rss = runs.rss(np.zeros_like(cut), cut) + runs.rss(cut, np.full_like(cut, n))
    return _fit_disc_one(x, y, values[ok][int(np.argmin(rss))])


def _search_disc_two(x: np.ndarray, y: np.ndarray, min_points: int = 3) -> dict | None:
    """The pair of observed areas T1 < T2 that minimises the discontinuous
    two-threshold model's RSS, with at least ``min_points`` values up to T1 and
    above T2 (the middle segment holds at least T2 itself). For each T1, every
    T2 is scored at once from the running sums."""
    values, n_left, runs = _disc_candidates(x, y)
    n = len(x)
    lasts = np.nonzero(n - n_left >= min_points)[0]
    firsts = np.nonzero(n_left >= min_points)[0]
    firsts = firsts[firsts < (lasts[-1] if len(lasts) else -1)]
    n_pairs = len(lasts) * len(firsts) - int(np.searchsorted(lasts, firsts, side="right").sum())
    _check_pairs(n_pairs, "the dataset has too many distinct areas")

    best_rss, best_pair = np.inf, None
    for i in firsts:
        js = lasts[lasts > i]
        a, b = n_left[i], n_left[js]
        rss = (
            runs.rss(np.array([0]), np.array([a]))[0]
            + runs.rss(np.full_like(b, a), b)
            + runs.rss(b, np.full_like(b, n))
        )
        w = int(np.argmin(rss))
        if rss[w] < best_rss:
            best_rss, best_pair = rss[w], (values[i], values[js[w]])
    if best_pair is None:
        return None
    return _fit_disc_two(x, y, *best_pair)


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

    Fits piecewise models to the data and selects the best by AICc. The
    models are R sars's:

    - **ContOne**: continuous two-slope model with one breakpoint
    - **ZslopeOne**: left-horizontal + right slope (small island effect)
    - **DiscOne**: discontinuous, a separate line either side of one breakpoint
    - **ContTwo**: continuous, three joined segments (two breakpoints)
    - **ZslopeTwo**: left-horizontal first segment, then two joined sloping ones
    - **DiscTwo**: discontinuous, a separate line in each of three segments
    - **Linear**: simple linear model (no breakpoint, baseline)

    ContOne, ZslopeOne and Linear are fitted unless ``models`` says otherwise.

    By default the models are fitted on the natural log of area, as R sars
    does (``logAreaT = TRUE``); ``log_area=False`` fits them on area itself.
    The breakpoint is reported on the fitted scale (``best_breakpoint`` and
    the ``breakpoint`` column of ``summary``) and in area units
    (``breakpoint_area``).

    Breakpoint selection uses a two-level procedure. Within each piecewise
    model class, a search selects the breakpoint (or pair of breakpoints) that
    minimises residual sum of squares (RSS): the continuous and left-horizontal
    models search a grid over the fitted scale, and the discontinuous models
    search the observed areas, since their breakpoints must sit at a data
    point (``interval`` doesn't apply to them). Then, across model classes —
    including the no-breakpoint Linear baseline — the best candidate from each
    class is ranked by AICc to choose the overall best model. The parameter
    counts are R sars's, each coefficient, breakpoint and sigma counting one
    (Linear: k=3, ZslopeOne: 4, ContOne: 5, DiscOne: 6, ZslopeTwo: 6,
    ContTwo: 7, DiscTwo: 9), so AICc penalises complexity at the cross-model
    level.

    Meaningful threshold detection requires at least 7 observations. The
    first and last segments must each hold at least 3 data points (R sars's
    ``nisl = 3``); a two-threshold model's middle segment may hold any number,
    as in R. With fewer than 7 observations the piecewise models may find no
    valid candidates and only the Linear baseline will be returned. A
    ``UserWarning`` is emitted when ``n < 7``. R sars advises against the
    two-threshold models for datasets of fewer than about 20 islands.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    models : list[str], optional
        Which models to fit, from "ContOne", "ZslopeOne", "DiscOne", "ContTwo",
        "ZslopeTwo", "DiscTwo" and "Linear". Default: ["ContOne", "ZslopeOne",
        "Linear"].
    interval : float, optional
        Grid spacing for the breakpoint search, on the fitted scale. Default:
        0.01 on the natural-log scale (R sars's default), and a thousandth of
        the breakpoint window (from the third-smallest to the third-largest
        area) when ``log_area`` is False. At most 100,000 grid points, and at
        most 10 million pairs of breakpoints for ContTwo and ZslopeTwo. (DiscTwo
        has the same limit on pairs of observed areas, which ``interval``
        doesn't change.) Island areas are right-skewed, so on the area scale even that
        grid is coarse among the small islands, where a threshold would be;
        the log scale is the default for that reason.
    log_area : bool
        Fit on the natural log of area (True, the default) or on area itself.

    Returns
    -------
    ThresholdFit
        Result with best model, breakpoint, segments, and comparison table.

    Raises
    ------
    ValueError
        If AICc is undefined for every model fitted (n <= k + 1 for each).

    Warns
    -----
    UserWarning
        If the dataset has fewer than 7 observations, or AICc is undefined for
        some of the models fitted (they rank last), as R sars's ``summary``
        warns.

    References
    ----------
    Matthews TJ & Rigal F (2021). Thresholds and the species-area
    relationship. Frontiers of Biogeography 13(1), e49404.
    """
    if models is None:
        models = list(_DEFAULT_MODELS)

    for m in models:
        if m not in _K:
            raise ValueError(f"Unknown threshold model: {m!r}. Choose from {sorted(_K)}")

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
        interval = 0.01  # as R sars; on the area scale the grid search sizes it from its window

    searches: dict[str, Callable[[], dict | None]] = {
        "Linear": lambda: _fit_linear(x, y),
        "ContOne": lambda: _grid_search_breakpoint(x, y, _fit_cont_one, interval),
        "ZslopeOne": lambda: _grid_search_breakpoint(x, y, _fit_zslope_one, interval),
        "DiscOne": lambda: _search_disc_one(x, y),
        "ContTwo": lambda: _grid_search_two_breakpoints(x, y, _fit_cont_two, True, interval),
        "ZslopeTwo": lambda: _grid_search_two_breakpoints(x, y, _fit_zslope_two, False, interval),
        "DiscTwo": lambda: _search_disc_two(x, y),
    }
    all_fits: dict[str, dict] = {}
    for name in searches:  # a fixed order, whatever order ``models`` gives
        if name in models:
            result = searches[name]()
            if result is not None:
                if name.startswith("Disc"):  # at observed areas: report them as given
                    result["breakpoints_area"] = tuple(
                        float(area[x == t][0]) for t in result["breakpoints"]
                    )
                else:
                    result["breakpoints_area"] = tuple(
                        _to_area(t, log_area) for t in result["breakpoints"]
                    )
                all_fits[name] = result

    if not all_fits:
        raise RuntimeError("No threshold models could be fitted to the data.")

    # Build summary table
    rows = []
    for name, fit in all_fits.items():
        second = fit["breakpoints"][1] if len(fit["breakpoints"]) > 1 else np.nan
        rows.append(
            {
                "model": name,
                "breakpoint": fit["breakpoint"],
                "breakpoint_area": fit["breakpoints_area"][0] if fit["breakpoints"] else None,
                "breakpoint2": second,
                "breakpoint2_area": fit["breakpoints_area"][1]
                if len(fit["breakpoints"]) > 1
                else np.nan,
                "AIC": fit["aic"],
                "AICc": fit["aicc"],
                "BIC": fit["bic"],
                "R2": fit["r2"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)

    undefined = summary.loc[np.isinf(summary["AICc"]), "model"].tolist()
    if len(undefined) == len(summary):
        raise ValueError(
            f"AICc is undefined for every model fitted ({', '.join(undefined)}): with "
            f"{n} observations, a model of k parameters needs n > k + 1. Include a "
            f"model with fewer parameters (such as Linear) or use more data."
        )
    if undefined:
        warnings.warn(
            f"AICc is undefined for {', '.join(undefined)} with {n} observations "
            f"(a model of k parameters needs n > k + 1); they rank last.",
            UserWarning,
            stacklevel=2,
        )

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
        best_breakpoints=best["breakpoints"],
    )
