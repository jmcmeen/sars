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

``threshold_ci`` gives a fitted model's breakpoints bootstrap confidence
intervals (Toms & Lesperance 2003, Ecology 84, 2034-2041): the model's
residuals are resampled onto its fitted values and the breakpoint search is
rerun on each resample.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from operator import index
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import special

from sars._models import _compute_ic, _validate_data
from sars._parallel import Workers, chunks, n_workers

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
    interval : float | None
        Grid spacing the breakpoint search used, on the fitted scale. None
        means a thousandth of the breakpoint window (from the third-smallest to
        the third-largest area), which is what ``sar_threshold`` searches on
        the area scale unless told otherwise; on the log scale it stores its
        0.01. ``threshold_ci`` searches its resamples with the same spacing.
    """

    best_model: str
    best_breakpoint: float | None
    best_segments: list[dict]
    summary: pd.DataFrame = field(repr=False)
    all_fits: dict[str, dict] = field(repr=False)
    data: pd.DataFrame = field(repr=False)
    log_area: bool = True
    best_breakpoints: tuple[float, ...] = ()
    interval: float | None = None

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
# Bootstrap resamples searched together.
_BOOT_BATCH = 500
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


def _fit_basis(columns: list[np.ndarray]) -> np.ndarray:
    """An orthonormal basis for the span of a fit's columns."""
    u, s, _ = np.linalg.svd(np.column_stack(columns), full_matrices=False)
    return u[:, s > s[0] * 1e-10]


def _rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b``, each row of the result computed the same way whatever else
    is in ``a``. A BLAS matrix product can sum in a different order for a
    different number of rows on some CPUs, so a resample searched alone, in a
    worker's chunk or in the whole batch could score its candidates
    differently in the last bits, and a near-tie then pick a different
    breakpoint. ``einsum`` without ``optimize`` uses numpy's own loops."""
    return np.einsum("ij,jk->ik", a, b)


def _hinge_gains(x: np.ndarray, q: np.ndarray, r_ys: np.ndarray, t: np.ndarray) -> np.ndarray:
    """How far the RSS falls when a hinge at each of ``t`` joins a fit.

    ``q`` is an orthonormal basis for the fit's columns and each row of
    ``r_ys`` a response with that fit projected out. With r_h the hinge with
    the fit projected out too, the fall is (r_y . r_h)^2 / (r_h . r_h). One
    row per response, one column per hinge.
    """
    hinges = np.maximum(x[:, None] - t[None, :], 0.0)
    r_h = hinges - q @ (q.T @ hinges)
    norms = np.sum(r_h**2, axis=0)
    # A hinge the fit can already make adds nothing. The cut is far below
    # where its remainder stops being resolvable: rounding leaves about 1e-16
    # of the hinge in r_h, so at 1e-10 of it the gain is still good to ~1e-5.
    usable = norms > 1e-20 * np.sum(hinges**2, axis=0)
    gains = np.zeros((len(r_ys), len(t)))
    gains[:, usable] = _rowwise(r_ys, r_h[:, usable]) ** 2 / norms[usable]
    return gains


def _tie(ys: np.ndarray) -> np.ndarray:
    """For each row of ``ys``, the difference in RSS below which two candidates tie.

    Candidates can fit exactly alike (two breakpoints between the same pair
    of neighbouring areas, say), and rounding would then pick among them at
    random, differently for one response alone and for many together. Calling
    such fits equal, and taking the first, makes the choice the same each time.
    """
    return np.asarray(1e-10 * np.sum((ys - ys.mean(axis=1, keepdims=True)) ** 2, axis=1))


def _best_hinge(
    x: np.ndarray, q: np.ndarray, ys: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``ys``, the hinge among ``t`` that leaves the least RSS
    when added to the fit with basis ``q``: its index in ``t`` and that RSS.
    The first wins a tie. The candidates are scored in blocks, so memory stays
    bounded however many there are."""
    r_ys = ys - _rowwise(_rowwise(ys, q), q.T)
    rss_without = np.sum(r_ys**2, axis=1)
    tie = _tie(ys)
    rows = np.arange(len(ys))
    best_gain = np.full(len(ys), -np.inf)
    best = np.zeros(len(ys), dtype=int)
    n_blocks = -(-len(t) * max(len(x), len(ys)) // _BLOCK)
    for block in np.array_split(np.arange(len(t)), n_blocks):
        gains = _hinge_gains(x, q, r_ys, t[block])
        k = np.argmax(gains >= (gains.max(axis=1) - tie)[:, None], axis=1)
        better = gains[rows, k] > best_gain + tie
        best_gain[better] = gains[rows, k][better]
        best[better] = block[k[better]]
    return best, rss_without - best_gain


def _best_breakpoints(
    x: np.ndarray,
    ys: np.ndarray,
    sloped_first: bool,
    interval: float | None,
    min_points: int = 3,
) -> np.ndarray | None:
    """For each row of ``ys``, the grid breakpoint that minimises RSS.

    The candidates and what each can fit depend only on ``x``, so every
    response is scored against them together: the bootstrap's resamples cost
    little more than one search. At least ``min_points`` values lie on each
    side of a candidate. None when there is no candidate.
    """
    grid = _breakpoint_grid(x, interval, min_points)
    if grid is None:
        return None
    n_left = np.searchsorted(np.sort(x), grid, side="right")
    grid = grid[(n_left >= min_points) & (len(x) - n_left >= min_points)]
    if not len(grid):
        return None
    q = _fit_basis([np.ones(len(x))] + ([x] if sloped_first else []))
    best, _ = _best_hinge(x, q, ys, grid)
    return np.asarray(grid[best])


def _best_breakpoint_pairs(
    x: np.ndarray,
    ys: np.ndarray,
    sloped_first: bool,
    interval: float | None,
    min_points: int = 3,
) -> np.ndarray | None:
    """For each row of ``ys``, the pair of grid breakpoints T1 < T2 that
    minimises RSS: one row per response, T1 then T2.

    The first segment (up to T1) and the last (above T2) hold at least
    ``min_points`` values each; the middle one may hold any number, as in R
    sars. For each T1, every T2 is scored at once for every response, as the
    fall in RSS from adding the T2 hinge to the fit with T1 alone. None when
    no pair is valid.
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

    tie = _tie(ys)
    best_rss = np.full(len(ys), np.inf)
    pairs = np.full((len(ys), 2), np.nan)
    for i in np.nonzero(first_ok)[0]:
        js = np.nonzero(last_ok[i + 1 :])[0] + i + 1
        if not len(js):
            continue
        columns = [np.ones(len(x))] + ([x] if sloped_first else []) + [_hinge(x, grid[i])]
        j, rss = _best_hinge(x, _fit_basis(columns), ys, grid[js])
        better = rss < best_rss - tie
        best_rss[better] = rss[better]
        pairs[better, 0] = grid[i]
        pairs[better, 1] = grid[js[j[better]]]
    return None if np.isnan(pairs).any() else pairs


def _grid_search_breakpoint(
    x: np.ndarray,
    y: np.ndarray,
    fit_func: Callable[[np.ndarray, np.ndarray, float], dict],
    sloped_first: bool,
    interval: float | None,
) -> dict | None:
    """The fit at the one grid breakpoint that minimises RSS; None without a candidate.

    ``fit_func`` is _fit_cont_one (``sloped_first`` True) or _fit_zslope_one
    (False). ``interval`` is the grid spacing on the scale of ``x``; None for a
    thousandth of the window the candidates can lie in.
    """
    best = _best_breakpoints(x, y[None, :], sloped_first, interval)
    return None if best is None else fit_func(x, y, float(best[0]))


def _grid_search_two_breakpoints(
    x: np.ndarray,
    y: np.ndarray,
    fit_func: Callable[[np.ndarray, np.ndarray, float, float], dict],
    sloped_first: bool,
    interval: float | None,
) -> dict | None:
    """The fit at the pair of grid breakpoints that minimises RSS; None without a valid pair."""
    best = _best_breakpoint_pairs(x, y[None, :], sloped_first, interval)
    return None if best is None else fit_func(x, y, float(best[0, 0]), float(best[0, 1]))


class _Runs:
    """Least-squares lines through runs [start, stop) of the data sorted by x,
    for many responses at once (the rows of ``ys``).

    Scores come from running sums of 1, x, y, x^2, xy and y^2 (centred on the
    means), so any run costs O(1) per response. Those sums lose precision for
    a run whose spread in x is tiny next to its distance from the mean of x (a
    cluster of areas far from the rest, say); such runs are fitted directly.
    Whether a run is such depends on x alone, so it is the same for every
    response.
    """

    # Below this ratio of a run's spread in x to its sum of squares about the
    # mean, the running-sum spread has lost too many digits to trust.
    _RELIABLE = 1e-8

    def __init__(self, x: np.ndarray, ys: np.ndarray) -> None:
        order = np.argsort(x, kind="stable")
        self.xs, self.ys = x[order], np.atleast_2d(ys)[:, order]
        xc = self.xs - self.xs.mean()
        yc = self.ys - self.ys.mean(axis=1, keepdims=True)

        def running(terms: np.ndarray) -> np.ndarray:
            zeros = np.zeros(terms.shape[:-1] + (1,))
            return np.concatenate([zeros, np.cumsum(terms, axis=-1)], axis=-1)

        self.count, self.sx, self.sxx = (running(t) for t in (np.ones_like(xc), xc, xc**2))
        self.sy, self.sxy, self.syy = (running(t) for t in (yc, xc * yc, yc**2))
        self._exact: dict[tuple[int, int], np.ndarray] = {}

    def rss(self, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
        """RSS of the line through each run, for each response: one row per
        response, one column per run. A run with one distinct x contributes
        its spread about the mean, as a flat line does."""
        count = self.count[stop] - self.count[start]
        sx = self.sx[stop] - self.sx[start]
        sxx = self.sxx[stop] - self.sxx[start]
        sy = self.sy[:, stop] - self.sy[:, start]
        sxy = self.sxy[:, stop] - self.sxy[:, start]
        syy = self.syy[:, stop] - self.syy[:, start]
        with np.errstate(divide="ignore", invalid="ignore"):
            cxx = sxx - sx**2 / count
            reliable = cxx > self._RELIABLE * sxx
            cxy = sxy - sx * sy / count
            cyy = syy - sy**2 / count
            explained = np.where(reliable, cxy**2 / np.where(reliable, cxx, 1.0), 0.0)
        rss = np.maximum(cyy - explained, 0.0)
        for k in np.nonzero(~reliable & (count > 1))[0]:
            rss[:, k] = self._fit(int(start[k]), int(stop[k]))
        return rss

    def _fit(self, start: int, stop: int) -> np.ndarray:
        key = (start, stop)
        if key not in self._exact:
            xs = self.xs[start:stop] - self.xs[start:stop].mean()
            design = np.column_stack([np.ones_like(xs), xs])
            ys = self.ys[:, start:stop]
            # A run with one distinct x has a column of zeros: lstsq's
            # minimum-norm answer is then the flat line through its mean.
            coef = np.linalg.lstsq(design, ys.T, rcond=None)[0]
            self._exact[key] = np.sum((ys - (design @ coef).T) ** 2, axis=1)
        return self._exact[key]


def _disc_candidates(x: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, _Runs]:
    """The distinct values of ``x`` (the discontinuous models' candidate
    breakpoints), how many points lie at or below each, and the runs to score.
    """
    runs = _Runs(x, ys)
    values = np.unique(runs.xs)
    n_left = np.searchsorted(runs.xs, values, side="right")
    return values, n_left, runs


def _first_least(rss: np.ndarray, tie: np.ndarray) -> np.ndarray:
    """For each row of ``rss``, the first column within ``tie`` of the least."""
    return np.argmax(rss <= (rss.min(axis=1) + tie)[:, None], axis=1)


def _disc_one_breakpoints(x: np.ndarray, ys: np.ndarray, min_points: int = 3) -> np.ndarray | None:
    """For each row of ``ys``, the observed area, with at least ``min_points``
    values on each side, that minimises the discontinuous one-threshold
    model's RSS. None when no area has that many either side."""
    ys = np.atleast_2d(ys)
    values, n_left, runs = _disc_candidates(x, ys)
    n = len(x)
    ok = (n_left >= min_points) & (n - n_left >= min_points)
    if not ok.any():
        return None
    cut = n_left[ok]
    rss = runs.rss(np.zeros_like(cut), cut) + runs.rss(cut, np.full_like(cut, n))
    return np.asarray(values[ok][_first_least(rss, _tie(ys))])


def _disc_two_breakpoints(x: np.ndarray, ys: np.ndarray, min_points: int = 3) -> np.ndarray | None:
    """For each row of ``ys``, the pair of observed areas T1 < T2 that
    minimises the discontinuous two-threshold model's RSS, with at least
    ``min_points`` values up to T1 and above T2 (the middle segment holds at
    least T2 itself): one row per response, T1 then T2. For each T1, every T2
    is scored at once for every response from the running sums. None when no
    pair is valid."""
    ys = np.atleast_2d(ys)
    values, n_left, runs = _disc_candidates(x, ys)
    n = len(x)
    lasts = np.nonzero(n - n_left >= min_points)[0]
    firsts = np.nonzero(n_left >= min_points)[0]
    firsts = firsts[firsts < (lasts[-1] if len(lasts) else -1)]
    n_pairs = len(lasts) * len(firsts) - int(np.searchsorted(lasts, firsts, side="right").sum())
    _check_pairs(n_pairs, "the dataset has too many distinct areas")
    if not len(firsts):
        return None

    tie = _tie(ys)
    rows = np.arange(len(ys))
    best_rss = np.full(len(ys), np.inf)
    pairs = np.full((len(ys), 2), np.nan)
    for i in firsts:
        js = lasts[lasts > i]
        a, b = n_left[i], n_left[js]
        rss = (
            runs.rss(np.array([0]), np.array([a]))
            + runs.rss(np.full_like(b, a), b)
            + runs.rss(b, np.full_like(b, n))
        )
        w = _first_least(rss, tie)
        better = rss[rows, w] < best_rss - tie
        best_rss[better] = rss[rows, w][better]
        pairs[better, 0] = values[i]
        pairs[better, 1] = values[js[w[better]]]
    return None if np.isnan(pairs).any() else pairs


def _search_disc_one(x: np.ndarray, y: np.ndarray, min_points: int = 3) -> dict | None:
    """The discontinuous one-threshold fit at the best observed area (see
    ``_disc_one_breakpoints``), or None."""
    found = _disc_one_breakpoints(x, y, min_points)
    return None if found is None else _fit_disc_one(x, y, float(found[0]))


def _search_disc_two(x: np.ndarray, y: np.ndarray, min_points: int = 3) -> dict | None:
    """The discontinuous two-threshold fit at the best pair of observed areas
    (see ``_disc_two_breakpoints``), or None."""
    found = _disc_two_breakpoints(x, y, min_points)
    return None if found is None else _fit_disc_two(x, y, *(float(t) for t in found[0]))


class _Search(NamedTuple):
    """How a model's breakpoints are searched for.

    ``fit`` searches one response and returns the fit (None without a valid
    candidate). ``breakpoints``, for every model with a breakpoint, finds the
    breakpoints of many responses at once, one row each.
    """

    fit: Callable[[np.ndarray, np.ndarray, float | None], dict | None]
    breakpoints: Callable[[np.ndarray, np.ndarray, float | None], np.ndarray | None] | None = None


# Every model, in the order they are fitted: the baseline, then one threshold,
# then two. ``sar_threshold`` and ``threshold_ci`` both search through here,
# so a bootstrap resample is searched as the data were.
_SEARCHES: dict[str, _Search] = {
    "Linear": _Search(lambda x, y, interval: _fit_linear(x, y)),
    "ContOne": _Search(
        lambda x, y, interval: _grid_search_breakpoint(x, y, _fit_cont_one, True, interval),
        lambda x, ys, interval: _best_breakpoints(x, ys, True, interval),
    ),
    "ZslopeOne": _Search(
        lambda x, y, interval: _grid_search_breakpoint(x, y, _fit_zslope_one, False, interval),
        lambda x, ys, interval: _best_breakpoints(x, ys, False, interval),
    ),
    "DiscOne": _Search(
        lambda x, y, interval: _search_disc_one(x, y),
        lambda x, ys, interval: _disc_one_breakpoints(x, ys),
    ),
    "ContTwo": _Search(
        lambda x, y, interval: _grid_search_two_breakpoints(x, y, _fit_cont_two, True, interval),
        lambda x, ys, interval: _best_breakpoint_pairs(x, ys, True, interval),
    ),
    "ZslopeTwo": _Search(
        lambda x, y, interval: _grid_search_two_breakpoints(x, y, _fit_zslope_two, False, interval),
        lambda x, ys, interval: _best_breakpoint_pairs(x, ys, False, interval),
    ),
    "DiscTwo": _Search(
        lambda x, y, interval: _search_disc_two(x, y),
        lambda x, ys, interval: _disc_two_breakpoints(x, ys),
    ),
}

# The threshold models, by name, in the order sar_threshold lists them.
THRESHOLD_MODELS: list[str] = list(_SEARCHES)
assert set(_SEARCHES) == set(_K)


def _search_rows(ys: np.ndarray, model: str, x: np.ndarray, interval: float | None) -> np.ndarray:
    """``_resample_breakpoints`` with the rows first, for ``map_in_order``."""
    return _resample_breakpoints(model, x, ys, interval)


def _resample_breakpoints(
    model: str, x: np.ndarray, ys: np.ndarray, interval: float | None
) -> np.ndarray:
    """The breakpoints the model's search finds for each row of ``ys``: one row
    per response, one column per breakpoint. The model has a breakpoint for
    these ``x`` (its candidates don't depend on the response)."""
    search = _SEARCHES[model]
    assert search.breakpoints is not None  # every model but Linear
    found = search.breakpoints(x, ys, interval)
    assert found is not None
    return found.reshape(len(ys), -1)


# The models whose breakpoint the inverted F test gives an interval for, and
# whether each one's first segment is sloped.
_F_TEST_MODELS = {"ContOne": True, "ZslopeOne": False}


def _search_grid(x: np.ndarray, interval: float | None, min_points: int = 3) -> np.ndarray:
    """The candidate breakpoints ``sar_threshold`` searches for a one-threshold
    continuous model: the grid, less any candidate without ``min_points``
    values on each side."""
    grid = _breakpoint_grid(x, interval, min_points)
    assert grid is not None  # the model was fitted, so it has candidates
    n_left = np.searchsorted(np.sort(x), grid, side="right")
    return grid[(n_left >= min_points) & (len(x) - n_left >= min_points)]


def _profile_rss(x: np.ndarray, y: np.ndarray, sloped_first: bool, grid: np.ndarray) -> np.ndarray:
    """The RSS of a one-threshold continuous model with its breakpoint fixed at
    each candidate in ``grid``."""
    q = _fit_basis([np.ones(len(x))] + ([x] if sloped_first else []))
    r_y = (y - q @ (q.T @ y))[None, :]
    rss_without = float(np.sum(r_y**2))
    gains = np.empty(len(grid))
    n_blocks = -(-len(grid) * len(x) // _BLOCK)
    for block in np.array_split(np.arange(len(grid)), n_blocks):
        gains[block] = _hinge_gains(x, q, r_y, grid[block])[0]
    # Rounding can take an exact fit's RSS a hair below zero.
    return np.maximum(rss_without - gains, 0.0)


def _inverted_f_bounds(
    grid: np.ndarray, rss: np.ndarray, residual_df: float, critical_df: float, conf: float
) -> tuple[float, float]:
    """The first and last candidate breakpoints whose fit is not rejected by
    the F test against the best fit at level ``1 - conf``:
    ``(RSS(T) - RSS_min) / (RSS_min / residual_df) <= F(conf; 1, critical_df)``.
    The two degrees of freedom are the same in the test itself (the model's
    residual degrees of freedom); they are separate here so the test can be
    checked against R sars's convention."""
    least = float(rss.min())
    # Fits within rounding of the best tie with it (as in the search), and an
    # exact fit (RSS 0) rejects every other candidate.
    tie = 1e-10 * float(np.max(rss))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.where(rss > least + tie, (rss - least) / (least / residual_df), 0.0)
    accepted = grid[statistic <= special.fdtri(1.0, critical_df, conf)]
    return float(accepted.min()), float(accepted.max())


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
        warns; or if a model fits the data exactly (residual sum of squares
        0), whose AICc is -inf for want of residual variance: it ranks first,
        and the ranking then reflects rounding, not evidence.

    References
    ----------
    Matthews TJ & Rigal F (2021). Thresholds and the species-area
    relationship. Frontiers of Biogeography 13(1), e49404.
    """
    if models is None:
        models = list(_DEFAULT_MODELS)

    for m in models:
        if m not in _SEARCHES:
            raise ValueError(f"Unknown threshold model: {m!r}. Choose from {sorted(_SEARCHES)}")

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

    all_fits: dict[str, dict] = {}
    for name, search in _SEARCHES.items():  # a fixed order, whatever order ``models`` gives
        if name in models:
            result = search.fit(x, y, interval)
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

    # +inf: too few observations for the model's parameters. -inf: a perfect
    # fit (RSS 0), whose likelihood is unbounded; it ranks first, as a fit
    # whose RSS rounds to a tiny positive number would anyway.
    undefined = summary.loc[summary["AICc"] == np.inf, "model"].tolist()
    perfect = summary.loc[summary["AICc"] == -np.inf, "model"].tolist()
    if perfect:
        warnings.warn(
            f"{', '.join(perfect)} fit the data exactly (RSS 0): with no residual "
            f"variance, AICc is -inf, so they rank first. Between such fits, and "
            f"against fits with a tiny RSS, the ranking reflects rounding, not evidence.",
            UserWarning,
            stacklevel=2,
        )
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
        interval=interval,
    )


# ---------------------------------------------------------------------------
# Confidence intervals for the breakpoints
# ---------------------------------------------------------------------------


@dataclass
class ThresholdCI:
    """Bootstrap confidence intervals for a threshold model's breakpoints.

    The tuples hold one value per breakpoint, in order: one for a
    one-threshold model, two for a two-threshold model.

    Attributes
    ----------
    model : str
        The threshold model the intervals are for.
    estimates : tuple[float, ...]
        The breakpoints fitted to the data, on the fitted scale: the natural
        log of area when ``log_area`` is True, area itself otherwise.
    lower : tuple[float, ...]
        Lower confidence bounds, on the fitted scale.
    upper : tuple[float, ...]
        Upper confidence bounds, on the fitted scale.
    estimates_area : tuple[float, ...]
        The breakpoints in area units.
    lower_area : tuple[float, ...]
        Lower confidence bounds in area units.
    upper_area : tuple[float, ...]
        Upper confidence bounds in area units.
    conf : float
        Confidence level (e.g. 0.95).
    n_boot : int
        Number of bootstrap resamples (0 for the inverted F test).
    samples : np.ndarray
        The breakpoints found in each resample, on the fitted scale: one row
        per resample, one column per breakpoint (no rows for the inverted F
        test).
    log_area : bool
        Whether the fitted scale is the natural log of area (True) or area.
    method : str
        How the intervals were made: ``"boot"`` (residual bootstrap) or
        ``"F"`` (inverted F test).
    """

    model: str
    estimates: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    estimates_area: tuple[float, ...]
    lower_area: tuple[float, ...]
    upper_area: tuple[float, ...]
    conf: float
    n_boot: int
    samples: np.ndarray = field(repr=False)
    log_area: bool = True
    method: str = "boot"

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{est:.4g} [{lo:.4g}, {hi:.4g}]"
            for est, lo, hi in zip(
                self.estimates_area, self.lower_area, self.upper_area, strict=True
            )
        )
        how = "method='F'" if self.method == "F" else f"n_boot={self.n_boot}"
        return (
            f"ThresholdCI(model='{self.model}', breakpoints_area=({parts}), "
            f"conf={self.conf}, {how})"
        )


def _ci_model(fit: ThresholdFit, model: str | None) -> str:
    """The model to give intervals: the one asked for, or the best that has a breakpoint."""
    if model is None:
        for name in fit.summary["model"]:  # best first
            if fit.all_fits[name]["breakpoints"]:
                return str(name)
        raise ValueError(
            "This fit holds no model with a breakpoint (only the Linear baseline), "
            "so there is no breakpoint to give an interval."
        )
    if model not in _SEARCHES:
        raise ValueError(f"Unknown threshold model: {model!r}. Choose from {sorted(_SEARCHES)}")
    if model == "Linear":
        raise ValueError("The Linear model has no breakpoint to give an interval.")
    if model not in fit.all_fits:
        raise ValueError(
            f"{model} is not among this fit's models ({', '.join(fit.all_fits)}); "
            f"include it in sar_threshold's models."
        )
    return model


def _fitted_data(fit: ThresholdFit, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fit's areas, the same on the fitted scale, and its richness.

    A fit holds the caller's DataFrame, not a copy, so it can have changed
    since. Whatever is drawn or resampled from it must be what the model was
    fitted to: the model's stored size and RSS have to come out again.
    """
    stored = fit.all_fits[model]
    area = np.asarray(fit.data["area"], dtype=float)
    y = np.asarray(fit.data["species"], dtype=float)
    same = len(area) == stored["n"]
    if same:
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(area) if fit.log_area else area
            rss = float(np.sum((y - _predict_segments(x, stored["segments"])) ** 2))
        scale = float(np.sum((y - np.mean(y)) ** 2))
        same = bool(np.isclose(rss, stored["rss"], rtol=1e-8, atol=1e-10 * scale))
    if not same:
        raise ValueError(
            f"The fit's data no longer give {model}'s fit: the DataFrame passed to "
            f"sar_threshold has changed since. Fit it again."
        )
    return area, x, y


def threshold_ci(
    fit: ThresholdFit,
    model: str | None = None,
    n_boot: int = 1000,
    conf: float = 0.95,
    rng: np.random.Generator | None = None,
    method: str = "boot",
    n_jobs: int = 1,
) -> ThresholdCI:
    """Confidence intervals for a threshold model's breakpoints.

    Two methods, as in R sars: a residual bootstrap (``method="boot"``, the
    default) for any model with a breakpoint, and an inverted F test
    (``method="F"``) for the continuous one-threshold models, ContOne and
    ZslopeOne.

    **Bootstrap.**
    A residual bootstrap, as Toms & Lesperance (2003) describe for piecewise
    regression: the model's residuals are resampled with replacement and added
    to its fitted values, the breakpoint search is rerun on each resample, and
    the interval for each breakpoint runs between the percentiles of the
    breakpoints found. The areas stay as observed, so every resample is
    searched over the same candidates as the data were, with the same grid
    spacing (``fit.interval``) and on the same scale.

    Works for each of the six models with a breakpoint. A two-threshold model
    gets an interval for each breakpoint, each from that breakpoint's own
    percentiles. A discontinuous model's breakpoints sit at observed areas, so
    its bounds are observed areas too.

    The intervals assume residuals that could have fallen at any island
    (independent, with the same spread throughout). With few islands they are
    wide and need not be symmetric about the estimate; a bound at the edge of
    the search window (the third-smallest or third-largest area) says the data
    don't pin the breakpoint down on that side. A continuous model's interval
    can't be finer than the grid: one only a step or two wide is too narrow,
    and the fit wants a smaller ``interval``. In simulations of 40 islands with
    one true breakpoint, on a grid of 0.02, 95% intervals covered it 91 to 95%
    of the time with either method; the inverted F test's were a little
    narrower on noisier data. Coverage falls when an interval is only a grid
    step or two wide.

    **Inverted F test** (Toms & Lesperance 2003, after Lerman 1980). The
    interval holds every candidate breakpoint ``T`` of the search grid whose
    fit, with the breakpoint fixed there, the F test doesn't reject against the
    best fit: ``(RSS(T) - RSS_min) / (RSS_min / (n - p)) <= F(conf; 1, n - p)``,
    with ``p`` the model's coefficients and breakpoint (ContOne 4, ZslopeOne 3).
    It needs no resampling, so it is exact to the grid and costs one search.
    The bounds are the first and last candidates accepted; if the accepted set
    has gaps, the interval spans them. R sars's ``threshold_ci(method = "F")``
    compares the same statistic with ``F(conf; 1, p)``, using ``p`` where the
    test's residual degrees of freedom belong, so for more than a handful of
    islands its intervals are much wider than the test's (on the 173 Aegean
    plant islands, ContOne's 95% interval is 1.45 to 3.02 in ln area there,
    1.775 to 2.695 here). Toms & Lesperance recommend it for larger samples
    and the bootstrap for small ones.

    The resamples are cheap for the models searched over a grid, because the
    candidates depend only on the areas and all the resamples are scored
    against them together: the default 1000 cost about as much as a few
    ``sar_threshold`` fits of the model, a fraction of a second for the
    one-threshold models and a few seconds at most for ContTwo and ZslopeTwo
    (more on hundreds of islands or with a finer ``interval``). DiscOne and
    DiscTwo score their resamples together in the same way, from running sums
    over the observed areas.

    Parameters
    ----------
    fit : ThresholdFit
        Result of ``sar_threshold()``.
    model : str, optional
        Which of the fit's models to give intervals: "ContOne", "ZslopeOne",
        "DiscOne", "ContTwo", "ZslopeTwo" or "DiscTwo". Default: the best
        model by AICc among those with a breakpoint.
    n_boot : int
        Number of bootstrap resamples (``method="boot"`` only).
    conf : float
        Confidence level (default 0.95).
    rng : np.random.Generator, optional
        Random number generator for reproducibility (``method="boot"`` only).
    method : str
        ``"boot"`` (default) or ``"F"``.
    n_jobs : int
        Processes to search the bootstrap resamples in: 1 (the default) in this
        one, -1 one per core. The result doesn't depend on it: the resamples are
        drawn from ``rng`` here, whatever searches them. The F method ignores
        it. In a script, run the call under ``if __name__ == "__main__":``:
        each worker process imports the script.

    Returns
    -------
    ThresholdCI

    Raises
    ------
    ValueError
        If ``model`` is "Linear" or isn't among the fit's models, if the fit
        holds no model with a breakpoint, if ``method`` is unknown, or "F" for
        a model other than ContOne or ZslopeOne, if ``n_boot`` isn't a
        positive whole number or ``n_jobs`` a positive whole number or -1 (checked for the
        bootstrap only; the F test ignores ``n_boot``, ``rng`` and ``n_jobs``)
        or ``conf`` isn't between 0 and 1, or
        if the fit's data have changed since it was fitted.

    References
    ----------
    Lerman PM (1980). Fitting segmented regression models by grid search.
    Journal of the Royal Statistical Society C 29(1), 77-84.

    Toms JD & Lesperance ML (2003). Piecewise regression: a tool for
    identifying ecological thresholds. Ecology 84(8), 2034-2041.
    """
    if method not in ("boot", "F"):
        raise ValueError(f"method must be 'boot' or 'F', got {method!r}")
    model = _ci_model(fit, model)
    if method == "F":
        return _f_test_ci(fit, model, conf)
    given = n_boot
    try:  # a whole number: not True, 2.5, None, NaN or infinity
        n_boot = 0 if isinstance(given, bool) else index(given)
    except TypeError:
        n_boot = 0
    if n_boot < 1:
        raise ValueError(f"n_boot must be a positive whole number, got {given!r}")
    n_workers(n_jobs)  # a bad n_jobs fails here, before the search
    if not 0 < conf < 1:
        raise ValueError(f"conf must be between 0 and 1, got {conf!r}")
    if rng is None:
        rng = np.random.default_rng()

    area, x, y = _fitted_data(fit, model)
    fitted_model = fit.all_fits[model]
    fitted = _predict_segments(x, fitted_model["segments"])
    # Every model has an intercept, so its residuals already average zero.
    residuals = y - fitted

    # The resamples are drawn here, a batch at a time to bound memory, so the
    # draws don't depend on n_jobs. Each batch's rows are split among the
    # workers: a resample's breakpoints depend on its own row alone.
    found = []
    shared = {"model": model, "x": x, "interval": fit.interval}
    with Workers(n_jobs, shared) as workers:
        for start in range(0, n_boot, _BOOT_BATCH):
            size = (min(_BOOT_BATCH, n_boot - start), len(y))
            rows = fitted + rng.choice(residuals, size=size, replace=True)
            found.extend(workers.map(_search_rows, chunks(rows, workers.workers)))
    samples = np.concatenate(found)
    estimates = tuple(float(t) for t in fitted_model["breakpoints"])

    tail = (1 - conf) / 2
    discontinuous = model.startswith("Disc")

    def bound(q: float) -> tuple[float, ...]:
        if discontinuous:
            # Breakpoints at observed areas: take the bound from the values
            # found, not from a point interpolated between two of them.
            values = np.quantile(samples, q, axis=0, method="inverted_cdf")
        else:
            values = np.quantile(samples, q, axis=0)
        return tuple(float(v) for v in values)

    lower, upper = bound(tail), bound(1 - tail)

    def in_area(values: tuple[float, ...]) -> tuple[float, ...]:
        if discontinuous:  # the observed areas as given, not exp(log(area))
            return tuple(float(area[x == v][0]) for v in values)
        return tuple(float(np.exp(v)) if fit.log_area else v for v in values)

    return ThresholdCI(
        model=model,
        estimates=estimates,
        lower=lower,
        upper=upper,
        estimates_area=in_area(estimates),
        lower_area=in_area(lower),
        upper_area=in_area(upper),
        conf=conf,
        n_boot=n_boot,
        samples=samples,
        log_area=fit.log_area,
    )


def _f_test_ci(fit: ThresholdFit, model: str, conf: float) -> ThresholdCI:
    """``threshold_ci(method="F")``: the inverted F test interval."""
    if model not in _F_TEST_MODELS:
        raise ValueError(
            f"The inverted F test gives intervals for ContOne and ZslopeOne, not {model}; "
            f"use method='boot', or name one of those models."
        )
    if not 0 < conf < 1:
        raise ValueError(f"conf must be between 0 and 1, got {conf!r}")
    area, x, y = _fitted_data(fit, model)
    grid = _search_grid(x, fit.interval)
    rss = _profile_rss(x, y, _F_TEST_MODELS[model], grid)
    residual_df = len(y) - (_K[model] - 1)  # k counts sigma too
    lower, upper = _inverted_f_bounds(grid, rss, residual_df, residual_df, conf)
    estimate = float(fit.all_fits[model]["breakpoints"][0])

    def in_area(value: float) -> tuple[float]:
        return (float(np.exp(value)) if fit.log_area else value,)

    return ThresholdCI(
        model=model,
        estimates=(estimate,),
        lower=(lower,),
        upper=(upper,),
        estimates_area=in_area(estimate),
        lower_area=in_area(lower),
        upper_area=in_area(upper),
        conf=conf,
        n_boot=0,
        samples=np.empty((0, 1)),
        log_area=fit.log_area,
        method="F",
    )
