"""Many least-squares fits of one model at once.

A fit's start grid holds tens to hundreds of starting points, and fitting each
with its own call of ``scipy.optimize.least_squares`` spent most of a fit's time
in the solver's per-call overhead. Here every start runs as one row of a batch:
a projected Levenberg-Marquardt, with forward-difference Jacobians, in array
operations over (starts x islands). The arrays can be numpy's or, with
``set_backend("cupy")``, CuPy's on a GPU: the model functions are plain numpy
expressions, which numpy dispatches to CuPy arrays.

The batch only ranks the starts. The best few distinct results go on to the
tight trust-region polish in ``_fit_nls``, so the reported fits keep that
precision whichever backend screened them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import numpy as np

_BACKENDS = ("numpy", "cupy")
_backend = "numpy"

# Steps of Levenberg-Marquardt on every start before the batch is cut to the
# most promising ones, and how many it keeps.
_WARMUP = 15
_KEEP = 12
_MAX_ITER = 200
# A start has converged when an accepted step lowers its cost by less than
# this share, or its damping has grown past _LAMBDA_MAX (no step helps).
_FTOL = 1e-10
_LAMBDA_MAX = 1e12
# Largest number of values (starts x islands, or starts x parameters x islands
# for a Jacobian) held in one array, so a large dataset doesn't exhaust memory:
# the warm-up runs over the starts a chunk at a time, and each model call over
# a block of rows.
_BLOCK = 2**24
# Relative step of the forward differences (as least_squares's "2-point" uses).
_FD_STEP = float(np.sqrt(np.finfo(float).eps))
_PENALTY = 1e10


def set_backend(name: str) -> None:
    """Choose the array library that screens each model's start grid.

    Every fit tries each model from a grid of starting values. They're run
    as one batch, on numpy (the default) or, for large datasets or many fits,
    on an NVIDIA GPU through CuPy (``pip install 'sars[gpu]'``). The tight
    final solve always runs on the CPU, so results agree between backends to
    the solver's tolerance wherever that solve converges.

    Parameters
    ----------
    name : {"numpy", "cupy"}
        The backend for every fit that follows, in this process and in the
        worker processes that ``n_jobs`` starts.

    Raises
    ------
    ValueError
        If ``name`` isn't one of the backends.
    ImportError
        If ``name`` is ``"cupy"`` and CuPy isn't installed or finds no GPU.
    """
    global _backend
    if name not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {name!r}")
    if name == "cupy":
        _check_cupy()
    _backend = name


def get_backend() -> str:
    """The array library screening start grids: ``"numpy"`` or ``"cupy"``."""
    return _backend


def _cupy() -> ModuleType:
    try:
        import cupy
    except ImportError as err:
        raise ImportError(
            "The cupy backend needs CuPy and an NVIDIA GPU: pip install 'sars[gpu]'"
        ) from err
    return cupy


def _check_cupy() -> None:
    """Fail now, not in the middle of a fit, if CuPy can't run on this machine."""
    cupy = _cupy()
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device found")
        # Compiles and runs the kinds of kernel a screen needs: elementwise
        # math, a reduction and a batched solve.
        a = cupy.eye(2)[None] * 2.0
        cupy.linalg.solve(a, cupy.exp(cupy.ones((1, 2, 1)))).sum().item()
    except Exception as err:
        raise ImportError(f"CuPy is installed but can't run on a GPU here: {err}") from err


def _xp() -> ModuleType:
    return _cupy() if _backend == "cupy" else np


def fd_steps(xp: ModuleType, p: Any, lower: Any, upper: Any) -> Any:
    """Forward-difference steps for the parameter rows ``p`` (..., k), as
    least_squares's "2-point" scheme takes them: a relative step, reversed
    or made one-sided to stay within the bounds, and returned as actually
    represented in floating point."""
    # scipy.optimize._numdiff: _compute_absolute_step, then
    # _adjust_scheme_to_bounds for a one-sided scheme.
    h = _FD_STEP * xp.where(p >= 0, 1.0, -1.0) * xp.maximum(1.0, xp.abs(p))
    lower_dist, upper_dist = p - lower, upper - p
    x = p + h
    violated = (x < lower) | (x > upper)
    fitting = xp.abs(h) <= xp.maximum(lower_dist, upper_dist)
    h = xp.where(violated & fitting, -h, h)
    forward = (upper_dist >= lower_dist) & ~fitting
    backward = (upper_dist < lower_dist) & ~fitting
    h = xp.where(forward, upper_dist, xp.where(backward, -lower_dist, h))
    return (p + h) - p


def screen(
    func: Callable[..., Any],
    area: np.ndarray,
    species: np.ndarray,
    starts: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
    log_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``species ~ func(area, *params)`` from every row of ``starts`` at once.

    Parameters are on the search scale: the ``log_idx`` ones as logs, within
    ``lower`` and ``upper``. Returns each start's end point and cost (half the
    sum of squared residuals), as numpy arrays. Starts dropped after the
    warm-up keep the point and cost they had then.
    """
    xp = _xp()
    area_d = xp.asarray(area, dtype=float)
    species_d = xp.asarray(species, dtype=float)
    lo = xp.asarray(lower, dtype=float)
    hi = xp.asarray(upper, dtype=float)
    x = xp.clip(xp.asarray(starts, dtype=float), lo, hi)
    n_starts, k = x.shape
    n = area_d.shape[0]
    eye = xp.eye(k)

    def natural(p: Any) -> Any:
        if not log_idx:
            return p
        p = p.copy()
        p[:, log_idx] = xp.exp(p[:, log_idx])
        return p

    def residuals(p: Any) -> Any:
        """Residuals of each row of p: (rows, islands)."""
        values = natural(p)
        pred = func(area_d[None, :], *(values[:, j : j + 1] for j in range(k)))
        res = species_d[None, :] - xp.broadcast_to(pred, (p.shape[0], n))
        return xp.where(xp.isfinite(res), res, _PENALTY)

    def cost_of(res: Any) -> Any:
        return 0.5 * xp.sum(res * res, axis=1)

    def normal_equations(p: Any, res: Any) -> tuple[Any, Any]:
        """J'J and J'r for each row, with a forward-difference J."""
        h = fd_steps(xp, p, lo, hi)
        rows = p.shape[0]
        stepped = p[:, None, :] + h[:, None, :] * eye[None, :, :]  # (rows, k, k)
        res_step = residuals(stepped.reshape(rows * k, k)).reshape(rows, k, n)
        jac = (res_step - res[:, None, :]) / h[:, :, None]  # (rows, k, n): J transposed
        return jac @ xp.swapaxes(jac, 1, 2), (jac @ res[:, :, None])[:, :, 0]

    def run(x: Any, iterations: int) -> tuple[Any, Any, Any]:
        """Levenberg-Marquardt on every row of x for up to ``iterations``
        steps. Returns the end points, their costs and which rows are still
        improving."""
        x = x.copy()
        res = _blocked(xp, residuals, x, n, 1)
        cost = cost_of(res)
        damping = xp.full(x.shape[0], 1e-3)
        active = xp.isfinite(cost)
        for _ in range(iterations):
            rows = xp.nonzero(active)[0]
            if rows.shape[0] == 0:
                break
            p, r, c, lam = x[rows], res[rows], cost[rows], damping[rows]
            jtj, jtr = _blocked_pair(xp, normal_equations, p, r, n, k + 1)
            diag = xp.maximum(xp.diagonal(jtj, axis1=1, axis2=2), 1e-12)
            system = jtj + (lam[:, None] * diag)[:, :, None] * eye
            try:
                delta = xp.linalg.solve(system, -jtr[:, :, None])[:, :, 0]
            except Exception:  # a singular system in the batch: solve one at a time
                delta = xp.stack(
                    [_solve_or_zero(xp, a, -b) for a, b in zip(system, jtr, strict=True)]
                )
            trial = xp.clip(p + delta, lo, hi)
            trial_res = _blocked(xp, residuals, trial, n, 1)
            trial_cost = cost_of(trial_res)
            better = xp.isfinite(trial_cost) & (trial_cost < c)
            gain = xp.where(better, (c - trial_cost) / xp.maximum(c, 1e-300), 0.0)
            x[rows] = xp.where(better[:, None], trial, p)
            res[rows] = xp.where(better[:, None], trial_res, r)
            cost[rows] = xp.where(better, trial_cost, c)
            lam = xp.where(better, lam / 3.0, lam * 4.0)
            damping[rows] = lam
            done = (better & (gain < _FTOL)) | (lam > _LAMBDA_MAX) | ~xp.isfinite(c)
            active[rows[done]] = False
        return x, cost, active

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        if n_starts <= _KEEP:
            x, cost, _ = run(x, _MAX_ITER)
        else:
            # Every start's iterations are independent of the others', so the
            # warm-up can run a chunk of starts at a time, holding only that
            # chunk's residuals; then the most promising starts still improving
            # carry on together.
            size = max(_KEEP, _rows_per_block(n, 1))
            warmup = min(_WARMUP, _MAX_ITER)
            parts = [run(x[i : i + size], warmup) for i in range(0, n_starts, size)]
            x = xp.concatenate([part[0] for part in parts])
            cost = xp.concatenate([part[1] for part in parts])
            active = xp.concatenate([part[2] for part in parts])
            keep = xp.argsort(xp.where(active, cost, xp.inf))[:_KEEP]
            keep = keep[active[keep]]
            if keep.shape[0]:
                x[keep], cost[keep], _ = run(x[keep], _MAX_ITER - warmup)
    if xp is not np:
        return xp.asnumpy(x), xp.asnumpy(cost)
    return x, cost


def _solve_or_zero(xp: ModuleType, a: Any, b: Any) -> Any:
    try:
        return xp.linalg.solve(a, b)
    except Exception:
        return xp.zeros_like(b)


def _rows_per_block(n: int, per_row: int) -> int:
    return max(1, _BLOCK // max(1, n * per_row))


def _blocked(xp: ModuleType, fn: Callable[[Any], Any], p: Any, n: int, per_row: int) -> Any:
    """fn over the rows of p, a block of rows at a time."""
    size = _rows_per_block(n, per_row)
    if p.shape[0] <= size:
        return fn(p)
    return xp.concatenate([fn(p[i : i + size]) for i in range(0, p.shape[0], size)])


def _blocked_pair(
    xp: ModuleType,
    fn: Callable[[Any, Any], tuple[Any, Any]],
    p: Any,
    r: Any,
    n: int,
    per_row: int,
) -> tuple[Any, Any]:
    size = _rows_per_block(n, per_row)
    if p.shape[0] <= size:
        return fn(p, r)
    parts = [fn(p[i : i + size], r[i : i + size]) for i in range(0, p.shape[0], size)]
    return xp.concatenate([a for a, _ in parts]), xp.concatenate([b for _, b in parts])
