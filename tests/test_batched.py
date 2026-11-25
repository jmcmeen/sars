"""The batched start-grid screen and its backends."""

import importlib.util
import logging
from itertools import product

import numpy as np
import pytest

import sars
from sars import _batched, _parallel
from sars._models import _MODEL_REGISTRY, _objective, _solve, _to_search_scale


def rescaled(data):
    area = data["area"].to_numpy(float)
    species = data["species"].to_numpy(float)
    return area / np.exp(np.mean(np.log(area))), species / species.max()


def grid_for(name):
    spec = _MODEL_REGISTRY[name]
    log_idx = [spec.param_names.index(p) for p in spec.log_params]
    lower = _to_search_scale(spec.bounds_lower, log_idx)
    upper = _to_search_scale(spec.bounds_upper, log_idx)
    grid = np.array(
        [np.clip(_to_search_scale(s, log_idx), lower, upper) for s in product(*spec.start_grid)]
    )
    return spec, log_idx, lower, upper, grid


def gpu_available():
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class TestBackend:
    def test_numpy_is_the_default(self):
        assert sars.get_backend() == "numpy"

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            sars.set_backend("torch")

    @pytest.mark.skipif(importlib.util.find_spec("cupy") is not None, reason="CuPy is installed")
    def test_cupy_missing_says_how_to_install(self):
        with pytest.raises(ImportError, match=r"sars\[gpu\]"):
            sars.set_backend("cupy")
        assert sars.get_backend() == "numpy"

    def test_workers_take_the_parents_backend(self, monkeypatch):
        chosen = []
        monkeypatch.setattr(_parallel, "_BLAS_THREADS", ())
        monkeypatch.setattr(_batched, "set_backend", chosen.append)
        _parallel._start_worker({}, "cupy")
        assert chosen == ["cupy"]

    def test_a_worker_without_cupy_stays_on_numpy(self, monkeypatch):
        def refuse(name):
            raise ImportError("no CuPy here")

        monkeypatch.setattr(_parallel, "_BLAS_THREADS", ())
        monkeypatch.setattr(_batched, "set_backend", refuse)
        _parallel._start_worker({}, "cupy")
        assert sars.get_backend() == "numpy"


def test_a_failed_gpu_screen_is_reported(galap, monkeypatch, caplog):
    def broken(*args):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(_batched, "screen", broken)
    monkeypatch.setattr(_batched, "_backend", "cupy")
    with caplog.at_level(logging.WARNING, logger="sars"):
        fit = sars.sar_power(galap)
    assert fit.converged
    assert "batched screen on cupy failed (out of memory)" in caplog.text


def test_steps_stay_within_the_bounds():
    rng = np.random.default_rng(0)
    lower = np.array([0.0, -1.0, 5.0])
    upper = np.array([1.0, -1.0 + 1e-9, 5.0 + 1e-12])  # the last two narrower than a step
    p = lower + rng.random((200, 3)) * (upper - lower)
    p[:10] = lower
    p[10:20] = upper
    h = _batched.fd_steps(np, p, lower, upper)
    assert (h != 0).all()
    assert ((p + h >= lower) & (p + h <= upper)).all()


@pytest.mark.parametrize("name", ["power", "loga", "epm1", "weibull3", "chapman", "betap"])
def test_screen_reaches_the_polished_optimum(name, galap):
    """The best screened start, polished as _fit_nls polishes it, gives the
    same cost as polishing the best of a start-by-start search."""
    spec, log_idx, lower, upper, grid = grid_for(name)
    area_s, species_s = rescaled(galap)
    xs, costs = _batched.screen(spec.func, area_s, species_s, grid, lower, upper, log_idx)
    assert xs.shape == grid.shape and costs.shape == (len(grid),)
    residuals, jacobian = _objective(spec, area_s, species_s, log_idx, lower, upper)
    best = _solve(residuals, jacobian, xs[np.argmin(costs)], lower, upper, ftol=1e-14, xtol=1e-14)
    one_by_one = min(
        (
            r
            for r in (_solve(residuals, jacobian, x0, lower, upper) for x0 in grid)
            if r is not None
        ),
        key=lambda r: r.cost,
    )
    assert best.cost <= one_by_one.cost * (1 + 1e-6) + 1e-12


def test_blocks_dont_change_the_result(galap, monkeypatch):
    spec, log_idx, lower, upper, grid = grid_for("weibull3")
    area_s, species_s = rescaled(galap)
    whole = _batched.screen(spec.func, area_s, species_s, grid, lower, upper, log_idx)
    monkeypatch.setattr(_batched, "_BLOCK", 7 * len(area_s))  # a few starts per block
    blocked = _batched.screen(spec.func, area_s, species_s, grid, lower, upper, log_idx)
    np.testing.assert_allclose(blocked[1], whole[1], rtol=1e-9)
    np.testing.assert_allclose(blocked[0], whole[0], rtol=1e-9, atol=1e-12)


def test_the_cut_keeps_the_best_start(galap, monkeypatch):
    spec, log_idx, lower, upper, grid = grid_for("epm1")
    area_s, species_s = rescaled(galap)
    monkeypatch.setattr(_batched, "_WARMUP", 10**9)  # never cut
    uncut = _batched.screen(spec.func, area_s, species_s, grid, lower, upper, log_idx)
    monkeypatch.setattr(_batched, "_WARMUP", 15)
    monkeypatch.setattr(_batched, "_KEEP", 4)
    cut = _batched.screen(spec.func, area_s, species_s, grid, lower, upper, log_idx)
    assert np.nanmin(cut[1]) <= np.nanmin(uncut[1]) * (1 + 1e-3) + 1e-12


def test_starts_outside_the_bounds_are_clipped(galap):
    spec, log_idx, lower, upper, grid = grid_for("power")
    area_s, species_s = rescaled(galap)
    far = grid * 1e6
    xs, _ = _batched.screen(spec.func, area_s, species_s, far, lower, upper, log_idx)
    assert (xs >= np.array(lower) - 1e-12).all() and (xs <= np.array(upper) + 1e-12).all()


@pytest.mark.skipif(not gpu_available(), reason="needs CuPy and an NVIDIA GPU")
class TestGPU:
    def test_cupy_matches_numpy(self, galap):
        spec, log_idx, lower, upper, grid = grid_for("weibull4")
        area_s, species_s = rescaled(galap)
        cpu = _batched.screen(spec.func, area_s, species_s, grid, lower, upper, log_idx)
        sars.set_backend("cupy")
        try:
            gpu = _batched.screen(spec.func, area_s, species_s, grid, lower, upper, log_idx)
        finally:
            sars.set_backend("numpy")
        assert np.nanmin(gpu[1]) == pytest.approx(np.nanmin(cpu[1]), rel=1e-6)

    def test_fits_match_on_both_backends(self, galap):
        cpu = sars.sar_multi(galap)
        sars.set_backend("cupy")
        try:
            gpu = sars.sar_multi(galap)
        finally:
            sars.set_backend("numpy")
        a = cpu.summary.set_index("model")["AICc"]
        b = gpu.summary.set_index("model")["AICc"]
        np.testing.assert_allclose(b.loc[a.index], a, rtol=1e-6)


def test_a_screen_of_only_penalized_points_isnt_a_fit(galap, monkeypatch):
    """Points where the model is undefined at some island are never taken as a
    fit: with nothing else from the screen, the start-by-start search runs."""
    from sars._models import _PENALTY_COST, _fit_nls

    alone = _fit_nls("power", galap)

    def undefined_everywhere(func, area, species, starts, *rest, **kw):
        return starts, np.full(len(starts), 2 * _PENALTY_COST)

    monkeypatch.setattr(_batched, "screen", undefined_everywhere)
    fit = _fit_nls("power", galap)
    assert fit.converged
    assert fit.aicc == pytest.approx(alone.aicc, rel=1e-9)
