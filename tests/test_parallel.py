"""n_jobs: bootstrap resamples fitted in worker processes give exactly the serial
result, since the resamples are drawn in the caller, in order."""

import operator
import os
import warnings

import numpy as np
import pytest

import sars
from sars._parallel import Workers, chunks, map_in_order, n_workers

FEW = ["power", "loga", "negexpo", "monod"]


@pytest.mark.parametrize(("n_jobs", "expected"), [(1, 1), (3, 3), (-1, os.cpu_count() or 1)])
def test_n_workers(n_jobs, expected):
    assert n_workers(n_jobs) == expected


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "2", None])
def test_n_workers_refuses_anything_else(bad):
    with pytest.raises(ValueError, match="n_jobs"):
        n_workers(bad)


def _warn_and_negate(item, scale=1):
    warnings.warn(f"item {item}", UserWarning, stacklevel=1)
    return -item * scale


@pytest.mark.filterwarnings("ignore:item:UserWarning")
def test_map_in_order_passes_shared_arguments():
    assert map_in_order(_warn_and_negate, [1, 2, 3], 2, {"scale": 10}) == [-10, -20, -30]


def test_warnings_from_workers_reach_the_caller_in_order():
    """A worker's warnings go through the caller's filters, as in one process."""
    with pytest.warns(UserWarning) as record:
        map_in_order(_warn_and_negate, [1, 2, 3, 4], 2)
    assert [str(w.message) for w in record] == ["item 1", "item 2", "item 3", "item 4"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing escapes a filter that ignores it
        warnings.filterwarnings("ignore", "item", UserWarning)
        map_in_order(_warn_and_negate, [1, 2], 2)


def test_one_pool_for_several_maps():
    with Workers(2) as workers:
        assert workers.map(operator.neg, [1, 2, 3]) == [-1, -2, -3]
        assert workers.map(operator.neg, [4]) == [-4]


@pytest.mark.parametrize(("n", "parts"), [(1, 1), (3, 3), (10, 4)])
def test_chunks_are_never_empty(n, parts):
    split = chunks(np.arange(n * 2).reshape(n, 2), 4)
    assert len(split) == parts
    assert all(len(part) for part in split)
    np.testing.assert_array_equal(np.concatenate(split), np.arange(n * 2).reshape(n, 2))


def test_map_in_order_keeps_the_order():
    items = list(range(20))
    expected = [-i for i in items]
    assert map_in_order(operator.neg, items, 1) == expected
    assert map_in_order(operator.neg, items, 3) == expected


def _same_ci(a, b):
    for name in ("mean", "median", "lower", "upper"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.n_boot == b.n_boot
    assert a.dropped == b.dropped
    assert a.convergence_counts == b.convergence_counts


@pytest.mark.parametrize("method", ["fast", "full"])
def test_bootstrap_ci_is_the_same_in_parallel(galap, method):
    def run(n_jobs):
        return sars.bootstrap_ci(
            galap, models=FEW, n_boot=12, method=method, rng=np.random.default_rng(3), n_jobs=n_jobs
        )

    _same_ci(run(1), run(2))


@pytest.mark.parametrize(
    "model", ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo"]
)
def test_threshold_ci_is_the_same_in_parallel(model):
    data = sars.load_aegean()
    fit = sars.sar_threshold(data, models=[model, "Linear"])

    def run(n_jobs):
        return sars.threshold_ci(
            fit, model=model, n_boot=120, rng=np.random.default_rng(5), n_jobs=n_jobs
        )

    serial, parallel = run(1), run(2)
    assert (serial.lower, serial.upper) == (parallel.lower, parallel.upper)
    assert (serial.lower_area, serial.upper_area) == (parallel.lower_area, parallel.upper_area)


@pytest.mark.parametrize(("n_boot", "n_jobs"), [(1, 2), (3, 2), (3, -1)])
def test_threshold_ci_with_fewer_resamples_than_workers(n_boot, n_jobs):
    fit = sars.sar_threshold(sars.load_aegean(), models=["ContOne", "Linear"])

    def run(jobs):
        return sars.threshold_ci(
            fit, model="ContOne", n_boot=n_boot, rng=np.random.default_rng(1), n_jobs=jobs
        )

    np.testing.assert_array_equal(run(1).samples, run(n_jobs).samples)


@pytest.mark.parametrize("model", ["ZslopeOne", "DiscTwo"])
def test_threshold_ci_beyond_one_batch_is_the_same_in_parallel(model):
    """Over 500 resamples (one batch), split three ways within each batch."""
    fit = sars.sar_threshold(sars.load_aegean(), models=[model, "Linear"])

    def run(jobs):
        return sars.threshold_ci(
            fit, model=model, n_boot=1200, rng=np.random.default_rng(2), n_jobs=jobs
        )

    np.testing.assert_array_equal(run(1).samples, run(3).samples)


def test_a_bad_n_jobs_fails_before_any_fitting(galap):
    with pytest.raises(ValueError, match="n_jobs"):
        sars.bootstrap_ci(galap, models=FEW, n_boot=5, n_jobs=0)
    fit = sars.sar_threshold(galap, models=["ContOne", "Linear"])
    with pytest.raises(ValueError, match="n_jobs"):
        sars.threshold_ci(fit, model="ContOne", n_boot=5, n_jobs=-3)
