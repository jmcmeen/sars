"""The benchmark script's helpers (benchmarks/run.py)."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run.py"
_spec = importlib.util.spec_from_file_location("benchmarks_run", _PATH)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_the_large_table_is_a_noisy_saturating_curve_over_large_areas():
    data = run.synthetic_large(2000)
    assert len(data) == 2000
    assert data["area"].between(1e4, 3e5).all()
    assert data["area"].is_monotonic_increasing
    assert data["species"].between(1, 300).all()
    top, bottom = data["species"].iloc[-200:].mean(), data["species"].iloc[:200].mean()
    assert 1.5 < top / bottom < 5  # rising, then flattening toward 200
    assert run.synthetic_large(2000).equals(data)  # the same table every run


def test_large_tables_are_recognized():
    assert run.is_large("large_100000")
    assert not run.is_large("synthetic_5000")


def test_speedups_leave_out_the_gpu_series():
    timings = pd.DataFrame(
        [
            {"language": "Python", "task": "average", "dataset": "galap", "n": 16, "seconds": 1.0},
            {"language": "R", "task": "average", "dataset": "galap", "n": 16, "seconds": 4.0},
            {
                "language": "Python GPU",
                "task": "average",
                "dataset": "galap",
                "n": 16,
                "seconds": 2.0,
            },
            {"language": "Python", "task": "power", "dataset": "galap", "n": 16, "seconds": 0.1},
            {"language": "R", "task": "power", "dataset": "galap", "n": 16, "seconds": 0.3},
        ]
    )
    wide = run.speedups(timings)
    assert list(wide.columns) == ["Python", "R", "R_over_Python"]
    assert len(wide) == 2  # the power job isn't dropped for lacking a GPU time
    assert wide.loc[("average", "galap", 16), "R_over_Python"] == 4.0
