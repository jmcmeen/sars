"""The benchmark script's helpers (benchmarks/run.py)."""

import importlib.util
import logging
from pathlib import Path

import numpy as np
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
    log_area, species = np.log(data["area"]), data["species"]
    thirds = np.array_split(np.arange(len(data)), 3)
    slopes = [np.polyfit(log_area.iloc[t], species.iloc[t], 1)[0] for t in thirds]
    # Rising throughout, sigmoid in log area: flattening toward 200 at the top.
    assert min(slopes) > 0 and slopes[2] < 0.75 * slopes[1]
    smooth = species.rolling(101, center=True, min_periods=1).median()
    assert (species - smooth).std() > 2  # noisy around the curve
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


@pytest.mark.parametrize("text", ["0", "-5"])
def test_a_large_table_needs_points(text):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        run.positive_int(text)


def test_gpu_timings_say_when_a_step_ran_on_the_cpu(monkeypatch):
    """A job where a GPU step fell back to the CPU is marked, and a failure
    part-way keeps the timings taken so far."""
    monkeypatch.setattr(run.sars, "set_backend", lambda name: None)
    monkeypatch.setattr(run.sars, "sar_multi", lambda data: None)

    def timed(jobs, datasets, repeats, n_boot, language):
        dataset = jobs["dataset"].iloc[0]
        if dataset == "broken":
            raise RuntimeError("the GPU went away")
        if dataset == "big":
            logging.getLogger("sars._models").warning("the batched screen on cupy failed")
        return pd.DataFrame([{"language": language, "dataset": dataset, "result": "best=power"}])

    monkeypatch.setattr(run, "time_python", timed)
    jobs = pd.DataFrame(
        [
            {"task": "average", "dataset": "small"},
            {"task": "average", "dataset": "big"},
            {"task": "average", "dataset": "broken"},
            {"task": "average", "dataset": "never"},
        ]
    )
    rows = run.time_gpu(jobs, {}, 1, 1)
    assert rows["dataset"].tolist() == ["small", "big"]
    assert rows["result"].tolist() == ["best=power", "best=power (partly on the CPU)"]
