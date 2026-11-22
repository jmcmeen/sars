"""The package on 34 published island datasets (Gooriah et al. 2021, CC0): every
dataset with enough islands fits with the multi-model average, the threshold
models and the log-log power model. See tests/data/gooriah2021/README.md."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars

DATA = Path(__file__).parent / "data" / "gooriah2021"
TABLES = sorted(DATA.glob("*.csv"))
MIN_ISLANDS = 5


def _islands_with_species(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["species"] > 0].reset_index(drop=True)


FITTED = [p for p in TABLES if len(_islands_with_species(p)) >= MIN_ISLANDS]


def test_there_are_34_tables_and_31_with_enough_islands():
    assert len(TABLES) == 34
    too_few = sorted(p.stem for p in TABLES if p not in FITTED)
    assert too_few == ["Theurkauf_2017", "Usher_1986", "Zagar_2013"]


@pytest.mark.parametrize("name", ["schrader", "xu"])
def test_bundled_copies_match(name):
    table = DATA / {"schrader": "Schrader_2019.csv", "xu": "Xu_2017.csv"}[name]
    bundled = getattr(sars, f"load_{name}")()
    pd.testing.assert_frame_equal(pd.read_csv(table), bundled)


@pytest.mark.slow
@pytest.mark.parametrize("path", FITTED, ids=[p.stem for p in FITTED])
def test_dataset_fits(path):
    data = _islands_with_species(path)
    with warnings.catch_warnings():
        # Small archipelagos (Barun_2010 has 6 islands) leave some threshold models
        # without a defined AICc; sar_threshold warns and ranks them last.
        warnings.filterwarnings("ignore", "AICc is undefined", UserWarning)
        warnings.filterwarnings("ignore", "Dataset has", UserWarning)
        average = sars.sar_average(data)
        threshold = sars.sar_threshold(data)
    assert average.multi.not_fitted == {}
    assert average.weights
    assert sum(average.weights.values()) == pytest.approx(1.0)
    assert np.all(np.isfinite(average.predict(data["area"].to_numpy())))
    assert threshold.best_model in threshold.summary["model"].tolist()
    log_log = sars.lin_pow(data)
    assert np.isfinite(log_log.z) and 0 <= log_log.r_squared <= 1
