"""The hierarchical log-log power model, against R's lme4."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars

REF = Path(__file__).parent / "r_reference"
DATA = pd.read_csv(REF / "mixed_data.csv")
FIT = pd.read_csv(REF / "mixed_fit.csv").iloc[0]
GROUPS = pd.read_csv(REF / "mixed_groups.csv").set_index("group")


def _datasets():
    return {
        "gala": sars.load_gala(),
        "aegean": sars.load_aegean(),
        "aegean2": sars.load_aegean2(),
        "niering": sars.load_niering(),
        "schrader": sars.load_schrader(),
        "xu": sars.load_xu(),
    }


@pytest.fixture(scope="module")
def fit():
    return sars.lin_pow_mixed(_datasets())


@pytest.mark.parametrize(
    "name", ["logc", "z", "logc_se", "z_se", "sd_logc", "sd_z", "corr", "sigma"]
)
def test_matches_lme4(fit, name):
    assert getattr(fit, name) == pytest.approx(FIT[name], rel=1e-5)


def test_reml_criterion_matches_lme4(fit):
    assert fit.reml_criterion == pytest.approx(FIT["reml"], rel=1e-9)


def test_pooled_coefficients_match_lme4(fit):
    for group, row in GROUPS.iterrows():
        assert fit.groups.loc[group, "logc"] == pytest.approx(row["logc"], rel=1e-5)
        assert fit.groups.loc[group, "z"] == pytest.approx(row["z"], rel=1e-5)


def test_alone_fits_and_con(fit):
    """Schrader's zeros put one on every richness value, so that all six share
    a log scale; each archipelago's own fit is on that scale too."""
    assert fit.con == 1
    g = fit.groups
    xu = sars.load_xu()
    alone = sars.lin_pow(xu.assign(species=xu["species"] + 1))
    assert g.loc["xu", "z_alone"] == pytest.approx(alone.z)


def test_one_frame_with_a_group_column_gives_the_same_fit(fit):
    long = DATA.drop(columns="con")
    again = sars.lin_pow_mixed(long, by="group")
    assert again.z == pytest.approx(fit.z, rel=1e-8)
    assert list(again.groups.index) == list(fit.groups.index)


def test_predictions(fit):
    mean = fit.predict([1.0, 100.0])
    np.testing.assert_allclose(mean, np.exp(fit.logc + fit.z * np.log([1.0, 100.0])) - fit.con)
    xu = fit.predict(10.0, group="xu")
    g = fit.groups.loc["xu"]
    assert xu == pytest.approx(np.exp(g["logc"] + g["z"] * np.log(10.0)) - fit.con)
    with pytest.raises(ValueError, match="No archipelago"):
        fit.predict(1.0, group="nope")
    with pytest.raises(ValueError, match="positive"):
        fit.predict(-1.0)


def test_a_shared_slope_is_a_singular_fit():
    """Archipelagos simulated with one z: its spread goes to 0, with a warning."""
    rng = np.random.default_rng(3)
    data = {}
    for j in range(5):
        area = np.exp(rng.uniform(-2, 5, 25))
        data[f"a{j}"] = pd.DataFrame(
            {
                "area": area,
                "species": np.exp(1 + 0.5 * j + 0.3 * np.log(area) + rng.normal(0, 0.1, 25)),
            }
        )
    with pytest.warns(UserWarning, match="singular"):
        fit = sars.lin_pow_mixed(data)
    assert fit.sd_z < 1e-3
    assert fit.z == pytest.approx(0.3, abs=0.03)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"a": sars.load_gala(), "b": sars.load_xu()}, "at least 3 archipelagos"),
        (
            {"a": sars.load_gala(), "b": sars.load_xu(), "c": sars.load_gala().head(1)},
            "at least two islands",
        ),
        (
            {
                "a": sars.load_gala(),
                "b": sars.load_xu(),
                "c": sars.load_gala().drop(columns="species"),
            },
            "no column",
        ),
    ],
)
def test_bad_data(data, match):
    with pytest.raises(ValueError, match=match):
        sars.lin_pow_mixed(data)


def test_zeros_need_a_positive_con():
    data = _datasets()
    with pytest.raises(ValueError, match="con must be positive"):
        sars.lin_pow_mixed(data, con=0)


@pytest.mark.parametrize("con", [np.inf, np.nan, True, "1"])
def test_con_must_be_a_finite_number(con):
    with pytest.raises(ValueError, match="'con' must be a finite number"):
        sars.lin_pow_mixed(_datasets(), con=con)


def test_without_zeros_no_con_is_added():
    data = {k: v for k, v in _datasets().items() if k != "schrader"}
    assert sars.lin_pow_mixed(data).con == 0


def test_large_archipelagos_are_quick():
    """The criterion uses each archipelago's 2 x 2 summaries, not n x n matrices."""
    import time

    rng = np.random.default_rng(0)
    big = {}
    for j in range(4):
        area = np.exp(rng.uniform(-3, 6, 5000))
        richness = np.exp(
            1 + 0.3 * j + (0.25 + 0.1 * rng.normal()) * np.log(area) + rng.normal(0, 0.3, 5000)
        )
        big[f"g{j}"] = pd.DataFrame({"area": area, "species": richness})
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sars.lin_pow_mixed(big)
    assert time.perf_counter() - start < 5


def test_an_exact_power_law_is_refused():
    area = np.array([1.0, 10.0, 100.0, 1000.0])
    data = {f"a{j}": pd.DataFrame({"area": area, "species": (j + 2) * area**0.3}) for j in range(3)}
    with pytest.raises(ValueError, match="no residual variation"):
        sars.lin_pow_mixed(data)


def test_saves_and_reloads(fit):
    again = sars.from_json(sars.to_json(fit))
    assert again.z == fit.z
    pd.testing.assert_frame_equal(again.groups, fit.groups)


def test_repr(fit):
    with warnings.catch_warnings():
        assert repr(fit).startswith("MixedLinPowFit(6 archipelagos, z=0.3254")
