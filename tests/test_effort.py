"""Richness standardised for sampling effort, against R's iNEXT and vegan."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars
from sars import _effort

REF = Path(__file__).parent / "r_reference"
TABLE = pd.read_csv(REF / "effort_abundance.csv", index_col=0)
INEXT = pd.read_csv(REF / "effort_inext.csv")
VEGAN = pd.read_csv(REF / "effort_vegan.csv")
CASES = list(INEXT.groupby(["base", "level"], dropna=False))


@pytest.mark.parametrize(
    ("key", "rows"), CASES, ids=[f"{b}-{'default' if pd.isna(lv) else lv}" for (b, lv), _ in CASES]
)
def test_matches_inext_estimated(key, rows):
    base, level = key
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = sars.standardised_richness(
            TABLE, base=base, level=None if pd.isna(level) else level
        ).set_index("site")
    for _, row in rows.iterrows():
        got = ours.loc[row["site"]]
        assert got["method"] == row["method"].lower()
        # Coverage targets are found numerically, here and in iNEXT.
        assert got["m"] == pytest.approx(row["m"], rel=1e-5)
        assert got["species"] == pytest.approx(row["qD"], rel=1e-5)
        assert _effort._coverage_at(TABLE[row["site"]].to_numpy(float), got["m"]) == pytest.approx(
            row["SC"], rel=1e-5
        )


def test_rarefaction_matches_vegan():
    for site, expected in zip(VEGAN["site"], VEGAN["rarefied10"], strict=True):
        assert _effort._rarefied(TABLE[site].to_numpy(float), 10) == pytest.approx(
            expected, rel=1e-10
        )


def test_the_default_level_is_reached_by_every_site():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sars.standardised_richness(TABLE)
    assert (result["m"] <= 2 * result["individuals"] + 1e-6).all()
    assert result["level"].nunique() == 1


def test_observed_level_gives_observed_richness():
    x = TABLE[["site05"]]
    n = int(x.sum().iloc[0])
    result = sars.standardised_richness(x, base="size", level=n)
    assert result["method"].iloc[0] == "observed"
    assert result["species"].iloc[0] == result["observed"].iloc[0]


def test_area_makes_it_ready_to_fit():
    xu = sars.load_xu()
    table = sars.load_xu_abundance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = sars.standardised_richness(table, area=xu["area"])
    assert list(data.columns[:3]) == ["site", "area", "species"]
    assert list(data["site"]) == list(xu["island"])
    fit = sars.lin_pow(data)
    # Standardised to equal coverage, the slope is shallower than on the
    # observed richness: more individuals were sampled on the larger islands.
    assert fit.z < sars.lin_pow(xu).z


def test_xu_abundance_matches_load_xu():
    table = sars.load_xu_abundance()
    xu = sars.load_xu()
    assert table.shape == (68, 42)
    assert list(table.columns) == list(xu["island"])
    np.testing.assert_array_equal((table > 0).sum().to_numpy(), xu["species"].to_numpy())


def test_extrapolating_far_warns():
    with pytest.warns(UserWarning, match="beyond twice their sample"):
        sars.standardised_richness(TABLE, base="size", level=5000)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"base": "effort"}, "base must be"),
        ({"base": "coverage", "level": 1.5}, "between 0 and 1"),
        ({"base": "size", "level": 0}, "at least one"),
        ({"area": [1.0, 2.0]}, "area has 2 values"),
    ],
)
def test_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sars.standardised_richness(TABLE, **kwargs)


@pytest.mark.parametrize(
    ("table", "match"),
    [
        (pd.DataFrame({"a": [1.5, 2.0]}), "whole numbers"),
        (pd.DataFrame({"a": [-1, 2]}), "whole numbers"),
        (pd.DataFrame({"a": [1, 2], "b": [0, 0]}), "no individuals"),
        (np.array([1, 2, 3]), "2-D"),
    ],
)
def test_bad_tables(table, match):
    with pytest.raises(ValueError, match=match):
        sars.standardised_richness(table)


def test_full_coverage_is_reached_only_in_the_limit():
    """level=1.0: m is infinite and richness is the Chao1 asymptote, with no
    numpy warnings on the way."""
    x = np.array([[3], [2], [1]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sars.standardised_richness(x, level=1.0)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert np.isinf(result["m"].iloc[0])
    assert result["species"].iloc[0] == pytest.approx(_effort._chao1(x[:, 0].astype(float)))
    assert result["coverage_at_m"].iloc[0] == 1.0


def test_a_single_individual():
    result = sars.standardised_richness(np.array([[1], [0]]), base="size", level=1)
    assert result["species"].iloc[0] == 1 and result["method"].iloc[0] == "observed"


def test_coverage_at_m_shows_where_the_minimum_size_was_reached():
    """A target below what one individual gives: m is held at 1, and the
    coverage there is reported."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sars.standardised_richness(TABLE[["site10"]], level=0.001)
    assert result["m"].iloc[0] == 1
    assert result["coverage_at_m"].iloc[0] > 0.001
    standard = sars.standardised_richness(TABLE[["site10"]], level=0.9)
    assert standard["coverage_at_m"].iloc[0] == pytest.approx(0.9, rel=1e-6)


def test_a_wrong_area_fails_before_any_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="area has 2 values"):
            sars.standardised_richness(TABLE, base="size", level=5000, area=[1, 2])
    assert not caught


def test_species_names_in_a_column_are_explained():
    frame = sars.load_xu_abundance().reset_index()
    with pytest.raises(ValueError, match="put the species names in the index"):
        sars.standardised_richness(frame)
