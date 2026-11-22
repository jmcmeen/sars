"""Species expected to be lost with habitat."""

import warnings

import numpy as np
import pandas as pd
import pytest

import sars

RISING = ["power", "loga", "koba", "negexpo", "monod", "ratio"]


@pytest.fixture(scope="module")
def gala():
    return sars.load_gala()


@pytest.fixture(scope="module")
def halved(gala):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sars.habitat_loss(
            gala,
            area=900,
            fraction_lost=0.5,
            models=RISING,
            n_boot=200,
            rng=np.random.default_rng(1),
        )


def test_point_estimates_come_from_the_average(gala, halved):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg = sars.sar_average(gala, models=RISING)
    row = halved.table.iloc[0]
    before, after = avg.predict(np.array([900.0, 450.0]))
    assert row["area"] == 900 and row["remaining"] == 450
    assert row["richness_before"] == pytest.approx(before)
    assert row["richness_after"] == pytest.approx(after)
    assert row["species_lost"] == pytest.approx(before - after)
    assert row["proportion_lost"] == pytest.approx((before - after) / before)


def test_intervals_come_from_each_resamples_own_loss(gala, halved):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kept = sars.habitat_loss(
            gala,
            area=900,
            fraction_lost=0.5,
            models=RISING,
            n_boot=200,
            rng=np.random.default_rng(1),
            keep_samples=True,
        )
    pd.testing.assert_frame_equal(kept.table, halved.table)
    row = halved.table.iloc[0]
    samples = kept.ci.samples  # the grid is sorted: 450, then 900
    assert samples.shape == (kept.ci.n_boot, 2)
    lost = samples[:, 1] - samples[:, 0]
    lo, hi = np.percentile(lost, [2.5, 97.5])
    assert row["lost_lower"] == pytest.approx(lo) and row["lost_upper"] == pytest.approx(hi)
    assert row["lost_lower"] < row["species_lost"] < row["lost_upper"]
    assert 0 < row["proportion_lower"] < row["proportion_lost"] < row["proportion_upper"] < 1


def test_remaining_and_fraction_lost_agree(gala):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sars.habitat_loss(gala, area=900, remaining=450, models=["power"], n_boot=0)
        b = sars.habitat_loss(gala, area=900, fraction_lost=0.5, models=["power"], n_boot=0)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.ci is None and a.table["lost_lower"].isna().all()


def test_power_law_loss_has_a_closed_form(gala):
    """For S = c A^z, losing a fraction f leaves a share (1 - f)^z."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = sars.habitat_loss(
            gala, area=500, fraction_lost=[0.1, 0.5, 0.9], models=["power"], n_boot=0
        )
    z = sars.sar_power(gala).params["z"]
    np.testing.assert_allclose(
        loss.table["proportion_lost"], 1 - (1 - np.array([0.1, 0.5, 0.9])) ** z, rtol=1e-6
    )


def test_scenarios_broadcast(gala):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = sars.habitat_loss(
            gala, area=[100, 900], fraction_lost=0.3, models=["power"], n_boot=0
        )
    assert list(loss.table["remaining"]) == pytest.approx([70, 630])


def test_extrapolation_warns(gala):
    with pytest.warns(UserWarning, match="outside the observed range"):
        sars.habitat_loss(gala, area=10_000, fraction_lost=0.2, models=["power"], n_boot=0)


def test_non_monotone_averages_warn(gala):
    """All 20 models on gala: a few resamples' averaged curves dip."""
    with pytest.warns(UserWarning, match="fewer species in the remaining area"):
        sars.habitat_loss(
            gala, area=900, fraction_lost=0.5, n_boot=200, rng=np.random.default_rng(1)
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"remaining": 100, "fraction_lost": 0.5}, "exactly one"),
        ({"fraction_lost": 1.5}, "between 0 and 1"),
        ({"fraction_lost": 1.0}, "must be positive"),
        ({"remaining": 1000}, "can't be larger"),
    ],
)
def test_bad_scenarios(gala, kwargs, match):
    with pytest.raises(ValueError, match=match):
        sars.habitat_loss(gala, area=900, models=["power"], n_boot=0, **kwargs)


def test_saves_and_reloads(halved):
    again = sars.from_json(sars.to_json(halved))
    pd.testing.assert_frame_equal(again.table, halved.table)
    np.testing.assert_array_equal(again.ci.samples, halved.ci.samples)


def test_bootstrap_keeps_samples_only_when_asked(gala):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sars.bootstrap_ci(gala, models=["power"], n_boot=20, rng=np.random.default_rng(0))
        kept = sars.bootstrap_ci(
            gala, models=["power"], n_boot=20, rng=np.random.default_rng(0), keep_samples=True
        )
    assert plain.samples.size == 0
    assert kept.samples.shape == (20, len(kept.area_grid))
    np.testing.assert_allclose(np.median(kept.samples, axis=0), kept.median)


def test_warnings_point_at_the_caller_once_each(gala):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sars.habitat_loss(
            gala, area=900, fraction_lost=0.5, n_boot=60, rng=np.random.default_rng(1)
        )
    assert caught
    assert all(w.filename == __file__ for w in caught)
    messages = [str(w.message) for w in caught]
    assert len(messages) == len(set(messages))


def test_scenarios_sharing_an_area_are_predicted_once(gala):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = sars.habitat_loss(
            gala,
            area=900,
            fraction_lost=[0.25, 0.5, 0.75],
            models=["power"],
            n_boot=30,
            rng=np.random.default_rng(0),
            keep_samples=True,
        )
    assert list(loss.ci.area_grid) == [225, 450, 675, 900]
    assert loss.ci.samples.shape == (30, 4)
    assert (loss.table["lost_lower"] < loss.table["species_lost"]).all()


def test_samples_are_dropped_unless_asked_for(halved):
    assert halved.ci.samples.size == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"area": [[900, 100]], "fraction_lost": 0.5}, "2-D"),
        ({"area": [], "remaining": []}, "empty"),
        ({"area": [900, 800, 700], "remaining": [100, 200]}, "don't line up"),
        ({"area": 900, "fraction_lost": 0.5, "n_boot": -1}, "n_boot"),
        ({"area": 900, "fraction_lost": 0.5, "conf": 95}, "conf"),
    ],
)
def test_more_bad_arguments(gala, kwargs, match):
    with pytest.raises(ValueError, match=match):
        sars.habitat_loss(gala, models=["power"], **kwargs)
