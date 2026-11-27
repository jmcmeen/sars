"""Habitat and countryside species-area models, against R sars 2.1.1."""

import numpy as np
import pandas as pd
import pytest

import sars

REF = __import__("pathlib").Path(__file__).parent / "r_reference"
HABITAT = pd.read_csv(REF / "habitat_models.csv")
CS_FITS = pd.read_csv(REF / "countryside_fits.csv")
CS_TOTAL = pd.read_csv(REF / "countryside_total.csv")
CS_RSS = pd.read_csv(REF / "countryside_rss.csv")
AREAS = ["Area_AG", "Area_SH", "Area_QF"]
GROUPS = ["Spcs_AG", "Spcs_SH", "Spcs_QF", "Spcs_UB"]


@pytest.fixture(scope="module")
def habitat_fits():
    return {
        mt: sars.sar_habitat(sars.load_habitat(), mod_type=mt)
        for mt in ("power", "logarithmic", "power_log")
    }


@pytest.fixture(scope="module")
def countryside_fits():
    cs = sars.load_countryside()
    return {
        mt: sars.sar_countryside(cs, habitats=AREAS, species=GROUPS, mod_type=mt)
        for mt in ("power", "logarithmic")
    }


@pytest.mark.parametrize(
    "row", [r for _, r in HABITAT.iterrows()], ids=lambda r: f"{r['modType']}-{r['model']}"
)
def test_habitat_models_match_r(habitat_fits, row):
    fit = habitat_fits[row["modType"]]
    got = fit.summary.set_index("model").loc[row["model"]]
    for column in ("rss", "R2", "adjR2", "AIC", "BIC", "AICc"):
        assert got[column] == pytest.approx(row[column], rel=1e-6, abs=1e-8), column
    assert got["rss"] <= row["rss"] * (1 + 1e-9)  # never a worse fit than R's
    p = fit.params[row["model"]]
    # The parameters agree to 1e-3: in the logarithmic jigsaw model c sits
    # inside a log, and c and z trade off along a ridge where the RSS is flat.
    assert p["c"] == pytest.approx(row["p1"], rel=1e-3)
    assert p["z"] == pytest.approx(row["p2"], rel=1e-3)
    if not np.isnan(row["p3"]):
        # R's jigsaw in the log-log form has log H's coefficient, which is d - z.
        d = p["d"] - p["z"] if (row["modType"], row["model"]) == ("power_log", "jigsaw") else p["d"]
        assert d == pytest.approx(row["p3"], rel=1e-3, abs=1e-7)


def test_habitat_summary_is_ranked_by_aicc(habitat_fits):
    summary = habitat_fits["power_log"].summary
    assert list(summary["model"]) == ["choros", "jigsaw", "power", "Kallimanis"]
    assert summary["AICc"].is_monotonic_increasing


@pytest.mark.parametrize("mod_type", ["power", "logarithmic", "power_log"])
def test_habitat_predictions_reproduce_the_fitted_values(habitat_fits, mod_type):
    fit = habitat_fits[mod_type]
    data = fit.data
    for model, row in fit.summary.set_index("model").iterrows():
        pred = fit.predict(data["area"], data["habitats"], model=model)
        observed = data["species"].to_numpy(dtype=float)
        if mod_type == "power_log":
            resid = np.log(observed) - np.log(pred)
        else:
            resid = observed - pred
        assert np.sum(resid**2) == pytest.approx(row["rss"], rel=1e-8)


def test_habitat_log_base_and_con():
    data = sars.load_habitat()
    natural = sars.sar_habitat(data)
    ten = sars.sar_habitat(data, log_base=10)
    a, h = data["area"].iloc[:3], data["habitats"].iloc[:3]
    np.testing.assert_allclose(ten.predict(a, h), natural.predict(a, h), rtol=1e-10)
    with_zero = data.assign(species=data["species"].where(data.index != 0, 0))
    with pytest.raises(ValueError, match="give con"):
        sars.sar_habitat(with_zero)
    fit = sars.sar_habitat(with_zero, con=1)
    assert fit.con == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mod_type": "linear"}, "mod_type"),
        ({"log_base": 3}, "log_base"),
        ({"habitats_col": "nope"}, "No column 'nope'"),
    ],
)
def test_habitat_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sars.sar_habitat(sars.load_habitat(), **kwargs)


@pytest.mark.parametrize("mod_type", ["power", "logarithmic"])
def test_countryside_matches_r(countryside_fits, mod_type):
    fit = countryside_fits[mod_type]
    ref = CS_FITS[CS_FITS["modType"] == mod_type].reset_index(drop=True)
    assert fit.groups == GROUPS and not fit.not_fitted
    for i, group in enumerate(GROUPS):
        got = fit.params.loc[group]
        assert got["rss"] == pytest.approx(ref["rss"][i], rel=1e-6)
        assert got["z"] == pytest.approx(ref["z"][i], rel=1e-3)
        assert fit.c[group] == pytest.approx(ref["c"][i], rel=1e-3)
        # Affinities near zero are where the fit is flat; compare the clear ones.
        for j, habitat in enumerate(AREAS):
            r = ref[f"aff{j + 1}"][i]
            if r > 1e-3:
                assert fit.affinity.loc[group, habitat] == pytest.approx(r, rel=1e-3)
            else:
                assert fit.affinity.loc[group, habitat] < 1e-2
    np.testing.assert_allclose(fit.predicted_total, CS_TOTAL[mod_type], atol=1e-4)
    rss = CS_RSS[CS_RSS["modType"] == mod_type].iloc[0]
    assert fit.rss["countryside"] == pytest.approx(rss["countryside_rss"], rel=1e-6)
    assert fit.rss[mod_type] == pytest.approx(rss["sar_rss"], rel=1e-6)


def test_countryside_predicts_for_any_mix_of_habitats(countryside_fits):
    fit = countryside_fits["power"]
    one = fit.predict([1000.0, 0.0, 5000.0])
    assert list(one.columns) == [*GROUPS, "total"]
    assert one["total"].iloc[0] == pytest.approx(one[GROUPS].iloc[0].sum())
    two = fit.predict([[1000.0, 0.0, 5000.0], [0.0, 6000.0, 0.0]])
    assert len(two) == 2
    with pytest.raises(ValueError, match="each of the 3 habitats"):
        fit.predict([1.0, 2.0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mod_type": "linear"}, "mod_type"),
        ({"grid_start": "some"}, "grid_start"),
        ({"habitats": ["nope"]}, "No columns"),
        ({"habitats": []}, "at least one"),
    ],
)
def test_countryside_bad_arguments(kwargs, match):
    args = {"habitats": AREAS, "species": GROUPS, **kwargs}
    with pytest.raises(ValueError, match=match):
        sars.sar_countryside(sars.load_countryside(), **args)


def test_both_save_and_reload(habitat_fits, countryside_fits):
    for fit in (habitat_fits["power"], countryside_fits["logarithmic"]):
        again = sars.from_json(sars.to_json(fit))
        assert type(again) is type(fit)
        if isinstance(fit, sars.CountrysideFit):
            pd.testing.assert_frame_equal(again.params, fit.params)
            np.testing.assert_array_equal(again.predicted_total, fit.predicted_total)
        else:
            pd.testing.assert_frame_equal(again.summary, fit.summary)


def test_the_datasets():
    habitat = sars.load_habitat()
    assert list(habitat.columns) == ["area", "habitats", "species"] and len(habitat) == 12
    cs = sars.load_countryside()
    assert list(cs.columns) == ["site", *AREAS, *GROUPS] and len(cs) == 425


def test_habitat_con_must_make_richness_positive_and_is_ignored_without_zeros():
    data = sars.load_habitat()
    with_zero = data.assign(species=data["species"].where(data.index != 0, 0))
    for bad in (0, -1):
        with pytest.raises(ValueError, match="con must make every richness positive"):
            sars.sar_habitat(with_zero, con=bad)
    with pytest.warns(UserWarning, match="con is added only when"):
        assert sars.sar_habitat(data, con=5).con is None


@pytest.mark.parametrize(("area", "habitats"), [(-1, 2), (10, 0), (np.nan, 3)])
def test_habitat_predict_refuses_impossible_inputs(habitat_fits, area, habitats):
    with pytest.raises(ValueError, match="positive numbers"):
        habitat_fits["power"].predict(area, habitats)


def test_countryside_predict_refuses_impossible_inputs(countryside_fits):
    with pytest.raises(ValueError, match="not negative"):
        countryside_fits["power"].predict([-1.0, 0.0, 10.0])
    # Oak-forest specialists barely use farmland: in the logarithmic form a
    # site of farmland alone has (almost) no weighted area for some group.
    log_fit = countryside_fits["logarithmic"]
    h = log_fit.params[[f"h_{a}" for a in AREAS]]
    if (h["h_Area_AG"] == 0).any():
        with pytest.raises(ValueError, match="weighted area is zero"):
            log_fit.predict([1000.0, 0.0, 0.0])


def test_countryside_totals_need_every_group(monkeypatch):
    from sars import _habitat

    real = _habitat._fit_curve

    def fail_one(f, x, y, starts, lower=None):
        if np.array_equal(y, sars.load_countryside()["Spcs_UB"].to_numpy(dtype=float)):
            return None
        return real(f, x, y, starts, lower)

    monkeypatch.setattr(_habitat, "_fit_curve", fail_one)
    with pytest.warns(UserWarning, match="Spcs_UB"):
        fit = sars.sar_countryside(sars.load_countryside(), habitats=AREAS, species=GROUPS)
    assert fit.not_fitted == ["Spcs_UB"] and fit.groups == GROUPS[:3]
    assert len(fit.predicted_total) == 425 and np.isnan(fit.predicted_total).all()
    assert fit.rss == {}
    assert np.isnan(fit.predict([1000.0, 0.0, 0.0])["total"].iloc[0])


def test_countryside_rss_compares_the_same_sites():
    """A site with no area counts against the single curve too."""
    cs = sars.load_countryside().copy()
    cs.loc[cs.index[-1], AREAS] = 0
    cs.loc[cs.index[-1], GROUPS] = [1, 1, 1, 0]
    with pytest.warns(UserWarning, match="total area of zero"):
        fit = sars.sar_countryside(cs, habitats=AREAS, species=GROUPS)
    total = cs[GROUPS].sum(axis=1).to_numpy(dtype=float)
    assert fit.rss["countryside"] == pytest.approx(np.sum((fit.predicted_total - total) ** 2))
    # The last site, with no area, contributes 3**2 to both.
    assert fit.rss["power"] > 9


@pytest.mark.parametrize("z_lower", [np.nan, "0"])
def test_countryside_z_lower_must_be_a_finite_number(z_lower):
    with pytest.raises(ValueError, match="z_lower"):
        sars.sar_countryside(
            sars.load_countryside(), habitats=AREAS, species=GROUPS, z_lower=z_lower
        )


def test_countryside_z_lower_accepts_numpy_numbers():
    fit = sars.sar_countryside(
        sars.load_countryside(), habitats=AREAS, species=GROUPS[:1], z_lower=np.int64(0)
    )
    assert fit.groups == GROUPS[:1]


def test_countryside_all_zero_weights_warn(monkeypatch):
    from sars import _habitat

    monkeypatch.setattr(
        _habitat,
        "_fit_curve",
        lambda f, x, y, starts, lower=None: (np.array([0.0, 0.0, 0.0, 0.5]), 1.0),
    )
    with pytest.warns(UserWarning, match="Every habitat weight is 0"):
        fit = sars.sar_countryside(sars.load_countryside(), habitats=AREAS, species=GROUPS[:1])
    assert fit.affinity.isna().all().all()


@pytest.mark.slow
@pytest.mark.parametrize("mod_type", ["power", "logarithmic"])
def test_screened_starts_find_the_full_grids_optimum(mod_type):
    """On groups other than the shipped ones, the screened starts (the
    default) give the fits R's whole grid does."""
    cs = sars.load_countryside()
    merged = cs.assign(G1=cs.Spcs_AG + cs.Spcs_UB, G2=cs.Spcs_SH + cs.Spcs_QF)
    args = {"habitats": AREAS, "species": ["G1", "G2"], "mod_type": mod_type}
    screened = sars.sar_countryside(merged, **args)
    full = sars.sar_countryside(merged, n_starts=None, **args)
    np.testing.assert_allclose(screened.params["rss"], full.params["rss"], rtol=1e-6)


@pytest.mark.parametrize(("grid_start", "k"), [("partial", 7), ("exhaustive", 5), ("none", 14)])
def test_countryside_refuses_a_grid_too_large_to_build(grid_start, k):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    data = pd.DataFrame({f"h{i}": rng.uniform(1, 10, 12) for i in range(k)})
    data["s"] = rng.integers(1, 20, 12)
    with pytest.raises(ValueError, match="starting values"):
        sars.sar_countryside(
            data, habitats=[f"h{i}" for i in range(k)], species=["s"], grid_start=grid_start
        )
