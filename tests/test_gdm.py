"""Tests for the general dynamic model (``sar_gdm``).

The reference numbers are R sars's ``gdm`` on the example from its help page
(galap with an island age column), written by
``tests/r_reference/generate_r_reference.R``. The known differences from R
are listed with the others in ``tests/test_r_reference.py``.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars

REF = Path(__file__).parent / "r_reference"

FORMS = ["loga", "linear", "power_area", "power_area_time"]
POWER_FORMS = ["power_area", "power_area_time"]
CANDIDATES = ["GDM", "A + T", "A", "Intercept"]
PARAMS = ["Int", "A", "Ti", "Ti2"]


@pytest.fixture(scope="module")
def islands():
    """galap with the island ages of R's gdm example."""
    path = REF / "gdm_fitted_galap.csv"
    if not path.exists():
        pytest.skip("Run generate_r_reference.R first")
    return pd.read_csv(path)[["area", "species", "time"]]


@pytest.fixture(scope="module")
def r_fits():
    return pd.read_csv(REF / "gdm_galap.csv").set_index(["model", "candidate"])


@pytest.fixture(scope="module")
def r_fitted():
    return pd.read_csv(REF / "gdm_fitted_galap.csv")


@pytest.fixture(scope="module")
def fits(islands):
    """Each form fitted once with its nested comparison."""
    return {m: sars.sar_gdm(islands, model=m, mod_sel=True) for m in FORMS}


def synthetic(model="loga", n=12, noise=0.0, seed=0):
    """Islands whose richness follows a known GDM of the given form."""
    rng = np.random.default_rng(seed)
    area = np.geomspace(2.0, 5000.0, n)[rng.permutation(n)]
    time = np.linspace(0.5, 12.0, n)
    truth = {
        "loga": lambda: 5.0 + 6.0 * np.log(area) + 9.0 * time - 0.7 * time**2,
        "linear": lambda: 20.0 + 0.01 * area + 9.0 * time - 0.7 * time**2,
        "power_area": lambda: np.exp(1.0 + 0.25 * np.log(area) + 0.4 * time - 0.03 * time**2),
        "power_area_time": lambda: np.exp(
            1.0 + 0.25 * np.log(area) + 1.5 * np.log(time) - 0.4 * np.log(time) ** 2
        ),
    }[model]()
    return pd.DataFrame(
        {"area": area, "species": truth + noise * rng.standard_normal(n), "time": time}
    )


TRUE_PARAMS = {
    "loga": [5.0, 6.0, 9.0, -0.7],
    "linear": [20.0, 0.01, 9.0, -0.7],
    "power_area": [1.0, 0.25, 0.4, -0.03],
    "power_area_time": [1.0, 0.25, 1.5, -0.4],
}


# ---------------------------------------------------------------------------
# Agreement with R
# ---------------------------------------------------------------------------


class TestAgainstR:
    @pytest.mark.parametrize("model", FORMS)
    @pytest.mark.parametrize("candidate", CANDIDATES)
    def test_fit_statistics(self, fits, r_fits, model, candidate):
        fit = fits[model].candidates[candidate]
        ref = r_fits.loc[(model, candidate)]
        assert len(fit.params) + 1 == ref["k"]
        rss = float(np.sum(fit.residuals**2))
        # R's nls stops at a relative tolerance of about 1e-5, so its RSS can
        # sit a hair above the minimum; never below it.
        assert rss <= ref["rss"] * (1 + 1e-9)
        assert rss == pytest.approx(ref["rss"], rel=1e-7)
        assert fit.aic == pytest.approx(ref["aic"], abs=1e-5)
        assert fit.aicc == pytest.approx(ref["aicc"], abs=1e-5)
        assert fit.bic == pytest.approx(ref["bic"], abs=1e-5)
        assert fit.r_squared == pytest.approx(ref["r2"], abs=1e-7)

    @pytest.mark.parametrize("model", FORMS)
    @pytest.mark.parametrize("candidate", ["GDM", "A + T", "A"])
    def test_coefficients_errors_and_p_values(self, fits, r_fits, model, candidate):
        fit = fits[model].candidates[candidate]
        ref = r_fits.loc[(model, candidate)]
        names = [p for p in PARAMS if not np.isnan(ref[p])]
        assert list(fit.params) == names
        for name in names:
            # A coefficient near zero (the power forms' A) has no meaningful
            # relative error, so differences are measured in standard errors.
            assert abs(fit.params[name] - ref[name]) < 1e-3 * ref[f"se_{name}"]
            assert fit.std_errors[name] == pytest.approx(ref[f"se_{name}"], rel=1e-4)
            assert fit.p_values[name] == pytest.approx(ref[f"p_{name}"], rel=1e-3)

    @pytest.mark.parametrize("model", ["loga", "linear"])
    def test_ols_coefficients_match_closely(self, fits, r_fits, model):
        ref = r_fits.loc[(model, "GDM")]
        for name in PARAMS:
            assert fits[model].params[name] == pytest.approx(ref[name], rel=1e-6)

    @pytest.mark.parametrize("model", FORMS)
    def test_intercept_only_candidate(self, fits, r_fits, model):
        """R fits the intercept-only model as a mean of richness for every form.
        A power form reports the same fit as exp(Int), with Int on the log scale."""
        fit = fits[model].candidates["Intercept"]
        ref = r_fits.loc[(model, "Intercept")]
        mean = np.exp(fit.params["Int"]) if model in POWER_FORMS else fit.params["Int"]
        assert mean == pytest.approx(ref["Int"], rel=1e-9)
        # The standard error of log(mean) is that of the mean over the mean.
        scale = ref["Int"] if model in POWER_FORMS else 1.0
        assert fit.std_errors["Int"] == pytest.approx(ref["se_Int"] / scale, rel=1e-7)
        if model not in POWER_FORMS:
            assert fit.p_values["Int"] == pytest.approx(ref["p_Int"], rel=1e-6)

    @pytest.mark.parametrize("model", FORMS)
    def test_fitted_values(self, fits, r_fitted, model):
        np.testing.assert_allclose(fits[model].fitted, r_fitted[model], rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("model", FORMS)
    def test_mod_sel_table(self, fits, r_fits, model):
        """Delta AIC and delta AICc of the four candidates, as R's table has them."""
        table = fits[model].comparison.set_index("model")
        ref = r_fits.loc[model]
        assert set(table.index) == set(CANDIDATES)
        for col, r_col in [("AIC", "aic"), ("AICc", "aicc")]:
            np.testing.assert_allclose(table.loc[CANDIDATES, col], ref[r_col], atol=1e-5)
            np.testing.assert_allclose(
                table.loc[CANDIDATES, f"delta_{col}"], ref[r_col] - ref[r_col].min(), atol=1e-5
            )
        r_weights = sars.akaike_weights(ref["aicc"].to_numpy())
        np.testing.assert_allclose(table.loc[CANDIDATES, "weight"], r_weights, atol=1e-6)

    def test_loga_mod_sel_ranking(self, fits):
        # R's table for the logarithmic form: the GDM has the lowest AIC, the
        # area-only model the lowest AICc (delta AICc 1.327 for the GDM).
        table = fits["loga"].comparison
        assert table["model"].tolist() == ["A", "GDM", "A + T", "Intercept"]
        assert table.set_index("model").loc["GDM", "delta_AIC"] == 0
        assert table.set_index("model").loc["GDM", "delta_AICc"] == pytest.approx(
            1.326982247, abs=1e-6
        )

    def test_all_forms_ranking(self, islands, r_fits):
        """model="all": R ranks power_area, power_area_time, loga, linear."""
        multi = sars.sar_gdm(islands, model="all")
        ref = r_fits.xs("GDM", level="candidate")["aicc"].sort_values()
        assert multi.summary["model"].tolist() == ref.index.tolist()
        assert multi.summary["model"].tolist() == [
            "power_area",
            "power_area_time",
            "loga",
            "linear",
        ]
        np.testing.assert_allclose(multi.summary["AICc"], ref, atol=1e-5)
        np.testing.assert_allclose(multi.summary["delta_AICc"], ref - ref.iloc[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestSarGdm:
    def test_default_is_loga_without_comparison(self, islands, fits):
        fit = sars.sar_gdm(islands)
        assert isinstance(fit, sars.GDMFit)
        assert fit.model == "loga" and fit.terms == "GDM"
        assert fit.params == pytest.approx(fits["loga"].params)
        assert fit.comparison is None and fit.candidates is None
        assert fit.n == 16 and fit.converged

    @pytest.mark.parametrize("model", FORMS)
    def test_recovers_known_parameters(self, model):
        fit = sars.sar_gdm(synthetic(model), model=model)
        assert list(fit.params) == PARAMS
        np.testing.assert_allclose(list(fit.params.values()), TRUE_PARAMS[model], rtol=1e-6)
        assert fit.r_squared == pytest.approx(1.0)
        assert fit.converged

    @pytest.mark.parametrize("model", POWER_FORMS)
    def test_power_fit_is_a_least_squares_minimum(self, model):
        """No nearby coefficients fit richness better, and the fit beats the
        regression of log richness that it starts from."""
        data = synthetic(model, n=20, noise=3.0, seed=3)
        data["species"] = data["species"].clip(lower=0.0)
        fit = sars.sar_gdm(data, model=model)
        y = data["species"].to_numpy()
        rss = np.sum((y - fit.fitted) ** 2)
        beta = np.array(list(fit.params.values()))
        log_t = model == "power_area_time"
        t = np.log(data["time"]) if log_t else data["time"]
        design = np.column_stack([np.ones(len(y)), np.log(data["area"]), t, t**2])
        rng = np.random.default_rng(1)
        for _ in range(200):
            trial = beta + 1e-3 * rng.standard_normal(4) * np.abs(beta)
            assert np.sum((y - np.exp(design @ trial)) ** 2) >= rss - 1e-9
        log_fit = np.linalg.lstsq(design, np.log(np.maximum(y, 0.5)), rcond=None)[0]
        assert rss < np.sum((y - np.exp(design @ log_fit)) ** 2)

    @pytest.mark.parametrize("model", FORMS)
    @pytest.mark.parametrize(
        ("area_units", "time_units"),
        [(1e6, 1e3), (1.0, 1e6), (1e6, 1.0), (1e14, 1e9), (1e-9, 1e-6), (1e14, 1e-6)],
    )
    def test_fit_does_not_depend_on_units(self, islands, fits, model, area_units, time_units):
        """Area in m2 for km2, or age in years for millions of years, is the
        same data: the same fit, standard errors relative to their coefficients,
        p values and comparison table, not a "collinear" error."""
        scaled = islands.assign(
            area=islands["area"] * area_units, time=islands["time"] * time_units
        )
        a = fits[model]
        b = sars.sar_gdm(scaled, model=model, mod_sel=True)
        np.testing.assert_allclose(a.fitted, b.fitted, rtol=1e-5, atol=1e-5)
        assert a.aic == pytest.approx(b.aic, abs=1e-6)
        assert list(a.comparison["model"]) == list(b.comparison["model"])
        np.testing.assert_allclose(a.comparison["AICc"], b.comparison["AICc"], atol=1e-6)
        # Time squared carries the time units squared and nothing else in
        # every form but the one that logs time.
        if model != "power_area_time":
            assert b.params["Ti2"] * time_units**2 == pytest.approx(a.params["Ti2"], rel=1e-6)
            assert b.std_errors["Ti2"] * time_units**2 == pytest.approx(
                a.std_errors["Ti2"], rel=1e-6
            )
            assert b.p_values["Ti2"] == pytest.approx(a.p_values["Ti2"], rel=1e-6)
        if model == "linear":
            assert b.params["A"] * area_units == pytest.approx(a.params["A"], rel=1e-6)
            assert b.std_errors["A"] * area_units == pytest.approx(a.std_errors["A"], rel=1e-6)
        else:
            # A is the coefficient of log(area): a change of units moves Int only.
            assert b.params["A"] == pytest.approx(a.params["A"], rel=1e-5, abs=1e-8)
            assert b.std_errors["A"] == pytest.approx(a.std_errors["A"], rel=1e-5)
            assert b.p_values["A"] == pytest.approx(a.p_values["A"], rel=1e-5)

    def test_standard_errors_keep_their_precision_in_large_units(self, islands, r_fits):
        """Ages in years (R's example, times 3e5): R's standard errors and p
        values, rescaled, to the precision they have in the original units."""
        units = 3e5
        fit = sars.sar_gdm(islands.assign(time=islands["time"] * units))
        ref = r_fits.loc[("loga", "GDM")]
        for name, power in [("Int", 0), ("A", 0), ("Ti", 1), ("Ti2", 2)]:
            scale = units**power
            assert fit.params[name] * scale == pytest.approx(ref[name], rel=1e-6)
            assert fit.std_errors[name] * scale == pytest.approx(ref[f"se_{name}"], rel=1e-6)
            assert fit.p_values[name] == pytest.approx(ref[f"p_{name}"], rel=1e-6)

    def test_information_criteria_count_coefficients_and_sigma(self, fits):
        fit = fits["loga"]
        rss = float(np.sum(fit.residuals**2))
        aic, aicc, bic = sars._models._compute_ic(5, fit.n, rss)
        assert (fit.aic, fit.aicc, fit.bic) == pytest.approx((aic, aicc, bic))
        assert fit.aicc == pytest.approx(sars.aicc(5, fit.n, rss))
        assert fit.rse == pytest.approx(np.sqrt(rss / (fit.n - 4)))

    def test_fitted_and_residuals(self, islands, fits):
        for fit in fits.values():
            np.testing.assert_allclose(fit.fitted + fit.residuals, islands["species"])
            np.testing.assert_allclose(fit.fitted, fit.predict(islands["area"], islands["time"]))

    def test_custom_column_names(self, islands, fits):
        renamed = islands.rename(columns={"area": "a", "species": "s", "time": "age"})
        renamed["extra"] = "x"
        fit = sars.sar_gdm(renamed, area_col="a", species_col="s", time_col="age")
        assert fit.params == pytest.approx(fits["loga"].params)
        assert list(fit.data.columns) == ["area", "species", "time"]

    def test_input_is_not_modified(self, islands):
        before = islands.copy()
        sars.sar_gdm(islands, model="all", mod_sel=True)
        pd.testing.assert_frame_equal(islands, before)

    def test_zero_richness_is_allowed(self):
        """An island with no species is data; nothing takes the log of richness."""
        data = synthetic("power_area", noise=1.0)
        data.loc[0, "species"] = 0.0
        for model in FORMS:
            fit = sars.sar_gdm(data, model=model)
            assert np.all(np.isfinite(list(fit.params.values())))

    def test_formula(self, fits):
        assert fits["loga"].formula == "S = Int + A*log(area) + Ti*time + Ti2*time^2"
        assert fits["linear"].formula == "S = Int + A*area + Ti*time + Ti2*time^2"
        assert fits["power_area"].formula == "S = exp(Int + A*log(area) + Ti*time + Ti2*time^2)"
        assert (
            fits["power_area_time"].formula
            == "S = exp(Int + A*log(area) + Ti*log(time) + Ti2*log(time)^2)"
        )
        assert fits["power_area"].candidates["A"].formula == "S = exp(Int + A*log(area))"
        assert fits["linear"].candidates["Intercept"].formula == "S = Int"

    def test_exact_fit_has_finite_criteria(self):
        """A GDM through every island: no log(0) warning, finite criteria, and
        a comparison table that ranks it first with all the weight."""
        data = synthetic("loga")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = sars.sar_gdm(data, mod_sel=True)
        assert fit.r_squared == pytest.approx(1.0)
        assert np.all(np.isfinite([fit.aic, fit.aicc, fit.bic]))
        table = fit.comparison
        assert table.iloc[0]["model"] == "GDM"
        assert table[["delta_AIC", "delta_AICc", "weight_AIC", "weight"]].notna().all().all()
        assert table.iloc[0]["weight"] == pytest.approx(1.0)
        assert table.iloc[0]["delta_AICc"] == 0

    def test_exact_fits_are_ranked_by_parameter_count(self):
        """Richness that is an exact function of log(area) alone: all three
        models with an area term fit exactly, tie on the likelihood, and the
        one with the fewest parameters is first."""
        data = synthetic("loga")
        data["species"] = 5.0 + 6.0 * np.log(data["area"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = sars.sar_gdm(data, mod_sel=True)
        table = fit.comparison
        assert table["model"].tolist() == ["A", "A + T", "GDM", "Intercept"]
        exact = table.iloc[:3]
        np.testing.assert_allclose(exact["AIC"].diff().dropna(), 2.0)
        assert np.all(np.isfinite(table[["AIC", "AICc", "BIC"]].to_numpy()))
        assert table["weight"].notna().all() and table["weight"].sum() == pytest.approx(1.0)
        assert fit.candidates["A"].params == pytest.approx({"Int": 5.0, "A": 6.0})

    def test_repr(self, fits):
        text = repr(fits["loga"])
        assert text.startswith("GDMFit(model='loga', Int=")
        assert "Ti2=" in text and "AICc=" in text
        assert "terms='A + T'" in repr(fits["loga"].candidates["A + T"])


class TestModSel:
    @pytest.mark.parametrize("model", FORMS)
    def test_candidates(self, fits, model):
        fit = fits[model]
        assert list(fit.candidates) == CANDIDATES
        assert fit.candidates["GDM"] is fit
        for terms, n_params in zip(CANDIDATES, [4, 3, 2, 1], strict=True):
            cand = fit.candidates[terms]
            assert cand.terms == terms and cand.model == model
            assert list(cand.params) == PARAMS[:n_params]
            assert cand.comparison is None or cand is fit

    @pytest.mark.parametrize("model", FORMS)
    def test_nested_models_fit_no_better(self, fits, model):
        rss = [float(np.sum(fits[model].candidates[c].residuals ** 2)) for c in CANDIDATES]
        assert rss == sorted(rss)
        assert fits[model].candidates["Intercept"].r_squared == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("model", FORMS)
    def test_table(self, fits, model):
        table = fits[model].comparison
        assert list(table.columns) == [
            "model",
            "k",
            "RSE",
            "R2",
            "AIC",
            "AICc",
            "BIC",
            "delta_AIC",
            "delta_AICc",
            "weight_AIC",
            "weight",
            "converged",
        ]
        assert table["converged"].all()
        assert np.all(np.diff(table["AICc"]) >= 0)
        assert table["delta_AICc"].iloc[0] == 0 and table["delta_AIC"].min() == 0
        assert table["weight"].sum() == pytest.approx(1.0)
        assert table["weight_AIC"].sum() == pytest.approx(1.0)
        np.testing.assert_allclose(table["weight"], sars.akaike_weights(table["AICc"]))
        np.testing.assert_allclose(table["weight_AIC"], sars.akaike_weights(table["AIC"]))
        assert dict(zip(table["model"], table["k"], strict=True)) == {
            "GDM": 5,
            "A + T": 4,
            "A": 3,
            "Intercept": 2,
        }

    def test_area_only_candidates_are_the_sar_models(self, islands, fits):
        """The "A" candidate of each form is the plain SAR of that form."""
        for model, sar in [
            ("loga", sars.sar_loga),
            ("linear", sars.sar_linear),
            ("power_area", sars.sar_power),
        ]:
            cand = fits[model].candidates["A"]
            plain = sar(islands)
            assert cand.aicc == pytest.approx(plain.aicc, abs=1e-4)
            np.testing.assert_allclose(
                cand.predict(islands["area"]), plain.predict(islands["area"]), rtol=1e-4
            )

    def test_supports_a_real_hump(self):
        data = synthetic("loga", n=20, noise=2.0, seed=2)
        fit = sars.sar_gdm(data, mod_sel=True)
        assert fit.comparison.iloc[0]["model"] == "GDM"
        assert fit.params["Ti"] > 0 > fit.params["Ti2"]
        assert fit.p_values["Ti2"] < 0.001

    def test_small_sample_ranks_by_aic_where_aicc_is_undefined(self):
        """Six islands: AICc needs more than k + 1 = 6 for the GDM, which is
        therefore infinite and ranked last."""
        data = synthetic("loga", n=6, noise=1.0)
        fit = sars.sar_gdm(data, mod_sel=True)
        assert np.isinf(fit.aicc) and np.isfinite(fit.aic)
        assert fit.comparison.iloc[-1]["model"] == "GDM"
        assert fit.comparison.set_index("model").loc["GDM", "weight"] == 0
        assert fit.comparison["weight"].sum() == pytest.approx(1.0)


@pytest.fixture(scope="module")
def multi(islands):
    return sars.sar_gdm(islands, model="all", mod_sel=True)


def stall(monkeypatch, n_params):
    """Make the solver give up after one evaluation on fits of ``n_params``
    coefficients, so those fits don't converge."""
    from sars import _gdm

    least_squares = _gdm.least_squares

    def limited(fun, x0, **kwargs):
        if len(x0) in n_params:
            kwargs["max_nfev"] = 1
        return least_squares(fun, x0, **kwargs)

    monkeypatch.setattr(_gdm, "least_squares", limited)


class TestConvergence:
    def test_warns_when_the_gdm_does_not_converge(self, islands, monkeypatch):
        """A solver that runs out of evaluations is reported, not passed off as a fit."""
        stall(monkeypatch, {4})
        with pytest.warns(UserWarning, match="'power_area': .* did not converge for GDM") as rec:
            fit = sars.sar_gdm(islands, model="power_area")
        assert not fit.converged
        assert rec[0].filename == __file__
        # The closed-form fits don't use the solver.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert sars.sar_gdm(islands, model="loga", mod_sel=True).converged

    def test_candidate_that_does_not_converge_is_named_and_ranked_last(self, islands, monkeypatch):
        stall(monkeypatch, {3, 2})
        with pytest.warns(UserWarning, match=r"did not converge for A \+ T, A\.") as rec:
            fit = sars.sar_gdm(islands, model="power_area", mod_sel=True)
        assert len(rec) == 1
        assert fit.converged
        assert not fit.candidates["A + T"].converged and not fit.candidates["A"].converged
        table = fit.comparison
        assert table["model"].tolist()[:2] == ["GDM", "Intercept"]
        assert set(table["model"].tolist()[2:]) == {"A + T", "A"}
        assert table["converged"].tolist() == [True, True, False, False]
        failed = table[~table["converged"]]
        assert (failed["weight"] == 0).all() and (failed["weight_AIC"] == 0).all()
        assert failed[["delta_AIC", "delta_AICc"]].isna().all().all()
        # The rest are weighed among themselves.
        ok = table[table["converged"]]
        assert ok["weight"].sum() == pytest.approx(1.0)
        np.testing.assert_allclose(ok["weight"], sars.akaike_weights(ok["AICc"]))
        assert ok["delta_AICc"].iloc[0] == 0

    def test_all_forms_warn_once_per_form_at_the_callers_line(self, islands, monkeypatch):
        stall(monkeypatch, {4})
        with pytest.warns(UserWarning, match="did not converge") as rec:
            multi = sars.sar_gdm(islands, model="all")
        assert [str(w.message).split(":")[0] for w in rec] == [
            "Model 'power_area'",
            "Model 'power_area_time'",
        ]
        assert all(w.filename == __file__ for w in rec)
        s = multi.summary
        assert s["model"].tolist()[:2] == ["loga", "linear"]
        assert s["converged"].tolist() == [True, True, False, False]
        assert multi.best.model == "loga"


class TestAllForms:
    def test_returns_every_form(self, multi, fits):
        assert isinstance(multi, sars.GDMMultiFit)
        assert list(multi.fits) == FORMS
        for model in FORMS:
            assert multi.fits[model].params == pytest.approx(fits[model].params)
            # mod_sel applies to each form.
            assert list(multi.fits[model].comparison["model"]) == list(
                fits[model].comparison["model"]
            )

    def test_without_mod_sel_has_no_nested_tables(self, islands):
        multi = sars.sar_gdm(islands, model="all")
        assert all(f.comparison is None for f in multi.fits.values())

    def test_summary(self, multi):
        s = multi.summary
        assert len(s) == 4 and set(s["model"]) == set(FORMS)
        assert np.all(np.diff(s["AICc"]) >= 0)
        assert (s["k"] == 5).all()
        assert s["weight"].sum() == pytest.approx(1.0)
        assert s["converged"].all()
        # The columns sar_multi's summary also has are named and ordered as there.
        shared = ["model", "R2", "AIC", "AICc", "BIC", "delta_AICc", "weight"]
        assert [c for c in s.columns if c in shared] == shared
        # Same k for every form, so AIC and AICc weights agree.
        np.testing.assert_allclose(s["weight"], s["weight_AIC"])
        for _, row in s.iterrows():
            assert row["AIC"] == multi.fits[row["model"]].aic
            assert row["R2"] == multi.fits[row["model"]].r_squared

    def test_best(self, multi):
        assert multi.best is multi.fits[multi.summary.iloc[0]["model"]]
        assert multi.best.model == "power_area"
        assert repr(multi) == "GDMMultiFit(4 models, best='power_area')"


class TestPredict:
    @pytest.mark.parametrize("model", FORMS)
    def test_matches_the_equation(self, fits, model):
        fit = fits[model]
        p = fit.params
        area, time = np.array([3.0, 40.0, 900.0]), np.array([2.0, 7.0, 12.0])
        x = area if model == "linear" else np.log(area)
        t = np.log(time) if model == "power_area_time" else time
        eta = p["Int"] + p["A"] * x + p["Ti"] * t + p["Ti2"] * t**2
        expected = np.exp(eta) if model in POWER_FORMS else eta
        np.testing.assert_allclose(fit.predict(area, time), expected)

    def test_scalars_and_broadcasting(self, fits):
        fit = fits["power_area"]
        assert fit.predict(100.0, 5.0).shape == ()
        assert fit.predict(100, [2, 8, 14]).shape == (3,)
        assert fit.predict([10.0, 100.0], 5).shape == (2,)
        grid = fit.predict(np.array([10.0, 100.0]), np.array([[2.0], [8.0], [14.0]]))
        assert grid.shape == (3, 2)
        assert grid[1, 0] == pytest.approx(float(fit.predict(10.0, 8.0)))
        with pytest.raises(ValueError, match="broadcast"):
            fit.predict([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_hump_in_time(self, fits):
        """Richness at a fixed area peaks at the age -Ti / (2 Ti2)."""
        fit = fits["loga"]
        peak = -fit.params["Ti"] / (2 * fit.params["Ti2"])
        ages = np.array([peak - 3, peak, peak + 3])
        low, top, high = fit.predict(100.0, ages)
        assert top > low and top > high
        assert low == pytest.approx(high)

    def test_time_is_required_when_the_fit_has_a_time_term(self, fits):
        with pytest.raises(ValueError, match="time is required"):
            fits["loga"].predict(10.0)
        with pytest.raises(ValueError, match="time is required"):
            fits["loga"].candidates["A + T"].predict(10.0)

    def test_candidates_without_time_predict_from_area_alone(self, fits, islands):
        area_only = fits["loga"].candidates["A"]
        np.testing.assert_allclose(area_only.predict(islands["area"]), area_only.fitted)
        # A time passed to them shapes the result and changes nothing else.
        np.testing.assert_allclose(area_only.predict(islands["area"], 3.0), area_only.fitted)
        for cand in (area_only, fits["power_area"].candidates["Intercept"]):
            out = cand.predict(100.0, [1.0, 2.0, 3.0])
            assert out.shape == (3,)
            np.testing.assert_allclose(out, cand.predict(100.0))
            assert cand.predict([10.0, 100.0], [[1.0], [2.0], [3.0]]).shape == (3, 2)
            with pytest.raises(ValueError, match="broadcast"):
                cand.predict([1.0, 2.0], [1.0, 2.0, 3.0])
        for model in FORMS:
            flat = fits[model].candidates["Intercept"]
            np.testing.assert_allclose(flat.predict([1.0, 50.0]), islands["species"].mean())

    def test_domain_errors(self, fits):
        with pytest.raises(ValueError, match="Areas must be positive"):
            fits["loga"].predict([1.0, 0.0], 3.0)
        with pytest.raises(ValueError, match="Times must be positive"):
            fits["power_area_time"].predict(1.0, 0.0)
        # An intercept-only fit doesn't use area, so takes no log of it.
        assert np.isfinite(fits["loga"].candidates["Intercept"].predict(0.0))
        # The linear form takes no logs.
        assert np.isfinite(fits["linear"].predict(0.0, 0.0))
        assert np.isfinite(fits["power_area"].predict(1.0, 0.0))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_values_are_rejected(self, fits, bad):
        for fit in (fits["linear"], fits["power_area"], fits["loga"].candidates["A"]):
            with pytest.raises(ValueError, match="area contains NaN or infinite"):
                fit.predict([1.0, bad], 3.0)
            with pytest.raises(ValueError, match="time contains NaN or infinite"):
                fit.predict(1.0, [3.0, bad])


class TestValidation:
    def test_not_a_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            sars.sar_gdm([[1, 2, 3]])

    def test_unknown_model(self, islands):
        with pytest.raises(ValueError, match="Unknown model: 'ATT3'"):
            sars.sar_gdm(islands, model="ATT3")

    def test_missing_time_column(self, galap):
        with pytest.raises(KeyError, match="time"):
            sars.sar_gdm(galap)

    def test_missing_named_column(self, islands):
        with pytest.raises(KeyError, match="age"):
            sars.sar_gdm(islands, time_col="age")

    def test_columns_must_differ(self, islands):
        with pytest.raises(ValueError, match="three different columns"):
            sars.sar_gdm(islands, time_col="area")

    def test_too_few_islands(self, islands):
        with pytest.raises(ValueError, match="at least 5 islands"):
            sars.sar_gdm(islands.iloc[[0, 2, 4, 6]])

    @pytest.mark.parametrize("col", ["area", "species", "time"])
    def test_missing_and_infinite_values(self, islands, col):
        data = islands.astype(float)
        data.loc[3, col] = np.nan
        with pytest.raises(ValueError, match=f"'{col}' contains missing values"):
            sars.sar_gdm(data)
        data.loc[3, col] = np.inf
        with pytest.raises(ValueError, match=f"'{col}' contains infinite values"):
            sars.sar_gdm(data)

    @pytest.mark.parametrize("col", ["area", "species", "time"])
    def test_missing_value_in_a_nullable_column(self, islands, col):
        """pandas' nullable dtypes hold NA, which is reported as missing."""
        data = islands.astype({col: "Float64"})
        data.loc[3, col] = pd.NA
        with pytest.raises(ValueError, match=f"'{col}' contains missing values"):
            sars.sar_gdm(data)

    def test_nullable_and_integer_columns_are_numeric(self, islands, fits):
        data = islands.astype({"species": "Int64", "time": "int32", "area": "Float64"})
        assert sars.sar_gdm(data).params == pytest.approx(fits["loga"].params)

    @pytest.mark.parametrize("col", ["area", "species", "time"])
    def test_non_numeric_columns_are_not_converted(self, islands, col):
        """Text is rejected even when it spells numbers, and so are booleans:
        either is more likely the wrong column than data."""
        as_text = islands.assign(**{col: islands[col].astype(str)})
        with pytest.raises(ValueError, match=f"'{col}' must be a numeric column"):
            sars.sar_gdm(as_text)
        as_object = islands.assign(**{col: islands[col].astype(object)})
        with pytest.raises(ValueError, match="pandas.to_numeric"):
            sars.sar_gdm(as_object)
        as_bool = islands.assign(**{col: islands[col] > islands[col].median()})
        with pytest.raises(ValueError, match=f"'{col}' must be a numeric column, got dtype bool"):
            sars.sar_gdm(as_bool)
        words = islands.assign(**{col: ["old"] * len(islands)})
        with pytest.raises(ValueError, match=f"'{col}' must be a numeric column"):
            sars.sar_gdm(words)

    def test_non_positive_area(self, islands):
        data = islands.astype(float)
        data.loc[0, "area"] = 0.0
        with pytest.raises(ValueError, match="'area' must contain only positive"):
            sars.sar_gdm(data, model="linear")

    def test_negative_richness_and_time(self, islands):
        data = islands.astype(float)
        data.loc[0, "species"] = -1.0
        with pytest.raises(ValueError, match="'species' must not contain negative"):
            sars.sar_gdm(data)
        data = islands.astype(float)
        data.loc[0, "time"] = -1.0
        with pytest.raises(ValueError, match="'time' must not contain negative"):
            sars.sar_gdm(data)

    def test_zero_time_only_stops_the_log_time_form(self, islands):
        data = islands.astype(float)
        data.loc[1, "time"] = 0.0
        for model in ["loga", "linear", "power_area"]:
            assert sars.sar_gdm(data, model=model).converged
        for model in ["power_area_time", "all"]:
            with pytest.raises(ValueError, match="power_area_time"):
                sars.sar_gdm(data, model=model)

    def test_too_few_distinct_ages(self, islands):
        data = islands.assign(time=[1.0, 2.0] * 8)
        with pytest.raises(ValueError, match="at least 3 different values"):
            sars.sar_gdm(data)

    def test_constant_area_or_richness(self, islands):
        with pytest.raises(ValueError, match="'area' needs at least 2 different"):
            sars.sar_gdm(islands.assign(area=5.0))
        with pytest.raises(ValueError, match="nothing to fit"):
            sars.sar_gdm(islands.assign(species=5.0))

    def test_collinear_area_and_time(self, islands):
        """Age an exact linear function of log(area): their coefficients can't
        be separated in the logarithmic form."""
        data = islands.assign(time=np.log(islands["area"]) + 10.0)
        with pytest.raises(ValueError, match="collinear"):
            sars.sar_gdm(data, model="loga")
        # The linear form uses area itself, which is not collinear with it.
        assert sars.sar_gdm(data, model="linear").converged

    def test_duplicated_column_name(self, islands):
        data = pd.concat([islands, islands[["time"]]], axis=1)
        with pytest.raises(ValueError, match="'time' names more than one column"):
            sars.sar_gdm(data)

    def test_no_warnings_on_the_example(self, islands):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sars.sar_gdm(islands, model="all", mod_sel=True)
