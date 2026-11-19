"""Agreement with R ``sars``, from the tables ``tests/r_reference/generate_r_reference.R``
writes (the R version is in ``r_version.txt``).

Two datasets: Galápagos (16 islands) and seven of its islands (rows 1, 3, ..., 13
by area), a small sample where R and this package often stop at different optima.

Every model on each dataset has an expected outcome:

- ``MATCH``: the same optimum as R. AICc agrees to ``AICC_TOL``, R² to ``R2_TOL``
  and every parameter to ``PARAM_RTOL`` (flat directions in the likelihood
  leave the last digits loose).
- ``BETTER``: a lower AICc than R's, where R stopped short of the optimum.
- ``SAME_FIT``: R's AICc to within ``RIDGE_TOL``, but the likelihood is flat along
  a ridge, so the parameters aren't identified and aren't compared: mmf's and
  weibull3's d growing toward a power law on the small sample, or weibull4 trading
  c toward 0 against a growing z. Where a solver stops on a ridge varies with the
  platform's floating point by a few thousandths of an AICc unit, which moves
  Akaike weights by about 0.1%.
- ``UNBOUNDED``: R's asymp fit has ``z > 1`` in ``d - c * z^A``, a curve that grows
  exponentially, which this package's form (``z > 0`` in ``d - c * exp(-z * A)``)
  deliberately can't reach. Its fit must keep an asymptote.
- a GitHub issue reference: a known difference, expected to fail the ``MATCH``
  checks until that issue is fixed (strict xfail, so the fix must update it).

Each matching model's parameter table (``SARFit.param_table``) is checked
against R's ``sigConf``: standard errors, t values and p values, to ``TABLE_RTOL``.
The two packages' optima differ in the fourth digit or so, which moves these
numbers more than the error in either's numerical Jacobian. R's interval columns
are the estimate ± 2 standard errors, not a t quantile, so they aren't compared.
Fits listed in ``TABLE_NOT_COMPARED`` are left out.

Averaging is checked on the models that match: R's weights and curve, rebuilt
from R's own AICc and fitted values. Over all models the average differs from
R's by what the per-model differences imply.

The threshold models match R's fits. For DiscOne and DiscTwo, R's
``sar_threshold`` searches outside its ``nisl`` window: DiscOne scores each
breakpoint as the one ``nisl - 1`` places lower, and both let the last segment
hold ``nisl - 1`` islands. Their reference breakpoints are therefore searched in
the generator with R's own model functions, over the window this package uses.

Coleman's random placement model is checked on R's simulated abundance matrix,
which the generator writes out alongside R's expected richness and standard
deviations.
The general dynamic model (``sar_gdm``) is checked against R's ``gdm`` in
``tests/test_gdm.py`` (``TestAgainstR``), from ``gdm_galap.csv`` and
``gdm_fitted_galap.csv``: every form and nested candidate matches R's fit. The
known differences are in what is reported, not in the fits:

- The intercept-only candidate of the power forms. R fits it as the mean
  richness for every form, so its ``Int`` is that mean; this package reports it
  as ``exp(Int)`` like the other power-form candidates, so ``Int`` is the log of
  the mean. Same fitted values and criteria.
- The comparison tables are sorted by AICc, R's by AIC. Both criteria, their
  deltas and their weights are in the table.
- R's ``"ATT2"`` option (the logarithmic form fitted by ``lm``) is ``"loga"``
  here: the same fit.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars

REF = Path(__file__).parent / "r_reference"

AICC_TOL = 1e-4
RIDGE_TOL = 1e-2
R2_TOL = 1e-6
PARAM_RTOL = 5e-3
PARAM_ATOL = 1e-5
TABLE_RTOL = 1e-2
P_VALUE_ATOL = 1e-3

MATCH = "match"
BETTER = "better"
SAME_FIT = "same fit, parameters along a flat ridge"
UNBOUNDED = "R's fit grows without bound"

EXPECTED = {
    "galap": {
        **dict.fromkeys(
            [
                "power",
                "powerR",
                "epm1",
                "epm2",
                "p1",
                "p2",
                "loga",
                "koba",
                "mmf",
                "monod",
                "negexpo",
                "chapman",
                "weibull3",
                "asymp",
                "ratio",
                "gompertz",
                "heleg",
                "linear",
            ],
            MATCH,
        ),
        "weibull4": SAME_FIT,
        "betap": BETTER,
    },
    "galap7": {
        **dict.fromkeys(
            [
                "power",
                "epm2",
                "p1",
                "p2",
                "loga",
                "koba",
                "monod",
                "negexpo",
                "ratio",
                "heleg",
                "linear",
            ],
            MATCH,
        ),
        **dict.fromkeys(["powerR", "epm1", "chapman", "gompertz", "betap"], BETTER),
        **dict.fromkeys(["mmf", "weibull3", "weibull4"], SAME_FIT),
        "asymp": UNBOUNDED,
    },
}


@cache
def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(REF / name)


# The models R's sar_average weighed (it has no mmf; this package has no logistic).
SHARED = _read("average_galap.csv")["model"].tolist()


def _data(stem: str) -> pd.DataFrame:
    return _read(f"{stem}.csv").rename(columns={"a": "area", "s": "species"})


@pytest.fixture(scope="session")
def fits(galap_fits):
    """Fits keyed by dataset, then model (galap reuses the shared session fits)."""
    galap7 = {f.model: f for f in sars.sar_multi(_data("galap7")).fits}
    return {"galap": galap_fits, "galap7": galap7}


def _r_params(text: str) -> dict[str, float]:
    return {k: float(v) for k, v in (kv.split("=") for kv in text.split("; "))}


def _params_on_r_scale(fit: sars.SARFit) -> dict[str, float]:
    params = dict(fit.params)
    if fit.model == "asymp":  # fitted as d - c*exp(-z*A); R fits d - c*z^A
        params["z"] = float(np.exp(-params["z"]))
    return params


def _is_issue(outcome: str) -> bool:
    return outcome.startswith("#")


def _cases(select):
    """One pytest param per (dataset, model) whose outcome ``select`` accepts;
    known differences are strict xfails."""
    for stem, models in EXPECTED.items():
        for model, outcome in models.items():
            if select(outcome):
                marks = []
                if _is_issue(outcome):
                    marks = [pytest.mark.xfail(strict=True, reason=outcome)]
                yield pytest.param(stem, model, marks=marks, id=f"{stem}-{model}")


def test_every_model_has_an_expected_outcome():
    for stem, models in EXPECTED.items():
        assert set(models) == set(_read(f"all_models_{stem}.csv")["model"]), stem


@pytest.mark.parametrize(("stem", "model"), list(_cases(lambda o: o == MATCH or _is_issue(o))))
def test_fit_matches_r(stem, model, fits):
    ref = _read(f"all_models_{stem}.csv").set_index("model").loc[model]
    fit = fits[stem][model]
    assert fit.aicc == pytest.approx(ref["aicc"], abs=AICC_TOL)
    assert fit.aic == pytest.approx(ref["aic"], abs=AICC_TOL)
    assert fit.bic == pytest.approx(ref["bic"], abs=AICC_TOL)
    assert fit.r_squared == pytest.approx(ref["r2"], abs=R2_TOL)
    r_params = _r_params(ref["params"])
    params = _params_on_r_scale(fit)
    assert set(params) == set(r_params)
    for name, value in r_params.items():
        assert params[name] == pytest.approx(value, rel=PARAM_RTOL, abs=PARAM_ATOL), name


@pytest.mark.parametrize(("stem", "model"), list(_cases(lambda o: o == BETTER)))
def test_fit_beats_r_where_r_stops_short(stem, model, fits):
    ref = _read(f"all_models_{stem}.csv").set_index("model").loc[model]
    assert fits[stem][model].aicc < ref["aicc"] - AICC_TOL


@pytest.mark.parametrize(("stem", "model"), list(_cases(lambda o: o == SAME_FIT)))
def test_fit_is_no_worse_than_r_on_a_flat_ridge(stem, model, fits):
    ref = _read(f"all_models_{stem}.csv").set_index("model").loc[model]
    fit = fits[stem][model]
    assert fit.aicc <= ref["aicc"] + RIDGE_TOL
    area = np.geomspace(1e-3, 1e5, 50)
    assert np.all(np.isfinite(fit.predict(area)))


@pytest.mark.parametrize(("stem", "model"), list(_cases(lambda o: o == UNBOUNDED)))
def test_asymp_keeps_its_asymptote_where_r_does_not(stem, model, fits):
    r_params = _r_params(_read(f"all_models_{stem}.csv").set_index("model").loc[model, "params"])
    assert r_params["z"] > 1
    assert fits[stem][model].params["z"] > 0


# ---------------------------------------------------------------------------
# Parameter tables
# ---------------------------------------------------------------------------


# Matching fits whose tables aren't compared with R's.
TABLE_NOT_COMPARED = {
    # The fit sits on a bound (p1's d and heleg's f at 0), where the two
    # packages stop at different tiny values and the usual standard errors
    # don't apply. This package's agree with the analytic Jacobian there; R's
    # differ from them by 3% (p1) and up to 60% (heleg).
    ("galap7", "p1"): "d is on its bound of 0",
    ("galap7", "heleg"): "f is on its bound of 0",
    ("galap7", "ratio"): "R gives no table (sigConf is NA)",
}


@pytest.mark.parametrize(
    ("stem", "model"),
    [
        pytest.param(stem, model, id=f"{stem}-{model}")
        for stem, models in EXPECTED.items()
        for model, outcome in models.items()
        if outcome == MATCH and (stem, model) not in TABLE_NOT_COMPARED
    ],
)
def test_param_table_matches_r(stem, model, fits):
    ref = _read(f"param_table_{stem}.csv")
    ref = ref[ref["model"] == model].set_index("param")
    table = fits[stem][model].param_table()
    assert set(table.index) == set(ref.index)
    for name, r in ref.iterrows():
        mine = table.loc[name]
        if model == "asymp" and name == "z":
            # R's z is exp(-z) here, so its standard error is this one times
            # exp(-z) (the delta method). The t test asks a different question
            # of each form (z = 0 here, exp(-z) = 0 in R), so it isn't compared.
            r_form = np.exp(-mine["estimate"])
            assert r_form * mine["std_error"] == pytest.approx(r["std_error"], rel=TABLE_RTOL)
            continue
        assert mine["std_error"] == pytest.approx(r["std_error"], rel=TABLE_RTOL), name
        assert mine["t_value"] == pytest.approx(r["t_value"], rel=TABLE_RTOL), name
        assert mine["p_value"] == pytest.approx(r["p_value"], rel=TABLE_RTOL, abs=P_VALUE_ATOL), (
            name
        )


def test_tables_not_compared_are_matching_fits():
    for stem, model in TABLE_NOT_COMPARED:
        assert EXPECTED[stem][model] == MATCH


def test_r_reports_a_standard_error_on_the_ridge_where_this_package_gives_nan(fits):
    """On weibull4's flat ridge R's table has a standard error for z a ten
    millionth of the estimate, which the fit can't support (R's own z differs
    from this package's in the third digit). This package reports NaN."""
    ref = _read("param_table_galap.csv")
    r = ref[ref["model"] == "weibull4"].set_index("param").loc["z"]
    assert r["std_error"] < 1e-5 * r["estimate"]
    with pytest.warns(UserWarning, match="weibull4"):
        table = fits["galap"]["weibull4"].param_table()
    assert np.isnan(table.loc["z", "std_error"])


# ---------------------------------------------------------------------------
# Multimodel weights and averaging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", ["galap", "galap7"])
def test_akaike_weights_reproduce_r_from_rs_aicc(stem):
    ref = _read(f"average_{stem}.csv")
    weights = sars.akaike_weights(ref["aicc"].to_numpy())
    np.testing.assert_allclose(weights, ref["weight"], rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("stem", ["galap", "galap7"])
def test_average_of_the_models_that_match_reproduces_r(stem):
    """Averaging the models that reach R's optimum gives R's weights and curve,
    rebuilt from R's own AICc values and fitted values."""
    data = _data(stem)
    models = [m for m in SHARED if EXPECTED[stem][m] == MATCH]
    ref = _read(f"all_models_{stem}.csv").set_index("model").loc[models]
    delta = ref["aicc"] - ref["aicc"].min()
    r_weights = np.exp(-0.5 * delta) / np.exp(-0.5 * delta).sum()
    r_curve = _read(f"fitted_{stem}.csv")[models].to_numpy() @ r_weights.to_numpy()

    avg = sars.sar_average(data, models=models)
    for model in models:
        assert avg.weights[model] == pytest.approx(r_weights[model], abs=1e-6), model
    # Flat directions leave fitted values loose in the fourth significant digit.
    np.testing.assert_allclose(avg.predict(data["area"].to_numpy()), r_curve, rtol=1e-4, atol=5e-3)


# ---------------------------------------------------------------------------
# Threshold models
# ---------------------------------------------------------------------------


def _threshold_ref() -> pd.DataFrame:
    return _read("threshold_galap.csv").set_index("model")


def test_threshold_models_match_r():
    """R searched ln(area) in steps of 0.01 (its default, and this package's) with
    three islands kept in the first and last segments (nisl = 3), the window this
    package searches. R's own search for DiscOne and DiscTwo strays from that
    window, so the reference searches the observed areas with R's model functions
    (see generate_r_reference.R)."""
    data = _data("galap")
    models = ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo", "Linear"]
    fit = sars.sar_threshold(data, models=models)
    ref = _threshold_ref()
    for model in models:
        mine = fit.all_fits[model]
        r = ref.loc[model]
        assert mine["k"] == r["k"], model
        assert mine["rss"] == pytest.approx(r["rss"], rel=1e-9), model
        assert mine["aicc"] == pytest.approx(r["aicc"], abs=1e-6), model
        assert mine["aic"] == pytest.approx(r["aic"], abs=1e-6), model
        assert mine["r2"] == pytest.approx(r["r2"], abs=1e-9), model
        expected = [t for t in (r["threshold"], r["threshold2"]) if not np.isnan(t)]
        assert mine["breakpoints"] == pytest.approx(tuple(expected), abs=1e-9), model
    assert fit.best_model == ref.drop("Intercept")["aicc"].idxmin()


# ---------------------------------------------------------------------------
# Coleman's random placement model
# ---------------------------------------------------------------------------


def test_coleman_matches_r():
    """R's simulated abundance matrix (22 species in rows, 11 sites in columns).
    R returns the expected richness, its standard deviation, the relative areas
    and the observed richness; the verdict follows from those."""
    abundance = _read("coleman_abundance.csv")
    ref = _read("coleman_sites.csv").set_index("site")
    assert abundance.shape == (22, 11)

    fit = sars.coleman(abundance, ref["area"].to_numpy())
    assert fit.sites.index.tolist() == ref.index.tolist()
    np.testing.assert_allclose(fit.sites["relative_area"], ref["relative_area"], rtol=1e-12)
    np.testing.assert_array_equal(fit.sites["observed"], ref["observed"])
    np.testing.assert_allclose(fit.sites["expected"], ref["expected"], rtol=1e-10)
    np.testing.assert_allclose(fit.sites["sd"], ref["sd"], rtol=1e-10)

    # The one-third rule applied to R's own numbers: 9 of the 11 sites lie
    # more than one standard deviation out, so random placement is rejected.
    outside = (ref["observed"] - ref["expected"]).abs() > ref["sd"]
    assert outside.sum() == 9
    assert (~fit.sites["within_sd"]).tolist() == outside.tolist()
    assert fit.fraction_outside == pytest.approx(9 / 11)
    assert fit.rejected


@pytest.mark.parametrize("name", ["aegean", "aegean2", "niering"])
def test_threshold_models_match_r_on_bundled_datasets(name):
    """The default models on the larger datasets, where thresholds are usually
    shown: R's breakpoints, fit and ranking. On niering ContOne has the lower
    RSS and ZslopeOne the lower AICc, so the ranking there rests on the
    parameter counts."""
    ref = _read("threshold_datasets.csv")
    ref = ref[ref["dataset"] == name].set_index("model")
    fit = sars.sar_threshold(getattr(sars, f"load_{name}")())
    assert set(fit.all_fits) == set(ref.index)
    for model, r in ref.iterrows():
        mine = fit.all_fits[model]
        assert mine["k"] == r["k"], model
        assert mine["rss"] == pytest.approx(r["rss"], rel=1e-9), model
        assert mine["aic"] == pytest.approx(r["aic"], abs=1e-6), model
        assert mine["aicc"] == pytest.approx(r["aicc"], abs=1e-6), model
        expected = () if np.isnan(r["threshold"]) else (r["threshold"],)
        assert mine["breakpoints"] == pytest.approx(expected, abs=1e-9), model
    assert fit.best_model == ref["aicc"].idxmin()
