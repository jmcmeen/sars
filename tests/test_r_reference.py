"""Agreement with R ``sars``, from the tables ``tests/r_reference/generate_r_reference.R``
writes (the R version is in ``r_version.txt``).

Two datasets: Galápagos (16 islands) and seven of its islands (rows 1, 3, ..., 13
by area), a small sample where R and this package often stop at different optima.

Every model on each dataset has an expected outcome:

- ``MATCH``: the same optimum as R. AICc agrees to ``AICC_TOL``, R² to ``R2_TOL``
  and every parameter to ``PARAM_RTOL`` (flat directions in the likelihood
  leave the last digits loose).
- ``BETTER``: a lower AICc than R's, where R stopped short of the optimum.
- ``SAME_FIT``: R's AICc (this package's is no worse), but the likelihood is flat
  along a ridge, so the parameters aren't identified and aren't compared: mmf's d
  and c growing together toward a power law on the small sample, or weibull4
  trading c toward 0 against a growing z.
- ``UNBOUNDED``: R's asymp fit has ``z > 1`` in ``d - c * z^A``, a curve that grows
  exponentially, which this package's form (``z > 0`` in ``d - c * exp(-z * A)``)
  deliberately can't reach. Its fit must keep an asymptote.
- a GitHub issue reference: a known difference, expected to fail the ``MATCH``
  checks until that issue is fixed (strict xfail, so the fix must update it).

Averaging is checked on the models that match: R's weights and curve, rebuilt
from R's own AICc and fitted values. Over all models the average differs from
R's by what the per-model differences imply.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars

REF = Path(__file__).parent / "r_reference"

AICC_TOL = 1e-4
R2_TOL = 1e-6
PARAM_RTOL = 5e-3
PARAM_ATOL = 1e-5

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
        **dict.fromkeys(["powerR", "epm1", "chapman", "weibull3", "gompertz", "betap"], BETTER),
        **dict.fromkeys(["mmf", "weibull4"], SAME_FIT),
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
    assert fit.aicc <= ref["aicc"] + AICC_TOL
    area = np.geomspace(1e-3, 1e5, 50)
    assert np.all(np.isfinite(fit.predict(area)))


@pytest.mark.parametrize(("stem", "model"), list(_cases(lambda o: o == UNBOUNDED)))
def test_asymp_keeps_its_asymptote_where_r_does_not(stem, model, fits):
    r_params = _r_params(_read(f"all_models_{stem}.csv").set_index("model").loc[model, "params"])
    assert r_params["z"] > 1
    assert fits[stem][model].params["z"] > 0


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
    three islands kept on each side (nisl = 3), the window this package searches."""
    data = _data("galap")
    fit = sars.sar_threshold(data)
    ref = _threshold_ref()
    for model in ["ContOne", "ZslopeOne", "Linear"]:
        mine = fit.all_fits[model]
        r = ref.loc[model]
        assert mine["rss"] == pytest.approx(r["rss"], rel=1e-9), model
        assert mine["aicc"] == pytest.approx(r["aicc"], abs=1e-6), model
        assert mine["aic"] == pytest.approx(r["aic"], abs=1e-6), model
        assert mine["r2"] == pytest.approx(r["r2"], abs=1e-9), model
        if model != "Linear":
            assert mine["breakpoint"] == pytest.approx(r["threshold"], abs=1e-9), model
    assert fit.best_model == ref.drop("Intercept")["aicc"].idxmin()
