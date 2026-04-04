import pandas as pd
import pytest

import sars


def test_power_law_spot_check():
    """Quick sanity check using confirmed R values (no R script needed)."""
    partial = pd.DataFrame({
        "area":    [0.20, 0.90, 1.00, 1.80, 1.87, 4.40, 7.10,
                    7.50, 18.00, 20.00],
        "species": [  48,    7,   52,   14,   42,   22,  103,
                        48,    79,   119],
    })
    fit = sars.sar_power(partial)
    assert fit.converged is True
    assert 0.0 < fit.params["z"] < 1.0
    assert fit.params["c"] > 0
    assert 0.0 < fit.r_squared <= 1.0


def test_power_law_matches_r(galap, r_reference):
    """Full validation: must match R sars::sar_power(galap) exactly."""
    fit = sars.sar_power(galap)
    ref = r_reference.loc["power"]
    assert fit.r_squared == pytest.approx(ref["r2"], abs=0.005)
    assert fit.aicc == pytest.approx(ref["aicc"], abs=0.1)
    assert fit.params["c"] == pytest.approx(33.1792, abs=0.05)
    assert fit.params["z"] == pytest.approx(0.2832, abs=0.005)


# ---------------------------------------------------------------------------
# Parametrized validation for all 20 models (uses cached galap_fits)
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "power", "powerR", "epm1", "epm2", "p1", "p2", "loga", "koba",
    "mmf", "monod", "negexpo", "chapman", "weibull3", "asymp",
    "ratio", "gompertz", "weibull4", "betap", "heleg", "linear",
]


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_matches_r(model_name, galap_fits, r_reference):
    """Validate each model against R sars reference values."""
    fit = galap_fits[model_name]
    ref = r_reference.loc[model_name]

    if pd.isna(ref["r2"]):
        pytest.skip(f"R produced NA for {model_name}")

    assert fit.converged is True, f"{model_name} did not converge"
    assert fit.r_squared >= ref["r2"] - 0.005, (
        f"{model_name} R² worse than R: {fit.r_squared} vs {ref['r2']}"
    )
    assert fit.aicc <= ref["aicc"] + 0.5, (
        f"{model_name} AICc worse than R: {fit.aicc} vs {ref['aicc']}"
    )


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_converges(model_name, galap_fits):
    """Every model should converge on the galap dataset."""
    fit = galap_fits[model_name]
    assert fit.converged is True, f"{model_name} did not converge"
    assert 0.0 < fit.r_squared <= 1.0, (
        f"{model_name} R² out of range: {fit.r_squared}"
    )


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_predict(model_name, galap, galap_fits):
    """Predict should return values for all models."""
    fit = galap_fits[model_name]
    if not fit.converged:
        pytest.skip(f"{model_name} did not converge")
    pred = fit.predict(galap["area"].values)
    assert len(pred) == len(galap)
    assert all(pd.notna(pred)), f"{model_name} predict returned NaN"
