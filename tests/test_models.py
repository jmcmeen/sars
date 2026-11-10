import numpy as np
import pandas as pd
import pytest

import sars  # noqa: I001

# ---------------------------------------------------------------------------
# Input validation (applies to all sar_* functions via _fit_nls)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Validation errors should be clear, not buried inside scipy."""

    def test_not_a_dataframe(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            sars.sar_power({"area": [1, 2], "species": [3, 4]})

    def test_missing_area_column(self):
        df = pd.DataFrame({"A": [1, 2], "species": [3, 4]})
        with pytest.raises(KeyError, match="area"):
            sars.sar_power(df)

    def test_missing_species_column(self):
        df = pd.DataFrame({"area": [1, 2], "S": [3, 4]})
        with pytest.raises(KeyError, match="species"):
            sars.sar_power(df)

    def test_missing_both_columns(self):
        df = pd.DataFrame({"A": [1, 2], "S": [3, 4]})
        with pytest.raises(KeyError, match="area.*species"):
            sars.sar_power(df)

    def test_too_few_observations(self):
        df = pd.DataFrame({"area": [1.0], "species": [5]})
        with pytest.raises(ValueError, match="at least 2"):
            sars.sar_power(df)

    def test_nan_in_area(self):
        df = pd.DataFrame({"area": [1.0, np.nan, 3.0], "species": [5, 10, 15]})
        with pytest.raises(ValueError, match="area.*NaN"):
            sars.sar_power(df)

    def test_nan_in_species(self):
        df = pd.DataFrame({"area": [1.0, 2.0, 3.0], "species": [5, np.nan, 15]})
        with pytest.raises(ValueError, match="species.*NaN"):
            sars.sar_power(df)

    def test_inf_in_area(self):
        df = pd.DataFrame({"area": [1.0, np.inf], "species": [5, 10]})
        with pytest.raises(ValueError, match="area.*infinite"):
            sars.sar_power(df)

    def test_zero_area(self):
        df = pd.DataFrame({"area": [0.0, 1.0, 2.0], "species": [5, 10, 15]})
        with pytest.raises(ValueError, match="area.*positive"):
            sars.sar_power(df)

    def test_negative_species(self):
        df = pd.DataFrame({"area": [1.0, 2.0, 3.0], "species": [-1, 10, 15]})
        with pytest.raises(ValueError, match="species.*positive"):
            sars.sar_power(df)

    def test_validation_applies_to_all_models(self):
        """Every sar_* function should reject bad input identically."""
        bad = pd.DataFrame({"wrong": [1, 2], "columns": [3, 4]})
        for name in dir(sars):
            if name.startswith("sar_") and name != "sar_threshold":
                fn = getattr(sars, name)
                with pytest.raises(KeyError):
                    fn(bad)


def test_power_law_spot_check():
    """Quick sanity check using confirmed R values (no R script needed)."""
    partial = pd.DataFrame(
        {
            "area": [0.20, 0.90, 1.00, 1.80, 1.87, 4.40, 7.10, 7.50, 18.00, 20.00],
            "species": [48, 7, 52, 14, 42, 22, 103, 48, 79, 119],
        }
    )
    fit = sars.sar_power(partial)
    assert fit.converged is True
    assert 0.0 < fit.params["z"] < 1.0
    assert fit.params["c"] > 0
    assert 0.0 < fit.r_squared <= 1.0


# ---------------------------------------------------------------------------
# Every model on the galap data (uses cached galap_fits); agreement with R is
# in test_r_reference.py
# ---------------------------------------------------------------------------

ALL_MODELS = [
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
    "weibull4",
    "betap",
    "heleg",
    "linear",
]


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_converges(model_name, galap_fits):
    """Every model should converge on the galap dataset."""
    fit = galap_fits[model_name]
    assert fit.converged is True, f"{model_name} did not converge"
    assert 0.0 < fit.r_squared <= 1.0, f"{model_name} R² out of range: {fit.r_squared}"


def test_fitting_is_quiet(galap):
    """The multi-start search visits parameters where a model overflows; those are
    penalised, not worth a numpy warning each (#82). epm2 is the worst offender."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sars.sar_epm2(galap)
    assert fit.converged
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_predict(model_name, galap, galap_fits):
    """Predict should return values for all models."""
    fit = galap_fits[model_name]
    if not fit.converged:
        pytest.skip(f"{model_name} did not converge")
    pred = fit.predict(galap["area"].values)
    assert len(pred) == len(galap)
    assert all(pd.notna(pred)), f"{model_name} predict returned NaN"
