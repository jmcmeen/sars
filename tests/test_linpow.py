"""Tests for the log-log power model (``lin_pow``).

The R numbers are in ``tests/r_reference/lin_pow_galap.csv``, written by
``generate_r_reference.R``: R ``sars``'s ``lin_pow`` on the Galápagos data and on
the same data with its smallest island emptied, for each log base.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars
from sars._models import _failed_fit, _fit_nls

REF = Path(__file__).parent / "r_reference"
R_LIN_POW = pd.read_csv(REF / "lin_pow_galap.csv", dtype={"log_base": str})

# The reference table holds 12 significant figures.
RTOL = 1e-9


@pytest.fixture(scope="module")
def galap_zero(galap):
    """galap with no species on its smallest island (the first row)."""
    data = galap.copy()
    assert data.loc[0, "area"] == data["area"].min()
    data.loc[0, "species"] = 0
    return data


@pytest.mark.parametrize("log_base", ["e", "2", "10"])
@pytest.mark.parametrize("dataset", ["galap", "galap_zero"])
def test_matches_r(dataset, log_base, galap, galap_zero):
    data = {"galap": galap, "galap_zero": galap_zero}[dataset]
    ref = R_LIN_POW[(R_LIN_POW["data"] == dataset) & (R_LIN_POW["log_base"] == log_base)]
    assert len(ref) == 1
    r = ref.iloc[0]

    fit = sars.lin_pow(data, log_base=log_base)

    assert fit.log_base == log_base
    assert fit.n == len(data)
    assert fit.con == (1.0 if dataset == "galap_zero" else 0.0)
    for attr, col in [
        ("logc", "logc"),
        ("z", "z"),
        ("logc_se", "logc_se"),
        ("z_se", "z_se"),
        ("logc_t", "logc_t"),
        ("z_t", "z_t"),
        ("logc_p", "logc_p"),
        ("z_p", "z_p"),
        ("r_squared", "r2"),
        ("adj_r_squared", "adj_r2"),
        ("sigma", "sigma"),
    ]:
        assert getattr(fit, attr) == pytest.approx(r[col], rel=RTOL), attr


@pytest.mark.parametrize("dataset", ["galap", "galap_zero"])
def test_compare_matches_r(dataset, galap, galap_zero):
    """compare=True fits the nonlinear power model to the richness as given,
    zeros included, as R does."""
    data = {"galap": galap, "galap_zero": galap_zero}[dataset]
    r = R_LIN_POW[(R_LIN_POW["data"] == dataset) & (R_LIN_POW["log_base"] == "e")].iloc[0]

    fit = sars.lin_pow(data, compare=True)

    assert isinstance(fit.power, sars.SARFit)
    assert fit.power.model == "power"
    # The nonlinear optimum is found iteratively by both packages.
    assert fit.power.params["c"] == pytest.approx(r["power_c"], rel=5e-3)
    assert fit.power.params["z"] == pytest.approx(r["power_z"], rel=5e-3)

    table = fit.comparison
    assert list(table.index) == ["log-log", "nonlinear"]
    assert list(table.columns) == ["c", "z", "converged"]
    assert list(table["converged"]) == [True, True]
    assert table.loc["log-log", "c"] == fit.c
    assert table.loc["log-log", "z"] == fit.z
    assert table.loc["nonlinear", "c"] == fit.power.params["c"]
    assert table.loc["nonlinear", "z"] == fit.power.params["z"]


def test_compare_without_zeros_is_sar_power(galap):
    fit = sars.lin_pow(galap, compare=True)
    assert fit.power.params == pytest.approx(sars.sar_power(galap).params)


def test_no_comparison_by_default(galap):
    fit = sars.lin_pow(galap)
    assert fit.power is None
    assert fit.comparison is None


def test_log_base_changes_only_the_intercept(galap):
    e, two, ten = (sars.lin_pow(galap, log_base=b) for b in ("e", "2", "10"))
    for other, base in [(two, 2.0), (ten, 10.0)]:
        assert other.z == pytest.approx(e.z, rel=1e-12)
        assert other.c == pytest.approx(e.c, rel=1e-12)
        assert other.c_se == pytest.approx(e.c_se, rel=1e-12)
        assert other.r_squared == pytest.approx(e.r_squared, rel=1e-12)
        assert other.logc == pytest.approx(e.logc / np.log(base), rel=1e-12)
        np.testing.assert_allclose(other.predict([1.0, 50.0]), e.predict([1.0, 50.0]), rtol=1e-12)


def test_c_is_the_back_transformed_intercept(galap):
    assert sars.lin_pow(galap).c == pytest.approx(np.exp(sars.lin_pow(galap).logc))
    assert sars.lin_pow(galap, log_base="10").c == pytest.approx(
        10 ** sars.lin_pow(galap, log_base="10").logc
    )
    fit = sars.lin_pow(galap)
    # Delta method: se(c) = c * se(ln c).
    assert fit.c_se == pytest.approx(fit.c * fit.logc_se)


@pytest.mark.parametrize(
    ("log_base", "key"),
    [
        (2, "2"),
        (2.0, "2"),
        (np.int64(2), "2"),
        (10, "10"),
        (10.0, "10"),
        (math.e, "e"),
        (np.e, "e"),
        (np.float64(np.e), "e"),
        ("E", "e"),
        (" 10 ", "10"),
    ],
)
def test_other_spellings_of_log_base(log_base, key, galap):
    fit = sars.lin_pow(galap, log_base=log_base)
    assert fit.log_base == key
    assert fit.logc == sars.lin_pow(galap, log_base=key).logc


@pytest.mark.parametrize("log_base", ["3", "ln", "log10", 2.5, 2.7, True, None, [10]])
def test_unknown_log_base_raises(log_base, galap):
    with pytest.raises(ValueError, match="log_base"):
        sars.lin_pow(galap, log_base=log_base)


def test_con_is_ignored_without_zeros(galap):
    """The constant is only added when an island has no species."""
    default = sars.lin_pow(galap)
    other = sars.lin_pow(galap, con=5)
    assert other.con == 0.0
    assert other.logc == default.logc
    assert other.z == default.z


def test_con_is_added_to_every_island(galap, galap_zero):
    fit = sars.lin_pow(galap_zero, con=0.5)
    assert fit.con == 0.5
    shifted = galap_zero.assign(species=galap_zero["species"] + 0.5)
    plain = sars.lin_pow(shifted)
    assert fit.logc == pytest.approx(plain.logc, rel=1e-12)
    assert fit.z == pytest.approx(plain.z, rel=1e-12)
    # The data kept on the fit are the ones given.
    assert fit.data["species"].iloc[0] == 0


@pytest.mark.parametrize("con", [0, -1, np.nan, np.inf])
def test_zero_species_needs_a_positive_con(con, galap_zero):
    with pytest.raises(ValueError, match="con"):
        sars.lin_pow(galap_zero, con=con)


@pytest.mark.parametrize("con", ["x", None, True, [1.0], np.nan, np.inf])
def test_con_must_be_a_finite_number(con, galap, galap_zero):
    """Checked whether or not the data have a zero for it to be added to."""
    for data in (galap, galap_zero):
        with pytest.raises(ValueError, match="'con' must be a finite number"):
            sars.lin_pow(data, con=con)


def test_con_may_be_a_numpy_number(galap_zero):
    assert sars.lin_pow(galap_zero, con=np.float64(0.5)).con == 0.5
    assert sars.lin_pow(galap_zero, con=np.int64(2)).con == 2.0


def test_predict_is_on_the_arithmetic_scale(galap):
    fit = sars.lin_pow(galap, log_base="10")
    area = np.array([1.0, 50.0, 10000.0])
    np.testing.assert_allclose(fit.predict(area), fit.c * area**fit.z)
    assert fit.predict(1.0) == pytest.approx(fit.c)
    assert fit.predict(50.0).shape == ()


def test_predict_takes_the_constant_off(galap_zero):
    """With a constant added, the line is of S + con, so predicted richness is
    what the line gives less the constant."""
    fit = sars.lin_pow(galap_zero)
    area = np.array([1.0, 50.0])
    np.testing.assert_allclose(fit.predict(area), fit.c * area**fit.z - 1.0)


def test_exact_power_law_is_recovered():
    area = np.array([1.0, 10.0, 100.0, 1000.0])
    data = pd.DataFrame({"area": area, "species": 3.0 * area**0.25})
    fit = sars.lin_pow(data)
    assert fit.c == pytest.approx(3.0)
    assert fit.z == pytest.approx(0.25)
    assert fit.r_squared == pytest.approx(1.0)


def test_exact_fit_has_no_error_to_test_against():
    """Rounding leaves a residual near 1e-16; it isn't an error estimate."""
    area = np.array([0.3, 2.0, 17.0, 450.0, 9000.0])
    data = pd.DataFrame({"area": area, "species": 3.0 * area**0.25})
    for log_base in ("e", "2", "10"):
        fit = sars.lin_pow(data, log_base=log_base)
        assert fit.sigma == 0.0
        assert fit.z_se == 0.0
        assert fit.logc_se == 0.0
        assert fit.c_se == 0.0
        assert fit.r_squared == 1.0
        assert fit.adj_r_squared == 1.0
        assert (fit.z_t, fit.z_p) == (np.inf, 0.0)
        assert (fit.logc_t, fit.logc_p) == (np.inf, 0.0)


def test_exact_fit_signs_and_a_zero_intercept():
    area = np.array([0.3, 2.0, 17.0, 450.0, 9000.0])
    # c = 1, so log c is 0: there is nothing to test it against.
    fit = sars.lin_pow(pd.DataFrame({"area": area, "species": area**0.5}))
    assert fit.logc == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(fit.logc_t)
    assert np.isnan(fit.logc_p)
    assert (fit.z_t, fit.z_p) == (np.inf, 0.0)
    # A falling line with c below 1: both t values are negative.
    fit = sars.lin_pow(pd.DataFrame({"area": area, "species": 0.5 * area**-0.2}))
    assert (fit.z_t, fit.z_p) == (-np.inf, 0.0)
    assert (fit.logc_t, fit.logc_p) == (-np.inf, 0.0)


@pytest.mark.parametrize("species", [5, 7, 1])
@pytest.mark.parametrize("log_base", ["e", "2", "10"])
def test_equal_richness_has_no_r_squared(species, log_base):
    """With five 7s, rounding in the mean leaves a total sum of squares that is
    tiny but not zero."""
    data = pd.DataFrame({"area": [1.0, 3.0, 10.0, 40.0, 100.0], "species": [species] * 5})
    fit = sars.lin_pow(data, log_base=log_base)
    assert fit.z == 0.0
    assert fit.c == pytest.approx(species)
    assert np.isnan(fit.r_squared)
    assert np.isnan(fit.adj_r_squared)
    assert fit.sigma == 0.0
    assert fit.z_se == 0.0
    # A slope of exactly zero can't be tested; nor can log 1 = 0.
    assert np.isnan(fit.z_t)
    assert np.isnan(fit.z_p)
    if species == 1:
        assert np.isnan(fit.logc_t)
        assert np.isnan(fit.logc_p)
    else:
        assert (fit.logc_t, fit.logc_p) == (np.inf, 0.0)
    np.testing.assert_allclose(fit.predict([1.0, 500.0]), species)


def test_huge_intercept_gives_infinite_c_and_finite_predictions():
    """A steep line over tiny areas puts c beyond the largest float, but the
    line itself is fine where the data are."""
    data = pd.DataFrame({"area": [1e-6, 1.01e-6, 1.02e-6, 1.03e-6], "species": [1, 40, 2, 9000]})
    fit = sars.lin_pow(data)
    assert fit.c == np.inf
    assert np.isfinite(fit.logc)
    assert np.isfinite(fit.logc_se)
    pred = fit.predict(data["area"].to_numpy())
    assert np.isfinite(pred).all()
    assert (pred > 0).all()
    area = data["area"].to_numpy()
    np.testing.assert_allclose(np.log(pred), fit.logc + fit.z * np.log(area))
    assert "c=inf" in repr(fit)


def test_tiny_intercept_gives_finite_predictions():
    """The mirror image: c underflows to 0 over huge areas."""
    data = pd.DataFrame({"area": [1e6, 1.01e6, 1.02e6, 1.03e6], "species": [1, 40, 2, 9000]})
    fit = sars.lin_pow(data)
    assert fit.c == 0.0
    pred = fit.predict(data["area"].to_numpy())
    assert np.isfinite(pred).all()
    assert (pred > 0).all()


def test_all_zero_richness_raises():
    data = pd.DataFrame({"area": [1.0, 10.0, 100.0], "species": [0, 0, 0]})
    for compare in (False, True):
        with pytest.raises(ValueError, match="No island has any species"):
            sars.lin_pow(data, compare=compare)


def test_nls_on_all_zero_richness_is_a_failed_fit():
    """The fitting loop admits zeros for lin_pow's comparison; all zeros leave
    nothing to rescale by, and must not come back as a converged fit."""
    data = pd.DataFrame({"area": [1.0, 10.0, 100.0], "species": [0, 0, 0]})
    fit = _fit_nls("power", data, allow_zero_species=True)
    assert fit.converged is False
    assert all(np.isnan(v) for v in fit.params.values())


def test_unconverged_comparison_warns_and_says_so(galap, monkeypatch):
    def failed(name, data, **kwargs):
        return _failed_fit(name, ["c", "z"], len(data), data)

    monkeypatch.setattr("sars._linpow._fit_nls", failed)
    with pytest.warns(UserWarning, match="did not converge"):
        fit = sars.lin_pow(galap, compare=True)
    assert fit.power.converged is False
    assert "nonlinear: not converged" in repr(fit)
    table = fit.comparison
    assert list(table["converged"]) == [True, False]
    assert np.isnan(table.loc["nonlinear", "c"])
    assert table.loc["log-log", "z"] == fit.z


def test_converged_comparison_is_quiet(galap):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sars.lin_pow(galap, compare=True)


def test_too_few_observations_raises():
    data = pd.DataFrame({"area": [1.0, 10.0], "species": [5, 9]})
    with pytest.raises(ValueError, match="at least 3"):
        sars.lin_pow(data)


def test_equal_areas_raise():
    data = pd.DataFrame({"area": [2.0, 2.0, 2.0], "species": [5, 9, 7]})
    with pytest.raises(ValueError, match="different areas"):
        sars.lin_pow(data)


def test_negative_species_raise(galap):
    data = galap.copy()
    data.loc[0, "species"] = -1
    with pytest.raises(ValueError, match="negative"):
        sars.lin_pow(data)


def test_bad_input_raises(galap):
    with pytest.raises(TypeError):
        sars.lin_pow([1, 2, 3])
    with pytest.raises(KeyError):
        sars.lin_pow(galap.rename(columns={"area": "a"}))
    with pytest.raises(ValueError, match="area"):
        sars.lin_pow(galap.assign(area=0.0))


def test_other_models_still_reject_zero_species(galap_zero):
    with pytest.raises(ValueError, match="positive"):
        sars.sar_power(galap_zero)
    with pytest.raises(ValueError, match="positive"):
        sars.sar_multi(galap_zero, models=["power", "loga"])


def test_repr(galap, galap_zero):
    text = repr(sars.lin_pow(galap))
    assert text.startswith("LinPowFit(logc=3.0186, c=20.4636, z=0.3385, R²=0.5298")
    assert "log_base='e'" in text
    assert "con=" not in text
    assert "nonlinear" not in text
    assert "area" not in text  # the data aren't printed

    text = repr(sars.lin_pow(galap_zero, compare=True))
    assert "con=1" in text
    assert "nonlinear: c=" in text
