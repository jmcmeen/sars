"""Residual checks: the tests themselves, their agreement with R, and the
models ``sar_multi``, ``sar_average`` and ``bootstrap_ci`` leave out.

The R numbers are in ``tests/r_reference/residual_checks_*.csv`` (written by
``generate_r_reference.R``): what R sars stores on each fit for the
Shapiro-Wilk test and the three correlations, base R's ``ks.test`` on the
fit's residuals, and whether ``sar_average`` kept the model. They are compared
for the models whose fits match R's (``EXPECTED`` in ``test_r_reference.py``).
"""

import dataclasses
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars
from sars import _checks
from tests.test_r_reference import EXPECTED, MATCH

REF = Path(__file__).parent / "r_reference"

# On the Galápagos data each of these fails a different check, and monod
# passes all three (the same outcomes as in R).
FAILS_NORMALITY = "linear"
FAILS_HOMOSCEDASTICITY = "power"
FAILS_NEGATIVE = "loga"
PASSES = "monod"
FOUR = [FAILS_HOMOSCEDASTICITY, FAILS_NEGATIVE, FAILS_NORMALITY, PASSES]
ALL_CHECKS = {"normality": "shapiro", "homoscedasticity": "cor.fitted", "neg_check": True}


def _ref(stem: str) -> pd.DataFrame:
    return pd.read_csv(REF / f"residual_checks_{stem}.csv").set_index("model")


def _matching(stem: str) -> list[str]:
    return [m for m, outcome in EXPECTED[stem].items() if outcome == MATCH]


@pytest.fixture(scope="module")
def fits(galap_fits):
    """Fits keyed by dataset, then model. Only the seven-island models compared
    with R are fitted: the others are the slow ones on that sample."""
    galap7 = pd.read_csv(REF / "galap7.csv").rename(columns={"a": "area", "s": "species"})
    galap7_fits = sars.sar_multi(galap7, models=_matching("galap7")).fits
    return {"galap": galap_fits, "galap7": {f.model: f for f in galap7_fits}}


CASES = [(stem, model) for stem in EXPECTED for model in _matching(stem)]
IDS = [f"{stem}-{model}" for stem, model in CASES]


# ---------------------------------------------------------------------------
# Agreement with R
# ---------------------------------------------------------------------------

# Fitted values agree with R's to about four significant digits, which is what
# limits the tests computed from the residuals' values. The rank correlations
# depend only on the order of the residuals, so they agree exactly.
VALUE_RTOL = 2e-3
RANK_RTOL = 1e-8


@pytest.mark.parametrize(("stem", "model"), CASES, ids=IDS)
def test_shapiro_and_spearman_match_r(stem, model, fits):
    ref = _ref(stem).loc[model]
    checks = fits[stem][model].residual_checks()
    assert (checks.normality, checks.homoscedasticity, checks.cor_method) == (
        "shapiro",
        "cor.fitted",
        "spearman",
    )
    assert checks.normality_statistic == pytest.approx(ref["shapiro_w"], rel=VALUE_RTOL)
    assert checks.normality_pvalue == pytest.approx(ref["shapiro_p"], rel=10 * VALUE_RTOL)
    assert checks.homoscedasticity_statistic == pytest.approx(ref["fitted_spearman"], abs=1e-12)
    assert checks.homoscedasticity_pvalue == pytest.approx(ref["fitted_spearman_p"], rel=RANK_RTOL)
    assert checks.negative_fitted == ref["neg_check"]


@pytest.mark.parametrize(("stem", "model"), CASES, ids=IDS)
def test_kolmogorov_and_kendall_match_r(stem, model, fits):
    ref = _ref(stem).loc[model]
    checks = fits[stem][model].residual_checks(
        normality="kolmogorov", homoscedasticity="cor.area", cor_method="kendall"
    )
    assert checks.normality_statistic == pytest.approx(ref["ks_d"], rel=VALUE_RTOL)
    assert checks.normality_pvalue == pytest.approx(ref["ks_p"], rel=10 * VALUE_RTOL)
    assert checks.homoscedasticity_statistic == pytest.approx(ref["area_kendall"], abs=1e-12)
    assert checks.homoscedasticity_pvalue == pytest.approx(ref["area_kendall_p"], rel=RANK_RTOL)


@pytest.mark.parametrize(("stem", "model"), CASES, ids=IDS)
def test_pearson_matches_r(stem, model, fits):
    ref = _ref(stem).loc[model]
    checks = fits[stem][model].residual_checks(normality=None, cor_method="pearson")
    # A correlation near zero keeps the fitted values' absolute error, not their
    # relative one: p1 on the seven islands sits on a flat ridge next to the
    # power law, where r is 0.076 here and 0.077 in R.
    assert checks.homoscedasticity_statistic == pytest.approx(
        ref["fitted_pearson"], rel=VALUE_RTOL, abs=2e-3
    )
    assert checks.homoscedasticity_pvalue == pytest.approx(
        ref["fitted_pearson_p"], rel=10 * VALUE_RTOL, abs=2e-3
    )


@pytest.mark.parametrize("stem", ["galap", "galap7"])
def test_models_left_out_match_r(stem, fits):
    """R's sar_average, asked for the Shapiro-Wilk test, the Spearman correlation
    with the fitted values and the negative-value check, keeps and drops the same
    models (R's average has no mmf)."""
    ref = _ref(stem)
    models = [m for m in _matching(stem) if m != "mmf"]
    kept = {m for m in models if fits[stem][m].residual_checks().failure(neg_check=True) is None}
    assert kept == {m for m in models if ref.loc[m, "kept"]}
    if stem == "galap":
        assert set(models) - kept == {"power", "epm2", "loga", "linear"}


def test_r_reference_exercises_every_check():
    """The comparison above means little unless R dropped models for each reason."""
    ref = _ref("galap").loc[_matching("galap")]
    assert (ref["shapiro_p"] < 0.05).any()
    assert (ref["fitted_spearman_p"] < 0.05).any()
    assert ref["neg_check"].any()
    assert ref["kept"].sum() > 0


# ---------------------------------------------------------------------------
# The correlation p values, away from the Galápagos fits
# ---------------------------------------------------------------------------

# Base R's cor.test of 1..n with each y: Spearman p values exact up to n = 9 and
# from the Edgeworth series beyond, Kendall's exact, and both tests' large-sample
# approximations when values are tied.
COR_TESTS = pd.read_csv(REF / "correlation_tests.csv")


@pytest.mark.parametrize("row", [r for _, r in COR_TESTS.iterrows()], ids=lambda r: r["method"])
def test_correlation_tests_match_r(row):
    y = np.array(row["y"].split(), dtype=float)
    estimate, p = _checks._correlation(np.arange(1.0, len(y) + 1), y, row["method"])
    assert estimate == pytest.approx(row["estimate"], abs=1e-12)
    if row["p"] == 0:
        # R's series for Spearman's rho turns negative far in the tail and is
        # reported as 0; here the t distribution takes over.
        assert 0 < p < 1e-4
    else:
        assert p == pytest.approx(row["p"], rel=1e-8)


def test_perfect_rank_correlation():
    x = np.arange(1.0, 6.0)
    # Two of the 5! rank orders are this extreme: the identical and the reversed.
    assert _checks._correlation(x, x**2, "spearman") == pytest.approx((1.0, 2 / 120))
    assert _checks._correlation(x, -x, "spearman") == pytest.approx((-1.0, 2 / 120))
    # Ties leave only the t distribution, which has no spread left at |rho| = 1.
    tied = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    assert _checks._correlation(tied, tied, "spearman") == pytest.approx((1.0, 0.0))


def test_correlation_reference_covers_each_p_value_method():
    spearman = COR_TESTS[COR_TESTS["method"] == "spearman"]
    sizes = spearman["y"].str.split().str.len()
    tied = spearman["y"].apply(lambda y: len(set(y.split())) < len(y.split()))
    assert (sizes[~tied] <= 9).any()
    assert (sizes[~tied] > 9).any()
    assert tied.any()
    assert (spearman["estimate"] < 0).any()
    assert (spearman["p"] == 0).any()


@pytest.mark.parametrize("n", [3, 5, 8])
def test_spearman_null_distribution_counts_every_rank_order(n):
    values, ways = _checks._spearman_null(n)
    assert ways.sum() == np.prod(np.arange(1, n + 1))
    assert values.min() == 0
    assert values.max() == n * (n * n - 1) // 3
    assert ways[0] == ways[-1] == 1  # identical and reversed ranks


def test_spearman_p_value_is_symmetric_and_at_most_one():
    x = np.arange(1.0, 13.0)
    y = np.array([3, 1, 2, 6, 4, 5, 9, 7, 8, 12, 10, 11], dtype=float)
    rho, p = _checks._correlation(x, y, "spearman")
    rho_rev, p_rev = _checks._correlation(x, -y, "spearman")
    assert rho_rev == pytest.approx(-rho)
    assert p_rev == pytest.approx(p, rel=1e-9)
    # Uncorrelated ranks: twice the nearer tail would exceed 1.
    _, p_null = _checks._correlation(
        np.arange(1.0, 6.0), np.array([3.0, 5.0, 1.0, 2.0, 4.0]), "spearman"
    )
    assert 0.9 < p_null <= 1.0


# ---------------------------------------------------------------------------
# SARFit.residual_checks
# ---------------------------------------------------------------------------


def test_a_test_not_asked_for_is_nan(galap_fits):
    checks = galap_fits["power"].residual_checks(normality=None, homoscedasticity=None)
    assert checks.normality is None
    assert checks.homoscedasticity is None
    assert np.isnan(checks.normality_statistic)
    assert np.isnan(checks.normality_pvalue)
    assert np.isnan(checks.homoscedasticity_statistic)
    assert np.isnan(checks.homoscedasticity_pvalue)
    assert checks.failure(neg_check=True) is None


def test_negative_fitted_looks_at_the_observed_areas(galap_fits):
    assert galap_fits["loga"].residual_checks().negative_fitted
    assert not galap_fits["power"].residual_checks().negative_fitted


def test_checks_do_not_depend_on_units(galap):
    """Areas in hectares and richness in tens give the same test results."""
    fit = sars.sar_power(galap)
    scaled = sars.sar_power(galap.assign(area=galap["area"] * 100, species=galap["species"] * 10))
    for kwargs in (
        {},
        {"normality": "kolmogorov", "homoscedasticity": "cor.area", "cor_method": "pearson"},
    ):
        a, b = fit.residual_checks(**kwargs), scaled.residual_checks(**kwargs)
        assert a.normality_pvalue == pytest.approx(b.normality_pvalue, rel=1e-4)
        assert a.homoscedasticity_pvalue == pytest.approx(b.homoscedasticity_pvalue, rel=1e-4)


def test_kolmogorov_uses_the_residuals_mean_and_sd(galap_fits):
    """Not R sars's "kolmo", which compares with a standard normal and so rejects
    any model whose residuals are larger than a few species."""
    from scipy import stats

    fit = galap_fits["power"]
    res = fit.data["species"].to_numpy() - fit.predict(fit.data["area"].to_numpy())
    checks = fit.residual_checks(normality="kolmogorov")
    expected = stats.kstest(res, stats.norm(res.mean(), res.std(ddof=1)).cdf)
    assert checks.normality_statistic == pytest.approx(expected.statistic)
    assert checks.normality_pvalue == pytest.approx(expected.pvalue)
    assert checks.normality_pvalue > 0.05
    assert stats.kstest(res, "norm").pvalue < 0.001


def test_a_perfect_fit_has_no_test_results():
    data = pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0, 5.0], "species": [3.0, 5.0, 7.0, 9.0, 11.0]})
    checks = sars.sar_linear(data).residual_checks()
    assert np.isnan(checks.normality_pvalue)
    assert checks.failure(neg_check=True) is None


# R's ks.test for x = sin(i) + (i mod 7) / 3 against a normal with x's mean and
# standard deviation: exact below 100 untied values, asymptotic otherwise.
KS_TESTS = pd.read_csv(REF / "ks_tests.csv")


@pytest.mark.parametrize(
    "row", [r for _, r in KS_TESTS.iterrows()], ids=lambda r: f"n{r['n']}-tied{r['tied']}"
)
def test_kolmogorov_matches_r_on_either_side_of_its_exact_limit(row):
    i = np.arange(1, row["n"] + 1)
    x = np.sin(i) + (i % 7) / 3
    if row["tied"]:
        x[1] = x[0]
    d, p = _checks._normality(x, "kolmogorov")
    assert d == pytest.approx(row["d"], abs=1e-12)
    # R sums the asymptotic series to a looser tolerance than scipy.
    assert p == pytest.approx(row["p"], rel=1e-5)


def test_kolmogorov_reference_covers_exact_and_asymptotic():
    assert set(KS_TESTS.loc[KS_TESTS["exact"], "n"]) == {20, 99}
    assert not KS_TESTS.loc[KS_TESTS["tied"] | (KS_TESTS["n"] >= 100), "exact"].any()
    # The two p values differ enough at n = 100 for the test above to tell them apart.
    from scipy import stats

    i = np.arange(1, 101)
    x = np.sin(i) + (i % 7) / 3
    exact = stats.kstest((x - x.mean()) / x.std(ddof=1), "norm", method="exact").pvalue
    r_value = KS_TESTS.loc[(KS_TESTS["n"] == 100) & ~KS_TESTS["tied"], "p"].iloc[0]
    assert abs(exact - r_value) > 1e-4


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_fitted_values_fail_any_check_asked_for(galap_fits, bad):
    """Without this a NaN prediction passes every check (a NaN p value rejects
    nothing, and NaN < 0 is false), and an infinite one gets a rank correlation."""
    fit = dataclasses.replace(galap_fits["power"], params={"c": bad, "z": 0.3})
    checks = fit.residual_checks()
    assert not checks.finite_fitted
    assert not checks.negative_fitted
    for value in (
        checks.normality_statistic,
        checks.normality_pvalue,
        checks.homoscedasticity_statistic,
        checks.homoscedasticity_pvalue,
    ):
        assert np.isnan(value)
    assert checks.failure() == "fitted values not finite"
    only_negative = fit.residual_checks(normality=None, homoscedasticity=None)
    assert only_negative.failure(neg_check=True) == "fitted values not finite"
    # No check asked for, nothing to fail.
    assert only_negative.failure() is None
    assert galap_fits["power"].residual_checks().finite_fitted


def test_sar_multi_leaves_out_a_model_with_non_finite_fitted_values(galap, monkeypatch):
    import sars._inference as inference

    real_fit = inference._fit_nls

    def fit(name, data, **kwargs):
        result = real_fit(name, data, **kwargs)
        if name == "power":
            return dataclasses.replace(result, params={"c": np.nan, "z": 0.3})
        return result

    monkeypatch.setattr(inference, "_fit_nls", fit)
    with pytest.warns(UserWarning, match="1 of 2"):
        multi = sars.sar_multi(galap, models=["power", PASSES], neg_check=True)
    assert multi.excluded == {"power": "fitted values not finite"}
    assert sars.sar_multi(galap, models=["power", PASSES]).excluded == {}


def test_only_constant_input_warnings_are_silenced():
    from scipy import stats

    rng = np.random.default_rng(0)
    n = 5001
    area = np.arange(1.0, n + 1)
    fitted = np.sqrt(area)
    options = _checks.CheckOptions(normality="shapiro", homoscedasticity="cor.fitted")
    # scipy's own caution about Shapiro-Wilk p values above 5000 observations.
    with pytest.warns(UserWarning, match="5000"):
        _checks.run_checks("power", area, fitted + rng.normal(size=n), fitted, options)
    # A constant fitted curve has no correlation to test: NaN, without scipy's warning.
    flat = np.full(10, 3.0)
    species = flat + rng.normal(size=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for method in ("pearson", "spearman", "kendall"):
            checks = _checks.run_checks(
                "linear",
                area[:10],
                species,
                flat,
                _checks.CheckOptions(homoscedasticity="cor.fitted", cor_method=method),
            )
            assert np.isnan(checks.homoscedasticity_pvalue)
    with pytest.warns(stats.ConstantInputWarning):
        stats.pearsonr(species, flat)


def test_importing_sars_does_not_load_scipy_stats():
    """scipy.stats takes about a quarter of a second to import and is only
    needed once a check is asked for."""
    code = "import sys, sars; sys.exit('scipy.stats' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_check_options_know_whether_anything_was_asked_for():
    assert not _checks.CheckOptions().requested
    assert not _checks.CheckOptions(cor_method="kendall", alpha_normality=0.1).requested
    assert _checks.CheckOptions(normality="shapiro").requested
    assert _checks.CheckOptions(homoscedasticity="cor.area").requested
    assert _checks.CheckOptions(neg_check=True).requested


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"normality": "lillie"}, "Lilliefors"),
        ({"normality": "kolmo"}, "Unknown normality test"),
        ({"homoscedasticity": "fitted"}, "Unknown homoscedasticity test"),
        ({"cor_method": "rank"}, "Unknown correlation method"),
    ],
)
def test_unknown_names_raise(galap_fits, kwargs, match):
    with pytest.raises(ValueError, match=match):
        galap_fits["power"].residual_checks(**kwargs)


def test_failure_reports_the_first_check_failed(galap_fits):
    checks = galap_fits["loga"].residual_checks()
    assert checks.failure() is None
    assert checks.failure(neg_check=True) == "negative fitted values"
    # With alpha raised, loga fails the earlier checks too, and the first is named.
    assert checks.normality_pvalue < 0.9
    assert checks.failure(alpha_normality=0.9, neg_check=True).startswith("residuals not normal")
    assert checks.failure(alpha_homoscedasticity=0.99, neg_check=True).startswith(
        "residual variance not constant"
    )


# ---------------------------------------------------------------------------
# sar_multi, sar_average, bootstrap_ci
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def checked(galap):
    with pytest.warns(UserWarning, match="failed a residual check") as record:
        multi = sars.sar_multi(galap, models=FOUR, **ALL_CHECKS)
    return multi, record


def test_no_checks_by_default(galap_multi):
    assert galap_multi.excluded == {}
    assert len(galap_multi.fits) == 20


def test_sar_multi_leaves_out_failing_models(checked):
    multi, _ = checked
    assert [f.model for f in multi.fits] == [PASSES]
    assert multi.summary["model"].tolist() == [PASSES]
    assert multi.summary["weight"].sum() == pytest.approx(1.0)
    assert list(multi.excluded) == [FAILS_HOMOSCEDASTICITY, FAILS_NEGATIVE, FAILS_NORMALITY]
    assert multi.excluded[FAILS_NORMALITY].startswith("residuals not normal (shapiro p = 0.0202")
    assert multi.excluded[FAILS_HOMOSCEDASTICITY].startswith(
        "residual variance not constant (spearman correlation of squared residuals "
        "with fitted values = 0.529, p = 0.0373"
    )
    assert multi.excluded[FAILS_NEGATIVE] == "negative fitted values"


def test_sar_multi_warns_once_listing_the_models(checked):
    _, record = checked
    assert len(record) == 1
    message = str(record[0].message)
    assert message.startswith("3 of 4 fitted models failed a residual check")
    for model in (FAILS_NORMALITY, FAILS_HOMOSCEDASTICITY, FAILS_NEGATIVE):
        assert f"{model}: " in message
    assert record[0].filename == __file__


def test_each_check_is_applied_only_when_asked_for(galap):
    models = [FAILS_NORMALITY, FAILS_NEGATIVE, PASSES]
    with pytest.warns(UserWarning, match="1 of 3"):
        normality = sars.sar_multi(galap, models=models, normality="shapiro")
    assert list(normality.excluded) == [FAILS_NORMALITY]
    with pytest.warns(UserWarning, match="1 of 3"):
        negative = sars.sar_multi(galap, models=models, neg_check=True)
    assert list(negative.excluded) == [FAILS_NEGATIVE]


def test_alpha_decides_what_fails(galap):
    models = [FAILS_NORMALITY, FAILS_HOMOSCEDASTICITY]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing left out, so nothing to warn about
        strict = sars.sar_multi(
            galap,
            models=models,
            normality="shapiro",
            homoscedasticity="cor.fitted",
            alpha_normality=0.01,
            alpha_homoscedasticity=0.01,
        )
    assert strict.excluded == {}
    assert len(strict.fits) == 2


def test_cor_method_is_passed_on(galap):
    """power's squared residuals are correlated with its fitted values by rank
    (Spearman p = 0.037) but not linearly (Pearson p = 0.35)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multi = sars.sar_multi(
            galap, models=["power"], homoscedasticity="cor.fitted", cor_method="pearson"
        )
    assert multi.excluded == {}


def test_sar_average_weighs_only_the_models_that_pass(galap):
    models = [FAILS_HOMOSCEDASTICITY, PASSES, "negexpo"]
    with pytest.warns(UserWarning, match="1 of 3") as record:
        avg = sars.sar_average(galap, models=models, homoscedasticity="cor.fitted")
    assert len(record) == 1
    assert record[0].filename == __file__
    assert set(avg.weights) == {PASSES, "negexpo"}
    assert sum(avg.weights.values()) == pytest.approx(1.0)
    assert list(avg.multi.excluded) == [FAILS_HOMOSCEDASTICITY]
    unchecked = sars.sar_average(galap, models=[PASSES, "negexpo"])
    area = galap["area"].to_numpy()
    np.testing.assert_allclose(avg.predict(area), unchecked.predict(area))


def test_sar_average_raises_when_every_model_fails(galap):
    with (
        pytest.warns(UserWarning, match="2 of 2"),
        pytest.raises(ValueError, match="Every fitted model failed a residual check.*loga"),
    ):
        sars.sar_average(galap, models=[FAILS_NORMALITY, FAILS_NEGATIVE], **ALL_CHECKS)


def test_sar_multi_returns_an_empty_result_when_every_model_fails(galap):
    with pytest.warns(UserWarning, match="1 of 1"):
        multi = sars.sar_multi(galap, models=[FAILS_NEGATIVE], neg_check=True)
    assert multi.fits == []
    assert len(multi.summary) == 0
    assert list(multi.excluded) == [FAILS_NEGATIVE]


@pytest.mark.parametrize("fn", [sars.sar_multi, sars.sar_average, sars.bootstrap_ci])
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"normality": "lillie"}, "Lilliefors"),
        ({"homoscedasticity": "cor"}, "Unknown homoscedasticity test"),
        ({"cor_method": "rank"}, "Unknown correlation method"),
        ({"alpha_normality": 0.0}, "alpha_normality must be a number between 0 and 1"),
        ({"alpha_homoscedasticity": 1.5}, "alpha_homoscedasticity must be a number between"),
        # Not numbers: a ValueError like the others, not the comparison's TypeError.
        ({"alpha_normality": "0.05"}, "alpha_normality must be a number between 0 and 1"),
        ({"alpha_homoscedasticity": None}, "alpha_homoscedasticity must be a number between"),
        ({"alpha_normality": True}, "alpha_normality must be a number between 0 and 1"),
    ],
)
def test_bad_check_arguments_raise_before_fitting(galap, fn, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fn(galap, models=["power"], **kwargs)


@pytest.mark.parametrize("method", ["fast", "full"])
def test_bootstrap_fixes_the_model_set_from_the_original_fit(galap, method, monkeypatch):
    """The checks run once, on the original data; every resample then fits only
    the models that passed, and is not tested again."""
    import sars._inference as inference

    fitted: list[str] = []
    real_fit = inference._fit_nls

    def spy(name, data, **kwargs):
        fitted.append(name)
        return real_fit(name, data, **kwargs)

    tested: list[str] = []
    real_checks = sars.SARFit._run_checks

    def count(self, options):
        tested.append(self.model)
        return real_checks(self, options)

    monkeypatch.setattr(inference, "_fit_nls", spy)
    monkeypatch.setattr(sars.SARFit, "_run_checks", count)
    with pytest.warns(UserWarning, match="3 of 4") as record:
        ci = sars.bootstrap_ci(
            galap, models=FOUR, n_boot=3, rng=np.random.default_rng(0), method=method, **ALL_CHECKS
        )
    assert len([w for w in record if "residual check" in str(w.message)]) == 1
    assert ci.n_boot == 3
    assert ci.n_models_attempted == 1
    # One fit of each model on the original data, then only the survivor.
    assert fitted[:4] == FOUR
    assert set(fitted[4:]) == {PASSES}
    assert tested == FOUR


def test_full_bootstrap_keeps_a_model_that_only_failed_to_converge(galap, monkeypatch):
    """A check removes the models that failed it and no others: a model that
    didn't converge on the original data is still tried on every resample, as
    it is when no checks are asked for."""
    import sars._inference as inference

    real_fit = inference._fit_nls
    resampled: list[str] = []

    def fit(name, data, **kwargs):
        result = real_fit(name, data, **kwargs)
        if data is galap:
            return dataclasses.replace(result, converged=name != "negexpo")
        resampled.append(name)
        return result

    monkeypatch.setattr(inference, "_fit_nls", fit)
    models = [FAILS_NEGATIVE, "negexpo", PASSES]
    kwargs = {"models": models, "n_boot": 2, "method": "full"}
    with pytest.warns(UserWarning, match="1 of 2"):
        ci = sars.bootstrap_ci(galap, rng=np.random.default_rng(0), neg_check=True, **kwargs)
    assert ci.n_models_attempted == 2
    assert resampled == ["negexpo", PASSES] * 2
    # Without the check, the same model is tried too.
    resampled.clear()
    plain = sars.bootstrap_ci(galap, rng=np.random.default_rng(0), **kwargs)
    assert plain.n_models_attempted == 3
    assert resampled == models * 2


def test_bootstrap_matches_a_bootstrap_of_the_models_that_pass(galap):
    with pytest.warns(UserWarning, match="3 of 4"):
        checked_ci = sars.bootstrap_ci(
            galap, models=FOUR, n_boot=5, rng=np.random.default_rng(1), **ALL_CHECKS
        )
    plain = sars.bootstrap_ci(galap, models=[PASSES], n_boot=5, rng=np.random.default_rng(1))
    np.testing.assert_allclose(checked_ci.lower, plain.lower)
    np.testing.assert_allclose(checked_ci.upper, plain.upper)


def test_bootstrap_raises_when_every_model_fails(galap):
    with (
        pytest.warns(UserWarning, match="1 of 1"),
        pytest.raises(ValueError, match="Every fitted model failed a residual check"),
    ):
        sars.bootstrap_ci(galap, models=[FAILS_NEGATIVE], n_boot=2, neg_check=True)
