"""Tests for multi-model inference, averaging, and bootstrap CI."""

import numpy as np
import pytest

import sars

# ---------------------------------------------------------------------------
# aicc and akaike_weights
# ---------------------------------------------------------------------------

class TestAicc:
    def test_basic(self):
        """AICc should return a finite value for reasonable inputs."""
        val = sars.aicc(k=3, n=16, rss=10000.0)
        assert np.isfinite(val)

    def test_matches_internal(self, galap):
        """aicc() should agree with the value computed during model fitting."""
        fit = sars.sar_power(galap)
        # power has 2 params + sigma = k=3
        rss = np.sum((galap["species"].values - fit.predict(galap["area"].values)) ** 2)
        expected = sars.aicc(k=3, n=fit.n, rss=rss)
        assert expected == pytest.approx(fit.aicc, abs=0.01)

    def test_small_sample_inf(self):
        """AICc should be inf when n - k - 1 <= 0."""
        val = sars.aicc(k=5, n=5, rss=100.0)
        assert val == np.inf


class TestAkaikeWeights:
    def test_sum_to_one(self):
        vals = np.array([180.0, 182.0, 185.0, 190.0])
        w = sars.akaike_weights(vals)
        assert w.sum() == pytest.approx(1.0)

    def test_best_model_highest(self):
        vals = np.array([180.0, 182.0, 185.0, 190.0])
        w = sars.akaike_weights(vals)
        assert w[0] > w[1] > w[2] > w[3]

    def test_equal_aicc_equal_weights(self):
        vals = np.array([100.0, 100.0, 100.0])
        w = sars.akaike_weights(vals)
        assert w[0] == pytest.approx(1 / 3)
        assert w[1] == pytest.approx(1 / 3)

    def test_nan_gets_zero_weight(self):
        vals = np.array([180.0, np.nan, 185.0])
        w = sars.akaike_weights(vals)
        assert w[1] == 0.0
        assert w.sum() == pytest.approx(1.0)

    def test_all_nan(self):
        vals = np.array([np.nan, np.nan])
        w = sars.akaike_weights(vals)
        assert np.all(w == 0.0)


# ---------------------------------------------------------------------------
# sar_multi
# ---------------------------------------------------------------------------

class TestSarMulti:
    def test_all_models(self, galap_multi):
        """sar_multi with all models should produce a summary table."""
        assert isinstance(galap_multi, sars.MultiSARFit)
        assert len(galap_multi.fits) > 0
        assert "model" in galap_multi.summary.columns
        assert "weight" in galap_multi.summary.columns
        assert "delta_AICc" in galap_multi.summary.columns

    def test_summary_sorted_by_aicc(self, galap_multi):
        aicc_vals = galap_multi.summary["AICc"].values
        assert np.all(aicc_vals[:-1] <= aicc_vals[1:])

    def test_weights_sum_to_one(self, galap_multi):
        assert galap_multi.summary["weight"].sum() == pytest.approx(1.0)

    def test_delta_aicc_min_is_zero(self, galap_multi):
        assert galap_multi.summary["delta_AICc"].min() == pytest.approx(0.0)

    def test_subset_models(self, galap):
        result = sars.sar_multi(galap, models=["power", "loga", "linear"])
        assert len(result.fits) == 3
        assert set(result.summary["model"]) == {"power", "loga", "linear"}

    def test_unknown_model_raises(self, galap):
        with pytest.raises(ValueError, match="Unknown model"):
            sars.sar_multi(galap, models=["power", "nonexistent"])

    def test_shape_column(self, galap_multi):
        shapes = set(galap_multi.summary["shape"])
        assert shapes <= {"convex", "sigmoid", "linear"}

    def test_asymptote_column(self, galap_multi):
        """Asymptotic models should have finite asymptote values."""
        for _, row in galap_multi.summary.iterrows():
            if row["shape"] == "sigmoid" or row["model"] in {
                "monod", "negexpo", "asymp", "ratio", "koba"
            }:
                if row["model"] != "koba":
                    assert row["asymptote"] is not None

    def test_all_20_converge(self, galap_multi):
        """All 20 models should converge on galap."""
        assert len(galap_multi.fits) == 20


# ---------------------------------------------------------------------------
# sar_average
# ---------------------------------------------------------------------------

class TestSarAverage:
    def test_basic(self, galap_average):
        assert isinstance(galap_average, sars.AveragedSAR)
        assert galap_average.ic == "AICc"
        assert len(galap_average.weights) > 0

    def test_predict(self, galap, galap_average):
        pred = galap_average.predict(galap["area"].values)
        assert len(pred) == len(galap)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)

    def test_predict_scalar(self, galap_average):
        pred = galap_average.predict(10.0)
        assert np.isfinite(pred)
        assert pred > 0

    def test_weights_sum_to_one(self, galap_average):
        assert sum(galap_average.weights.values()) == pytest.approx(1.0)

    def test_subset_models(self, galap):
        avg = sars.sar_average(galap, models=["power", "negexpo"])
        assert len(avg.weights) == 2

    def test_unsupported_ic(self, galap):
        with pytest.raises(ValueError, match="Unsupported IC"):
            sars.sar_average(galap, ic="WAIC")

    def test_ic_aic(self, galap):
        avg = sars.sar_average(galap, ic="AIC")
        assert avg.ic == "AIC"
        assert len(avg.weights) > 0
        assert sum(avg.weights.values()) == pytest.approx(1.0)

    def test_ic_bic(self, galap):
        avg = sars.sar_average(galap, ic="BIC")
        assert avg.ic == "BIC"
        assert len(avg.weights) > 0
        assert sum(avg.weights.values()) == pytest.approx(1.0)

    def test_ic_weights_differ(self, galap):
        """Different ICs should generally produce different weight vectors."""
        avg_aicc = sars.sar_average(galap, ic="AICc")
        avg_bic = sars.sar_average(galap, ic="BIC")
        w_aicc = [avg_aicc.weights.get(m, 0) for m in sorted(avg_aicc.weights)]
        w_bic = [avg_bic.weights.get(m, 0) for m in sorted(avg_bic.weights)]
        assert w_aicc != w_bic


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_basic(self, galap):
        """Bootstrap with small n_boot should complete and return CI."""
        ci = sars.bootstrap_ci(
            galap, models=["power", "loga"], n_boot=10,
            rng=np.random.default_rng(42),
        )
        assert isinstance(ci, sars.BootstrappedCI)
        assert ci.n_boot > 0
        assert len(ci.area_grid) == 100
        assert len(ci.mean) == 100
        assert len(ci.lower) == 100
        assert len(ci.upper) == 100

    def test_ci_ordering(self, galap):
        """Lower bound should be <= mean <= upper bound."""
        ci = sars.bootstrap_ci(
            galap, models=["power", "loga"], n_boot=10,
            rng=np.random.default_rng(42),
        )
        assert np.all(ci.lower <= ci.mean + 1e-10)
        assert np.all(ci.mean <= ci.upper + 1e-10)

    def test_custom_area_grid(self, galap):
        grid = np.array([1.0, 5.0, 10.0, 50.0])
        ci = sars.bootstrap_ci(
            galap, models=["power"], n_boot=5,
            area_grid=grid, rng=np.random.default_rng(42),
        )
        assert len(ci.area_grid) == 4

    def test_conf_level(self, galap):
        ci = sars.bootstrap_ci(
            galap, models=["power"], n_boot=10,
            conf=0.90, rng=np.random.default_rng(42),
        )
        assert ci.conf == 0.90

    def test_method_full(self, galap):
        """method='full' uses complete grid search per resample."""
        ci = sars.bootstrap_ci(
            galap, models=["power", "loga"], n_boot=3,
            rng=np.random.default_rng(42), method="full",
        )
        assert isinstance(ci, sars.BootstrappedCI)
        assert ci.n_boot > 0

    def test_invalid_method(self, galap):
        with pytest.raises(ValueError, match="method must be"):
            sars.bootstrap_ci(galap, n_boot=1, method="invalid")

    def test_convergence_diagnostics(self, galap):
        """BootstrappedCI should expose per-replicate convergence counts."""
        ci = sars.bootstrap_ci(
            galap, models=["power", "loga"], n_boot=5,
            rng=np.random.default_rng(42),
        )
        assert len(ci.convergence_counts) == ci.n_boot
        assert ci.n_models_attempted == 2
        assert all(0 < c <= 2 for c in ci.convergence_counts)

    def test_convergence_diagnostics_fast(self, galap):
        """BootstrappedCI should expose convergence counts in fast mode."""
        ci = sars.bootstrap_ci(
            galap, models=["power", "loga"], n_boot=5,
            rng=np.random.default_rng(42), method="fast",
        )
        assert len(ci.convergence_counts) == ci.n_boot
        assert ci.n_models_attempted == 2
        assert all(0 < c <= 2 for c in ci.convergence_counts)
