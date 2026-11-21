"""Tests for multi-model inference, averaging, and bootstrap CI."""

import numpy as np
import pandas as pd
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

    def test_all_non_finite_is_undefined(self):
        """No finite IC means no weights, not zero weights (issue #50)."""
        for vals in (np.array([np.nan, np.nan]), np.array([np.inf, np.inf])):
            w = sars.akaike_weights(vals)
            assert w.shape == vals.shape
            assert np.all(np.isnan(w))


# Four islands: AICc is infinite for every model (n - k - 1 <= 0), while AIC
# and BIC are finite. Used to check that undefined weights fail loudly.
@pytest.fixture
def tiny():
    import pandas as pd

    return pd.DataFrame({"area": [1.0, 4.0, 16.0, 64.0], "species": [5.0, 9.0, 15.0, 24.0]})


TINY_MODELS = ["power", "loga", "koba"]


class TestUndefinedWeights:
    def test_summary_shows_nan_weights(self, tiny):
        multi = sars.sar_multi(tiny, models=TINY_MODELS)
        assert np.all(np.isinf(multi.summary["AICc"]))
        assert multi.summary["weight"].isna().all()

    def test_sar_average_raises(self, tiny):
        with pytest.raises(ValueError, match="no model has a finite AICc"):
            sars.sar_average(tiny, models=TINY_MODELS)

    def test_sar_average_works_with_finite_ic(self, tiny):
        avg = sars.sar_average(tiny, models=TINY_MODELS, ic="AIC")
        assert sum(avg.weights.values()) == pytest.approx(1.0)
        pred = avg.predict(np.array([2.0, 8.0, 32.0]))
        assert np.all(pred > 0)

    def test_bootstrap_ci_reports_nan_not_zero(self, tiny):
        ci = sars.bootstrap_ci(tiny, models=TINY_MODELS, n_boot=5, rng=np.random.default_rng(0))
        assert np.all(np.isnan(ci.mean))
        assert np.all(np.isnan(ci.median))
        assert np.all(np.isnan(ci.lower))
        assert np.all(np.isnan(ci.upper))


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
        """Each model's general shape: Matthews et al. (2019, Table 1), as corrected
        in R sars."""
        expected = {
            "power": "convex", "powerR": "convex", "loga": "convex", "koba": "convex",
            "linear": "linear", "epm1": "convex/sigmoid", "epm2": "sigmoid",
            "p1": "convex/sigmoid", "p2": "sigmoid", "monod": "convex",
            "negexpo": "convex", "ratio": "convex", "asymp": "convex/sigmoid",
            "mmf": "convex/sigmoid", "heleg": "convex/sigmoid", "gompertz": "sigmoid",
            "weibull3": "sigmoid", "weibull4": "sigmoid", "chapman": "sigmoid",
            "betap": "sigmoid",
        }  # fmt: skip
        shapes = galap_multi.summary.set_index("model")["shape"].to_dict()
        assert shapes == {m: expected[m] for m in shapes}

    def test_observed_shape_column(self, galap_multi):
        summary = galap_multi.summary
        assert list(summary.columns)[-1] == "observed_shape"  # appended: positions kept
        observed = summary.set_index("model")["observed_shape"].to_dict()
        # Convex models stay convex; a sigmoid model's fit is sigmoid only when its
        # inflection point is among the islands; fits that turn down have no shape.
        expected = {
            "power": "convex", "powerR": "convex", "loga": "convex", "koba": "convex",
            "monod": "convex", "negexpo": "convex", "ratio": "convex", "linear": "linear",
            "asymp": "convex", "mmf": "convex", "heleg": "convex", "chapman": "convex",
            "weibull3": "convex", "betap": "convex", "gompertz": "sigmoid", "p2": "sigmoid",
            "p1": "none", "epm1": "none", "epm2": "none",
        }  # fmt: skip
        for model, shape in expected.items():
            assert observed[model] == shape, model
        # weibull4 sits on a flat ridge, where its curve's shape isn't pinned down.
        assert observed["weibull4"] in {"convex", "sigmoid", "none"}

    def test_bootstrap_resamples_skip_the_shape(self, galap_fits):
        from sars._inference import _build_summary

        summary = _build_summary([galap_fits["power"]], shapes=False)
        assert summary["observed_shape"].isna().all()

    def test_observed_shape_of_an_empty_summary(self):
        from sars._inference import _build_summary

        assert "observed_shape" in _build_summary([]).columns

    def test_asymptote_column(self, galap_multi):
        """Models with an asymptote report it; the others (koba among them) don't."""
        asymptotic = {"monod", "negexpo", "ratio", "asymp", "mmf", "heleg", "gompertz",
                      "weibull3", "weibull4", "chapman", "betap"}  # fmt: skip
        for _, row in galap_multi.summary.iterrows():
            has_one = not pd.isna(row["asymptote"])
            assert has_one is (row["model"] in asymptotic), row["model"]

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


class TestAverageOfNoModels:
    def test_sar_average_raises_when_no_model_converges(self, galap, monkeypatch):
        """It used to return an average with no weights, predicting zero
        species everywhere."""
        from sars import _inference

        failed = _inference._fit_nls("power", galap)
        failed.converged = False
        monkeypatch.setattr(_inference, "_fit_nls", lambda name, data, **kw: failed)
        with pytest.raises(ValueError, match="nothing to average: no model converged"):
            sars.sar_average(galap, models=["power", "loga"])

    def test_sar_average_of_no_models_raises(self, galap):
        with pytest.raises(ValueError, match="nothing to average"):
            sars.sar_average(galap, models=[])
        with pytest.raises(ValueError, match="No models given"):
            sars.bootstrap_ci(galap, models=[], method="full", n_boot=3)

    def test_weights_on_none_of_the_fitted_models_are_refused(self, galap):
        multi = sars.sar_multi(galap, models=["power"])
        with pytest.raises(ValueError, match="no weight to any of the fitted models"):
            sars.AveragedSAR(multi=multi, ic="AICc", weights={"loga": 1.0})

    def test_a_full_bootstrap_counts_resamples_where_no_model_converged(self, galap, monkeypatch):
        from sars import _inference

        real = _inference.sar_multi
        calls = {"n": 0}

        def multi(data, models="all"):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                return _inference.MultiSARFit(
                    fits=[], data=data, summary=_inference._build_summary([])
                )
            return real(data, models=models)

        monkeypatch.setattr(_inference, "sar_multi", multi)
        with pytest.warns(UserWarning, match="dropped"):
            ci = sars.bootstrap_ci(
                galap, models=["power"], method="full", n_boot=6, rng=np.random.default_rng(0)
            )
        assert ci.dropped == {"no model converged": 3}
        assert ci.n_boot == 3


class TestBootstrapCI:
    def test_median_resists_runaway_resamples_and_the_mean_warns(self, galap, monkeypatch):
        """bootstrap_ci calls AveragedSAR.predict once per resample."""
        real = sars.AveragedSAR.predict
        calls = {"n": 0}

        def predict(self, area):
            calls["n"] += 1
            pred = real(self, area)
            return pred * 1e9 if calls["n"] == 50 else pred  # one call per resample: one runs away

        monkeypatch.setattr(sars.AveragedSAR, "predict", predict)
        with pytest.warns(UserWarning, match="mean is outside the central 95%"):
            ci = sars.bootstrap_ci(
                galap, models=["power", "loga"], n_boot=80, rng=np.random.default_rng(0)
            )
        assert np.all((ci.median >= ci.lower) & (ci.median <= ci.upper))
        assert np.any((ci.mean < ci.lower) | (ci.mean > ci.upper))

    def test_basic(self, galap):
        """Bootstrap with small n_boot should complete and return CI."""
        ci = sars.bootstrap_ci(
            galap,
            models=["power", "loga"],
            n_boot=10,
            rng=np.random.default_rng(42),
        )
        assert isinstance(ci, sars.BootstrappedCI)
        assert ci.n_boot > 0
        assert 90 <= len(ci.area_grid) <= 100
        for values in (ci.mean, ci.lower, ci.upper, ci.median):
            assert len(values) == len(ci.area_grid)
        assert ci.dropped == {}

    def test_default_grid_covers_the_small_areas(self, galap):
        """The Galápagos areas run from 0.52 to 1,008 km²: an even grid of 100
        puts five points below 50 km², where most islands are."""
        ci = sars.bootstrap_ci(galap, models=["power"], n_boot=3, rng=np.random.default_rng(0))
        area = galap["area"].to_numpy()
        assert ci.area_grid[0] == area.min() and ci.area_grid[-1] == area.max()
        assert np.all(np.diff(ci.area_grid) > 0)
        assert np.sum(ci.area_grid < 50) >= 30

    def test_dropped_resamples_are_counted_and_reported(self, galap, monkeypatch):
        """Each way a resample can fail is counted, and one warning says how many
        were dropped and why."""
        from sars import _inference

        real_fast = _inference._sar_multi_fast
        real_predict = sars.AveragedSAR.predict
        calls = {"fit": 0, "predict": 0}

        def fast(data, warm_starts):
            calls["fit"] += 1
            if calls["fit"] % 5 == 1:
                return _inference.MultiSARFit(
                    fits=[], data=data, summary=_inference._build_summary([])
                )
            if calls["fit"] % 5 == 2:
                raise RuntimeError("boom")
            return real_fast(data, warm_starts)

        def predict(self, area):
            calls["predict"] += 1
            pred = real_predict(self, area)
            return pred * np.inf if calls["predict"] % 3 == 0 else pred

        monkeypatch.setattr(_inference, "_sar_multi_fast", fast)
        monkeypatch.setattr(sars.AveragedSAR, "predict", predict)
        with pytest.warns(UserWarning, match=r"of 20 bootstrap resamples were dropped"):
            ci = sars.bootstrap_ci(galap, models=["power"], n_boot=20, rng=np.random.default_rng(0))
        assert ci.dropped["no model converged"] == 4
        assert ci.dropped["RuntimeError: boom"] == 4
        assert ci.dropped["prediction not finite"] == 4
        assert ci.n_boot + sum(ci.dropped.values()) == 20

    def test_every_resample_dropped_is_reported(self, galap, monkeypatch):
        from sars import _inference

        def fast(data, warm_starts):
            raise RuntimeError("no luck\nwith details on a second line")

        monkeypatch.setattr(_inference, "_sar_multi_fast", fast)
        with pytest.warns(
            UserWarning, match="5 of 5 bootstrap resamples were dropped .*no interval"
        ):
            ci = sars.bootstrap_ci(galap, models=["power"], n_boot=5, rng=np.random.default_rng(0))
        assert ci.n_boot == 0 and np.isnan(ci.upper).all()
        assert ci.dropped == {"RuntimeError: no luck": 5}  # one key per kind of failure

    def test_a_list_grid_works_where_the_bound_runs_away(self, galap, monkeypatch):
        real = sars.AveragedSAR.predict
        calls = {"n": 0}

        def predict(self, area):
            calls["n"] += 1
            pred = real(self, area)
            return pred * 1e6 if calls["n"] % 5 == 0 else pred

        monkeypatch.setattr(sars.AveragedSAR, "predict", predict)
        with pytest.warns(UserWarning, match="upper bound exceeds"):
            ci = sars.bootstrap_ci(
                galap,
                models=["power"],
                n_boot=20,
                area_grid=[1.0, 10.0, 100.0],
                rng=np.random.default_rng(0),
            )
        assert isinstance(ci.area_grid, np.ndarray)

    def test_bad_data_gets_the_packages_own_message(self):
        data = pd.DataFrame({"area": [0.0, 1, 2, 3, 4], "species": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="'area' must contain only positive values"):
            sars.bootstrap_ci(data, models=["power"], n_boot=2)

    def test_nothing_to_resample_is_reported(self, galap, monkeypatch):
        from sars import _inference

        monkeypatch.setattr(
            _inference,
            "_sar_multi",
            lambda data, models, checks, stacklevel: _inference.MultiSARFit(
                fits=[], data=data, summary=_inference._build_summary([])
            ),
        )
        with pytest.warns(UserWarning, match="No model converged on the original data"):
            ci = sars.bootstrap_ci(galap, models=["power", "loga"], n_boot=3)
        assert ci.n_models_attempted == 2 and np.isnan(ci.upper).all()

    def test_a_prediction_that_overflows_is_dropped_without_numpys_warning(
        self, galap, monkeypatch
    ):
        import warnings

        real = sars.AveragedSAR.predict
        calls = {"n": 0}

        def predict(self, area):
            calls["n"] += 1
            pred = real(self, area)
            return pred * np.exp(1000.0) if calls["n"] == 2 else pred  # overflows

        monkeypatch.setattr(sars.AveragedSAR, "predict", predict)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            ci = sars.bootstrap_ci(galap, models=["power"], n_boot=10, rng=np.random.default_rng(0))
        assert ci.dropped == {"prediction not finite": 1}

    def test_an_upper_bound_that_runs_away_is_reported(self, galap, monkeypatch):
        """One resample in ten runs away at the smallest areas: more than the
        2.5% the upper bound sits above, so the bound runs away there."""
        real = sars.AveragedSAR.predict
        calls = {"n": 0}

        def predict(self, area):
            calls["n"] += 1
            pred = real(self, area)
            area = np.asarray(area)
            return np.where(area < 1.0, pred * 1e6, pred) if calls["n"] % 10 == 0 else pred

        monkeypatch.setattr(sars.AveragedSAR, "predict", predict)
        with pytest.warns(UserWarning, match=r"upper bound exceeds 10 times the largest observed"):
            ci = sars.bootstrap_ci(galap, models=["power"], n_boot=60, rng=np.random.default_rng(0))
        assert np.all(ci.upper[ci.area_grid >= 1.0] < 10 * galap["species"].max())

    def test_a_narrow_interval_alone_does_not_warn(self, galap):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sars.bootstrap_ci(
                galap, models=["power"], n_boot=20, conf=0.1, rng=np.random.default_rng(3)
            )

    def test_ci_ordering(self, galap):
        """Lower bound should be <= median <= upper bound."""
        ci = sars.bootstrap_ci(
            galap,
            models=["power", "loga"],
            n_boot=10,
            rng=np.random.default_rng(42),
        )
        assert np.all(ci.lower <= ci.median + 1e-10)
        assert np.all(ci.median <= ci.upper + 1e-10)

    def test_custom_area_grid(self, galap):
        grid = np.array([1.0, 5.0, 10.0, 50.0])
        ci = sars.bootstrap_ci(
            galap,
            models=["power"],
            n_boot=5,
            area_grid=grid,
            rng=np.random.default_rng(42),
        )
        assert len(ci.area_grid) == 4

    def test_conf_level(self, galap):
        ci = sars.bootstrap_ci(
            galap,
            models=["power"],
            n_boot=10,
            conf=0.90,
            rng=np.random.default_rng(42),
        )
        assert ci.conf == 0.90

    def test_method_full(self, galap):
        """method='full' uses complete grid search per resample."""
        ci = sars.bootstrap_ci(
            galap,
            models=["power", "loga"],
            n_boot=3,
            rng=np.random.default_rng(42),
            method="full",
        )
        assert isinstance(ci, sars.BootstrappedCI)
        assert ci.n_boot > 0

    def test_invalid_method(self, galap):
        with pytest.raises(ValueError, match="method must be"):
            sars.bootstrap_ci(galap, n_boot=1, method="invalid")

    def test_convergence_diagnostics(self, galap):
        """BootstrappedCI should expose per-replicate convergence counts."""
        ci = sars.bootstrap_ci(
            galap,
            models=["power", "loga"],
            n_boot=5,
            rng=np.random.default_rng(42),
        )
        assert len(ci.convergence_counts) == ci.n_boot
        assert ci.n_models_attempted == 2
        assert all(0 < c <= 2 for c in ci.convergence_counts)

    def test_convergence_diagnostics_fast(self, galap):
        """BootstrappedCI should expose convergence counts in fast mode."""
        ci = sars.bootstrap_ci(
            galap,
            models=["power", "loga"],
            n_boot=5,
            rng=np.random.default_rng(42),
            method="fast",
        )
        assert len(ci.convergence_counts) == ci.n_boot
        assert ci.n_models_attempted == 2
        assert all(0 < c <= 2 for c in ci.convergence_counts)


class TestMultiInputValidation:
    """sar_multi must surface input errors instead of returning zero fits."""

    def test_too_few_rows_raises(self):
        df = pd.DataFrame({"area": [1.0], "species": [5]})
        with pytest.raises(ValueError, match="at least 2"):
            sars.sar_multi(df)

    def test_nan_raises(self):
        df = pd.DataFrame({"area": [1.0, 2.0, np.nan], "species": [5, 6, 7]})
        with pytest.raises(ValueError, match="area.*NaN"):
            sars.sar_multi(df)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "species": [5, 6, 7]})
        with pytest.raises(KeyError, match="area"):
            sars.sar_multi(df)

    def test_average_inherits_validation(self):
        df = pd.DataFrame({"area": [1.0], "species": [5]})
        with pytest.raises(ValueError, match="at least 2"):
            sars.sar_average(df)
