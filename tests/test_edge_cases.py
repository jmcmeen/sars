"""Tests for edge cases, repr methods, and coverage gaps."""

import numpy as np
import pandas as pd
import pytest

import sars
from sars._models import SARFit, _failed_fit


class TestSARFitRepr:
    def test_repr_format(self, galap):
        fit = sars.sar_power(galap)
        r = repr(fit)
        assert "SARFit(model='power'" in r
        assert "R²=" in r
        assert "AICc=" in r

    def test_repr_contains_params(self, galap):
        fit = sars.sar_power(galap)
        r = repr(fit)
        assert "c=" in r
        assert "z=" in r


class TestSARFitPredictUnknownModel:
    def test_predict_unknown_model_raises(self):
        fit = SARFit(
            model="nonexistent",
            params={"a": 1.0},
            r_squared=0.5,
            aic=100.0,
            aicc=101.0,
            bic=102.0,
            n=10,
            converged=True,
            data=pd.DataFrame({"area": [1.0], "species": [10]}),
        )
        with pytest.raises(NotImplementedError, match="nonexistent"):
            fit.predict(np.array([1.0, 2.0]))


class TestFailedFit:
    def test_failed_fit_returns_unconverged(self):
        data = pd.DataFrame({"area": [1.0], "species": [10]})
        fit = _failed_fit("power", ["c", "z"], 1, data)
        assert fit.converged is False
        assert np.isnan(fit.r_squared)
        assert np.isnan(fit.aic)
        assert all(np.isnan(v) for v in fit.params.values())


class TestMultiSARFitRepr:
    def test_repr(self, galap):
        result = sars.sar_multi(galap, models=["power", "loga"])
        r = repr(result)
        assert "MultiSARFit" in r
        assert "2 models" in r

    def test_empty_multi(self):
        """sar_multi with data that causes all models to fail."""
        # A single data point can't fit any model
        tiny = pd.DataFrame({"area": [1.0], "species": [10]})
        result = sars.sar_multi(tiny, models=["power"])
        r = repr(result)
        assert "MultiSARFit" in r


class TestAveragedSARRepr:
    def test_repr(self, galap):
        avg = sars.sar_average(galap, models=["power", "loga"])
        r = repr(avg)
        assert "AveragedSAR" in r
        assert "2 models" in r


class TestBootstrappedCIRepr:
    def test_repr(self, galap):
        ci = sars.bootstrap_ci(
            galap, models=["power"], n_boot=5,
            rng=np.random.default_rng(42),
        )
        r = repr(ci)
        assert "BootstrappedCI" in r
        assert "n_boot=" in r


class TestAICcEdge:
    def test_small_n_large_k(self):
        """When n - k - 1 <= 0, AICc should be inf."""
        val = sars.aicc(k=10, n=10, rss=100.0)
        assert val == np.inf


class TestPlotsImport:
    def test_plots_module_importable(self):
        from sars import _plots  # noqa: F401
