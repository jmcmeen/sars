"""Tests for threshold / piecewise SAR models."""

import numpy as np
import pytest

import sars


class TestSarThreshold:
    def test_basic(self, galap):
        """sar_threshold should return a ThresholdFit."""
        result = sars.sar_threshold(galap)
        assert isinstance(result, sars.ThresholdFit)
        assert result.best_model in {"ContOne", "ZslopeOne", "Linear"}

    def test_summary_table(self, galap):
        result = sars.sar_threshold(galap)
        assert "model" in result.summary.columns
        assert "AICc" in result.summary.columns
        assert "R2" in result.summary.columns
        assert len(result.summary) == 3  # all three models

    def test_summary_sorted_by_aicc(self, galap):
        result = sars.sar_threshold(galap)
        aicc_vals = result.summary["AICc"].values
        assert np.all(aicc_vals[:-1] <= aicc_vals[1:])

    def test_breakpoint_models_have_breakpoint(self, galap):
        result = sars.sar_threshold(galap)
        for name, fit in result.all_fits.items():
            if name == "Linear":
                assert fit["breakpoint"] is None
            else:
                assert fit["breakpoint"] is not None

    def test_predict(self, galap):
        result = sars.sar_threshold(galap)
        pred = result.predict(galap["area"].values)
        assert len(pred) == len(galap)
        assert np.all(np.isfinite(pred))

    def test_predict_scalar(self, galap):
        result = sars.sar_threshold(galap)
        pred = result.predict(5.0)
        assert np.isfinite(pred)

    def test_linear_baseline(self, galap):
        """Linear model should have one segment, no breakpoint."""
        result = sars.sar_threshold(galap, models=["Linear"])
        assert result.best_model == "Linear"
        assert result.best_breakpoint is None
        assert len(result.best_segments) == 1

    def test_cont_one_segments(self, galap):
        """ContOne should have two segments."""
        result = sars.sar_threshold(galap, models=["ContOne"])
        assert result.best_model == "ContOne"
        assert result.best_breakpoint is not None
        assert len(result.best_segments) == 2

    def test_zslope_one_flat_left(self, galap):
        """ZslopeOne left segment should have zero slope."""
        result = sars.sar_threshold(galap, models=["ZslopeOne"])
        assert result.best_model == "ZslopeOne"
        assert len(result.best_segments) == 2
        assert result.best_segments[0]["slope"] == 0.0

    def test_cont_one_continuity(self, galap):
        """ContOne segments should be continuous at breakpoint."""
        result = sars.sar_threshold(galap, models=["ContOne"])
        seg1, seg2 = result.best_segments
        bp = result.best_breakpoint
        y_left = seg1["intercept"] + seg1["slope"] * bp
        y_right = seg2["intercept"] + seg2["slope"] * bp
        assert y_left == pytest.approx(y_right, abs=1e-8)

    def test_zslope_one_continuity(self, galap):
        """ZslopeOne segments should be continuous at breakpoint."""
        result = sars.sar_threshold(galap, models=["ZslopeOne"])
        seg1, seg2 = result.best_segments
        bp = result.best_breakpoint
        y_left = seg1["intercept"] + seg1["slope"] * bp
        y_right = seg2["intercept"] + seg2["slope"] * bp
        assert y_left == pytest.approx(y_right, abs=1e-8)

    def test_r2_between_0_and_1(self, galap):
        result = sars.sar_threshold(galap)
        for _, row in result.summary.iterrows():
            assert 0.0 <= row["R2"] <= 1.0, (
                f"{row['model']} R2 out of range: {row['R2']}"
            )

    def test_custom_interval(self, galap):
        """Smaller interval should still work."""
        result = sars.sar_threshold(galap, interval=0.01)
        assert isinstance(result, sars.ThresholdFit)

    def test_unknown_model_raises(self, galap):
        with pytest.raises(ValueError, match="Unknown threshold model"):
            sars.sar_threshold(galap, models=["FakeModel"])

    def test_subset_models(self, galap):
        result = sars.sar_threshold(galap, models=["Linear", "ContOne"])
        assert len(result.summary) == 2
        assert set(result.summary["model"]) == {"Linear", "ContOne"}

    def test_repr(self, galap):
        result = sars.sar_threshold(galap)
        r = repr(result)
        assert "ThresholdFit" in r
