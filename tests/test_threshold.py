"""Tests for threshold / piecewise SAR models."""

import numpy as np
import pandas as pd
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

    @pytest.mark.parametrize("model", ["ContOne", "ZslopeOne", "Linear"])
    def test_predict_is_never_zero_outside_the_fitted_range(self, galap, model):
        result = sars.sar_threshold(galap, models=[model])
        below, above = galap["area"].min() / 10, galap["area"].max() * 10
        pred = result.predict([below, above])
        assert np.all(np.isfinite(pred)) and np.all(pred != 0)

    def test_predict_continues_the_fitted_line_outside_the_range(self):
        area = np.geomspace(1, 1000, 12)
        data = pd.DataFrame({"area": area, "species": 2.0 + 3.0 * np.log(area)})
        result = sars.sar_threshold(data, models=["Linear"])
        out = np.array([0.01, 1e6])
        np.testing.assert_allclose(result.predict(out), 2.0 + 3.0 * np.log(out))

    def test_predict_extends_each_end_segment(self):
        # Flat at 5 up to ln(area) = 3, then rising with slope 2.
        area = np.geomspace(1, np.exp(6), 30)
        x = np.log(area)
        data = pd.DataFrame({"area": area, "species": np.where(x <= 3, 5.0, 5.0 + 2 * (x - 3))})
        result = sars.sar_threshold(data, models=["ZslopeOne"])
        assert result.breakpoint_area == pytest.approx(np.exp(3), rel=0.05)
        below, above = np.exp(-4), np.exp(10)
        np.testing.assert_allclose(result.predict([below, above]), [5.0, 19.0], atol=0.2)

    def test_predict_on_the_area_scale_extends_too(self):
        area = np.linspace(1, 100, 12)
        data = pd.DataFrame({"area": area, "species": 4.0 + 0.5 * area})
        result = sars.sar_threshold(data, models=["Linear"], log_area=False)
        np.testing.assert_allclose(result.predict([0.0, 300.0]), [4.0, 154.0])

    def test_predict_nan_gives_nan(self, galap):
        assert np.isnan(sars.sar_threshold(galap).predict([np.nan]))[0]

    @pytest.mark.parametrize("area", [0.0, -1.0])
    def test_predict_refuses_non_positive_areas_on_the_log_scale(self, galap, area):
        with pytest.raises(ValueError, match="positive"):
            sars.sar_threshold(galap).predict([area])

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
            assert 0.0 <= row["R2"] <= 1.0, f"{row['model']} R2 out of range: {row['R2']}"

    def test_custom_interval(self, galap):
        """A coarser interval than the default still works."""
        result = sars.sar_threshold(galap, interval=0.1)
        x = np.sort(np.log(galap["area"].to_numpy()))
        for model in ["ContOne", "ZslopeOne"]:
            steps = (result.all_fits[model]["breakpoint"] - x[2]) / 0.1
            assert steps == pytest.approx(round(steps), abs=1e-9), model

    def test_too_fine_an_interval_raises(self, galap):
        with pytest.raises(ValueError, match="breakpoint candidates"):
            sars.sar_threshold(galap, interval=1e-9)

    def test_unknown_model_raises(self, galap):
        with pytest.raises(ValueError, match="Unknown threshold model"):
            sars.sar_threshold(galap, models=["FakeModel"])

    def test_subset_models(self, galap):
        result = sars.sar_threshold(galap, models=["Linear", "ContOne"])
        assert len(result.summary) == 2
        assert set(result.summary["model"]) == {"Linear", "ContOne"}

    def test_breakpoint_is_reported_in_area_units_too(self, galap):
        """The fit is on the natural log of area; breakpoint_area undoes that (#87)."""
        result = sars.sar_threshold(galap, models=["ZslopeOne"])
        assert result.log_area is True
        assert result.breakpoint_area == pytest.approx(np.exp(result.best_breakpoint))
        row = result.summary.set_index("model").loc["ZslopeOne"]
        assert row["breakpoint_area"] == pytest.approx(np.exp(row["breakpoint"]))
        assert "breakpoint_area" in repr(result)

    def test_linear_has_no_breakpoint_area(self, galap):
        result = sars.sar_threshold(galap, models=["Linear"])
        assert result.breakpoint_area is None
        assert result.summary["breakpoint_area"].isna().all()

    def test_log_area_false_fits_on_area_itself(self, galap):
        result = sars.sar_threshold(galap, log_area=False)
        assert result.log_area is False
        assert set(result.summary["model"]) == {"ContOne", "ZslopeOne", "Linear"}
        bps = result.summary.dropna(subset=["breakpoint"])
        # On the untransformed scale the breakpoint already is an area, inside the data.
        assert (bps["breakpoint_area"] == bps["breakpoint"]).all()
        assert bps["breakpoint"].between(galap["area"].min(), galap["area"].max()).all()
        # The linear baseline is then a straight line in area, not in log(area).
        linear = result.all_fits["Linear"]["segments"][0]
        areas = np.array([5.0, 500.0])
        expected = linear["intercept"] + linear["slope"] * areas
        assert sars.sar_threshold(galap, models=["Linear"], log_area=False).predict(
            areas
        ) == pytest.approx(expected)
        on_log = sars.sar_threshold(galap, models=["Linear"]).all_fits["Linear"]["r2"]
        assert result.all_fits["Linear"]["r2"] != pytest.approx(on_log)

    def test_log_area_false_recovers_a_breakpoint_in_area_units(self):
        """A flat-then-rising relationship in area: the default grid on that scale has to
        be fine enough to find the knee, not just the first candidate."""
        import pandas as pd

        area = np.linspace(1.0, 100.0, 40)
        species = 10.0 + np.where(area > 30.0, 0.5 * (area - 30.0), 0.0)
        result = sars.sar_threshold(pd.DataFrame({"area": area, "species": species}),
                                    models=["ZslopeOne", "Linear"], log_area=False)  # fmt: skip
        assert result.best_model == "ZslopeOne"
        assert result.breakpoint_area == pytest.approx(30.0, abs=0.2)
        assert result.predict([10.0, 90.0]) == pytest.approx([10.0, 40.0], abs=0.1)
        assert "breakpoint_area=" in repr(result) and "ln area" not in repr(result)

    @pytest.mark.parametrize("bad", [0, -0.1, float("nan"), float("inf")])
    def test_interval_must_be_positive_and_finite(self, galap, bad):
        with pytest.raises(ValueError, match="interval must be positive and finite"):
            sars.sar_threshold(galap, interval=bad)

    def test_area_scale_grid_covers_the_small_islands_despite_huge_ones(self):
        """The default grid on the area scale is sized from the breakpoint window, not
        the full range, so a few huge areas don't push every candidate past the knee."""
        import pandas as pd

        small = np.linspace(1.0, 20.0, 12)
        area = np.concatenate([small, [2000.0, 5000.0, 9000.0]])
        species = 10.0 + np.where(area > 8.0, 0.8 * (area - 8.0), 0.0)
        result = sars.sar_threshold(pd.DataFrame({"area": area, "species": species}),
                                    models=["ZslopeOne"], log_area=False)  # fmt: skip
        assert result.breakpoint_area == pytest.approx(8.0, abs=2.0)

    def test_equal_areas_fall_back_to_the_linear_baseline_on_either_scale(self):
        import pandas as pd

        data = pd.DataFrame({"area": [10.0] * 8, "species": range(1, 9)})
        for log_area in (True, False):
            assert sars.sar_threshold(data, log_area=log_area).best_model == "Linear"

    def test_repr(self, galap):
        result = sars.sar_threshold(galap)
        r = repr(result)
        assert "ThresholdFit" in r
