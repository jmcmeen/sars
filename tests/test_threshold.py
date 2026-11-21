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


ALL_MODELS = ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo", "Linear"]


def _two_knees(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Flat, then rising, then rising faster, on ln(area), with knees at ln 2 and ln 5."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 7.0, n))
    y = 10.0 + 4.0 * np.maximum(x - 2.0, 0.0) + 8.0 * np.maximum(x - 5.0, 0.0)
    return pd.DataFrame({"area": np.exp(x), "species": y + rng.normal(0, 0.05, n)})


class TestMoreThresholdModels:
    """DiscOne, ContTwo, ZslopeTwo and DiscTwo, as in R sars."""

    def test_default_models_are_unchanged(self, galap):
        assert set(sars.sar_threshold(galap).all_fits) == {"ContOne", "ZslopeOne", "Linear"}

    def test_each_model_reports_its_parameters_breakpoints_and_segments(self, galap):
        fit = sars.sar_threshold(galap, models=ALL_MODELS)
        shape = {"ContOne": (5, 1), "ZslopeOne": (4, 1), "DiscOne": (6, 1),
                 "ContTwo": (7, 2), "ZslopeTwo": (6, 2), "DiscTwo": (9, 2), "Linear": (3, 0)}  # fmt: skip
        for name, (k, n_breaks) in shape.items():
            result = fit.all_fits[name]
            assert result["k"] == k, name
            assert len(result["breakpoints"]) == n_breaks, name
            assert len(result["segments"]) == n_breaks + 1, name
            assert list(result["breakpoints"]) == sorted(result["breakpoints"]), name

    def test_outer_segments_hold_three_islands(self, galap):
        fit = sars.sar_threshold(galap, models=ALL_MODELS)
        x = np.log(galap["area"].to_numpy())
        for name, result in fit.all_fits.items():
            if result["breakpoints"]:
                assert np.sum(x <= result["breakpoints"][0]) >= 3, name
                assert np.sum(x > result["breakpoints"][-1]) >= 3, name

    def test_discontinuous_breakpoints_are_observed_areas(self, galap):
        fit = sars.sar_threshold(galap, models=["DiscOne", "DiscTwo"])
        x = np.log(galap["area"].to_numpy())
        for name in ["DiscOne", "DiscTwo"]:
            for t in fit.all_fits[name]["breakpoints"]:
                assert np.min(np.abs(x - t)) == 0.0, name

    @pytest.mark.parametrize("model", ["ContTwo", "ZslopeTwo"])
    def test_continuous_two_threshold_segments_meet(self, galap, model):
        result = sars.sar_threshold(galap, models=[model]).all_fits[model]
        segs = result["segments"]
        for t, left, right in zip(result["breakpoints"], segs[:-1], segs[1:], strict=True):
            at_left = left["intercept"] + left["slope"] * t
            assert at_left == pytest.approx(right["intercept"] + right["slope"] * t, abs=1e-8)
        if model == "ZslopeTwo":
            assert segs[0]["slope"] == 0.0

    @pytest.mark.parametrize("model", ["ContTwo", "ZslopeTwo"])
    def test_two_knees_are_recovered(self, model):
        data = _two_knees()
        fit = sars.sar_threshold(data, models=[model, "ContOne", "Linear"])
        assert fit.best_model == model
        assert fit.best_breakpoints == pytest.approx((2.0, 5.0), abs=0.05)
        assert fit.breakpoints_area == pytest.approx(np.exp(fit.best_breakpoints))
        assert fit.predict(np.exp([1.0, 6.5])) == pytest.approx([10.0, 40.0], abs=0.2)

    def test_jumps_are_recovered_by_the_discontinuous_models(self):
        x = np.arange(1.0, 13.0)
        y = np.where(x <= 4, 5.0 + x, np.where(x <= 8, 30.0 + 2 * x, 80.0 - x))
        data = pd.DataFrame({"area": np.exp(x), "species": y})
        one = sars.sar_threshold(data, models=["DiscOne"]).all_fits["DiscOne"]
        assert one["breakpoints"][0] in (4.0, 8.0)  # at one of the jumps
        two = sars.sar_threshold(data, models=["DiscTwo", "DiscOne", "Linear"])
        assert two.best_model == "DiscTwo"
        assert two.best_breakpoints == pytest.approx((4.0, 8.0))
        assert two.all_fits["DiscTwo"]["rss"] == pytest.approx(0.0, abs=1e-18)
        # A point at a breakpoint is in the segment to its left, as in the fit.
        assert two.predict(np.exp([4.0, 8.0])) == pytest.approx([9.0, 46.0])

    def test_vectorised_searches_match_trying_every_pair(self):
        """The two-threshold searches score pairs in bulk; the pair they pick must be the
        one fitting every pair separately would."""
        from sars import _threshold as th

        rng = np.random.default_rng(3)
        x = np.sort(rng.uniform(0, 5, 14))
        y = 20 + 3 * x + rng.normal(0, 2, 14)
        grid = th._breakpoint_grid(x, 0.05, 3)
        values = np.unique(x)
        for fit_func, sloped, cands in [
            (th._fit_cont_two, True, grid),
            (th._fit_zslope_two, False, grid),
            (th._fit_disc_two, None, values),
        ]:
            best = min(
                (fit_func(x, y, a, b) for a in cands for b in cands
                 if a < b and np.sum(x <= a) >= 3 and np.sum(x > b) >= 3),
                key=lambda r: r["rss"],
            )  # fmt: skip
            if sloped is None:
                found = th._search_disc_two(x, y)
            else:
                found = th._grid_search_two_breakpoints(x, y, fit_func, sloped, 0.05)
            assert found["rss"] == pytest.approx(best["rss"], rel=1e-12)
            assert found["breakpoints"] == pytest.approx(best["breakpoints"])
        best_one = min(
            (
                th._fit_disc_one(x, y, t)
                for t in values
                if np.sum(x <= t) >= 3 and np.sum(x > t) >= 3
            ),
            key=lambda r: r["rss"],
        )
        assert th._search_disc_one(x, y)["breakpoints"] == best_one["breakpoints"]

    def test_discontinuous_searches_of_many_responses_match_trying_every_candidate(self):
        """The discontinuous searches score many responses at once, from running
        sums: each response must get the breakpoints fitting every candidate
        separately would."""
        from sars import _threshold as th

        rng = np.random.default_rng(8)
        x = np.sort(rng.uniform(0, 5, 13))
        ys = 20 + 3 * x + rng.normal(0, 2, (6, 13))
        values = np.unique(x)
        ones = th._disc_one_breakpoints(x, ys)
        twos = th._disc_two_breakpoints(x, ys)
        for row, y in enumerate(ys):
            best_one = min(
                (th._fit_disc_one(x, y, t) for t in values
                 if np.sum(x <= t) >= 3 and np.sum(x > t) >= 3),
                key=lambda r: r["rss"],
            )  # fmt: skip
            best_two = min(
                (th._fit_disc_two(x, y, a, b) for a in values for b in values
                 if a < b and np.sum(x <= a) >= 3 and np.sum(x > b) >= 3),
                key=lambda r: r["rss"],
            )  # fmt: skip
            assert ones[row] == best_one["breakpoints"][0]
            assert tuple(twos[row]) == pytest.approx(best_two["breakpoints"])

    def test_summary_reports_the_second_breakpoint(self, galap):
        fit = sars.sar_threshold(galap, models=ALL_MODELS)
        table = fit.summary.set_index("model")
        assert table.loc["ContTwo", "breakpoint2"] == fit.all_fits["ContTwo"]["breakpoints"][1]
        assert table.loc["ContTwo", "breakpoint2_area"] == pytest.approx(
            np.exp(table.loc["ContTwo", "breakpoint2"])
        )
        assert table.loc[["ContOne", "DiscOne", "Linear"], "breakpoint2"].isna().all()

    def test_repr_shows_both_breakpoints(self):
        fit = sars.sar_threshold(_two_knees(), models=["ContTwo"])
        assert "breakpoints=(" in repr(fit) and "breakpoints_area=(" in repr(fit)
        assert fit.best_breakpoint == fit.best_breakpoints[0]
        on_area = sars.sar_threshold(_two_knees(), models=["ContTwo"], log_area=False)
        assert "breakpoints_area=(" in repr(on_area) and "ln area" not in repr(on_area)

    def test_one_threshold_and_linear_fits_report_their_breakpoints_as_a_tuple(self, galap):
        fit = sars.sar_threshold(galap, models=["ContOne"])
        assert fit.best_breakpoints == (fit.best_breakpoint,)
        assert sars.sar_threshold(galap, models=["Linear"]).best_breakpoints == ()

    def test_too_many_pairs_raises(self, galap, monkeypatch):
        from sars import _threshold as th

        monkeypatch.setattr(th, "_MAX_PAIRS", 1000)
        with pytest.raises(ValueError, match="pairs of breakpoints"):
            sars.sar_threshold(galap, models=["ContTwo"])
        with pytest.raises(ValueError, match="too many distinct areas"):
            sars.sar_threshold(_two_knees(n=60), models=["DiscTwo"])

    def test_small_data_still_falls_back_to_linear(self):
        data = pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0, 5.0], "species": [1, 2, 3, 4, 5]})
        with pytest.warns(UserWarning):
            fit = sars.sar_threshold(data, models=ALL_MODELS)
        assert set(fit.all_fits) == {"Linear"}

    def test_second_breakpoint_columns_are_float_without_a_two_threshold_model(self, galap):
        summary = sars.sar_threshold(galap).summary
        assert summary["breakpoint2"].dtype == float
        assert summary["breakpoint2_area"].dtype == float
        assert summary["breakpoint2"].isna().all()

    def test_discontinuous_breakpoints_in_area_units_are_the_observed_areas(self, galap):
        fit = sars.sar_threshold(galap, models=["DiscOne", "DiscTwo"])
        areas = set(galap["area"])
        for name in ["DiscOne", "DiscTwo"]:
            assert set(fit.all_fits[name]["breakpoints_area"]) <= areas, name
        assert fit.breakpoint_area in areas
        row = fit.summary.set_index("model").loc["DiscTwo"]
        assert {row["breakpoint_area"], row["breakpoint2_area"]} <= areas

    def test_a_fit_built_with_one_breakpoint_lists_it(self, galap):
        fit = sars.ThresholdFit("ContOne", 1.2, [], pd.DataFrame(), {}, galap)
        assert fit.best_breakpoints == (1.2,)
        assert fit.breakpoints_area == pytest.approx((np.exp(1.2),))

    def test_runs_far_from_the_mean_are_scored_exactly(self):
        """A cluster of areas far from the rest: running sums would lose its slope."""
        from sars import _threshold as th

        rng = np.random.default_rng(0)
        x = np.concatenate([np.linspace(1, 6, 6), 1e8 + np.linspace(0, 7, 8)])
        y = np.concatenate([np.full(6, 50.0), 10 + 5 * np.linspace(0, 7, 8)])
        y = y + rng.normal(0, 0.1, 14)
        _, _, runs = th._disc_candidates(x, y)
        xs, ys = x[6:] - x[6:].mean(), y[6:]
        slope, intercept = np.polyfit(xs, ys, 1)
        exact = np.sum((ys - intercept - slope * xs) ** 2)
        assert runs.rss(np.array([6]), np.array([14]))[0] == pytest.approx(exact, rel=1e-6)
        assert exact < 1.0  # a sloped run, not a flat one

    def test_two_threshold_search_in_small_blocks_matches_trying_every_pair(self, monkeypatch):
        from sars import _threshold as th

        monkeypatch.setattr(th, "_BLOCK", 50)  # a few candidates per block
        rng = np.random.default_rng(5)
        x = np.sort(rng.uniform(0, 5, 14))
        y = 20 + 3 * x + rng.normal(0, 2, 14)
        grid = th._breakpoint_grid(x, 0.05, 3)
        best = min(
            (th._fit_cont_two(x, y, a, b) for a in grid for b in grid
             if a < b and np.sum(x <= a) >= 3 and np.sum(x > b) >= 3),
            key=lambda r: r["rss"],
        )  # fmt: skip
        found = th._grid_search_two_breakpoints(x, y, th._fit_cont_two, True, 0.05)
        assert found["breakpoints"] == pytest.approx(best["breakpoints"])

    def test_undefined_aicc_warns_and_ranks_last(self):
        data = _two_knees(n=10)
        with pytest.warns(UserWarning, match="AICc is undefined for DiscTwo"):
            fit = sars.sar_threshold(data, models=["DiscTwo", "ContOne", "Linear"])
        assert fit.summary["model"].iloc[-1] == "DiscTwo"

    def test_undefined_aicc_for_every_model_raises(self):
        data = _two_knees(n=8)
        with pytest.raises(ValueError, match="AICc is undefined for every model"):
            sars.sar_threshold(data, models=["ContTwo", "DiscTwo"])


BREAKPOINT_MODELS = ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo"]
# A coarse grid keeps the bootstrap tests quick; the default 0.01 is searched
# the same way, a hundred candidates at a time instead of ten.
COARSE = 0.1


def _one_knee(noise: float, n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Flat at 40 species up to ln(area) = 3, then rising 8 per unit, plus noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 8, n)
    y = 40 + 8 * np.maximum(x - 3, 0) + rng.normal(0, noise, n)
    return pd.DataFrame({"area": np.exp(x), "species": y})


@pytest.fixture(scope="module")
def galap_all(galap):
    return sars.sar_threshold(galap, models=[*BREAKPOINT_MODELS, "Linear"], interval=COARSE)


class TestThresholdCI:
    @pytest.mark.parametrize("model", ["ContOne", "ZslopeOne"])
    def test_interval_covers_a_known_breakpoint(self, model):
        fit = sars.sar_threshold(_one_knee(noise=3.0), models=[model], interval=COARSE)
        ci = sars.threshold_ci(fit, n_boot=200, rng=np.random.default_rng(1))
        assert ci.lower[0] < 3.0 < ci.upper[0]
        assert ci.lower[0] <= ci.estimates[0] <= ci.upper[0]
        # The data pin the breakpoint down: the interval is far narrower than
        # the window the search covers (ln(area) from 0.4 to 7.6).
        assert ci.upper[0] - ci.lower[0] < 2.0

    @pytest.mark.parametrize("model", ["ContOne", "ZslopeOne"])
    def test_f_interval_covers_a_known_breakpoint(self, model):
        fit = sars.sar_threshold(_one_knee(noise=3.0), models=[model], interval=COARSE)
        ci = sars.threshold_ci(fit, method="F")
        assert ci.method == "F" and ci.n_boot == 0 and ci.samples.shape == (0, 1)
        assert ci.lower[0] < 3.0 < ci.upper[0]
        assert ci.lower[0] <= ci.estimates[0] <= ci.upper[0]
        assert ci.upper[0] - ci.lower[0] < 2.0
        assert repr(ci).endswith("method='F')")

    def test_f_bounds_are_where_the_test_stops_accepting(self):
        """Inside the bounds the F statistic is at most the critical value; just
        outside, above it."""
        from scipy import stats

        from sars import _threshold as th

        data = _one_knee(noise=3.0)
        fit = sars.sar_threshold(data, models=["ContOne"], interval=COARSE)
        ci = sars.threshold_ci(fit, method="F", conf=0.9)
        x = np.log(data["area"].to_numpy())
        y = data["species"].to_numpy()
        grid = th._search_grid(x, COARSE)
        rss = th._profile_rss(x, y, True, grid)
        n, p = len(y), 4
        stat = (rss - rss.min()) / (rss.min() / (n - p))
        crit = stats.f.ppf(0.9, 1, n - p)
        inside = (grid >= ci.lower[0] - 1e-12) & (grid <= ci.upper[0] + 1e-12)
        assert stat[inside].min() == 0.0
        lo, hi = np.flatnonzero(inside)[[0, -1]]
        assert stat[lo] <= crit and stat[hi] <= crit
        assert lo == 0 or stat[lo - 1] > crit
        assert hi == len(grid) - 1 or stat[hi + 1] > crit
        # Each fixed-breakpoint RSS is the fit's own.
        for k in (0, lo, hi, len(grid) - 1):
            assert rss[k] == pytest.approx(th._fit_cont_one(x, y, grid[k])["rss"], rel=1e-9)

    def test_f_interval_of_an_exact_fit_is_the_breakpoint(self):
        """Data on a hinge exactly: rounding can leave the best RSS a hair below
        zero, which must not turn the interval into the whole window."""
        from sars import _threshold as th

        area = np.exp(np.linspace(0, 8, 30))
        x = np.log(area)
        knee = th._search_grid(x, 0.01)[150]
        for slope in (5.0, 5.1, 7.3):
            y = 100 + 2 * x + slope * np.maximum(x - knee, 0)
            fit = sars.sar_threshold(pd.DataFrame({"area": area, "species": y}), models=["ContOne"])
            ci = sars.threshold_ci(fit, method="F")
            assert ci.lower[0] == pytest.approx(knee) and ci.upper[0] == pytest.approx(knee)

    def test_f_interval_coverage(self):
        """Over noisy data with one true breakpoint, 95% intervals cover it about
        95% of the time."""
        covered = []
        for seed in range(150):
            data = _one_knee(noise=4.0, seed=seed)
            fit = sars.sar_threshold(data, models=["ContOne"], interval=COARSE)
            ci = sars.threshold_ci(fit, method="F")
            covered.append(ci.lower[0] <= 3.0 <= ci.upper[0])
        assert 0.88 <= np.mean(covered) <= 0.99

    @pytest.mark.parametrize("model", ["DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo"])
    def test_f_interval_is_for_continuous_one_threshold_models(self, galap_all, model):
        with pytest.raises(ValueError, match="inverted F test gives intervals for ContOne"):
            sars.threshold_ci(galap_all, model=model, method="F")

    def test_unknown_method_raises(self, galap_all):
        with pytest.raises(ValueError, match="method must be 'boot' or 'F'"):
            sars.threshold_ci(galap_all, model="ContOne", method="bca")

    def test_interval_narrows_with_less_noise(self):
        widths = []
        for noise in (6.0, 2.0, 0.5):
            fit = sars.sar_threshold(_one_knee(noise), models=["ContOne"], interval=COARSE)
            ci = sars.threshold_ci(fit, n_boot=150, rng=np.random.default_rng(1))
            widths.append(ci.upper[0] - ci.lower[0])
        assert widths[0] > widths[1] > widths[2] > 0

    def test_same_rng_gives_the_same_interval(self, galap_all):
        a = sars.threshold_ci(galap_all, "ContOne", n_boot=30, rng=np.random.default_rng(7))
        b = sars.threshold_ci(galap_all, "ContOne", n_boot=30, rng=np.random.default_rng(7))
        c = sars.threshold_ci(galap_all, "ContOne", n_boot=30, rng=np.random.default_rng(8))
        np.testing.assert_array_equal(a.samples, b.samples)
        assert (a.lower, a.upper) == (b.lower, b.upper)
        assert not np.array_equal(a.samples, c.samples)

    @pytest.mark.parametrize("model", BREAKPOINT_MODELS)
    def test_each_model_gets_an_interval_per_breakpoint(self, galap_all, model):
        ci = sars.threshold_ci(galap_all, model, n_boot=40, rng=np.random.default_rng(3))
        n_breakpoints = 2 if model.endswith("Two") else 1
        assert ci.model == model
        assert ci.conf == 0.95
        assert ci.n_boot == 40
        assert ci.samples.shape == (40, n_breakpoints)
        assert ci.estimates == galap_all.all_fits[model]["breakpoints"]
        assert ci.estimates_area == galap_all.all_fits[model]["breakpoints_area"]
        for values in (ci.lower, ci.upper, ci.lower_area, ci.upper_area):
            assert len(values) == n_breakpoints
        assert all(lo <= hi for lo, hi in zip(ci.lower, ci.upper, strict=True))
        # The bounds are percentiles of the breakpoints the resamples gave.
        np.testing.assert_allclose(ci.lower, ci.samples.min(axis=0), atol=np.ptp(ci.samples))
        assert np.all(ci.samples.min(axis=0) <= ci.lower)
        assert np.all(ci.upper <= ci.samples.max(axis=0))
        if n_breakpoints == 2:  # every resample's pair is in order
            assert np.all(ci.samples[:, 0] < ci.samples[:, 1])

    @pytest.mark.parametrize("model", ["ContOne", "ZslopeTwo"])
    def test_area_bounds_are_the_log_bounds_exponentiated(self, galap_all, model):
        ci = sars.threshold_ci(galap_all, model, n_boot=30, rng=np.random.default_rng(3))
        assert ci.log_area
        np.testing.assert_allclose(ci.lower_area, np.exp(ci.lower))
        np.testing.assert_allclose(ci.upper_area, np.exp(ci.upper))

    @pytest.mark.parametrize("model", ["DiscOne", "DiscTwo"])
    def test_discontinuous_bounds_are_observed_areas(self, galap_all, galap, model):
        ci = sars.threshold_ci(galap_all, model, n_boot=40, rng=np.random.default_rng(3))
        areas = set(galap["area"])
        assert set(ci.lower_area) <= areas and set(ci.upper_area) <= areas
        assert set(ci.samples.ravel()) <= set(np.log(galap["area"]))

    def test_resamples_are_searched_on_the_fits_grid(self, galap):
        """The fit's own interval is reused: on a grid of 0.5 every resample's
        breakpoint is a whole number of steps from the first candidate."""
        fit = sars.sar_threshold(galap, models=["ContOne"], interval=0.5)
        assert fit.interval == 0.5
        ci = sars.threshold_ci(fit, n_boot=30, rng=np.random.default_rng(0))
        steps = (ci.samples[:, 0] - np.sort(np.log(galap["area"]))[2]) / 0.5
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)

    def test_default_interval_is_stored_on_the_fit(self, galap):
        assert sars.sar_threshold(galap).interval == 0.01
        assert sars.sar_threshold(galap, log_area=False).interval is None

    def test_area_scale_fit_reports_bounds_in_area_units(self):
        rng = np.random.default_rng(0)
        area = np.linspace(1, 100, 40)
        species = 30 + 2.0 * np.maximum(area - 40, 0) + rng.normal(0, 3, 40)
        data = pd.DataFrame({"area": area, "species": species})
        fit = sars.sar_threshold(data, models=["ZslopeOne"], log_area=False, interval=1.0)
        ci = sars.threshold_ci(fit, n_boot=100, rng=np.random.default_rng(1))
        assert not ci.log_area
        assert ci.lower_area == ci.lower and ci.upper_area == ci.upper
        assert ci.lower[0] < 40 < ci.upper[0]

    def test_default_model_is_the_best_with_a_breakpoint(self, galap):
        fit = sars.sar_threshold(galap)  # Linear ranks first on these data
        assert fit.best_model == "Linear"
        ranked = [m for m in fit.summary["model"] if m != "Linear"]
        ci = sars.threshold_ci(fit, n_boot=5, rng=np.random.default_rng(0))
        assert ci.model == ranked[0]

    def test_conf_sets_the_percentiles(self, galap_all):
        wide = sars.threshold_ci(galap_all, "ContOne", n_boot=60, rng=np.random.default_rng(2))
        narrow = sars.threshold_ci(
            galap_all, "ContOne", n_boot=60, conf=0.5, rng=np.random.default_rng(2)
        )
        np.testing.assert_array_equal(wide.samples, narrow.samples)
        assert narrow.conf == 0.5
        assert narrow.lower[0] == pytest.approx(np.quantile(narrow.samples[:, 0], 0.25))
        assert narrow.upper[0] == pytest.approx(np.quantile(narrow.samples[:, 0], 0.75))
        assert wide.lower[0] <= narrow.lower[0] <= narrow.upper[0] <= wide.upper[0]

    def test_linear_has_no_breakpoint_to_bound(self, galap_all):
        with pytest.raises(ValueError, match="Linear model has no breakpoint"):
            sars.threshold_ci(galap_all, "Linear")

    def test_a_fit_with_only_linear_raises(self, galap):
        fit = sars.sar_threshold(galap, models=["Linear"])
        with pytest.raises(ValueError, match="no model with a breakpoint"):
            sars.threshold_ci(fit)

    def test_a_model_the_fit_lacks_raises(self, galap):
        fit = sars.sar_threshold(galap)
        with pytest.raises(ValueError, match="DiscOne is not among this fit's models"):
            sars.threshold_ci(fit, "DiscOne")
        with pytest.raises(ValueError, match="Unknown threshold model"):
            sars.threshold_ci(fit, "Bogus")

    @pytest.mark.parametrize(
        "n_boot", [0, -5, 2.5, True, None, float("nan"), float("inf"), "10", 10.0]
    )
    def test_n_boot_must_be_a_positive_whole_number(self, galap_all, n_boot):
        with pytest.raises(ValueError, match="n_boot must be a positive whole number"):
            sars.threshold_ci(galap_all, "DiscOne", n_boot=n_boot)

    def test_n_boot_may_be_a_numpy_integer(self, galap_all):
        ci = sars.threshold_ci(galap_all, "DiscOne", n_boot=np.int64(7))
        assert ci.n_boot == 7 and len(ci.samples) == 7

    @pytest.mark.parametrize("model", BREAKPOINT_MODELS)
    def test_resamples_searched_together_match_searching_each_alone(self, galap, model):
        """The bootstrap scores all its resamples against the candidates at once;
        each must get the breakpoints sar_threshold's own search gives it."""
        from sars import _threshold as th

        rng = np.random.default_rng(11)
        x = np.log(galap["area"].to_numpy(dtype=float))
        ys = 100 + 20 * x + rng.normal(0, 30, (25, len(x)))
        together = th._resample_breakpoints(model, x, ys, 0.05)
        alone = [th._SEARCHES[model].fit(x, y, 0.05)["breakpoints"] for y in ys]
        np.testing.assert_array_equal(together, np.array(alone))

    @pytest.mark.parametrize("block", [1_000_000, 40])
    def test_grid_searches_match_trying_every_candidate(self, monkeypatch, block):
        """For several responses at once, in one block of candidates or many."""
        from sars import _threshold as th

        monkeypatch.setattr(th, "_BLOCK", block)
        rng = np.random.default_rng(4)
        x = np.sort(rng.uniform(0, 5, 14))
        ys = 20 + 3 * x + rng.normal(0, 2, (6, 14))
        grid = th._breakpoint_grid(x, 0.05, 3)
        for fit_func, sloped in [(th._fit_cont_one, True), (th._fit_zslope_one, False)]:
            found = th._best_breakpoints(x, ys, sloped, 0.05)
            for y, t in zip(ys, found, strict=True):
                best = min(
                    (fit_func(x, y, c) for c in grid
                     if np.sum(x <= c) >= 3 and np.sum(x > c) >= 3),
                    key=lambda r: r["rss"],
                )  # fmt: skip
                assert t == pytest.approx(best["breakpoint"])
                assert th._grid_search_breakpoint(x, y, fit_func, sloped, 0.05)[
                    "rss"
                ] == pytest.approx(best["rss"], rel=1e-12)
        for fit_func, sloped in [(th._fit_cont_two, True), (th._fit_zslope_two, False)]:
            found = th._best_breakpoint_pairs(x, ys, sloped, 0.05)
            for y, pair in zip(ys, found, strict=True):
                best = min(
                    (fit_func(x, y, a, b) for a in grid for b in grid
                     if a < b and np.sum(x <= a) >= 3 and np.sum(x > b) >= 3),
                    key=lambda r: r["rss"],
                )  # fmt: skip
                # Pairs between the same two areas fit alike, so compare the fits.
                assert fit_func(x, y, *pair)["rss"] == pytest.approx(best["rss"], rel=1e-9)

    def test_one_search_covers_many_resamples(self, galap_all, monkeypatch):
        """The resamples go to the search in batches, not one at a time."""
        from sars import _threshold as th

        calls = []
        real = th._resample_breakpoints
        monkeypatch.setattr(
            th,
            "_resample_breakpoints",
            lambda m, x, ys, i: calls.append(len(ys)) or real(m, x, ys, i),
        )
        monkeypatch.setattr(th, "_BOOT_BATCH", 40)
        ci = sars.threshold_ci(galap_all, "ContOne", n_boot=100, rng=np.random.default_rng(0))
        assert calls == [40, 40, 20]
        assert ci.samples.shape == (100, 1) and ci.n_boot == 100

    @pytest.mark.parametrize("model", ["ContOne", "DiscOne"])
    def test_data_changed_since_the_fit_raises(self, galap, model):
        data = galap.copy()
        fit = sars.sar_threshold(data, models=[model, "Linear"], interval=COARSE)
        data.loc[data.index[0], "species"] += 50
        with pytest.raises(ValueError, match="has changed since"):
            sars.threshold_ci(fit, model, n_boot=5)
        data.loc[data.index[0], "species"] -= 50
        assert sars.threshold_ci(fit, model, n_boot=5).n_boot == 5  # restored: fine again
        data["area"] = data["area"] * 2  # a Disc breakpoint is no longer an observed area
        with pytest.raises(ValueError, match="has changed since"):
            sars.threshold_ci(fit, model, n_boot=5)

    def test_rows_dropped_since_the_fit_raises(self, galap):
        data = galap.copy()
        fit = sars.sar_threshold(data, interval=COARSE)
        data.drop(index=data.index[-1], inplace=True)
        with pytest.raises(ValueError, match="has changed since"):
            sars.threshold_ci(fit, n_boot=5)

    def test_a_fit_without_an_interval_is_searched_over_a_thousandth_of_the_window(self, galap):
        from dataclasses import replace

        fit = replace(sars.sar_threshold(galap, models=["ContOne"]), interval=None)
        ci = sars.threshold_ci(fit, n_boot=20, rng=np.random.default_rng(0))
        window = np.sort(np.log(galap["area"]))[[2, -3]]
        steps = (ci.samples[:, 0] - window[0]) / ((window[1] - window[0]) / 1000)
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-6)

    @pytest.mark.parametrize("conf", [0, 1, 1.5, -0.1, 95])
    def test_conf_must_be_a_proportion(self, galap_all, conf):
        with pytest.raises(ValueError, match="conf"):
            sars.threshold_ci(galap_all, "DiscOne", conf=conf)

    def test_repr_gives_the_intervals_in_area_units(self, galap_all):
        ci = sars.threshold_ci(galap_all, "ContTwo", n_boot=20, rng=np.random.default_rng(0))
        text = repr(ci)
        assert text.startswith("ThresholdCI(model='ContTwo', breakpoints_area=(")
        assert text.count("[") == 2 and "n_boot=20" in text and "conf=0.95" in text
