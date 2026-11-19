"""Tests for SAR plotting functions."""

import matplotlib  # noqa: I001

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest

from sars import (
    load_galap,
    plot_average,
    plot_fit,
    plot_multi,
    plot_residuals,
    plot_threshold,
    sar_average,
    sar_multi,
    sar_power,
    sar_threshold,
    threshold_ci,
)
from sars._inference import bootstrap_ci


@pytest.fixture
def galap_fit():
    data = load_galap()
    return sar_power(data)


@pytest.fixture
def galap_multi():
    data = load_galap()
    return sar_multi(data, models=["power", "loga", "negexpo", "monod", "linear"])


@pytest.fixture
def galap_avg():
    data = load_galap()
    return sar_average(data, models=["power", "loga", "negexpo", "monod", "linear"])


class TestPlotFit:
    def test_returns_axes(self, galap_fit):
        ax = plot_fit(galap_fit)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_accepts_existing_axes(self, galap_fit):
        fig, ax = plt.subplots()
        result = plot_fit(galap_fit, ax=ax)
        assert result is ax
        plt.close("all")

    def test_log_scale(self, galap_fit):
        ax = plot_fit(galap_fit, log=True)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close("all")

    def test_has_scatter_and_line(self, galap_fit):
        ax = plot_fit(galap_fit)
        # Should have at least one PathCollection (scatter) and one Line2D
        assert len(ax.collections) >= 1
        assert len(ax.lines) >= 1
        plt.close("all")


class TestPlotMulti:
    def test_returns_axes(self, galap_multi):
        ax = plot_multi(galap_multi)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_top_n(self, galap_multi):
        ax = plot_multi(galap_multi, top_n=3)
        # scatter + up to 3 model lines
        assert len(ax.lines) <= 3
        plt.close("all")

    def test_accepts_existing_axes(self, galap_multi):
        fig, ax = plt.subplots()
        result = plot_multi(galap_multi, ax=ax)
        assert result is ax
        plt.close("all")


class TestPlotAverage:
    def test_returns_axes(self, galap_avg):
        ax = plot_average(galap_avg)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_with_bootstrap_ci(self, galap_avg):
        data = load_galap()
        boot = bootstrap_ci(
            data,
            models=["power", "loga"],
            n_boot=10,
            rng=np.random.default_rng(42),
        )
        ax = plot_average(galap_avg, ci=True, boot=boot)
        assert isinstance(ax, plt.Axes)
        # Should have a PolyCollection for the CI band
        assert len(ax.collections) >= 2  # scatter + fill_between
        plt.close("all")

    def test_no_ci(self, galap_avg):
        ax = plot_average(galap_avg, ci=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestPlotResiduals:
    def test_returns_axes(self, galap_fit):
        ax = plot_residuals(galap_fit)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_has_zero_line(self, galap_fit):
        ax = plot_residuals(galap_fit)
        # Should have the zero reference line
        assert len(ax.lines) >= 1
        plt.close("all")

    def test_accepts_existing_axes(self, galap_fit):
        fig, ax = plt.subplots()
        result = plot_residuals(galap_fit, ax=ax)
        assert result is ax
        plt.close("all")


THRESHOLD_MODELS = ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo", "Linear"]


@pytest.fixture(scope="module")
def galap_threshold():
    return sar_threshold(load_galap(), models=THRESHOLD_MODELS, interval=0.1)


def _segments_drawn(ax):
    """The fitted segments' lines: everything but the vertical breakpoint markers."""
    return [ln for ln in ax.lines if len(set(ln.get_xdata())) > 1]


def _breakpoints_drawn(ax):
    return [ln.get_xdata()[0] for ln in ax.lines if len(set(ln.get_xdata())) == 1]


class TestPlotThreshold:
    def test_returns_axes(self, galap_threshold):
        ax = plot_threshold(galap_threshold)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_accepts_existing_axes(self, galap_threshold):
        fig, ax = plt.subplots()
        assert plot_threshold(galap_threshold, ax=ax) is ax
        plt.close("all")

    def test_default_model_is_the_best(self, galap_threshold):
        ax = plot_threshold(galap_threshold)
        assert galap_threshold.best_model in ax.get_title()
        plt.close("all")

    @pytest.mark.parametrize("model", THRESHOLD_MODELS)
    def test_each_model_draws_its_segments_and_breakpoints(self, galap_threshold, model):
        ax = plot_threshold(galap_threshold, model=model)
        fitted = galap_threshold.all_fits[model]
        assert len(_segments_drawn(ax)) == len(fitted["segments"])
        assert _breakpoints_drawn(ax) == pytest.approx(list(fitted["breakpoints"]))
        assert model in ax.get_title()
        plt.close("all")

    def test_data_are_plotted_on_the_log_scale_and_labelled_so(self, galap_threshold):
        ax = plot_threshold(galap_threshold, model="ContOne")
        data = load_galap()
        points = ax.collections[0].get_offsets()
        np.testing.assert_allclose(points[:, 0], np.log(data["area"]))
        np.testing.assert_allclose(points[:, 1], data["species"])
        assert ax.get_xlabel() == "ln(Area)"
        assert ax.get_xscale() == "linear"
        plt.close("all")

    def test_area_scale_fit_is_plotted_and_labelled_in_area(self):
        data = load_galap()
        fit = sar_threshold(data, models=["ContOne"], log_area=False)
        ax = plot_threshold(fit)
        np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 0], data["area"])
        assert ax.get_xlabel() == "Area"
        plt.close("all")

    @pytest.mark.parametrize("model", ["ContOne", "ContTwo", "ZslopeTwo"])
    def test_continuous_segments_join(self, galap_threshold, model):
        lines = _segments_drawn(plot_threshold(galap_threshold, model=model))
        for left, right in zip(lines[:-1], lines[1:], strict=True):
            assert left.get_xdata()[-1] == right.get_xdata()[0]
            assert left.get_ydata()[-1] == pytest.approx(right.get_ydata()[0])
        plt.close("all")

    @pytest.mark.parametrize("model", ["DiscOne", "DiscTwo"])
    def test_discontinuous_segments_are_not_joined_across_a_jump(self, galap_threshold, model):
        lines = _segments_drawn(plot_threshold(galap_threshold, model=model))
        fit = galap_threshold.all_fits[model]
        assert len(lines) == len(fit["segments"]) > 1
        x = np.sort(np.log(load_galap()["area"].to_numpy(dtype=float)))
        for left, right, t in zip(lines[:-1], lines[1:], fit["breakpoints"], strict=True):
            # Separate lines: one ends at the breakpoint, its own last island, and
            # the next starts at the first island above it, leaving the gap empty.
            assert len(left.get_xdata()) == 2 and len(right.get_xdata()) == 2
            assert left.get_xdata()[-1] == t
            assert right.get_xdata()[0] == x[x > t][0] > t
        plt.close("all")

    def test_ci_label_keeps_the_confidence_level(self, galap_threshold):
        ci = threshold_ci(
            galap_threshold, "DiscOne", n_boot=5, conf=0.999, rng=np.random.default_rng(0)
        )
        labels = plot_threshold(galap_threshold, ci=ci).get_legend_handles_labels()[1]
        assert "99.9% CI" in labels
        plt.close("all")

    def test_ci_from_another_fit_raises(self, galap_threshold):
        other = load_galap().iloc[:-2]
        ci = threshold_ci(
            sar_threshold(other, models=["ContOne"], interval=0.1),
            n_boot=5,
            rng=np.random.default_rng(0),
        )
        with pytest.raises(ValueError, match="ci is for another fit"):
            plot_threshold(galap_threshold, ci=ci)
        plt.close("all")

    def test_data_changed_since_the_fit_raises(self):
        data = load_galap()
        fit = sar_threshold(data, models=["DiscOne"])
        data["species"] = data["species"] + 10
        with pytest.raises(ValueError, match="has changed since"):
            plot_threshold(fit)
        plt.close("all")

    def test_lines_follow_the_models_predictions(self, galap_threshold):
        from sars._threshold import _predict_segments

        segments = galap_threshold.all_fits["DiscTwo"]["segments"]
        for line in _segments_drawn(plot_threshold(galap_threshold, model="DiscTwo")):
            inside = np.linspace(*line.get_xdata(), 5)[1:-1]
            drawn = np.interp(inside, line.get_xdata(), line.get_ydata())
            np.testing.assert_allclose(drawn, _predict_segments(inside, segments))
        plt.close("all")

    @pytest.mark.parametrize("model", ["ContOne", "DiscTwo"])
    def test_ci_is_shaded_for_each_breakpoint(self, galap_threshold, model):
        ci = threshold_ci(galap_threshold, model, n_boot=30, rng=np.random.default_rng(0))
        bare = plot_threshold(galap_threshold, model=model)
        ax = plot_threshold(galap_threshold, ci=ci)  # the model comes from the interval
        assert model in ax.get_title()
        assert len(ax.patches) - len(bare.patches) == len(ci.estimates)
        spans = sorted((p.get_x(), p.get_x() + p.get_width()) for p in ax.patches)
        np.testing.assert_allclose(spans, sorted(zip(ci.lower, ci.upper, strict=True)))
        labels = ax.get_legend_handles_labels()[1]
        assert "95% CI" in labels and "Breakpoint" in labels
        plt.close("all")

    def test_ci_for_another_model_raises(self, galap_threshold):
        ci = threshold_ci(galap_threshold, "DiscOne", n_boot=5, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="ci is for DiscOne, not ContOne"):
            plot_threshold(galap_threshold, model="ContOne", ci=ci)
        plt.close("all")

    def test_ci_on_another_scale_raises(self, galap_threshold):
        on_area = sar_threshold(load_galap(), models=["DiscOne"], log_area=False)
        ci = threshold_ci(on_area, n_boot=5, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="different scales"):
            plot_threshold(galap_threshold, ci=ci)
        plt.close("all")

    def test_a_model_the_fit_lacks_raises(self):
        fit = sar_threshold(load_galap())
        with pytest.raises(ValueError, match="DiscOne is not among this fit's models"):
            plot_threshold(fit, model="DiscOne")
        plt.close("all")

    def test_keyword_overrides(self, galap_threshold):
        ax = plot_threshold(
            galap_threshold,
            model="DiscOne",
            scatter_kw={"color": "green"},
            line_kw={"color": "orange", "label": "mine"},
        )
        assert all(ln.get_color() == "orange" for ln in _segments_drawn(ax))
        labels = ax.get_legend_handles_labels()[1]
        assert labels.count("mine") == 1  # one legend entry, however many segments
        plt.close("all")
