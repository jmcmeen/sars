"""Tests for SAR plotting functions."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sars import (
    load_galap,
    plot_average,
    plot_fit,
    plot_multi,
    plot_residuals,
    sar_average,
    sar_multi,
    sar_power,
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
