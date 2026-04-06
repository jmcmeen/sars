"""Plotting functions for SAR fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from sars._inference import AveragedSAR, MultiSARFit
from sars._models import SARFit

if TYPE_CHECKING:
    from sars._inference import BootstrappedCI


def plot_fit(
    fit: SARFit,
    ax: plt.Axes | None = None,
    log: bool = False,
    scatter_kw: dict | None = None,
    line_kw: dict | None = None,
) -> plt.Axes:
    """Plot a single SAR model fit with observed data points.

    Parameters
    ----------
    fit : SARFit
        A fitted SAR model.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    log : bool
        If True, use log-log axes.
    scatter_kw : dict, optional
        Extra keyword arguments passed to ``ax.scatter()`` for the
        observed data points.
    line_kw : dict, optional
        Extra keyword arguments passed to ``ax.plot()`` for the fitted
        curve.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    area = fit.data["area"].values
    species = fit.data["species"].values

    s_defaults = dict(color="black", zorder=3, label="Observed")
    ax.scatter(area, species, **{**s_defaults, **(scatter_kw or {})})

    a_grid = np.linspace(area.min(), area.max(), 200)
    pred = fit.predict(a_grid)
    l_defaults = dict(color="dodgerblue", linewidth=2,
                      label=f"{fit.model} (R²={fit.r_squared:.3f})")
    ax.plot(a_grid, pred, **{**l_defaults, **(line_kw or {})})

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Area")
    ax.set_ylabel("Species")
    ax.set_title(f"SAR fit: {fit.model}")
    ax.legend()
    return ax


def plot_multi(
    multi_fit: MultiSARFit,
    top_n: int = 5,
    ax: plt.Axes | None = None,
    scatter_kw: dict | None = None,
    line_kw: dict | None = None,
) -> plt.Axes:
    """Plot the top N models from a multi-model fit.

    Parameters
    ----------
    multi_fit : MultiSARFit
        Result of ``sar_multi()``.
    top_n : int
        Number of top models (by AICc) to display.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    scatter_kw : dict, optional
        Extra keyword arguments passed to ``ax.scatter()`` for the
        observed data points.
    line_kw : dict, optional
        Extra keyword arguments passed to ``ax.plot()`` for each
        fitted curve.  Per-model ``color`` and ``label`` are set
        automatically but can be overridden.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    data = multi_fit.data
    area = data["area"].values
    species = data["species"].values

    s_defaults = dict(color="black", zorder=3, label="Observed")
    ax.scatter(area, species, **{**s_defaults, **(scatter_kw or {})})

    a_grid = np.linspace(area.min(), area.max(), 200)

    # Get top N models by AICc from summary
    top_models = multi_fit.summary.head(top_n)["model"].tolist()
    fits_by_name = {f.model: f for f in multi_fit.fits}

    cmap = plt.cm.tab20 if top_n > 10 else plt.cm.tab10
    colors = cmap(np.linspace(0, 1, top_n))

    extra_line = line_kw or {}
    for i, name in enumerate(top_models):
        fit = fits_by_name.get(name)
        if fit is None:
            continue
        pred = fit.predict(a_grid)
        w = multi_fit.summary.loc[
            multi_fit.summary["model"] == name, "weight"
        ].iloc[0]
        l_defaults = dict(color=colors[i], linewidth=1.5,
                          label=f"{name} (w={w:.3f})")
        ax.plot(a_grid, pred, **{**l_defaults, **extra_line})

    ax.set_xlabel("Area")
    ax.set_ylabel("Species")
    ax.set_title(f"Top {top_n} SAR models")
    ax.legend(fontsize="small")
    return ax


def plot_average(
    averaged: AveragedSAR,
    ci: bool = True,
    boot: BootstrappedCI | None = None,
    ax: plt.Axes | None = None,
    scatter_kw: dict | None = None,
    line_kw: dict | None = None,
    fill_kw: dict | None = None,
) -> plt.Axes:
    """Plot the model-averaged SAR prediction.

    Parameters
    ----------
    averaged : AveragedSAR
        Result of ``sar_average()``.
    ci : bool
        If True and ``boot`` is provided, shade the confidence interval.
    boot : BootstrappedCI, optional
        Bootstrap confidence intervals from ``bootstrap_ci()``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    scatter_kw : dict, optional
        Extra keyword arguments passed to ``ax.scatter()`` for the
        observed data points.
    line_kw : dict, optional
        Extra keyword arguments passed to ``ax.plot()`` for the
        averaged curve.
    fill_kw : dict, optional
        Extra keyword arguments passed to ``ax.fill_between()`` for the
        confidence band.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    data = averaged.multi.data
    area = data["area"].values
    species = data["species"].values

    s_defaults = dict(color="black", zorder=3, label="Observed")
    ax.scatter(area, species, **{**s_defaults, **(scatter_kw or {})})

    a_grid = np.linspace(area.min(), area.max(), 200)
    pred = averaged.predict(a_grid)
    l_defaults = dict(color="dodgerblue", linewidth=2, label="Model-averaged")
    ax.plot(a_grid, pred, **{**l_defaults, **(line_kw or {})})

    if ci and boot is not None and np.any(np.isfinite(boot.lower)):
        f_defaults = dict(alpha=0.2, color="dodgerblue",
                          label=f"{boot.conf:.0%} CI")
        ax.fill_between(
            boot.area_grid, boot.lower, boot.upper,
            **{**f_defaults, **(fill_kw or {})},
        )

    ax.set_xlabel("Area")
    ax.set_ylabel("Species")
    ax.set_title("Model-averaged SAR")
    ax.legend()
    return ax


def plot_residuals(
    fit: SARFit,
    ax: plt.Axes | None = None,
    scatter_kw: dict | None = None,
) -> plt.Axes:
    """Plot residuals vs fitted values for a SAR model fit.

    Parameters
    ----------
    fit : SARFit
        A fitted SAR model.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    scatter_kw : dict, optional
        Extra keyword arguments passed to ``ax.scatter()`` for the
        residual points.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    area = fit.data["area"].values
    observed = fit.data["species"].values
    fitted = fit.predict(area)
    residuals = observed - fitted

    s_defaults = dict(color="black")
    ax.scatter(fitted, residuals, **{**s_defaults, **(scatter_kw or {})})
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals: {fit.model}")
    return ax
