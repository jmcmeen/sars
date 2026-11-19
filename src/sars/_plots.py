"""Plotting functions for SAR fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from sars._inference import AveragedSAR, MultiSARFit
from sars._models import SARFit
from sars._threshold import _fitted_data

if TYPE_CHECKING:
    from sars._inference import BootstrappedCI
    from sars._threshold import ThresholdCI, ThresholdFit


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

    area = fit.data["area"].to_numpy(dtype=float)
    species = fit.data["species"].to_numpy(dtype=float)

    s_defaults = {"color": "black", "zorder": 3, "label": "Observed"}
    ax.scatter(area, species, **{**s_defaults, **(scatter_kw or {})})

    a_grid = np.linspace(area.min(), area.max(), 200)
    pred = fit.predict(a_grid)
    l_defaults = {
        "color": "dodgerblue",
        "linewidth": 2,
        "label": f"{fit.model} (R²={fit.r_squared:.3f})",
    }
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
    area = data["area"].to_numpy(dtype=float)
    species = data["species"].to_numpy(dtype=float)

    s_defaults = {"color": "black", "zorder": 3, "label": "Observed"}
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
        w = multi_fit.summary.loc[multi_fit.summary["model"] == name, "weight"].iloc[0]
        l_defaults = {"color": colors[i], "linewidth": 1.5, "label": f"{name} (w={w:.3f})"}
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
    area = data["area"].to_numpy(dtype=float)
    species = data["species"].to_numpy(dtype=float)

    s_defaults = {"color": "black", "zorder": 3, "label": "Observed"}
    ax.scatter(area, species, **{**s_defaults, **(scatter_kw or {})})

    a_grid = np.linspace(area.min(), area.max(), 200)
    pred = averaged.predict(a_grid)
    l_defaults = {"color": "dodgerblue", "linewidth": 2, "label": "Model-averaged"}
    ax.plot(a_grid, pred, **{**l_defaults, **(line_kw or {})})

    if ci and boot is not None and np.any(np.isfinite(boot.lower)):
        f_defaults = {"alpha": 0.2, "color": "dodgerblue", "label": f"{boot.conf:.0%} CI"}
        ax.fill_between(
            boot.area_grid,
            boot.lower,
            boot.upper,
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

    area = fit.data["area"].to_numpy(dtype=float)
    observed = fit.data["species"].to_numpy(dtype=float)
    fitted = fit.predict(area)
    residuals = observed - fitted

    s_defaults = {"color": "black"}
    ax.scatter(fitted, residuals, **{**s_defaults, **(scatter_kw or {})})
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals: {fit.model}")
    return ax


def plot_threshold(
    fit: ThresholdFit,
    model: str | None = None,
    ci: ThresholdCI | None = None,
    ax: plt.Axes | None = None,
    scatter_kw: dict | None = None,
    line_kw: dict | None = None,
) -> plt.Axes:
    """Plot a threshold model: the data, its fitted segments and its breakpoints.

    The x axis is the scale the model was fitted on: the natural log of area
    by default, area itself for a fit made with ``log_area=False``. A
    continuous model is drawn as one bending line; a discontinuous model as a
    separate line for each segment, with nothing drawn across a jump: a
    breakpoint belongs to the segment on its left, so the next segment starts
    at the first area above it. A dashed vertical line marks each breakpoint,
    and with ``ci`` each breakpoint's confidence interval is shaded.

    Parameters
    ----------
    fit : ThresholdFit
        Result of ``sar_threshold()``.
    model : str, optional
        Which of the fit's models to draw. Default: the model ``ci`` is for
        when ``ci`` is given, the best model by AICc otherwise.
    ci : ThresholdCI, optional
        Breakpoint confidence intervals from ``threshold_ci()`` for the same
        fit and model.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    scatter_kw : dict, optional
        Extra keyword arguments passed to ``ax.scatter()`` for the
        observed data points.
    line_kw : dict, optional
        Extra keyword arguments passed to ``ax.plot()`` for the fitted
        segments.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``model`` isn't among the fit's models, if ``ci`` is for another
        model, scale or fit than the one drawn, or if the fit's data have
        changed since it was fitted.
    """
    if model is None:
        model = ci.model if ci is not None else fit.best_model
    if model not in fit.all_fits:
        raise ValueError(
            f"{model} is not among this fit's models ({', '.join(fit.all_fits)}); "
            f"include it in sar_threshold's models."
        )
    if ci is not None and ci.model != model:
        raise ValueError(f"ci is for {ci.model}, not {model}.")
    if ci is not None and ci.log_area != fit.log_area:
        raise ValueError("ci and fit are on different scales (one on ln(area), one on area).")
    fitted = fit.all_fits[model]
    if ci is not None and not (
        len(ci.estimates) == len(fitted["breakpoints"])
        and np.allclose(ci.estimates, fitted["breakpoints"])
    ):
        raise ValueError(f"ci is for another fit: its breakpoints aren't this fit's for {model}.")
    _, x, species = _fitted_data(fit, model)
    if ax is None:
        _, ax = plt.subplots()

    s_defaults = {"color": "black", "zorder": 3, "label": "Observed"}
    ax.scatter(x, species, **{**s_defaults, **(scatter_kw or {})})

    l_defaults = {
        "color": "dodgerblue",
        "linewidth": 2,
        "label": f"{model} (R²={fitted['r2']:.3f})",
    }
    line = {**l_defaults, **(line_kw or {})}
    # Each segment is a straight line, so its two ends draw it. Continuous
    # segments share their ends and join up. A discontinuous model's breakpoint
    # is the last area of the segment on its left, so the next segment is drawn
    # from the first area above it, not back across the gap it never fitted.
    for i, seg in enumerate(fitted["segments"]):
        start = seg["x_min"]
        if i > 0 and model.startswith("Disc"):
            start = x[x > start].min()
        ends = np.array([start, seg["x_max"]])
        if i > 0:  # one legend entry for the whole model
            line = {**line, "label": "_nolegend_"}
        ax.plot(ends, seg["intercept"] + seg["slope"] * ends, **line)

    for i, t in enumerate(fitted["breakpoints"]):
        ax.axvline(
            t,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Breakpoint" if i == 0 else "_nolegend_",
        )
    if ci is not None:
        for i, (lo, hi) in enumerate(zip(ci.lower, ci.upper, strict=True)):
            ax.axvspan(
                lo,
                hi,
                alpha=0.15,
                color="red",
                linewidth=0,
                label=f"{ci.conf * 100:g}% CI" if i == 0 else "_nolegend_",
            )

    ax.set_xlabel("ln(Area)" if fit.log_area else "Area")
    ax.set_ylabel("Species")
    ax.set_title(f"Threshold model: {model}")
    ax.legend()
    return ax
