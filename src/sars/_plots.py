"""Plotting functions for SAR fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from sars._coleman import ColemanFit
from sars._inference import AveragedSAR, MultiSARFit
from sars._linpow import LinPowFit
from sars._models import SARFit, _area_grid
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

    a_grid = _area_grid(float(area.min()), float(area.max()), 200)
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
        Number of top models (by AICc) to display. Fewer are drawn when the
        fit holds fewer, and the title gives the number drawn.
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

    a_grid = _area_grid(float(area.min()), float(area.max()), 200)

    # The top N models by AICc, as many as there are: a set of fewer models,
    # or fewer that converged, draws fewer curves.
    fits_by_name = {f.model: f for f in multi_fit.fits}
    drawn = list(
        dict.fromkeys(  # a model asked for twice is drawn once
            name for name in multi_fit.summary.head(top_n)["model"] if name in fits_by_name
        )
    )

    # A qualitative map, one colour per curve drawn, in order. tab20 pairs a dark
    # and a light shade of each hue, so take the dark shades first, then the light.
    if len(drawn) > 10:
        cmap = plt.cm.tab20
        colour_index = [(2 * i) % 20 + (2 * i) // 20 for i in range(len(drawn))]
    else:
        cmap = plt.cm.tab10
        colour_index = list(range(len(drawn)))

    extra_line = line_kw or {}
    for i, name in enumerate(drawn):
        pred = fits_by_name[name].predict(a_grid)
        w = multi_fit.summary.loc[multi_fit.summary["model"] == name, "weight"].iloc[0]
        l_defaults = {
            "color": cmap(colour_index[i]),
            "linewidth": 1.5,
            "label": f"{name} (w={w:.3f})",
        }
        ax.plot(a_grid, pred, **{**l_defaults, **extra_line})

    ax.set_xlabel("Area")
    ax.set_ylabel("Species")
    if not multi_fit.fits:
        ax.set_title("No fitted SAR models")
    else:
        ax.set_title(f"Top {len(drawn)} SAR model{'s' if len(drawn) != 1 else ''}")
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

    a_grid = _area_grid(float(area.min()), float(area.max()), 200)
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


def plot_lin_pow(
    fit: LinPowFit,
    ax: plt.Axes | None = None,
    scatter_kw: dict | None = None,
    line_kw: dict | None = None,
) -> plt.Axes:
    """Plot a log-log power model fit: the data and the fitted line on log-log axes.

    Parameters
    ----------
    fit : LinPowFit
        A fitted log-log power model.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    scatter_kw : dict, optional
        Extra keyword arguments passed to ``ax.scatter()`` for the
        observed data points.
    line_kw : dict, optional
        Extra keyword arguments passed to ``ax.plot()`` for the fitted line.

    Returns
    -------
    matplotlib.axes.Axes
        The axes are logarithmic and labelled in area and richness units,
        whatever the fit's log base. When a constant was added to the richness
        (``fit.con``), the points and the line are of richness plus that
        constant, the values the line was fitted to.
    """
    if ax is None:
        _, ax = plt.subplots()

    area = fit.data["area"].to_numpy(dtype=float)
    species = fit.data["species"].to_numpy(dtype=float) + fit.con

    s_defaults = {"color": "black", "zorder": 3, "label": "Observed"}
    ax.scatter(area, species, **{**s_defaults, **(scatter_kw or {})})

    # A power law is straight on log-log axes, so its two ends draw it.
    ends = np.array([area.min(), area.max()])
    l_defaults = {
        "color": "dodgerblue",
        "linewidth": 2,
        "label": f"z={fit.z:.3f} (R²={fit.r_squared:.3f})",
    }
    ax.plot(ends, fit.predict(ends) + fit.con, **{**l_defaults, **(line_kw or {})})

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Area")
    ax.set_ylabel(f"Species + {fit.con:g}" if fit.con else "Species")
    ax.set_title("Log-log power model")
    ax.legend()
    return ax


def plot_coleman(
    fit: ColemanFit,
    ax: plt.Axes | None = None,
    log: bool = True,
    scatter_kw: dict | None = None,
    line_kw: dict | None = None,
    fill_kw: dict | None = None,
) -> plt.Axes:
    """Plot Coleman's random placement model against observed richness.

    Draws the expected richness against relative area (a site's share of the
    total area), a band one standard deviation either side of it, and the
    observed richness of each site. Random placement is rejected when more
    than a third of the points fall outside the band.

    Parameters
    ----------
    fit : ColemanFit
        Result of ``coleman()``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    log : bool
        If True (the default), put relative area on a log axis, the usual
        way to draw this plot. Richness stays on a linear axis.
    scatter_kw : dict, optional
        Extra keyword arguments passed to ``ax.scatter()`` for the
        observed data points.
    line_kw : dict, optional
        Extra keyword arguments passed to ``ax.plot()`` for the expected
        curve.
    fill_kw : dict, optional
        Extra keyword arguments passed to ``ax.fill_between()`` for the
        band of one standard deviation.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    alpha = fit.sites["relative_area"].to_numpy(dtype=float)
    observed = fit.sites["observed"].to_numpy(dtype=float)

    s_defaults = {"color": "black", "zorder": 3, "label": "Observed"}
    ax.scatter(alpha, observed, **{**s_defaults, **(scatter_kw or {})})

    lo, hi = float(alpha.min()), float(alpha.max())
    if np.isclose(lo, hi):
        # Sites of one size (a common design) give a single relative area and
        # no range to draw over, so show the curve a factor of ten either side
        # of it, stopping at the whole area.
        lo, hi = lo / 10, min(1.0, hi * 10)
    # Evenly spaced on the axis in use, so the curve is smooth at the small
    # sites on a log axis too.
    grid = np.geomspace(lo, hi, 200) if log else np.linspace(lo, hi, 200)
    # The grid's top can round a hair above 1; _curve allows for that.
    expected, sd = fit._curve(grid * fit.total_area)

    l_defaults = {"color": "dodgerblue", "linewidth": 2, "label": "Random placement"}
    ax.plot(grid, expected, **{**l_defaults, **(line_kw or {})})
    f_defaults = {"alpha": 0.2, "color": "dodgerblue", "label": "±1 sd"}
    ax.fill_between(grid, expected - sd, expected + sd, **{**f_defaults, **(fill_kw or {})})

    if log:
        ax.set_xscale("log")

    ax.set_xlabel("Relative area")
    ax.set_ylabel("Species")
    verdict = "rejected" if fit.rejected else "not rejected"
    ax.set_title(f"Coleman's random placement: {verdict}")
    ax.legend()
    return ax
