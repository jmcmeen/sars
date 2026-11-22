"""Species expected to be lost with habitat, from a model-averaged curve."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sars._inference import AveragedSAR, BootstrappedCI, bootstrap_ci, sar_average

# More than this share of resamples with a negative loss is worth a warning.
_NEGATIVE_SHARE = 0.01


@dataclass
class HabitatLoss:
    """Species expected to be lost when habitat is lost.

    Attributes
    ----------
    table : pd.DataFrame
        One row per scenario: ``area`` and ``remaining`` (the area before and
        after the loss), ``richness_before`` and ``richness_after`` (the
        averaged curve at each), ``species_lost`` and ``proportion_lost``, and
        their bootstrap intervals (``lost_lower``, ``lost_upper``,
        ``proportion_lower``, ``proportion_upper``; NaN without a bootstrap).
    average : AveragedSAR
        The model-averaged fit the point estimates come from.
    ci : BootstrappedCI or None
        The bootstrap the intervals come from, at the scenarios' distinct
        areas (``ci.area_grid``), with each resample's predictions in
        ``ci.samples`` if ``keep_samples`` was asked for; None when ``n_boot``
        was 0.
    conf : float
        The intervals' confidence level.
    """

    table: pd.DataFrame = field(repr=False)
    average: AveragedSAR = field(repr=False)
    ci: BootstrappedCI | None = field(default=None, repr=False)
    conf: float = 0.95

    def __repr__(self) -> str:
        n = len(self.table)
        if n == 0:
            return "HabitatLoss(no scenarios)"
        first = self.table.iloc[0]
        return (
            f"HabitatLoss({n} scenario{'s' if n != 1 else ''}, first: "
            f"{first['species_lost']:.3g} species lost of {first['richness_before']:.3g})"
        )


def _scenarios(values: float | ArrayLike, name: str) -> np.ndarray:
    """One value or a list of them, as a 1-D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        raise ValueError(f"{name} must be one value or a list of them, not a {array.ndim}-D array.")
    array = np.atleast_1d(array)
    if array.size == 0:
        raise ValueError(f"{name} is empty: give at least one scenario.")
    return array


def _broadcast(a: np.ndarray, b: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        x, y = np.broadcast_arrays(a, b)
    except ValueError:
        raise ValueError(
            f"area ({len(a)} values) and {name} ({len(b)} values) don't line up: give one "
            "value for either, or the same number of each."
        ) from None
    return x.astype(float).copy(), y.astype(float).copy()


def habitat_loss(
    data: pd.DataFrame,
    area: float | ArrayLike,
    remaining: float | ArrayLike | None = None,
    fraction_lost: float | ArrayLike | None = None,
    models: str | list[str] = "all",
    ic: str = "AICc",
    n_boot: int = 1000,
    conf: float = 0.95,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
    keep_samples: bool = False,
) -> HabitatLoss:
    """Species expected to be lost when an area loses some of its habitat.

    Fits the models, averages them (as ``sar_average``), and compares the
    averaged curve's richness at the area before and after the loss. The
    interval for the loss comes from a bootstrap (as ``bootstrap_ci``): each
    resample's own averaged curve gives its own loss, so the interval is for
    the difference itself, not two separate predictions.

    What the number means, and doesn't: it is the eventual loss once the
    remaining area holds the richness an island of its size would, not the
    loss right away (species can persist for a long time before dying out,
    the "extinction debt"). It treats what remains as one piece of habitat,
    not fragments. And it tends to overstate the loss: a curve fitted across
    islands of different sizes describes how many species an area holds, not
    how many disappear when part of it goes, and the second is smaller (He
    and Hubbell 2011, *Nature* 473, 368-371). Read it as an upper estimate.

    The intervals resample islands, so they share ``bootstrap_ci``'s limits.
    At the edge of the data, the resamples that leave out the largest (or
    smallest) island extrapolate, and their predictions there can run away
    (``bootstrap_ci``'s warnings say so). And averaging all 20 models brings in curves that fall with
    area in places, so a few resamples can show a negative loss (also
    warned). Averaging a set of models whose curves only rise avoids that.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species', in the same units as
        ``area`` and ``remaining``.
    area : float or array-like
        The area before the loss: one value, or one per scenario.
    remaining : float or array-like, optional
        The area left after the loss.
    fraction_lost : float or array-like, optional
        Instead of ``remaining``: the share of ``area`` lost, from 0 to 1.
        Give exactly one of the two. Either broadcasts against ``area``.
    models : str or list[str]
        Models to average, or "all" (the default).
    ic : str
        Criterion for the weights: "AICc" (the default), "AIC" or "BIC".
    n_boot : int
        Bootstrap resamples for the intervals (1000 by default); 0 for no
        intervals.
    conf : float
        Confidence level of the intervals (0.95 by default).
    rng : np.random.Generator, optional
        Random generator for the resamples, for reproducible intervals.
    n_jobs : int
        Processes to fit the resamples in; -1 for one per core.
    keep_samples : bool
        Keep each resample's predictions at the distinct areas in
        ``ci.samples`` (False by default: the result, saved, stays small).

    Returns
    -------
    HabitatLoss

    Raises
    ------
    ValueError
        If neither or both of ``remaining`` and ``fraction_lost`` are given,
        an area or fraction is out of range, the scenarios don't line up, or
        ``n_boot`` or ``conf`` is invalid.

    Warns
    -----
    UserWarning
        If an area lies outside the observed range: the curve is then
        extrapolated, and models that agree inside the range can disagree
        widely outside it.

    Examples
    --------
    >>> loss = sars.habitat_loss(sars.load_gala(), area=900, fraction_lost=0.5, n_boot=200)
    >>> loss.table[["species_lost", "lost_lower", "lost_upper"]]  # doctest: +SKIP
    """
    if (remaining is None) == (fraction_lost is None):
        raise ValueError("Give exactly one of remaining and fraction_lost.")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise ValueError(f"n_boot must be a whole number, 0 or more, got {n_boot!r}.")
    if not 0 < conf < 1:
        raise ValueError(f"conf must be between 0 and 1, got {conf!r}.")
    before = _scenarios(area, "area")
    if remaining is not None:
        after = _scenarios(remaining, "remaining")
        before, after = _broadcast(before, after, "remaining")
    else:
        assert fraction_lost is not None  # exactly one of the two, checked above
        fraction = _scenarios(fraction_lost, "fraction_lost")
        if not np.all((fraction >= 0) & (fraction <= 1)):
            raise ValueError(f"fraction_lost must be between 0 and 1, got {fraction}.")
        before, fraction = _broadcast(before, fraction, "fraction_lost")
        after = before * (1.0 - fraction)
    if not (np.all(np.isfinite(before)) and np.all(before > 0)):
        raise ValueError(f"area must be positive, got {before}.")
    if not (np.all(np.isfinite(after)) and np.all(after > 0)):
        raise ValueError(
            "The remaining area must be positive: the curves say nothing about an area of "
            f"zero. Got {after}."
        )
    if np.any(after > before):
        raise ValueError("The remaining area can't be larger than the area before the loss.")

    lo, hi = float(data["area"].min()), float(data["area"].max())
    outside = sorted({float(a) for a in np.concatenate([before, after]) if a < lo or a > hi})
    if outside:
        shown = ", ".join(f"{a:.4g}" for a in outside[:4]) + (", ..." if len(outside) > 4 else "")
        warnings.warn(
            f"Area{'s' if len(outside) > 1 else ''} {shown} outside the observed range "
            f"({lo:.4g} to {hi:.4g}): the curve is extrapolated there, and models that agree "
            "inside the range can disagree widely outside it.",
            stacklevel=2,
        )

    # The average and the bootstrap each fit the data and warn about it; their
    # warnings are collected, said once each, and pointed at the caller.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        average = sar_average(data, models=models, ic=ic)
        grid, where = np.unique(np.concatenate([before, after]), return_inverse=True)
        ci = None
        if n_boot > 0:
            ci = bootstrap_ci(
                data,
                models=models,
                n_boot=n_boot,
                conf=conf,
                area_grid=grid,
                rng=rng,
                ic=ic,
                n_jobs=n_jobs,
                keep_samples=True,
            )
    said: set[str] = set()
    for w in caught:
        if str(w.message) not in said:
            said.add(str(w.message))
            warnings.warn(
                w.message, w.category if isinstance(w.category, type) else UserWarning, stacklevel=2
            )

    rich_before = np.asarray(average.predict(before), dtype=float)
    rich_after = np.asarray(average.predict(after), dtype=float)
    lost = rich_before - rich_after
    with np.errstate(divide="ignore", invalid="ignore"):
        proportion = np.where(rich_before > 0, lost / rich_before, np.nan)

    k = len(before)
    nan = np.full(k, np.nan)
    lost_lower, lost_upper, prop_lower, prop_upper = nan.copy(), nan.copy(), nan.copy(), nan.copy()
    if ci is not None and ci.samples.size:
        s_before, s_after = ci.samples[:, where[:k]], ci.samples[:, where[k:]]
        # A resample with no species at the area before has no proportion to
        # lose; it is left out of both intervals, so they describe the same
        # resamples.
        usable = s_before > 0
        boot_lost = np.where(usable, s_before - s_after, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot_prop = np.where(usable, boot_lost / s_before, np.nan)
        alpha = 1.0 - conf
        q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a column with none usable
            lost_lower, lost_upper = np.nanpercentile(boot_lost, q, axis=0)
            prop_lower, prop_upper = np.nanpercentile(boot_prop, q, axis=0)
        negative = np.nanmean(np.where(usable, boot_lost < 0, np.nan), axis=0)
        if np.any(negative > _NEGATIVE_SHARE):
            below = bool(np.any(lost_lower < 0))
            warnings.warn(
                f"In up to {np.nanmax(negative):.0%} of the bootstrap resamples the averaged "
                "curve has fewer species in the remaining area than in the whole: some "
                "averaged models' curves fall with area"
                + (", and the interval's lower bound is negative" if below else "")
                + ". Average models whose curves only rise (models=[...]).",
                stacklevel=2,
            )
        if not keep_samples:
            ci = replace(ci, samples=np.empty((0, 0)))

    table = pd.DataFrame(
        {
            "area": before,
            "remaining": after,
            "richness_before": rich_before,
            "richness_after": rich_after,
            "species_lost": lost,
            "lost_lower": lost_lower,
            "lost_upper": lost_upper,
            "proportion_lost": proportion,
            "proportion_lower": prop_lower,
            "proportion_upper": prop_upper,
        }
    )
    return HabitatLoss(table=table, average=average, ci=ci, conf=conf)
