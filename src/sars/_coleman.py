"""Coleman's random placement model.

Coleman (1981), Mathematical Biosciences 54, 191-215, and Coleman et al.
(1982), Ecology 63, 1121-1133. If the individuals of every species are
scattered at random over the sites, without regard to each other, a site
holding a fraction ``alpha`` of the total area misses a species with ``n``
individuals with probability ``(1 - alpha)^n``. The richness expected at that
site and its variance follow from the species' total abundances alone:

    S(alpha)       = sum_i [1 - (1 - alpha)^n_i]
    sigma^2(alpha) = sum_i (1 - alpha)^n_i - sum_i (1 - alpha)^(2 n_i)

Nothing is fitted: the model has no free parameters. It is a null model, and
the question is whether the observed richness of the sites stays close to the
expected curve. The usual rule (Coleman et al. 1982; Wang et al. 2010,
Diversity and Distributions 16, 862-873) rejects random placement when more
than a third of the sites lie more than one standard deviation from it. That
rule is rough: with few sites it often rejects data placed at random.
``ColemanFit.placement_test`` gives a calibrated alternative, a p value from
simulating random placement of the observed species totals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# Random placement is rejected when more than this share of the sites lies
# further than one standard deviation from the expected curve.
REJECTION_FRACTION = 1 / 3

# How far above the total area an area may be and still count as the total:
# room for rounding in a sum, nothing more.
_AREA_RTOL = 1e-9


def _is_labelled(index: pd.Index) -> bool:
    """Whether an index carries labels someone chose, rather than the 0, 1, ...
    pandas gives when none are supplied."""
    return not (isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1)


def _counts_array(abundance: pd.DataFrame | ArrayLike) -> np.ndarray:
    """The abundance table as a 2-D float array, or a ValueError saying what is
    wrong with it. Missing values are looked for before anything else, so a
    table with gaps is reported as that, whatever type pandas gave it."""
    not_numbers = "abundance must hold numbers only (species in rows, sites in columns)."
    try:
        raw = (
            abundance.to_numpy(dtype=object)
            if isinstance(abundance, pd.DataFrame)
            else np.asarray(abundance)
        )
    except (TypeError, ValueError) as exc:
        # numpy refuses rows of unequal length.
        raise ValueError(
            "abundance must be a rectangular table (species in rows, sites in columns)."
        ) from exc
    if raw.ndim != 2:
        raise ValueError(
            f"abundance must be two-dimensional (species in rows, sites in columns), "
            f"got {raw.ndim} dimension(s)."
        )
    if raw.dtype.kind == "O" and pd.isna(raw).any():
        raise ValueError("abundance holds missing or non-finite values.")
    # Text is refused even when it spells a number: float() would read "3" as a
    # count, and a column of strings usually means the table was read wrongly.
    if raw.dtype.kind in "USO" and any(isinstance(v, (str, bytes)) for v in raw.ravel()):
        raise ValueError(not_numbers)
    try:
        counts = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(not_numbers) from exc
    if not np.all(np.isfinite(counts)):
        raise ValueError("abundance holds missing or non-finite values.")
    return counts


def _site_areas(area: ArrayLike, sites: pd.Index | None, n_sites: int) -> np.ndarray:
    """Site areas in the order of the abundance table's columns. ``sites`` is
    the table's column labels, or None when it has none."""
    if (
        isinstance(area, pd.Series)
        and sites is not None
        and _is_labelled(sites)
        and _is_labelled(area.index)
        and len(area) == n_sites
        and not area.index.equals(sites)
    ):
        # Both sides name their sites, in different orders or with different
        # names. Matching by position would quietly give sites the wrong areas.
        same_labels = (
            sites.is_unique and area.index.is_unique and len(sites.difference(area.index)) == 0
        )
        if not same_labels:
            raise ValueError(
                "area's labels don't match abundance's column labels. Give area one "
                "value per column under the same labels, or pass it without labels "
                "(a list or array) to match by position."
            )
        area = area.reindex(sites)
    areas = np.asarray(area, dtype=float)
    if areas.ndim != 1 or len(areas) != n_sites:
        raise ValueError(
            f"area must have one value per site: abundance has {n_sites} columns, "
            f"area has shape {areas.shape}."
        )
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("Site areas must be positive and finite.")
    return areas


def _expected_and_sd(alpha: np.ndarray, abundance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected richness and its standard deviation at relative areas ``alpha``,
    for species with total abundances ``abundance``."""
    # One row per relative area, one column per species: the chance that a
    # species is absent from a site of that relative area.
    absent = np.power(1.0 - alpha[:, None], abundance[None, :])
    expected = (1.0 - absent).sum(axis=1)
    # absent - absent^2, written as a product so rounding can't make a term
    # (and so the variance) negative.
    variance = (absent * (1.0 - absent)).sum(axis=1)
    return expected, np.sqrt(variance)


def _distance(richness: np.ndarray, expected: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Sum over sites (the last axis) of the squared distance of richness from
    the expected value, in standard deviations. A site whose richness is fixed
    under random placement (sd 0: it holds all the area, or none of any
    species) contributes nothing if it matches and makes the distance infinite
    if it doesn't, since random placement can't produce it."""
    gap = richness - expected
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(sd > 0, (gap / np.where(sd > 0, sd, 1.0)) ** 2, 0.0)
    fixed_off = (sd == 0) & (np.abs(gap) > 1e-9)
    z2 = np.where(fixed_off, np.inf, z2)
    return z2.sum(axis=-1)


@dataclass(frozen=True)
class ColemanTest:
    """A simulation test of Coleman's random placement model
    (``ColemanFit.placement_test``).

    Attributes
    ----------
    statistic : float
        Sum over sites of the squared distance of observed from expected
        richness, in standard deviations.
    p_value : float
        Share of the simulated random placements, the data counted among them,
        whose statistic is at least the data's. A small value is evidence
        against random placement.
    n_sim : int
        Number of simulated placements.
    rule_rejection_rate : float
        Share of the simulated random placements that the one-third rule
        rejects: how often that rule would reject random placement for these
        species totals and site areas when it holds.
    """

    statistic: float
    p_value: float
    n_sim: int
    rule_rejection_rate: float

    def __repr__(self) -> str:
        return (
            f"ColemanTest(statistic={self.statistic:.3g}, p_value={self.p_value:.3g}, "
            f"n_sim={self.n_sim}, rule_rejection_rate={self.rule_rejection_rate:.0%})"
        )


@dataclass(eq=False)
class ColemanFit:
    """Coleman's random placement model compared with observed richness.

    Attributes
    ----------
    sites : pd.DataFrame
        One row per site, in the order given, indexed by site label (the
        abundance table's column labels, or 0, 1, ... for an array). Columns:
        ``area``, ``relative_area`` (area over total area), ``observed``
        (species present at the site), ``expected`` (richness under random
        placement), ``sd`` (its standard deviation) and ``within_sd`` (whether
        ``observed`` lies within one ``sd`` of ``expected``, bounds included).
    abundance : pd.Series
        Total abundance of each species over all sites, indexed by species
        label (the abundance table's row labels, or row positions for an
        array). Species absent from every site are left out.
    total_area : float
        Sum of the site areas.

    Notes
    -----
    Two fits are equal only if they are the same object: the tables they hold
    have no single true-or-false comparison.
    """

    sites: pd.DataFrame = field(repr=False)
    abundance: pd.Series = field(repr=False)
    total_area: float

    @property
    def fraction_outside(self) -> float:
        """Share of the sites whose observed richness lies more than one
        standard deviation from the expected value."""
        return float((~self.sites["within_sd"].to_numpy(dtype=bool)).mean())

    @property
    def n_sites(self) -> int:
        """Number of sites."""
        return len(self.sites)

    @property
    def n_species(self) -> int:
        """Number of species present at one site or more."""
        return len(self.abundance)

    @property
    def rejected(self) -> bool:
        """Whether random placement is rejected: True when more than a third of
        the sites lie more than one standard deviation from the expected curve.

        This is the conventional rule, and a rough one: about a third of the
        sites are expected outside one standard deviation when placement is
        random, so with few sites it often rejects data that are."""
        return bool(self.fraction_outside > REJECTION_FRACTION)

    def _curve(self, area: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Expected richness and its standard deviation at the given area(s),
        each with the shape of ``area``."""
        area = np.asarray(area, dtype=float)
        # A total passed back in (say, the sum of the site areas taken in
        # another order) can differ from total_area in the last digit, so the
        # upper bound has a little give and the share is clipped to 1.
        too_large = area > self.total_area * (1 + _AREA_RTOL)
        if np.any(~np.isfinite(area)) or np.any(area < 0) or np.any(too_large):
            raise ValueError(
                f"Areas must lie between 0 and the total area ({self.total_area:g}): "
                "the model describes a part of the area the species were counted over."
            )
        alpha = np.clip(area / self.total_area, 0.0, 1.0)
        expected, sd = _expected_and_sd(alpha.ravel(), self.abundance.to_numpy(dtype=float))
        return expected.reshape(alpha.shape), sd.reshape(alpha.shape)

    def predict(self, area: ArrayLike) -> np.ndarray:
        """Expected species richness of a site of the given area(s).

        Parameters
        ----------
        area : float or array-like
            Area value(s), in the units of the site areas, between 0 and
            ``total_area``.

        Returns
        -------
        np.ndarray
            Expected richness under random placement, with the shape of ``area``.

        Raises
        ------
        ValueError
            If an area is negative, not finite, or larger than ``total_area``.
        """
        return self._curve(area)[0]

    def predict_sd(self, area: ArrayLike) -> np.ndarray:
        """Standard deviation of the richness of a site of the given area(s).

        Parameters
        ----------
        area : float or array-like
            Area value(s), in the units of the site areas, between 0 and
            ``total_area``.

        Returns
        -------
        np.ndarray
            Standard deviation of richness under random placement, with the
            shape of ``area``.

        Raises
        ------
        ValueError
            If an area is negative, not finite, or larger than ``total_area``.
        """
        return self._curve(area)[1]

    def placement_test(
        self, n_sim: int = 999, rng: np.random.Generator | None = None
    ) -> ColemanTest:
        """A calibrated test of random placement, by simulation.

        The one-third rule (``rejected``) has no stated error rate, and with
        few sites it often rejects data placed at random. This test simulates
        random placement itself: each species' individuals, as many as were
        counted, are scattered over the sites in proportion to their areas,
        ``n_sim`` times. The statistic is the sum over sites of the squared
        distance of observed from expected richness, in standard deviations,
        and the p value is the share of simulations at least as far from the
        curve as the data, counting the data as one of them.

        Parameters
        ----------
        n_sim : int, default 999
            Number of simulated placements. The smallest p value possible is
            ``1 / (n_sim + 1)``.
        rng : numpy.random.Generator or int, optional
            Random number generator, or a seed for one, for a reproducible
            result.

        Returns
        -------
        ColemanTest
            The statistic, its p value, and how often the one-third rule
            rejected the simulated random placements: that rule's false
            rejection rate for these species totals and site areas.

        Raises
        ------
        ValueError
            If ``n_sim`` is not a positive whole number.
        """
        if isinstance(n_sim, bool) or not isinstance(n_sim, (int, np.integer)) or n_sim < 1:
            raise ValueError(f"n_sim must be a positive whole number, got {n_sim!r}")
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)  # None, or a seed
        alpha = self.sites["relative_area"].to_numpy(dtype=float)
        alpha = alpha / alpha.sum()  # the sites' shares sum to 1 exactly
        expected = self.sites["expected"].to_numpy(dtype=float)
        sd = self.sites["sd"].to_numpy(dtype=float)
        observed = self.sites["observed"].to_numpy(dtype=float)

        # Richness of every site in every simulated placement: each species
        # is present at a site that receives one of its individuals or more.
        richness = np.zeros((int(n_sim), len(alpha)))
        for total in self.abundance.to_numpy(dtype=np.int64):
            richness += rng.multinomial(total, alpha, size=int(n_sim)) > 0

        statistic = _distance(observed, expected, sd)
        simulated = _distance(richness, expected, sd)
        # A simulation that ties with the data counts as at least as far off;
        # the slack keeps rounding from deciding a tie.
        slack = 1e-9 * max(1.0, abs(float(statistic))) if np.isfinite(statistic) else 0.0
        p_value = (1 + int(np.sum(simulated >= statistic - slack))) / (int(n_sim) + 1)
        outside = np.abs(richness - expected) > sd
        rule_rate = float(np.mean(outside.mean(axis=1) > REJECTION_FRACTION))
        return ColemanTest(
            statistic=float(statistic),
            p_value=p_value,
            n_sim=int(n_sim),
            rule_rejection_rate=rule_rate,
        )

    def __repr__(self) -> str:
        outside = int((~self.sites["within_sd"]).sum())
        verdict = "rejected" if self.rejected else "not rejected"
        return (
            f"ColemanFit({self.n_species} species, {self.n_sites} sites, "
            f"{outside} outside ±1 sd ({self.fraction_outside:.0%}), "
            f"random placement {verdict})"
        )


def coleman(abundance: pd.DataFrame | ArrayLike, area: ArrayLike) -> ColemanFit:
    """Compare site richness with Coleman's random placement model.

    Under random placement every individual lands on a site with probability
    equal to the site's share of the total area, independently of the others.
    A species with ``n`` individuals is then absent from a site of relative
    area ``alpha`` with probability ``(1 - alpha)^n``, which gives the expected
    richness ``S(alpha) = sum_i [1 - (1 - alpha)^n_i]`` and the variance
    ``sum_i (1 - alpha)^n_i - sum_i (1 - alpha)^(2 n_i)`` (Coleman 1981). The
    model has no parameters to estimate.

    Parameters
    ----------
    abundance : pd.DataFrame or 2-D array-like
        Species-by-site abundance table: **one row per species, one column per
        site**, each value the number of individuals of that species counted at
        that site. This is the orientation R ``sars``'s ``coleman(data, area)``
        takes. A table with sites in rows needs transposing first (``df.T``).
        Counts must be whole numbers, zero or more. Presence-absence data
        won't do: the model needs the numbers of individuals.
    area : array-like
        Area of each site; all must be positive. A list or array is matched
        to ``abundance``'s columns by position. A ``pd.Series`` with its own
        labels, given with a DataFrame whose columns are labelled, is matched
        by label instead, whatever its order, and must hold exactly the
        column labels. A Series or DataFrame with the default 0, 1, ...
        labels counts as unlabelled and is matched by position.

    Returns
    -------
    ColemanFit
        The per-site table (``sites``), the share of sites more than one
        standard deviation from the expected curve (``fraction_outside``) and
        the verdict (``rejected``: True when that share is over a third).

    Raises
    ------
    ValueError
        If ``abundance`` isn't two-dimensional, holds a missing, negative or
        fractional value or text (numbers stored as strings included), or
        holds no individuals at all; if there are fewer than two sites; if
        ``area`` doesn't have one positive, finite value per site; or if a
        labelled ``area`` doesn't carry ``abundance``'s column labels.

    Warns
    -----
    UserWarning
        If no value in ``abundance`` is above 1, which is what
        presence-absence data looks like.

    Notes
    -----
    Species absent from every site are dropped. They add nothing to the
    expected richness or to its variance (``(1 - alpha)^0 = 1``), so the
    result is the same with or without them; ``ColemanFit.n_species`` counts
    only the species that were seen.

    A site counts as within one standard deviation when
    ``|observed - expected| <= sd``.

    The one-third rule is a convention, not a test with a known error rate.
    One standard deviation either side holds roughly two thirds of the spread,
    so about a third of the sites are expected outside it under random
    placement itself, and with few sites the rule often rejects data that
    were placed at random. Read ``rejected`` alongside the plot
    (``plot_coleman``) and the pattern of the sites that fall outside.

    References
    ----------
    Coleman BD (1981). On random placement and species-area relations.
    Mathematical Biosciences 54, 191-215.

    Coleman BD, Mares MA, Willig MR, Hsieh Y-H (1982). Randomness, area, and
    species richness. Ecology 63, 1121-1133.

    Examples
    --------
    >>> import numpy as np
    >>> import sars
    >>> rng = np.random.default_rng(1)
    >>> area = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    >>> totals = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
    >>> counts = np.array([rng.multinomial(n, area / area.sum()) for n in totals])
    >>> fit = sars.coleman(counts, area)
    >>> fit.n_sites
    6
    """
    site_labels: pd.Index | None = None
    species: pd.Index | None = None
    if isinstance(abundance, pd.DataFrame):
        site_labels, species = abundance.columns, abundance.index
    counts = _counts_array(abundance)
    n_sites = counts.shape[1]
    if n_sites < 2:
        raise ValueError(
            f"Need at least 2 sites (columns of abundance), got {n_sites}: a site's "
            "relative area is its share of the total over all sites."
        )
    if np.any(counts < 0):
        raise ValueError("abundance holds negative values; counts must be zero or more.")
    if np.any(counts != np.round(counts)):
        raise ValueError(
            "abundance holds fractional values; the model needs counts of individuals "
            "(not densities, cover or biomass)."
        )

    areas = _site_areas(area, site_labels, n_sites)

    if species is None:
        species = pd.RangeIndex(counts.shape[0])
    totals = pd.Series(counts.sum(axis=1), index=species, name="abundance")
    totals.index.name = "species"
    totals = totals[totals > 0]
    if len(totals) == 0:
        raise ValueError("abundance holds no individuals.")
    if counts.max() <= 1:
        warnings.warn(
            "No value in abundance is above 1, which looks like presence-absence data. "
            "The random placement model needs the number of individuals of each species.",
            UserWarning,
            stacklevel=2,
        )

    total_area = float(areas.sum())
    alpha = areas / total_area
    expected, sd = _expected_and_sd(alpha, totals.to_numpy(dtype=float))
    observed = (counts > 0).sum(axis=0)
    within = np.abs(observed - expected) <= sd

    sites = pd.DataFrame(
        {
            "area": areas,
            "relative_area": alpha,
            "observed": observed,
            "expected": expected,
            "sd": sd,
            "within_sd": within,
        },
        index=site_labels if site_labels is not None else pd.RangeIndex(n_sites),
    )
    sites.index.name = "site"
    return ColemanFit(sites=sites, abundance=totals, total_area=total_area)
