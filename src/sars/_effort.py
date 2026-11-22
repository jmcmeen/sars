"""Richness standardised for sampling effort: rarefaction and extrapolation by
sample size or sample coverage (Chao & Jost 2012; Chao et al. 2014), as in
R's iNEXT."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import gammaln


def _counts(x: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    x = x[x > 0]
    return x, int(x.sum()), int(np.sum(x == 1)), int(np.sum(x == 2))


def _chao1(x: np.ndarray) -> float:
    """Chao1, the bias-corrected form iNEXT uses: S_obs + f0."""
    x, n, f1, f2 = _counts(x)
    return len(x) + _f0(n, f1, f2)


def _f0(n: int, f1: int, f2: int) -> float:
    if f2 == 0:
        return (n - 1) / n * f1 * (f1 - 1) / 2
    return (n - 1) / n * f1**2 / (2 * f2)


def _rarefied(x: np.ndarray, m: int) -> float:
    """Expected richness of a sample of m individuals (m <= n), without
    replacement: sum over species of 1 - C(n - x_i, m) / C(n, m)."""
    x, n, _, _ = _counts(x)
    if m >= n:
        return float(len(x))
    keep = n - x >= m
    log_ratio = (
        gammaln(n - x[keep] + 1)
        - gammaln(n - x[keep] - m + 1)
        - gammaln(n + 1)
        + gammaln(n - m + 1)
    )
    return float(len(x) - np.sum(np.exp(log_ratio)))


def _coverage_at(x: np.ndarray, m: float) -> float:
    """Sample coverage at sample size m, as iNEXT's Chat.Ind: interpolated
    below n (linearly between whole sizes), extrapolated above it."""
    x, n, f1, f2 = _counts(x)
    f0 = _f0(n, f1, f2)
    a = n * f0 / (n * f0 + f1) if f1 > 0 else 1.0

    def whole(k: int) -> float:
        if k >= n:
            return 1.0 - f1 / n * a
        keep = n - x >= k
        xx = x[keep]
        log_ratio = gammaln(n - xx + 1) - gammaln(n - xx - k + 1) - gammaln(n) + gammaln(n - k)
        return float(1.0 - np.sum(xx / n * np.exp(log_ratio)))

    if m > n:
        return float(1.0 - f1 / n * a ** (m - n + 1))
    if m == n:
        return whole(n)
    lo, hi = int(np.floor(m)), int(np.ceil(m))
    if lo == hi:
        return whole(lo)
    return (hi - m) * whole(lo) + (m - lo) * whole(hi)


def _richness_at(x: np.ndarray, m: float) -> float:
    """Richness at sample size m, as iNEXT's TD.m.est for q = 0:
    rarefaction below n (linear between whole sizes), and above n
    S_obs + (Chao1 - S_obs)(1 - (1 - beta)^(m - n)), with beta from the
    rarefied richness at n - 1."""
    x, n, _, _ = _counts(x)
    s_obs = float(len(x))
    if m < n:
        lo, hi = int(np.floor(m)), int(np.ceil(m))
        if lo == hi:
            return _rarefied(x, lo)
        return (hi - m) * _rarefied(x, lo) + (m - lo) * _rarefied(x, hi)
    if m == n:
        return s_obs
    asy = _chao1(x)
    below = _rarefied(x, n - 1)
    beta = (s_obs - below) / (asy - below) if asy != below else 0.0
    return s_obs + (asy - s_obs) * (1.0 - (1.0 - beta) ** (m - n))


def _size_for_coverage(x: np.ndarray, target: float) -> float:
    """The sample size at which coverage reaches ``target``, as iNEXT's
    invChat.Ind: found numerically below the sample's own coverage, in
    closed form above it, and at least 1."""
    from scipy.optimize import minimize_scalar

    x, n, f1, f2 = _counts(x)
    ref = _coverage_at(x, n)
    if target == ref:
        m = float(n)
    elif target < ref:
        res = minimize_scalar(
            lambda m: abs(_coverage_at(x, m) - target),
            bounds=(0.0, float(n)),
            method="bounded",
            options={"xatol": 1e-8},
        )
        m = float(res.x)
    else:
        if f1 > 0 and f2 > 0:
            a = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)
        elif f1 > 1 and f2 == 0:
            a = (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)
        else:
            a = 0.0
        if target >= 1.0:
            m = np.inf  # full coverage is reached only in the limit, as in iNEXT
        else:
            m = n + (0.0 if a == 0 else (np.log(n / f1) + np.log(1 - target)) / np.log(a) - 1)
    return max(m, 1.0)


def standardised_richness(
    abundance: pd.DataFrame | ArrayLike,
    area: ArrayLike | None = None,
    base: str = "coverage",
    level: float | None = None,
) -> pd.DataFrame:
    """Species richness standardised for sampling effort, ready to fit.

    The species found at a site depend on how hard it was sampled as well as
    on how many there are: a larger island often gets more transects, and
    then its richness is inflated by effort as well as area (Gooriah et al.
    2021). This puts every site on an equal footing before a species-area
    curve is fitted, by rarefying (or extrapolating) each site's sample to
    the same sample coverage or the same number of individuals, as R's iNEXT
    does (Chao & Jost 2012; Chao et al. 2014).

    Parameters
    ----------
    abundance : pd.DataFrame or 2-D array-like
        Species-by-site abundance table: **one row per species, one column
        per site**, counts of individuals (as `coleman` and iNEXT take). A
        table with sites in rows needs transposing first (``df.T``).
    area : array-like, optional
        Each site's area, in the order of the columns. If given, the result
        has an ``area`` column and can go straight to the fitting functions.
    base : str
        "coverage" (the default): standardise to the same sample coverage,
        the share of the community's individuals that belong to species in
        the sample; sites are then compared at equal completeness. "size":
        standardise to the same number of individuals.
    level : float, optional
        The coverage (between 0 and 1) or number of individuals to
        standardise to. By default, as in iNEXT, the largest that every site
        reaches by extrapolating to twice its sample: the smallest coverage
        at twice each site's sample size, or the smallest doubled sample
        size.

    Returns
    -------
    pd.DataFrame
        One row per site: ``site``, ``area`` (if given), ``species`` (the
        standardised richness), ``observed`` (the richness in the sample),
        ``individuals``, ``coverage`` (the sample's own coverage), ``m`` (the
        sample size standardised to), ``coverage_at_m`` (the coverage at
        ``m``: the level asked for, unless ``m`` was held at its minimum of
        one individual), ``method`` ("rarefaction", "extrapolation" or
        "observed") and ``level``.

    Raises
    ------
    ValueError
        If the table isn't a non-negative whole-number table with at least
        one individual at every site, ``base`` is unknown, ``level`` is out
        of range, or ``area`` doesn't have one value per site.

    Warns
    -----
    UserWarning
        If a site is extrapolated beyond twice its sample, where the estimate
        is unreliable.

    References
    ----------
    Chao A, Jost L (2012). Coverage-based rarefaction and extrapolation:
    standardizing samples by completeness rather than size. Ecology 93,
    2533-2547. Chao A et al. (2014). Rarefaction and extrapolation with Hill
    numbers. Ecological Monographs 84, 45-67. Gooriah L et al. (2021).
    Synthesis reveals that island species-area relationships emerge from
    processes beyond passive sampling. Global Ecology and Biogeography 30,
    2119-2131.

    Examples
    --------
    >>> table = pd.DataFrame({"a": [10, 5, 1, 1], "b": [40, 20, 9, 3]})
    >>> sars.standardised_richness(table, area=[1.0, 4.0])  # doctest: +SKIP
    """
    if base not in ("coverage", "size"):
        raise ValueError(f"base must be 'coverage' or 'size', got {base!r}.")
    if isinstance(abundance, pd.DataFrame):
        sites = [str(c) for c in abundance.columns]
        text = [
            str(c) for c in abundance.columns if not pd.api.types.is_numeric_dtype(abundance[c])
        ]
        if text:
            raise ValueError(
                f"Columns that aren't counts: {', '.join(text)}. Every column is a site; put the "
                "species names in the index (e.g. pd.read_csv(..., index_col=0))."
            )
        table = abundance.to_numpy(dtype=float)
    else:
        table = np.asarray(abundance, dtype=float)
        if table.ndim != 2:
            raise ValueError("abundance must be a 2-D species-by-site table.")
        sites = [str(i) for i in range(table.shape[1])]
    if table.ndim != 2 or table.shape[1] == 0:
        raise ValueError("abundance must be a 2-D species-by-site table.")
    if not np.all(np.isfinite(table)) or np.any(table < 0) or np.any(table != np.round(table)):
        raise ValueError("abundance must hold counts: whole numbers, 0 or more.")
    totals = table.sum(axis=0)
    if np.any(totals == 0):
        empty = [s for s, t in zip(sites, totals, strict=True) if t == 0]
        raise ValueError(f"Sites with no individuals can't be standardised: {', '.join(empty)}.")
    columns = [table[:, j] for j in range(table.shape[1])]
    areas = None
    if area is not None:
        areas = np.asarray(area, dtype=float).ravel()
        if len(areas) != len(sites):
            raise ValueError(f"area has {len(areas)} values; the table has {len(sites)} sites.")

    if level is None:
        if base == "coverage":
            level = min(_coverage_at(x, 2 * x.sum()) for x in columns)
        else:
            level = float(min(2 * x.sum() for x in columns))
    elif base == "coverage" and not 0 < level <= 1:
        raise ValueError(f"A coverage level must be between 0 and 1, got {level!r}.")
    elif base == "size" and not level >= 1:
        raise ValueError(f"A size level must be at least one individual, got {level!r}.")

    rows = []
    for name, x in zip(sites, columns, strict=True):
        n = int(x.sum())
        m = _size_for_coverage(x, float(level)) if base == "coverage" else float(level)
        rows.append(
            {
                "site": name,
                "species": _richness_at(x, m),
                "observed": int(np.sum(x > 0)),
                "individuals": n,
                "coverage": _coverage_at(x, n),
                "m": m,
                "coverage_at_m": _coverage_at(x, m) if np.isfinite(m) else 1.0,
                "method": "extrapolation" if m > n else ("rarefaction" if m < n else "observed"),
                "level": float(level),
            }
        )
    result = pd.DataFrame(rows)
    far = result.loc[result["m"].round() > 2 * result["individuals"], "site"].tolist()
    if far:
        warnings.warn(
            f"Sites extrapolated beyond twice their sample, where the estimate is unreliable: "
            f"{', '.join(far)}. Choose a lower level.",
            stacklevel=2,
        )
    if areas is not None:
        result.insert(1, "area", areas)
    return result
