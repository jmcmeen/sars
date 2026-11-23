"""Species-area models with habitat: the habitat SARs and the countryside SAR."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sars._models import _compute_ic

_LOG_BASES = {"e": np.log, "2": np.log2, "10": np.log10}


def _log_fn(log_base: str | float) -> Any:
    key = str(log_base).removesuffix(".0")
    if key not in _LOG_BASES:
        raise ValueError(f"log_base must be 'e', 2 or 10, got {log_base!r}.")
    return _LOG_BASES[key]


def _fit_curve(
    func: Any,
    x: tuple[np.ndarray, ...],
    y: np.ndarray,
    starts: list[list[float]],
    lower: list[float] | None = None,
) -> tuple[np.ndarray, float] | None:
    """Least squares from each start; the best (lowest RSS) fit, or None."""
    from scipy.optimize import least_squares

    best: tuple[np.ndarray, float] | None = None
    k = len(starts[0])
    bounds = (lower, [np.inf] * k) if lower is not None else (-np.inf, np.inf)

    def resid(p: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            r = func(p, *x) - y
        return np.where(np.isfinite(r), r, 1e150)

    for start in starts:
        p0 = np.asarray(start, dtype=float)
        if lower is not None:
            p0 = np.maximum(p0, lower)
        with np.errstate(all="ignore"):
            if not np.all(np.isfinite(resid(p0))) or np.any(np.abs(resid(p0)) >= 1e150):
                continue
            try:
                res = least_squares(
                    resid,
                    p0,
                    bounds=bounds,
                    method="trf",
                    x_scale="jac",
                    max_nfev=2000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                )
            except (ValueError, FloatingPointError):
                continue
        rss = float(np.sum(res.fun**2))
        if not np.isfinite(rss) or np.any(np.abs(res.fun) >= 1e150):
            continue
        if best is None or rss < best[1]:
            best = (res.x, rss)
    return best


def _fit_stats(y: np.ndarray, rss: float, n_params: int, linear: bool) -> dict[str, float]:
    """R2, adjusted R2 and the information criteria, as R sars computes them:
    its adjusted R2 counts sigma among the parameters for a non-linear fit,
    and not for a linear one."""
    n = len(y)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    p = n_params if linear else n_params + 1
    adj = 1.0 - (n - 1) * rss / ((n - p) * tss) if tss > 0 and n > p else np.nan
    aic, aicc, bic = _compute_ic(n_params + 1, n, rss, y)
    return {"R2": r2, "adjR2": adj, "AIC": aic, "BIC": bic, "AICc": aicc}


# ---------------------------------------------------------------------------
# Habitat SARs
# ---------------------------------------------------------------------------

_HABITAT_GRID = [
    list(p)
    for p in itertools.product(
        np.arange(0, 51, 5.0),
        [0.1, 0.25, 0.75, 1.0],
        [-1.0, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 50.0],
    )
]


@dataclass
class HabitatFit:
    """The habitat species-area models fitted to one dataset.

    Attributes
    ----------
    mod_type : str
        "power", "logarithmic" or "power_log" (see `sar_habitat`).
    summary : pd.DataFrame
        One row per model, best first by AICc: its parameters ``c``, ``z``
        and ``d`` (NaN where the model has none), ``R2``, ``adjR2``, ``AIC``,
        ``BIC``, ``AICc`` and ``rss``. In the power_log form ``c`` is the
        intercept on the log scale, and the jigsaw model's ``d`` is the
        coefficient of log habitats plus ``z`` (so ``d - z`` is that
        coefficient, as R reports it).
    params : dict[str, dict[str, float]]
        Each fitted model's parameters.
    data : pd.DataFrame
        The data: columns ``area``, ``habitats`` and ``species``.
    log_base : str
        The logarithm used by the logarithmic and power_log forms.
    con : float or None
        The constant added to richness before taking logs (power_log form,
        data with zeros).
    not_fitted : list[str]
        Models that couldn't be fitted.
    """

    mod_type: str
    summary: pd.DataFrame = field(repr=False)
    params: dict[str, dict[str, float]] = field(repr=False)
    data: pd.DataFrame = field(repr=False)
    log_base: str = "e"
    con: float | None = None
    not_fitted: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        best = self.summary.iloc[0]["model"] if len(self.summary) else "?"
        return f"HabitatFit(mod_type={self.mod_type!r}, {len(self.summary)} models, best={best!r})"

    def predict(
        self, area: float | ArrayLike, habitats: float | ArrayLike, model: str | None = None
    ) -> np.ndarray:
        """Richness predicted by a model (the best by AICc by default) at
        given areas and numbers of habitats. In the power_log form the
        prediction is back-transformed from the log scale (minus ``con``)."""
        name = model or str(self.summary.iloc[0]["model"])
        if name not in self.params:
            raise ValueError(f"No fitted model {name!r}; fitted: {list(self.params)}.")
        a, h = np.broadcast_arrays(np.asarray(area, dtype=float), np.asarray(habitats, dtype=float))
        if not (
            np.all(np.isfinite(a)) and np.all(a > 0) and np.all(np.isfinite(h)) and np.all(h > 0)
        ):
            raise ValueError("Areas and numbers of habitats must be positive numbers.")
        p = self.params[name]
        log = _log_fn(self.log_base)
        out = _habitat_value(self.mod_type, name, p, a, h, log)
        if self.mod_type == "power_log":
            base = {"e": np.e, "2": 2.0, "10": 10.0}[self.log_base]
            out = np.power(base, out) - (self.con or 0.0)
        return np.asarray(out, dtype=float)


def _habitat_value(
    mod_type: str, name: str, p: dict[str, float], a: np.ndarray, h: np.ndarray, log: Any
) -> np.ndarray:
    """A habitat model's value: richness, or log richness in the power_log form."""
    c, z, d = p.get("c", np.nan), p.get("z", np.nan), p.get("d", np.nan)
    with np.errstate(all="ignore"):
        if mod_type == "power":
            if name == "choros":
                return c * (a * h) ** z
            if name == "jigsaw":
                return c * h**d * (a / h) ** z
            if name == "Kallimanis":
                return c * a ** (z + d * h)
            return c * a**z
        if mod_type == "logarithmic":
            if name == "choros":
                return c + z * log(a * h)
            if name == "jigsaw":
                return h**d * log(c * (a / h) ** z)
            if name == "Kallimanis":
                return c + (z + d * h) * log(a)
            return c + z * log(a)
        # power_log: linear in the logs
        if name == "choros":
            return c + z * log(a * h)
        if name == "jigsaw":
            return c + z * log(a) + (d - z) * log(h)
        if name == "Kallimanis":
            return c + z * log(a) + d * h * log(a)
        return c + z * log(a)


def sar_habitat(
    data: pd.DataFrame,
    mod_type: str = "power_log",
    area_col: str = "area",
    habitats_col: str = "habitats",
    species_col: str = "species",
    con: float | None = None,
    log_base: str | float = "e",
) -> HabitatFit:
    """Fit the habitat species-area models, as R sars's ``sar_habitat``.

    Richness may depend on the number of habitats on an island as well as on
    its area. Four models are compared:

    - **choros** (Triantis et al. 2003): richness depends on the product of
      area and habitats, K = A·H. Power form S = c·K^z.
    - **jigsaw** (Tjørve 2009): S = c·H^d·(A/H)^z, the habitats as pieces of
      area each with its own species.
    - **Kallimanis** (Kallimanis et al. 2008): the slope rises with habitat
      heterogeneity, S = c·A^(z + d·H).
    - the classical power (or logarithmic) curve on area alone.

    Parameters
    ----------
    data : pd.DataFrame
        One row per island.
    mod_type : str
        "power_log" (the default, as in R): the power forms fitted as linear
        regressions on the logs (log S on log A and log H). "power": the
        power forms fitted by non-linear least squares. "logarithmic": the
        logarithmic (S = c + z·log A) forms.
    area_col, habitats_col, species_col : str
        The columns holding area, the number of habitats (or another
        heterogeneity measure), and species richness.
    con : float, optional
        For the power_log form, a constant added to every richness value
        before taking logs when any island has no species (it must make every
        value positive). As in R, it is ignored (with a warning) otherwise.
    log_base : str or float
        The logarithm: "e" (default), 2 or 10.

    Returns
    -------
    HabitatFit

    Raises
    ------
    ValueError
        For an unknown ``mod_type`` or ``log_base``, missing or non-finite
        values, or richness of zero in the power_log form without ``con``.

    References
    ----------
    Triantis KA, Mylonas M, Lika K, Vardinoyannis K (2003). A model for the
    species-area-habitat relationship. Journal of Biogeography 30, 19-27.
    Tjørve E (2009). Shapes and functions of species-area curves (II).
    Journal of Biogeography 36, 1435-1445. Kallimanis AS et al. (2008). How
    does habitat diversity affect the species-area relationship? Global
    Ecology and Biogeography 17, 532-538.

    Examples
    --------
    >>> fit = sars.sar_habitat(sars.load_habitat())
    >>> fit.summary[["model", "z", "d", "AICc"]]  # doctest: +SKIP
    """
    if mod_type not in ("power", "logarithmic", "power_log"):
        raise ValueError(
            f"mod_type must be 'power', 'logarithmic' or 'power_log', got {mod_type!r}."
        )
    log = _log_fn(log_base)
    base = str(log_base).removesuffix(".0")
    for column in (area_col, habitats_col, species_col):
        if column not in data.columns:
            raise ValueError(f"No column {column!r}; the columns are {list(data.columns)}.")
    frame = pd.DataFrame(
        {
            "area": pd.to_numeric(data[area_col], errors="coerce"),
            "habitats": pd.to_numeric(data[habitats_col], errors="coerce"),
            "species": pd.to_numeric(data[species_col], errors="coerce"),
        }
    ).reset_index(drop=True)
    if not np.all(np.isfinite(frame.to_numpy(dtype=float))):
        raise ValueError("area, habitats and species must all be finite numbers.")
    if (frame["area"] <= 0).any() or (frame["habitats"] <= 0).any():
        raise ValueError("Every island needs a positive area and number of habitats.")
    a = frame["area"].to_numpy(dtype=float)
    h = frame["habitats"].to_numpy(dtype=float)
    s = frame["species"].to_numpy(dtype=float)

    used_con = None
    if mod_type == "power_log":
        if (s <= 0).any():
            if con is None:
                raise ValueError(
                    "Some islands have no species: give con, a constant added to every "
                    "richness value before taking logs."
                )
            used_con = float(con)
            if not (np.isfinite(used_con) and np.all(s + used_con > 0)):
                raise ValueError(f"con must make every richness positive, got {con!r}.")
        elif con is not None:
            warnings.warn(
                "con is added only when some island has no species, as in R; it is ignored.",
                stacklevel=2,
            )
        y = log(s + (used_con or 0.0))
    else:
        y = s

    params: dict[str, dict[str, float]] = {}
    rss: dict[str, float] = {}
    n_params: dict[str, int] = {}
    classical = "logarithmic" if mod_type == "logarithmic" else "power"
    names = ["choros", "jigsaw", "Kallimanis", classical]

    if mod_type == "power_log":
        designs = {
            "choros": [log(a * h)],
            "jigsaw": [log(a), log(h)],
            "Kallimanis": [log(a), h * log(a)],
            "power": [log(a)],
        }
        for name, cols in designs.items():
            x = np.column_stack([np.ones_like(y), *cols])
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            r = float(np.sum((x @ coef - y) ** 2))
            if name == "jigsaw":
                p = {"c": coef[0], "z": coef[1], "d": coef[2] + coef[1]}
            elif name == "Kallimanis":
                p = {"c": coef[0], "z": coef[1], "d": coef[2]}
            else:
                p = {"c": coef[0], "z": coef[1]}
            params[name] = {k: float(v) for k, v in p.items()}
            rss[name] = r
            n_params[name] = len(coef)
    else:

        def model(name: str) -> tuple[Any, list[list[float]], list[str]]:
            def f(p: np.ndarray, a: np.ndarray, h: np.ndarray) -> np.ndarray:
                keys = ["c", "z", "d"][: len(p)]
                return _habitat_value(mod_type, name, dict(zip(keys, p, strict=True)), a, h, log)

            if name in ("choros", classical):
                return f, [[5.0, 0.25]] if mod_type == "power" else [[5.0, 0.25]], ["c", "z"]
            if mod_type == "power":
                return f, _HABITAT_GRID, ["c", "z", "d"]
            return f, [[5.0, 1.0, 0.6]], ["c", "z", "d"]

        for name in names:
            f, starts, keys = model(name)
            fit = _fit_curve(f, (a, h), y, starts)
            if fit is None:
                continue
            params[name] = {k: float(v) for k, v in zip(keys, fit[0], strict=True)}
            rss[name] = fit[1]
            n_params[name] = len(keys)

    not_fitted = [n for n in names if n not in params]
    if not params:
        raise RuntimeError("No habitat model could be fitted to these data.")
    if not_fitted:
        warnings.warn(
            f"These habitat models couldn't be fitted: {', '.join(not_fitted)}.", stacklevel=2
        )
    rows = []
    for name in names:
        if name not in params:
            continue
        p = params[name]
        stats = _fit_stats(y, rss[name], n_params[name], linear=mod_type == "power_log")
        rows.append(
            {
                "model": name,
                "c": p.get("c", np.nan),
                "z": p.get("z", np.nan),
                "d": p.get("d", np.nan),
                **stats,
                "rss": rss[name],
            }
        )
    summary = pd.DataFrame(rows).sort_values("AICc", kind="stable").reset_index(drop=True)
    return HabitatFit(
        mod_type=mod_type,
        summary=summary,
        params=params,
        data=frame,
        log_base=base,
        con=used_con,
        not_fitted=not_fitted,
    )


# ---------------------------------------------------------------------------
# The countryside SAR
# ---------------------------------------------------------------------------

_GRIDS: dict[str, list[float | None]] = {
    "partial": [1e-10, 1e-6, 0.1, 5000.0, 1e5, 1e7, 1e8, None],
    "exhaustive": [1e-10, 1e-8, 1e-6, 1e-4, 0.01, 1.0, 1000.0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10, None],
    "none": [1e-6, 1.0, None],
}
_Z_GRIDS: dict[str, list[float | None]] = {
    "partial": [0.01, 0.1, 0.7, None],
    "exhaustive": [0.001, 0.01, 0.1, 0.25, 0.5, 1.0, None],
    "none": [None],
}


@dataclass
class CountrysideFit:
    """The countryside species-area model fitted to one dataset.

    Each group of species (by habitat affinity) has its own curve of the
    habitat areas: S = (h_1·A_1 + ... + h_k·A_k)^z in the power form, or
    S = z·log(h_1·A_1 + ... + h_k·A_k) in the logarithmic form. The h's say
    how much each habitat counts for that group.

    Attributes
    ----------
    mod_type : str
        "power" or "logarithmic".
    habitats : list[str]
        The habitat (area) columns, in order.
    groups : list[str]
        The species groups (richness columns) that could be fitted.
    params : pd.DataFrame
        One row per group: ``h_<habitat>`` for each habitat, ``z``, ``rss``
        and the group's ``AIC``.
    affinity : pd.DataFrame
        One row per group: each h divided by the group's largest, so 1 marks
        the habitat the group depends on most and values near 0 habitats it
        barely uses.
    c : pd.Series
        Each group's c: (largest h)^z (power) or z·log(largest h)
        (logarithmic), the richness a unit area of its best habitat holds.
    predicted_total : np.ndarray
        The model's total richness at each site (the sum over groups); NaN
        if any group couldn't be fitted.
    rss : dict[str, float]
        The total richness's RSS over every site under the countryside model
        (``"countryside"``) and under a single power (or logarithmic) curve
        of total area (``"power"`` or ``"logarithmic"``); empty if any group
        couldn't be fitted.
    data : pd.DataFrame
        The data's habitat area and richness columns.
    not_fitted : list[str]
        Groups whose model couldn't be fitted.
    """

    mod_type: str
    habitats: list[str]
    groups: list[str]
    params: pd.DataFrame = field(repr=False)
    affinity: pd.DataFrame = field(repr=False)
    c: pd.Series = field(repr=False)
    predicted_total: np.ndarray = field(repr=False)
    rss: dict[str, float] = field(repr=False)
    data: pd.DataFrame = field(repr=False)
    not_fitted: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"CountrysideFit(mod_type={self.mod_type!r}, {len(self.habitats)} habitats, "
            f"{len(self.groups)} groups)"
        )

    def predict(self, areas: ArrayLike) -> pd.DataFrame:
        """Each group's richness, and the total, for given habitat areas.

        Parameters
        ----------
        areas : array-like
            One value per habitat, in the order of ``habitats``, or a 2-D
            array with one row per site.

        Returns
        -------
        pd.DataFrame
            One row per site: a column per group and ``total`` (NaN if any
            group couldn't be fitted).
        """
        a = np.atleast_2d(np.asarray(areas, dtype=float))
        if a.shape[1] != len(self.habitats):
            raise ValueError(
                f"Give an area for each of the {len(self.habitats)} habitats "
                f"({', '.join(self.habitats)}), not {a.shape[1]}."
            )
        if not (np.all(np.isfinite(a)) and np.all(a >= 0)):
            raise ValueError("Habitat areas must be finite and not negative.")
        out = {}
        h_cols = [f"h_{name}" for name in self.habitats]
        hs = self.params.loc[self.groups, h_cols].to_numpy(dtype=float)
        zs = self.params.loc[self.groups, "z"].to_numpy(dtype=float)
        for group, h, z in zip(self.groups, hs, zs, strict=True):
            if self.mod_type == "logarithmic" and np.any(a @ h <= 0):
                raise ValueError(
                    f"The logarithmic model of {group} needs some area of a habitat it uses "
                    "(its weighted area is zero at some site)."
                )
            out[group] = _countryside_value(self.mod_type, h, float(z), a)
        frame = pd.DataFrame(out)
        # A total needs every group: without one, it would be too low.
        frame["total"] = frame.sum(axis=1) if not self.not_fitted else np.nan
        return frame


def _countryside_value(mod_type: str, h: np.ndarray, z: float, a: np.ndarray) -> np.ndarray:
    weighted = a @ h
    with np.errstate(all="ignore"):
        return z * np.log(weighted) if mod_type == "logarithmic" else weighted**z


def sar_countryside(
    data: pd.DataFrame,
    habitats: list[str],
    species: list[str],
    mod_type: str = "power",
    grid_start: str = "partial",
    z_lower: float = 0.0,
    n_starts: int | None = 40,
) -> CountrysideFit:
    """Fit the countryside species-area model, as R sars's ``sar_countryside``.

    In a landscape of several habitats, species differ in which habitats they
    use. The countryside SAR (Pereira & Daily 2006; Proença & Pereira 2013)
    splits the species into groups by habitat affinity (forest species,
    farmland species, ...) and fits each group's richness as a curve of a
    weighted sum of the habitat areas, the weights (h) saying how much each
    habitat counts for that group. The fit then predicts richness for any mix
    of habitats, such as a landscape after land-use change.

    Parameters
    ----------
    data : pd.DataFrame
        One row per site.
    habitats : list[str]
        The columns holding each habitat's area in the site.
    species : list[str]
        The columns holding each species group's richness in the site.
    mod_type : str
        "power" (default), S = (Σ h_i·A_i)^z, or "logarithmic",
        S = z·log(Σ h_i·A_i).
    grid_start : str
        The starting values tried for each group, as in R: "partial" (the
        default), "exhaustive" (more, slower) or "none" (few).
    z_lower : float
        The smallest z allowed (0 by default). The h's are at least 0.
    n_starts : int or None
        How many of the grid's starting values to fit from: the ``n_starts``
        whose curves are closest to the data before fitting, plus the closest
        for each habitat as the most weighted (40 by default). None fits from
        every one, as R does: on the bundled data the two give the same fits,
        and the full grid takes 50 to 70 times as long.

    Returns
    -------
    CountrysideFit

    Raises
    ------
    ValueError
        For unknown columns or arguments, non-finite values, or a site with
        no area under the logarithmic model.

    Warns
    -----
    UserWarning
        If a group's model couldn't be fitted, or a site has no area.

    References
    ----------
    Pereira HM, Daily GC (2006). Modeling biodiversity dynamics in
    countryside landscapes. Ecology 87, 1877-1885. Proença V, Pereira HM
    (2013). Species-area models to assess biodiversity change in
    multi-habitat landscapes: the importance of species habitat affinity.
    Basic and Applied Ecology 14, 102-114.

    Examples
    --------
    >>> cs = sars.load_countryside()
    >>> fit = sars.sar_countryside(
    ...     cs, habitats=["Area_AG", "Area_SH", "Area_QF"],
    ...     species=["Spcs_AG", "Spcs_SH", "Spcs_QF", "Spcs_UB"])
    >>> fit.affinity  # doctest: +SKIP
    """
    if mod_type not in ("power", "logarithmic"):
        raise ValueError(f"mod_type must be 'power' or 'logarithmic', got {mod_type!r}.")
    if grid_start not in _GRIDS:
        raise ValueError(f"grid_start must be one of {list(_GRIDS)}, got {grid_start!r}.")
    if not habitats or not species:
        raise ValueError("Give at least one habitat column and one species column.")
    missing = [c for c in [*habitats, *species] if c not in data.columns]
    if missing:
        raise ValueError(f"No columns {missing}; the columns are {list(data.columns)}.")
    if isinstance(z_lower, bool) or not isinstance(z_lower, Real) or not np.isfinite(z_lower):
        raise ValueError(f"z_lower must be a finite number, got {z_lower!r}.")
    if n_starts is not None and (
        isinstance(n_starts, bool) or not isinstance(n_starts, int) or n_starts < 1
    ):
        raise ValueError(f"n_starts must be a positive whole number or None, got {n_starts!r}.")
    frame = data[[*habitats, *species]].apply(pd.to_numeric, errors="coerce")
    values = frame.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Every habitat area and richness must be a finite number.")
    if np.any(values < 0):
        raise ValueError("Habitat areas and richness can't be negative.")
    areas = frame[habitats].to_numpy(dtype=float)
    total_area = areas.sum(axis=1)
    if np.any(total_area == 0):
        if mod_type == "logarithmic":
            raise ValueError("Some sites have no area, which the logarithmic model can't take.")
        warnings.warn("Some sites have a total area of zero.", stacklevel=2)

    k = len(habitats)
    fits: dict[str, tuple[np.ndarray, float]] = {}
    for group in species:
        s = frame[group].to_numpy(dtype=float)
        starts = _countryside_starts(total_area, s, mod_type, grid_start, k)

        def f(p: np.ndarray, a: np.ndarray) -> np.ndarray:
            return _countryside_value(mod_type, p[:-1], p[-1], a)

        lower = [0.0] * k + [float(z_lower)]
        chosen = (
            starts if n_starts is None else _best_starts(f, areas, s, starts, lower, k, n_starts)
        )
        group_fit = _fit_curve(f, (areas,), s, chosen, lower)
        if group_fit is not None:
            fits[group] = group_fit
    not_fitted = [g for g in species if g not in fits]
    if not fits:
        raise RuntimeError("No species group's model could be fitted.")
    if not_fitted:
        warnings.warn(
            f"The models of these species groups couldn't be fitted: {', '.join(not_fitted)}.",
            stacklevel=2,
        )
    groups = [g for g in species if g in fits]
    h_cols = [f"h_{name}" for name in habitats]
    n = len(frame)
    param_rows = []
    for g in groups:
        p, r = fits[g]
        aic = _compute_ic(k + 2, n, r, frame[g].to_numpy(dtype=float))[0]
        param_rows.append(
            {**dict(zip(h_cols, p[:-1], strict=True)), "z": p[-1], "rss": r, "AIC": aic}
        )
    params = pd.DataFrame(param_rows, index=pd.Index(groups, name="group"))
    hs = params[h_cols].to_numpy(dtype=float)
    hmax = hs.max(axis=1)
    if np.any(hmax == 0):
        zero = [g for g, m in zip(groups, hmax, strict=True) if m == 0]
        warnings.warn(
            f"Every habitat weight is 0 for {', '.join(zero)}: the group's affinities are "
            "undefined (NaN).",
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        affinity = pd.DataFrame(hs / hmax[:, None], index=params.index, columns=list(habitats))
        if mod_type == "logarithmic":
            c = np.log(hmax) * params["z"].to_numpy()
        else:
            c = hmax ** params["z"].to_numpy()
    c_series = pd.Series(c, index=params.index, name="c")

    fit = CountrysideFit(
        mod_type=mod_type,
        habitats=list(habitats),
        groups=groups,
        params=params,
        affinity=affinity,
        c=c_series,
        predicted_total=np.array([]),
        rss={},
        data=frame.reset_index(drop=True),
        not_fitted=not_fitted,
    )
    total_rich = frame[species].sum(axis=1).to_numpy(dtype=float)
    rss: dict[str, float] = {}
    if not_fitted:
        # A total needs every group: without one, it would be too low.
        predicted = np.full(n, np.nan)
    else:
        predicted = fit.predict(areas)["total"].to_numpy()
        rss["countryside"] = float(np.sum((predicted - total_rich) ** 2))
        single = _single_curve_rss(total_area, total_rich, mod_type)
        if single is not None:
            rss[mod_type] = single
    fit.predicted_total = predicted
    fit.rss = rss
    return fit


def _countryside_starts(
    total_area: np.ndarray, s: np.ndarray, mod_type: str, grid_start: str, k: int
) -> list[list[float]]:
    """R's starting grid: each h over a range of magnitudes ending at the
    value a single curve of total area implies, and z over a few values."""
    c1, z1 = 5.0, 0.25
    single = _single_curve(total_area, s, mod_type)
    if single is not None:
        c1, z1 = single
    with np.errstate(all="ignore"):
        hmax = float(np.exp(c1 / z1) if mod_type == "logarithmic" else c1 ** (1.0 / z1))
    if not np.isfinite(hmax) or hmax <= 0:
        hmax = 1.0
    h_vals = [hmax if v is None else v for v in _GRIDS[grid_start]]
    z_vals = [z1 if v is None else v for v in _Z_GRIDS[grid_start]]
    grid = [list(p) for p in itertools.product(*([h_vals] * k), z_vals)]
    if grid_start == "none" and hmax > 5:
        kept = [g for g in grid if hmax < sum(g[:-1]) < 2 * hmax]
        grid = kept or grid
    return grid


def _best_starts(
    f: Any,
    areas: np.ndarray,
    s: np.ndarray,
    starts: list[list[float]],
    lower: list[float],
    n_habitats: int,
    keep: int = 40,
) -> list[list[float]]:
    """The starts whose curves are closest to the data before any fitting,
    plus, so that no region of the grid goes untried, the closest among those
    where each habitat has the largest weight. The grid can have thousands of
    points, and refining a few finds the same optimum far faster."""
    scored = []
    for start in starts:
        p = np.maximum(np.asarray(start, dtype=float), lower)
        with np.errstate(all="ignore"):
            r = f(p, areas) - s
        rss = float(np.sum(r**2))
        if np.isfinite(rss):
            scored.append((rss, start))
    if not scored:
        return starts[:keep]
    scored.sort(key=lambda t: t[0])
    chosen = [start for _, start in scored[:keep]]
    for i in range(n_habitats):
        best = next(
            (st for _, st in scored if int(np.argmax(st[:n_habitats])) == i and st not in chosen),
            None,
        )
        if best is not None:
            chosen.append(best)
    return chosen


def _single_curve(area: np.ndarray, s: np.ndarray, mod_type: str) -> tuple[float, float] | None:
    """c and z of a power (or logarithmic) curve of total area, or None."""
    from sars._models import _fit_nls

    keep = area > 0
    if keep.sum() < 3:
        return None
    frame = pd.DataFrame({"area": area[keep], "species": s[keep]})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = _fit_nls(
                "loga" if mod_type == "logarithmic" else "power", frame, allow_zero_species=True
            )
    except (ValueError, RuntimeError):
        return None
    if not fit.converged:
        return None
    return float(fit.params["c"]), float(fit.params["z"])


def _single_curve_rss(area: np.ndarray, s: np.ndarray, mod_type: str) -> float | None:
    """The RSS of a power (or logarithmic) curve of total area, fitted to the
    sites with area and scored on every site (a site with no area has none
    of the curve's species), the same sites as the countryside RSS."""
    curve = _single_curve(area, s, mod_type)
    if curve is None:
        return None
    c, z = curve
    with np.errstate(all="ignore"):
        pred = c + z * np.log(area) if mod_type == "logarithmic" else c * area**z
    return float(np.sum((pred - s) ** 2))
