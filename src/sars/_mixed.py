"""A hierarchical log-log power model: z pooled across archipelagos."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sars._many import _named_datasets


@dataclass
class MixedLinPowFit:
    """The log-log power model fitted to several archipelagos at once.

    log S = (log c + u_j) + (z + v_j)·log A, with each archipelago's
    deviations (u_j, v_j) drawn from a normal distribution with standard
    deviations ``sd_logc`` and ``sd_z`` and correlation ``corr``, and normal
    residuals with standard deviation ``sigma``. Natural logarithms.

    Attributes
    ----------
    logc, c, z : float
        The mean intercept (log c, and c), and the mean slope, across
        archipelagos.
    logc_se, z_se : float
        Their standard errors.
    sd_logc, sd_z : float
        How much archipelagos' intercepts and slopes vary around the means.
        ``sd_z`` near 0 means the data are consistent with one slope for all.
    corr : float
        The correlation of an archipelago's intercept and slope deviations
        (NaN when either standard deviation is 0).
    sigma : float
        The residual standard deviation, on the log scale.
    reml_criterion : float
        -2 times the restricted log-likelihood at the fit (lme4's "REML
        criterion at convergence").
    n : int
        The number of islands.
    con : float
        The constant added to every richness value before taking logs: the
        ``con`` given when some island has no species, else 0.
    groups : pd.DataFrame
        One row per archipelago: ``n``, its partially pooled ``logc``, ``c``
        and ``z`` (the means plus its predicted deviations), and its own
        ``z_alone`` (and ``z_alone_se``) from a log-log fit of its data alone.
    data : pd.DataFrame
        The data: ``group``, ``area``, ``species``.
    """

    logc: float
    c: float
    z: float
    logc_se: float
    z_se: float
    sd_logc: float
    sd_z: float
    corr: float
    sigma: float
    reml_criterion: float
    n: int
    con: float = 0.0
    groups: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    data: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def __repr__(self) -> str:
        return (
            f"MixedLinPowFit({len(self.groups)} archipelagos, z={self.z:.4f} ± "
            f"{self.z_se:.4f}, sd_z={self.sd_z:.4f})"
        )

    def predict(self, area: float | ArrayLike, group: str | None = None) -> np.ndarray:
        """Richness predicted at given areas: by an archipelago's own pooled
        curve, or with ``group=None`` by the mean curve (for an archipelago
        not in the data). Back-transformed from the log scale, minus ``con``."""
        a = np.asarray(area, dtype=float)
        if not (np.all(np.isfinite(a)) and np.all(a > 0)):
            raise ValueError("Areas must be positive numbers.")
        if group is None:
            return np.exp(self.logc + self.z * np.log(a)) - self.con
        if group not in self.groups.index:
            raise ValueError(f"No archipelago {group!r}; they are {list(self.groups.index)}.")
        logc, z = self.groups.loc[group, ["logc", "z"]].to_numpy(dtype=float)
        return np.exp(logc + z * np.log(a)) - self.con


def _lower(theta: np.ndarray) -> np.ndarray:
    return np.array([[theta[0], 0.0], [theta[1], theta[2]]])


def _profiled(
    theta: np.ndarray, stats: list[tuple[np.ndarray, np.ndarray, float]], n: int
) -> tuple[float, dict]:
    """lme4's profiled REML criterion at relative covariance factor theta,
    with the pieces the fit needs.

    Each archipelago enters through its sufficient statistics A = X'X,
    b = X'y and c = y'y (X = [1, log A], y = log richness). With
    V = I + X G X' (G = L L'), the push-through identity gives
    X'V^-1X = A - A K A, X'V^-1y = b - A K b and y'V^-1y = c - b'K b with
    K = G (I + A G)^-1, and log|V| = log|I + A G|, so no n_j x n_j matrix is
    formed and G may be singular.
    """
    lam = _lower(theta)
    g = lam @ lam.T
    p = 2
    eye = np.eye(p)
    xtvx = np.zeros((p, p))
    xtvy = np.zeros(p)
    ytvy = 0.0
    logdet_v = 0.0
    for a, b, c in stats:
        m = eye + a @ g
        sign, logdet = np.linalg.slogdet(m)
        if sign <= 0:
            raise np.linalg.LinAlgError("I + A G is not positive definite")
        logdet_v += logdet
        k = g @ np.linalg.inv(m)
        xtvx += a - a @ k @ a
        xtvy += b - a @ k @ b
        ytvy += c - b @ k @ b
    beta = np.linalg.solve(xtvx, xtvy)
    rvr = ytvy - 2.0 * beta @ xtvy + beta @ xtvx @ beta
    # Relative to y'V^-1y: an exact power law leaves only rounding error.
    if not rvr > 1e-12 * max(abs(ytvy), 1.0):
        raise np.linalg.LinAlgError("no residual variation")
    sigma2 = rvr / (n - p)
    sign, logdet_x = np.linalg.slogdet(xtvx)
    if sign <= 0:
        raise np.linalg.LinAlgError("X'V^-1X is not positive definite")
    criterion = logdet_v + logdet_x + (n - p) * (1.0 + np.log(2.0 * np.pi * sigma2))
    return float(criterion), {"beta": beta, "sigma2": sigma2, "xtvx": xtvx, "g": g}


def _prepare(
    datasets: Mapping[str, pd.DataFrame] | pd.DataFrame,
    by: str | None,
    con: float,
    kind: str = "mixed model",
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], float]:
    """The archipelagos' areas and richness, checked, and the constant added
    to richness before taking logs (``con`` if any island has no species,
    else 0): the input of the pooled log-log models."""
    if isinstance(con, bool) or not isinstance(con, Real) or not np.isfinite(con):
        raise ValueError(f"'con' must be a finite number, got {con!r}")
    named = _named_datasets(datasets, by)
    if len(named) < 3:
        raise ValueError(
            f"A {kind} needs at least 3 archipelagos to estimate how they vary, not {len(named)}."
        )
    cleaned = []
    for name, frame in named.items():
        missing = [c for c in ("area", "species") if c not in frame.columns]
        if missing:
            raise ValueError(f"{name!r} has no column(s) {missing}.")
        area = pd.to_numeric(frame["area"], errors="coerce").to_numpy(dtype=float)
        species = pd.to_numeric(frame["species"], errors="coerce").to_numpy(dtype=float)
        if not (np.all(np.isfinite(area)) and np.all(np.isfinite(species))):
            raise ValueError(f"{name!r} has missing or non-numeric areas or richness.")
        if np.any(area <= 0) or np.any(species < 0):
            raise ValueError(f"{name!r} needs positive areas and richness of 0 or more.")
        if len(area) < 2 or np.ptp(area) == 0:
            raise ValueError(f"{name!r} needs at least two islands of different areas.")
        cleaned.append((name, area, species))
    # As lin_pow does, con is added only if some island has no species; here
    # to every archipelago, so that all are on the same log scale.
    added = 0.0
    if any(np.any(sp == 0) for _, _, sp in cleaned):
        if not con > 0:
            raise ValueError("Some islands have no species: con must be positive to take logs.")
        added = float(con)
    return cleaned, added


def lin_pow_mixed(
    datasets: Mapping[str, pd.DataFrame] | pd.DataFrame,
    by: str | None = None,
    con: float = 1.0,
) -> MixedLinPowFit:
    """Fit the log-log power model to several archipelagos at once, pooling z.

    Fitting each archipelago on its own gives noisy slopes for small ones;
    fitting them all as one ignores that they differ. This mixed model does
    both: log S = (log c + u_j) + (z + v_j)·log A, where each archipelago's
    intercept and slope deviate from the means by amounts drawn from one
    normal distribution, whose spread is estimated too. Each archipelago's
    intercept and slope are then pulled towards the means, the more so the
    less its own data say. They move together, so when the deviations are
    correlated one archipelago's slope can end up a little further from the
    mean than its own fit's. Fitted by restricted maximum likelihood (REML), as R's
    ``lme4::lmer(log(S) ~ log(A) + (log(A) | archipelago))``, which it
    matches.

    Parameters
    ----------
    datasets : mapping of name to DataFrame, or one DataFrame
        The archipelagos, each with columns 'area' and 'species' (as
        `sar_many` takes them); or one DataFrame with a column, named by
        ``by``, saying which archipelago each island is in.
    by : str, optional
        That column, for a single DataFrame.
    con : float
        If any island has no species, added to every richness value (in every
        archipelago, so that all share one log scale) before taking logs, as
        `lin_pow` does (1 by default); otherwise unused.

    Returns
    -------
    MixedLinPowFit

    Raises
    ------
    ValueError
        With fewer than three archipelagos (there is no spread to estimate),
        an archipelago with fewer than two islands or no spread of areas,
        missing columns, non-positive areas, or zeros without a positive
        ``con``.

    Warns
    -----
    UserWarning
        If the fit is singular: a standard deviation at 0 or a correlation of
        ±1, which says the data can't separate that part of the variation.

    Examples
    --------
    >>> fit = sars.lin_pow_mixed({"gala": sars.load_gala(), "aegean": sars.load_aegean(),
    ...                           "xu": sars.load_xu()})
    >>> fit.groups[["z", "z_alone"]]  # doctest: +SKIP
    """
    from scipy.optimize import minimize

    cleaned, added = _prepare(datasets, by, con)
    blocks = []
    stats = []
    rows = []
    long_parts = []
    for name, area, species in cleaned:
        x = np.column_stack([np.ones_like(area), np.log(area)])
        y = np.log(species + added)
        blocks.append((x, y))
        stats.append((x.T @ x, x.T @ y, float(y @ y)))
        alone, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ alone
        dof = len(y) - 2
        if dof > 0:
            s2 = float(resid @ resid) / dof
            z_alone_se = float(np.sqrt(s2 * np.linalg.inv(x.T @ x)[1, 1]))
        else:
            z_alone_se = np.nan
        rows.append({"group": name, "n": len(y), "z_alone": alone[1], "z_alone_se": z_alone_se})
        long_parts.append(pd.DataFrame({"group": name, "area": area, "species": species}))
    n = sum(len(y) for _, y in blocks)

    def objective(t: np.ndarray) -> float:
        try:
            return _profiled(t, stats, n)[0]
        except np.linalg.LinAlgError:
            return np.inf

    # Lines fitted to each archipelago on its own leave the least residual
    # variation any of these models can; with none, sigma goes to 0.
    within = sum(
        float(np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2)) for x, y in blocks
    )
    total = sum(float(np.sum((y - y.mean()) ** 2)) for _, y in blocks)
    if within <= 1e-12 * max(total, 1.0):
        raise ValueError(
            "Each archipelago's data lie exactly on a power law, leaving no residual variation "
            "to estimate the model from."
        )
    results = []
    for start in ([1.0, 0.0, 1.0], [0.1, 0.0, 0.1], [3.0, -1.0, 0.5], [0.5, 0.0, 0.05]):
        res = minimize(
            objective,
            np.array(start),
            method="L-BFGS-B",
            bounds=[(0.0, None), (None, None), (0.0, None)],
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
        )
        if np.isfinite(res.fun):
            results.append(res)
    if not results:
        raise RuntimeError("The REML criterion couldn't be evaluated from any starting point.")
    best = min(results, key=lambda r: r.fun)
    # L-BFGS-B can report a failed line search at the optimum itself, where it
    # can't improve further; independent starts that reach the same criterion
    # are the better sign of convergence.
    agreed = sum(abs(r.fun - best.fun) < 1e-6 for r in results) > 1
    if not (best.success or agreed):
        warnings.warn(
            f"The REML optimisation may not have converged ({best.message}); treat the "
            "estimates with caution.",
            stacklevel=2,
        )
    theta = best.x
    criterion, parts = _profiled(theta, stats, n)
    beta, sigma2, xtvx, g = parts["beta"], parts["sigma2"], parts["xtvx"], parts["g"]
    cov_beta = sigma2 * np.linalg.inv(xtvx)
    sigma_cov = sigma2 * g
    sd = np.sqrt(np.maximum(np.diag(sigma_cov), 0.0))
    corr = float(sigma_cov[0, 1] / (sd[0] * sd[1])) if sd[0] > 0 and sd[1] > 0 else np.nan
    # lme4's isSingular: a variance parameter at its lower bound (0).
    if theta[0] < 1e-4 or theta[2] < 1e-4:
        warnings.warn(
            "The fit is singular (a standard deviation at 0 or a correlation of ±1): the data "
            "can't separate that part of the variation between archipelagos.",
            stacklevel=2,
        )
    for row, (a, bvec, _) in zip(rows, stats, strict=True):
        # BLUP: G X'V^-1 r, with X'V^-1 r = X'r - A K X'r and X'r = b - A beta.
        k = g @ np.linalg.inv(np.eye(2) + a @ g)
        xtr = bvec - a @ beta
        b = g @ (xtr - a @ k @ xtr)
        row["logc"] = float(beta[0] + b[0])
        row["c"] = float(np.exp(row["logc"]))
        row["z"] = float(beta[1] + b[1])
    groups = pd.DataFrame(rows).set_index("group")
    groups = groups[["n", "logc", "c", "z", "z_alone", "z_alone_se"]]
    return MixedLinPowFit(
        logc=float(beta[0]),
        c=float(np.exp(beta[0])),
        z=float(beta[1]),
        logc_se=float(np.sqrt(cov_beta[0, 0])),
        z_se=float(np.sqrt(cov_beta[1, 1])),
        sd_logc=float(sd[0]),
        sd_z=float(sd[1]),
        corr=corr,
        sigma=float(np.sqrt(sigma2)),
        reml_criterion=criterion,
        n=n,
        con=added,
        groups=groups,
        data=pd.concat(long_parts, ignore_index=True),
    )
