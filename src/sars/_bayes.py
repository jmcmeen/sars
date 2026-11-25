"""A Bayesian hierarchical log-log power model (the ``bayes`` extra, PyMC).

The model of ``lin_pow_mixed``, log S = (log c + u_j) + (z + v_j) log A, with
each archipelago's deviations (u_j, v_j) drawn from one bivariate normal, but
fitted by sampling the posterior instead of by REML: every archipelago's
intercept and slope, the group means, the spread between archipelagos and
their correlation each get a full posterior distribution, and small
archipelagos are pulled toward the group as far as their data allow.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sars._extras import require_extra
from sars._mixed import _prepare


@dataclass(frozen=True)
class BayesLinPowFit:
    """Result of `lin_pow_bayes`: posterior summaries of the hierarchical
    log-log power model.

    Attributes
    ----------
    summary : pd.DataFrame
        The group-level parameters, one row each: ``logc`` and ``z`` (the
        means), ``sd_logc`` and ``sd_z`` (how much archipelagos vary),
        ``corr`` (the correlation of their deviations) and ``sigma`` (the
        residual standard deviation, in log richness); with the posterior
        ``mean``, ``sd``, the ``lower`` and ``upper`` ends of the central
        ``conf`` interval, ``r_hat`` and ``ess_bulk``.
    groups : pd.DataFrame
        One row per archipelago: ``n`` islands, the posterior mean and interval
        of its ``logc``, ``c`` and ``z``, and ``z_alone``, the slope of a
        least-squares fit to it alone, for comparison.
    conf : float
        The level of the intervals.
    divergences : int
        Divergent transitions in the sampler; any at all deserve attention
        (raise ``target_accept``).
    converged : bool
        Every r-hat below 1.01 and no divergences.
    n : int
        Islands, over all archipelagos.
    con : float
        The constant added to richness before taking logs (0 unless an
        island has no species).
    idata : arviz.InferenceData
        The full posterior, for plots and diagnostics with ArviZ.
    """

    summary: pd.DataFrame
    groups: pd.DataFrame
    conf: float
    divergences: int
    converged: bool
    n: int
    con: float = 0.0
    idata: Any = field(default=None, repr=False)

    def predict(self, area: ArrayLike, group: str | None = None) -> np.ndarray:
        """The posterior mean of richness at ``area``: for one archipelago, or
        for the group's mean curve when ``group`` is None. Like `lin_pow`'s,
        on the arithmetic scale, less ``con``."""
        post = self.idata.posterior
        a = np.atleast_1d(np.asarray(area, dtype=float))
        if group is None:
            logc = post["logc"].to_numpy().ravel()
            z = post["z"].to_numpy().ravel()
        else:
            names = list(self.groups["group"])
            if group not in names:
                raise ValueError(f"No archipelago {group!r}; they are {names}.")
            j = names.index(group)
            logc = post["logc_group"].to_numpy()[..., j].ravel()
            z = post["z_group"].to_numpy()[..., j].ravel()
        draws = np.exp(logc[:, None] + z[:, None] * np.log(a)[None, :])
        return draws.mean(axis=0) - self.con


def lin_pow_bayes(
    datasets: Mapping[str, pd.DataFrame] | pd.DataFrame,
    by: str | None = None,
    con: float = 1.0,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    conf: float = 0.95,
    target_accept: float = 0.99,
    seed: int | None = None,
    progressbar: bool = False,
) -> BayesLinPowFit:
    """Fit the log-log power model to several archipelagos with partial
    pooling, by Bayesian sampling (needs ``pip install 'sars[bayes]'``).

    log S_ij = (log c + u_j) + (z + v_j) log A_ij + e_ij, with (u_j, v_j)
    bivariate normal with standard deviations ``sd_logc`` and ``sd_z`` and
    correlation ``corr``, and e_ij normal with standard deviation ``sigma``:
    the model of `lin_pow_mixed`, fitted by NUTS in PyMC rather than REML, so
    that every quantity has a posterior distribution. The priors are weakly
    informative, scaled to the data: the mean intercept is normal around the
    mean log richness with a standard deviation of ten times its spread, the
    mean slope normal(0.25, 1), the standard deviations half-normal (the
    intercepts' on the scale of log richness's spread, the slopes' 0.5, the
    residuals' that spread), and the correlation LKJ(2). The deviations are
    sampled non-centered, which suits the few archipelagos a study usually has.

    Parameters
    ----------
    datasets : mapping of name to DataFrame, or one DataFrame
        The archipelagos, as `lin_pow_mixed` takes them (three or more).
    by : str, optional
        With one DataFrame, the column naming each island's archipelago.
    con : float
        Added to every richness value if any island has no species (1 by
        default), as `lin_pow_mixed` does.
    draws, tune, chains : int
        Posterior draws kept per chain, tuning draws per chain, and chains.
    conf : float
        Level of the central posterior intervals reported (0.95 by default).
    target_accept : float
        NUTS's target acceptance rate. The default, 0.99, takes small steps:
        the funnel-shaped posteriors of hierarchical models with few groups
        gave divergent transitions at 0.9 and 0.95 on the bundled
        archipelagos, and none at 0.99, at about 1.5 times the time.
    seed : int, optional
        Seed for the sampler, for reproducible draws.
    progressbar : bool
        Show PyMC's progress bar and its log messages (quiet by default).

    Returns
    -------
    BayesLinPowFit

    Raises
    ------
    MissingExtraError
        If PyMC isn't installed.
    ValueError
        As `lin_pow_mixed` (fewer than three archipelagos, an archipelago
        with fewer than two islands or one area, bad values), or a ``conf``
        outside 0 to 1.

    Examples
    --------
    >>> fit = sars.lin_pow_bayes({"gala": sars.load_gala(), "aegean": sars.load_aegean(),
    ...                           "xu": sars.load_xu()}, seed=1)  # doctest: +SKIP
    >>> fit.groups[["group", "z", "z_lower", "z_upper", "z_alone"]]  # doctest: +SKIP
    """
    require_extra("bayes", "lin_pow_bayes")
    if not 0 < conf < 1:
        raise ValueError(f"'conf' must be between 0 and 1, got {conf}")
    cleaned, added = _prepare(datasets, by, con, kind="hierarchical model")

    import arviz as az
    import pymc as pm

    names = [name for name, _, _ in cleaned]
    log_area = np.concatenate([np.log(area) for _, area, _ in cleaned])
    log_s = np.concatenate([np.log(species + added) for _, _, species in cleaned])
    index = np.concatenate([np.full(len(area), j) for j, (_, area, _) in enumerate(cleaned)])
    spread = float(np.std(log_s)) or 1.0

    with pm.Model(coords={"group": names, "effect": ["logc", "z"]}):
        logc = pm.Normal("logc", mu=float(np.mean(log_s)), sigma=10 * spread)
        z = pm.Normal("z", mu=0.25, sigma=1.0)
        chol, corr, sds = pm.LKJCholeskyCov(
            "chol",
            n=2,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(sigma=np.array([spread, 0.5]), shape=2),
            compute_corr=True,
        )
        pm.Deterministic("sd_logc", sds[0])
        pm.Deterministic("sd_z", sds[1])
        pm.Deterministic("corr", corr[0, 1])
        raw = pm.Normal("raw", 0.0, 1.0, dims=("group", "effect"))
        deviations = pm.math.dot(raw, chol.T)
        logc_group = pm.Deterministic("logc_group", logc + deviations[:, 0], dims="group")
        z_group = pm.Deterministic("z_group", z + deviations[:, 1], dims="group")
        sigma = pm.HalfNormal("sigma", sigma=spread)
        mean = logc_group[index] + z_group[index] * log_area
        pm.Normal("log_species", mu=mean, sigma=sigma, observed=log_s)
        pymc_log = logging.getLogger("pymc")
        level = pymc_log.level
        if not progressbar:
            pymc_log.setLevel(logging.WARNING)
        try:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=seed,
                progressbar=progressbar,
                compute_convergence_checks=False,
            )
        finally:
            pymc_log.setLevel(level)

    tail = (1 - conf) / 2
    post = idata.posterior
    stats = az.summary(
        idata,
        var_names=["logc", "z", "sd_logc", "sd_z", "corr", "sigma"],
        kind="diagnostics",
    )
    rows = []
    for name in ["logc", "z", "sd_logc", "sd_z", "corr", "sigma"]:
        values = post[name].to_numpy().ravel()
        rows.append(
            {
                "parameter": name,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "lower": float(np.quantile(values, tail)),
                "upper": float(np.quantile(values, 1 - tail)),
                "r_hat": float(stats.loc[name, "r_hat"]),
                "ess_bulk": float(stats.loc[name, "ess_bulk"]),
            }
        )
    summary = pd.DataFrame(rows).set_index("parameter")

    groups = []
    logc_draws = post["logc_group"].to_numpy()
    z_draws = post["z_group"].to_numpy()
    for j, (name, area, species) in enumerate(cleaned):
        lc = logc_draws[..., j].ravel()
        zz = z_draws[..., j].ravel()
        x = np.column_stack([np.ones_like(area), np.log(area)])
        alone = np.linalg.lstsq(x, np.log(species + added), rcond=None)[0]
        groups.append(
            {
                "group": name,
                "n": len(area),
                "logc": float(lc.mean()),
                "logc_lower": float(np.quantile(lc, tail)),
                "logc_upper": float(np.quantile(lc, 1 - tail)),
                "c": float(np.exp(lc).mean()),
                "z": float(zz.mean()),
                "z_lower": float(np.quantile(zz, tail)),
                "z_upper": float(np.quantile(zz, 1 - tail)),
                "z_alone": float(alone[1]),
            }
        )
    divergences = int(idata.sample_stats["diverging"].to_numpy().sum())
    group_stats = az.summary(idata, var_names=["logc_group", "z_group"], kind="diagnostics")
    r_hats = np.concatenate([summary["r_hat"].to_numpy(), group_stats["r_hat"].to_numpy()])
    converged = bool(np.all(r_hats < 1.01) and divergences == 0)
    return BayesLinPowFit(
        summary=summary,
        groups=pd.DataFrame(groups),
        conf=conf,
        divergences=divergences,
        converged=converged,
        n=len(log_s),
        con=added,
        idata=idata,
    )
