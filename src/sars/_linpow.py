"""The log-log power model, fitted by ordinary least squares."""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import stdtr

from sars._models import SARFit, _fit_nls, _validate_data

# The accepted spellings of each log base, and the base itself.
_LOG_BASES: dict[str, float] = {"e": float(np.e), "2": 2.0, "10": 10.0}

# Sums of squares smaller than this fraction of their scale are rounding error
# (which is near 1e-32 of it), not variation in the data.
_NOISE = 1e-20


def _parse_log_base(log_base: object) -> str:
    """The key in ``_LOG_BASES`` for a base given as a string or a number."""
    if isinstance(log_base, str):
        key = log_base.strip().lower()
        if key in _LOG_BASES:
            return key
    elif isinstance(log_base, numbers.Real) and not isinstance(log_base, bool):
        for key, value in _LOG_BASES.items():
            if np.isclose(float(log_base), value, rtol=1e-9, atol=0.0):
                return key
    raise ValueError(f"Unsupported log_base: {log_base!r}. Choose from 'e', '2' or '10'.")


@dataclass
class LinPowFit:
    """Result of fitting the log-log power model, log S = log c + z log A.

    The standard errors, t and p values, and both R² are those of the straight
    line on the log-log axes, not of the curve on the arithmetic scale, so R²
    isn't comparable with a ``SARFit``'s.

    Attributes
    ----------
    logc : float
        Intercept of the line: log c, in the base given by ``log_base``.
    c : float
        ``log_base ** logc``, the power model's c. It is the same whatever
        the log base.
    z : float
        Slope of the line, the power model's z. It doesn't depend on the log
        base.
    logc_se, z_se : float
        Standard errors of ``logc`` and ``z``. Both are 0 when the line passes
        through every point.
    c_se : float
        Approximate standard error of ``c``, from ``logc_se`` by the delta
        method (``c * ln(base) * logc_se``). c's sampling distribution is
        skewed, so prefer ``logc`` and ``logc_se`` for inference.
        ``c`` itself is infinite when ``log_base ** logc`` is beyond the
        largest float (a steep line over very small areas); ``predict`` works
        from ``logc`` and isn't affected.
    logc_t, z_t : float
        t statistics (estimate over standard error) on n - 2 degrees of freedom.
    logc_p, z_p : float
        Two-sided p values for ``logc = 0`` (that is, c = 1) and ``z = 0``.
        When the line passes through every point there is no residual error
        to test against: a coefficient that isn't zero then has an infinite t
        and a p of 0, and one that is zero has NaN for both.
    r_squared : float
        Coefficient of determination of the line, in log space. NaN when every
        island has the same richness.
    adj_r_squared : float
        ``r_squared`` adjusted for the two coefficients.
    sigma : float
        Residual standard error in log space (in units of ``log_base``).
    n : int
        Number of observations.
    log_base : str
        Base of the logarithms: "e", "2" or "10".
    con : float
        The constant that was added to every richness value before taking logs:
        the ``con`` argument when an island has no species, otherwise 0.
    data : pd.DataFrame
        Original data (columns: area, species), before ``con`` was added.
    power : SARFit | None
        The nonlinear power model (``sar_power``'s least squares on the
        arithmetic scale) fitted to the same data, when ``compare=True``.
        ``comparison`` sets its c and z beside this fit's. Its ``converged``
        is False, and its parameters NaN, if that fit failed.
    """

    logc: float
    c: float
    z: float
    logc_se: float
    c_se: float
    z_se: float
    logc_t: float
    z_t: float
    logc_p: float
    z_p: float
    r_squared: float
    adj_r_squared: float
    sigma: float
    n: int
    log_base: str
    con: float
    data: pd.DataFrame = field(repr=False)
    power: SARFit | None = None

    def predict(self, area: float | np.ndarray) -> np.ndarray:
        """Predict species richness for given area value(s).

        Parameters
        ----------
        area : float or array-like
            Area value(s) in original (non-log) space.

        Returns
        -------
        np.ndarray
            Predicted richness on the arithmetic scale, ``c * area ** z``, less
            ``con`` when a constant was added to the richness before fitting
            (which can take a small island's prediction below zero). The line
            is fitted to log richness, so this estimates the median richness at
            that area rather than the mean: no correction for the
            back-transformation is applied.
        """
        area = np.asarray(area, dtype=float)
        base = _LOG_BASES[self.log_base]
        # From the line itself rather than c * area ** z: c can overflow or
        # underflow where the prediction at these areas is an ordinary number.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_s = self.logc + self.z * (np.log(area) / np.log(base))
            return base**log_s - self.con

    @property
    def comparison(self) -> pd.DataFrame | None:
        """c and z from this fit and from the nonlinear power model, one row
        each ("log-log" and "nonlinear"), with whether each fit converged (the
        nonlinear one's c and z are NaN if it didn't). None unless fitted with
        ``compare=True``."""
        if self.power is None:
            return None
        return pd.DataFrame(
            {
                "c": [self.c, self.power.params["c"]],
                "z": [self.z, self.power.params["z"]],
                "converged": [True, self.power.converged],
            },
            index=pd.Index(["log-log", "nonlinear"], name="fit"),
        )

    def __repr__(self) -> str:
        parts = [
            f"logc={self.logc:.4f}",
            f"c={self.c:.4f}",
            f"z={self.z:.4f}",
            f"R²={self.r_squared:.4f}",
            f"log_base='{self.log_base}'",
        ]
        if self.con:
            parts.append(f"con={self.con:g}")
        if self.power is not None and not self.power.converged:
            parts.append("nonlinear: not converged")
        elif self.power is not None:
            p = self.power.params
            parts.append(f"nonlinear: c={p['c']:.4f} z={p['z']:.4f}")
        return f"LinPowFit({', '.join(parts)})"


def lin_pow(
    data: pd.DataFrame,
    con: float = 1.0,
    log_base: str | float = "e",
    compare: bool = False,
) -> LinPowFit:
    """Fit the log-log power model: log S = log c + z log A.

    The power model made linear by taking logs of both axes and fitted by
    ordinary least squares, the commonest way the species-area relationship has
    been fitted. It minimises squared error in log richness, so it gives a
    different c and z from ``sar_power``, which minimises it in richness itself.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'. Unlike the other fitting
        functions, islands with no species are accepted.
    con : float
        Constant added to every richness value when at least one island has no
        species, so that the logs exist. It isn't used otherwise.
    log_base : {"e", "2", "10"}
        Base of the logarithms. The bases may also be given as numbers (2,
        10.0, ``math.e``). It changes ``logc`` and its standard error, not c,
        z or the fit.
    compare : bool
        Also fit the nonlinear power model to the same data, as
        ``LinPowFit.power``, to compare its c and z with this fit's. A
        warning is given if that fit doesn't converge.

    Returns
    -------
    LinPowFit

    Raises
    ------
    ValueError
        If ``log_base`` isn't one of the bases above, ``con`` isn't a finite
        number, there are fewer than three observations (the standard errors
        need a residual degree of freedom), the areas are all equal, richness
        is negative or zero on every island, or an island has no species and
        ``con`` isn't positive.

    References
    ----------
    Arrhenius O (1921) Species and area. Journal of Ecology 9:95-99.

    Rosenzweig ML (1995) Species Diversity in Space and Time. Cambridge
    University Press.
    """
    base_key = _parse_log_base(log_base)
    base = _LOG_BASES[base_key]
    if isinstance(con, bool) or not isinstance(con, numbers.Real) or not np.isfinite(con):
        raise ValueError(f"'con' must be a finite number, got {con!r}")

    _validate_data(data, allow_zero_species=True)
    area = data["area"].to_numpy(dtype=float)
    species = data["species"].to_numpy(dtype=float)
    n = len(area)
    if n < 3:
        raise ValueError(f"Need at least 3 observations for the log-log fit, got {n}")
    if not (species > 0).any():
        raise ValueError("No island has any species: there is no relationship to fit")

    # The constant is added to every island, and only when one has no species:
    # data without zeros are fitted as they are, whatever con is.
    added = 0.0
    if (species == 0).any():
        if con <= 0:
            raise ValueError(
                f"An island has no species, so 'con' must be positive to take logs; got {con!r}"
            )
        added = float(con)

    x = np.log(area) / np.log(base)
    y = np.log(species + added) / np.log(base)

    # scipy.stats.linregress would give the slope's statistics, but not the
    # intercept's t and p values or sigma, and it compares exactly where the
    # degenerate cases below need a tolerance, so the line is fitted here.
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    sxx = float(x_dev @ x_dev)
    tss = float(y_dev @ y_dev)
    # Areas that agree to rounding leave the slope undefined.
    if sxx <= _NOISE * max(1.0, x.mean() ** 2):
        raise ValueError("The log-log fit needs at least two different areas")
    # Equal richness leaves only rounding error in y_dev: the line is flat.
    flat = tss <= _NOISE * float(y @ y)
    z = 0.0 if flat else float(x_dev @ y_dev) / sxx
    logc = float(y.mean() - z * x.mean())

    resid = y - (logc + z * x)
    rss = float(resid @ resid)
    df = n - 2
    # A line through every point leaves no residual error, only rounding.
    exact = flat or rss <= _NOISE * tss
    if exact:
        sigma = z_se = logc_se = 0.0
        # With no error, a coefficient that isn't zero is infinitely far from
        # it (t infinite, p 0); for one that is zero the test is undefined.
        z_t, z_p = (float("nan"),) * 2 if flat else (float(np.copysign(np.inf, z)), 0.0)
        if abs(logc) <= 1e-12 * max(1.0, float(np.abs(y).max())):
            logc_t = logc_p = float("nan")
        else:
            logc_t, logc_p = float(np.copysign(np.inf, logc)), 0.0
    else:
        sigma = float(np.sqrt(rss / df))
        z_se = float(sigma / np.sqrt(sxx))
        logc_se = float(sigma * np.sqrt(1.0 / n + x.mean() ** 2 / sxx))
        logc_t = logc / logc_se
        z_t = z / z_se
        logc_p = float(2.0 * stdtr(df, -abs(logc_t)))
        z_p = float(2.0 * stdtr(df, -abs(z_t)))

    if flat:
        r_squared = adj_r_squared = float("nan")
    elif exact:
        r_squared = adj_r_squared = 1.0
    else:
        r_squared = 1.0 - rss / tss
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / df

    # A steep line over tiny areas can put c beyond the largest float: it is
    # then infinite, and predict() still works from logc.
    with np.errstate(over="ignore", invalid="ignore"):
        c = float(np.float64(base) ** logc)
        c_se = float(c * np.log(base) * logc_se) if logc_se else 0.0

    power = None
    if compare:
        # The comparison fit is to the richness as given, zeros included: least
        # squares on the arithmetic scale needs no constant.
        power = _fit_nls("power", data, allow_zero_species=True)
        if not power.converged:
            warnings.warn(
                "The nonlinear power model did not converge, so there are no "
                "nonlinear c and z to compare with.",
                stacklevel=2,
            )

    return LinPowFit(
        logc=logc,
        c=c,
        z=z,
        logc_se=logc_se,
        c_se=c_se,
        z_se=z_se,
        logc_t=float(logc_t),
        z_t=float(z_t),
        logc_p=logc_p,
        z_p=z_p,
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        sigma=sigma,
        n=n,
        log_base=base_key,
        con=added,
        data=data,
        power=power,
    )
