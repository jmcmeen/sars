"""SAR model definitions and fitting routines."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import least_squares

from sars._checks import CheckOptions, ResidualChecks, run_checks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SARFit dataclass
# ---------------------------------------------------------------------------


@dataclass
class SARFit:
    """Result of fitting a single SAR model.

    Attributes
    ----------
    model : str
        Short model identifier matching R sars function suffix (e.g. 'power').
    params : dict[str, float]
        Fitted parameter values keyed by parameter name.
    r_squared : float
        Coefficient of determination in arithmetic space.
    aic : float
        Akaike Information Criterion (normal log-likelihood convention).
    aicc : float
        AIC corrected for small sample size.
    bic : float
        Bayesian Information Criterion.
    n : int
        Number of observations used in fit.
    converged : bool
        Whether the NLS solver converged.
    data : pd.DataFrame
        Original data (columns: area, species).

    See Also
    --------
    SARFit.param_table : standard errors, t tests and confidence intervals
        for the fitted parameters.
    """

    model: str
    params: dict
    r_squared: float
    aic: float
    aicc: float
    bic: float
    n: int
    converged: bool
    data: pd.DataFrame

    def predict(self, area: float | np.ndarray) -> np.ndarray:
        """Predict species richness for given area value(s)."""
        area = np.asarray(area, dtype=float)
        spec = _MODEL_REGISTRY.get(self.model)
        if spec is None:
            raise NotImplementedError(f"predict not yet implemented for '{self.model}'")
        return spec.func(area, *[self.params[k] for k in spec.param_names])

    def observed_shape(self) -> str:
        """The shape the fitted curve takes over the observed range of areas.

        A model's general shape (the ``shape`` column of ``sar_multi``'s
        summary) says what the model can do; this says what the fit did. The
        curve is evaluated between the smallest and largest observed area, on
        arithmetic axes, and classified by how its slope changes:

        - ``"linear"``: the curve follows the straight line through its end
          points (a flat curve included).
        - ``"convex"``: richness rises and the slope only falls (the curve
          decelerates; "convex upward" in the SAR literature).
        - ``"sigmoid"``: richness rises and the slope first rises, then falls
          (one inflection point inside the observed range).
        - ``"none"``: anything else: richness falls somewhere, the curve only
          accelerates, or it has more than one inflection point.

        Each test allows a departure of a thousandth of the curve's range of
        richness, so a bend or a dip too small to see doesn't change the
        label. A sigmoid model whose inflection point lies below the smallest
        area is therefore ``"convex"`` here: over the data it only decelerates.

        Returns
        -------
        str
            ``"linear"``, ``"convex"``, ``"sigmoid"`` or ``"none"``. ``"none"``
            too if the fit didn't converge or the curve isn't finite.
        """
        if not self.converged:
            return "none"
        area = np.asarray(self.data["area"], dtype=float)
        return _curve_shape(self.predict, float(area.min()), float(area.max()))

    def residual_checks(
        self,
        normality: str | None = "shapiro",
        homoscedasticity: str | None = "cor.fitted",
        cor_method: str = "spearman",
    ) -> ResidualChecks:
        """Test the residuals for normality and constant variance.

        Least squares, and the information criteria built on it, assume
        normal residuals of constant variance. These are the checks R sars
        offers before model averaging (``normaTest``, ``homoTest`` and
        ``neg_check``). ``sar_multi``, ``sar_average`` and ``bootstrap_ci``
        can leave out the models that fail them.

        Parameters
        ----------
        normality : str or None
            Normality test for the residuals: ``"shapiro"`` (Shapiro-Wilk, the
            default), ``"kolmogorov"`` (Kolmogorov-Smirnov against a normal
            distribution with the residuals' mean and standard deviation), or
            None for no test. Estimating the mean and standard deviation from
            the residuals makes the Kolmogorov-Smirnov p value too large, so
            that test rejects less often than its alpha suggests. R sars's
            ``"kolmo"`` is a different test: it compares the residuals with a
            standard normal, so its result depends on the units of richness.
            R's ``"lillie"`` (Lilliefors) is not available.
        homoscedasticity : str or None
            Homoscedasticity test: the correlation of the squared residuals
            with the fitted values (``"cor.fitted"``, the default) or with
            area (``"cor.area"``), or None for no test.
        cor_method : str
            The correlation used by the homoscedasticity test: ``"spearman"``
            (default), ``"pearson"`` or ``"kendall"``. p values are computed
            as R's ``ks.test`` and ``cor.test`` compute them: exact for the
            Kolmogorov-Smirnov test below 100 untied residuals and asymptotic
            otherwise, exact for Kendall's tau below
            50 untied observations and for Spearman's rho up to 9, a series
            approximation for Spearman's rho beyond that, and the usual t or
            normal approximation when values are tied. (Where that series
            gives R a Spearman p value of exactly 0, far in the tail, this
            reports the t distribution's small positive value.)

        Returns
        -------
        ResidualChecks
            The test statistics and p values, and whether any fitted value at
            the observed areas is negative. If a fitted value isn't finite,
            no test is run and ``finite_fitted`` is False.

        Raises
        ------
        ValueError
            If a test or correlation name is not one of those above.
        """
        return self._run_checks(
            CheckOptions(
                normality=normality, homoscedasticity=homoscedasticity, cor_method=cor_method
            )
        )

    def _run_checks(self, options: CheckOptions) -> ResidualChecks:
        """``residual_checks`` for options that are already validated."""
        area = self.data["area"].to_numpy(dtype=float)
        species = self.data["species"].to_numpy(dtype=float)
        return run_checks(self.model, area, species, self.predict(area), options)

    def param_table(self, conf: float = 0.95) -> pd.DataFrame:
        """Standard errors, t tests and confidence intervals for the parameters.

        The usual large-sample (Wald) inference for nonlinear least squares:
        with ``J`` the Jacobian of the model with respect to its parameters at
        the fitted values, ``n`` observations and ``p`` parameters, the
        parameters' covariance is ``sigma^2 (J'J)^-1`` with
        ``sigma^2 = RSS / (n - p)``. Each parameter's t value is its estimate
        over its standard error, tested against zero on ``n - p`` degrees of
        freedom (two-sided), and its interval is
        ``estimate ± t(1 - (1 - conf) / 2, n - p) * std_error``.

        The table is computed on each call from the fit's data and parameters.
        The Jacobian is numerical (central differences).

        Parameters
        ----------
        conf : float, default 0.95
            Confidence level of the interval, between 0 and 1.

        Returns
        -------
        pd.DataFrame
            One row per parameter (the index, in the model's parameter order),
            with columns ``estimate``, ``std_error``, ``t_value``, ``p_value``,
            ``ci_lower`` and ``ci_upper``.

        Warns
        -----
        UserWarning
            If the data don't identify some of the parameters: the model's
            curve doesn't change when they do, or changes the same way as when
            other parameters do (a flat ridge in the fit). Their standard
            errors, and the columns computed from them, are NaN. Likewise for
            a parameter the curve can't be differentiated reliably in at the
            fitted values, and any other whose standard error depends on it,
            and for a parameter that stopped on one of the bounds the fit
            searches within, where the approximation doesn't hold (the other
            parameters' standard errors are given as if it were free).

        Raises
        ------
        ValueError
            If ``conf`` is not between 0 and 1, or if ``data`` has changed
            since the fit (the table is computed from it).

        Notes
        -----
        Every column but ``estimate`` is NaN when the fit has no residual
        degrees of freedom (``n <= p``), and the whole table is NaN for a fit
        that did not converge.

        The standard errors, t values and p values are the ones R ``sars``
        reports in a fit's ``sigConf``. Its interval columns are
        ``estimate ± 2 * std_error`` whatever the sample size; the t quantile
        used here is larger than 2 for small samples (2.14 for 16 islands and
        two parameters), so these intervals are slightly wider.

        These are approximations that assume the curve is close to linear in
        the parameters near the optimum, and that the optimum is not on one of
        the bounds the fit searches within. An interval can extend past values
        the model allows (a negative ``c`` for the power model). For
        uncertainty in the fitted curve itself, see ``bootstrap_ci``.

        Examples
        --------
        >>> import sars
        >>> fit = sars.sar_power(sars.load_galap())
        >>> fit.param_table().round(4)
                   estimate  std_error  t_value  p_value  ci_lower  ci_upper
        parameter
        c           33.1803    19.2423   1.7243   0.1066   -8.0903   74.4509
        z            0.2832     0.0994   2.8499   0.0129    0.0701    0.4963
        """
        if not 0.0 < conf < 1.0:
            raise ValueError(f"conf must be between 0 and 1, got {conf}")
        names = list(self.params)
        estimate = np.array([self.params[k] for k in names], dtype=float)
        std_error = np.full(len(names), np.nan)
        dof = self.n - len(names)
        if self.converged and dof > 0:
            spec = _MODEL_REGISTRY.get(self.model)
            if spec is None:
                raise NotImplementedError(f"param_table not yet implemented for '{self.model}'")
            area, species = self._fitted_data(spec)
            std_error, rough = _param_std_errors(spec.func, area, species, estimate)
            missing = [n for n, se in zip(names, std_error, strict=True) if np.isnan(se)]
            on_bound = _on_bound(self.model, estimate, area, species)
            if on_bound.any():
                std_error[on_bound] = np.nan
                warnings.warn(
                    f"Model '{self.model}': "
                    f"{', '.join(n for n, b in zip(names, on_bound, strict=True) if b)} "
                    f"stopped on a bound of the fit's search, where a standard error "
                    f"doesn't apply (NaN in param_table); the fit's optimum there is the "
                    f"bound's, not the data's",
                    UserWarning,
                    stacklevel=2,
                )
            if rough.any():
                which = ", ".join(n for n, bad in zip(names, rough, strict=True) if bad)
                warnings.warn(
                    f"Model '{self.model}': the curve can't be differentiated reliably with "
                    f"respect to {which} at the fitted values, as happens far along a flat "
                    f"ridge of the fit, so {', '.join(missing)} have no standard error "
                    f"(NaN in param_table)",
                    UserWarning,
                    stacklevel=2,
                )
            elif missing:
                warnings.warn(
                    f"Model '{self.model}': the data don't identify {', '.join(missing)}, "
                    f"so no standard error can be given (NaN in param_table)",
                    UserWarning,
                    stacklevel=2,
                )
        # A perfect fit has standard errors of 0, and so infinite t values.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_value = estimate / std_error
        if dof > 0:
            # Student's t from scipy.special: scipy.stats would add a tenth of a
            # second to every import of the package.
            p_value = 2.0 * special.stdtr(dof, -np.abs(t_value))
            half_width = special.stdtrit(dof, 0.5 + conf / 2.0) * std_error
        else:
            p_value = half_width = np.full(len(names), np.nan)
        return pd.DataFrame(
            {
                "estimate": estimate,
                "std_error": std_error,
                "t_value": t_value,
                "p_value": p_value,
                "ci_lower": estimate - half_width,
                "ci_upper": estimate + half_width,
            },
            index=pd.Index(names, name="parameter"),
        )

    def _fitted_data(self, spec: _ModelSpec) -> tuple[np.ndarray, np.ndarray]:
        """The area and richness the model was fitted to, as arrays.

        ``data`` is the caller's DataFrame, not a copy, so it can have changed
        since the fit. Statistics computed from data the parameters weren't
        fitted to would be wrong without any sign of it, so this raises instead.
        """
        area = np.asarray(self.data["area"], dtype=float)
        species = np.asarray(self.data["species"], dtype=float)
        changed = len(area) != self.n
        if not changed and np.isfinite(self.r_squared):
            # The fit's R² is reproduced exactly by the data it was fitted to.
            with np.errstate(all="ignore"):
                values = [self.params[k] for k in spec.param_names]
                rss = float(np.sum((species - spec.func(area, *values)) ** 2))
                ss_tot = float(np.sum((species - np.mean(species)) ** 2))
                r_squared = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan
            changed = not np.isclose(r_squared, self.r_squared, rtol=1e-9, atol=1e-9)
        if changed:
            raise ValueError(
                f"the data of this '{self.model}' fit have changed since it was fitted "
                f"({self.n} observations, R² {self.r_squared:.6g}); fit the model again"
            )
        return area, species

    def __repr__(self) -> str:
        p = "  ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        return f"SARFit(model='{self.model}', {p}, R²={self.r_squared:.4f}, AICc={self.aicc:.2f})"


# ---------------------------------------------------------------------------
# Observed shape
# ---------------------------------------------------------------------------

# How far, as a fraction of the curve's range of richness, a curve may stray
# from a shape and still be given it. One tolerance for every test, in richness
# and not in slope: areas often span many orders of magnitude, and a slope
# among the largest islands is tiny beside one among the smallest.
_SHAPE_TOL = 1e-3
_SHAPE_GRID = 300


def _gap_below_hull(x: np.ndarray, y: np.ndarray) -> float:
    """The most the curve dips below the tightest concave curve drawn over it
    (its upper hull): zero for a curve whose slope only falls."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # Drop b if it lies on or below the line from a to the new point.
            if (y[b] - y[a]) * (x[i] - x[a]) <= (y[i] - y[a]) * (x[b] - x[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    return float(np.max(np.interp(x, x[hull], y[hull]) - y))


def _curve_shape(predict: Callable[[np.ndarray], np.ndarray], lo: float, hi: float) -> str:
    """Classify ``predict`` on [lo, hi] as linear, convex, sigmoid or none (see
    ``SARFit.observed_shape``)."""
    if not hi > lo:
        return "none"
    # Island areas are right-skewed, and a curve bends most among the small
    # ones: an even grid alone would step over that, so add a geometric one.
    x = np.unique(
        np.concatenate([np.linspace(lo, hi, _SHAPE_GRID), np.geomspace(lo, hi, _SHAPE_GRID)])
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = np.asarray(predict(x), dtype=float)
    if not np.all(np.isfinite(y)):
        return "none"

    spread = float(y.max() - y.min())
    if spread <= 1e-9 * float(np.max(np.abs(y))):
        return "linear"  # flat: a line of slope zero
    tol = _SHAPE_TOL * spread
    chord = y[0] + (y[-1] - y[0]) * (x - lo) / (hi - lo)
    if np.max(np.abs(y - chord)) <= tol:
        return "linear"
    if np.max(np.maximum.accumulate(y) - y) > tol:
        return "none"  # richness falls somewhere
    if _gap_below_hull(x, y) <= tol:
        return "convex"
    if _gap_below_hull(x, -y) <= tol:
        return "none"  # it only accelerates

    # Sigmoid: accelerating up to some point, decelerating after it. The
    # longer the accelerating stretch, the easier the rest passes, so find the
    # longest one (its gap only grows with length: a binary search).
    first, last = 1, len(x) - 1
    while first < last:
        mid = (first + last + 1) // 2
        if _gap_below_hull(x[: mid + 1], -y[: mid + 1]) <= tol:
            first = mid
        else:
            last = mid - 1
    return "sigmoid" if _gap_below_hull(x[first:], y[first:]) <= tol else "none"


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------


def _compute_ic(k: int, n: int, rss: float) -> tuple[float, float, float]:
    """Compute AIC, AICc, and BIC from residual sum of squares.

    Uses the normal log-likelihood convention consistent with R sars:
        logL = -n/2 * log(2*pi*rss/n) - n/2

    Parameters
    ----------
    k : int
        Number of estimated parameters (including sigma).
    n : int
        Number of observations.
    rss : float
        Residual sum of squares.

    Returns
    -------
    tuple[float, float, float]
        (AIC, AICc, BIC)
    """
    log_lik = -n / 2.0 * np.log(2.0 * np.pi * rss / n) - n / 2.0
    aic = -2.0 * log_lik + 2.0 * k
    if n - k - 1 > 0:
        aicc = aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    else:
        aicc = np.inf
    bic = -2.0 * log_lik + k * np.log(n)
    return aic, aicc, bic


# ---------------------------------------------------------------------------
# Model specification registry
# ---------------------------------------------------------------------------


# A SAR model function, f(area, p1, p2, ...) -> predicted S. Each model takes a
# fixed number of parameters, named by its spec's ``param_names``, which a
# Callable type can't express for every arity at once.
_SARFunc = Callable[..., np.ndarray]


@dataclass
class _ModelSpec:
    """Internal specification for a SAR model."""

    name: str
    func: _SARFunc
    param_names: list[str]
    # Search limits on the rescaled parameters (see _RESCALE), not on the ones
    # reported: a fit in other units can report values beyond them.
    bounds_lower: list[float]
    bounds_upper: list[float]
    # One list of starting values per param, on the rescaled data the fit runs on
    # (see _RESCALE): area over its geometric mean, species over its maximum.
    start_grid: list[list[float]]
    # Parameters the solver searches on the natural-log scale, for ones whose
    # optimum can sit many orders of magnitude from the start values (weibull4's
    # c falls below 1e-20 on the Galápagos data). Their bounds and start values are still given, and their
    # fitted values reported, on the natural scale; a lower bound of 0 becomes
    # exp(_LOG_FLOOR).
    log_params: tuple[str, ...] = ()


# Smallest log value a log-scale parameter can take (exp(-700) is about 1e-304).
_LOG_FLOOR = -700.0


def _to_search_scale(values: Sequence[float] | np.ndarray, log_idx: list[int]) -> list[float]:
    out = [float(v) for v in values]
    for i in log_idx:
        out[i] = max(float(np.log(out[i])) if out[i] > 0 else -np.inf, _LOG_FLOOR)
    return out


def _to_natural_scale(values: Sequence[float] | np.ndarray, log_idx: list[int]) -> np.ndarray:
    out = np.array(values, dtype=float)
    out[log_idx] = np.exp(out[log_idx])
    return out


_MODEL_REGISTRY: dict[str, _ModelSpec] = {}


def _register(spec: _ModelSpec) -> _ModelSpec:
    if spec.name not in _RESCALE:
        raise ValueError(f"model {spec.name!r} needs a rescaling entry in _RESCALE")
    _MODEL_REGISTRY[spec.name] = spec
    return spec


# ---------------------------------------------------------------------------
# Rescaling
# ---------------------------------------------------------------------------

# Each model is fitted on rescaled data, A' = A / a and S' = S / s, where a is
# the geometric mean of the areas and s the largest richness, and its
# parameters converted back. The start grids and bounds are then in units
# near 1 whatever the data's units, and a model whose form survives a change
# of units gives the same fit (the same R2, and parameters converted exactly)
# in km2 or m2, for tens of species or thousands. Each entry maps rescaled
# parameters p' to the data's units, and back, in the spec's parameter order.
# epm1 and epm2 change form with the area units (their exponent depends on A
# itself), so only their richness is rescaled.
_Rescale = Callable[[np.ndarray, float, float], list[float]]
_RESCALE: dict[str, tuple[_Rescale, _Rescale]] = {
    # S = c A^z: c = s c' a^-z
    "power": (
        lambda p, a, s: [s * p[0] * a ** -p[1], p[1]],
        lambda p, a, s: [p[0] * a ** p[1] / s, p[1]],
    ),
    # S = f + c A^z: f = s f', c = s c' a^-z
    "powerR": (
        lambda p, a, s: [s * p[0], s * p[1] * a ** -p[2], p[2]],
        lambda p, a, s: [p[0] / s, p[1] * a ** p[2] / s, p[2]],
    ),
    # S = c + z log A: z = s z', c = s (c' - z' log a)
    "loga": (
        lambda p, a, s: [s * (p[0] - p[1] * np.log(a)), s * p[1]],
        lambda p, a, s: [p[0] / s + p[1] / s * np.log(a), p[1] / s],
    ),
    # S = c + m A: c = s c', m = s m' / a
    "linear": (
        lambda p, a, s: [s * p[0], s * p[1] / a],
        lambda p, a, s: [p[0] / s, p[1] * a / s],
    ),
    # epm1 and epm2: c = s c' (richness only)
    "epm1": (
        lambda p, a, s: [s * p[0], p[1], p[2]],
        lambda p, a, s: [p[0] / s, p[1], p[2]],
    ),
    "epm2": (
        lambda p, a, s: [s * p[0], p[1], p[2]],
        lambda p, a, s: [p[0] / s, p[1], p[2]],
    ),
    # S = c A^z exp(-d A): c = s c' a^-z, d = d' / a
    "p1": (
        lambda p, a, s: [s * p[0] * a ** -p[1], p[1], p[2] / a],
        lambda p, a, s: [p[0] * a ** p[1] / s, p[1], p[2] * a],
    ),
    # S = c A^z exp(-d / A): c = s c' a^-z, d = d' a
    "p2": (
        lambda p, a, s: [s * p[0] * a ** -p[1], p[1], p[2] * a],
        lambda p, a, s: [p[0] * a ** p[1] / s, p[1], p[2] / a],
    ),
    # S = c log(1 + A / z): c = s c', z = z' a
    "koba": (
        lambda p, a, s: [s * p[0], p[1] * a],
        lambda p, a, s: [p[0] / s, p[1] / a],
    ),
    # S = d A / (c + A): d = s d', c = c' a
    "monod": (
        lambda p, a, s: [s * p[0], p[1] * a],
        lambda p, a, s: [p[0] / s, p[1] / a],
    ),
    # S = d (1 - exp(-z A)): d = s d', z = z' / a
    "negexpo": (
        lambda p, a, s: [s * p[0], p[1] / a],
        lambda p, a, s: [p[0] / s, p[1] * a],
    ),
    # S = d - c exp(-z A): d = s d', c = s c', z = z' / a
    "asymp": (
        lambda p, a, s: [s * p[0], s * p[1], p[2] / a],
        lambda p, a, s: [p[0] / s, p[1] / s, p[2] * a],
    ),
    # S = (c + z A) / (1 + d A): c = s c', z = s z' / a, d = d' / a
    "ratio": (
        lambda p, a, s: [s * p[0], s * p[1] / a, p[2] / a],
        lambda p, a, s: [p[0] / s, p[1] * a / s, p[2] * a],
    ),
    # S = d / (1 + c A^-z): d = s d', c = c' a^z
    "mmf": (
        lambda p, a, s: [s * p[0], p[1] * a ** p[2], p[2]],
        lambda p, a, s: [p[0] / s, p[1] * a ** -p[2], p[2]],
    ),
    # S = d exp(-exp(-z (A - c))): d = s d', z = z' / a, c = c' a
    "gompertz": (
        lambda p, a, s: [s * p[0], p[1] / a, p[2] * a],
        lambda p, a, s: [p[0] / s, p[1] * a, p[2] / a],
    ),
    # S = d (1 - exp(-c A^z)): d = s d', c = c' a^-z
    "weibull3": (
        lambda p, a, s: [s * p[0], p[1] * a ** -p[2], p[2]],
        lambda p, a, s: [p[0] / s, p[1] * a ** p[2], p[2]],
    ),
    # S = d (1 - exp(-c A^z))^f: d = s d', c = c' a^-z
    "weibull4": (
        lambda p, a, s: [s * p[0], p[1] * a ** -p[2], p[2], p[3]],
        lambda p, a, s: [p[0] / s, p[1] * a ** p[2], p[2], p[3]],
    ),
    # S = d (1 - exp(-z A))^c: d = s d', z = z' / a
    "chapman": (
        lambda p, a, s: [s * p[0], p[1] / a, p[2]],
        lambda p, a, s: [p[0] / s, p[1] * a, p[2]],
    ),
    # S = d (1 - (1 + (A / c)^z)^-f): d = s d', c = c' a
    "betap": (
        lambda p, a, s: [s * p[0], p[1] * a, p[2], p[3]],
        lambda p, a, s: [p[0] / s, p[1] / a, p[2], p[3]],
    ),
    # S = c / (f + A^-z): c = s c' a^-z, f = f' a^-z
    "heleg": (
        lambda p, a, s: [s * p[0] * a ** -p[2], p[1] * a ** -p[2], p[2]],
        lambda p, a, s: [p[0] * a ** p[2] / s, p[1] * a ** p[2], p[2]],
    ),
}
# Models whose area isn't rescaled (see above).
_AREA_UNSCALED = {"epm1", "epm2"}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_data(data: pd.DataFrame, allow_zero_species: bool = False) -> None:
    """Validate a SAR input DataFrame.

    Checks that *data* has the required ``area`` and ``species`` columns,
    contains at least two rows, and that all values are finite and positive
    (richness may be zero with ``allow_zero_species``).

    Parameters
    ----------
    data : pd.DataFrame
    allow_zero_species : bool
        Accept islands with no species (richness still can't be negative).
        Only the log-log power model takes them, by adding a constant.

    Raises
    ------
    TypeError
        If *data* is not a DataFrame.
    KeyError
        If required columns are missing.
    ValueError
        If data is empty, contains non-positive values, or non-finite values.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")

    missing = [c for c in ("area", "species") if c not in data.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. DataFrame has columns: {list(data.columns)}"
        )

    if len(data) < 2:
        raise ValueError(f"Need at least 2 observations, got {len(data)}")

    area = data["area"]
    species = data["species"]

    if not np.isfinite(area.to_numpy(dtype=float)).all():
        raise ValueError("'area' contains NaN or infinite values")
    if not np.isfinite(species.to_numpy(dtype=float)).all():
        raise ValueError("'species' contains NaN or infinite values")
    if (area.to_numpy(dtype=float) <= 0).any():
        raise ValueError("'area' must contain only positive values")
    if allow_zero_species:
        if (species.to_numpy(dtype=float) < 0).any():
            raise ValueError("'species' must not contain negative values")
    elif (species.to_numpy(dtype=float) <= 0).any():
        raise ValueError("'species' must contain only positive values")


# ---------------------------------------------------------------------------
# Generic NLS fitter
# ---------------------------------------------------------------------------


def _fit_nls(
    name: str,
    data: pd.DataFrame,
    start_from: dict[str, float] | None = None,
    allow_zero_species: bool = False,
) -> SARFit:
    """Fit a model from the registry using multi-start NLS.

    Parameters
    ----------
    name : str
        Model name (key in _MODEL_REGISTRY).
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    start_from : dict[str, float], optional
        If provided, use these parameter values as the single starting point
        instead of the full grid search. Used by bootstrap to avoid the
        expensive grid search on each resample.
    allow_zero_species : bool
        Accept islands with no species. Least squares on the arithmetic scale
        has no trouble with them; the public fitting functions reject them.

    Returns
    -------
    SARFit
    """
    _validate_data(data, allow_zero_species=allow_zero_species)
    spec = _MODEL_REGISTRY[name]
    area = np.asarray(data["area"], dtype=float)
    species = np.asarray(data["species"], dtype=float)
    n = len(area)
    if not (species > 0).any():
        # Only reachable with allow_zero_species. There is no richness to
        # rescale by, and a curve through all zeros has no likelihood (RSS 0).
        logger.info("Model '%s': no island has any species; nothing to fit", name)
        return _failed_fit(name, spec.param_names, n, data)

    # The fit runs on rescaled data (see _RESCALE).
    a_scale = 1.0 if name in _AREA_UNSCALED else float(np.exp(np.mean(np.log(area))))
    s_scale = float(np.max(species))
    to_natural, to_scaled = _RESCALE[name]
    area_s, species_s = area / a_scale, species / s_scale

    log_idx = [spec.param_names.index(p) for p in spec.log_params]
    lower = _to_search_scale(spec.bounds_lower, log_idx)
    upper = _to_search_scale(spec.bounds_upper, log_idx)

    def residuals(p: np.ndarray) -> np.ndarray:
        pred = spec.func(area_s, *(_to_natural_scale(p, log_idx) if log_idx else p))
        res = species_s - pred
        # Replace any NaN/Inf with a large penalty
        return np.where(np.isfinite(res), res, 1e10)

    best_cost = np.inf
    best_result = None

    if start_from is not None:
        expected = set(spec.param_names)
        provided = set(start_from)
        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if extra:
                parts.append(f"unexpected {extra}")
            raise ValueError(
                f"start_from keys don't match model '{name}' "
                f"params {spec.param_names}: {', '.join(parts)}"
            )
        bad = [k for k in spec.log_params if not start_from[k] > 0]
        if bad:
            raise ValueError(
                f"start_from values for {bad} must be positive: model '{name}' "
                f"searches them on a log scale"
            )
        given = np.array([start_from[k] for k in spec.param_names], dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            warm = np.array(to_scaled(given, a_scale, s_scale), dtype=float)
        if np.all(np.isfinite(warm)) and all(warm[i] > 0 for i in log_idx):
            starts: Iterable[Sequence[float]] = [warm.tolist()]
        else:  # it doesn't survive rescaling (overflow, or a log parameter at 0)
            logger.debug("Model '%s': warm start %s can't be rescaled; using the grid", name, given)
            starts = product(*spec.start_grid)
    else:
        starts = product(*spec.start_grid)

    # Early-termination: stop after this many consecutive successful solver
    # calls that don't improve the best cost.  For large grids this avoids
    # hundreds of redundant solver calls once the optimum has stabilised.
    # The limit is generous enough to avoid premature stopping.
    n_params = len(spec.param_names)
    stale_limit = 50 * n_params  # 100 for 2-p, 150 for 3-p, 200 for 4-p

    n_starts = 0
    n_errors = 0
    n_unsuccessful = 0
    stale_runs = 0
    for start in starts:
        n_starts += 1
        try:
            # The search visits parameters where the model overflows; residuals()
            # turns the non-finite values into a penalty, so numpy's warnings about
            # them carry nothing and would flood an interactive session.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                result = least_squares(
                    residuals,
                    x0=np.clip(_to_search_scale(start, log_idx), lower, upper),
                    bounds=(lower, upper),
                    method="trf",
                    max_nfev=5000,
                )
            if not result.success:
                # Stopped on max_nfev or similar: not a converged solution,
                # so it may neither win on cost nor count as a stale run.
                n_unsuccessful += 1
                continue
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                stale_runs = 0
            else:
                stale_runs += 1
        except Exception:
            n_errors += 1
            logger.debug(
                "Model '%s' NLS failed for start %s",
                name,
                list(start),
                exc_info=True,
            )

        if stale_runs >= stale_limit and best_result is not None:
            logger.debug(
                "Model '%s': early stop after %d starts (%d without improvement)",
                name,
                n_starts,
                stale_limit,
            )
            break

        if n_starts % 100 == 0:
            logger.debug(
                "Model '%s': %d starts evaluated (best cost=%.4g)",
                name,
                n_starts,
                best_cost,
            )

    if n_errors:
        logger.info(
            "Model '%s': %d/%d starting points raised exceptions",
            name,
            n_errors,
            n_starts,
        )
    if n_unsuccessful:
        logger.info(
            "Model '%s': %d/%d starting points stopped without converging",
            name,
            n_unsuccessful,
            n_starts,
        )

    if best_result is None:
        return _failed_fit(name, spec.param_names, n, data)

    if start_from is None:
        # The default tolerances stop early along flat ridges of the likelihood,
        # leaving parameters loose in the third digit; one tight solve from the best
        # grid start pins them down. A warm start (the bootstrap's) skips it: its
        # resamples need the fit, not the last digits.
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                polished = least_squares(
                    residuals,
                    x0=best_result.x,
                    bounds=(lower, upper),
                    method="trf",
                    max_nfev=5000,
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14,
                )
            if polished.success and polished.cost <= best_result.cost:
                best_result = polished
        except Exception:
            logger.debug("Model '%s': polishing the best fit failed", name, exc_info=True)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = np.array(
            to_natural(_to_natural_scale(best_result.x, log_idx), a_scale, s_scale), dtype=float
        )
        # RSS in the data's units, from the converted parameters: it is what's
        # reported, and it checks the conversion.
        rss = float(np.sum((species - spec.func(area, *values)) ** 2))
    if not (np.all(np.isfinite(values)) and np.isfinite(rss)):
        logger.info("Model '%s': the fit's parameters overflow in the data's units", name)
        return _failed_fit(name, spec.param_names, n, data)
    params = dict(zip(spec.param_names, (float(v) for v in values), strict=True))
    ss_tot = np.sum((species - np.mean(species)) ** 2)
    # R² is undefined when every island has the same richness.
    r_squared = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan

    k = len(spec.param_names) + 1  # +1 for sigma
    aic, aicc, bic = _compute_ic(k, n, rss)

    return SARFit(
        model=name,
        params=params,
        r_squared=r_squared,
        aic=aic,
        aicc=aicc,
        bic=bic,
        n=n,
        converged=True,
        data=data,
    )


def _failed_fit(name: str, param_names: list[str], n: int, data: pd.DataFrame) -> SARFit:
    return SARFit(
        model=name,
        params=dict.fromkeys(param_names, np.nan),
        r_squared=np.nan,
        aic=np.nan,
        aicc=np.nan,
        bic=np.nan,
        n=n,
        converged=False,
        data=data,
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# --- Non-asymptotic models (8) ---

# 1. power: S = c * A^z
_register(
    _ModelSpec(
        name="power",
        func=lambda a, c, z: c * a**z,
        param_names=["c", "z"],
        bounds_lower=[1e-10, -5.0],
        bounds_upper=[1e6, 5.0],
        start_grid=[
            [0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
            [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
        ],
    )
)

# 2. powerR: S = c * A^z + d
_register(
    _ModelSpec(
        name="powerR",
        func=lambda a, f, c, z: f + c * a**z,
        param_names=["f", "c", "z"],
        bounds_lower=[-1e6, 1e-10, -5.0],
        bounds_upper=[1e6, 1e6, 5.0],
        start_grid=[
            [-1.0, -0.3, 0.0, 0.1],
            [0.1, 0.3, 1.0, 3.0],
            [0.05, 0.15, 0.3, 0.5, 1.0],
        ],
    )
)

# 3. loga: S = c + z * log(A)
_register(
    _ModelSpec(
        name="loga",
        func=lambda a, c, z: c + z * np.log(a),
        param_names=["c", "z"],
        bounds_lower=[-1e6, -1e6],
        bounds_upper=[1e6, 1e6],
        start_grid=[
            [-1.0, -0.3, 0.0, 0.3, 1.0],
            [0.01, 0.03, 0.1, 0.3, 1.0],
        ],
    )
)

# 4. linear: S = c + m * A  (OLS)
_register(
    _ModelSpec(
        name="linear",
        func=lambda a, c, m: c + m * a,
        param_names=["c", "m"],
        bounds_lower=[-1e6, -1e6],
        bounds_upper=[1e6, 1e6],
        start_grid=[
            [0.0, 0.1, 0.3, 0.5, 1.0],
            [0.001, 0.01, 0.03, 0.1, 1.0],
        ],
    )
)


# 5. epm1: S = c * A^(z * A^-d)  (Extended power model 1, Tjørve 2009)
def _epm1_func(a, c, z, d):
    return c * a ** (z * a ** (-d))


_register(
    _ModelSpec(
        name="epm1",
        func=_epm1_func,
        param_names=["c", "z", "d"],
        bounds_lower=[0.0, -50.0, -50.0],
        bounds_upper=[np.inf, 50.0, 50.0],
        start_grid=[
            [1e-08, 1e-05, 0.001, 0.03, 0.3],
            [0.05, 0.3, 1.0, 3.0, 8.0],
            [-1.0, -0.3, 0.0, 0.1, 0.5, 1.0],
        ],
        # c spans dozens of orders of magnitude with the area units (1e-49 in m²).
        log_params=("c",),
    )
)


# 6. epm2: S = c * A^(z - d/A)  (Extended power model 2, Tjørve 2009). z lies in
#    [0, 1], as R sars bounds it; d is unbounded because it scales with the area
#    units (d/A).
def _epm2_func(a, c, z, d):
    return c * a ** (z - d / a)


_register(
    _ModelSpec(
        name="epm2",
        func=_epm2_func,
        param_names=["c", "z", "d"],
        bounds_lower=[0.0, 0.0, -np.inf],
        bounds_upper=[np.inf, 1.0, np.inf],
        start_grid=[
            [0.002, 0.02, 0.06, 0.15, 0.3],
            [0.05, 0.25, 0.5, 0.9],
            [-10.0, -1.0, 0.0, 0.5, 1.0, 10.0],
        ],
    )
)


# 7. p1: S = c * A^z * exp(-d * A)  (Persistence function 1)
def _p1_func(a, c, z, d):
    return c * a**z * np.exp(-d * a)


_register(
    _ModelSpec(
        name="p1",
        func=_p1_func,
        param_names=["c", "z", "d"],
        bounds_lower=[1e-10, -5.0, 0.0],
        bounds_upper=[1e6, 5.0, 10.0],
        start_grid=[
            [0.05, 0.1, 0.3, 0.5, 1.0],
            [0.1, 0.3, 0.5, 0.7, 1.0],
            [0.001, 0.01, 0.03, 0.1, 0.3],
        ],
    )
)


# 8. p2: S = c * A^z * exp(-d / A)  (Persistence function 2)
def _p2_func(a, c, z, d):
    return c * a**z * np.exp(-d / a)


_register(
    _ModelSpec(
        name="p2",
        func=_p2_func,
        param_names=["c", "z", "d"],
        bounds_lower=[1e-10, -5.0, -np.inf],
        bounds_upper=[1e6, 5.0, np.inf],
        start_grid=[
            [0.1, 0.3, 0.5, 1.0, 2.0],
            [0.001, 0.01, 0.05, 0.1, 0.5],
            [0.03, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)

# --- koba (convex, no asymptote), then the asymptotic convex models (4) ---

# 9. koba: S = c * log(1 + A/z)
_register(
    _ModelSpec(
        name="koba",
        func=lambda a, c, z: c * np.log(1.0 + a / z),
        param_names=["c", "z"],
        bounds_lower=[1e-10, 1e-10],
        bounds_upper=[1e6, 1e6],
        start_grid=[
            [0.01, 0.03, 0.1, 0.2, 0.3, 0.5],
            [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)

# 10. monod: S = d * A / (c + A)
_register(
    _ModelSpec(
        name="monod",
        func=lambda a, d, c: d * a / (c + a),
        param_names=["d", "c"],
        bounds_lower=[1e-10, 1e-10],
        bounds_upper=[1e6, 1e6],
        start_grid=[
            [0.3, 0.5, 0.8, 1.0, 1.5, 3.0],
            [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        ],
    )
)

# negexpo, weibull3, weibull4, chapman and betap compute 1 - exp(-x) as
# -expm1(-x) (betap's 1 - (1 + u)^-f as -expm1(-f log1p(u))): for small x the
# plain difference loses most of its digits, and weibull4's Galápagos fit has
# x near 1e-14 at the smallest islands.
# 11. negexpo: S = d * (1 - exp(-z * A))
_register(
    _ModelSpec(
        name="negexpo",
        func=lambda a, d, z: d * -np.expm1(-z * a),
        param_names=["d", "z"],
        bounds_lower=[1e-10, 1e-10],
        bounds_upper=[1e6, 100.0],
        start_grid=[
            [0.3, 0.5, 0.8, 1.0, 1.5, 3.0],
            [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)

# 12. asymp: S = d - c * exp(-z * A)
_register(
    _ModelSpec(
        name="asymp",
        func=lambda a, d, c, z: d - c * np.exp(-z * a),
        param_names=["d", "c", "z"],
        bounds_lower=[1e-10, 1e-10, 1e-10],
        bounds_upper=[1e6, 1e6, 100.0],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0],
            [0.3, 0.6, 1.0, 2.0],
            [0.003, 0.03, 0.1, 0.3, 1.0],
        ],
    )
)

# 13. ratio: S = (c + z * A) / (1 + d * A)
_register(
    _ModelSpec(
        name="ratio",
        func=lambda a, c, z, d: (c + z * a) / (1.0 + d * a),
        param_names=["c", "z", "d"],
        bounds_lower=[-1e6, 1e-10, 1e-10],
        bounds_upper=[1e6, 1e6, 100.0],
        start_grid=[
            [-0.1, 0.0, 0.03, 0.1, 0.3],
            [0.03, 0.1, 0.3, 1.0, 3.0],
            [0.03, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)

# --- Asymptotic sigmoid models (7) ---

# 14. mmf: S = d / (1 + c * A^(-z))
_register(
    _ModelSpec(
        name="mmf",
        func=lambda a, d, c, z: d / (1.0 + c * a ** (-z)),
        param_names=["d", "c", "z"],
        bounds_lower=[1e-10, 1e-10, 1e-10],
        bounds_upper=[np.inf, np.inf, 10.0],
        start_grid=[
            [0.3, 0.5, 1.0, 2.0, 5.0],
            [0.1, 0.5, 1.0, 3.0, 10.0],
            [0.1, 0.5, 1.0, 1.5, 2.0],
        ],
        # On small samples d and c grow together toward a power-law limit.
        log_params=("d", "c"),
    )
)

# 15. gompertz: S = d * exp(-exp(-z * (A - c)))
_register(
    _ModelSpec(
        name="gompertz",
        func=lambda a, d, z, c: d * np.exp(-np.exp(-z * (a - c))),
        param_names=["d", "z", "c"],
        bounds_lower=[1e-10, 1e-10, -1e6],
        bounds_upper=[np.inf, 100.0, 1e6],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0, 5.0],
            [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            [0.0, 0.1, 0.3, 1.0, 3.0],
        ],
    )
)


# 16. weibull3: S = d * (1 - exp(-c * A^z))
def _weibull3_func(a, d, c, z):
    return d * -np.expm1(-c * a**z)


_register(
    _ModelSpec(
        name="weibull3",
        func=_weibull3_func,
        param_names=["d", "c", "z"],
        bounds_lower=[1e-10, 0.0, 1e-10],
        bounds_upper=[np.inf, 100.0, 10.0],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0, 5.0],
            [0.01, 0.1, 0.3, 1.0, 3.0],
            [0.1, 0.5, 1.0, 1.5, 2.0],
        ],
        # On small samples d runs off toward a power-law limit, with c shrinking.
        log_params=("d", "c"),
    )
)


# 17. weibull4: S = d * (1 - exp(-c * A^z))^f
def _weibull4_func(a, d, c, z, f):
    inner = -np.expm1(-c * a**z)
    return d * np.sign(inner) * np.abs(inner) ** f


_register(
    _ModelSpec(
        name="weibull4",
        func=_weibull4_func,
        param_names=["d", "c", "z", "f"],
        bounds_lower=[1e-10, 0.0, 1e-10, 1e-10],
        bounds_upper=[np.inf, 100.0, 10.0, 100.0],
        start_grid=[
            [0.5, 1.0, 2.0],
            [1e-08, 1e-05, 0.001, 0.03, 0.3, 1.0],
            [0.5, 1.0, 3.0, 6.0, 9.0],
            [0.05, 0.1, 0.5, 1.0, 2.0],
        ],
        log_params=("c",),
    )
)


# 18. chapman: S = d * (1 - exp(-z * A))^c
def _chapman_func(a, d, z, c):
    inner = -np.expm1(-z * a)
    return d * np.sign(inner) * np.abs(inner) ** c


_register(
    _ModelSpec(
        name="chapman",
        func=_chapman_func,
        param_names=["d", "z", "c"],
        bounds_lower=[1e-10, 1e-10, 1e-10],
        bounds_upper=[1e6, 100.0, 100.0],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0, 5.0],
            [0.03, 0.1, 0.3, 1.0, 3.0],
            [0.1, 0.3, 0.5, 1.0, 2.0],
        ],
    )
)


# 19. betap: S = d * (1 - (1 + (A/c)^z)^(-f))
def _betap_func(a, d, c, z, f):
    return d * -np.expm1(-f * np.log1p((a / c) ** z))


_register(
    _ModelSpec(
        name="betap",
        func=_betap_func,
        param_names=["d", "c", "z", "f"],
        bounds_lower=[1e-10, 1e-10, 1e-10, 1e-10],
        bounds_upper=[1e6, 1e6, 10.0, 100.0],
        start_grid=[
            [0.5, 0.8, 1.0, 2.0],
            [0.3, 1.0, 3.0, 10.0, 30.0],
            [0.1, 0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 2.0, 5.0, 10.0],
        ],
    )
)


# 20. heleg: S = c / (f + A^(-z))
#     Parameters c, f, z (all positive). Asymptote = c / f.
def _heleg_func(a, c, f, z):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = c / (f + a ** (-z))
    return np.where(np.isfinite(result), result, 0.0)


_register(
    _ModelSpec(
        name="heleg",
        func=_heleg_func,
        param_names=["c", "f", "z"],
        bounds_lower=[1e-10, 1e-10, 1e-10],
        bounds_upper=[1e6, 1e6, 1e6],
        start_grid=[
            [0.03, 0.1, 0.3, 1.0, 3.0],
            [0.03, 0.1, 0.3, 1.0, 3.0],
            [0.1, 0.5, 1.0, 2.0, 5.0],
        ],
    )
)


# ---------------------------------------------------------------------------
# Public API — one function per model
# ---------------------------------------------------------------------------


def sar_power(data: pd.DataFrame) -> SARFit:
    """Fit the power law SAR model: S = c * A^z

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit

    References
    ----------
    Arrhenius O (1921) Species and area. Journal of Ecology 9:95-99.
    """
    return _fit_nls("power", data)


def sar_powerR(data: pd.DataFrame) -> SARFit:  # noqa: N802
    """Fit the power-R SAR model: S = f + c * A^z

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("powerR", data)


def sar_loga(data: pd.DataFrame) -> SARFit:
    """Fit the logarithmic SAR model: S = c + z * log(A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit

    References
    ----------
    Gleason HA (1922) On the relation between species and area.
    Ecology 3:158-162.
    """
    return _fit_nls("loga", data)


def sar_linear(data: pd.DataFrame) -> SARFit:
    """Fit the linear SAR model: S = c + m * A

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("linear", data)


def sar_epm1(data: pd.DataFrame) -> SARFit:
    """Fit the extended power model 1: S = c * A^(z * A^-d)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("epm1", data)


def sar_epm2(data: pd.DataFrame) -> SARFit:
    """Fit the extended power model 2: S = c * A^(z - d/A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("epm2", data)


def sar_p1(data: pd.DataFrame) -> SARFit:
    """Fit persistence function 1: S = c * A^z * exp(-d * A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("p1", data)


def sar_p2(data: pd.DataFrame) -> SARFit:
    """Fit persistence function 2: S = c * A^z * exp(-d / A)

    ``d`` may be negative, as in R ``sars``; the curve then rises steeply toward
    the smallest areas, so predictions below the data's range can be very large.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("p2", data)


def sar_koba(data: pd.DataFrame) -> SARFit:
    """Fit the Kobayashi logarithmic model: S = c * log(1 + A/z)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("koba", data)


def sar_monod(data: pd.DataFrame) -> SARFit:
    """Fit the Monod model: S = d * A / (c + A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("monod", data)


def sar_negexpo(data: pd.DataFrame) -> SARFit:
    """Fit the negative exponential model: S = d * (1 - exp(-z * A))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("negexpo", data)


def sar_asymp(data: pd.DataFrame) -> SARFit:
    """Fit the asymptotic model: S = d - c * exp(-z * A)

    R ``sars`` writes the same curve as ``d - c * z^A``, so its ``z`` is
    ``exp(-z)`` here. ``z`` is kept positive, which keeps the asymptote; R can
    also reach ``z > 1`` in its form, a curve that grows without bound.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("asymp", data)


def sar_ratio(data: pd.DataFrame) -> SARFit:
    """Fit the rational function model: S = (c + z * A) / (1 + d * A)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("ratio", data)


def sar_mmf(data: pd.DataFrame) -> SARFit:
    """Fit the Morgan-Mercer-Flodin model: S = d / (1 + c * A^(-z))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("mmf", data)


def sar_gompertz(data: pd.DataFrame) -> SARFit:
    """Fit the Gompertz model: S = d * exp(-exp(-z * (A - c)))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("gompertz", data)


def sar_weibull3(data: pd.DataFrame) -> SARFit:
    """Fit the 3-parameter Weibull model: S = d * (1 - exp(-c * A^z))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("weibull3", data)


def sar_weibull4(data: pd.DataFrame) -> SARFit:
    """Fit the 4-parameter Weibull model: S = d * (1 - exp(-c * A^z))^f

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("weibull4", data)


def sar_chapman(data: pd.DataFrame) -> SARFit:
    """Fit the Chapman-Richards model: S = d * (1 - exp(-z * A))^c

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("chapman", data)


def sar_betap(data: pd.DataFrame) -> SARFit:
    """Fit the Beta-P model: S = d * (1 - (1 + (A/c)^z)^(-f))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("betap", data)


def sar_heleg(data: pd.DataFrame) -> SARFit:
    """Fit the Heleg model: S = c / (f + A^(-z))

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Returns
    -------
    SARFit
    """
    return _fit_nls("heleg", data)


# A finite difference is taken over a step that moves the curve by between
# these fractions of the richness values: enough for the difference to be more
# than rounding noise (about 1e-16 of them), and little enough that the curve's
# bending over the step doesn't show.
_JAC_MIN_RESPONSE = 1e-7


_JAC_MAX_RESPONSE = 1e-5


# The first step tried, relative to the parameter.
_JAC_REL_STEP = float(np.finfo(float).eps ** (1.0 / 3.0))


# Two step sizes must agree this closely for a derivative to be trusted.
_JAC_AGREE_TOL = 1e-6


# The agreement of a derivative of a smooth curve is no better than this,
# whatever the two steps happen to give.
_JAC_BEST_AGREEMENT = 1e-9


# A direction in parameter space is undetermined when the curve responds to it
# this much less than to the best-determined one. Standard errors that depend
# on such a direction are reported as unknown rather than as huge numbers.
_IDENT_RCOND = 1e-4


def _difference(
    func: _SARFunc, area: np.ndarray, values: np.ndarray, i: int, base: np.ndarray, h: float
) -> tuple[np.ndarray, float, int] | None:
    """A second-order difference of the curve in parameter ``i`` with step ``h``.

    Central where the curve is defined on both sides. Where a step one way
    leaves the model's domain (a positive parameter stepped past zero), the
    three-point difference on the other side. Returns the derivative, the size
    of the change in the curve it was taken over, and which side it used (0 for
    central), or None if the curve isn't defined a step away on either side.
    """

    def at(step: float) -> np.ndarray:
        moved = values.copy()
        moved[i] += step
        return np.asarray(func(area, *moved), dtype=float)

    up, down = at(h), at(-h)
    up_ok, down_ok = bool(np.all(np.isfinite(up))), bool(np.all(np.isfinite(down)))
    if up_ok and down_ok:
        return (up - down) / (2.0 * h), float(np.linalg.norm(up - down)) / 2.0, 0
    for side, near, ok in ((1, up, up_ok), (-1, down, down_ok)):
        if ok:
            far = at(2.0 * side * h)
            if np.all(np.isfinite(far)):
                slope = side * (-3.0 * base + 4.0 * near - far) / (2.0 * h)
                return slope, float(np.linalg.norm(near - base)), side
    return None


def _jacobian_column(
    func: _SARFunc, area: np.ndarray, values: np.ndarray, i: int, base: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Derivative of the model's curve with respect to parameter ``i``.

    Returns two estimates of it, from a step and from half that step, and how
    far apart they are relative to their size; the second estimate is the
    better one, and is extrapolated from both when they agree. A parameter the
    curve doesn't respond to gets zeros. None means no step was found at which
    the curve could be differenced.
    """
    # The right step is the one the curve responds to by the right amount,
    # which the parameter's own size doesn't tell: p1's d can sit at 0 or 1e-42
    # on its bound with a derivative of thousands, and a log-scale parameter
    # at 1e-300. So the step is searched for, growing or shrinking from a first
    # guess in proportion to how far the response is from the one wanted, and
    # bisecting once it has been both too small and too large.
    lowest, highest = _JAC_MIN_RESPONSE * scale, _JAC_MAX_RESPONSE * scale
    wanted = float(np.sqrt(lowest * highest))
    h = _JAC_REL_STEP * abs(values[i])
    if not (np.isfinite(h) and h > 0):
        h = _JAC_REL_STEP
    too_small, too_big = 0.0, np.inf
    coarse = None
    for _ in range(200):
        found = _difference(func, area, values, i, base, h)
        response = found[1] if found is not None else np.inf
        if lowest <= response <= highest:
            coarse = found
            break
        if response < lowest:
            too_small = h
            h *= min(wanted / response, 1e8) if response > 0 else 1e8
            h = max(h, 2.0 * too_small)
        else:
            too_big = h
            h *= max(wanted / response, 1e-8) if np.isfinite(response) else 1e-3
            h = min(h, 0.5 * too_big)
        if np.isinf(h):
            # No step moves the curve: it ignores this parameter.
            zeros = np.zeros(len(area))
            return zeros, zeros, 0.0
        if not too_small < h < too_big:
            h = float(np.sqrt(too_small * too_big))
            if not too_small < h < too_big:
                # No step between one too small and the next one up: the curve
                # jumps there, or stops being defined.
                return None
    fine = _difference(func, area, values, i, base, h / 2.0)
    if coarse is None or fine is None:
        return None
    size = float(np.linalg.norm(fine[0]))
    apart = float(np.linalg.norm(fine[0] - coarse[0])) / size if size > 0 else np.inf
    best = fine[0]
    if apart <= _JAC_AGREE_TOL and fine[2] == coarse[2]:
        # Both have an error proportional to the step squared, so Richardson
        # extrapolation removes most of it.
        best = (4.0 * fine[0] - coarse[0]) / 3.0
    return coarse[0], best, apart


def _std_errors_from_jacobian(jac: np.ndarray, sigma2: float, accuracy: float) -> np.ndarray:
    """sqrt(diag(sigma^2 (J'J)^-1)), NaN for parameters the Jacobian doesn't identify.

    ``accuracy`` is the relative accuracy of the Jacobian's columns.
    """
    std_error = np.full(jac.shape[1], np.nan)
    # A parameter the curve ignores has no standard error; the others' are the
    # ones they'd have with it held fixed.
    norms = np.linalg.norm(jac, axis=0)
    live = norms > 0
    if not live.any():
        return std_error
    # Columns are scaled to unit length so that the test for an undetermined
    # direction doesn't depend on the parameters' units.
    _, sing, vt = np.linalg.svd(jac[:, live] / norms[live], full_matrices=False)
    weak = sing < _IDENT_RCOND * sing[0]
    # Each parameter's variance is a sum over the singular directions, of its
    # squared loading on the direction over the squared singular value.
    loadings = vt.T
    known = ((loadings[:, ~weak] / sing[~weak]) ** 2).sum(axis=1)
    variance = known
    lost = np.zeros(len(known), dtype=bool)
    if weak.any():
        # A parameter whose variance comes mostly from weak directions is
        # unidentified: two parameters on a ridge, say. A third that merely
        # shares the model with them isn't, though its loading on the ridge is
        # rarely exactly zero. What the weak directions add to its variance is
        # judged with what the Jacobian's accuracy allows to be known: a
        # loading within the error of the singular vectors is taken as zero,
        # and a singular value isn't taken to be smaller than could be
        # resolved. If that is the smaller part, it is counted in, unless more
        # than 1% of the variance would come from directions whose singular
        # values are too small to have two good digits.
        loading_error = accuracy * sing[0] / sing[~weak].min()
        resolved = np.where(np.abs(loadings[:, weak]) > loading_error, loadings[:, weak], 0.0)
        parts = (resolved / np.maximum(sing[weak], accuracy * sing[0])) ** 2
        unsure = sing[weak] < 100.0 * accuracy * sing[0]
        lost = (parts.sum(axis=1) > known) | (parts[:, unsure].sum(axis=1) > 0.01 * known)
        variance = known + parts.sum(axis=1)
    std_error[live] = np.where(lost, np.nan, np.sqrt(sigma2 * variance) / norms[live])
    return std_error


def _on_bound(name: str, values: np.ndarray, area: np.ndarray, species: np.ndarray) -> np.ndarray:
    """Which of a fit's parameters sit on a bound of its search. The bounds are
    on the rescaled parameters (see ``_RESCALE``), so the fitted values are
    rescaled as ``_fit_nls`` rescales them; a log-scale parameter is compared on
    the log scale, where its lower bound of 0 is ``exp(_LOG_FLOOR)``."""
    spec = _MODEL_REGISTRY[name]
    if name not in _RESCALE:  # a model registered by hand, as tests do
        return np.zeros(len(values), dtype=bool)
    a_scale = 1.0 if name in _AREA_UNSCALED else float(np.exp(np.mean(np.log(area))))
    s_scale = float(np.max(species))
    with np.errstate(all="ignore"):
        scaled = np.array(_RESCALE[name][1](values, a_scale, s_scale), dtype=float)
    log_idx = [spec.param_names.index(p) for p in spec.log_params]
    found = np.zeros(len(values), dtype=bool)
    for i, (v, lo, hi) in enumerate(zip(scaled, spec.bounds_lower, spec.bounds_upper, strict=True)):
        if i in log_idx:
            (v,) = _to_search_scale([v], [0]) if v > 0 else (-np.inf,)
            lo, hi = _to_search_scale([lo, hi], [0, 1])
        found[i] = any(np.isfinite(b) and abs(v - b) <= 1e-6 * max(1.0, abs(b)) for b in (lo, hi))
    return found


def _param_std_errors(
    func: _SARFunc, area: np.ndarray, species: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standard errors of fitted parameters, from sigma^2 (J'J)^-1.

    Parameters
    ----------
    func : callable
        The model function, ``f(area, *params)``.
    area, species : np.ndarray
        The data the model was fitted to.
    values : np.ndarray
        The fitted parameters, in the function's order. There must be fewer of
        them than observations.

    Returns
    -------
    std_error : np.ndarray
        One per parameter; NaN where none can be given.
    rough : np.ndarray of bool
        The parameters the curve couldn't be differentiated reliably in. They
        have no standard error, and neither do any that depend on them.
    """
    n, p = len(area), len(values)
    nan = np.full(p, np.nan)
    with np.errstate(all="ignore"):
        base = np.asarray(func(area, *values), dtype=float)
        if not np.all(np.isfinite(base)):
            return nan, np.ones(p, dtype=bool)
        scale = float(np.linalg.norm(species)) or float(np.linalg.norm(base)) or 1.0
        columns = [_jacobian_column(func, area, values, i, base, scale) for i in range(p)]
    undefined = np.array([c is None for c in columns])
    if undefined.any():
        # With no estimate at all of a column, nothing can be said of how the
        # other parameters depend on it.
        return nan, undefined
    found = [c for c in columns if c is not None]
    apart = np.array([c[2] for c in found])
    rough = apart > _JAC_AGREE_TOL
    accuracy = float(np.clip(apart[~rough].max(initial=0.0), _JAC_BEST_AGREEMENT, _JAC_AGREE_TOL))
    residuals = species - base
    sigma2 = float(residuals @ residuals) / (n - p)
    std_error = _std_errors_from_jacobian(np.column_stack([c[1] for c in found]), sigma2, accuracy)
    if rough.any():
        # A parameter whose derivative is rough has no standard error. Another
        # keeps its own only if it doesn't depend on that derivative: it must
        # come out the same from the other estimate of the rough columns, and
        # with the rough parameters held fixed instead (noise in a column can
        # hide a ridge the parameter lies on).
        best = np.column_stack([c[1] for c in found])
        for alternative in (np.column_stack([c[0] for c in found]), np.zeros_like(best)):
            other = _std_errors_from_jacobian(np.where(rough, alternative, best), sigma2, accuracy)
            with np.errstate(invalid="ignore"):
                same = np.abs(other - std_error) <= 0.01 * std_error
            std_error = np.where(same & ~rough, std_error, np.nan)
    return std_error, rough
