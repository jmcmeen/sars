"""Residual checks on a fitted model: normality, homoscedasticity, negative fits."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cache
from numbers import Real

import numpy as np

# scipy.stats is imported inside the functions that use it: it takes about a
# quarter of a second to load, and by default no checks are run.

# The names each argument accepts. "kolmogorov" is not R sars's "kolmo", which
# compares the residuals with a standard normal, so its result depends on the
# units of species richness; here the normal has the residuals' own mean and
# standard deviation. "lillie" is R's name for the Lilliefors test.
NORMALITY_TESTS = ("shapiro", "kolmogorov", "lillie")
_NORMALITY_ALIASES = {"lilliefors": "lillie"}
HOMOSCEDASTICITY_TESTS = ("cor.fitted", "cor.area")
COR_METHODS = ("spearman", "pearson", "kendall")

_NO_KOLMO = (
    "R sars's normaTest='kolmo' compares the residuals with a standard normal, so its "
    "answer depends on the units of species richness; it isn't offered. Use 'lillie' "
    "(the Lilliefors test, which allows for the mean and standard deviation being "
    "estimated) or 'kolmogorov'."
)


@dataclass(frozen=True)
class CheckOptions:
    """The residual checks asked for, validated once when built.

    The public functions take these as separate keyword arguments (described
    in ``sar_multi``) and carry them around as one of these.
    """

    normality: str | None = None
    homoscedasticity: str | None = None
    cor_method: str = "spearman"
    neg_check: bool = False
    alpha_normality: float = 0.05
    alpha_homoscedasticity: float = 0.05

    def __post_init__(self) -> None:
        if self.normality == "kolmo":
            raise ValueError(_NO_KOLMO)
        if isinstance(self.normality, str) and self.normality in _NORMALITY_ALIASES:
            object.__setattr__(self, "normality", _NORMALITY_ALIASES[self.normality])
        if self.normality is not None and self.normality not in NORMALITY_TESTS:
            raise ValueError(
                f"Unknown normality test: {self.normality!r}. "
                f"Choose from {list(NORMALITY_TESTS)} or None."
            )
        if (
            self.homoscedasticity is not None
            and self.homoscedasticity not in HOMOSCEDASTICITY_TESTS
        ):
            raise ValueError(
                f"Unknown homoscedasticity test: {self.homoscedasticity!r}. "
                f"Choose from {list(HOMOSCEDASTICITY_TESTS)} or None."
            )
        if self.cor_method not in COR_METHODS:
            raise ValueError(
                f"Unknown correlation method: {self.cor_method!r}. Choose from {list(COR_METHODS)}."
            )
        for name in ("alpha_normality", "alpha_homoscedasticity"):
            alpha = getattr(self, name)
            # A string or None would raise TypeError from the comparison.
            is_number = isinstance(alpha, Real) and not isinstance(alpha, bool)
            if not is_number or not 0.0 < alpha < 1.0:
                raise ValueError(f"{name} must be a number between 0 and 1, got {alpha!r}")

    @property
    def requested(self) -> bool:
        """Whether any check was asked for."""
        return self.normality is not None or self.homoscedasticity is not None or self.neg_check


@dataclass
class ResidualChecks:
    """Residual checks on one fitted model.

    A test that wasn't asked for, or can't be computed (fewer than three
    observations, or five for the Lilliefors test; residuals that are all
    equal; or fitted values that aren't all finite), has NaN for its statistic
    and p value.

    Attributes
    ----------
    model : str
        The model checked.
    normality : str or None
        The normality test run on the residuals: "shapiro" (Shapiro-Wilk),
        "kolmogorov" (Kolmogorov-Smirnov against a normal distribution with
        the residuals' mean and standard deviation), "lillie" (Lilliefors),
        or None.
    normality_statistic : float
        The test statistic: W for "shapiro", D for "kolmogorov" and "lillie".
    normality_pvalue : float
        Its p value. A small value is evidence against normal residuals.
    homoscedasticity : str or None
        What the squared residuals were correlated with: "cor.fitted" (the
        fitted values), "cor.area" (the areas), or None.
    cor_method : str
        The correlation used: "spearman", "pearson" or "kendall".
    homoscedasticity_statistic : float
        The correlation coefficient (rho, r or tau).
    homoscedasticity_pvalue : float
        Two-sided p value of the test that the correlation is zero. A small
        value is evidence that the residual variance changes along the curve.
    negative_fitted : bool
        Whether the model predicts negative richness at any observed area.
    finite_fitted : bool
        Whether the model's prediction is finite at every observed area. When
        it isn't, no test is run: there are no residuals to test.
    """

    model: str
    normality: str | None
    normality_statistic: float
    normality_pvalue: float
    homoscedasticity: str | None
    cor_method: str
    homoscedasticity_statistic: float
    homoscedasticity_pvalue: float
    negative_fitted: bool
    finite_fitted: bool = True

    def failure(
        self,
        alpha_normality: float = 0.05,
        alpha_homoscedasticity: float = 0.05,
        neg_check: bool = False,
    ) -> str | None:
        """The first check the model fails, in words, or None if it passes.

        The checks run in R sars's order: normality, homoscedasticity, then
        negative fitted values (only when ``neg_check`` is true). A test with
        a NaN p value can't reject anything, so it passes, with one exception:
        a model whose fitted values aren't all finite fails whenever any check
        is asked for, since none of them can be run on it.

        Parameters
        ----------
        alpha_normality : float
            The normality test fails when its p value is below this.
        alpha_homoscedasticity : float
            The homoscedasticity test fails when its p value is below this.
        neg_check : bool
            Whether a negative fitted value is a failure.

        Returns
        -------
        str or None
            Which check failed and the numbers behind it.
        """
        asked = self.normality is not None or self.homoscedasticity is not None or neg_check
        if asked and not self.finite_fitted:
            return "fitted values not finite"
        if self.normality_pvalue < alpha_normality:
            return (
                f"residuals not normal ({self.normality} "
                f"p = {self.normality_pvalue:.3g} < {alpha_normality:g})"
            )
        if self.homoscedasticity_pvalue < alpha_homoscedasticity:
            against = "fitted values" if self.homoscedasticity == "cor.fitted" else "area"
            return (
                f"residual variance not constant ({self.cor_method} correlation of squared "
                f"residuals with {against} = {self.homoscedasticity_statistic:.3g}, "
                f"p = {self.homoscedasticity_pvalue:.3g} < {alpha_homoscedasticity:g})"
            )
        if neg_check and self.negative_fitted:
            return "negative fitted values"
        return None


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


# R's ks.test gives an exact p value below this many untied observations and
# the asymptotic one otherwise. scipy on its own stays exact up to 10000.
_KS_EXACT_BELOW_N = 100


def _normality(residuals: np.ndarray, test: str) -> tuple[float, float]:
    """Statistic and p value of a normality test on the residuals."""
    from scipy import stats

    n = len(residuals)
    # Neither test says anything about residuals that are all (nearly) the same,
    # such as those of a fit through every point.
    if n < 3 or np.ptp(residuals) <= 1e-12 * max(1.0, float(np.max(np.abs(residuals)))):
        return np.nan, np.nan
    if test == "shapiro":
        res = stats.shapiro(residuals)
    elif test == "lillie":
        return _lilliefors(residuals)
    else:
        # The mean and standard deviation come from the residuals themselves,
        # which the Kolmogorov-Smirnov distribution doesn't allow for: the p
        # value is too large, so the test rejects less often than its alpha
        # (the Lilliefors test is the correction).
        standardised = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)
        exact = n < _KS_EXACT_BELOW_N and len(np.unique(residuals)) == n
        res = stats.kstest(standardised, "norm", method="exact" if exact else "asymp")
    return float(res.statistic), float(res.pvalue)


# The Lilliefors test's null distribution of D depends only on n: the residuals
# are standardised by their own mean and standard deviation. It is simulated
# once per n, with a fixed seed so a p value is reproducible.
_LILLIEFORS_DRAWS = 20_000
_LILLIEFORS_SEED = 20_261_122
_LILLIEFORS_MIN_N = 5  # as R's nortest::lillie.test
# Above this many residuals the null isn't simulated (20,000 samples of n would
# take seconds); D is rescaled to this n by (n / 1000)^0.49, the scaling
# Dallal and Wilkinson use above 100. Within about 0.01 up to n = 5000.
_LILLIEFORS_MAX_SIMULATED_N = 1000


def _lilliefors_statistic(x: np.ndarray) -> float:
    """D: the largest gap between the empirical distribution of x and a
    normal with x's mean and (n - 1) standard deviation."""
    from scipy import stats

    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    p = stats.norm.cdf((x - x.mean()) / x.std(ddof=1))
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - p), np.max(p - (i - 1) / n)))


def _dallal_wilkinson(d: float, n: int) -> float:
    """Dallal and Wilkinson's (1986) approximation to the Lilliefors p value,
    accurate below 0.1. Above n = 100, D is rescaled to n = 100."""
    if n > 100:
        d, n = d * (n / 100) ** 0.49, 100
    return float(
        np.exp(
            -7.01256 * d**2 * (n + 2.78019)
            + 2.99587 * d * np.sqrt(n + 2.78019)
            - 0.122119
            + 0.974598 / np.sqrt(n)
            + 1.67997 / n
        )
    )


@cache
def _lilliefors_null(n: int) -> np.ndarray:
    """Simulated values of D for n normal observations, sorted."""
    from scipy import stats

    rng = np.random.default_rng(_LILLIEFORS_SEED)
    i = np.arange(1, n + 1)
    chunk = max(1, min(_LILLIEFORS_DRAWS, 2_000_000 // n))  # bounded memory
    out = []
    remaining = _LILLIEFORS_DRAWS
    while remaining:
        size = min(chunk, remaining)
        x = np.sort(rng.standard_normal((size, n)), axis=1)
        p = stats.norm.cdf(
            (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
        )
        out.append(np.maximum((i / n - p).max(axis=1), (p - (i - 1) / n).max(axis=1)))
        remaining -= size
    return np.sort(np.concatenate(out))


def _lilliefors(x: np.ndarray) -> tuple[float, float]:
    """D and its p value: Dallal-Wilkinson where that is below 0.1 (as R's
    nortest), else the share of the simulated null at least as large, and at
    least 0.1."""
    n = len(x)
    if n < _LILLIEFORS_MIN_N:
        warnings.warn(
            f"The Lilliefors test needs at least {_LILLIEFORS_MIN_N} residuals, not {n}; "
            "it wasn't run, so it leaves no model out.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan, np.nan
    d = _lilliefors_statistic(x)
    p = _dallal_wilkinson(d, n)
    if p <= 0.1:
        return d, p
    if n > _LILLIEFORS_MAX_SIMULATED_N:
        scaled = d * (n / _LILLIEFORS_MAX_SIMULATED_N) ** 0.49
        simulated = _lilliefors_p_from_null(scaled, _lilliefors_null(_LILLIEFORS_MAX_SIMULATED_N))
    else:
        simulated = _lilliefors_p_from_null(d, _lilliefors_null(n))
    # Where Dallal-Wilkinson is above 0.1, so is the p value: the simulated
    # null can dip just below it at small n (as R's own fit does), which would
    # make p rise with D across the switch.
    return d, max(simulated, 0.1)


def _lilliefors_p_from_null(d: float, null: np.ndarray) -> float:
    """The share of the sorted simulated values at least as large as d,
    counting d itself among them, so it is never 0."""
    at_least = len(null) - int(np.searchsorted(null, d, side="left"))
    return (at_least + 1) / (len(null) + 1)


@cache
def _spearman_null(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Null distribution of Spearman's S = sum of squared rank differences.

    Returns the values S can take for ``n`` untied pairs and how many of the
    n! rank orders give each. Built one position at a time over the set of
    ranks already used, so it costs about 2^n steps instead of n!.
    """
    # counts[used] maps a partial sum to the number of ways of reaching it
    # after assigning the ranks in the bit set ``used`` to the first positions.
    counts: dict[int, dict[int, int]] = {0: {0: 1}}
    for position in range(n):
        nxt: dict[int, dict[int, int]] = {}
        for used, sums in counts.items():
            for rank in range(n):
                if used & (1 << rank):
                    continue
                target = nxt.setdefault(used | (1 << rank), {})
                d2 = (position - rank) ** 2
                for s, ways in sums.items():
                    target[s + d2] = target.get(s + d2, 0) + ways
        counts = nxt
    final = counts[(1 << n) - 1]
    values = np.array(sorted(final))
    return values, np.array([final[v] for v in values], dtype=float)


# Above this many pairs the exact distribution is replaced by a series.
_SPEARMAN_EXACT_MAX_N = 9
# Above this many pairs, or with ties, the p value comes from the t distribution.
_SPEARMAN_SERIES_MAX_N = 1289


def _spearman_upper_tail(s: float, n: int) -> float:
    """P(S >= s) for Spearman's S with ``n`` untied pairs.

    Exact for n up to 9. Beyond that, the Edgeworth series of Best & Roberts
    (1975, Algorithm AS 89, Applied Statistics 24: 377-379), which is what R's
    ``cor.test`` reports, so p values agree with R's. The series can leave
    [0, 1] far in the tails, where it is clipped.
    """
    from scipy import stats

    if n <= _SPEARMAN_EXACT_MAX_N:
        values, ways = _spearman_null(n)
        return float(ways[values >= s].sum() / ways.sum())
    b = 1.0 / n
    # S is always even without ties; s - 1 is the continuity correction.
    x = (6.0 * (s - 1.0) * b / (n * n - 1.0) - 1.0) * np.sqrt(1.0 / b - 1.0)
    y = x * x
    inner = 0.0072 - 0.0831 * b + y * b * (0.0131 - 0.00046 * y)
    u = (
        x
        * b
        * (
            0.2274
            + b * (0.2531 + 0.1745 * b)
            + y * (-0.0758 + b * (0.1033 + 0.3932 * b) - y * b * (0.0879 + 0.0151 * b - y * inner))
        )
    )
    p = u / np.exp(y / 2.0) + stats.norm.sf(x)
    return float(min(max(p, 0.0), 1.0))


def _t_pvalue(r: float, n: int) -> float:
    """Two-sided p value of a correlation ``r`` of ``n`` pairs from the t distribution."""
    from scipy import stats

    if abs(r) >= 1.0:
        return 0.0
    t = r * np.sqrt((n - 2) / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), n - 2))


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman's rho and its two-sided p value, computed as R's ``cor.test`` does.

    scipy's ``spearmanr`` always uses the t distribution, which is a few
    percent off for the handful of islands a species-area dataset has (for
    the Galápagos power fit, p = 0.035 against 0.037), enough to move a model
    across alpha.
    """
    from scipy import stats

    n = len(x)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    tied = len(np.unique(x)) < n or len(np.unique(y)) < n
    if tied or n > _SPEARMAN_SERIES_MAX_N:
        return rho, _t_pvalue(rho, n)
    s = float(round(np.sum((rx - ry) ** 2)))
    if s > n * (n * n - 1) / 6.0:  # negative correlation: the upper tail of S
        tail = _spearman_upper_tail(s, n)
    else:  # P(S <= s) = 1 - P(S >= s + 2), S being even
        tail = 1.0 - _spearman_upper_tail(s + 2.0, n)
    if tail <= 0.0:
        # Far in the tail the series turns negative and R reports p = 0. The t
        # distribution gives a small positive value there instead.
        return rho, _t_pvalue(rho, n)
    return rho, float(min(2.0 * tail, 1.0))


# R's cor.test gives an exact Kendall p value below this many untied pairs.
_KENDALL_EXACT_BELOW_N = 50


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    """Correlation coefficient and two-sided p value of the test that it is zero."""
    from scipy import stats

    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    if method == "spearman":
        return _spearman(x, y)
    if method == "pearson":
        res = stats.pearsonr(x, y)
    else:
        tied = len(np.unique(x)) < n or len(np.unique(y)) < n
        exact = not tied and n < _KENDALL_EXACT_BELOW_N
        res = stats.kendalltau(x, y, method="exact" if exact else "asymptotic")
    # Unpacked rather than read as .statistic: scipy before 1.10 named
    # kendalltau's estimate .correlation, and the tuple works for every version.
    estimate, p_value = res
    return float(estimate), float(p_value)


def run_checks(
    model: str,
    area: np.ndarray,
    species: np.ndarray,
    fitted: np.ndarray,
    options: CheckOptions,
) -> ResidualChecks:
    """Run the tests ``options`` names on one model's fitted values (see
    ``SARFit.residual_checks``). ``options.neg_check`` and the alphas are not
    used here: they decide what counts as a failure, not what is measured."""
    from scipy import stats

    finite = bool(np.all(np.isfinite(fitted)))
    norm_stat = norm_p = homo_stat = homo_p = np.nan
    # Residuals from a non-finite prediction are NaN or infinite. The rank
    # correlations would still return a number for them, a meaningless one.
    if finite:
        residuals = species - fitted
        with warnings.catch_warnings():
            # A correlation with (nearly) constant values comes back as NaN,
            # which is answer enough. Other warnings are left to be seen, such
            # as Shapiro-Wilk's above 5000 observations.
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            warnings.simplefilter("ignore", stats.NearConstantInputWarning)
            if options.normality is not None:
                norm_stat, norm_p = _normality(residuals, options.normality)
            if options.homoscedasticity is not None:
                against = fitted if options.homoscedasticity == "cor.fitted" else area
                homo_stat, homo_p = _correlation(residuals**2, against, options.cor_method)
    return ResidualChecks(
        model=model,
        normality=options.normality,
        normality_statistic=norm_stat,
        normality_pvalue=norm_p,
        homoscedasticity=options.homoscedasticity,
        cor_method=options.cor_method,
        homoscedasticity_statistic=homo_stat,
        homoscedasticity_pvalue=homo_p,
        negative_fitted=bool(np.any(fitted < 0)),
        finite_fitted=finite,
    )
