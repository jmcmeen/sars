# SAR Models

`sars` implements all 20 models from the R `sars` package. Each model has a
dedicated fitting function (e.g., `sars.sar_power()`) that returns a `SARFit`
object.

Each model's general shape is as classified by Matthews et al. (2019, Table 1), with the
corrections R `sars` has made since: convex (decelerating), sigmoid (S-shaped, with an
inflection point), or convex/sigmoid where the parameters decide which. It is the `shape`
column of `sar_multi`'s summary: a property of the model, not a test of the fitted curve.

The summary's `observed_shape` column, and `SARFit.observed_shape()`, say which shape a
fitted curve took. The curve is evaluated between the smallest and largest observed area,
on arithmetic axes, and classified by how its slope changes:

| Observed shape | The fitted curve, over the observed areas |
|----------------|-------------------------------------------|
| `linear` | follows the straight line through its end points |
| `convex` | rises, and its slope only falls |
| `sigmoid` | rises, and its slope first rises, then falls (one inflection point) |
| `none` | anything else: it falls somewhere, only accelerates, or has several inflection points |

Each test allows a departure of a thousandth of the curve's range of richness, so a bend
or a dip too small to see doesn't change the label. The two columns can differ. A sigmoid model whose inflection point lies below the smallest island
is `convex` over the data, and a model that can turn down at large areas (p1, epm1, epm2)
is `none` when its fit does.

`sars.lin_pow()` fits the power model the traditional way instead, as a straight line
on log-log axes (log S = log c + z log A) by ordinary least squares, and returns a
`LinPowFit`. It isn't one of the 20 models `sar_multi` compares: its R² is in log
space, so it can't be ranked against theirs.

## Non-asymptotic models

These models have no asymptote parameter. Most increase without bound as area
increases; p1 can peak and then fall, and epm1 levels off at c when d > 0. The sigmoid
ones (epm2, p2) have an inflection point but no asymptote.

| Function | Formula | Parameters | Shape |
|----------|---------|------------|-------|
| `sar_power` | S = c &middot; A^z | c, z | convex |
| `sar_powerR` | S = f + c &middot; A^z | f, c, z | convex |
| `sar_loga` | S = c + z &middot; log(A) | c, z | convex |
| `sar_koba` | S = c &middot; log(1 + A/z) | c, z | convex |
| `sar_linear` | S = c + m &middot; A | c, m | linear |
| `sar_epm1` | S = c &middot; A^(z &middot; A^-d) | c, z, d | convex/sigmoid |
| `sar_epm2` | S = c &middot; A^(z - d / A) | c, z (0 to 1), d | sigmoid |
| `sar_p1` | S = c &middot; A^z &middot; exp(-d &middot; A) | c, z, d | convex/sigmoid |
| `sar_p2` | S = c &middot; A^z &middot; exp(-d / A) | c, z, d | sigmoid |

## Asymptotic models

These models approach a finite asymptote as area increases.

| Function | Formula | Parameters | Shape |
|----------|---------|------------|-------|
| `sar_monod` | S = d &middot; A / (c + A) | d, c | convex |
| `sar_negexpo` | S = d &middot; (1 - exp(-z &middot; A)) | d, z | convex |
| `sar_ratio` | S = (c + z &middot; A) / (1 + d &middot; A) | c, z, d | convex |
| `sar_asymp` | S = d - c &middot; exp(-z &middot; A) | d, c, z | convex/sigmoid |
| `sar_mmf` | S = d / (1 + c &middot; A^(-z)) | d, c, z | convex/sigmoid |
| `sar_heleg` | S = c / (f + A^(-z)) | c, f, z | convex/sigmoid |
| `sar_gompertz` | S = d &middot; exp(-exp(-z &middot; (A - c))) | d, z, c | sigmoid |
| `sar_weibull3` | S = d &middot; (1 - exp(-c &middot; A^z)) | d, c, z | sigmoid |
| `sar_weibull4` | S = d &middot; (1 - exp(-c &middot; A^z))^f | d, c, z, f | sigmoid |
| `sar_chapman` | S = d &middot; (1 - exp(-z &middot; A))^c | d, z, c | sigmoid |
| `sar_betap` | S = d &middot; (1 - (1 + (A/c)^z)^(-f)) | d, c, z, f | sigmoid |

R `sars` writes asymp as S = d - c &middot; z^A, so R's z is exp(-z) here.

## Fitting details

All models are fitted using nonlinear least squares (scipy `least_squares`)
with a multi-start grid of initial values to avoid local minima:

- **2-parameter models**: &ge;36 starting points
- **3-parameter models**: &ge;100 starting points
- **4-parameter models**: &ge;200 starting points

The fit runs on rescaled data: area divided by its geometric mean and richness by its
maximum, with the parameters converted back afterwards. The starting values and bounds
are therefore near 1 whatever your units, and every model except epm1 and epm2 gives the
same fit in km² or m², for tens of species or thousands. epm1 and epm2 are the exception
because their exponent depends on area itself, so a change of area units changes their
fit; only their richness is rescaled.

If fitting fails for all starting values, a `SARFit` with `converged=False`
is returned rather than raising an exception.

`SARFit.param_table()` gives the parameters' standard errors, t tests and confidence
intervals, with `NaN` for parameters the data don't identify (see
[Getting Started](getting-started.md#parameter-uncertainty)).

Information criteria (AIC, AICc, BIC) are computed using the normal
log-likelihood convention, consistent with the R `sars` package.
