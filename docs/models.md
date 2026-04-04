# SAR Models

`sars` implements all 20 models from the R `sars` package. Each model has a
dedicated fitting function (e.g., `sars.sar_power()`) that returns a `SARFit`
object.

## Non-asymptotic models

These models increase without bound as area increases.

| Function | Formula | Parameters |
|----------|---------|------------|
| `sar_power` | S = c &middot; A^z | c, z |
| `sar_powerR` | S = f + c &middot; A^z | f, c, z |
| `sar_loga` | S = c + z &middot; log(A) | c, z |
| `sar_linear` | S = c + m &middot; A | c, m |
| `sar_epm1` | S = c &middot; A^z &middot; exp(d &middot; (log A)^2) | c, z, d |
| `sar_epm2` | S = c &middot; A^(z1 &middot; A^z2) | c, z1, z2 |
| `sar_p1` | S = c &middot; A^z &middot; exp(-d &middot; A) | c, z, d |
| `sar_p2` | S = c &middot; A^z &middot; exp(-d / A) | c, z, d |

## Asymptotic convex models

These models approach a finite asymptote with a convex (decelerating) curve.

| Function | Formula | Parameters |
|----------|---------|------------|
| `sar_koba` | S = c &middot; log(1 + A/z) | c, z |
| `sar_monod` | S = d &middot; A / (c + A) | d, c |
| `sar_negexpo` | S = d &middot; (1 - exp(-z &middot; A)) | d, z |
| `sar_asymp` | S = d - c &middot; exp(-z &middot; A) | d, c, z |
| `sar_ratio` | S = (c + z &middot; A) / (1 + d &middot; A) | c, z, d |

## Asymptotic sigmoid models

These models have an S-shaped curve, approaching an asymptote via an
inflection point.

| Function | Formula | Parameters |
|----------|---------|------------|
| `sar_mmf` | S = d / (1 + c &middot; A^(-z)) | d, c, z |
| `sar_gompertz` | S = d &middot; exp(-exp(-z &middot; (A - c))) | d, z, c |
| `sar_weibull3` | S = d &middot; (1 - exp(-c &middot; A^z)) | d, c, z |
| `sar_weibull4` | S = d &middot; (1 - exp(-c &middot; A^z))^f | d, c, z, f |
| `sar_chapman` | S = d &middot; (1 - exp(-z &middot; A))^c | d, z, c |
| `sar_betap` | S = d &middot; (1 - (1 + (A/c)^z)^(-f)) | d, c, z, f |
| `sar_heleg` | S = d / (1 + slope^log(c / A)) | d, slope, c |

## Fitting details

All models are fitted using nonlinear least squares (scipy `least_squares`)
with a multi-start grid of initial values to avoid local minima:

- **2-parameter models**: &ge;36 starting points
- **3-parameter models**: &ge;100 starting points
- **4-parameter models**: &ge;200 starting points

If fitting fails for all starting values, a `SARFit` with `converged=False`
is returned rather than raising an exception.

Information criteria (AIC, AICc, BIC) are computed using the normal
log-likelihood convention, consistent with the R `sars` package.
