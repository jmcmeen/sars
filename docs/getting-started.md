# Getting Started

## Installation

```bash
pip install sars
```

### Optional dependencies

```bash
# Interactive Plotly plots
pip install sars[interactive]

# GeoDataFrame support (geopandas + shapely)
pip install sars[geo]

# Everything
pip install sars[all]
```

## Your first SAR fit

The simplest workflow is to load data and fit a single model:

```python
import sars

galap = sars.load_galap()
fit = sars.sar_power(galap)
print(fit)
```

The returned `SARFit` object contains fitted parameters, goodness-of-fit
statistics, a `predict()` method, and `param_table()` for the parameters'
standard errors (see [Parameter uncertainty](#parameter-uncertainty)):

```python
fit.params      # {'c': 33.1803, 'z': 0.2832}
fit.r_squared   # 0.4912
fit.aicc        # 189.03
fit.predict(100.0)  # predicted richness for 100 km²
```

## The log-log power model

The power model is traditionally fitted as a straight line on log-log axes,
log S = log c + z log A, by ordinary least squares. `lin_pow()` does that and
reports the line's coefficients with their standard errors, t and p values, and
R² in log space:

```python
lp = sars.lin_pow(galap)
print(lp)
# LinPowFit(logc=3.0186, c=20.4636, z=0.3385, R²=0.5298, log_base='e')
lp.z, lp.z_se, lp.z_p   # slope, its standard error and p value
lp.predict(100.0)       # c * 100 ** z, on the arithmetic scale
```

It minimises error in log richness, so its c and z differ from `sar_power()`'s,
which minimises error in richness itself. `compare=True` fits both:

```python
lp = sars.lin_pow(galap, compare=True)
print(lp.comparison[["c", "z"]].round(3))
#                 c      z
# fit
# log-log    20.464  0.339
# nonlinear  33.180  0.283
```

`log_base` takes "e" (the default), "2" or "10"; only `logc` depends on it. If an
island has no species, `con` (default 1) is added to every richness value so the
logs exist, and `predict()` takes it off again. `plot_lin_pow(lp)` draws the data
and the line on log-log axes.
`load_galap()` is one of four bundled datasets; `load_aegean()`, `load_aegean2()` and
`load_niering()` load the others. [Datasets](datasets.md) lists what each holds and
where it was published.
## Parameter uncertainty

`param_table()` gives each parameter's standard error, a t test against zero, and a
confidence interval:

```python
fit.param_table().round(4)
#            estimate  std_error  t_value  p_value  ci_lower  ci_upper
# parameter
# c           33.1803    19.2423   1.7243   0.1066   -8.0903   74.4509
# z            0.2832     0.0994   2.8499   0.0129    0.0701    0.4963

fit.param_table(conf=0.9)  # 90% intervals
```

These come from the usual nonlinear least-squares approximation: the covariance of the
parameters is `sigma² (JᵀJ)⁻¹`, where `J` is the Jacobian of the model at the fitted
values and `sigma² = RSS / (n - p)` for `n` islands and `p` parameters. The t test and the
interval use the t distribution with `n - p` degrees of freedom.

The standard errors, t values and p values are the ones R `sars` reports (a fit's
`sigConf`). R's interval columns are the estimate ± 2 standard errors whatever the sample
size, where this package uses the t quantile, so the intervals here are slightly wider
for small samples (± 2.14 standard errors for the 16 Galápagos islands).

Some fits can't support a standard error. When the data don't identify a parameter at all
(the curve stays the same as it changes, or as it and another parameter change together),
its standard error is `NaN` and `param_table()` warns, naming the model and the
parameters. The same goes for a parameter the curve can't be differentiated reliably in,
and any other whose standard error depends on it, and for a parameter that stopped on one
of the bounds the fit searches within: there the optimum is the bound's, not the data's.
`sar_weibull4` on the Galápagos data runs along a nearly flat ridge, `c` falling toward 0
as `z` grows, until `z` reaches its upper bound of 10, so `z` has no standard error and
`c`'s dwarfs its estimate. A fit with as many parameters as islands has `NaN` too, without
a warning. A large but finite standard error is the ordinary case of a parameter the data
only weakly determine.

The table is computed from the fit's `data`, which is the DataFrame you passed in, so
`param_table()` raises `ValueError` if that DataFrame has been changed since the fit.

The standard errors assume the fitted curve is close to linear in its parameters near the
optimum, so treat them as approximate for small samples, and an interval can include
values the model doesn't allow. For uncertainty in the curve itself, use
[`bootstrap_ci`](#bootstrap-confidence-intervals).

## Comparing all 20 models

Use `sar_multi()` to fit every model and get a ranked summary:

```python
multi = sars.sar_multi(galap)
print(multi.summary[["model", "R2", "AICc", "delta_AICc", "weight"]])
```

The summary table is sorted by AICc. The `weight` column gives Akaike weights
that sum to 1. `shape` is each model's general shape and `observed_shape` the shape its
fitted curve took over the observed areas (see [Models](models.md)).

## Model averaging

Rather than picking a single best model, use `sar_average()` for
information-theoretic model averaging:

```python
avg = sars.sar_average(galap)
avg.predict([1.0, 10.0, 100.0, 1000.0])
```

## Predicting richness

`sar_pred()` tabulates the richness a fit predicts at any areas, including ones
larger than any sampled. It takes a single fit, a `sar_multi()` result (each
model's prediction, best model first), a `sar_average()` result (the weighted
average, as model "average"), a `sar_threshold()` result (its best model) or a
`lin_pow()` fit:

```python
sars.sar_pred(sars.sar_power(galap), [50, 500, 10000]).round(1)
#    model     area  prediction
# 0  power     50.0       100.5
# 1  power    500.0       192.8
# 2  power  10000.0       450.4

multi = sars.sar_multi(galap, models=["power", "loga", "negexpo"])
sars.sar_pred(multi, 5000)        # one row per model
sars.sar_pred(avg, [5000, 10000]) # the averaged prediction
```

Areas must be positive (a threshold fit made with `log_area=False` takes any finite
area). An averaged prediction depends on which models were
averaged, so compare it with R `sars`'s only over the same set of models.
## Residual checks

Least squares assumes residuals that are normal with constant variance. `residual_checks()`
on a fit tests both, and reports whether the model predicts negative richness at any
observed area:

```python
checks = sars.sar_power(galap).residual_checks()
checks.normality_pvalue          # 0.150 (Shapiro-Wilk)
checks.homoscedasticity_pvalue   # 0.037 (squared residuals vs fitted values, Spearman)
checks.negative_fitted           # False
```

`normality` is `"shapiro"` (the default), `"kolmogorov"` or `None`. `homoscedasticity`
correlates the squared residuals with the fitted values (`"cor.fitted"`, the default) or
with area (`"cor.area"`), using `cor_method` `"spearman"`, `"pearson"` or `"kendall"`.

`sar_multi`, `sar_average` and `bootstrap_ci` run no checks by default, as in R `sars`.
Ask for them to leave out the models that fail, at `alpha_normality` and
`alpha_homoscedasticity` (0.05 each):

```python
avg = sars.sar_average(
    galap, normality="shapiro", homoscedasticity="cor.fitted", neg_check=True
)
avg.multi.excluded
# {'power': 'residual variance not constant (spearman correlation of squared residuals
#            with fitted values = 0.529, p = 0.0373 < 0.05)',
#  'epm2': 'residual variance not constant (...)',
#  'loga': 'negative fitted values',
#  'linear': 'residuals not normal (shapiro p = 0.0202 < 0.05)'}
```

A warning lists the models left out, and the weights are shared among the rest. A model
that fails more than one check is listed under the first, in the order normality,
homoscedasticity, negative values. If every model fails, `sar_average` and `bootstrap_ci`
raise `ValueError`. `bootstrap_ci` applies the checks once, to the fits on the original
data: every resample averages the models that passed, so the interval belongs to the same
average `sar_average` reports.

Statistics and p values match R (`shapiro.test`, `ks.test`, `cor.test`), including where
R switches from an exact p value to an approximate one (the Kolmogorov-Smirnov test at
100 residuals, or when values are tied). On the Galápagos data the same models are left
out as by R's `sar_average`. A model whose fitted values aren't all finite can't be
tested, so it is left out whenever any check is asked for. Two differences from
R `sars`:

- `"kolmogorov"` compares the residuals with a normal distribution that has their own mean
  and standard deviation. R's `normaTest = "kolmo"` compares them with a standard normal,
  so its answer depends on the units of richness. Because the mean and standard deviation
  are estimated, the test is conservative: it rejects less often than its alpha.
- R's `normaTest = "lillie"` (the Lilliefors test, which corrects for that) is not
  available. scipy doesn't have it, and no published formula gives its p value over the
  whole range.

## Bootstrap confidence intervals

```python
ci = sars.bootstrap_ci(galap, n_boot=100)
# ci.lower, ci.median and ci.upper are arrays over ci.area_grid; for a central
# curve use the averaged fit's prediction (sars.sar_average(galap).predict(ci.area_grid))
```

By default, `bootstrap_ci` uses `method="fast"`, which fits all models once on
the original data and warm-starts each bootstrap resample from those parameters.
Use `method="full"` for a complete grid search on every resample (much slower
but maximally robust):

```python
ci = sars.bootstrap_ci(galap, n_boot=100, method="full")
```

The default grid of areas is half evenly and half geometrically spaced, so the small
islands are covered when the areas span orders of magnitude; pass `area_grid` to choose
your own. A resample is dropped when no model converges on it, a fit raises, or its
averaged prediction isn't finite somewhere on the grid: `ci.n_boot` counts the resamples
that made the interval, `ci.dropped` counts the others by reason, and a warning says how
many were dropped. Another warning marks where the upper bound runs away, past ten times
the largest observed richness: there more than the tail's share of resamples extrapolate,
typically near areas few observations lie close to, and the interval is best read as
unbounded.

## Threshold models

Detect breakpoints in the species-area relationship (e.g., the small island
effect) using piecewise regression:

```python
th = sars.sar_threshold(galap)
print(th)
print(th.summary)
```

The models are fitted on the natural log of area, as in R `sars`, so `best_breakpoint`
and the summary's `breakpoint` column are on that scale. `breakpoint_area` (the attribute
and the column next to it) gives the same point in your area units. Pass `log_area=False`
to fit on area itself.

R `sars`'s other threshold models are available too: DiscOne, ContTwo, ZslopeTwo and DiscTwo.
Choose models with `models=`. The summary's `breakpoint2` columns hold a two-threshold
model's second breakpoint. `best_breakpoints` and `breakpoints_area` give all of the best
model's breakpoints:

```python
models = ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo", "Linear"]
th = sars.sar_threshold(galap, models=models)
print(th.summary)
print(th.all_fits["ContTwo"]["breakpoints"])
```

### Confidence intervals for a breakpoint

`threshold_ci` gives a model's breakpoints bootstrap confidence intervals (Toms &
Lesperance 2003). It resamples the model's residuals onto its fitted values, reruns the
breakpoint search on each resample, and takes percentiles of the breakpoints found:

```python
import numpy as np

ci = sars.threshold_ci(th, model="ContOne", rng=np.random.default_rng(1))
print(ci)                             # estimate [lower, upper] in area units
print(ci.estimates, ci.lower, ci.upper)   # on the fitted scale (ln area)
print(ci.lower_area, ci.upper_area)
```

Without `model` it takes the best model that has a breakpoint; Linear has none. A
two-threshold model gets an interval for each breakpoint. The default 1000 resamples are
scored together for the models searched over a grid, so they cost about as much as a few
fits of the model for ContOne and ZslopeOne (a fraction of a second) and roughly ten fits
for ContTwo and ZslopeTwo (a few seconds on the 16 Galápagos islands). DiscOne and DiscTwo
are searched one resample at a time: a second or a few on tens of islands, some tens of
seconds for DiscTwo on a hundred. `ci.samples` holds every resample's breakpoints.

Read the intervals with the data in mind. With few islands they are wide: a bound at the
third-smallest or third-largest area (the edge of the search) means the data don't place
the breakpoint on that side. A discontinuous model's breakpoint is always an observed
area, so its bounds are too, and when one jump dominates every resample can land on the
same island, giving an interval of no width. That says the resamples agree, not that the
threshold is known exactly. Percentile intervals for a breakpoint also run a little narrow:
in simulations of 40 islands with one true breakpoint, 95% intervals covered it about 85 to
90% of the time. R `sars`'s `threshold_ci` offers the bootstrap for ContOne and
ZslopeOne only; its draws differ, so its intervals agree with these only roughly.

`plot_threshold` draws a model on the scale it was fitted on, with its breakpoints marked
and, given `ci`, their intervals shaded:

```python
sars.plot_threshold(th, ci=ci)
sars.plot_threshold(th, model="DiscOne")   # separate lines either side of the jump
```
## Random placement

Coleman's (1981) random placement model asks whether species-area data need any ecology at
all: if every individual lands on a site by chance, in proportion to the site's area, how
many species should each site hold? The answer depends only on each species' total
abundance, so nothing is fitted. `coleman()` takes a species-by-site abundance table,
**species in rows and sites in columns** (as R `sars` does; transpose a sites-by-species
table with `.T`), and the site areas in column order. Areas in a `pd.Series` labelled
with the table's column names are matched by name instead, whatever their order.

This example simulates a community by random placement, so the model should hold:

```python
import numpy as np
import pandas as pd
import sars

rng = np.random.default_rng(3)
area = np.array([1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0])
totals = rng.geometric(0.05, size=30)  # total abundance of each of 30 species
# Each individual lands on a site with probability equal to its share of the area.
counts = np.array([rng.multinomial(n, area / area.sum()) for n in totals])
abundance = pd.DataFrame(counts, columns=[f"site{i + 1}" for i in range(len(area))])

fit = sars.coleman(abundance, area)
print(fit)
# ColemanFit(30 species, 8 sites, 1 outside ±1 sd (12%), random placement not rejected)
print(fit.sites)
```

`fit.sites` has a row per site: `area`, `relative_area`, `observed` and `expected`
richness, the standard deviation `sd`, and `within_sd`. Random placement is rejected
(`fit.rejected`) when more than a third of the sites lie more than one standard deviation
from the expected curve; `fit.fraction_outside` is that share. `fit.predict(area)` and
`fit.predict_sd(area)` give the curve and its standard deviation at any area up to the
total, and `sars.plot_coleman(fit)` draws both with the observed points.

Treat the verdict as a rough guide. One standard deviation either side holds about two
thirds of a roughly normal spread, so close to a third of the sites are expected outside
it even when placement really is random, and with few sites the rule often rejects such
data: rerun the simulation above with other seeds to see it happen. Look at the plot and at
which sites fall outside (the small ones, the large ones, all on one side) before reading
much into `rejected`.

The model needs counts of individuals. Presence-absence data won't do, and species absent
from every site are ignored.
## The general dynamic model

On oceanic islands richness depends on island age as well as area: it rises while a
volcanic island is being built and falls as the island erodes and subsides. The general
dynamic model (GDM) of Whittaker, Triantis & Ladle (2008) therefore adds a hump-shaped
function of age to the species-area relationship: area, time and time squared, their
"ATT²" model. `sar_gdm` fits it to a DataFrame with a column of island ages:

```python
islands = sars.load_galap()
# Illustrative ages, from the example in R sars's gdm help page (not real island ages)
islands["time"] = [4, 1, 13, 16, 15, 2, 6, 4, 5, 11, 3, 9, 8, 10, 12, 7]

gdm = sars.sar_gdm(islands, model="loga", mod_sel=True)
print(gdm)             # coefficients Int, A, Ti, Ti2
print(gdm.p_values)    # also gdm.std_errors, gdm.fitted, gdm.r_squared, gdm.aic
print(gdm.comparison)  # the GDM against simpler models
gdm.predict(area=100, time=[2, 8, 14])
```

A hump in age shows as a positive `Ti` (time) and a negative `Ti2` (time squared).
`mod_sel=True` checks whether the data support it, ranking the GDM by AICc against three
simpler models: area and time ("A + T"), area only ("A") and an intercept only. The
table has both AIC and AICc, with their differences and weights. If the solver doesn't
converge for a power-form candidate, `sar_gdm` warns, naming it, and lists it last with no
weight.

`model` chooses how area enters, with the forms and parameter names of R `sars`'s `gdm`:

| `model` | Equation | Fitted by |
|---------|----------|-----------|
| `"loga"` (default) | S = Int + A·log(area) + Ti·T + Ti2·T² | ordinary least squares |
| `"linear"` | S = Int + A·area + Ti·T + Ti2·T² | ordinary least squares |
| `"power_area"` | S = exp(Int + A·log(area) + Ti·T + Ti2·T²) | nonlinear least squares |
| `"power_area_time"` | S = exp(Int + A·log(area) + Ti·log(T) + Ti2·log(T)²) | nonlinear least squares |

`"loga"` is the original ATT² model (R's `"loga"` and `"ATT2"` are the same fit). The power
forms are fitted to richness itself, not to its log, so all four share a response and can
be compared by AIC. Their coefficients, `Int` included, are on the log scale.
`model="all"` fits the four and ranks them:

```python
every = sars.sar_gdm(islands, model="all")
print(every.summary)
print(every.fits["power_area"])   # or every.best
```

Use `area_col`, `species_col` and `time_col` if your columns have other names. The fit
doesn't depend on the units of area or age (years or millions of years). To fit a
variant with log-transformed richness or age (Steinbauer et al. 2013), transform the
column before passing it.

Reference: Whittaker RJ, Triantis KA, Ladle RJ (2008). A general dynamic theory of
oceanic island biogeography. *Journal of Biogeography*, 35, 977-994.

Among the three default models (ContOne, ZslopeOne and Linear), the straight line has the
lowest AICc on the 16 Galápagos islands. The 173 Aegean islands of `load_aegean2()` favour
a threshold:

```python
th = sars.sar_threshold(sars.load_aegean2())
print(th)
# ThresholdFit(best='ContOne', breakpoint=2.305 (ln area), breakpoint_area=10.02)
```

## Bringing your own data

`sars` expects a DataFrame with `area` and `species` columns:

```python
import pandas as pd

data = pd.DataFrame({
    "area": [0.5, 1.0, 2.0, 5.0, 10.0, 50.0],
    "species": [12, 18, 25, 40, 55, 90],
})
fit = sars.sar_power(data)
```

Or use the I/O adapters:

```python
# From CSV
data = sars.from_csv("my_islands.csv", area_col="island_area", species_col="richness")

# From a GeoDataFrame (computes area from geometries)
data = sars.from_geodataframe(gdf, species_col="n_species")
```

`from_geodataframe` needs the `geo` extra (`pip install sars[geo]`). It measures each
polygon in the GeoDataFrame's own coordinate system, so reproject longitude/latitude
data to an equal-area projection first, for example `gdf.to_crs("EPSG:6933")`; it raises
an error for coordinates in degrees, whose "areas" mean nothing. `crs_units` chooses
km² (the default), hectares or m². The
[areas from polygons](examples/geodataframe.ipynb) example works through it with real
county outlines.
