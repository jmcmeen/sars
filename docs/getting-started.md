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
statistics, and a `predict()` method:

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
