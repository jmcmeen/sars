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
fit.params      # {'c': 33.1792, 'z': 0.2832}
fit.r_squared   # 0.4912
fit.aicc        # 189.03
fit.predict(100.0)  # predicted richness for 100 km²
```

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
