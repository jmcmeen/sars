# Getting Started

## Installation

```bash
pip install sars
```

sars needs Python 3.11 or later.

### Optional dependencies

```bash
# GeoDataFrame support (geopandas + shapely)
pip install "sars[geo]"

# The HTTP server, `sars serve` (FastAPI + uvicorn)
pip install "sars[server]"

# Everything
pip install "sars[all]"
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
`load_galap()` is one of seven bundled datasets; `load_gala()`, `load_aegean()`,
`load_aegean2()`, `load_niering()`, `load_schrader()` and `load_xu()` load the
others. [Datasets](datasets.md) lists what each holds and where it was published.

### Many archipelagos at once

Fitting each archipelago on its own gives noisy slopes for the small ones; pooling them
into one curve ignores that they differ. `lin_pow_mixed` fits the log-log model to all of
them at once as a mixed model. Each archipelago's intercept and slope deviate from
shared means, and the spread of those deviations is estimated too. It matches R's
`lme4::lmer(log(S) ~ log(A) + (log(A) | archipelago))`:

```python
fit = sars.lin_pow_mixed({
    "gala": sars.load_gala(), "aegean": sars.load_aegean(), "aegean2": sars.load_aegean2(),
    "niering": sars.load_niering(), "schrader": sars.load_schrader(), "xu": sars.load_xu(),
})
fit.z, fit.z_se, fit.sd_z     # (0.325, 0.034, 0.079)
fit.groups[["n", "z", "z_alone"]].round(3)
#             n      z  z_alone
# gala       30  0.352    0.366
# aegean     90  0.199    0.195
# aegean2   173  0.338    0.336
# niering    32  0.349    0.348
# schrader   60  0.426    0.432
# xu         42  0.288    0.302
```

Across these six archipelagos the mean slope is 0.33 ± 0.03. Archipelagos' slopes vary
around it with a standard deviation of 0.08. Allowing for the mean's own uncertainty too,
a new archipelago's z would most likely fall between about 0.16 and 0.49. Each archipelago's pooled `z` sits beside its own
`z_alone`. The pooled values move towards the mean, most for archipelagos with few
islands or a poor fit. Intercept and slope move together, so with correlated deviations
a slope can move slightly the other way, as Xu's does here. `fit.predict(area)` gives
the mean curve, and `fit.predict(area, group="xu")` an archipelago's own. As with
`lin_pow`, when an island has no species `con` (1) is added to every richness value
before taking logs. That happens here because of Schrader's empty islets, and it applies
to all six archipelagos, so that they share one log scale.

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
`c`'s dwarfs its estimate. `fit.at_bounds` names such parameters without the warning
(`("z",)` here), for code that compares many fits. A fit with as many parameters as islands has `NaN` too, without
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

A model that doesn't converge, or whose fit fails, is left out of the summary, the
weights and any average. `sar_multi` and `sar_average` warn, naming it, and list it with
the reason in `not_fitted`:

```python
multi.not_fitted
# {} when every model converged, as on these data; otherwise e.g.
# {'betap': 'did not converge'}
```

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

`normality` is `"shapiro"` (the default), `"lillie"` (Lilliefors), `"kolmogorov"` or
`None`. `homoscedasticity`
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

Statistics and p values match R (`shapiro.test`, `ks.test`, `cor.test`, and
`nortest::lillie.test` below p = 0.1, see below), including where
R switches from an exact p value to an approximate one (the Kolmogorov-Smirnov test at
100 residuals, or when values are tied). On the Galápagos data the same models are left
out as by R's `sar_average`. A model whose fitted values aren't all finite can't be
tested, so it is left out whenever any check is asked for. Two differences from
R `sars`:

- R's `normaTest = "kolmo"` compares the residuals with a standard normal, so its answer
  depends on the units of richness, and it isn't offered. `"kolmogorov"` compares them
  with a normal that has their own mean and standard deviation. Because those are
  estimated, that test is conservative: it rejects less often than its alpha. `"lillie"`,
  the Lilliefors test, uses the same statistic with a p value that allows for this.
- The Lilliefors p value matches R's below 0.1, where both use Dallal and Wilkinson's
  (1986) approximation. Above 0.1, R uses a polynomial fit of its own, and `sars` instead
  counts how often 20,000 simulated normal samples of the same size give a statistic at
  least as large. The result is within about 0.01 of the true value, where R's can be off
  by 0.03 (for 5 residuals with D = 0.177, R gives 0.873, `sars` 0.900, and a
  400,000-sample simulation 0.898). Where Dallal and Wilkinson's value is above 0.1,
  `sars` never reports less than 0.1, so the p value only falls as D grows; R's dips to
  0.091 just past its switch for 5 residuals. At any alpha up to 0.09, including the
  usual 0.05, both leave out the same models. The test needs at least five residuals:
  with fewer it isn't run (R refuses), and `sars` warns.

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

The resamples are independent, so they can be fitted in several processes at once:
`n_jobs=-1` uses one per core (`threshold_ci` takes `n_jobs` too). The interval is the
same for any `n_jobs`, given the same `rng`. On an 8-core machine, 1000 fast resamples
of the Galápagos data took 26 s instead of 149 s. In a script, put the call under
`if __name__ == "__main__":`, since each worker process imports the script:

```python
if __name__ == "__main__":
    ci = sars.bootstrap_ci(galap, n_boot=1000, n_jobs=-1)
```

## Habitat loss

The main conservation use of a species–area curve is to ask how many species an area
loses when part of its habitat goes. `habitat_loss` compares the model-averaged curve
at the area before and after, with a bootstrap interval for the loss:

```python
import numpy as np

gala = sars.load_gala()
rising = ["power", "loga", "koba", "negexpo", "monod", "ratio"]
loss = sars.habitat_loss(
    gala, area=900, fraction_lost=[0.25, 0.5, 0.9], models=rising, rng=np.random.default_rng(42)
)
loss.table[["remaining", "richness_after", "species_lost", "lost_lower", "lost_upper",
            "proportion_lost"]]
#    remaining  richness_after  species_lost  lost_lower  lost_upper  proportion_lost
# 0      675.0          262.99         16.34        2.21       93.73             0.06
# 1      450.0          238.36         40.96        5.83      187.59             0.15
# 2       90.0          131.23        148.09       48.57      339.84             0.53
```

An island of 900 km² holds about 279 plant species on this curve, and one of half that
size about 238. Losing half the habitat therefore costs about 41 species, or 15%. Give
`remaining=` instead of `fraction_lost=` for the area left. Each resample's own averaged
curve gives its own loss, so the interval is for the loss itself. It is wide: 30 islands
pin down the curve's shape only loosely.

Read the numbers with three things in mind:

- **It is the eventual loss.** The estimate is how many species an area of the remaining
  size holds once it settles. Species can hang on for a long time before they die out
  (an "extinction debt").
- **It is likely an overestimate.** A curve fitted across islands of different sizes
  describes how many species an area holds. Losing part of an area costs fewer than
  that (He and Hubbell 2011). The calculation also treats what remains as one block
  rather than fragments.
- **The interval has the bootstrap's limits.** Areas at the edge of the data are
  extrapolated in the resamples that leave the largest island out, and a warning flags
  predictions there that run away. Averaging all 20 models brings in curves that fall with area in
  places, so a few resamples show negative losses (also warned). That is why the example
  averages six models whose curves only rise.

## Cross-validation

AICc ranks models by how well they fit, with a penalty for parameters. With few islands,
models it ranks closely can predict quite differently, and its small-sample correction
is itself an approximation. `loo_cv` asks the question directly: leave each island out,
fit every model to the rest, and see how well it predicts the island it didn't see.

```python
gala = sars.load_gala()
cv = sars.loo_cv(gala, n_jobs=-1)
cv.summary[["model", "rmse", "max_error", "worst", "cv_rank", "ic_rank"]].head()
#       model   rmse  max_error       worst  cv_rank  ic_rank
# 0      koba  64.43     203.43  Santa Cruz        1          2
# 1     ratio  66.31     213.88  Fernandina        2          7
# 2     monod  69.04     217.29  Fernandina        3          8
# ...
```

`rmse` and `mae` are the root mean square and mean absolute errors of the left-out
predictions, in species. `max_error` and `worst` give the largest error and the island it
was for (named from an `island`, `site`, `locality` or `name` column, or the one `label`
gives). `ic_rank` is the model's rank by AICc on all the data; `ic="AIC"` or `"BIC"`
compares with those instead, and with very few islands, where AICc is undefined, it has
to. On Johnson and Raven's 30 Galápagos islands, AICc's best model, p1, comes 16th
by cross-validation: it predicts Isabela, the largest island, poorly when Isabela is left
out. The row `"average"` is the AICc model average, refitted in each fold. One model that
extrapolates wildly in some fold can spoil it, as epm2 does on the 16-island `galap` data.
`cv.predictions` holds every left-out prediction. A model whose fit fails in some fold
is ranked after those that never fail; `rmse_common` compares every model on the islands
all of them predicted.

Each fold runs the full multi-start search, so `n_jobs=-1` is worth using. On these 30
islands it takes about 3 minutes on one core. `warm_start=True` instead starts each fold
from the fit to all the data, as the bootstrap's fast method does: 15 to 30 times faster,
but those starting values have seen the island left out. A flexible model can stay near
them rather than find the best fit to the other islands, and its error comes out too
small. Here, warm starts move weibull4 from 15th to 4th.

## Saving and loading results

Any result `sars` returns (a single fit, a multi-model fit, an average, a threshold fit,
a bootstrap interval, the result of `sar_many`) can be saved as JSON and loaded back,
data, parameters and settings included:

```python
avg = sars.sar_average(galap)
sars.save(avg, "galap_average.json")

avg = sars.load("galap_average.json")  # an AveragedSAR again
avg.predict([10.0, 100.0])
```

`sars.to_json(result)` and `sars.from_json(text)` do the same with a string. The file is
standard JSON (an infinite AICc is written as a string and read back as a number), and
it records the version of `sars` that wrote it. Loading rebuilds only `sars`'s own result
types, so a file from elsewhere can't run code, unlike a pickle.

## Many datasets at once

`sar_many` fits the same models to several datasets and puts the results in one table,
one row per dataset: the multi-model average's best model and its weight, the log-log
power fit, and the threshold model `sar_threshold` chooses. Pass a dict of named
datasets, or one long DataFrame with `by=` naming the column that says which dataset
each row belongs to:

```python
many = sars.sar_many({"galap": sars.load_galap(), "gala": sars.load_gala(), "xu": sars.load_xu()})
print(many.summary[["dataset", "n", "z", "R2", "best_model", "threshold_model", "breakpoint_area"]])
#   dataset   n     z    R2 best_model threshold_model  breakpoint_area
# 0   galap  16  0.34  0.53    negexpo          Linear              NaN
# 1    gala  30  0.39  0.76         p1       ZslopeOne            10.85
# 2      xu  42  0.35  0.79      power         ContOne            17.00

sars.sar_many(long_table, by="archipelago", threshold=False, n_jobs=-1)
```

Each fit runs on its own: one that fails (a dataset too small for the threshold models,
say) is named with its error in the `error` column, without stopping the dataset's other
fits or the other datasets, and the warnings each dataset raised are listed against it
in `warnings`.
The fits themselves are kept, by name, in `many.averages`, `many.log_log` and
`many.thresholds`. `n_jobs` fits the datasets in parallel, as for `bootstrap_ci`.

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

`threshold_ci` gives a model's breakpoints confidence intervals by either of the methods
Toms & Lesperance (2003) describe. The default bootstrap resamples the model's residuals
onto its fitted values, reruns the breakpoint search on each resample, and takes
percentiles of the breakpoints found:

```python
import numpy as np

ci = sars.threshold_ci(th, model="ContOne", rng=np.random.default_rng(1))
print(ci)                             # estimate [lower, upper] in area units
print(ci.estimates, ci.lower, ci.upper)   # on the fitted scale (ln area)
print(ci.lower_area, ci.upper_area)
```

Without `model` it takes the best model that has a breakpoint; Linear has none. A
two-threshold model gets an interval for each breakpoint. The default 1000 resamples are
scored together against the candidate breakpoints, which depend only on the areas, so they
cost about as much as a few fits of the model: a fraction of a second for the
one-threshold models, and a few seconds at most for ContTwo and ZslopeTwo. `ci.samples`
holds every resample's breakpoints.

For ContOne and ZslopeOne, `method="F"` inverts an F test instead: the interval holds
every candidate breakpoint whose fit, with the breakpoint fixed there, the F test at level
`1 - conf` doesn't reject against the best fit. It needs no resampling:

```python
ci_f = sars.threshold_ci(th, model="ContOne", method="F")
```

Read the intervals with the data in mind. With few islands they are wide: a bound at the
third-smallest or third-largest area (the edge of the search) means the data don't place
the breakpoint on that side. A discontinuous model's breakpoint is always an observed
area, so its bounds are too, and when one jump dominates every resample can land on the
same island, giving an interval of no width. That says the resamples agree, not that the
threshold is known exactly. In simulations of 40 islands with one true breakpoint, 95%
intervals covered it 91 to 95% of the time with either method, and less when an interval
is only a grid step or two wide (use a smaller `interval` then). Toms & Lesperance
recommend the bootstrap for small samples and the F test for larger ones.

R `sars`'s `threshold_ci` offers both methods for ContOne and ZslopeOne only. Its
bootstrap draws differ, so its bootstrap intervals agree with these only roughly. Its F
intervals compare the statistic with F(conf; 1, p), p the number of regression
coefficients, where the test's residual degrees of freedom, n − p, belong, and it searches
every candidate from the smallest area to the largest, ignoring the three-island window.
On many islands that makes its intervals much wider: on the 173 Aegean plant islands,
ContOne's 95% interval in ln area is 1.45 to 3.02 in R and 1.78 to 2.70 here.

`plot_threshold` draws a model on the scale it was fitted on, with its breakpoints marked
and, given `ci`, their intervals shaded:

```python
sars.plot_threshold(th, ci=ci)
sars.plot_threshold(th, model="DiscOne")   # separate lines either side of the jump
```
## Sampling effort

The richness recorded on an island is a sample, and when larger islands are sampled
harder, part of the species-area relationship is the effort's (Gooriah et al. 2021).
Given each site's species counts, `standardised_richness` puts the sites on an equal
footing first. It rarefies or extrapolates every sample to the same sample coverage,
the share of the community's individuals that belong to species already found, as R's
iNEXT does:

```python
xu = sars.load_xu()                  # 42 islands: area and observed richness
counts = sars.load_xu_abundance()    # the same islands' butterfly counts, species by island
data = sars.standardised_richness(counts, area=xu["area"])
data[["site", "area", "species", "observed", "individuals", "coverage", "m"]].head(3)
#        site     area  species  observed  individuals  coverage       m
# 0  Zhoushan  476.165   17.051        41          732     0.992  48.853
# 1   Liuheng   93.658   19.564        39          294     0.949  54.477
# 2   Daishan  104.967   12.429        32         1102     0.995  35.742

sars.lin_pow(xu).z, sars.lin_pow(data).z
# (0.351, 0.263)
```

From 2 to 1,102 butterflies were counted on these islands, more on the larger ones
(r = 0.84 on log scales). At an equal coverage of 85.5%, the log-log slope falls from
0.35 to 0.26, so about a quarter of the apparent effect of area on the observed
richness came from sampling. `m` is the number of individuals each site was rarefied (or
extrapolated) to; `coverage` is its sample's own coverage, and `coverage_at_m` the
coverage it was standardised to. `data` goes to any fitting function like other data.

`base="size"` standardises to the same number of individuals instead. `level` sets the
coverage, or the number of individuals. By default it is, as in iNEXT, the largest
value every site reaches by extrapolating to twice its sample, and a warning flags any
site extrapolated further. The estimates match iNEXT's `estimateD` for richness (q = 0).

## Habitat models

Richness can depend on how many habitats an island has as well as on its size.
`sar_habitat` fits R sars's four habitat models:

- **choros**, on area times the number of habitats;
- **jigsaw**, with habitats as pieces of area, each with its own species;
- **Kallimanis**, whose slope rises with habitat diversity;
- **the classical curve**, on area alone.

```python
fit = sars.sar_habitat(sars.load_habitat())  # Skyros land snails: area, habitats, species
fit.summary[["model", "z", "d", "R2", "AICc"]]
#         model      z      d     R2    AICc
# 0      choros  0.147    NaN  0.914   2.228
# 1      jigsaw  0.086  0.501  0.940   2.614
# 2       power  0.177    NaN  0.877   6.444
# 3  Kallimanis  0.179 -0.000  0.877  11.150
```

By default (`mod_type="power_log"`, as in R) the power forms are fitted as regressions on
the logs. `mod_type="power"` fits them by non-linear least squares, and `"logarithmic"`
fits the logarithmic forms instead. On these 12 islands, choros and jigsaw beat area
alone, and Kallimanis doesn't. In the jigsaw model richness rises faster with habitats (d = 0.50) than
with area (z = 0.09). `fit.predict(area, habitats)` uses the best model, or the one named
in `model=`.

`sar_countryside` fits the countryside model, for landscapes of several habitats whose
species differ in which habitats they use. Each group of species, sorted by habitat
affinity, gets a curve of a weighted sum of the habitat areas. The weights say how much
each habitat counts for that group:

```python
cs = sars.load_countryside()  # Portuguese plants: three habitats, four species groups
fit = sars.sar_countryside(
    cs,
    habitats=["Area_AG", "Area_SH", "Area_QF"],
    species=["Spcs_AG", "Spcs_SH", "Spcs_QF", "Spcs_UB"],
)
fit.affinity.round(3)
#          Area_AG  Area_SH  Area_QF
# Spcs_AG    1.000    0.000      0.0
# Spcs_SH    0.000    1.000      0.0
# Spcs_QF    0.000    0.000      1.0
# Spcs_UB    0.420    0.106      1.0
```

Each habitat's specialists depend on it alone. The ubiquitous species use all three,
oak forest most. The fit predicts richness for any mix of habitats. Here are 26 ha of
each habitat alone, then a third of each:

```python
fit.predict([[262144, 0, 0], [0, 262144, 0], [0, 0, 262144], [87381, 87381, 87382]])
#    Spcs_AG  Spcs_SH  Spcs_QF  Spcs_UB  total
# 0     50.3      0.6      5.3      7.7   63.9
# 1      3.8     16.6      4.1      6.0   30.4
# 2      4.6      5.1     33.7      9.1   52.5
# 3     44.0     14.7     28.9      8.0   95.6
```

On these data, the countryside model's error in total richness is half that of one power
curve of total area (`fit.rss`: 5,256 against 10,573). Both functions reproduce R sars's
fits, and their results save as JSON like the others.

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

For a test with a known error rate, `fit.placement_test()` simulates random placement
itself: it scatters each species' individuals over the sites in proportion to their areas
(999 times by default), and compares the data's distance from the expected curve (the
squared deviations of the sites, in standard deviations, summed) with the simulations':

```python
test = fit.placement_test(rng=np.random.default_rng(1))
print(test)
# ColemanTest(statistic=2.21, p_value=0.973, n_sim=999, rule_rejection_rate=50%)
```

A small `p_value` is evidence against random placement. `rule_rejection_rate` is how often
the one-third rule rejected the simulated communities, all of them placed at random: for
these species totals and site areas the rule would reject random placement half the time.

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

`from_geodataframe` needs the `geo` extra (`pip install "sars[geo]"`). It measures each
polygon in the GeoDataFrame's own coordinate system, so reproject longitude/latitude
data to an equal-area projection first, for example `gdf.to_crs("EPSG:6933")`; it raises
an error for coordinates in degrees, whose "areas" mean nothing. `area_units` chooses
km² (the default), hectares or m². The
[areas from polygons](examples/geodataframe.ipynb) example works through it with real
county outlines.
