# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions up to 0.7.5 predate this file; their entries below were
reconstructed from the commit history and the
[GitHub releases](https://github.com/jmcmeen/sars/releases) page.

## [Unreleased]

### Added

- `sars.save()` and `sars.load()`, and `sars.to_json()` and `sars.from_json()`, save any
  result (`SARFit`, `MultiSARFit`, `AveragedSAR`, `BootstrappedCI`, `LinPowFit`,
  `ThresholdFit`, `ThresholdCI`, `ManySARFit`) as standard JSON and load it back, with
  its data, parameters and settings, the `sars` version that wrote it and a format
  version. Non-finite numbers are kept, and loading rebuilds only `sars`'s own result
  types (#215)
- A `sars` command (also `python -m sars`), with no extra dependencies: `sars fit`,
  `average`, `threshold`, `predict` and `datasets`. It reads a CSV file, standard input
  or a bundled dataset, saves results with `--save`, and with `--json` prints a JSON
  envelope (`sars`, `command`, `ok`, `result` or `error`, `warnings`) (#217)
- The Lilliefors normality test in the residual checks: `normality="lillie"` (or
  `"lilliefors"`) in `SARFit.residual_checks`, `sar_multi`, `sar_average`,
  `bootstrap_ci` and the CLI. Its statistic and its p value below 0.1 match R's
  `nortest::lillie.test`; above 0.1 the p value comes from a simulated null (and is at
  least 0.1), within about 0.01 of the true value, where R's polynomial fit is off by up
  to 0.03 (#152)
- `sars.loo_cv()`: leave-one-island-out cross-validation. It ranks the models by the
  error of their predictions for left-out islands, beside their AICc (or AIC, BIC) rank, with the
  largest error and its island, the AICc average refitted in each fold, and every
  prediction. Each fold runs the full search (warm starts are optional, and optimistic
  for flexible models), and folds can run in parallel (`n_jobs`). The
  CLI runs it as `sars cv`, and its result saves as JSON (#222)
- The CLI's JSON tables keep whole numbers as integers (#222)
- `sars.habitat_loss()`: species expected to be lost when an area loses habitat (given as
  the area remaining or the fraction lost), from the model-averaged curve, with bootstrap
  intervals built from each resample's own loss. It warns when an area is extrapolated
  and when averaged curves fall with area. The docs explain what the estimate means and
  its known upward bias. The CLI runs it as `sars loss` (#224)
- `bootstrap_ci(keep_samples=True)` keeps every resample's prediction in
  `BootstrappedCI.samples` (#224)
- An optional HTTP server, `sars serve` (the `server` extra: FastAPI and uvicorn). Each
  endpoint (`/v1/fit`, `average`, `threshold`, `cv`, `loss`, `predict`, `datasets`)
  runs the matching `sars` command in a subprocess and returns its JSON envelope. It
  listens on localhost by default, takes an optional bearer token, and stops commands
  after a time limit. Its OpenAPI document is in the docs (#228)
- A Galápagos example for `from_geodataframe`: 28 island outlines from OpenStreetMap,
  joined by name to `load_gala()`'s plant counts, fitted over 0.02 to 4,700 km² and
  compared with the published areas (#153)

### Changed

- Asking for R sars's `normality="kolmo"` now raises an error that explains why it isn't
  offered: it compares the residuals with a standard normal, so its answer depends on the
  units of richness (#152)

### Fixed

- `sars cv --label-col` now labels the islands by that column; before, only an
  auto-detected name column was used (#228)
- `n_jobs` from a script the worker processes can't open, one read from standard input
  (`python - < script.py`) or a pipe (`python <(...)`), runs in the calling process with a
  warning, instead of failing with an obscure error from the worker pool (#225)

## [0.11.3] - 2026-11-22

### Added

- `sar_many()` fits the same models to many datasets and summarises them in one table,
  one row per dataset: islands, area range, the log-log power fit's z and R², the
  multi-model average's best model and its weight, and the threshold model and its
  breakpoint, with the fits kept in a `ManySARFit`. Datasets come as a dict of name to
  DataFrame or as one long DataFrame with `by=`. A fit that fails is named with its
  error in the table, without stopping the dataset's other fits or the other datasets;
  each dataset's warnings are listed against it; `n_jobs` fits them in parallel (#209)
- `bootstrap_ci` and `threshold_ci` take `n_jobs`, to fit the bootstrap resamples in
  several processes (`-1` for one per core), with the standard library's
  `concurrent.futures`, so no new dependency. The default, 1, runs as before, and the
  result doesn't depend on it: the resamples are drawn from `rng` in the calling
  process, in order, and the workers' warnings are raised again in it. With 8 workers,
  1000 fast resamples of the Galápagos data take 26 s instead of 149 s, 48 full ones
  46 s instead of 288. In a script the call has to be under
  `if __name__ == "__main__":`, since each worker imports the script (#207)
- The test suite fits all 34 island datasets compiled by Gooriah et al. (2021), released
  under CC0, with `sar_average`, `sar_threshold` and `lin_pow`, and a new Validation
  page in the docs reports the results: all 31 datasets with at least five islands fit
  without an error. The tables, with a script that rebuilds them from the deposit, are
  under `tests/data/gooriah2021/`; the test is marked `slow` and runs in CI's notebooks
  job (#187)

### Changed

- CI's test jobs install exactly the versions in `uv.lock` with uv, instead of resolving
  the newest releases with pip on every run, so a failure comes from the change under
  test. Two jobs run on pushes to `main`, manual runs and a new weekly schedule: one on
  the minimum versions of the runtime dependencies, which gates the release, and one on
  the newest releases, ignoring the lockfile, which only reports (#211, #212, #213)

### Fixed

- The Kendall homoscedasticity check (`cor_method="kendall"`) no longer fails with an
  AttributeError on scipy before 1.10, which `pyproject.toml` allows: those versions
  name the estimate differently. `coleman` reports a ragged abundance table as not
  rectangular on numpy before 1.24 too, as it does on later versions (#213)
- `plot_multi` raises `ValueError` for a negative `top_n`, which it used to read as
  "all but the last n" models (`top_n=-1` drew 19 of 20). `top_n=0` still draws the
  observed data alone, now documented (#184)

## [0.11.2] - 2026-11-22

### Changed

- Fitting is several times faster: `sar_multi` on the Galápagos data takes about a
  seventh of the time, and weibull4, which took three quarters of it, about a tenth.
  The search from the grid of starting values uses looser tolerances, and the tight
  final solve starts from the best three distinct grid results instead of one; starts
  that only match the best fit to within the search's own scatter no longer delay its
  early stop. On every bundled dataset the fits are unchanged, apart from two that run
  along a ridge to a bound and stop at a point on it within 0.0001 of the old AICc
  (betap on `load_xu`, epm2 on `load_schrader`) (#193)
- CI checks pull requests faster. They test Python 3.11 and 3.14, with every supported
  version tested on pushes to `main` and manual runs; the example notebooks run once,
  in a job of their own, instead of on every Python; a failure on the pre-release 3.15
  no longer fails the run; coverage runs on one Python; and three `sar_average` tests
  fit five models instead of all 20 to check the IC weighting. A threshold test now
  allows the warning it triggers on some machines (#194, #195, #196, #197)

### Security

- The development lockfile moves to Pillow 13.0.0, tornado 6.6.1 and soupsieve 4.1.3,
  which fix the vulnerabilities GitHub reported in the versions it locked. soupsieve
  stays at 2.10 for Python before 3.11.5, which no fixed release supports. None of
  this changes what pip installs for users (#202)

## [0.11.1] - 2026-11-22

### Added

- `load_gala()`: Johnson and Raven's (1973) Galápagos plants, 30 islands from 0.01 to
  4,669 km², with island names and endemic species counts, as in R `faraway`'s `gala`.
  A different compilation from `load_galap` (Preston 1962, 16 islands) with a much
  tighter species-area relationship (log-log z = 0.39, R² = 0.76), and a threshold
  near 11 km² (#185)
- `load_schrader()` and `load_xu()`: plants on 60 small islands in Raja Ampat (Schrader
  et al. 2019) and butterflies on 42 islands of the Zhoushan Archipelago (Xu et al.
  2017), from the CC0 compilation of Gooriah et al. (2021), with areas converted from
  hectares to km². Both relationships are tight: log-log z = 0.42 and R² = 0.91 on
  Schrader's 40 islands with plants, z = 0.35 and R² = 0.79 on Xu's. Twenty of
  Schrader's islets, all under 30 m², have no plants recorded; they are kept, and only
  `lin_pow` fits them, so drop them for the other models (#186)

### Fixed

- `plot_multi` titles the plot with the number of models it draws ("Top 2 SAR models"),
  not `top_n`, and gives each curve its own colour from the start of the colour map. A
  fit with fewer models than `top_n` used to be titled "Top 5 SAR models", with colours
  spread for five curves. Above ten curves it takes tab20's dark shades before its light
  ones, so neighbouring ranks differ in hue, and a model asked for twice is drawn once
  (#180)

## [0.11.0] - 2026-11-22

### Added

- The Galápagos example shows the newer features: a fit's parameter table, the observed
  shape of each fitted curve, residual checks before averaging, `sar_pred`, and a
  threshold with bootstrap and inverted F intervals and its plot, on the Aegean plants,
  where a threshold shows (#148)
- `ColemanFit.placement_test(n_sim=999, rng=None)`, a calibrated test of Coleman's random
  placement model: it simulates random placement of the observed species totals and gives
  a p value for the sites' distance from the expected curve (their squared deviations in
  standard deviations, summed), with `rule_rejection_rate`, how often the one-third rule
  rejected the simulated random placements. On communities placed at random over eight
  sites the test rejects at about its level, where the rule rejects about half (#143)
- `threshold_ci(method="F")`, intervals for ContOne's and ZslopeOne's breakpoint from an
  inverted F test (Toms & Lesperance 2003): every candidate breakpoint whose fit the F test
  at level `1 - conf` doesn't reject against the best fit. It needs no resampling.
  `ThresholdCI.method` says which method made an interval. R `sars`'s F intervals compare
  the statistic with F(conf; 1, p), p its number of regression coefficients, instead of
  F(conf; 1, n - p), so for more than a few islands they are much wider; with R's degrees
  of freedom and R's grid this package reproduces them to a grid step (#144)

### Changed

- **Behavior change:** `sar_multi` and `sar_average` warn when a model doesn't converge
  or its fit raises, naming each, and list them with the reason in the new
  `MultiSARFit.not_fitted`. They used to drop them without a word, so the only way to
  find a missing model was to compare names with the full list. `bootstrap_ci` warns
  once, never per resample: for the fit to the original data, or, with
  `method="full"`, for a model that couldn't be fitted in any resample (#179)
- **Behavior change:** sars requires Python 3.11 or later. Python 3.10 reached end of
  life in October 2026; on 3.10, pip keeps installing 0.10. The minimum versions of
  numpy, scipy, pandas and matplotlib rise to their first releases with Python 3.11
  wheels (numpy 1.23.2, scipy 1.9.2, pandas 1.5, matplotlib 3.6) (#169)
- **Behavior change:** the `interactive` extra is gone. It installed plotly, which no part
  of sars uses: every plot is matplotlib. `sars[all]` now installs the `geo` extra only
  (#170)
- **Behavior change:** `from_geodataframe` keeps the GeoDataFrame's index when the areas
  come from `area_col`, as it already did for areas computed from the geometries (and as
  `from_df` keeps its input's). It used to return a fresh `RangeIndex` on that path only,
  so the rows couldn't be joined back to the polygons by index. Its `area` column, passed
  to `coleman`, is now matched to the abundance table's columns by label, as areas
  computed from geometries already were. An unknown unit now raises with `area_col` too,
  where it was ignored (#146)
- **Behavior change:** a `SARFit` whose `model` isn't one of the package's (built or
  edited by hand) raises `ValueError` ("Unknown model: ...", listing the models) instead
  of `NotImplementedError` from `predict`, `param_table` and everything that predicts from
  it (`observed_shape`, `residual_checks`, the plots, `sar_pred`): nothing is
  unimplemented, the name is wrong. `sar_multi`'s error for an unknown name lists the
  models too, and a fit missing one of its model's parameters says which, instead of a
  bare `KeyError` (#163)
- The test suite fits the models its reference tests share once per run, not once in each
  pytest-xdist worker: every worker used to refit all 20 models on both reference datasets
  at the same time. A full run on 18 cores takes about a quarter less time (#147)
- `threshold_ci` scores DiscOne's and DiscTwo's resamples together, as it does the other
  models': 1000 resamples of DiscTwo on 100 islands take a twentieth of a second, where
  they took about 25 seconds. Their search, in `sar_threshold` too, now treats two
  candidates whose RSS differs by under 1e-10 of the total sum of squares as tied and takes
  the first, as the grid searches do, so rounding no longer decides between fits that are
  equal (#144)
- **Behavior change:** `bootstrap_ci`'s default `area_grid` is half evenly and half
  geometrically spaced across the data's areas (about 100 points), and `plot_fit`,
  `plot_multi` and `plot_average` draw their curves on the same kind of grid. An even grid
  alone put almost no points among the small islands when areas span orders of magnitude:
  5 of 100 points below 50 km² on the Galápagos data, where most of its islands are,
  against 32 now (#141)
- `make lint`, `make fmt-check` and CI lint and format-check the example notebooks in
  `docs/examples` as well as `src` and `tests`. Their code cells are reformatted to match,
  and the geodataframe example drops an unused import and pairs counties with their label
  points with `zip(..., strict=True)` (#171)
- The documentation site uses Material for MkDocs' standard theme, with light and dark
  schemes that follow the reader's system setting and a toggle in the header, in place of
  a custom dark-only stylesheet with its own fonts (#175)

### Deprecated

- `from_geodataframe`'s `crs_units` is renamed `area_units`, which says what it sets: the
  unit of the areas returned, not a property of the CRS. `crs_units` still works, with a
  `FutureWarning` (#146)

### Fixed

- negexpo, weibull3, weibull4, chapman and betap lost precision where `1 - exp(-x)` is
  computed for small `x`: weibull4's Galápagos fit predicted 7.99 species at the smallest
  island where its formula gives 7.52. They now use `expm1` (betap's `1 - (1 + u)^-f` as
  `-expm1(-f log1p(u))`). The rounding had propped up false optima on flat ridges. weibull4
  on the Galápagos data now runs along its ridge to `z`'s upper bound of 10, with a lower
  AICc than R's (193.44 against 193.57, a value set by the bound), and `c`'s standard error,
  finite now, dwarfs its estimate; on the seven-island sample it now matches R's AICc. The
  other fits move only in their last digits (#140)
- `SARFit.param_table` gives no standard error (`NaN`, with a warning naming them) for
  parameters that stopped on a bound of the fit's search, where the approximation behind
  it doesn't hold and the estimate is the bound's, not the data's: weibull4's `z` on the
  Galápagos data, and on the seven-island sample p1's `d` and heleg's `f` at 0 among
  others. Until now they were reported as if the optimum were interior (#145)
- `bootstrap_ci` no longer drops resamples without saying so: `BootstrappedCI.dropped`
  counts them by reason (no model converged, a fit raised, or the averaged prediction
  wasn't finite), and a warning reports how many. numpy's warnings from a resample's fits and
  prediction no longer leak out, since the resample is dropped and counted. A new warning
  marks the areas where the upper bound runs away, past ten times the largest observed
  richness, so the interval there reads as unbounded (#141)
- `sar_average` raises `ValueError` when no model converges, or none is given, instead of
  returning an average that predicts zero species everywhere. `AveragedSAR` refuses to be
  built without a model, or with weights that give none to the models fitted. With
  `method="full"`, `bootstrap_ci` no longer puts a resample where no model converged into
  the interval as a prediction of zero (which pulled the lower bound down): it is dropped
  and counted, as with the default method; and it raises at once when `models` is empty
  (#142)
- The notebook tests shut each notebook's kernel down and close its zmq sockets when the
  notebook finishes. They used to leave them for garbage collection, which reported them as
  `ResourceWarning`s on whichever test ran next in the same worker (#161)
- Two tests no longer drop rows with `inplace=True`, which pandas 3.1 deprecates and
  pandas 4 removes. They change a fit's DataFrame by adding a row in place with `.loc`
  instead, which still works under pandas 4 (#166)

## [0.10.0] - 2026-11-21

### Added

- `sar_multi`'s summary has an `observed_shape` column, and `SARFit` an `observed_shape()`
  method: the shape the fitted curve takes between the smallest and largest observed area
  ("convex", "sigmoid", "linear" or "none"), from how its slope changes. The `shape`
  column still gives the model's general shape, which for five models is "convex/sigmoid"
  and doesn't say which a fit took (#116)
- `threshold_ci` gives the breakpoints of a fitted threshold model bootstrap confidence
  intervals: it resamples the model's residuals onto its fitted values, reruns that model's
  breakpoint search on each resample (same grid, same scale) and takes percentiles of the
  breakpoints found (Toms & Lesperance 2003). It works for all six models with a breakpoint,
  where R `sars`'s `threshold_ci` covers ContOne and ZslopeOne, and returns a `ThresholdCI`
  with the estimates and bounds on the fitted scale and in area units, plus every resample's
  breakpoints. The resamples of a model searched over a grid are scored together, so the
  default 1000 cost about as much as a few fits of the model for ContOne and ZslopeOne and
  roughly ten for ContTwo and ZslopeTwo. `ThresholdFit` gains an `interval` attribute, the
  grid spacing its search used (#119)
- `plot_threshold` plots a threshold model on the scale it was fitted on: the data, the
  fitted segments (a discontinuous model's drawn apart at each jump), a line at each
  breakpoint and, given a `ThresholdCI`, its interval shaded (#119)
- `lin_pow` fits the log-log power model, log S = log c + z log A, by ordinary least
  squares, as R `sars`'s `lin_pow` does. It returns a `LinPowFit` with log c, c and z, their
  standard errors, t and p values, R² and adjusted R² in log space, and a `predict` on the
  arithmetic scale. `log_base` chooses natural, base-2 or base-10 logs. When an island has
  no species, `con` (default 1) is added to every richness value. `compare=True` also fits
  the nonlinear power model, so the two fits' c and z can be set side by side
  (`LinPowFit.comparison`), with a warning if that fit doesn't converge. `plot_lin_pow` draws the data and the fitted line on log-log
  axes. The coefficients, standard errors, p values and R² match R on the Galápagos data
  for each log base, with and without an empty island (#122)
- `sar_pred(fit, area)` returns a table of predicted richness (columns `model`, `area`,
  `prediction`) at one or more areas, as R `sars`'s `sar_pred` does. It takes a `SARFit`,
  a `MultiSARFit` (a row per model and area, models in summary order), an `AveragedSAR`
  (model "average"), a `ThresholdFit` (its best model) or a `LinPowFit`. Areas that are
  zero, negative or not finite raise `ValueError`, as does a fit that didn't converge.
  Predictions for power, loga and negexpo
  match R on the Galápagos data; an averaged prediction matches R's when both average the
  same models (#125)
- `coleman(abundance, area)` compares site richness with Coleman's (1981) random placement
  model, as R `sars`'s `coleman` does. It takes a species-by-site abundance table (species in
  rows, sites in columns) and the site areas, and returns a `ColemanFit`: a per-site table
  of area, relative area, observed richness, expected richness, its standard deviation and
  whether the site lies within one standard deviation; the share of sites outside it
  (`fraction_outside`); and the verdict (`rejected`, True when that share is over a third).
  Areas given as a labelled `pd.Series` are matched to the table's columns by label.
  `predict` and `predict_sd` give the expected curve at any area, and `plot_coleman` draws
  it with its band and the observed points. Expected richness and standard deviations match
  R's on its simulated matrix (#123)
- `sar_gdm` fits the general dynamic model of oceanic island biogeography (Whittaker,
  Triantis & Ladle 2008): richness as a function of island area plus a hump-shaped function
  of island age (time and time squared). `model=` chooses the form, with the four of R
  `sars`'s `gdm`: "loga" (the original ATT² model) and "linear", fitted by ordinary least
  squares, and "power_area" and "power_area_time", fitted by nonlinear least squares on
  richness; "all" fits the four and ranks them. `mod_sel=True` compares the GDM with the
  simpler area + time, area-only and intercept-only models by AIC and AICc. The returned
  `GDMFit` has the coefficients with standard errors and p values, fitted values, R², AIC,
  AICc, BIC and `predict(area, time)`. The fit doesn't depend on the units of area or age.
  Coefficients and AIC match R on the example from its help page (#124)
- Three more bundled datasets, with the values R `sars` ships: `load_aegean()` (terrestrial
  isopods on 90 Aegean islands; Sfenthourakis & Triantis 2009), `load_aegean2()` (plants on
  173 Aegean islands; Matthews et al. 2020) and `load_niering()` (plants on 32 islands of
  Kapingamarangi Atoll; Niering 1963). Areas are in km². A new Datasets page in the
  documentation lists them with their sources and a threshold example (#126)
- Residual checks before model averaging, as in R `sars`. `SARFit.residual_checks()` returns
  a `ResidualChecks` with a normality test of the residuals (`"shapiro"` or `"kolmogorov"`),
  a homoscedasticity test (the Spearman, Pearson or Kendall correlation of the squared
  residuals with the fitted values, `"cor.fitted"`, or with area, `"cor.area"`) and whether
  any fitted value at the observed areas is negative. `sar_multi`, `sar_average` and
  `bootstrap_ci` take `normality`, `homoscedasticity`, `cor_method`, `neg_check`,
  `alpha_normality` and `alpha_homoscedasticity`: models that fail a check asked for are
  left out, listed with the reason in the new `MultiSARFit.excluded` and in one warning.
  The default is no checks, so results are unchanged unless you ask.
  A model whose fitted values aren't all finite fails any check asked for. `sar_average` and
  `bootstrap_ci` raise `ValueError` if every model fails; `bootstrap_ci` applies the checks
  to the fits on the original data only, which fixes the models every resample averages.
  Statistics, p values and the models left out match R on the Galápagos data.
  `"kolmogorov"` uses a normal distribution with the residuals' mean and standard deviation
  (R's `"kolmo"` uses a standard normal, which depends on the units of richness), and R's
  Lilliefors test is not offered (#121)
- `SARFit.param_table()` gives each fitted parameter's standard error, t value, p value
  and confidence interval, from the Jacobian of the model at the fitted values
  (covariance `sigma² (JᵀJ)⁻¹`, with `n - p` degrees of freedom). The standard errors,
  t values and p values match R `sars`'s `sigConf` on the Galápagos data. The interval
  uses the t distribution at the level `conf` (default 0.95), so it is slightly wider
  for small samples than R's estimate ± 2 standard errors. Parameters the data don't
  identify, and those that depend on a derivative of the curve that can't be computed
  reliably (`sar_weibull4`'s on the Galápagos data), get `NaN` and a `UserWarning`
  instead of a meaningless number (#120)
- A second worked example in the documentation: the 26 independent sites of Stout, Jessee &
  McMeen (2025) on southern Appalachian amphibians, read from a CSV. It sets the classical
  log-log power fit beside `sar_power`, compares all twenty models, and takes the model
  average through a bootstrap interval, a prediction for an unstudied site and a threshold
  check (#127)
- A worked example of `from_geodataframe` in the documentation: outlines of ten southern
  Appalachian counties from Natural Earth, joined to published amphibian counts,
  reprojected to an equal-area coordinate system, with the computed areas compared against
  the published ones and a model fitted to them (#128)

### Fixed

- The warm-start test compared weibull4's AICc on the Galápagos data to 1e-6, tighter than
  that fit's flat ridge allows, and failed on Python 3.10; it now allows the tolerance the
  R-reference tests use for fits on a ridge (#136)
- `from_geodataframe` no longer returns meaningless areas for polygons in a geographic
  coordinate system. It used to compute them in squared degrees and divide by a million,
  so an island of 1,000 km² came back as 0.00000008. It now raises `ValueError` and says to
  reproject, for any coordinate system that isn't projected and for a GeoDataFrame with no
  coordinate system whose coordinates look like longitude and latitude; other
  GeoDataFrames with no coordinate system get a warning and are taken to be in metres. It
  also reads the length unit from a projected coordinate system, so one in feet gives the
  same areas as one in metres, and raises `ValueError`, naming the rows, for geometries
  that are missing, empty or have no area (points, lines, degenerate polygons), which used
  to count as zero area. Passing `area_col` is unaffected (#128)

## [0.9.0] - 2026-11-19

### Added

- `sar_threshold` fits the rest of R `sars`'s threshold models: DiscOne (a separate line
  either side of a breakpoint at an observed area), ContTwo (three joined segments),
  ZslopeTwo (a flat segment, then two joined sloping ones) and DiscTwo (three separate
  lines), with R's parameter counts (6, 7, 6 and 9). Choose them with `models=`; the default
  stays ContOne, ZslopeOne and Linear. Two-threshold fits report both breakpoints in
  `ThresholdFit.best_breakpoints` and `breakpoints_area`, and in new `breakpoint2` and
  `breakpoint2_area` columns of `summary`. ContTwo and ZslopeTwo match R's `sar_threshold`
  on the Galápagos data. For DiscOne and DiscTwo, R's own search strays outside its `nisl`
  window, so this package keeps three islands in the first and last segments, as `nisl = 3`
  documents; their RSS matches R's model functions over that window (#109)
- `sar_threshold` warns when AICc is undefined for some of the models it fits (a model of k
  parameters needs more than k + 1 observations), as R's `summary` does, and raises
  `ValueError` when that's true of every model, instead of naming one of them best (#109)

### Changed

- **Behavior change:** the `shape` column of `sar_multi`'s summary follows Matthews et al.
  (2019, Table 1) with R `sars`'s later corrections. p2 is now "sigmoid" (was "convex");
  epm1, p1 and asymp are "convex/sigmoid" (were "convex"), and mmf and heleg are
  "convex/sigmoid" (were "sigmoid"), for models whose parameters decide the shape. Code
  that filters on `shape` gets different models. The model reference and README give each
  model's shape (#113)

### Fixed

- Fits no longer depend on the data's units. The start grids and bounds were fixed numbers,
  many of them the Galápagos fit's own values, so in other units fits went wrong while still
  reporting convergence. With areas in m², negexpo, asymp, gompertz and chapman fitted
  nothing (R² 0) and p1 did worse than a flat line. Each model is now fitted on rescaled data
  (area over its geometric mean, richness over its maximum) from neutral start values, and
  its parameters converted back. Every model except epm1 and epm2, whose form changes with
  the area units, gives the same fit in any units (weibull4 to within its flat ridge). Fitting all 20 models on Galápagos is
  about a third faster (#111)
- `SARFit.r_squared` is NaN, not `-inf` or NaN with a warning, when every island has the
  same richness, where R² is undefined (#111)
- Docs: heleg's formula in the model reference (`c / (f + A^(-z))`, as fitted), koba
  listed with the non-asymptotic models (it has no asymptote), the full Preston (1962)
  citation for the bundled Galápagos data, and two statements in the Galápagos notebook:
  `summary`'s shape column is each model's typical shape, and the bootstrap's central
  curve (#112)

## [0.8.1] - 2026-11-10

### Fixed

- Fits that stopped short of R `sars` because of narrower parameter limits or a start grid tied
  to the area units now reach R's optimum or better: p2's `d` may be negative, the asymptote
  `d` (and mmf's `c`) is no longer capped at 1e6, and weibull3's `c` may approach 0. Parameters
  whose optimum can lie many orders of magnitude from the start values (weibull4's `c`, epm1's
  `c`, mmf's and weibull3's `d` and `c`) are searched on a log scale, so epm1 now fits areas in
  m² as well as in km². A final tight solve pins down parameters left loose along flat
  ridges. Fitting all 20 models on the R reference datasets is about 25% faster, and
  `bootstrap_ci` on Galápagos about twice as fast (#100)
- `sar_asymp`'s docstring and the model docs say how its `z` relates to R's (#100)

## [0.8.0] - 2026-11-10

### Changed

- **Behavior change:** `sar_epm1` and `sar_epm2` now fit the extended power models R `sars`
  fits under those names (Tjørve 2009): epm1 is `S = c * A^(z * A^-d)` and epm2 is
  `S = c * A^(z - d/A)`, with `z` between 0 and 1. The old epm1 was a log-quadratic power model
  (`c * A^z * exp(d * log(A)^2)`) that R doesn't have, and the old epm2 was R's epm1 under other
  parameter names. Both now match R on the Galápagos data, so `sar_multi` and `sar_average`
  results that include them change, and epm2's parameters are `c, z, d` instead of
  `c, z1, z2`. epm2's shape is listed as sigmoid, as in R (#99)
- **Behavior change:** `sar_threshold` steps its breakpoint grid by 0.01 on ln(area) by default,
  as R `sars` does, instead of 0.1. The coarser grid could miss the RSS minimum, so default
  breakpoints and AICc values can change, and with them which model is best. An interval
  giving more than 100,000 grid points now raises a `ValueError` instead of running for a
  very long time (#101)

### Fixed

- The R reference tables are reproducible: R's default fits use random starting values, so
  `generate_r_reference.R` now seeds each fit (#103)

## [0.7.20] - 2026-11-10

### Added

- Tests against R `sars` 2.1.1 for model-averaging weights and curves (on the models that match), threshold models, and
  a seven-island sample, alongside the per-model fits. Models that reach R's optimum must agree
  to 1e-4 in AICc, 1e-6 in R² and 0.5% in each parameter, replacing the old check (no more than
  0.5 AICc worse than R). Each known difference from R is listed with its issue (#99, #100,
  #101) (#98)

## [0.7.19] - 2026-11-09

### Added

- `BootstrappedCI.median`, the median prediction over resamples: the centre of the percentile
  interval. `mean` is unchanged, but a few resamples whose prediction runs away can dominate
  it, so `bootstrap_ci` now warns when the mean falls outside the central 95% of the resample
  predictions (#83)

### Changed

- The package type-checks with mypy (with pandas-stubs), and CI and `make check` run it
  (`make typecheck`). Model functions are typed as plain callables, the threshold search's
  `fit_func` has a real signature, and the plots read their data with
  `to_numpy(dtype=float)` (#94)

### Fixed

- `ThresholdFit.predict` no longer returns 0 for areas outside the fitted range: below it the
  first segment's line continues, above it the last one's, as the SAR models' `predict`
  extrapolates. A NaN area gives NaN, and on a fit on ln(area) an area of zero or less raises
  `ValueError` (#88)

## [0.7.17] - 2026-11-08

### Changed

- CI gives each test worker one math-library thread (`OMP_NUM_THREADS` and friends), so
  four workers don't run sixteen BLAS threads on four vCPUs, and lists the slowest tests (#92)

### Changed

- Tests run on every logical core (`-n logical`; `-n auto` counted 2 of CI's 4), CI traces
  coverage with sys.monitoring on Python 3.12+, caches pip downloads, and cancels a PR's
  superseded runs (#90)

## [0.7.15] - 2026-11-06

### Added

- Threshold fits report the breakpoint in area units as well: `ThresholdFit.breakpoint_area`
  and a `breakpoint_area` column in `summary` next to `breakpoint`, which stays on the fitted
  scale (the natural log of area by default). The repr shows both. `sar_threshold()` gains
  `log_area`: with `False` the piecewise models are fitted on area itself, and `interval`
  (now optional) defaults to a thousandth of the breakpoint window on that scale (#87)

- A worked example, `docs/examples/galapagos.ipynb`, rendered in the documentation under
  Examples: the Galápagos plant data through one model, all twenty, model averaging,
  bootstrap confidence intervals and threshold models, with plots. The test suite runs it
  (#81)

### Fixed

- Fitting no longer floods the console with numpy `RuntimeWarning`s (overflow, invalid
  value) from parameter combinations the multi-start search tries and rejects; `epm2`
  alone emitted hundreds (#82)

## [0.7.10] - 2026-10-27

### Added

- CI tests and PyPI classifiers cover Python 3.14 and 3.15 (#64).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), and a
  support section in the README and docs.
- JOSS paper source under `paper/`, compiled to PDF by a GitHub Actions
  workflow on every change.

### Changed

- PyPI development status classifier raised from Alpha to Beta.

### Fixed

- `sar_multi`, and therefore `sar_average` and `bootstrap_ci`, raise input
  validation errors again instead of returning an empty result; the
  per-model `ValueError` skip added in 0.7.5 had swallowed them (#66).
- A fit whose best start stopped on the solver's iteration limit is no
  longer reported as converged; only successful solver results are
  candidates for the best fit (#65).

## [0.7.9] - 2026-10-17

### Changed

- Ruff lint and format checks run in their own CI job, at the version pinned
  in `uv.lock`.

## [0.7.8] - 2026-10-17

### Fixed

- Model averaging no longer predicts zero everywhere when every information
  criterion value is non-finite (e.g. AICc on four or fewer islands).
  `akaike_weights` returns NaN weights in that case, `sar_average` raises a
  `ValueError` suggesting `ic="AIC"` or `"BIC"`, and `bootstrap_ci` reports a
  NaN interval instead of a confident interval of zero (#50).

## [0.7.7] - 2026-10-17

### Changed

- Release workflow hardened against `main` moving on while a release is in
  progress; publishing happens only from release tags.

## [0.7.6] - 2026-10-11

### Changed

- Requires Python 3.10 or newer (Python 3.9 reached end of life in October 2025).
  CI now tests 3.10 through 3.13.

## [0.7.5] - 2026-04-07

### Fixed

- `sar_multi` skips a model whose fit raises a `ValueError` instead of
  aborting the whole run. This also swallowed input validation errors; see
  #66 under Unreleased.

## [0.7.4] - 2026-04-06

### Added

- `from_df()` adapter for existing DataFrames (#14).
- `ic` argument on `sar_average()` to weight by AIC or BIC as well as
  AICc (#22).
- `**kwargs` passthrough to the plotting functions (#24).
- `sars.__version__` (#25).
- Input validation with clear error messages on every `sar_*` function (#19).
- Early termination of the multi-start grid search once the best fit has
  stabilised (#21).

### Fixed

- `load_galap()` failed for installed (non-editable) packages because the
  dataset was not bundled (#16).
- Exceptions inside the NLS loop are logged instead of silently
  swallowed (#17).
- `from_geodataframe` type annotation referenced an undefined name (#15).
- Colour cycling in `plot_multi()` repeated colours past ten models (#23).
- Heleg model parameter mapping checked against R `sars` (#26).
- Threshold models document their minimum sample size (#18) and breakpoint
  selection (#28).

## [0.7.3] - 2026-04-04

### Added

- `BootstrappedCI` reports `n_models_attempted` and the mean convergence
  rate; a warning is raised when fewer than half the models converge.

### Fixed

- The low-convergence warning was suppressed when no model converged (#10).
- `n_attempted` is computed once per bootstrap rather than per resample (#9).

## [0.7.2] - 2026-04-04

### Added

- `start_from` warm starts in the fitting loop, and `method="fast"` in
  `bootstrap_ci()` that reuses the original fits as starting values instead
  of repeating the grid search on every resample (#2).

### Fixed

- Failed fits in the fast bootstrap path are counted and reported instead of
  dropped silently (#5).

## [0.7.1] - 2026-04-04

### Added

- Getting-started guide and API reference pages on the docs site.

## [0.7.0] - 2026-04-04

### Added

- mkdocs-material documentation site with API pages built from docstrings.
- Edge-case tests covering `repr` methods, failed fits, and predict errors.

### Removed

- Dead `_data_anchors()` helper.

## [0.6.0] - 2026-04-03

### Added

- Plotting: `plot_fit`, `plot_multi`, `plot_average`, `plot_residuals`.

## [0.5.0] - 2026-04-03

### Added

- I/O adapters: `from_csv`, `from_geodataframe`, `from_pyinaturalist`, and
  `load_galap`.

## [0.4.0] - 2026-04-03

### Added

- `sar_threshold()` piecewise models with breakpoint grid search.

## [0.3.0] - 2026-04-03

### Added

- `sar_multi()`, `sar_average()`, `bootstrap_ci()`, and the information
  criteria helpers.

## [0.2.0] - 2026-04-03

### Added

- All 20 SAR models on a shared multi-start NLS engine, returning `SARFit`.

## [0.1.1] - 2026-04-03

### Added

- R reference generation script and the Galapagos reference dataset.

## [0.1.0] - 2026-04-03

### Added

- Project skeleton: package layout, CI, publishing workflow, license, and
  citation file.

[Unreleased]: https://github.com/jmcmeen/sars/compare/v0.11.3...HEAD
[0.11.3]: https://github.com/jmcmeen/sars/compare/v0.11.2...v0.11.3
[0.11.2]: https://github.com/jmcmeen/sars/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/jmcmeen/sars/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/jmcmeen/sars/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/jmcmeen/sars/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/jmcmeen/sars/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/jmcmeen/sars/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/jmcmeen/sars/compare/v0.7.20...v0.8.0
[0.7.20]: https://github.com/jmcmeen/sars/compare/v0.7.19...v0.7.20
[0.7.19]: https://github.com/jmcmeen/sars/compare/v0.7.18...v0.7.19
[0.7.17]: https://github.com/jmcmeen/sars/compare/v0.7.16...v0.7.17
[0.7.15]: https://github.com/jmcmeen/sars/compare/v0.7.14...v0.7.15
[0.7.10]: https://github.com/jmcmeen/sars/compare/v0.7.9...v0.7.10
[0.7.9]: https://github.com/jmcmeen/sars/compare/v0.7.8...v0.7.9
[0.7.8]: https://github.com/jmcmeen/sars/compare/v0.7.7...v0.7.8
[0.7.7]: https://github.com/jmcmeen/sars/compare/v0.7.6...v0.7.7
[0.7.6]: https://github.com/jmcmeen/sars/compare/v0.7.5...v0.7.6
[0.7.5]: https://github.com/jmcmeen/sars/compare/v0.7.4...v0.7.5
[0.7.4]: https://github.com/jmcmeen/sars/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/jmcmeen/sars/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/jmcmeen/sars/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/jmcmeen/sars/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/jmcmeen/sars/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jmcmeen/sars/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jmcmeen/sars/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jmcmeen/sars/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jmcmeen/sars/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jmcmeen/sars/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/jmcmeen/sars/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/jmcmeen/sars/releases/tag/v0.1.0
