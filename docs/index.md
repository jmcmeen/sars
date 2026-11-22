# sars

**Species-area relationship curve fitting in Python.**

[![PyPI](https://img.shields.io/pypi/v/sars)](https://pypi.org/project/sars/)
[![DOI](https://zenodo.org/badge/1200249864.svg)](https://doi.org/10.5281/zenodo.19410393)
[![CI](https://github.com/jmcmeen/sars/actions/workflows/ci.yml/badge.svg)](https://github.com/jmcmeen/sars/actions/workflows/ci.yml)

A conceptual mirror of the R [`sars`](https://cran.r-project.org/package=sars) package
(Matthews et al. 2019), native to the Python scientific stack.

## Features

- **20 SAR models** -- power, logarithmic, asymptotic, sigmoid, and more
- **Log-log power model** -- the classic straight-line fit of log S on log A, with standard errors and p values
- **Multi-model inference** -- fit all models at once, ranked by AICc with Akaike weights
- **Habitat loss** -- species expected to be lost with habitat, with model-averaged bootstrap intervals
- **Cross-validation** -- rank models by how well they predict a left-out island, beside AICc
- **Save and reload results** -- any fit or result as standard JSON, data and settings included
- **[HTTP server](server.md)** -- the command line as a small JSON API (`sars serve`, optional extra)
- **[Command line](cli.md)** -- `sars fit|average|threshold|cv|loss|predict|datasets`, with JSON output for scripts
- **Many datasets at once** -- the same models fitted to many datasets, summarised in one table
- **Parameter uncertainty** -- standard errors, t tests and confidence intervals for each fitted parameter
- **Model averaging** -- weighted-average predictions across candidate models
- **Prediction** -- a table of predicted richness at any areas, from any fit
- **Residual checks** -- normality, homoscedasticity and negative-prediction checks on each fit, optionally leaving out the models that fail before averaging
- **Bootstrap confidence intervals** -- percentile-based CIs for averaged predictions
- **Threshold models** -- piecewise regression for detecting breakpoints (small island effect), with confidence intervals for the breakpoints by bootstrap or by inverted F test
- **Random placement** -- Coleman's null model for richness from a species-by-site abundance table, with a calibrated simulation test
- **I/O adapters** -- load from CSV, GeoDataFrame, or pyiNaturalist summaries
- **General dynamic model** -- richness as a function of island area and a hump in island age (Whittaker et al. 2008), with nested-model comparison
- **Bundled datasets** -- seven published island datasets (two from the Galápagos, two from the Aegean, Kapingamarangi Atoll, Raja Ampat and the Zhoushan Archipelago); see [Datasets](datasets.md)
- **R-validated** -- fits, AICc weights, model averaging, threshold models, random placement and the general dynamic model are checked against R `sars` reference output, with every known difference listed in `tests/test_r_reference.py`

## Quick start

```python
import sars

# Load the built-in Galapagos dataset (Preston 1962)
galap = sars.load_galap()

# Fit a single model
fit = sars.sar_power(galap)
print(fit)
# SARFit(model='power', c=33.1803  z=0.2832, R²=0.4912, AICc=189.03)

# Fit all 20 models and compare
multi = sars.sar_multi(galap)
print(multi.summary[["model", "AICc", "delta_AICc", "weight"]].head())

# Model-averaged predictions
avg = sars.sar_average(galap)
predictions = avg.predict([1.0, 10.0, 100.0])

# Bootstrap confidence intervals
ci = sars.bootstrap_ci(galap, n_boot=100)
```

## Worked examples

[The Galápagos plant species–area relationship](examples/galapagos.ipynb) takes the
bundled dataset through the whole workflow: one model, all twenty, model averaging,
bootstrap confidence intervals and threshold models, with the plots for each step.

[Southern Appalachian amphibians](examples/appalachian.ipynb) does the same for
published counts from 26 independent sites, read from a CSV. It compares the classical
log–log power fit with the non-linear one, predicts richness for an unstudied site and
shows what a weak fit and a wide bootstrap interval look like.

[Areas from polygons](examples/geodataframe.ipynb) reads county outlines with geopandas,
reprojects them to an equal-area coordinate system, and builds the dataset with
`from_geodataframe`, checking the computed areas against published ones.

[Galápagos islands from polygons](examples/galapagos_polygons.ipynb) joins 28 island
outlines from OpenStreetMap to Johnson and Raven's plant counts by name, and fits a curve
to areas computed from the polygons over 0.02 to 4,700 km².

## Support and contributing

Bug reports, questions, and feature requests go in the
[issue tracker](https://github.com/jmcmeen/sars/issues). The project has a
single maintainer; expect a reply within a week. See
[CONTRIBUTING.md](https://github.com/jmcmeen/sars/blob/main/CONTRIBUTING.md) for the development setup, how to add a
model, and the release process, and [CODE_OF_CONDUCT.md](https://github.com/jmcmeen/sars/blob/main/CODE_OF_CONDUCT.md)
for community expectations.

## Citation

If you use this software, please cite it:

> McMeen, J. (2026). *sars: Species-area relationship curve fitting in Python*.
> [![DOI](https://zenodo.org/badge/1200249864.svg)](https://doi.org/10.5281/zenodo.19410393)
