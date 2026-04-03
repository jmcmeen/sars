# sars

[![DOI](https://zenodo.org/badge/1200249864.svg)](https://doi.org/10.5281/zenodo.19410393)

Species-area relationship curve fitting in Python.

A conceptual mirror of the R [`sars`](https://cran.r-project.org/package=sars) package (Matthews et al. 2019), native to the Python scientific stack.

## Installation

```bash
pip install sars
```

## Features

- **20 SAR models** — power, logarithmic, asymptotic, sigmoid, and more
- **Multi-model inference** — fit all models at once, ranked by AICc with Akaike weights
- **Model averaging** — weighted-average predictions across candidate models
- **Bootstrap confidence intervals** — percentile-based CIs for averaged predictions
- **R-validated** — all models tested against R `sars` package reference values

## Quick start

```python
import sars

# Load the built-in Galapagos dataset (Preston 1962)
galap = sars.load_galap()

# Fit a single model
fit = sars.sar_power(galap)
print(fit)
# SARFit(model='power', c=33.1792  z=0.2832, R²=0.4912, AICc=189.03)

# Fit all 20 models and compare
multi = sars.sar_multi(galap)
print(multi.summary[["model", "AICc", "delta_AICc", "weight"]].head())

# Model-averaged predictions
avg = sars.sar_average(galap)
predictions = avg.predict([1.0, 10.0, 100.0])

# Bootstrap confidence intervals
ci = sars.bootstrap_ci(galap, n_boot=100)
```

## Available models

| Type | Models |
| ------ | -------- |
| Non-asymptotic | `power`, `powerR`, `loga`, `linear`, `epm1`, `epm2`, `p1`, `p2` |
| Asymptotic convex | `koba`, `monod`, `negexpo`, `asymp`, `ratio` |
| Asymptotic sigmoid | `mmf`, `gompertz`, `weibull3`, `weibull4`, `chapman`, `betap`, `heleg` |

Each model has a dedicated function (e.g. `sars.sar_power()`, `sars.sar_negexpo()`) and returns a `SARFit` object with parameters, R², AIC, AICc, and BIC.

## Citation

If you use this software, please cite it:

> McMeen, J. (2026). *sars: Species-area relationship curve fitting in Python*. [![DOI](https://zenodo.org/badge/1200249864.svg)](https://doi.org/10.5281/zenodo.19410393)
