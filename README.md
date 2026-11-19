# sars

Species-area relationship curve fitting in Python. A conceptual mirror of the R [`sars`](https://cran.r-project.org/package=sars) package (Matthews et al. 2019), native to the Python scientific stack.

[![PyPI](https://img.shields.io/pypi/v/sars)](https://pypi.org/project/sars/)
[![Python](https://img.shields.io/pypi/pyversions/sars)](https://pypi.org/project/sars/)
[![CI](https://github.com/jmcmeen/sars/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jmcmeen/sars/actions/workflows/ci.yml)
[![Docs](https://github.com/jmcmeen/sars/actions/workflows/docs.yml/badge.svg?branch=main)](https://jmcmeen.github.io/sars/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1200249864.svg)](https://doi.org/10.5281/zenodo.19410393)

## Installation

```bash
pip install sars
```

## Features

- **20 SAR models** — power, logarithmic, asymptotic, sigmoid, and more
- **Log-log power model** — the classic straight-line fit of log S on log A, with standard errors and p values
- **Multi-model inference** — fit all models at once, ranked by AICc with Akaike weights
- **Model averaging** — weighted-average predictions across candidate models
- **Prediction** — a table of predicted richness at any areas, from any fit
- **Bootstrap confidence intervals** — percentile-based CIs for averaged predictions
- **Threshold models** — piecewise regression for detecting breakpoints (small island effect), with bootstrap confidence intervals for the breakpoints
- **R-validated** — fits, AICc weights, model averaging and threshold models are checked against R `sars` reference output, with every known difference listed in `tests/test_r_reference.py`

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

For a complete analysis of these data, see the
[worked example](https://jmcmeen.github.io/sars/examples/galapagos/) in the documentation.

## Available models

| Type | Models |
| ------ | -------- |
| Linear | `linear` |
| Non-asymptotic, convex | `power`, `powerR`, `loga`, `koba` |
| Non-asymptotic, convex or sigmoid | `epm1`, `p1` |
| Non-asymptotic, sigmoid | `epm2`, `p2` |
| Asymptotic, convex | `monod`, `negexpo`, `ratio` |
| Asymptotic, convex or sigmoid | `asymp`, `mmf`, `heleg` |
| Asymptotic, sigmoid | `gompertz`, `weibull3`, `weibull4`, `chapman`, `betap` |

Each model has a dedicated function (e.g. `sars.sar_power()`, `sars.sar_negexpo()`) and returns a `SARFit` object with parameters, R², AIC, AICc, and BIC.

## Support and contributing

Bug reports, questions, and feature requests go in the
[issue tracker](https://github.com/jmcmeen/sars/issues). The project has a
single maintainer; expect a reply within a week. See
[CONTRIBUTING.md](https://github.com/jmcmeen/sars/blob/main/CONTRIBUTING.md) for the development setup, how to add a
model, and the release process, and [CODE_OF_CONDUCT.md](https://github.com/jmcmeen/sars/blob/main/CODE_OF_CONDUCT.md)
for community expectations.

## Citation

If you use this software, please cite it:

> McMeen, J. (2026). *sars: Species-area relationship curve fitting in Python*. [![DOI](https://zenodo.org/badge/1200249864.svg)](https://doi.org/10.5281/zenodo.19410393)

## References

This package is a Python implementation inspired by the R [`sars`](https://github.com/txm676/sars) package. The 20 SAR models and multi-model inference framework draw on the following works:

### R sars package

- Matthews TJ, Triantis KA, Whittaker RJ, Guilhaumon F (2019). sars: an R package for fitting, evaluating and comparing species-area relationship models. *Ecography*, 42, 1446-1455.
- Matthews TJ, Guilhaumon F, Triantis KA, Borregaard MK, Whittaker RJ (2016). On the form of species-area relationships in habitat islands and true islands. *Global Ecology and Biogeography*, 25, 847-858.

### SAR model reviews and synthesis

- Triantis KA, Guilhaumon F, Whittaker RJ (2012). The island species-area relationship: biology and statistics. *Journal of Biogeography*, 39, 215-231.
- Guilhaumon F, Mouillot D, Gimenez O (2010). mmSAR: an R-package for multimodel species-area relationship inference. *Ecography*, 33, 420-424.
- Tjorve E (2003). Shapes and functions of species-area curves: a review of possible models. *Journal of Biogeography*, 30, 827-835.
- Tjorve E (2009). Shapes and functions of species-area curves (II): a review of new models and parameterizations. *Journal of Biogeography*, 36, 1435-1445.

### Original model formulations

- Arrhenius O (1921). Species and area. *Journal of Ecology*, 9, 95-99. (`power`)
- Gleason HA (1922). On the relation between species and area. *Ecology*, 3, 158-162. (`loga`)
- Gompertz B (1825). On the nature of the function expressive of the law of human mortality. *Philosophical Transactions of the Royal Society of London*, 115, 513-583. (`gompertz`)
- Kobayashi S (1975). The species-area relation. I. A model for discrete sampling. *Researches on Population Ecology*, 17, 87-96. (`koba`)
- Monod J (1950). La technique de culture continue: theorie et applications. *Annales de l'Institut Pasteur*, 79, 390-410. (`monod`)
- Morgan PH, Mercer LP, Flodin NW (1975). General model for nutritional responses of higher organisms. *Proceedings of the National Academy of Sciences*, 72, 4327-4331. (`mmf`)
- Ratkowsky DA (1990). *Handbook of Nonlinear Regression Models*. Marcel Dekker, New York. (`asymp`, `ratio`)
- Richards FJ (1959). A flexible growth function for empirical use. *Journal of Experimental Botany*, 10, 290-301. (`chapman`)
- Rosenzweig ML (1995). *Species Diversity in Space and Time*. Cambridge University Press. (`powerR`)
- Ulrich W, Buszko J (2003). Self-similarity and the species-area relation of Polish butterflies. *Basic and Applied Ecology*, 4, 263-270. (`p1`, `p2`)
- Weibull W (1951). A statistical distribution function of wide applicability. *Journal of Applied Mechanics*, 18, 293-297. (`weibull3`, `weibull4`)
- Minami M, Lennert-Cody CE, Gao W, Roman-Verdesoto M (2007). Modeling shark bycatch: the zero-inflated negative binomial regression model with smoothing. *Fisheries Research*, 84, 210-221. (`betap`)

### Data

- Preston FW (1962). The canonical distribution of commonness and rarity. *Ecology*, 43, 185-215, 410-432. (the bundled Galápagos dataset, `load_galap`)

### Threshold models

- Matthews TJ, Rigal F (2021). Thresholds and the species-area relationship. *Frontiers of Biogeography*, 13, e49404.
- Toms JD, Lesperance ML (2003). Piecewise regression: a tool for identifying ecological thresholds. *Ecology*, 84, 2034-2041. (`threshold_ci`)
