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

### Threshold models

- Matthews TJ, Rigal F (2021). Thresholds and the species-area relationship. *Frontiers of Biogeography*, 13, e49404.
