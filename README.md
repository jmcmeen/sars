# sars

Species-area relationship curve fitting in Python. A conceptual mirror of the R [`sars`](https://cran.r-project.org/package=sars) package (Matthews et al. 2019), native to the Python scientific stack.

[![PyPI](https://img.shields.io/pypi/v/sars)](https://pypi.org/project/sars/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/sars/)
[![CI](https://github.com/jmcmeen/sars/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jmcmeen/sars/actions/workflows/ci.yml)
[![Docs](https://github.com/jmcmeen/sars/actions/workflows/docs.yml/badge.svg?branch=main)](https://jmcmeen.github.io/sars/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1200249864.svg)](https://doi.org/10.5281/zenodo.19410393)

## Installation

```bash
pip install sars
```

sars needs Python 3.11 or later. Optional extras add:

```bash
pip install "sars[geo]"     # site areas from polygons (from_geodataframe): geopandas, shapely
pip install "sars[server]"  # sars serve, the HTTP API: FastAPI, uvicorn
pip install "sars[gpu]"     # start grids screened on an NVIDIA GPU: CuPy (CUDA 13)
```

## Features

**Fitting**

- **20 SAR models**: power, logarithmic, asymptotic, sigmoid and more, each from a grid
  of starting values, so a fit doesn't hang on one lucky guess
- **Log-log power model**: the classic straight-line fit of log S on log A, with
  standard errors and p values
- **Threshold models**: piecewise regression for breakpoints (the small-island effect),
  with breakpoint intervals by bootstrap or inverted F test
- **Fast**: each model's start grid is fitted as one batch of array operations, and
  optionally on an NVIDIA GPU (`sars.set_backend("cupy")`); see the
  [benchmarks](benchmarks/) against R `sars`

**Comparing models and their uncertainty**

- **Multi-model inference**: all models ranked by AICc, with Akaike weights
- **Model averaging**: weighted-average predictions across the candidate models
- **Cross-validation**: models ranked by how well they predict a left-out island
- **Residual checks**: normality, homoscedasticity and negative predictions, optionally
  leaving out the models that fail before averaging
- **Parameter uncertainty**: standard errors, t tests and confidence intervals
- **Bootstrap intervals** on averaged predictions, by a fast or a full method

**Applications and other models**

- **Habitat loss**: species expected to be lost with habitat, with model-averaged
  bootstrap intervals
- **Sampling effort**: richness standardized to equal sample coverage or size (as iNEXT)
  before fitting
- **Many datasets at once**: the same models on many datasets in one table, and a
  mixed-effects log-log model pooling z across archipelagos (matches lme4)
- **Habitat models**: the choros, jigsaw and Kallimanis models of area and habitat
  diversity, and the countryside model for multi-habitat landscapes
- **Random placement**: Coleman's null model, with a calibrated simulation test
- **General dynamic model**: richness from island area and age (Whittaker et al. 2008)

**Using it**

- **Command line**: `sars fit`, `average`, `ci`, `threshold`, `cv`, `loss`, `loglog`,
  `many`, `predict` and `datasets`, with JSON output for scripts; see the
  [example scripts](examples/cli/)
- **HTTP server**: the same operations as a small JSON API (`sars serve`)
- **Save and reload**: any fit or result as standard JSON, data and settings included
- **Bundled datasets**: seven published island datasets (`sars datasets`)
- **R-validated**: fits, weights, averages, thresholds, random placement and the general
  dynamic model are checked against R `sars` output, every known difference listed in
  `tests/test_r_reference.py`

New to species-area relationships? The documentation's
[primer](https://jmcmeen.github.io/sars/primer/) explains the ideas, and the tools above,
in plain terms.

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

The same from the command line:

```console
$ sars fit galap -m all          # all 20 models, ranked by AICc
$ sars average galap             # the weighted average
$ sars ci galap --n-boot 100     # bootstrap intervals on the averaged curve
$ sars --help                    # every command
```

For a complete analysis of these data, see the
[worked example](https://jmcmeen.github.io/sars/examples/galapagos/) in the documentation.
A [second example](https://jmcmeen.github.io/sars/examples/appalachian/) takes published
amphibian counts from 26 southern Appalachian sites, read from a CSV, through the same
steps, and compares the classical log-log power fit with the non-linear one. A
[third](https://jmcmeen.github.io/sars/examples/geodataframe/) computes site areas from
polygons with `from_geodataframe`, and a
[fourth](https://jmcmeen.github.io/sars/examples/galapagos_polygons/) does so for 28
Galápagos islands from OpenStreetMap, over 0.02 to 4,700 km².

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
- Kobayashi S (1974). The species-area relation. I. A model for discrete sampling. *Researches on Population Ecology*, 15, 223-237. (`koba`)
- Mielke PW, Johnson ES (1974). Some generalized beta distributions of the second kind having desirable application features in hydrology and meteorology. *Water Resources Research*, 10, 223-226. (`betap`)
- Monod J (1950). La technique de culture continue: theorie et applications. *Annales de l'Institut Pasteur*, 79, 390-410. (`monod`)
- Morgan PH, Mercer LP, Flodin NW (1975). General model for nutritional responses of higher organisms. *Proceedings of the National Academy of Sciences*, 72, 4327-4331. (`mmf`)
- Ratkowsky DA (1990). *Handbook of Nonlinear Regression Models*. Marcel Dekker, New York. (`asymp`, `ratio`)
- Richards FJ (1959). A flexible growth function for empirical use. *Journal of Experimental Botany*, 10, 290-301. (`chapman`)
- Rosenzweig ML (1995). *Species Diversity in Space and Time*. Cambridge University Press. (`powerR`)
- Ulrich W, Buszko J (2003). Self-similarity and the species-area relation of Polish butterflies. *Basic and Applied Ecology*, 4, 263-270. (`p1`, `p2`)
- Weibull W (1951). A statistical distribution function of wide applicability. *Journal of Applied Mechanics*, 18, 293-297. (`weibull3`, `weibull4`)

### Data

- Preston FW (1962). The canonical distribution of commonness and rarity. *Ecology*, 43, 185-215, 410-432. (the bundled Galápagos plant dataset, `load_galap`; areas in km²)
- Johnson MP, Raven PH (1973). Species number and endemism: the Galápagos Archipelago revisited. *Science*, 179, 893-895. (plants and endemic plants on 30 Galápagos islands, `load_gala`; areas in km²)
- Sfenthourakis S, Triantis KA (2009). Habitat diversity, ecological requirements of species and the Small Island Effect. *Diversity and Distributions*, 15, 131-140. (terrestrial isopods on 90 Aegean islands, `load_aegean`; areas in km²)
- Matthews TJ, Rigal F, Kougioumoutzis K, Trigas P, Triantis KA (2020). Unravelling the small-island effect through phylogenetic community ecology. *Journal of Biogeography*, 47, 2341-2352. (plants on 173 Aegean islands, `load_aegean2`; areas in km²)
- Niering WA (1963). Terrestrial ecology of Kapingamarangi Atoll, Caroline Islands. *Ecological Monographs*, 33, 131-160. (plants on 32 islands of Kapingamarangi Atoll, `load_niering`; areas in km²)
- Schrader J, Moeljono S, Keppel G, Kreft H (2019). Plants on small islands revisited: the effects of spatial scale and habitat quality on the species-area relationship. *Ecography*, 42, 1405-1414. (plants on 60 small islands in Raja Ampat, `load_schrader`; areas in km²)
- Xu A, Han X, Zhang X, Millien V, Wang Y (2017). Nestedness of butterfly assemblages in the Zhoushan Archipelago, China: area effects, life-history traits and conservation implications. *Biodiversity and Conservation*, 26, 1375-1392. (butterflies on 42 islands, `load_xu`; areas in km²)
- Gooriah L, Blowes SA, Sagouis A, Schrader J, Karger DN, Kreft H, Chase JM (2021). Synthesis reveals that island species-area relationships emerge from processes beyond passive sampling. *Global Ecology and Biogeography*, 30, 2119-2131. (the CC0 compilation `load_schrader` and `load_xu` come from)

### Threshold models

- Lerman PM (1980). Fitting segmented regression models by grid search. *Journal of the Royal Statistical Society C*, 29, 77-84. (`threshold_ci`, `method="F"`)
- Matthews TJ, Rigal F (2021). Thresholds and the species-area relationship. *Frontiers of Biogeography*, 13, e49404.
- Toms JD, Lesperance ML (2003). Piecewise regression: a tool for identifying ecological thresholds. *Ecology*, 84, 2034-2041. (`threshold_ci`)

### Random placement

- Coleman BD (1981). On random placement and species-area relations. *Mathematical Biosciences*, 54, 191-215. (`coleman`)
- Coleman BD, Mares MA, Willig MR, Hsieh Y-H (1982). Randomness, area, and species richness. *Ecology*, 63, 1121-1133. (`coleman`)

### General dynamic model

- Whittaker RJ, Triantis KA, Ladle RJ (2008). A general dynamic theory of oceanic island biogeography. *Journal of Biogeography*, 35, 977-994. (`sar_gdm`)
