---
title: 'sars: A Python Package for Species-Area Relationship Curve Fitting and Multi-Model Inference'
tags:
  - Python
  - ecology
  - macroecology
  - biodiversity
  - species-area relationship
  - biogeography
  - conservation biology
authors:
  - name: John McMeen
    orcid: 0009-0003-8141-567X
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 26 October 2026
bibliography: paper.bib
---

# Summary

The species-area relationship (SAR), the near-universal empirical pattern
that larger areas harbour more species, is among the most replicated findings
in ecology and a cornerstone of conservation biology [@Arrhenius1921;
@Rosenzweig1995]. Fitting
an SAR to observed area-richness data yields parameters used to forecast
extinction from habitat loss, delineate protected areas, and compare
biodiversity across systems and scales [@Guilhaumon2008; @Triantis2012].
However, rigorous SAR analysis is complicated by the existence of more than
twenty candidate models with substantively different ecological assumptions,
and no single model performs best across all taxa, spatial scales, or island
types [@Tjorve2003; @Tjorve2009; @Dengler2009; @Matthews2016]. The current
methodological standard is therefore multi-model inference: fitting all
plausible models, comparing them with information criteria, and constructing a
model-averaged prediction that propagates uncertainty across the candidate set
[@Burnham2002; @Guilhaumon2008].

`sars` is an open-source Python package providing a production-ready
implementation of this workflow. All twenty SAR models catalogued by
@Triantis2012 are fitted using bounded nonlinear least squares from a
grid of starting values, ranked by AICc with Akaike weights, and
combined into a model-averaged SAR curve with bootstrapped confidence
intervals. A piecewise threshold module implements the small-island effect
detection framework of @Matthews2021. Data can be supplied as `pandas`
DataFrames, `geopandas` GeoDataFrames, or `pyinaturalist` observation
lists, enabling `sars` to participate natively in Python-based biodiversity
data pipelines without format conversion.

# Statement of Need

No Python package currently provides the full multi-model SAR workflow. This
gap imposes a concrete cost on researchers: data acquisition and pre-processing
in Python, through APIs for GBIF, iNaturalist, or NEON, must be followed
by a context switch to R for SAR analysis, then a return to Python for
downstream modelling or visualisation. Every crossing of this boundary
introduces friction, version-management overhead, and risk of silent
discrepancy between environments.

The cost is growing as the biodiversity informatics community consolidates
around Python. Citizen-science occurrence data are accessed primarily through
Python APIs; species distribution models are built with scikit-learn and
PyTorch; remote-sensing pipelines run in Google Earth Engine and rioxarray. A
researcher working at the intersection of occurrence data and SAR theory
has, until now, had no native Python tooling for the statistical inference
step. `sars` is designed to close that gap completely, enabling end-to-end
SAR analysis, from raw occurrence records to a published multi-model curve
with confidence intervals, within a single Python environment.

The target audience is ecologists, conservation biologists, and biodiversity
data scientists who are already committed to Python workflows and need
publication-quality SAR analysis without adopting a second language for one
step in their pipeline.

# State of the Field

The R `sars` package [@Matthews2019] is the most complete existing
implementation of multi-model SAR inference. It fits twenty models,
computes AICc weights, generates bootstrapped confidence intervals, and
provides piecewise threshold analysis. No Python equivalent exists. The
`macroeco` package [@Kitzes2016] provides SAR functionality in Python but
within a maximum entropy framework limited to the power law and endemics-area
relationship; it does not implement multi-model inference or the full model
catalogue of @Triantis2012. The `scipy.optimize` module can fit individual
SAR functions but provides no information-theoretic infrastructure, no
SAR-specific starting-value strategy, and no awareness of the convergence
difficulties that arise with four-parameter sigmoid models across wide area
ranges. The `vegan` R package [@Oksanen2022] provides species accumulation
curve tools but is focused on community ecology and does not implement island
SAR multi-model inference.

`sars` mirrors the R package in scope and function naming (`sar_power()`,
`sar_loga()`, `sar_average()`) so that researchers familiar with the R
ecosystem can transfer their methodological knowledge directly. The
implementation is fully independent, built from first principles on
`scipy`, `numpy`, `pandas`, and `matplotlib`, and validated against published
R output on the Galapagos benchmark dataset [@Preston1962].

# Software Design

**Universal return type.** Every model-fitting function returns a `SARFit`
dataclass carrying fitted parameters, R², AIC, AICc, BIC, convergence
status, and the original data. This uniformity is the architectural decision
that makes the package composable: `sar_multi()` accepts any heterogeneous
list of `SARFit` objects, and `sar_average()` accepts the output of
`sar_multi()` without transformation. Users can assemble custom model subsets,
pass them through the inference pipeline, and obtain a weighted average curve
without adapters or format negotiations between functions.

**Multi-start optimisation with early termination.** Nonlinear least
squares is sensitive to starting values, and a single initialisation fails
reliably on the more complex sigmoid models when area spans several orders of
magnitude. Each model in `sars` therefore carries a grid of starting values
per parameter, chosen to span plausible ranges for that model, and the fit is
attempted from every point in the Cartesian product of the grid. The
trust-region reflective solver (`scipy.optimize.least_squares` with
`method='trf'`) is used throughout for its robustness to parameter bounds.
Because the larger grids contain hundreds of points, the search stops once
fifty consecutive starts per model parameter have failed to improve the best
residual sum of squares, which keeps three- and four-parameter fits tractable
without abandoning the global search. A fitted parameter set can also be
passed back as a single warm start, which is how the bootstrap avoids
repeating the full grid search on every resample. This strategy follows the
grid-start approach of @Matthews2019. On the Galapagos benchmark dataset every
model converges and matches or exceeds the goodness of fit of the R `sars`
reference output (R² within 0.005 and AICc within 0.5), and the power-law
parameters agree with R to two decimal places; these checks run in the test
suite on every commit.

**Information criteria matched to R.** AIC, AICc, and BIC are computed using
the normal log-likelihood convention consistent with R's `nls()` and the R
`sars` package: $\ell = -n/2 \cdot \log(2\pi \hat{\sigma}^2) - n/2$, where
$\hat{\sigma}^2 = \text{RSS}/n$. The parameter count $k$ includes the
residual variance as an estimated parameter, matching R's treatment. AICc
is the default criterion because SAR datasets routinely contain fewer than
thirty observations (the standard Galapagos benchmark has sixteen islands),
making the small-sample correction material rather than cosmetic [@Burnham2002].

**Graceful convergence handling.** When the NLS solver exhausts its starting
grid without converging, `sars` returns a `SARFit` with `converged=False`
rather than raising an exception. The multi-model functions skip non-converged
fits automatically and report which models were excluded, preserving the
composability of the pipeline under partial failures.

**Layered optional dependencies.** The base install requires only `numpy`,
`scipy`, `pandas`, and `matplotlib`. Geospatial I/O via `geopandas` and
`shapely` is an optional `geo` extra; interactive visualisation via `plotly`
is a separate `interactive` extra. This design avoids imposing the geospatial
dependency chain, which can be substantial on some platforms, on users who
work with tabular area-species data only.

# Research Impact Statement

`sars` is in use in published and in-progress research. Two public
repositories, `appalachian-sar` and `etn-sar`, reanalyse the datasets of
@Stout2025 and @Jessee2022 with `sars`, replacing the log-linear ordinary
least squares fits reported in those papers with nonlinear least squares,
multi-model comparison across all twenty models, and threshold analysis. A
manuscript reporting that reanalysis for Southern Appalachian amphibians and
reptiles is in preparation. `sars` is also the fitting engine for
`genetic-sar`, a package in development that constructs nested SARs from
biodiversity occurrence records and applies a genetic algorithm to optimise
the nesting sequence; its associated papers will cite this software directly.
The Zenodo archive of `sars` provides a citable DOI for these dependencies.

Beyond the author's own programme, `sars` addresses a demand that is well
established in the ecological literature. The R `sars` package has accumulated
more than one hundred citations since its 2019 publication [@Matthews2019],
and the earlier `mmSAR` R package [@Guilhaumon2010] demonstrated sustained
community appetite for multi-model SAR software in the decade before it. As Python
continues to consolidate its position as the primary language for biodiversity
data science, demand for Python-native SAR analysis will grow in step with
the community using Python-first occurrence data pipelines.

# AI Usage Disclosure

Claude (Anthropic) was used during the development of this project, through
the Claude Code command-line tool: claude-opus-4-6 for the initial
development in April 2026 and claude-fable-5-1 for revisions in October 2026.
Claude assisted with: planning the package architecture and
public API design; drafting initial implementations of model-fitting functions
subsequently validated by the human author against R `sars` output; generating
the R reference-generation script and pytest scaffolding used for validation;
drafting this paper; and producing supporting materials including the
`CITATION.cff`, contributing guide, and changelog. All
AI-assisted outputs were reviewed, tested, and substantially edited by the
human author, who made all core design decisions, including the grid-based
multi-start strategy, the matched log-likelihood convention, and the layered
dependency structure, and who is fully responsible for the accuracy,
correctness, and originality of the submitted materials. No AI tools were
used in any direct communication with editors or reviewers.

# Acknowledgements

The author acknowledges the foundational contribution of Thomas J. Matthews,
Kostas A. Triantis, Robert J. Whittaker, and François Guilhaumon, whose R
`sars` package [@Matthews2019] established the multi-model SAR inference
framework that this package brings to the Python ecosystem. The Python
implementation is independent and does not derive from the R source code.
Validation uses the Galapagos plant dataset compiled by @Preston1962 and
distributed with the R `sars` package. This work received no external
funding. The author declares no competing interests.

# References
