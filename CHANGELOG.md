# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions up to 0.7.5 predate this file; their entries below were
reconstructed from the commit history and the
[GitHub releases](https://github.com/jmcmeen/sars/releases) page.

## [Unreleased]

## [0.7.10] - 2026-10-27

### Added

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

[Unreleased]: https://github.com/jmcmeen/sars/compare/v0.7.10...HEAD
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
