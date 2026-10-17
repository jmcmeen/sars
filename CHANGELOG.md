# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions up to 0.7.5 predate this file; their notes are on the
[GitHub releases](https://github.com/jmcmeen/sars/releases) page.

## [Unreleased]

## [0.7.8] - 2026-10-17

### Fixed

- Model averaging no longer predicts zero everywhere when every information
  criterion value is non-finite (e.g. AICc on four or fewer islands).
  `akaike_weights` returns NaN weights in that case, `sar_average` raises a
  `ValueError` suggesting `ic="AIC"` or `"BIC"`, and `bootstrap_ci` reports a
  NaN interval instead of a confident interval of zero (#50).

## [0.7.6] - 2026-10-11

### Changed

- Requires Python 3.10 or newer (Python 3.9 reached end of life in October 2025).
  CI now tests 3.10 through 3.13.

## [0.7.5] - 2026-04-07

## [0.7.4] - 2026-04-06

## [0.7.3] - 2026-04-04

## [0.7.2] - 2026-04-04

## [0.7.1] - 2026-04-04

## [0.7.0] - 2026-04-04

## [0.6.0] - 2026-04-03

## [0.5.0] - 2026-04-03

## [0.4.0] - 2026-04-03

## [0.3.0] - 2026-04-03

## [0.2.0] - 2026-04-03

## [0.1.1] - 2026-04-03

## [0.1.0] - 2026-04-03

[Unreleased]: https://github.com/jmcmeen/sars/compare/v0.7.8...HEAD
[0.7.8]: https://github.com/jmcmeen/sars/compare/v0.7.7...v0.7.8
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
