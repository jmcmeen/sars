# Validation across 34 datasets

The reference tests check `sars` against R `sars` on the bundled datasets. To check
the package on a wider range of data, the test suite also fits all 34 island and
habitat-island datasets compiled by Gooriah et al. (2021), released under CC0
(Zenodo record 7311286): plants, birds, beetles, other invertebrates, herpetofauna
and mammals, on true islands, lake islands, forest fragments, barrier islands and
atolls.

Every one of the 31 datasets with at least five islands with species fits without an
error: `sar_average` (all 20 models converge on each), `sar_threshold` and `lin_pow`.
Theurkauf 2017, Usher 1986 and Zagar 2013 have too few islands with an area to fit.
Islands with no species recorded are left out, since only `lin_pow` fits zero richness,
and so are islands the deposit gives no area for (see the tables' README).

| Measure | Result |
| --- | --- |
| Median log-log z | 0.16 |
| Datasets with log-log R² ≥ 0.5 | 16 of 31 |
| Best by AICc | `linear` and `power` 7 each, `loga` and `negexpo` 5, `koba` 3, others 4 |
| Threshold model chosen | Linear 25, ZslopeOne 4, ContOne 2 |

Most of these relationships are weak: richness here counts the species recorded in
sampled assemblages, not complete island lists, and the deposit marks sampling effort
as not standardised between islands in 27 of the 34 studies. Gooriah et al.'s point is exactly that sampled
richness and island richness differ. A threshold model wins in only 6 of 31, mostly on
the datasets with the strongest relationships.

| Dataset | Islands | Taxon | Islands are | log-log z | R² | Best by AICc | Threshold model |
| --- | ---: | --- | --- | ---: | ---: | --- | --- |
| Andrade 2002 | 12 | Birds | forest islands | 0.64 | 0.60 | `linear` | Linear |
| As 1984 | 10 | Beetles | true islands | 0.19 | 0.53 | `linear` | ZslopeOne |
| Barun 2010 | 6 | Herpetofauna | true islands | -0.06 | 0.02 | `power` | Linear |
| Bell 2017 | 30 | Beetles | lake islands | -0.02 | 0.05 | `power` | Linear |
| Borges 2016 | 7 | Invertebrates | true islands | 0.14 | 0.49 | `koba` | Linear |
| Choi 2011 | 17 | Invertebrates | true islands | 0.35 | 0.36 | `loga` | Linear |
| Dasilva 2019 | 14 | Invertebrates | forest islands | 0.15 | 0.67 | `loga` | Linear |
| Davidar 2001 | 38 | Birds | true islands | 0.16 | 0.63 | `p2` | Linear |
| Devy 1998 | 11 | Invertebrates | true islands | 0.18 | 0.61 | `monod` | Linear |
| Dueser 1979 | 10 | Mammals | barrier islands | 0.34 | 0.56 | `linear` | Linear |
| Evans 2016 | 13 | Birds | true islands | 0.11 | 0.07 | `linear` | Linear |
| Hatteland 2008 | 14 | Beetles | true islands | 0.14 | 0.06 | `linear` | Linear |
| Jonsson 2011 | 22 | Birds | lake islands | 0.32 | 0.57 | `power` | ZslopeOne |
| Karger 2014 | 12 | Plants | true islands | 0.18 | 0.55 | `power` | Linear |
| Kotze 2000 | 24 | Beetles | true islands | 0.11 | 0.47 | `koba` | Linear |
| Macdonald 2018 | 29 | Invertebrates | lake islands | 0.54 | 0.69 | `power` | ZslopeOne |
| Pereira 2017 | 11 | Invertebrates | forest islands | 0.12 | 0.06 | `negexpo` | Linear |
| Perillo 2017 | 14 | Invertebrates | forest islands | 0.03 | 0.01 | `negexpo` | Linear |
| Pierce 2003 | 10 | Birds | atolls | 0.04 | 0.06 | `monod` | Linear |
| Puspitasari 2016 | 10 | Beetles | true islands | 0.79 | 0.69 | `loga` | Linear |
| Rangasamy frogs 2014 | 9 | Herpetofauna | true islands | 0.11 | 0.13 | `negexpo` | Linear |
| Rangasamy lizards 2014 | 14 | Herpetofauna | true islands | 0.28 | 0.62 | `negexpo` | Linear |
| Rocamora 2003 | 11 | Birds | atolls | 0.04 | 0.01 | `negexpo` | Linear |
| Schilthuizen 2013 | 13 | Invertebrates | true islands | 0.12 | 0.77 | `loga` | Linear |
| Schrader 2019 | 40 | Plants | true islands | 0.42 | 0.91 | `p1` | ContOne |
| Surendran Vasudevan 2015 frogs | 10 | Herpetofauna | true islands | 0.21 | 0.43 | `linear` | Linear |
| Surendran Vasudevan 2015 lizards | 14 | Herpetofauna | true islands | 0.22 | 0.90 | `koba` | Linear |
| Waqasakiti 2018 | 10 | Beetles | true islands | 0.22 | 0.53 | `power` | Linear |
| Werden 2012 | 8 | Mammals | barrier islands | 0.13 | 0.31 | `loga` | Linear |
| Werner Zalewski 2006 | 14 | Beetles | lake islands | -0.17 | 0.38 | `linear` | ZslopeOne |
| Xu 2017 | 42 | Invertebrates | true islands | 0.35 | 0.79 | `power` | ContOne |

"Best by AICc" is the top model of `sar_average`; "Threshold model" is the one
`sar_threshold` chooses (Linear means no threshold). The tables are in the repository
under `tests/data/gooriah2021/`, with a script that rebuilds them from the deposit and
the list of the original studies, and `tests/test_gooriah.py` fits them all.
Schrader 2019 and Xu 2017 also ship as `load_schrader()` and `load_xu()`.

## Source

- Gooriah L, Blowes SA, Sagouis A, Schrader J, Karger DN, Kreft H, Chase JM (2021).
  Synthesis reveals that island species-area relationships emerge from processes
  beyond passive sampling. *Global Ecology and Biogeography*, 30, 2119-2131.

If you use one of these datasets, cite its original study as well as the synthesis.
