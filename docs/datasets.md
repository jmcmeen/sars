# Datasets

Seven published island datasets ship with the package, two more in the formats of the
habitat models, and the species counts behind one of them (below). Each of the seven
returns a DataFrame with `area` and `species` columns; `load_gala`, `load_schrader` and
`load_xu` add `island` (the island's name or survey code), and `load_gala`
`endemics`.

| Loader | Islands | Taxon | Area units | Source |
| ------ | ------- | ----- | ---------- | ------ |
| `load_galap()` | 16, Galápagos | Plants | km² | Preston (1962) |
| `load_gala()` | 30, Galápagos | Plants, with endemics | km² | Johnson & Raven (1973) |
| `load_aegean()` | 90, Aegean Sea, Greece | Terrestrial isopods | km² | Sfenthourakis & Triantis (2009) |
| `load_aegean2()` | 173, Aegean Sea, Greece | Plants | km² | Matthews et al. (2020) |
| `load_niering()` | 32, Kapingamarangi Atoll, Micronesia | Plants | km² | Niering (1963) |
| `load_schrader()` | 60, Raja Ampat, Indonesia | Plants | km² | Schrader et al. (2019) |
| `load_xu()` | 42, Zhoushan Archipelago, China | Butterflies | km² | Xu et al. (2017) |

The values of `load_galap`, `load_aegean`, `load_aegean2` and `load_niering` are
those shipped with R `sars` 2.1.1, in the same row order, so results can be compared
with R's directly; `load_gala`'s are those in R `faraway`'s `gala`. `load_schrader` and
`load_xu` come from the compilation of Gooriah et al. (2021), released under CC0, with
the areas converted from hectares. Only `load_schrader` has islands with no species,
which have to be dropped before most fits (see below).

Two more come in the formats the habitat models take, from R `sars` 2.1.1:

| Loader | Sites | Taxon | Columns | Source |
| ------ | ----- | ----- | ------- | ------ |
| `load_habitat()` | 12 islands, Skyros, Greece | Land snails | `area` (km²), `habitats`, `species` | Triantis et al. (2005) |
| `load_countryside()` | 425 nested plots, Portugal | Understory plants | three habitat areas (m²), four habitat-affinity groups' richness | Proença & Pereira (2013) |

`load_xu_abundance()` has the counts behind `load_xu`: individuals of 68 butterfly
species on each of its 42 islands, species by island, in `load_xu`'s order, for
`standardised_richness`. It comes from the same CC0 compilation of Gooriah et al. (2021).

`load_countryside`'s plots are nested (1 m² plots within 64 m², 4,096 m² and 26.2 ha
squares), so its rows aren't independent.

Some things to know about each:

- **`load_galap`** leaves out Albemarle (Isabela), which is almost six times
  the size of the next largest island.
- **`load_gala`** is a different compilation of the Galápagos flora: 30
  islands, Isabela included, from 0.01 to 4,669 km², with island names and the
  number of endemic species on each as well. Its species-area relationship is
  much tighter than `load_galap`'s (log-log z = 0.39, R² = 0.76). To fit
  endemic richness, use `endemics` as `species` after dropping Daphne Minor,
  which has none:

  ```python
  gala = sars.load_gala()
  endemic = gala[gala["endemics"] > 0].assign(species=lambda d: d["endemics"])
  sars.sar_multi(endemic)
  ```
- **`load_aegean`** and **`load_aegean2`** come from separate studies of
  different taxa and are different sets of islands. Both run from islets of
  well under a hectare to Crete, six to seven orders of magnitude in area.
- **`load_schrader`** and **`load_xu`** count the species recorded on each
  island's transects, not complete island lists, and the survey effort varied
  between islands (Schrader's number of transects rose with island size). Both
  relationships are tight: log-log z = 0.42 and R² = 0.91 for Schrader's 40
  islands with plants (0.43 and 0.89 on all 60 through `lin_pow`, which adds 1
  to every richness), z = 0.35 and R² = 0.79 for Xu's butterflies.
- **`load_schrader`**'s islands are tiny, from rocks of 3 m² to 1.2 ha
  (2.8e-6 to 0.0118 km²); the deposit gives areas from 0.01 ha up rounded to
  the nearest 0.01 ha, so four islands share 0.0001 km². Twenty islets, all
  under 30 m², have no plants
  recorded. They are kept, since they are the small-island end of the study,
  but only `lin_pow` (which adds `con` before taking logs) fits zero richness;
  drop them for the other models:

  ```python
  schrader = sars.load_schrader()
  sars.sar_multi(schrader[schrader["species"] > 0])
  ```
- **`load_niering`**'s islands are all small: 0.00012 to 0.318 km² (120 m² to
  32 ha). Fitted parameters are in those units, so the power model's `c` is
  the richness expected on 1 km², three times the size of the largest island.

## Example: the small-island effect

On very small islands richness often varies independently of area, and only
starts to rise with it above some threshold. The two Aegean datasets come
from studies of this effect, and are the usual examples for
[`sar_threshold`](api/threshold.md):

```python
import sars

aegean2 = sars.load_aegean2()
th = sars.sar_threshold(aegean2)
print(th)
# ThresholdFit(best='ContOne', breakpoint=2.305 (ln area), breakpoint_area=10.02)

print(th.summary[["model", "breakpoint_area", "AICc", "R2"]].round(2))
#        model  breakpoint_area     AICc    R2
# 0    ContOne            10.02  2046.17  0.94
# 1  ZslopeOne             3.95  2061.77  0.94
# 2     Linear              NaN  2305.19  0.74
```

Both one-threshold models fit far better than a single line through species
against ln(area), with the breakpoint at about 10 km² for the best of them
(two joined sloping segments). The isopods and the atoll's plants favour a
threshold model too, at their own scales:

```python
sars.sar_threshold(sars.load_aegean())
# ThresholdFit(best='ContOne', breakpoint=1.263 (ln area), breakpoint_area=3.538)

sars.sar_threshold(sars.load_niering())
# ThresholdFit(best='ZslopeOne', breakpoint=-4.692 (ln area), breakpoint_area=0.009166)
```

For Kapingamarangi the best model is flat below the breakpoint: under about
0.009 km² (0.9 ha) richness doesn't depend on area. Johnson and Raven's
Galápagos islands show the same kind of threshold, near 11 km²:

```python
sars.sar_threshold(sars.load_gala())
# ThresholdFit(best='ZslopeOne', breakpoint=2.384 (ln area), breakpoint_area=10.85)
```

## Sources

- Gooriah L, Blowes SA, Sagouis A, Schrader J, Karger DN, Kreft H, Chase JM
  (2021). Synthesis reveals that island species-area relationships emerge from
  processes beyond passive sampling. *Global Ecology and Biogeography*, 30,
  2119-2131. Data: Zenodo record 7311286, CC0. (`load_schrader`, `load_xu`, `load_xu_abundance`)
- Johnson MP, Raven PH (1973). Species number and endemism: the Galápagos
  Archipelago revisited. *Science*, 179, 893-895. (`load_gala`)
- Matthews TJ, Rigal F, Kougioumoutzis K, Trigas P, Triantis KA (2020).
  Unravelling the small-island effect through phylogenetic community ecology.
  *Journal of Biogeography*, 47, 2341-2352. (`load_aegean2`)
- Niering WA (1963). Terrestrial ecology of Kapingamarangi Atoll, Caroline
  Islands. *Ecological Monographs*, 33, 131-160. (`load_niering`)
- Preston FW (1962). The canonical distribution of commonness and rarity.
  *Ecology*, 43, 185-215, 410-432. (`load_galap`)
- Schrader J, Moeljono S, Keppel G, Kreft H (2019). Plants on small islands
  revisited: the effects of spatial scale and habitat quality on the
  species-area relationship. *Ecography*, 42, 1405-1414. (`load_schrader`)
- Sfenthourakis S, Triantis KA (2009). Habitat diversity, ecological
  requirements of species and the Small Island Effect. *Diversity and
  Distributions*, 15, 131-140. (`load_aegean`)
- Xu A, Han X, Zhang X, Millien V, Wang Y (2017). Nestedness of butterfly
  assemblages in the Zhoushan Archipelago, China: area effects, life-history
  traits and conservation implications. *Biodiversity and Conservation*, 26,
  1375-1392. (`load_xu`, `load_xu_abundance`)

If you publish results from one of these datasets, cite its source.
