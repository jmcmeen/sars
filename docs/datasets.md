# Datasets

Five published island datasets ship with the package. Each loader returns a
DataFrame with `area` and `species` columns, ready for any fitting function;
`load_gala` adds `island` (the island's name) and `endemics`.

| Loader | Islands | Taxon | Area units | Source |
| ------ | ------- | ----- | ---------- | ------ |
| `load_galap()` | 16, Galápagos | Plants | km² | Preston (1962) |
| `load_gala()` | 30, Galápagos | Plants, with endemics | km² | Johnson & Raven (1973) |
| `load_aegean()` | 90, Aegean Sea, Greece | Terrestrial isopods | km² | Sfenthourakis & Triantis (2009) |
| `load_aegean2()` | 173, Aegean Sea, Greece | Plants | km² | Matthews et al. (2020) |
| `load_niering()` | 32, Kapingamarangi Atoll, Micronesia | Plants | km² | Niering (1963) |

The values of all but `load_gala` are those shipped with R `sars` 2.1.1, in the same
row order, so results can be compared with R's directly; `load_gala`'s are those in R
`faraway`'s `gala`. No island in any of them has zero
species, so none needs filtering before a fit.

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

- Preston FW (1962). The canonical distribution of commonness and rarity.
  *Ecology*, 43, 185-215, 410-432. (`load_galap`)
- Johnson MP, Raven PH (1973). Species number and endemism: the Galápagos
  Archipelago revisited. *Science*, 179, 893-895. (`load_gala`)
- Sfenthourakis S, Triantis KA (2009). Habitat diversity, ecological
  requirements of species and the Small Island Effect. *Diversity and
  Distributions*, 15, 131-140. (`load_aegean`)
- Matthews TJ, Rigal F, Kougioumoutzis K, Trigas P, Triantis KA (2020).
  Unravelling the small-island effect through phylogenetic community ecology.
  *Journal of Biogeography*, 47, 2341-2352. (`load_aegean2`)
- Niering WA (1963). Terrestrial ecology of Kapingamarangi Atoll, Caroline
  Islands. *Ecological Monographs*, 33, 131-160. (`load_niering`)

If you publish results from one of these datasets, cite its source.
