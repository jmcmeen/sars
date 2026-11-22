# Data for the example notebooks

## `appalachian_island.csv`

Amphibian species richness at 26 independent sites in the southern Appalachians: the
"island model" data, Appendix 2 of

> Stout JB, Jessee LD, McMeen JN (2025). Nested and island models for determining the
> species-area relationship of southern Appalachian amphibians. *Journal of North
> American Herpetology*, 2025(1), 1–7. <https://doi.org/10.17161/jnah.v2025i1.21974>

| Column | Meaning |
| --- | --- |
| `locality` | Site name |
| `area_km2` | Site area in km² |
| `amphibians` | Amphibian species recorded (frogs plus salamanders) |
| `frogs` | Frog species recorded |
| `salamanders` | Salamander species recorded |

The article's page states that it is licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). The table is
reproduced here by one of its authors. Cite the paper if you use it.

Used by `appalachian.ipynb`.

## `appalachian_counties.geojson`

Outlines of ten counties in Tennessee, North Carolina and Georgia, in longitude and
latitude (WGS 84): the ten sites of `appalachian_island.csv` that are counties.

The polygons are from [Natural Earth](https://www.naturalearthdata.com/)'s 1:10m cultural
vectors, admin 2 counties layer (`ne_10m_admin_2_counties`). Natural Earth is in the
public domain: no permission is needed to use, modify or redistribute it. Coordinates
are rounded to five decimal places, about a metre.

| Property | Meaning |
| --- | --- |
| `county` | County name, Natural Earth's `NAME` |
| `state` | Two-letter state abbreviation, Natural Earth's `REGION` |

The file holds no species counts. `geodataframe.ipynb` joins it to
`appalachian_island.csv` by county name and state.

## `galapagos_islands.geojson`

Outlines of 28 Galápagos islands, in longitude and latitude (WGS 84): the islands of
`sars.load_gala()` (Johnson and Raven 1973) other than its two Gardners, which it doesn't
say apart. Las Plazas is one feature holding North and South Plaza, as Johnson and Raven
give one figure for the two.

The polygons are from [OpenStreetMap](https://www.openstreetmap.org/), © OpenStreetMap
contributors. **This file is a derivative of the OpenStreetMap database and is licensed
under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
(ODbL), not under the repository's MIT licence.** If you reuse it, keep the attribution
and share any adapted version under the ODbL (see
[OpenStreetMap's copyright page](https://www.openstreetmap.org/copyright)). They were taken as each island's way or multipolygon relation (the `osm` property
names it), and each outline was simplified with a tolerance of 1.5% of the square root of
its area, at most 60 m, so that no island's area changed by more than 0.75%. Coordinates
are rounded to five decimal places, about a metre.

| Property | Meaning |
| --- | --- |
| `island` | The island's name as `load_gala()` spells it |
| `osm_name` | OpenStreetMap's `name` for it |
| `osm` | The OpenStreetMap feature: `way/<id>` or `relation/<id>` |

The file holds no species counts. `galapagos_polygons.ipynb` joins it to `load_gala()` by
`island`.
