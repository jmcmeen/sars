"""Data loading and I/O adapters."""

from __future__ import annotations

import importlib.resources
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


def _load_dataset(name: str) -> pd.DataFrame:
    """Read a bundled dataset, ``sars/data/<name>.csv``, as an area/species frame.

    The files taken from R sars keep its column names (``a``, ``s``) and row
    order, so they can be compared with R's copies line for line; the others
    already name their columns ``area`` and ``species``.
    """
    filename = f"{name}.csv"
    # Only the lookup is guarded: an error from parsing a file that is there
    # is a different problem, and shouldn't be reported as a missing file.
    try:
        ref = importlib.resources.files("sars.data").joinpath(filename)
        found = ref.is_file()
    except (TypeError, ModuleNotFoundError):
        found = False
    if not found:
        raise FileNotFoundError(
            f"{filename} is not available. The sars package data may not "
            "have been installed correctly. Try reinstalling with: "
            "pip install --force-reinstall sars"
        )
    with importlib.resources.as_file(ref) as path:
        df = pd.read_csv(path)
    return df.rename(columns={"a": "area", "s": "species"})


def load_galap() -> pd.DataFrame:
    """Load the Galapagos plant species-area dataset.

    Returns the 16-island dataset of plant species counts and island area
    (km²) as shipped in the R sars package (Albemarle/Isabela excluded), from
    Preston FW (1962), The canonical distribution of commonness and rarity,
    Ecology 43, 185-215 and 410-432.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    """
    return _load_dataset("galap")


def load_gala() -> pd.DataFrame:
    """Load Johnson & Raven's Galápagos plant species-area dataset.

    Plant species counts, endemic species counts and island area (km²) for
    30 islands of the Galápagos, from Isabela (4669 km²) down to islets of
    0.01 km², as compiled by Johnson and Raven (1973). The values are those
    in R faraway's ``gala`` dataset (its unfilled original, ``galamiss``,
    has the same species, endemic and area values), in its row order. The
    island names are faraway's with the words separated ("Santa Cruz" for
    "SantaCruz"); its two Gardner islands stay "Gardner1" and "Gardner2".

    A different compilation from `load_galap` (Preston 1962), which has 16
    islands and leaves out Isabela. Every island has at least two species, so
    the data can go straight to the fitting functions, which use the 'area'
    and 'species' columns. To fit endemic richness, use the 'endemics' column
    as 'species', after dropping Daphne Minor, which has none.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'island', 'area', 'species' and 'endemics'.

    References
    ----------
    Johnson MP, Raven PH (1973). Species number and endemism: the Galápagos
    Archipelago revisited. Science 179, 893-895.
    """
    return _load_dataset("gala")


def load_aegean() -> pd.DataFrame:
    """Load the Aegean terrestrial isopod species-area dataset.

    Terrestrial isopod species counts and island area (km²) for 90 islands
    of the Aegean Sea, Greece, from Crete (8261 km²) down to islets of a few
    thousand square metres. The values are those shipped in the R sars
    package; rows are in its order, largest island first. Every island has
    at least one species (1 to 34), so the data can go straight to the
    fitting functions.

    The source used these islands to study the small-island effect, which
    makes them a common example for `sar_threshold`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    References
    ----------
    Sfenthourakis S, Triantis KA (2009). Habitat diversity, ecological
    requirements of species and the Small Island Effect. Diversity and
    Distributions 15, 131-140.
    """
    return _load_dataset("aegean")


def load_aegean2() -> pd.DataFrame:
    """Load the Aegean plant species-area dataset.

    Plant species counts and island area (km²) for 173 islands of the Aegean
    Sea, Greece, from islets of 600 m² (0.0006 km²) up to Crete (8270 km²).
    The values are those shipped in the R sars package; rows are in its
    order, smallest island first. Every island has at least one species (2
    to 1751), so the data can go straight to the fitting functions.

    A different set of islands from `load_aegean`: the two come from separate
    studies of different taxa. With areas spanning seven orders of magnitude
    this is the usual example for `sar_threshold`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    References
    ----------
    Matthews TJ, Rigal F, Kougioumoutzis K, Trigas P, Triantis KA (2020).
    Unravelling the small-island effect through phylogenetic community
    ecology. Journal of Biogeography 47, 2341-2352.
    """
    return _load_dataset("aegean2")


def load_niering() -> pd.DataFrame:
    """Load the Kapingamarangi Atoll plant species-area dataset.

    Plant species counts and island area (km²) for 32 islands of
    Kapingamarangi Atoll in the Caroline Islands, Micronesia. The islands are
    tiny, 0.00012 to 0.318 km² (120 m² to 32 ha), so a model fitted to them
    has its parameters in those units. The values are those shipped in the R
    sars package; rows are in its order, smallest island first. Every island
    has at least one species (5 to 60), so the data can go straight to the
    fitting functions.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    References
    ----------
    Niering WA (1963). Terrestrial ecology of Kapingamarangi Atoll, Caroline
    Islands. Ecological Monographs 33, 131-160.
    """
    return _load_dataset("niering")


def load_schrader() -> pd.DataFrame:
    """Load Schrader et al.'s plants on small tropical islands.

    Plant species counts and island area (km²) for 60 small islands in
    Raja Ampat, Indonesia, from rocks of 3 m² (2.8e-6 km²) to 1.2 ha
    (0.0118 km²), surveyed by Schrader et al. (2019) and compiled by Gooriah
    et al. (2021), whose deposit is released under CC0. Areas are converted
    from the deposit's hectares; islands are identified by the survey's codes.
    Richness counts the species recorded on each island's transects (one to
    six, more on larger islands), not a complete flora.

    The deposit gives areas from 0.01 ha up rounded to the nearest 0.01 ha.
    Twenty islets, all under 30 m², have no plants recorded. They are kept,
    since they are the small-island end of the study, but only `lin_pow`
    (which adds ``con`` before taking logs) fits zero richness; drop them for
    the other models (see Examples).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'island', 'area' and 'species'.

    Examples
    --------
    >>> schrader = sars.load_schrader()
    >>> planted = schrader[schrader["species"] > 0]
    >>> sars.sar_multi(planted)
    MultiSARFit(20 models, best='p1')

    References
    ----------
    Schrader J, Moeljono S, Keppel G, Kreft H (2019). Plants on small islands
    revisited: the effects of spatial scale and habitat quality on the
    species-area relationship. Ecography 42, 1405-1414.

    Gooriah L, Blowes SA, Sagouis A, Schrader J, Karger DN, Kreft H, Chase JM
    (2021). Synthesis reveals that island species-area relationships emerge
    from processes beyond passive sampling. Global Ecology and Biogeography
    30, 2119-2131.
    """
    return _load_dataset("schrader")


def load_xu() -> pd.DataFrame:
    """Load Xu et al.'s butterflies of the Zhoushan Archipelago.

    Butterfly species counts and island area (km²) for 42 islands of the
    Zhoushan Archipelago, China, from 0.0255 km² to Zhoushan Island
    (476 km²), surveyed by Xu et al. (2017) and compiled by Gooriah et al.
    (2021), whose deposit is released under CC0. Areas are converted from the
    deposit's hectares. Richness counts the species recorded along each
    island's transects, whose length varied between islands, not a complete
    list. Every island has at least one species, so the data can go straight
    to the fitting functions.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'island', 'area' and 'species'.

    References
    ----------
    Xu A, Han X, Zhang X, Millien V, Wang Y (2017). Nestedness of butterfly
    assemblages in the Zhoushan Archipelago, China: area effects, life-history
    traits and conservation implications. Biodiversity and Conservation 26,
    1375-1392.

    Gooriah L, Blowes SA, Sagouis A, Schrader J, Karger DN, Kreft H, Chase JM
    (2021). Synthesis reveals that island species-area relationships emerge
    from processes beyond passive sampling. Global Ecology and Biogeography
    30, 2119-2131.
    """
    return _load_dataset("xu")


def from_df(
    df: pd.DataFrame,
    area_col: str = "area",
    species_col: str = "species",
) -> pd.DataFrame:
    """Create SAR-formatted DataFrame from an existing DataFrame.

    Selects and renames the specified columns to the standard 'area' and
    'species' names expected by all ``sars`` model functions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing area and species richness columns.
    area_col : str
        Name of the column containing area values.
    species_col : str
        Name of the column containing species richness values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Raises
    ------
    KeyError
        If the specified columns are not found.
    """
    if area_col not in df.columns:
        raise KeyError(f"Column {area_col!r} not found. Available columns: {list(df.columns)}")
    if species_col not in df.columns:
        raise KeyError(f"Column {species_col!r} not found. Available columns: {list(df.columns)}")
    return df[[area_col, species_col]].rename(columns={area_col: "area", species_col: "species"})


def from_csv(
    path: str | Path,
    area_col: str = "area",
    species_col: str = "species",
) -> pd.DataFrame:
    """Load species-area data from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    area_col : str
        Name of the column containing area values.
    species_col : str
        Name of the column containing species richness values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the specified columns are not found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if area_col not in df.columns:
        raise KeyError(f"Column {area_col!r} not found. Available columns: {list(df.columns)}")
    if species_col not in df.columns:
        raise KeyError(f"Column {species_col!r} not found. Available columns: {list(df.columns)}")
    return df[[area_col, species_col]].rename(columns={area_col: "area", species_col: "species"})


def from_pyinaturalist(
    observations: list[dict],
    area_key: str = "area",
    species_key: str = "species_count",
) -> pd.DataFrame:
    """Convert pyiNaturalist observation summaries to SAR format.

    Expects a list of dicts where each dict represents an island or site
    with area and species count fields. This is not raw iNaturalist
    observation data — it should be pre-aggregated by site.

    Parameters
    ----------
    observations : list[dict]
        List of dicts, each with at least an area field and a species
        count field.
    area_key : str
        Key in each dict for the area value.
    species_key : str
        Key in each dict for the species richness value.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Raises
    ------
    KeyError
        If the specified keys are not found in the observation dicts.
    """
    if not observations:
        return pd.DataFrame(columns=["area", "species"])

    rows = []
    for i, obs in enumerate(observations):
        if area_key not in obs:
            raise KeyError(
                f"Key {area_key!r} not found in observation {i}. Available keys: {list(obs.keys())}"
            )
        if species_key not in obs:
            raise KeyError(
                f"Key {species_key!r} not found in observation {i}. "
                f"Available keys: {list(obs.keys())}"
            )
        rows.append({"area": obs[area_key], "species": obs[species_key]})

    return pd.DataFrame(rows)


def from_geodataframe(
    gdf: GeoDataFrame,
    species_col: str = "species",
    area_col: str | None = None,
    area_units: str | None = None,
    *,
    crs_units: str | None = None,
) -> pd.DataFrame:
    """Convert a GeoDataFrame of polygons to SAR format.

    Computes area from polygon geometries if no area column is specified.
    Requires geopandas and shapely (install with ``pip install "sars[geo]"``).

    Areas are measured in the GeoDataFrame's own coordinate reference system
    (CRS), so it must be a projected one: reproject longitude/latitude data
    first, e.g. ``gdf.to_crs("EPSG:6933")``. Use an equal-area projection.
    A projected CRS that isn't equal-area (Web Mercator, or UTM far from its
    zone) is accepted but distorts the areas, and this function cannot tell.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with polygon geometries and a species count column.
    species_col : str
        Column name for species richness values.
    area_col : str, optional
        Column name for pre-computed area values. If None, area is
        computed from geometries.
    area_units : str, optional
        Unit of the computed areas: "km2" (the default), "ha" or "m2". The CRS's
        own length unit is read from the CRS, so a CRS in feet gives the same
        areas as one in metres. Ignored when ``area_col`` is given.
    crs_units : str, optional
        Deprecated alias of ``area_units``, which names what it sets: the unit
        of the areas returned, not a property of the CRS.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species', with the GeoDataFrame's
        index whether the areas are computed or taken from ``area_col``.

    Raises
    ------
    ImportError
        If geopandas is not installed.
    KeyError
        If the specified columns are not found.
    TypeError
        If both ``area_units`` and ``crs_units`` are given.
    ValueError
        If ``area_units`` is not one of the units above (even with
        ``area_col``, where it is otherwise ignored) or, when area is
        computed from the geometries, if any geometry is missing, empty or
        has no area (a point, a line, a degenerate polygon), if the CRS is
        not a projected one (geographic CRSs, in degrees, included), or if
        there is no CRS and the coordinates look like longitude and latitude
        (all within ±180 and ±90).

    Warns
    -----
    FutureWarning
        If ``crs_units`` is used.
    UserWarning
        If area is computed from geometries that have no CRS and whose
        coordinates don't look like degrees. They are then taken to be metres.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "geopandas is required for from_geodataframe(). Install with: pip install 'sars[geo]'"
        ) from e

    if crs_units is not None:
        if area_units is not None:
            raise TypeError("Pass area_units or its deprecated alias crs_units, not both.")
        warnings.warn(
            "crs_units is deprecated and will be removed in a future release; use "
            "area_units, which names the unit of the areas returned.",
            FutureWarning,
            stacklevel=2,
        )
        area_units = crs_units
    if area_units is None:
        area_units = "km2"
    unit_divisors = {"km2": 1e6, "m2": 1.0, "ha": 1e4}
    if area_units not in unit_divisors:
        # Checked on both paths, so a misspelt unit isn't passed over when it is ignored.
        raise ValueError(
            f"Unknown area_units {area_units!r}. Choose from {list(unit_divisors.keys())}"
        )

    if species_col not in gdf.columns:
        raise KeyError(f"Column {species_col!r} not found. Available columns: {list(gdf.columns)}")

    if area_col is not None:
        if area_col not in gdf.columns:
            raise KeyError(f"Column {area_col!r} not found. Available columns: {list(gdf.columns)}")
        areas = gdf[area_col].to_numpy()
    else:
        geometry = gdf.geometry
        if geometry.isna().any() or geometry.is_empty.any():
            raise ValueError(
                "Some rows have a missing or empty geometry, which has no area. Drop them, "
                "or pass area_col to use areas from a column."
            )
        # Squared degrees are not an area: a degree of longitude shrinks towards the
        # poles. Dividing them by 1e6 would give numbers that look like tiny islands.
        reproject = (
            'Reproject to an equal-area projected CRS first, e.g. gdf.to_crs("EPSG:6933"), '
            "or pass area_col to use areas from a column."
        )
        crs = gdf.crs
        if crs is None:
            min_x, min_y, max_x, max_y = geometry.total_bounds
            if min_x >= -180 and max_x <= 180 and min_y >= -90 and max_y <= 90:
                # Projected coordinates this close to their origin are rare; longitude
                # and latitude saved without a CRS are not.
                raise ValueError(
                    "The GeoDataFrame has no CRS and every coordinate lies within ±180 and "
                    "±90, so they look like longitude and latitude in degrees, and areas "
                    "computed from them would be meaningless. Set the CRS with "
                    f'gdf.set_crs("EPSG:4326") if that is what they are. {reproject}'
                )
            warnings.warn(
                "The GeoDataFrame has no CRS, so its coordinates are taken to be metres. "
                "Set one with gdf.set_crs() to be sure of the units.",
                UserWarning,
                stacklevel=2,
            )
            metres_per_unit = 1.0
        elif not crs.is_projected:
            # Geographic (degrees), but also geocentric and engineering systems, whose
            # coordinates don't lie on a map plane either.
            kind = (
                "geographic: its coordinates are degrees" if crs.is_geographic else "not projected"
            )
            raise ValueError(
                f"The GeoDataFrame's CRS ({crs.name}) is {kind}, and areas computed from "
                f"it are meaningless. {reproject}"
            )
        else:
            # State plane and some national grids are in feet, not metres.
            metres_per_unit = crs.axis_info[0].unit_conversion_factor
        raw_area = geometry.area
        # A point, a line, or a polygon whose outline encloses nothing has area zero,
        # which would otherwise pass for a very small island.
        no_area = ~(raw_area > 0)
        if no_area.any():
            kinds = sorted(set(geometry[no_area].geom_type))
            raise ValueError(
                f"Rows {list(gdf.index[no_area])} have no area (geometry type {kinds}). Area "
                "can only be computed from valid polygons: fix or drop these rows, or pass "
                "area_col to use areas from a column."
            )
        areas = (raw_area * metres_per_unit**2 / unit_divisors[area_units]).to_numpy()

    # Both paths keep the GeoDataFrame's index, as from_df keeps its input's, so
    # the rows can be joined back to the polygons.
    return pd.DataFrame(
        {"area": areas, "species": gdf[species_col].to_numpy()},
        index=gdf.index,
    )
