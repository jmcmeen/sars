"""Data loading and I/O adapters."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


def _load_dataset(name: str) -> pd.DataFrame:
    """Read a bundled dataset, ``sars/data/<name>.csv``, as an area/species frame.

    The files keep the column names R sars uses (``a``, ``s``) and its row
    order, so they can be compared with R's copies line for line.
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
    crs_units: str = "km2",
) -> pd.DataFrame:
    """Convert a GeoDataFrame of polygons to SAR format.

    Computes area from polygon geometries if no area column is specified.
    Requires geopandas and shapely (install with ``pip install sars[geo]``).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with polygon geometries and a species count column.
    species_col : str
        Column name for species richness values.
    area_col : str, optional
        Column name for pre-computed area values. If None, area is
        computed from geometries.
    crs_units : str
        Unit conversion for computed areas. "km2" divides by 1e6 (assumes
        CRS in metres), "m2" uses raw area, "ha" divides by 1e4.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Raises
    ------
    ImportError
        If geopandas is not installed.
    KeyError
        If the specified columns are not found.
    ValueError
        If geometries are not polygons or CRS units are invalid.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "geopandas is required for from_geodataframe(). Install with: pip install sars[geo]"
        ) from e

    if species_col not in gdf.columns:
        raise KeyError(f"Column {species_col!r} not found. Available columns: {list(gdf.columns)}")

    if area_col is not None:
        if area_col not in gdf.columns:
            raise KeyError(f"Column {area_col!r} not found. Available columns: {list(gdf.columns)}")
        areas = gdf[area_col].values
    else:
        raw_area = gdf.geometry.area
        unit_divisors = {"km2": 1e6, "m2": 1.0, "ha": 1e4}
        if crs_units not in unit_divisors:
            raise ValueError(
                f"Unknown crs_units {crs_units!r}. Choose from {list(unit_divisors.keys())}"
            )
        areas = raw_area / unit_divisors[crs_units]

    return pd.DataFrame(
        {
            "area": areas,
            "species": gdf[species_col].values,
        }
    )
