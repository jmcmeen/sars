"""Data loading and I/O adapters."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


def load_galap() -> pd.DataFrame:
    """Load the Galapagos plant species-area dataset.

    Returns the Preston (1962) 16-island dataset as shipped in the R sars
    package (Albemarle/Isabela excluded).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    """
    try:
        ref = importlib.resources.files("sars.data").joinpath("galap.csv")
        with importlib.resources.as_file(ref) as path:
            df = pd.read_csv(path)
    except (TypeError, FileNotFoundError, ModuleNotFoundError) as exc:
        raise FileNotFoundError(
            "galap.csv is not available. The sars package data may not "
            "have been installed correctly. Try reinstalling with: "
            "pip install --force-reinstall sars"
        ) from exc
    df = df.rename(columns={"a": "area", "s": "species"})
    return df


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
        raise KeyError(
            f"Column {area_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    if species_col not in df.columns:
        raise KeyError(
            f"Column {species_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    return df[[area_col, species_col]].rename(
        columns={area_col: "area", species_col: "species"}
    )


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
        raise KeyError(
            f"Column {area_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    if species_col not in df.columns:
        raise KeyError(
            f"Column {species_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    return df[[area_col, species_col]].rename(
        columns={area_col: "area", species_col: "species"}
    )


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
                f"Key {area_key!r} not found in observation {i}. "
                f"Available keys: {list(obs.keys())}"
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
            "geopandas is required for from_geodataframe(). "
            "Install with: pip install sars[geo]"
        ) from e

    if species_col not in gdf.columns:
        raise KeyError(
            f"Column {species_col!r} not found. "
            f"Available columns: {list(gdf.columns)}"
        )

    if area_col is not None:
        if area_col not in gdf.columns:
            raise KeyError(
                f"Column {area_col!r} not found. "
                f"Available columns: {list(gdf.columns)}"
            )
        areas = gdf[area_col].values
    else:
        raw_area = gdf.geometry.area
        unit_divisors = {"km2": 1e6, "m2": 1.0, "ha": 1e4}
        if crs_units not in unit_divisors:
            raise ValueError(
                f"Unknown crs_units {crs_units!r}. "
                f"Choose from {list(unit_divisors.keys())}"
            )
        areas = raw_area / unit_divisors[crs_units]

    return pd.DataFrame({
        "area": areas,
        "species": gdf[species_col].values,
    })
