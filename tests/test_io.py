"""Tests for I/O adapters."""

import importlib.resources
from pathlib import Path

import pandas as pd
import pytest

import sars
from sars._io import _load_dataset
from sars._models import _validate_data

REF = Path(__file__).parent / "r_reference"


class TestLoadGalap:
    def test_returns_dataframe(self):
        df = sars.load_galap()
        assert "area" in df.columns
        assert "species" in df.columns

    def test_16_rows(self):
        df = sars.load_galap()
        assert len(df) == 16

    def test_values_positive(self):
        df = sars.load_galap()
        assert (df["area"] > 0).all()
        assert (df["species"] > 0).all()


# Rows, then the sums of area and species over R sars's copy of each dataset
# (R: nrow(d), sum(d$a), sum(d$s)), so a changed value shows even where the
# reference tables aren't to hand.
DATASETS = {
    "galap": (16, 3117.76, 1703),
    "aegean": (90, 11943.8839, 878),
    "aegean2": (173, 22593.1625, 53686),
    "niering": (32, 1.10292, 580),
}


@pytest.mark.parametrize("name", DATASETS)
class TestBundledDatasets:
    def test_shape_and_columns(self, name):
        df = getattr(sars, f"load_{name}")()
        assert list(df.columns) == ["area", "species"]
        assert len(df) == DATASETS[name][0]
        assert df["area"].dtype == float
        assert pd.api.types.is_integer_dtype(df["species"])

    def test_passes_validation(self, name):
        """No island has zero species (or area), which the fitters would reject."""
        df = getattr(sars, f"load_{name}")()
        _validate_data(df)
        assert (df["species"] >= 1).all()

    def test_sums(self, name):
        _, area_sum, species_sum = DATASETS[name]
        df = getattr(sars, f"load_{name}")()
        assert df["area"].sum() == pytest.approx(area_sum, rel=1e-12)
        assert df["species"].sum() == species_sum

    def test_file_is_r_export(self, name):
        """The bundled file is, byte for byte, the one the reference generator
        writes from R sars's copy of the dataset."""
        bundled = importlib.resources.files("sars.data").joinpath(f"{name}.csv")
        assert bundled.read_bytes() == (REF / f"{name}.csv").read_bytes()

    def test_file_is_package_data(self, name):
        """Each file is found through the installed package, which is where the
        loaders look. The wheel target ships the package directory whole
        (``packages = ["src/sars"]`` in pyproject.toml), data files included."""
        assert importlib.resources.files("sars.data").joinpath(f"{name}.csv").is_file()

    def test_fresh_copy(self, name):
        """Changing a loaded frame doesn't change what the next call returns."""
        df = getattr(sars, f"load_{name}")()
        df.loc[0, "species"] = -1
        assert getattr(sars, f"load_{name}")().loc[0, "species"] > 0

    def test_fits(self, name):
        fit = sars.sar_power(getattr(sars, f"load_{name}")())
        assert fit.converged
        assert 0 < fit.params["z"] < 1


class TestLoadGala:
    """Johnson & Raven (1973), as in R faraway's gala: four columns, not R
    sars's two, so not in DATASETS."""

    def test_shape_columns_and_sums(self):
        df = sars.load_gala()
        assert list(df.columns) == ["island", "area", "species", "endemics"]
        assert len(df) == 30
        assert df["area"].dtype == float
        assert pd.api.types.is_integer_dtype(df["species"])
        assert pd.api.types.is_integer_dtype(df["endemics"])
        assert df["area"].sum() == pytest.approx(7851.26, rel=1e-12)
        assert df["species"].sum() == 2557
        assert df["endemics"].sum() == 783
        assert (df["island"].iloc[0], df["island"].iloc[-1]) == ("Baltra", "Wolf")

    def test_ranges(self):
        df = sars.load_gala()
        assert df["area"].min() == 0.01
        assert df["area"].max() == 4669.32  # Isabela, which load_galap leaves out
        assert (df["endemics"] <= df["species"]).all()

    def test_passes_validation_and_fits(self):
        df = sars.load_gala()
        _validate_data(df)
        log_log = sars.lin_pow(df)
        assert log_log.z == pytest.approx(0.3886, abs=1e-4)
        assert log_log.r_squared == pytest.approx(0.7571, abs=1e-4)
        threshold = sars.sar_threshold(df)
        assert threshold.best_model == "ZslopeOne"
        assert threshold.breakpoint_area == pytest.approx(10.85, abs=0.01)

    def test_endemic_richness_fits_without_the_island_that_has_none(self):
        df = sars.load_gala()
        assert df.loc[df["endemics"] == 0, "island"].tolist() == ["Daphne Minor"]
        endemic = df[df["endemics"] > 0].assign(species=lambda d: d["endemics"])
        assert sars.sar_power(endemic).converged

    def test_file_is_package_data_and_copies_are_fresh(self):
        assert importlib.resources.files("sars.data").joinpath("gala.csv").is_file()
        df = sars.load_gala()
        df.loc[0, "species"] = -1
        assert sars.load_gala().loc[0, "species"] == 58


class TestLoadDataset:
    def test_missing_file_names_it(self):
        with pytest.raises(FileNotFoundError, match=r"nosuch\.csv is not available.*reinstall"):
            _load_dataset("nosuch")

    def test_parse_error_is_not_reported_as_missing(self, monkeypatch):
        def broken(*_args, **_kwargs):
            raise TypeError("bad parse")

        monkeypatch.setattr(pd, "read_csv", broken)
        with pytest.raises(TypeError, match="bad parse"):
            sars.load_galap()

    def test_missing_data_package(self, monkeypatch):
        def no_package(_name):
            raise ModuleNotFoundError("No module named 'sars.data'")

        monkeypatch.setattr("importlib.resources.files", no_package)
        for loader in (
            sars.load_galap,
            sars.load_gala,
            sars.load_aegean,
            sars.load_aegean2,
            sars.load_niering,
        ):
            with pytest.raises(FileNotFoundError, match="reinstall"):
                loader()


class TestFromDf:
    def test_default_columns(self):
        df = pd.DataFrame({"area": [1.0, 2.0], "species": [10, 20]})
        result = sars.from_df(df)
        assert list(result.columns) == ["area", "species"]
        assert len(result) == 2

    def test_custom_columns(self):
        df = pd.DataFrame({"km2": [1.0, 5.0], "richness": [10, 30]})
        result = sars.from_df(df, area_col="km2", species_col="richness")
        assert list(result.columns) == ["area", "species"]
        assert result["area"].iloc[1] == 5.0
        assert result["species"].iloc[1] == 30

    def test_extra_columns_dropped(self):
        df = pd.DataFrame({"area": [1.0], "species": [10], "island": ["A"]})
        result = sars.from_df(df)
        assert list(result.columns) == ["area", "species"]

    def test_missing_area_col(self):
        df = pd.DataFrame({"x": [1.0], "species": [10]})
        with pytest.raises(KeyError, match="area"):
            sars.from_df(df)

    def test_missing_species_col(self):
        df = pd.DataFrame({"area": [1.0], "count": [10]})
        with pytest.raises(KeyError, match="species"):
            sars.from_df(df)

    def test_usable_with_sar_power(self):
        df = pd.DataFrame(
            {
                "A": [0.5, 1.0, 2.0, 5.0, 10.0],
                "S": [10, 15, 22, 40, 55],
            }
        )
        result = sars.from_df(df, area_col="A", species_col="S")
        fit = sars.sar_power(result)
        assert fit.converged


class TestFromCsv:
    def test_basic(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("area,species\n1.0,10\n2.0,20\n5.0,35\n")
        df = sars.from_csv(csv)
        assert list(df.columns) == ["area", "species"]
        assert len(df) == 3

    def test_custom_columns(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("island_area,richness\n1.0,10\n2.0,20\n")
        df = sars.from_csv(csv, area_col="island_area", species_col="richness")
        assert list(df.columns) == ["area", "species"]
        assert df["area"].iloc[0] == 1.0

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sars.from_csv(tmp_path / "nonexistent.csv")

    def test_missing_area_col(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("x,species\n1.0,10\n")
        with pytest.raises(KeyError, match="area"):
            sars.from_csv(csv)

    def test_missing_species_col(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("area,count\n1.0,10\n")
        with pytest.raises(KeyError, match="species"):
            sars.from_csv(csv)

    def test_usable_with_sar_power(self, tmp_path):
        """from_csv output should be directly usable with model functions."""
        csv = tmp_path / "data.csv"
        csv.write_text("area,species\n0.5,10\n1.0,15\n2.0,22\n5.0,40\n10.0,55\n")
        df = sars.from_csv(csv)
        fit = sars.sar_power(df)
        assert fit.converged


class TestFromPyinaturalist:
    def test_basic(self):
        obs = [
            {"area": 1.0, "species_count": 10},
            {"area": 5.0, "species_count": 30},
            {"area": 10.0, "species_count": 50},
        ]
        df = sars.from_pyinaturalist(obs)
        assert list(df.columns) == ["area", "species"]
        assert len(df) == 3

    def test_custom_keys(self):
        obs = [
            {"island_km2": 1.0, "n_species": 10},
            {"island_km2": 5.0, "n_species": 30},
        ]
        df = sars.from_pyinaturalist(obs, area_key="island_km2", species_key="n_species")
        assert df["area"].iloc[0] == 1.0
        assert df["species"].iloc[1] == 30

    def test_empty_list(self):
        df = sars.from_pyinaturalist([])
        assert list(df.columns) == ["area", "species"]
        assert len(df) == 0

    def test_missing_area_key(self):
        obs = [{"x": 1.0, "species_count": 10}]
        with pytest.raises(KeyError, match="area"):
            sars.from_pyinaturalist(obs)

    def test_missing_species_key(self):
        obs = [{"area": 1.0, "count": 10}]
        with pytest.raises(KeyError, match="species_count"):
            sars.from_pyinaturalist(obs)


class TestFromGeoDataFrame:
    @pytest.fixture()
    def simple_gdf(self):
        """Create a simple GeoDataFrame with square polygons."""
        try:
            import geopandas as gpd
            from shapely.geometry import box
        except ImportError:
            pytest.skip("geopandas/shapely not installed")
        # Squares with areas 1e6, 4e6, 9e6 m^2 = 1, 4, 9 km^2, at the middle of the UTM
        # zone (easting 500 km) and off the equator, as real projected data would be.
        x, y = 500_000, 4_000_000
        polys = [box(x, y, x + side, y + side) for side in (1000, 2000, 3000)]
        return gpd.GeoDataFrame(
            {"species": [10, 25, 40], "geometry": polys},
            crs="EPSG:32617",
        )

    def test_computed_area_km2(self, simple_gdf):
        df = sars.from_geodataframe(simple_gdf)
        assert list(df.columns) == ["area", "species"]
        assert len(df) == 3
        assert df["area"].iloc[0] == pytest.approx(1.0, abs=0.01)
        assert df["area"].iloc[1] == pytest.approx(4.0, abs=0.01)

    def test_computed_area_m2(self, simple_gdf):
        df = sars.from_geodataframe(simple_gdf, area_units="m2")
        assert df["area"].iloc[0] == pytest.approx(1e6, rel=0.01)

    def test_computed_area_ha(self, simple_gdf):
        df = sars.from_geodataframe(simple_gdf, area_units="ha")
        assert df["area"].iloc[0] == pytest.approx(100.0, abs=1.0)

    def test_precomputed_area(self, simple_gdf):
        simple_gdf["my_area"] = [2.0, 8.0, 18.0]
        df = sars.from_geodataframe(simple_gdf, area_col="my_area")
        assert df["area"].iloc[0] == 2.0

    def test_missing_species_col(self, simple_gdf):
        with pytest.raises(KeyError, match="richness"):
            sars.from_geodataframe(simple_gdf, species_col="richness")

    def test_missing_area_col(self, simple_gdf):
        with pytest.raises(KeyError, match="nonexistent"):
            sars.from_geodataframe(simple_gdf, area_col="nonexistent")

    def test_invalid_area_units(self, simple_gdf):
        with pytest.raises(ValueError, match="area_units"):
            sars.from_geodataframe(simple_gdf, area_units="acres")

    def test_crs_units_is_a_deprecated_alias(self, simple_gdf):
        with pytest.warns(FutureWarning, match="crs_units is deprecated.*area_units"):
            df = sars.from_geodataframe(simple_gdf, crs_units="ha")
        assert df["area"].iloc[0] == pytest.approx(100.0, abs=1.0)

    def test_area_units_is_the_fourth_positional_argument(self, simple_gdf):
        """Where crs_units was, so positional calls keep their meaning."""
        df = sars.from_geodataframe(simple_gdf, "species", None, "ha")
        assert df["area"].iloc[0] == pytest.approx(100.0, abs=1.0)

    @pytest.mark.parametrize("area_units", ["ha", "km2"])
    def test_area_units_and_crs_units_together_are_refused(self, simple_gdf, area_units):
        """Even an explicit "km2", the default, which a check against the default missed."""
        with pytest.raises(TypeError, match="not both"):
            sars.from_geodataframe(simple_gdf, area_units=area_units, crs_units="m2")

    def test_invalid_area_units_with_area_col(self, simple_gdf):
        """Ignored with area_col, but a misspelt unit is still reported."""
        simple_gdf["my_area"] = [1.0, 4.0, 9.0]
        with pytest.raises(ValueError, match="area_units 'acres'"):
            sars.from_geodataframe(simple_gdf, area_col="my_area", area_units="acres")

    @pytest.mark.parametrize("area_col", [None, "my_area"], ids=["computed", "column"])
    def test_keeps_the_geodataframe_index(self, simple_gdf, area_col):
        """Both paths keep the index, as from_df does: the column path used to
        return a fresh RangeIndex."""
        gdf = simple_gdf.set_index(pd.Index(["Crete", "Rhodes", "Kos"], name="island"))
        gdf["my_area"] = [1.0, 4.0, 9.0]
        df = sars.from_geodataframe(gdf, area_col=area_col)
        pd.testing.assert_index_equal(df.index, gdf.index)
        assert df.loc["Rhodes", "area"] == pytest.approx(4.0)
        assert df.loc["Kos", "species"] == 40

    def test_geographic_crs_is_refused(self, simple_gdf):
        """Longitude/latitude coordinates give areas in squared degrees, which used to be
        divided by 1e6 and returned as km²."""
        lonlat = simple_gdf.to_crs("EPSG:4326")
        with pytest.raises(ValueError, match="geographic.*to_crs"):
            sars.from_geodataframe(lonlat)

    def test_geographic_crs_with_area_col(self, simple_gdf):
        """The CRS doesn't matter when the areas come from a column."""
        lonlat = simple_gdf.to_crs("EPSG:4326")
        lonlat["my_area"] = [1.0, 4.0, 9.0]
        df = sars.from_geodataframe(lonlat, area_col="my_area")
        assert df["area"].tolist() == [1.0, 4.0, 9.0]

    def test_reprojected_to_equal_area(self, simple_gdf):
        """Reprojecting to an equal-area CRS recovers the areas. The squares sit on UTM's
        central meridian, where its scale factor of 0.9996 shrinks areas by 0.08%."""
        df = sars.from_geodataframe(simple_gdf.to_crs("EPSG:6933"))
        assert df["area"].tolist() == pytest.approx([1.0, 4.0, 9.0], rel=2e-3)
        assert df["area"].tolist() == pytest.approx(
            [a / 0.9996**2 for a in (1.0, 4.0, 9.0)], rel=1e-4
        )

    def test_crs_in_feet(self):
        """Areas come out in the unit asked for whatever the CRS's own length unit."""
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import box

        # North Carolina state plane, in US survey feet (1200/3937 m).
        foot = 1200 / 3937
        side = 1000 / foot  # a 1 km square
        gdf = gpd.GeoDataFrame(
            {"species": [10], "geometry": [box(2e6, 5e5, 2e6 + side, 5e5 + side)]},
            crs="EPSG:2264",
        )
        assert sars.from_geodataframe(gdf)["area"].iloc[0] == pytest.approx(1.0)
        assert sars.from_geodataframe(gdf, area_units="ha")["area"].iloc[0] == pytest.approx(100.0)

    def test_missing_crs_warns(self, simple_gdf):
        gpd = pytest.importorskip("geopandas")
        no_crs = gpd.GeoDataFrame(
            {"species": simple_gdf["species"]}, geometry=list(simple_gdf.geometry)
        )
        assert no_crs.crs is None
        with pytest.warns(UserWarning, match="no CRS"):
            df = sars.from_geodataframe(no_crs)
        assert df["area"].iloc[0] == pytest.approx(1.0)

    def test_missing_crs_with_degree_like_coordinates(self):
        """Longitude/latitude saved without a CRS must not come back as km² either."""
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import box

        lonlat = gpd.GeoDataFrame(
            {"species": [10, 20]}, geometry=[box(-91, -1, -90, 0), box(-90, 0, -89.5, 0.5)]
        )
        with pytest.raises(ValueError, match="no CRS.*longitude and latitude"):
            sars.from_geodataframe(lonlat)
        lonlat["my_area"] = [3.0, 1.0]
        assert sars.from_geodataframe(lonlat, area_col="my_area")["area"].tolist() == [3.0, 1.0]

    def test_unprojected_crs_is_refused(self, simple_gdf):
        """A geocentric CRS is not geographic, but its coordinates aren't on a map plane."""
        geocentric = simple_gdf.to_crs("EPSG:4978")
        with pytest.raises(ValueError, match="not projected"):
            sars.from_geodataframe(geocentric)

    def test_non_polygon_geometry(self, simple_gdf):
        """A point has area zero, which must not pass for an island."""
        points = simple_gdf.set_geometry(simple_gdf.centroid)
        with pytest.raises(ValueError, match=r"Rows \[0, 1, 2\] have no area.*Point"):
            sars.from_geodataframe(points)

    def test_degenerate_polygons(self, simple_gdf):
        """A bow-tie whose lobes cancel and a polygon with collinear vertices have area
        zero though their type is Polygon. The error names their rows."""
        from shapely.geometry import Polygon

        x, y = 500_000, 4_000_000
        simple_gdf.loc[0, "geometry"] = Polygon(
            [(x, y), (x + 1000, y + 1000), (x + 1000, y), (x, y + 1000)]
        )
        simple_gdf.loc[2, "geometry"] = Polygon([(x, y), (x + 500, y), (x + 1000, y)])
        with pytest.raises(ValueError, match=r"Rows \[0, 2\] have no area"):
            sars.from_geodataframe(simple_gdf)

    def test_missing_geometry(self, simple_gdf):
        simple_gdf.loc[1, "geometry"] = None
        with pytest.raises(ValueError, match="missing or empty"):
            sars.from_geodataframe(simple_gdf)

    def test_multipolygon(self, simple_gdf):
        """An island with an islet is one row; its area is the sum of the parts."""
        from shapely.geometry import MultiPolygon, box

        x, y = 500_000, 4_000_000
        simple_gdf.loc[0, "geometry"] = MultiPolygon(
            [box(x, y, x + 1000, y + 1000), box(x + 5000, y, x + 6000, y + 1000)]
        )
        df = sars.from_geodataframe(simple_gdf)
        assert df["area"].iloc[0] == pytest.approx(2.0)

    def test_import_error(self, monkeypatch):
        """Should raise ImportError if geopandas not available."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "geopandas":
                raise ImportError("No module named 'geopandas'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="geopandas"):
            sars.from_geodataframe(None)
