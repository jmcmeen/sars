"""Saving results as JSON and loading them back."""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import sars
from sars._serialize import FORMAT_VERSION

FEW = ["power", "loga", "negexpo"]


@pytest.fixture(scope="module")
def results():
    galap = sars.load_galap()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        threshold = sars.sar_threshold(galap)
        return {
            "SARFit": sars.sar_power(galap),
            "MultiSARFit": sars.sar_multi(galap, models=FEW),
            "AveragedSAR": sars.sar_average(galap, models=FEW),
            "BootstrappedCI": sars.bootstrap_ci(
                galap, models=FEW, n_boot=8, rng=np.random.default_rng(0)
            ),
            "LinPowFit": sars.lin_pow(galap, compare=True),
            "ThresholdFit": threshold,
            "ThresholdCI": sars.threshold_ci(threshold, n_boot=20, rng=np.random.default_rng(0)),
            "ManySARFit": sars.sar_many(
                {"galap": galap, "xu": sars.load_xu()}, models=FEW, threshold=False
            ),
        }


def _assert_same(a, b):
    """Field by field, for dataclasses holding frames and arrays."""
    if hasattr(a, "__dataclass_fields__"):
        assert type(a) is type(b)
        for name in a.__dataclass_fields__:
            _assert_same(getattr(a, name), getattr(b, name))
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_same(x, y)
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    else:
        assert a == b


@pytest.mark.parametrize(
    "name",
    [
        "SARFit",
        "MultiSARFit",
        "AveragedSAR",
        "BootstrappedCI",
        "LinPowFit",
        "ThresholdFit",
        "ThresholdCI",
        "ManySARFit",
    ],
)
def test_round_trip(results, name):
    original = results[name]
    again = sars.from_json(sars.to_json(original))
    _assert_same(original, again)


def test_a_reloaded_fit_predicts_the_same(results):
    area = np.array([0.5, 10.0, 1000.0])
    for name in ("SARFit", "AveragedSAR", "ThresholdFit", "LinPowFit"):
        again = sars.from_json(sars.to_json(results[name]))
        np.testing.assert_array_equal(again.predict(area), results[name].predict(area))


def test_infinite_and_missing_numbers_survive_as_standard_json():
    """Four islands: AICc is infinite for every model."""
    tiny = sars.load_galap().head(4)
    fit = sars.sar_power(tiny)
    assert math.isinf(fit.aicc)
    text = sars.to_json(fit)

    def refuse(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    json.loads(text, parse_constant=refuse)  # no bare NaN or Infinity
    assert math.isinf(sars.from_json(text).aicc)


def test_the_document_records_its_versions(results):
    document = json.loads(sars.to_json(results["SARFit"]))
    assert document["sars_json"] == FORMAT_VERSION
    assert document["sars_version"] == sars.__version__


def test_save_and_load(results, tmp_path):
    path = tmp_path / "fit.json"
    sars.save(results["AveragedSAR"], path)
    assert json.loads(path.read_text())["object"]["__type__"] == "AveragedSAR"
    _assert_same(results["AveragedSAR"], sars.load(path))


def test_a_labelled_index_and_extra_columns_survive():
    gala = sars.load_gala().set_index("island")
    fit = sars.sar_power(gala)
    again = sars.from_json(sars.to_json(fit))
    pd.testing.assert_frame_equal(again.data, gala)


def test_a_newer_format_is_refused(results):
    document = json.loads(sars.to_json(results["SARFit"]))
    document["sars_json"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="Upgrade sars"):
        sars.from_json(json.dumps(document))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("not json", "Not JSON"),
        ("{}", "format marker"),
        (
            '{"sars_json": 1, "object": {"__type__": "os.system", "fields": {}}}',
            "isn't a sars result",
        ),
    ],
)
def test_bad_documents_are_refused(text, match):
    with pytest.raises(ValueError, match=match):
        sars.from_json(text)


def test_only_sars_results_can_be_saved():
    with pytest.raises(TypeError, match="saves sars results"):
        sars.to_json({"c": 1.0})


def _frame_round_trip(frame):
    from sars._serialize import _decode, _encode

    return _decode(json.loads(json.dumps(_encode(frame), allow_nan=False)))


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"x": pd.array([1, None], dtype="Int64")}),
        pd.DataFrame({"x": pd.array([1.5, None], dtype="Float64")}),
        pd.DataFrame({"d": pd.to_datetime(["2020-01-01", None])}),
        pd.DataFrame({"d": pd.to_datetime(["2020-01-01T05:00"]).tz_localize("UTC")}),
        pd.DataFrame({"t": pd.to_timedelta(["1 day", None])}),
        pd.DataFrame({"c": pd.Categorical(["b", None], categories=["c", "b", "a"], ordered=True)}),
        pd.DataFrame({"s": pd.array(["a", None], dtype="string"), "b": [True, False]}),
        pd.DataFrame({True: [1.0], None: [2.0]}),
        pd.DataFrame(
            {"v": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["k", "j"])
        ),
        pd.DataFrame({"v": [1]}, index=pd.CategoricalIndex(["x"], name="c")),
        pd.DataFrame({"v": pd.Series([], dtype=float)}, index=pd.Index([], dtype="int64")),
        pd.DataFrame({"v": [1.0]}, index=pd.DatetimeIndex(["2021-03-04"], name="when")),
        pd.DataFrame({"f": [1.0, np.nan, np.inf]}),
    ],
    ids=[
        "Int64",
        "Float64",
        "datetime",
        "datetime-tz",
        "timedelta",
        "categorical",
        "string-bool",
        "non-string-names",
        "multiindex",
        "categorical-index",
        "empty-index",
        "date-index",
        "nan-inf",
    ],
)
def test_frames_keep_their_dtypes_names_and_index(frame):
    pd.testing.assert_frame_equal(_frame_round_trip(frame), frame)


@pytest.mark.parametrize(
    "array",
    [np.empty((0, 1)), np.arange(6).reshape(2, 3), np.array([[1, 2], [3, 4]], dtype=object)],
    ids=["empty-2d", "2d", "object"],
)
def test_arrays_keep_their_shape(array):
    from sars._serialize import _decode, _encode

    again = _decode(_encode(array))
    assert again.shape == array.shape and again.dtype == array.dtype


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('{"sars_json": 1}', "no 'object'"),
        ('{"sars_json": true, "object": 1}', "format version"),
        ('{"sars_json": 1, "object": 5}', "the object is a int"),
        ('{"sars_json": 1, "object": {"__type__": "SARFit"}}', "Malformed"),
        ('{"sars_json": 1, "object": {"__type__": "SARFit", "fields": {"x": 1}}}', "Malformed"),
    ],
)
def test_malformed_documents_raise_value_error(text, match):
    with pytest.raises(ValueError, match=match):
        sars.from_json(text)


def test_a_look_alike_class_is_refused():
    import dataclasses

    @dataclasses.dataclass
    class LinPowFit:
        z: float

    with pytest.raises(TypeError, match="saves sars results"):
        sars.to_json(LinPowFit(0.3))


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1.5, np.nan], index=["a", "b"], name="c"),
        pd.Series([1, 2], name=None),
        pd.Series(pd.array([1, None], dtype="Int64"), index=pd.Index(["x", "y"], name="group")),
        pd.Series([1.0, 2.0]).rename_axis("g"),
    ],
    ids=["named-float", "unnamed-int", "nullable-with-index-name", "named-range-index"],
)
def test_series_round_trip(series):
    from sars._serialize import _decode, _encode

    again = _decode(json.loads(json.dumps(_encode(series), allow_nan=False)))
    pd.testing.assert_series_equal(again, series)


def _gdm_islands():
    """Islands whose richness has a hump in island age."""
    rng = np.random.default_rng(3)
    area = np.geomspace(2.0, 5000.0, 14)[rng.permutation(14)]
    time = np.linspace(0.5, 12.0, 14)
    species = 5 + 6 * np.log(area) + 9 * time - 0.7 * time**2 + rng.normal(0, 2, 14)
    return pd.DataFrame({"area": area, "species": species, "time": time})


def test_a_gdm_fit_round_trips_with_its_candidates():
    """A GDM fit is one of its own candidates: the reference back to it is
    saved as one, and restored."""
    fit = sars.sar_gdm(_gdm_islands(), mod_sel=True)
    again = sars.from_json(sars.to_json(fit))
    assert again.candidates["GDM"] is again
    assert list(again.candidates) == list(fit.candidates)
    assert again.candidates["A + T"].params == fit.candidates["A + T"].params
    assert again.candidates["A"].candidates is None
    pd.testing.assert_frame_equal(again.comparison, fit.comparison)
    np.testing.assert_array_equal(
        again.predict([10.0, 100.0], 3.0), fit.predict([10.0, 100.0], 3.0)
    )

    plain = sars.from_json(sars.to_json(sars.sar_gdm(_gdm_islands())))
    assert plain.candidates is None and plain.params == sars.sar_gdm(_gdm_islands()).params


def test_a_gdm_multi_fit_round_trips():
    multi = sars.sar_gdm(_gdm_islands(), model="all", mod_sel=True)
    again = sars.from_json(sars.to_json(multi))
    assert list(again.fits) == list(multi.fits)
    pd.testing.assert_frame_equal(again.summary, multi.summary)
    for name, fit in again.fits.items():
        assert fit.candidates["GDM"] is fit
        assert fit.params == multi.fits[name].params
        assert fit.candidates["A + T"].params == multi.fits[name].candidates["A + T"].params
    assert again.best.model == multi.best.model


@pytest.mark.parametrize(
    "tamper",
    [
        lambda fields: fields.__setitem__("candidates", {"__self__": True}),
        lambda fields: fields.__setitem__("n", {"__self__": True}),
        lambda fields: fields["params"]["__dict__"].__setitem__("Int", {"__self__": True}),
        lambda fields: fields["candidates"]["__dict__"].__setitem__("A", [{"__self__": True}]),
    ],
    ids=["as a field", "as a number", "among numbers", "in a list"],
)
def test_a_reference_back_is_read_only_among_candidates(tamper):
    import json

    document = json.loads(sars.to_json(sars.sar_gdm(_gdm_islands(), mod_sel=True)))
    tamper(document["object"]["fields"])
    with pytest.raises(ValueError):
        sars.from_json(json.dumps(document))


def test_a_coleman_fit_round_trips():
    fit = sars.coleman(sars.load_xu_abundance(), sars.load_xu()["area"].to_numpy())
    again = sars.from_json(sars.to_json(fit))
    pd.testing.assert_frame_equal(again.sites, fit.sites)
    pd.testing.assert_series_equal(again.abundance, fit.abundance)
    assert again.total_area == fit.total_area and again.rejected == fit.rejected
