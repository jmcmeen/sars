"""Saving fits and other results as JSON, and loading them back."""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORMAT_VERSION = 1


def _result_types() -> dict[str, type]:
    """The result classes JSON may rebuild, by name. Nothing else is ever
    constructed from a file, so loading one can't run arbitrary code."""
    from sars._coleman import ColemanFit
    from sars._crossval import LOOCV
    from sars._gdm import GDMFit, GDMMultiFit
    from sars._habitat import CountrysideFit, HabitatFit
    from sars._inference import AveragedSAR, BootstrappedCI, MultiSARFit
    from sars._linpow import LinPowFit
    from sars._loss import HabitatLoss
    from sars._many import ManySARFit
    from sars._mixed import MixedLinPowFit
    from sars._models import SARFit
    from sars._threshold import ThresholdCI, ThresholdFit

    classes = [
        SARFit,
        MultiSARFit,
        AveragedSAR,
        BootstrappedCI,
        LinPowFit,
        ThresholdFit,
        ThresholdCI,
        ManySARFit,
        LOOCV,
        HabitatLoss,
        HabitatFit,
        CountrysideFit,
        MixedLinPowFit,
        GDMFit,
        GDMMultiFit,
        ColemanFit,
    ]
    return {cls.__name__: cls for cls in classes}


def to_json(obj: Any, indent: int | None = None) -> str:
    """A sars result as JSON text, for `from_json` to load again.

    Works for every result the package returns (`SARFit`, `MultiSARFit`,
    `AveragedSAR`, `BootstrappedCI`, `LinPowFit`, `ThresholdFit`,
    `ThresholdCI`, `ManySARFit`, `LOOCV`, `HabitatLoss`, `HabitatFit`,
    `CountrysideFit`, `MixedLinPowFit`, `GDMFit`, `GDMMultiFit`, `ColemanFit`), with
    its data, parameters and settings; but not `BayesLinPowFit`, whose posterior ArviZ saves. The
    text is standard JSON: NaN and infinite numbers (an AICc that is
    undefined, say) are written as strings and read back as numbers. It
    records the format version and the sars version that wrote it.

    Parameters
    ----------
    obj : a sars result
        The object to save.
    indent : int, optional
        Indent nested levels by this many spaces, for reading; None (the
        default) writes it compactly.

    Returns
    -------
    str

    Raises
    ------
    TypeError
        If ``obj`` isn't a sars result, or holds a value JSON can't store.

    Examples
    --------
    >>> fit = sars.sar_power(sars.load_galap())
    >>> again = sars.from_json(sars.to_json(fit))
    >>> again.params == fit.params
    True
    """
    from sars import __version__

    if type(obj) not in _result_types().values():
        raise TypeError(
            f"to_json saves sars results ({', '.join(sorted(_result_types()))}), "
            f"not {type(obj).__name__}."
        )
    document = {"sars_json": FORMAT_VERSION, "sars_version": __version__, "object": _encode(obj)}
    return json.dumps(document, indent=indent, allow_nan=False)


def from_json(text: str) -> Any:
    """Load a sars result saved with `to_json`.

    Parameters
    ----------
    text : str
        JSON written by `to_json`.

    Returns
    -------
    The result, of the type it was saved as.

    Raises
    ------
    ValueError
        If the text isn't JSON written by `to_json`, comes from a newer format
        than this version of sars reads, or names a type that isn't a sars
        result.
    """
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not JSON: {exc}") from exc
    if not isinstance(document, dict) or "sars_json" not in document:
        raise ValueError("Not a saved sars result: no 'sars_json' format marker.")
    version = document["sars_json"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"Not a saved sars result: format version {version!r}.")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"Saved with sars JSON format {version} (sars {document.get('sars_version')}); "
            f"this version of sars reads format {FORMAT_VERSION} and older. Upgrade sars."
        )
    if "object" not in document:
        raise ValueError("Not a saved sars result: no 'object'.")
    try:
        obj = _decode(document["object"])
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"Malformed saved sars result: {type(exc).__name__}: {exc}") from exc
    if type(obj) not in _result_types().values():
        raise ValueError(f"Not a saved sars result: the object is a {type(obj).__name__}.")
    return obj


def save(obj: Any, path: str | Path, indent: int | None = 2) -> None:
    """Save a sars result to a JSON file; `load` reads it back.

    Parameters
    ----------
    obj : a sars result
        See `to_json`.
    path : str or Path
        The file to write (overwritten if it exists).
    indent : int, optional
        Spaces per nested level (2 by default); None writes it compactly.
    """
    Path(path).write_text(to_json(obj, indent=indent), encoding="utf-8")


def load(path: str | Path) -> Any:
    """Load a sars result saved with `save` (or `to_json`).

    Parameters
    ----------
    path : str or Path
        The JSON file.

    Returns
    -------
    The result, of the type it was saved as.
    """
    return from_json(Path(path).read_text(encoding="utf-8"))


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _result_types().get(name) is not type(value):
            raise TypeError(f"Can't save a {name}: not a sars result.")
        fields = {
            f.name: _encode_field(value, getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.init
        }
        return {"__type__": name, "fields": fields}
    if isinstance(value, pd.DataFrame):
        return {"__frame__": _encode_frame(value)}
    if isinstance(value, pd.Series):
        # As a one-column frame, so it keeps its dtype and index; its name
        # (which may be None) goes alongside.
        return {
            "__series__": {
                "name": _encode(value.name),
                "frame": _encode_frame(value.to_frame(name="values")),
            }
        }
    if isinstance(value, np.ndarray):
        return {
            "__array__": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": _encode(value.ravel().tolist()),
            }
        }
    if isinstance(value, (np.generic,)):
        return _encode(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f"Can't save a dict with non-string keys: {list(value)[:3]}")
        return {"__dict__": {k: _encode(v) for k, v in value.items()}}
    raise TypeError(f"Can't save a value of type {type(value).__name__}.")


def _encode_field(owner: Any, value: Any) -> Any:
    """A field's value; a dict entry that is the object holding it (a GDM fit
    is one of its own candidates) is written as a reference back to it."""
    if isinstance(value, dict) and any(v is owner for v in value.values()):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f"Can't save a dict with non-string keys: {list(value)[:3]}")
        return {
            "__dict__": {
                k: {"__self__": True} if v is owner else _encode(v) for k, v in value.items()
            }
        }
    return _encode(value)


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    names = list(frame.columns)
    if len({repr(n) for n in names}) != len(names):
        raise TypeError(f"Can't save a DataFrame with repeated column names: {names}")
    index = frame.index
    default = isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1
    if default and index.name is None:
        encoded_index = None  # the default index, rebuilt from the length
    elif isinstance(index, pd.MultiIndex):
        encoded_index = {
            "names": _encode(list(index.names)),
            "levels": [
                _encode_values(index.get_level_values(i).to_series(index=None))
                for i in range(index.nlevels)
            ],
        }
    else:
        encoded_index = {
            "names": _encode([index.name]),
            "levels": [_encode_values(index.to_series(index=None))],
        }
    return {
        "columns": _encode(names),
        "data": [_encode_values(frame.iloc[:, i]) for i in range(frame.shape[1])],
        "length": len(frame),
        "index": encoded_index,
    }


def _encode_values(series: pd.Series) -> dict[str, Any]:
    """A column (or index level) as its dtype and JSON values: missing values
    as None, dates and durations as ISO strings, categories and their order."""
    dtype = series.dtype
    spec: dict[str, Any] = {"dtype": str(dtype)}
    values = series.tolist()
    if isinstance(dtype, pd.CategoricalDtype):
        spec["categories"] = _encode(_plain(list(dtype.categories)))
        spec["ordered"] = bool(dtype.ordered)
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        out = [None if _missing(v) else v.isoformat() for v in values]
    elif dtype.kind == "f" and not pd.api.types.is_extension_array_dtype(dtype):
        out = [float(v) for v in values]  # NaN stays NaN, written as a tagged float
    else:
        out = [None if _missing(v) else _plain(v) for v in values]
    spec["values"] = _encode(out)
    return spec


def _missing(value: Any) -> bool:
    """A missing scalar (None, NaN, pd.NA, NaT). A list (as in `ManySARFit`'s
    summary) is a value, whatever it holds."""
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    return bool(pd.isna(value))


def _plain(value: Any) -> Any:
    """A numpy or pandas scalar as the Python value it holds."""
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__type__" in value:
        types = _result_types()
        name = value["__type__"]
        if name not in types:
            raise ValueError(f"Saved object names type {name!r}, which isn't a sars result.")
        # A reference back to the object ({"__self__": true}) is read only as
        # a direct entry of one of its dict fields, beside objects of its own
        # type (a GDM fit among its candidates); anywhere else it is refused.
        fields: dict[str, Any] = {}
        back: list[tuple[dict[str, Any], str]] = []
        for field_name, raw in value["fields"].items():
            entries = raw.get("__dict__") if isinstance(raw, dict) else None
            if isinstance(entries, dict) and any(v == {"__self__": True} for v in entries.values()):
                decoded: dict[str, Any] = {}
                for key, item in entries.items():
                    if item == {"__self__": True}:
                        back.append((decoded, key))
                    else:
                        decoded[key] = _decode(item)
                        if type(decoded[key]) is not types[name]:
                            raise ValueError(
                                f"A reference back to a {name} sits beside a "
                                f"{type(decoded[key]).__name__} in {field_name!r}."
                            )
                    decoded.setdefault(key, None)
                fields[field_name] = decoded
            else:
                fields[field_name] = _decode(raw)
        obj = types[name](**fields)
        for holder, key in back:
            holder[key] = obj
        return obj
    if "__frame__" in value:
        return _decode_frame(value["__frame__"])
    if "__series__" in value:
        spec = value["__series__"]
        series = _decode_frame(spec["frame"])["values"]
        series.name = _decode(spec["name"])
        return series
    if "__array__" in value:
        spec = value["__array__"]
        flat = _decode(spec["data"])
        shape = tuple(spec["shape"])
        if spec["dtype"] == "object":
            # Element by element: np.array would turn equal-length lists into
            # a further dimension.
            out = np.empty(len(flat), dtype=object)
            for i, item in enumerate(flat):
                out[i] = item
            return out.reshape(shape)
        return np.array(flat, dtype=spec["dtype"]).reshape(shape)
    if "__float__" in value:
        return float(value["__float__"])
    if "__tuple__" in value:
        return tuple(_decode(v) for v in value["__tuple__"])
    if "__dict__" in value:
        return {k: _decode(v) for k, v in value["__dict__"].items()}
    raise ValueError(f"Unrecognised entry in saved sars JSON: keys {sorted(value)}.")


def _decode_frame(spec: dict[str, Any]) -> pd.DataFrame:
    names = _decode(spec["columns"])
    columns = [_decode_values(column) for column in spec["data"]]
    if spec.get("index") is not None:
        levels = [_decode_values(level) for level in spec["index"]["levels"]]
        index_names = _decode(spec["index"]["names"])
        if len(levels) > 1:
            index: pd.Index = pd.MultiIndex.from_arrays(levels, names=index_names)
        else:
            index = pd.Index(levels[0], name=index_names[0])
    else:
        index = pd.RangeIndex(spec["length"])
    # Built by position, then named: inserting by name would turn a None name into NaN.
    frame = pd.DataFrame(
        {position: column.array for position, column in enumerate(columns)}, index=index
    )
    frame.columns = pd.Index(names, dtype=object) if names else frame.columns
    if len(frame.columns) and all(isinstance(n, str) for n in names):
        frame.columns = pd.Index(names)  # the usual string column labels
    return frame


def _decode_values(spec: dict[str, Any]) -> pd.Series:
    values = _decode(spec["values"])
    dtype = spec["dtype"]
    if "categories" in spec:
        categories = _decode(spec["categories"])
        cat_dtype = pd.CategoricalDtype(categories, ordered=spec["ordered"])
        return pd.Series(pd.Categorical(values, dtype=cat_dtype))
    series = pd.Series(values, dtype=object if dtype == "object" else None)
    if str(series.dtype) == dtype:
        return series
    try:
        return series.astype(dtype)
    except (TypeError, ValueError) as exc:
        warnings.warn(
            f"Couldn't restore a column's dtype {dtype!r} ({exc}); it is loaded as {series.dtype}.",
            stacklevel=4,
        )
        return series
