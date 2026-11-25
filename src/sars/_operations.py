"""The work behind each command, shared by the command line and the HTTP API.

Each operation takes plain arguments (a data frame, model names, options) and
returns its result; ``result`` turns a result into JSON-ready values, and
``envelope`` runs an operation and wraps what it gives, the warnings it raised,
or its error, in the envelope both front ends return::

    {"sars": "<version>", "command": "fit", "ok": true,
     "result": {...}, "warnings": [...]}

or, on failure, ``"ok": false`` and ``"error": {"type": ..., "message": ...}``.

Neither front end imports the other: the command line reads its arguments and
files, calls an operation and prints; the server validates a request, calls
the same operation and responds.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

# The datasets bundled with the package, by the name the front ends take, and
# the function that loads each.
DATASETS = {
    "galap": "load_galap",
    "gala": "load_gala",
    "aegean": "load_aegean",
    "aegean2": "load_aegean2",
    "niering": "load_niering",
    "schrader": "load_schrader",
    "xu": "load_xu",
}

THRESHOLD_MODELS = ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo"]


# The most resample indices a bootstrap may hold at once (resamples times
# islands): 200 million, 1.6 GB, so a request can't exhaust the machine.
MAX_RESAMPLE_VALUES = 200_000_000


def _check_resamples(n_boot: int, n: int) -> None:
    if n_boot * n > MAX_RESAMPLE_VALUES:
        raise InputError(
            f"{n_boot} resamples of {n} islands is more than the {MAX_RESAMPLE_VALUES:,} "
            "resampled values a bootstrap here may hold; use fewer resamples."
        )


class InputError(Exception):
    """A problem with what an operation was given (a missing column, an
    unknown dataset, no model fitted): reported as an error, without a
    traceback."""


# Errors that mean the request was at fault (bad input, an unknown model, data
# that can't be fitted), not a bug; and their names, as an envelope reports
# them (with the command line's own UsageError, for arguments it can't parse).
INPUT_ERRORS: tuple[type[Exception], ...] = (
    InputError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)
INPUT_ERROR_NAMES = frozenset(
    {"InputError", "UsageError", "ValueError", "KeyError", "RuntimeError", "OSError"}
)


def version() -> str:
    try:
        from importlib.metadata import version as installed

        return installed("sars")
    except Exception:  # pragma: no cover - only in an uninstalled checkout
        return "0.0.0.dev0"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def bundled(name: str) -> Any:
    """A bundled dataset by its short name."""
    import sars

    if name not in DATASETS:
        raise InputError(f"No bundled dataset {name!r}; choose from {', '.join(DATASETS)}.")
    return getattr(sars, DATASETS[name])()


def data_frame(
    area: Sequence[float], species: Sequence[float], label: Sequence[str] | None = None
) -> Any:
    """Species-area data from arrays, with a name per site if given."""
    import pandas as pd

    import sars

    frame = sars.from_df(pd.DataFrame({"area": list(area), "species": list(species)}))
    if label is not None:
        frame["label"] = list(label)
    return frame


def model_names(values: Sequence[str] | None) -> list[str] | None:
    """Model names from repeated or comma-separated options."""
    if not values:
        return None
    names = [name.strip() for value in values for name in value.split(",") if name.strip()]
    if not names:
        raise InputError(f"No model names in {' '.join(repr(v) for v in values)}.")
    return names


def _models(names: Sequence[str] | None, default: str) -> str | list[str]:
    names = list(names) if names else [default]
    return "all" if names == ["all"] else names


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def datasets() -> dict[str, Any]:
    """The bundled datasets: name, size, area range and columns."""
    rows = []
    for name in DATASETS:
        frame = bundled(name)
        rows.append(
            {
                "name": name,
                "n": len(frame),
                "area_min": float(frame["area"].min()),
                "area_max": float(frame["area"].max()),
                "columns": [str(c) for c in frame.columns],
            }
        )
    return {"datasets": rows}


def fit(
    data: Any,
    models: Sequence[str] | None = None,
    normality: str | None = None,
    homoscedasticity: str | None = None,
    neg_check: bool = False,
) -> Any:
    """One model (power by default), or several (or "all") ranked by AICc."""
    import sars

    names = list(models) if models else ["power"]
    checks = normality or homoscedasticity or neg_check
    if len(names) == 1 and names[0] != "all" and not checks:
        from sars._models import _model_spec

        try:
            _model_spec(names[0])
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        return getattr(sars, f"sar_{names[0]}")(data)
    multi = sars.sar_multi(
        data,
        models=_models(names, "power"),
        normality=normality,
        homoscedasticity=homoscedasticity,
        neg_check=neg_check,
    )
    if multi.summary.empty:
        reasons = {**multi.excluded, **multi.not_fitted}
        raise InputError(
            "No model was fitted"
            + (" or passed the checks: " if reasons else ".")
            + "; ".join(f"{m}: {r}" for m, r in reasons.items())
        )
    return multi


def average(
    data: Any,
    models: Sequence[str] | None = None,
    ic: str = "AICc",
    normality: str | None = None,
    homoscedasticity: str | None = None,
    neg_check: bool = False,
) -> Any:
    """The model average, weighted by an information criterion."""
    import sars

    return sars.sar_average(
        data,
        models=_models(models, "all"),
        ic=ic,
        normality=normality,
        homoscedasticity=homoscedasticity,
        neg_check=neg_check,
    )


def threshold(
    data: Any,
    models: Sequence[str] | None = None,
    interval: float | None = None,
    log_area: bool = True,
) -> Any:
    """Piecewise (threshold) models, one chosen by AICc."""
    import sars

    return sars.sar_threshold(
        data, models=list(models) if models else None, interval=interval, log_area=log_area
    )


def cv(
    data: Any,
    models: Sequence[str] | None = None,
    warm: bool = False,
    ic: str = "AICc",
    label: str = "auto",
    n_jobs: int = 1,
) -> Any:
    """Leave-one-island-out cross-validation."""
    import sars

    return sars.loo_cv(
        data, models=_models(models, "all"), warm_start=warm, ic=ic, label=label, n_jobs=n_jobs
    )


def loss(
    data: Any,
    area: Sequence[float],
    remaining: Sequence[float] | None = None,
    fraction_lost: Sequence[float] | None = None,
    models: Sequence[str] | None = None,
    ic: str = "AICc",
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int | None = None,
    n_jobs: int = 1,
) -> Any:
    """Species expected to be lost with habitat, with bootstrap intervals."""
    import numpy as np

    import sars

    _check_resamples(n_boot, len(data))

    return sars.habitat_loss(
        data,
        area=list(area),
        remaining=None if remaining is None else list(remaining),
        fraction_lost=None if fraction_lost is None else list(fraction_lost),
        models=_models(models, "all"),
        ic=ic,
        n_boot=n_boot,
        conf=conf,
        rng=np.random.default_rng(seed),
        n_jobs=n_jobs,
    )


def ci(
    data: Any,
    areas: Sequence[float] | None = None,
    models: Sequence[str] | None = None,
    ic: str = "AICc",
    method: str = "fast",
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int | None = None,
    n_jobs: int = 1,
    normality: str | None = None,
    homoscedasticity: str | None = None,
    neg_check: bool = False,
) -> Any:
    """Bootstrap intervals on the model-averaged curve at ``areas`` (by
    default ten areas spread evenly in log over the data's range)."""
    import numpy as np

    import sars
    from sars._models import _validate_data

    if n_boot < 1:
        raise InputError(f"The number of resamples must be at least 1, got {n_boot}.")
    if not 0 < conf < 1:
        raise InputError(f"The interval level must be between 0 and 1, got {conf}.")
    _validate_data(data)  # before the default areas are taken from it
    _check_resamples(n_boot, len(data))
    if areas is None:
        at = np.geomspace(float(data["area"].min()), float(data["area"].max()), 10)
    else:
        at = np.asarray(areas, dtype=float)
        if not (np.isfinite(at).all() and (at > 0).all()):
            raise InputError("Areas must be positive, finite numbers.")
    return sars.bootstrap_ci(
        data,
        models=_models(models, "all"),
        n_boot=n_boot,
        conf=conf,
        area_grid=at,
        rng=np.random.default_rng(seed),
        method=method,
        ic=ic,
        normality=normality,
        homoscedasticity=homoscedasticity,
        neg_check=neg_check,
        n_jobs=n_jobs,
    )


def loglog(data: Any, log_base: str = "e", con: float = 1.0, compare: bool = False) -> Any:
    """The log-log power model, log S = log c + z log A, by least squares."""
    import sars

    return sars.lin_pow(data, con=con, log_base=log_base, compare=compare)


def predict(obj: Any, areas: Sequence[float]) -> dict[str, Any]:
    """Predicted richness at ``areas`` from a fitted result."""
    import numpy as np

    if not hasattr(obj, "predict"):
        raise InputError(
            f"A saved {type(obj).__name__} doesn't predict; save a single fit, an "
            "average or a threshold fit instead."
        )
    at = np.asarray(areas, dtype=float)
    if not np.isfinite(at).all():
        raise InputError("Areas must be finite numbers.")
    species = np.asarray(obj.predict(at), dtype=float)
    rows = [{"area": float(a), "species": number(s)} for a, s in zip(at, species, strict=True)]
    return {"type": type(obj).__name__, "predictions": rows}


def from_saved(saved: Any) -> Any:
    """A result from what ``sars.save`` wrote, as parsed JSON."""
    import sars

    try:
        return sars.from_json(json.dumps(saved))
    except (ValueError, KeyError, TypeError) as exc:
        raise InputError(f"Not a saved sars result: {exc}") from exc


# ---------------------------------------------------------------------------
# Results as JSON
# ---------------------------------------------------------------------------


def result(obj: Any, include_object: bool, saved: str | None = None) -> dict[str, Any]:
    """A result for the envelope: its summary, where it was saved, and with
    ``include_object`` the whole result in ``sars.save``'s format."""
    import sars

    out: dict[str, Any] = {"summary": summary(obj), "saved": saved}
    if include_object:
        out["object"] = json.loads(sars.to_json(obj))
    return out


def number(value: Any) -> Any:
    """A JSON-safe number: None where it's undefined or infinite (an AICc at
    small n), so the envelope stays standard JSON."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def records(frame: Any) -> list[dict[str, Any]]:
    """A table's rows as plain JSON values: whole numbers stay integers,
    undefined numbers and missing values become None."""
    return [
        {key: _cell_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _cell_value(value: Any) -> Any:
    import numpy as np
    import pandas as pd

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return number(value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    return value


def summary(obj: Any) -> dict[str, Any]:
    """The readable part of a result, as plain JSON values."""
    kind = type(obj).__name__
    if kind == "SARFit":
        return {
            "type": kind,
            "model": obj.model,
            "params": {k: number(v) for k, v in obj.params.items()},
            "R2": number(obj.r_squared),
            "AIC": number(obj.aic),
            "AICc": number(obj.aicc),
            "BIC": number(obj.bic),
            "n": obj.n,
            "converged": bool(obj.converged),
        }
    if kind == "MultiSARFit":
        return {
            "type": kind,
            "best": obj.summary.iloc[0]["model"] if len(obj.summary) else None,
            "models": records(obj.summary),
            "excluded": dict(obj.excluded),
            "not_fitted": dict(obj.not_fitted),
        }
    if kind == "AveragedSAR":
        return {
            "type": kind,
            "ic": obj.ic,
            "weights": {k: number(v) for k, v in obj.weights.items()},
            "models": records(_averaged_table(obj)),
            "excluded": dict(obj.multi.excluded),
            "not_fitted": dict(obj.multi.not_fitted),
        }
    if kind == "ThresholdFit":
        return {
            "type": kind,
            "best": obj.best_model,
            "breakpoints": [number(b) for b in obj.best_breakpoints],
            "log_area": obj.log_area,
            "segments": [{k: number(v) for k, v in s.items()} for s in obj.best_segments],
            "models": records(obj.summary),
        }
    if kind == "HabitatLoss":
        return {
            "type": kind,
            "conf": obj.conf,
            "ic": obj.average.ic,
            "scenarios": records(obj.table),
        }
    if kind == "LinPowFit":
        names = ("logc", "c", "z", "logc_se", "c_se", "z_se", "logc_t", "z_t", "logc_p", "z_p")
        out = {"type": kind, **{name: number(getattr(obj, name)) for name in names}}
        out.update(
            R2=number(obj.r_squared),
            adj_R2=number(obj.adj_r_squared),
            sigma=number(obj.sigma),
            n=obj.n,
            log_base=obj.log_base,
            con=number(obj.con),
            power=None if obj.power is None else summary(obj.power),
        )
        return out
    if kind == "BootstrappedCI":
        rows = [
            {
                "area": number(a),
                "mean": number(m),
                "median": number(md),
                "lower": number(lo),
                "upper": number(hi),
            }
            for a, m, md, lo, hi in zip(
                obj.area_grid, obj.mean, obj.median, obj.lower, obj.upper, strict=True
            )
        ]
        return {
            "type": kind,
            "conf": obj.conf,
            "n_boot": obj.n_boot,
            "n_models_attempted": obj.n_models_attempted,
            "dropped": dict(obj.dropped),
            "intervals": rows,
        }
    if kind == "LOOCV":
        return {
            "type": kind,
            "best": obj.summary.iloc[0]["model"],
            "warm_start": obj.warm_start,
            "models": records(obj.summary),
            "not_fitted": dict(obj.multi.not_fitted),
        }
    raise InputError(f"Can't summarise a {kind}.")  # pragma: no cover


def _averaged_table(obj: Any) -> Any:
    """The fitted models ranked by the average's criterion, with its
    differences and weights (``multi.summary`` is always by AICc)."""
    table = obj.multi.summary[["model", "R2", "AIC", "AICc", "BIC"]].copy()
    table[f"delta_{obj.ic}"] = table[obj.ic] - table[obj.ic].min()
    table["weight"] = table["model"].map(obj.weights).astype(float)
    return table.sort_values(obj.ic, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def envelope(
    command: str | None, call: Callable[[], dict[str, Any]], reraise_bugs: bool = False
) -> dict[str, Any]:
    """Run ``call`` (an operation, returning its JSON-ready result) and wrap
    its outcome: the result, or the error it raised, and every distinct
    warning. An error that isn't the request's fault (a bug) is re-raised with
    ``reraise_bugs``, to keep its traceback."""
    out: dict[str, Any] = {"sars": version(), "command": command}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
        except Exception as exc:
            if reraise_bugs and not isinstance(exc, INPUT_ERRORS):
                raise
            message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
            out.update(ok=False, error={"type": type(exc).__name__, "message": message})
        else:
            out.update(ok=True, result=value)
    seen: list[str] = []
    for warning in caught:
        text = str(warning.message)
        if text not in seen:
            seen.append(text)
    out["warnings"] = seen
    return out


def envelope_json(env: dict[str, Any]) -> str:
    """The envelope as standard JSON; a value JSON can't hold (a NaN that
    slipped through) is reported as the envelope's error instead."""
    try:
        return json.dumps(env, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:  # a bug, not the request's fault
        failed = {key: env.get(key) for key in ("sars", "command", "warnings")}
        message = f"The result can't be written as JSON: {exc}"
        failed.update(ok=False, error={"type": "SerializationError", "message": message})
        return json.dumps(failed, indent=2)
