"""What each HTTP endpoint runs in its worker process (``sars._server_worker``):
the request's plain values in, the operation (``sars._operations``) called,
its JSON-ready result out. Data are reduced to area and species as the command
line reduces them, with a name per site kept only where an operation uses it
(cross-validation), so a result's stored data match the command line's.
"""

from __future__ import annotations

from typing import Any

from sars import _operations as ops


def frame(data: dict[str, Any], labels: bool = False) -> Any:
    """The data to fit: a bundled dataset (``{"dataset": name}``), or arrays
    (``{"area": ..., "species": ..., "label": ...}``); with ``labels``, the
    names too."""
    import sars

    if "dataset" in data:
        table = ops.bundled(data["dataset"])
        out = sars.from_df(table)
        if labels:
            from sars._crossval import _LABEL_GUESSES

            lower = {str(c).lower(): c for c in table.columns}
            name = next((lower[g] for g in _LABEL_GUESSES if g in lower), None)
            if name is not None:
                out[name] = table[name].to_numpy()
        return out
    return ops.data_frame(data["area"], data["species"], data.get("label") if labels else None)


def datasets() -> dict[str, Any]:
    return ops.datasets(tables=True)


def table(data: dict[str, Any]) -> Any:
    """A table of named columns: a bundled table (``{"dataset": name}``), or
    ``{"columns": {name: values}}``."""
    import pandas as pd

    if "dataset" in data:
        return ops.bundled(data["dataset"])
    return pd.DataFrame({name: list(values) for name, values in data["columns"].items()})


def abundance(data: dict[str, Any]) -> tuple[Any, list[float] | None]:
    """A species-by-site table of counts and its sites' areas (None if none
    were sent): a bundled table with the areas that go with it, or
    ``{"sites", "counts", "species", "areas"}`` with a row of counts per
    species."""
    import pandas as pd

    if "dataset" in data:
        counts = ops.bundled(data["dataset"])
        areas = ops.bundled(ops.TABLES[data["dataset"]]["areas"])
        return counts, ops.site_areas([str(c) for c in counts.columns], areas)
    species = data.get("species") or [str(i + 1) for i in range(len(data["counts"]))]
    counts = pd.DataFrame(data["counts"], index=pd.Index(species, name="species"))
    counts.columns = list(data["sites"])
    return counts, data.get("areas")


def gdm(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.gdm(table(data), **options), include_object=True)


def habitat(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.habitat(table(data), **options), include_object=True)


def countryside(
    data: dict[str, Any],
    habitats: list[str] | None = None,
    species: list[str] | None = None,
    **options: Any,
) -> dict[str, Any]:
    defaults = ops.TABLES[data["dataset"]]["columns"] if "dataset" in data else {}
    obj = ops.countryside(
        table(data),
        habitats=habitats or defaults["habitats"],
        species=species or defaults["species"],
        **options,
    )
    return ops.result(obj, include_object=True)


def coleman(data: dict[str, Any], n_sim: int = 999, seed: int | None = None) -> dict[str, Any]:
    counts, areas = abundance(data)
    if areas is None:
        raise ops.InputError("Random placement needs the area of each site: send areas.")
    fit, test = ops.coleman(counts, areas, n_sim=n_sim, seed=seed)
    return ops.coleman_result(fit, test, include_object=True)


def effort(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    counts, areas = abundance(data)
    return ops.effort(counts, areas, **options)


def fit(
    data: dict[str, Any], params: bool = False, conf: float = 0.95, **options: Any
) -> dict[str, Any]:
    obj = ops.fit(frame(data), **options)
    result = ops.result(obj, include_object=True)
    if params:
        result["summary"]["parameters"] = ops.parameters(obj, conf=conf)
    return result


def average(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.average(frame(data), **options), include_object=True)


def threshold(
    data: dict[str, Any], ci: dict[str, Any] | None = None, **options: Any
) -> dict[str, Any]:
    obj = ops.threshold(frame(data), **options)
    result = ops.result(obj, include_object=True)
    if ci is not None:
        result["summary"]["ci"] = ops.summary(ops.breakpoint_ci(obj, **ci))
    return result


def cv(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    label = "label" if data.get("label") is not None else "auto"
    return ops.result(ops.cv(frame(data, labels=True), label=label, **options), include_object=True)


def loss(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.loss(frame(data), **options), include_object=True)


def many(datasets: dict[str, dict[str, Any]], **options: Any) -> dict[str, Any]:
    frames = {name: frame(data) for name, data in datasets.items()}
    return ops.result(ops.many(frames, **options), include_object=True)


def loglog(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.loglog(frame(data), **options), include_object=True)


def ci(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.ci(frame(data), **options), include_object=True)


def predict(
    result: dict[str, Any], areas: list[float], time: list[float] | None = None
) -> dict[str, Any]:
    return ops.predict(ops.from_saved(result), areas, time=time)
