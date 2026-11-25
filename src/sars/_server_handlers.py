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
    return ops.datasets()


def fit(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.fit(frame(data), **options), include_object=True)


def average(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.average(frame(data), **options), include_object=True)


def threshold(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.threshold(frame(data), **options), include_object=True)


def cv(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    label = "label" if data.get("label") is not None else "auto"
    return ops.result(ops.cv(frame(data, labels=True), label=label, **options), include_object=True)


def loss(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.loss(frame(data), **options), include_object=True)


def loglog(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.loglog(frame(data), **options), include_object=True)


def ci(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.ci(frame(data), **options), include_object=True)


def predict(result: dict[str, Any], areas: list[float]) -> dict[str, Any]:
    return ops.predict(ops.from_saved(result), areas)
