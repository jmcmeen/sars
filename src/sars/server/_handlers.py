"""What each endpoint runs in its worker process: the request's plain values
in, the operation (``sars._operations``) called, its JSON-ready result out.

Kept apart from the FastAPI app, so a worker imports only the library.
"""

from __future__ import annotations

from typing import Any

from sars import _operations as ops


def frame(data: dict[str, Any]) -> Any:
    """The data to fit: a bundled dataset (``{"dataset": name}``), or arrays
    (``{"area": ..., "species": ..., "label": ...}``)."""
    if "dataset" in data:
        return ops.bundled(data["dataset"])
    return ops.data_frame(data["area"], data["species"], data.get("label"))


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
    return ops.result(ops.cv(frame(data), label=label, **options), include_object=True)


def loss(data: dict[str, Any], **options: Any) -> dict[str, Any]:
    return ops.result(ops.loss(frame(data), **options), include_object=True)


def predict(result: dict[str, Any], areas: list[float]) -> dict[str, Any]:
    return ops.predict(ops.from_saved(result), areas)
