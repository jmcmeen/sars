"""The HTTP API: each request runs one of the library's operations and
returns its JSON envelope.

The operations (``sars._operations``) are the ones the command line runs; the
two front ends share them and nothing else. Each request runs its operation in
a fresh worker process (``sars.server._handlers``), with a time limit: the
request's values are passed as data, nothing is evaluated and no shell runs.
"""

from __future__ import annotations

import hmac
import importlib
import json
import multiprocessing
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

API_VERSION = "v1"
DEFAULT_TIMEOUT = 600.0  # seconds; a cold cross-validation of every model can take minutes
MAX_ROWS = 100_000
MAX_SCENARIOS = 1_000
MAX_MODELS = 50
MAX_JOBS = 64

IC = Literal["AICc", "AIC", "BIC"]
# The bundled datasets (sars._operations.DATASETS; a test keeps the two the same).
Dataset = Literal["galap", "gala", "aegean", "aegean2", "niering", "schrader", "xu"]
# A model name: letters and digits only, so it can't be read as a flag.
ModelName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9]*$", max_length=32)]
Models = Annotated[list[ModelName], Field(min_length=1, max_length=MAX_MODELS)]
Jobs = Annotated[int, Field(ge=-1, le=MAX_JOBS, description="processes; -1: the server's limit")]
Numbers = Annotated[list[float], Field(min_length=1, max_length=MAX_SCENARIOS)]


@dataclass(frozen=True)
class ServerConfig:
    """How the server runs requests.

    Attributes
    ----------
    token : str or None
        If set, every request but ``/v1/version`` needs the header
        ``Authorization: Bearer <token>``.
    timeout : float
        Seconds a request may run before it is stopped, with any worker
        processes it started.
    python : str
        The interpreter each request's worker process runs (this one by
        default).
    max_jobs : int
        The most processes one request's ``n_jobs`` may use (this machine's
        cores by default; -1 in a request means this many).
    max_concurrent : int
        How many requests run at once; further requests wait.
    """

    token: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    python: str = sys.executable
    max_jobs: int = field(default_factory=lambda: os.cpu_count() or 1)
    max_concurrent: int = 4


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class Data(BaseModel):
    """Species-area data: an area and a richness per site, and optionally a
    name per site (used to label results)."""

    area: list[float] = Field(min_length=1, max_length=MAX_ROWS)
    species: list[float] = Field(min_length=1, max_length=MAX_ROWS)
    label: list[str] | None = Field(default=None, max_length=MAX_ROWS)

    @model_validator(mode="after")
    def _same_length(self) -> Data:
        lengths = {len(self.area), len(self.species)} | (
            {len(self.label)} if self.label is not None else set()
        )
        if len(lengths) != 1:
            raise ValueError("area, species and label must have the same length")
        return self


class DataRequest(BaseModel):
    """The data to fit: inline arrays, or a bundled dataset's name."""

    data: Data | None = None
    dataset: Dataset | None = Field(
        default=None, description="a bundled dataset (GET /v1/datasets)"
    )

    @model_validator(mode="after")
    def _one_source(self) -> DataRequest:
        if (self.data is None) == (self.dataset is None):
            raise ValueError("give exactly one of data and dataset")
        return self


class Checks(BaseModel):
    normality: Literal["shapiro", "lillie", "kolmogorov"] | None = None
    homoscedasticity: Literal["cor.fitted", "cor.area"] | None = None
    neg_check: bool = False


class FitRequest(DataRequest, Checks):
    models: Models = Field(default=["power"], description='one model, several, or ["all"]')


class AverageRequest(DataRequest, Checks):
    models: Models = Field(default=["all"])
    ic: IC = "AICc"


class ThresholdRequest(DataRequest):
    models: Models | None = Field(default=None, description="default: ContOne, ZslopeOne, Linear")
    interval: float | None = None
    log_area: bool = True


class CVRequest(DataRequest):
    models: Models = Field(default=["all"])
    ic: IC = "AICc"
    warm: bool = False
    n_jobs: Jobs = 1


class LossRequest(DataRequest):
    area: Numbers = Field(description="the area before the loss")
    remaining: Numbers | None = None
    fraction_lost: Numbers | None = None
    models: Models = Field(default=["all"])
    ic: IC = "AICc"
    n_boot: int = Field(default=1000, ge=0, le=100_000)
    conf: float = Field(default=0.95, gt=0, lt=1)
    seed: int | None = None
    n_jobs: Jobs = 1

    @model_validator(mode="after")
    def _one_after(self) -> LossRequest:
        if (self.remaining is None) == (self.fraction_lost is None):
            raise ValueError("give exactly one of remaining and fraction_lost")
        return self


class PredictRequest(BaseModel):
    result: dict[str, Any] = Field(
        description="a result saved by --save, or an envelope's result.object"
    )
    areas: list[float] = Field(min_length=1, max_length=MAX_ROWS)


# ---------------------------------------------------------------------------
# Running a request
# ---------------------------------------------------------------------------


def _work(command: str, target: str, kwargs: dict[str, Any], conn: Any) -> None:
    """In the worker process: run ``target`` (a handler's name, or
    ``module:function``) and send back the envelope, as JSON."""
    if os.name == "posix":
        os.setsid()  # a process group of its own, with any workers it starts
    os.environ.pop("SARS_SERVER_TOKEN", None)  # the work never sees the token
    from sars import _operations as ops

    if ":" in target:
        module, name = target.split(":")
        function = getattr(importlib.import_module(module), name)
    else:
        from sars.server import _handlers

        function = getattr(_handlers, target)
    conn.send(ops.envelope_json(ops.envelope(command, lambda: function(**kwargs))))
    conn.close()


def _status(envelope: dict[str, Any]) -> int:
    """200 when the operation succeeded; 422 when the request was at fault (an
    unknown model, data it couldn't fit); 500 for any other failure (a bug)."""
    from sars._operations import INPUT_ERROR_NAMES

    if envelope.get("ok"):
        return 200
    return 422 if (envelope.get("error") or {}).get("type") in INPUT_ERROR_NAMES else 500


def _execute(
    config: ServerConfig, command: str, target: str, kwargs: dict[str, Any]
) -> JSONResponse:
    """Run one request's operation in a worker process and return its
    envelope (see ``_status``); 500 if the worker ended without one, 504 if it
    ran out of time, after which it and any worker processes it started are
    stopped."""
    context = multiprocessing.get_context("spawn")
    if config.python != sys.executable:
        context.set_executable(config.python)
    receive, send = context.Pipe(duplex=False)
    worker = context.Process(target=_work, args=(command, target, kwargs, send))
    worker.start()
    send.close()
    try:
        if not receive.poll(config.timeout):
            _stop(worker)
            raise HTTPException(
                504, f"The request took longer than {config.timeout:g} s and was stopped."
            )
        try:
            text = receive.recv()
        except EOFError:
            text = None
    finally:
        receive.close()
        worker.join()
    if text is None:
        raise HTTPException(
            500, f"The request's worker ended without a result (exit code {worker.exitcode})."
        )
    envelope = json.loads(text)
    return JSONResponse(envelope, status_code=_status(envelope))


def _stop(worker: Any) -> None:
    """Stop a worker process and the processes it started."""
    if os.name == "posix":
        try:
            os.killpg(worker.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):  # not yet in its own group
            pass
    worker.kill()


def _data(request: DataRequest) -> dict[str, Any]:
    """The request's data as plain values, for the worker."""
    if request.dataset is not None:
        return {"dataset": request.dataset}
    assert request.data is not None
    return {
        "area": [float(a) for a in request.data.area],
        "species": [float(s) for s in request.data.species],
        "label": request.data.label,
    }


def _checks(request: Checks) -> dict[str, Any]:
    return {
        "normality": request.normality,
        "homoscedasticity": request.homoscedasticity,
        "neg_check": request.neg_check,
    }


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """The FastAPI application."""
    config = config or ServerConfig()
    from sars import __version__

    slots = threading.BoundedSemaphore(config.max_concurrent)

    def jobs(n: int) -> int:
        """A request's n_jobs, within the server's limit (-1: the limit)."""
        return config.max_jobs if n == -1 else max(1, min(n, config.max_jobs))

    app = FastAPI(
        title="sars",
        version=API_VERSION,
        summary="Species-area relationship models over HTTP",
        description=(
            "Each endpoint runs one of the library's operations, the same ones the "
            "command line runs, and returns its JSON envelope: "
            "`{sars, command, ok, result | error, warnings}`. "
            "A request the operation refuses returns 422 with `ok: false`."
        ),
    )

    def authorise(authorization: Annotated[str | None, Header()] = None) -> None:
        if config.token is None:
            return
        expected = f"Bearer {config.token}".encode()
        # Compared as bytes: compare_digest refuses non-ASCII str, and a
        # header holds whatever bytes a client sends.
        given = (authorization or "").encode("latin-1", errors="replace")
        if not hmac.compare_digest(given, expected):
            raise HTTPException(401, "A valid bearer token is required.")

    guarded = [Depends(authorise)]

    def run(command: str, **kwargs: Any) -> JSONResponse:
        with slots:
            return _execute(config, command, command, kwargs)

    @app.get("/v1/version", tags=["service"])
    def version() -> dict[str, str]:
        """The sars version and the API version."""
        return {"sars": __version__, "api": API_VERSION}

    @app.get("/v1/datasets", tags=["service"], dependencies=guarded)
    def datasets() -> JSONResponse:
        """The bundled datasets (as `sars datasets`)."""
        return run("datasets")

    @app.post("/v1/fit", tags=["models"], dependencies=guarded)
    def fit(request: FitRequest) -> JSONResponse:
        """Fit one model, or several ranked by AICc (as `sars fit`)."""
        return run("fit", data=_data(request), models=request.models, **_checks(request))

    @app.post("/v1/average", tags=["models"], dependencies=guarded)
    def average(request: AverageRequest) -> JSONResponse:
        """The model average (as `sars average`)."""
        return run(
            "average",
            data=_data(request),
            models=request.models,
            ic=request.ic,
            **_checks(request),
        )

    @app.post("/v1/threshold", tags=["models"], dependencies=guarded)
    def threshold(request: ThresholdRequest) -> JSONResponse:
        """Piecewise (threshold) models (as `sars threshold`)."""
        return run(
            "threshold",
            data=_data(request),
            models=request.models,
            interval=request.interval,
            log_area=request.log_area,
        )

    @app.post("/v1/cv", tags=["models"], dependencies=guarded)
    def cv(request: CVRequest) -> JSONResponse:
        """Leave-one-island-out cross-validation (as `sars cv`)."""
        return run(
            "cv",
            data=_data(request),
            models=request.models,
            warm=request.warm,
            ic=request.ic,
            n_jobs=jobs(request.n_jobs),
        )

    @app.post("/v1/loss", tags=["models"], dependencies=guarded)
    def loss(request: LossRequest) -> JSONResponse:
        """Species expected to be lost with habitat (as `sars loss`)."""
        return run(
            "loss",
            data=_data(request),
            area=request.area,
            remaining=request.remaining,
            fraction_lost=request.fraction_lost,
            models=request.models,
            ic=request.ic,
            n_boot=request.n_boot,
            conf=request.conf,
            seed=request.seed,
            n_jobs=jobs(request.n_jobs),
        )

    @app.post("/v1/predict", tags=["models"], dependencies=guarded)
    def predict(request: PredictRequest) -> JSONResponse:
        """Predictions from a saved result (as `sars predict`)."""
        return run("predict", result=request.result, areas=request.areas)

    return app


def openapi_spec() -> dict[str, Any]:
    """The API's OpenAPI document (the same for every configuration)."""
    return create_app().openapi()


def write_openapi_spec(path: str | Path) -> None:
    """Write the OpenAPI document as JSON (as committed in docs/api/openapi.json)."""
    text = json.dumps(openapi_spec(), indent=2, sort_keys=True, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")
