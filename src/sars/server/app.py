"""The HTTP API: each request runs one of the library's operations and
returns its JSON envelope.

The operations (``sars._operations``) are the ones the command line runs; the
two front ends share them and nothing else. Each request runs its operation in
a fresh worker process (``python -m sars._server_worker``, through
``sars._server_handlers``) with a time limit: the request's values reach it as
JSON data, nothing is evaluated and no shell runs.
"""

from __future__ import annotations

import hmac
import json
import os
import signal
import subprocess
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
# A column, site or species name: no space at either end, so two names can't
# differ only by one.
Name = Annotated[str, Field(pattern=r"^\S(.*\S)?$", max_length=100)]
Column = Annotated[list[float], Field(min_length=1, max_length=MAX_ROWS)]
# The most values an abundance or countryside table may hold (rows times
# columns), and the most habitats or species groups.
MAX_COUNTS = 1_000_000
MAX_COLUMNS = 50
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
        The interpreter that runs each request's worker,
        ``-m sars._server_worker`` (this one by default).
    max_jobs : int
        The most processes one request's ``n_jobs`` may use (this machine's
        cores by default; -1 in a request means this many).
    max_concurrent : int
        How many requests run at once; further requests wait.
    backend : {"numpy", "cupy"}
        The array library each request's start-grid screens run on (see
        ``sars.set_backend``); "cupy" is checked when the app is created.
    """

    token: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    python: str = sys.executable
    max_jobs: int = field(default_factory=lambda: os.cpu_count() or 1)
    max_concurrent: int = 4
    backend: str = "numpy"


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
    params: bool = Field(default=False, description="add each model's parameter table")
    conf: float = Field(default=0.95, gt=0, lt=1, description="the parameter tables' level")


class AverageRequest(DataRequest, Checks):
    models: Models = Field(default=["all"])
    ic: IC = "AICc"


class BreakpointCI(BaseModel):
    model: ModelName | None = Field(default=None, description="default: the chosen model")
    method: Literal["boot", "F"] = "boot"
    n_boot: int = Field(default=1000, ge=1, le=100_000)
    conf: float = Field(default=0.95, gt=0, lt=1)
    seed: int | None = None
    n_jobs: Jobs = 1


class ThresholdRequest(DataRequest):
    models: Models | None = Field(default=None, description="default: ContOne, ZslopeOne, Linear")
    interval: float | None = None
    log_area: bool = True
    ci: BreakpointCI | None = Field(default=None, description="intervals for the breakpoints")


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


DatasetName = Annotated[str, Field(min_length=1, max_length=100)]
MAX_DATASETS = 1_000


class ManyRequest(BaseModel):
    """Many datasets: inline arrays by name, or bundled datasets' names."""

    datasets: dict[DatasetName, Data] | None = Field(default=None, max_length=MAX_DATASETS)
    bundled: list[Dataset] | None = Field(default=None, max_length=MAX_DATASETS)
    models: Models = Field(default=["all"])
    ic: IC = "AICc"
    threshold: bool = True
    n_jobs: Jobs = 1

    @model_validator(mode="after")
    def _one_source(self) -> ManyRequest:
        given = [d for d in (self.datasets, self.bundled) if d is not None]
        if len(given) != 1:
            raise ValueError("give exactly one of datasets and bundled")
        if len(given[0]) < 2:
            raise ValueError("give two or more datasets")
        if self.bundled is not None and len(set(self.bundled)) != len(self.bundled):
            raise ValueError("give each bundled dataset once")
        return self


class LogLogRequest(DataRequest):
    log_base: Literal["e", "2", "10"] = "e"
    con: float = Field(default=1.0, description="added to every richness when an island has none")
    compare: bool = False


class CIRequest(DataRequest, Checks):
    areas: Numbers | None = Field(
        default=None, description="areas to predict at (default: ten over the data's range)"
    )
    models: Models = Field(default=["all"])
    ic: IC = "AICc"
    method: Literal["fast", "full"] = "fast"
    n_boot: int = Field(default=1000, ge=1, le=100_000)
    conf: float = Field(default=0.95, gt=0, lt=1)
    seed: int | None = None
    n_jobs: Jobs = 1


class GDMData(BaseModel):
    """An area, a richness and an age per island."""

    area: Column
    species: Column
    time: Column

    @model_validator(mode="after")
    def _same_length(self) -> GDMData:
        if len({len(self.area), len(self.species), len(self.time)}) != 1:
            raise ValueError("area, species and time must have the same length")
        return self


class GDMRequest(BaseModel):
    data: GDMData
    model: Literal["loga", "linear", "power_area", "power_area_time", "all"] = "loga"
    mod_sel: bool = Field(default=False, description="also compare with the simpler models")


class HabitatData(BaseModel):
    """An area, a number of habitats and a richness per island."""

    area: Column
    habitats: Column
    species: Column

    @model_validator(mode="after")
    def _same_length(self) -> HabitatData:
        if len({len(self.area), len(self.habitats), len(self.species)}) != 1:
            raise ValueError("area, habitats and species must have the same length")
        return self


class HabitatRequest(BaseModel):
    """The data to fit: inline arrays, or the bundled table."""

    data: HabitatData | None = None
    dataset: Literal["habitat"] | None = None
    mod_type: Literal["power_log", "power", "logarithmic"] = "power_log"
    con: float | None = Field(
        default=None, description="added to every richness before logs (power_log)"
    )
    log_base: Literal["e", "2", "10"] = "e"

    @model_validator(mode="after")
    def _one_source(self) -> HabitatRequest:
        if (self.data is None) == (self.dataset is None):
            raise ValueError("give exactly one of data and dataset")
        return self


class CountrysideData(BaseModel):
    """Per site: the area of each habitat and the richness of each species
    group, each by name."""

    habitats: dict[Name, Column] = Field(min_length=1, max_length=MAX_COLUMNS)
    species: dict[Name, Column] = Field(min_length=1, max_length=MAX_COLUMNS)

    @model_validator(mode="after")
    def _consistent(self) -> CountrysideData:
        if set(self.habitats) & set(self.species):
            raise ValueError("a habitat and a species group can't share a name")
        columns = [*self.habitats.values(), *self.species.values()]
        if len({len(v) for v in columns}) != 1:
            raise ValueError("every habitat and species group must have a value per site")
        if len(columns) * len(columns[0]) > MAX_COUNTS:
            raise ValueError(f"at most {MAX_COUNTS} values (sites times columns)")
        return self


class CountrysideRequest(BaseModel):
    """The data to fit: inline columns, or the bundled table."""

    data: CountrysideData | None = None
    dataset: Literal["countryside"] | None = None
    mod_type: Literal["power", "logarithmic"] = "power"
    grid_start: Literal["partial", "exhaustive", "none"] = "partial"
    z_lower: float = 0.0
    n_starts: int | None = Field(
        default=40, ge=1, le=100_000, description="starting values to fit from; null: all"
    )

    @model_validator(mode="after")
    def _one_source(self) -> CountrysideRequest:
        if (self.data is None) == (self.dataset is None):
            raise ValueError("give exactly one of data and dataset")
        return self


class Abundance(BaseModel):
    """Counts of individuals: a row per species, a value per site."""

    sites: list[Name] = Field(min_length=1, max_length=MAX_ROWS, description="the sites' names")
    counts: list[list[float]] = Field(
        min_length=1, max_length=MAX_ROWS, description="a row per species, a count per site"
    )
    species: list[Name] | None = Field(
        default=None, max_length=MAX_ROWS, description="the species' names (default: 1, 2, ...)"
    )
    areas: list[float] | None = Field(
        default=None, max_length=MAX_ROWS, description="each site's area, in the order of sites"
    )

    @model_validator(mode="after")
    def _consistent(self) -> Abundance:
        if len(set(self.sites)) != len(self.sites):
            raise ValueError("give each site once")
        if any(len(row) != len(self.sites) for row in self.counts):
            raise ValueError("every row of counts must have a value per site")
        if len(self.counts) * len(self.sites) > MAX_COUNTS:
            raise ValueError(f"at most {MAX_COUNTS} counts (species times sites)")
        if self.species is not None:
            if len(self.species) != len(self.counts):
                raise ValueError("species must name each row of counts")
            if len(set(self.species)) != len(self.species):
                raise ValueError("give each species once")
        if self.areas is not None and len(self.areas) != len(self.sites):
            raise ValueError("areas must have a value per site")
        return self


class AbundanceRequest(BaseModel):
    """The counts: inline, or the bundled table (with its sites' areas)."""

    data: Abundance | None = None
    dataset: Literal["xu_abundance"] | None = None

    @model_validator(mode="after")
    def _one_source(self) -> AbundanceRequest:
        if (self.data is None) == (self.dataset is None):
            raise ValueError("give exactly one of data and dataset")
        return self


class ColemanRequest(AbundanceRequest):
    n_sim: int = Field(default=999, ge=0, le=100_000, description="0: no simulation test")
    seed: int | None = Field(default=None, ge=0)


class EffortRequest(AbundanceRequest):
    base: Literal["coverage", "size"] = "coverage"
    level: float | None = Field(default=None, description="default: as iNEXT chooses")


class PredictRequest(BaseModel):
    result: dict[str, Any] = Field(
        description="a result saved by --save, or an envelope's result.object"
    )
    areas: list[float] = Field(min_length=1, max_length=MAX_ROWS)
    time: list[float] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ROWS,
        description="island ages, for a general dynamic model: one per area, or one for all",
    )


# ---------------------------------------------------------------------------
# Running a request
# ---------------------------------------------------------------------------


def _status(envelope: dict[str, Any]) -> int:
    """200 when the operation succeeded; 422 when the request was at fault (an
    unknown model, data it couldn't fit); 500 for any other failure (a bug)."""
    from sars._operations import INPUT_ERROR_NAMES

    if envelope.get("ok"):
        return 200
    return 422 if (envelope.get("error") or {}).get("type") in INPUT_ERROR_NAMES else 500


def _execute(
    config: ServerConfig,
    command: str,
    target: str,
    kwargs: dict[str, Any],
    handlers: str = "sars._server_handlers",
) -> JSONResponse:
    """Run one request's handler (``handlers.target``) in a worker process,
    ``python -m sars._server_worker``, and return its envelope (see
    ``_status``): 500 if the worker printed none, 504 if it ran out of time,
    after which it and any processes it started are stopped. ``handlers`` is
    chosen by the server (tests give stand-ins), never by a request."""
    request = json.dumps(
        {
            "command": command,
            "handlers": handlers,
            "target": target,
            "kwargs": kwargs,
            "backend": config.backend,
        }
    )
    # -P: the working directory isn't put on sys.path. The worker's own
    # environment: without the token.
    env = {k: v for k, v in os.environ.items() if k != "SARS_SERVER_TOKEN"}
    env["PYTHONWARNINGS"] = "default"
    posix = os.name == "posix"
    with subprocess.Popen(
        [config.python, "-P", "-m", "sars._server_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=posix,  # its own process group, with its workers
    ) as worker:
        try:
            stdout, stderr = worker.communicate(request, timeout=config.timeout)
        except subprocess.TimeoutExpired:
            _stop(worker, posix)
            worker.communicate()
            raise HTTPException(
                504, f"The request took longer than {config.timeout:g} s and was stopped."
            ) from None
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        tail = (stderr or stdout).strip().splitlines()[-1:] or ["no output"]
        raise HTTPException(
            500,
            f"The request's worker ended without a result (exit code {worker.returncode}): "
            f"{tail[0]}",
        ) from None
    return JSONResponse(envelope, status_code=_status(envelope))


def _stop(worker: subprocess.Popen[str], posix: bool) -> None:
    """Stop a worker process and the processes it started."""
    if posix:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            pass
    worker.kill()  # pragma: no cover - Windows, or already gone


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
    from sars._batched import _check_cupy

    if config.backend not in ("numpy", "cupy"):
        raise ValueError(f"backend must be numpy or cupy, got {config.backend!r}")
    if config.backend == "cupy":
        _check_cupy()  # fail now, not on every request

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
        """The bundled datasets and tables (as `sars datasets`)."""
        return run("datasets")

    @app.post("/v1/fit", tags=["models"], dependencies=guarded)
    def fit(request: FitRequest) -> JSONResponse:
        """Fit one model, or several ranked by AICc (as `sars fit`)."""
        return run(
            "fit",
            data=_data(request),
            models=request.models,
            params=request.params,
            conf=request.conf,
            **_checks(request),
        )

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
        ci = None
        if request.ci is not None:
            ci = request.ci.model_dump()
            ci["n_jobs"] = jobs(request.ci.n_jobs)
        return run(
            "threshold",
            data=_data(request),
            models=request.models,
            interval=request.interval,
            log_area=request.log_area,
            ci=ci,
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

    @app.post("/v1/many", tags=["models"], dependencies=guarded)
    def many(request: ManyRequest) -> JSONResponse:
        """The same models on many datasets, in one table (as `sars many`)."""
        datasets: dict[str, dict[str, Any]]
        if request.datasets is not None:
            datasets = {
                name: {"area": list(d.area), "species": list(d.species)}
                for name, d in request.datasets.items()
            }
        else:
            assert request.bundled is not None
            datasets = {name: {"dataset": name} for name in request.bundled}
        return run(
            "many",
            datasets=datasets,
            models=request.models,
            ic=request.ic,
            threshold=request.threshold,
            n_jobs=jobs(request.n_jobs),
        )

    @app.post("/v1/loglog", tags=["models"], dependencies=guarded)
    def loglog(request: LogLogRequest) -> JSONResponse:
        """The log-log power model, by least squares (as `sars loglog`)."""
        return run(
            "loglog",
            data=_data(request),
            log_base=request.log_base,
            con=request.con,
            compare=request.compare,
        )

    @app.post("/v1/ci", tags=["models"], dependencies=guarded)
    def ci(request: CIRequest) -> JSONResponse:
        """Bootstrap intervals on the model-averaged curve (as `sars ci`)."""
        return run(
            "ci",
            data=_data(request),
            areas=request.areas,
            models=request.models,
            ic=request.ic,
            method=request.method,
            n_boot=request.n_boot,
            conf=request.conf,
            seed=request.seed,
            n_jobs=jobs(request.n_jobs),
            **_checks(request),
        )

    @app.post("/v1/predict", tags=["models"], dependencies=guarded)
    def predict(request: PredictRequest) -> JSONResponse:
        """Predictions from a saved result (as `sars predict`)."""
        return run("predict", result=request.result, areas=request.areas, time=request.time)

    def source(request: Any, columns: dict[str, list[float]] | None) -> dict[str, Any]:
        """A handler's data: the bundled table's name, or the inline columns."""
        if request.dataset is not None:
            return {"dataset": request.dataset}
        return {"columns": columns}

    @app.post("/v1/gdm", tags=["other tables"], dependencies=guarded)
    def gdm(request: GDMRequest) -> JSONResponse:
        """The general dynamic model: area and island age (as `sars gdm`)."""
        return run(
            "gdm",
            data={"columns": request.data.model_dump()},
            model=request.model,
            mod_sel=request.mod_sel,
        )

    @app.post("/v1/habitat", tags=["other tables"], dependencies=guarded)
    def habitat(request: HabitatRequest) -> JSONResponse:
        """The habitat models: area and number of habitats (as `sars habitat`)."""
        return run(
            "habitat",
            data=source(request, request.data.model_dump() if request.data else None),
            mod_type=request.mod_type,
            con=request.con,
            log_base=request.log_base,
        )

    @app.post("/v1/countryside", tags=["other tables"], dependencies=guarded)
    def countryside(request: CountrysideRequest) -> JSONResponse:
        """The countryside model: the areas of several habitats (as `sars countryside`)."""
        columns = habitats = species = None
        if request.data is not None:
            columns = {**request.data.habitats, **request.data.species}
            habitats, species = list(request.data.habitats), list(request.data.species)
        return run(
            "countryside",
            data=source(request, columns),
            habitats=habitats,
            species=species,
            mod_type=request.mod_type,
            grid_start=request.grid_start,
            z_lower=request.z_lower,
            n_starts=request.n_starts,
        )

    def counts(request: AbundanceRequest) -> dict[str, Any]:
        if request.dataset is not None:
            return {"dataset": request.dataset}
        assert request.data is not None
        return request.data.model_dump()

    @app.post("/v1/coleman", tags=["other tables"], dependencies=guarded)
    def coleman(request: ColemanRequest) -> JSONResponse:
        """Coleman's random placement model, from counts (as `sars coleman`)."""
        return run("coleman", data=counts(request), n_sim=request.n_sim, seed=request.seed)

    @app.post("/v1/effort", tags=["other tables"], dependencies=guarded)
    def effort(request: EffortRequest) -> JSONResponse:
        """Richness standardized for sampling effort, from counts (as `sars effort`)."""
        return run("effort", data=counts(request), base=request.base, level=request.level)

    return app


def openapi_spec() -> dict[str, Any]:
    """The API's OpenAPI document (the same for every configuration)."""
    return create_app().openapi()


def write_openapi_spec(path: str | Path) -> None:
    """Write the OpenAPI document as JSON (as committed in docs/api/openapi.json)."""
    text = json.dumps(openapi_spec(), indent=2, sort_keys=True, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")
