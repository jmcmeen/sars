"""The HTTP API: each request runs the ``sars`` command line and returns its JSON.

The command line is the contract (see `sars.cli`): every endpoint builds the
argv of one ``sars`` command, runs ``python -m sars ... --json`` in a
subprocess with a time limit, and returns the envelope it printed. Nothing is
evaluated, no shell runs, and data reach the command as a CSV file in a fresh
temporary directory.
"""

from __future__ import annotations

import csv
import hmac
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
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
# The bundled datasets (sars.cli.DATASETS; a test keeps the two the same).
Dataset = Literal["galap", "gala", "aegean", "aegean2", "niering", "schrader", "xu"]
# A model name: letters and digits only, so it can't be read as a flag.
ModelName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9]*$", max_length=32)]
Models = Annotated[list[ModelName], Field(min_length=1, max_length=MAX_MODELS)]
Jobs = Annotated[int, Field(ge=-1, le=MAX_JOBS, description="processes; -1: the server's limit")]
Numbers = Annotated[list[float], Field(min_length=1, max_length=MAX_SCENARIOS)]
# The error types the command line reports for a bad request; any other means a bug.
_CLIENT_ERRORS = {"CLIError", "UsageError", "ValueError", "KeyError", "RuntimeError", "OSError"}


@dataclass(frozen=True)
class ServerConfig:
    """How the server runs commands.

    Attributes
    ----------
    token : str or None
        If set, every request but ``/v1/version`` needs the header
        ``Authorization: Bearer <token>``.
    timeout : float
        Seconds a command may run before it is stopped, with any worker
        processes it started.
    python : str
        The interpreter that runs ``-m sars`` (this one by default).
    max_jobs : int
        The most processes one request's ``n_jobs`` may use (this machine's
        cores by default; -1 in a request means this many).
    max_concurrent : int
        How many commands run at once; further requests wait.
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
# Running the command line
# ---------------------------------------------------------------------------


def _run(config: ServerConfig, argv: list[str], workdir: Path) -> JSONResponse:
    """Run ``sars <argv> --json`` and return its envelope: 200 when the
    command succeeded; 422 when the request was at fault (arguments it
    couldn't parse, an unknown model, data it couldn't fit); 500 when the
    command failed some other way or printed no envelope; 504 when it ran out
    of time, after which it and any worker processes it started are stopped."""
    # -P: the work directory, the child's cwd, isn't put on sys.path.
    command = [config.python, "-P", "-m", "sars", *argv, "--json"]
    env = {k: v for k, v in os.environ.items() if k != "SARS_SERVER_TOKEN"}
    env["PYTHONWARNINGS"] = "default"
    posix = os.name == "posix"
    with subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=posix,  # its own process group, with its workers
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=config.timeout)
        except subprocess.TimeoutExpired:
            if posix:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:  # pragma: no cover - Windows: the child only
                proc.kill()
            proc.communicate()
            raise HTTPException(
                504, f"The command took longer than {config.timeout:g} s and was stopped."
            ) from None
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        tail = (stderr or stdout).strip().splitlines()[-1:] or ["no output"]
        raise HTTPException(500, f"The command printed no envelope: {tail[0]}") from None
    if envelope.get("ok"):
        status = 200
    else:
        kind = (envelope.get("error") or {}).get("type")
        status = 422 if kind in _CLIENT_ERRORS else 500
    return JSONResponse(envelope, status_code=status)


def _data_args(request: DataRequest, workdir: Path, labels: bool = False) -> list[str]:
    """The DATA argument (a CSV written into ``workdir``, or a dataset name),
    and ``--label-col`` for a command that labels islands (``labels``)."""
    if request.dataset is not None:
        return [request.dataset]
    assert request.data is not None
    out = io.StringIO()
    writer = csv.writer(out)
    names = request.data.label
    writer.writerow(["area", "species", *(["label"] if names is not None else [])])
    for i, (a, sp) in enumerate(zip(request.data.area, request.data.species, strict=True)):
        writer.writerow(
            [repr(float(a)), repr(float(sp)), *([names[i]] if names is not None else [])]
        )
    (workdir / "data.csv").write_text(out.getvalue(), encoding="utf-8")
    args = ["data.csv"]
    if labels and request.data.label is not None:
        args += ["--label-col", "label"]
    return args


def _models(names: list[str] | None) -> list[str]:
    # -m is every command's models option; attached, so a name can't be read as a flag.
    return [f"-m{','.join(names)}"] if names else []


def _check_args(request: Checks) -> list[str]:
    args = []
    if request.normality:
        args.append(f"--normality={request.normality}")
    if request.homoscedasticity:
        args.append(f"--homoscedasticity={request.homoscedasticity}")
    if request.neg_check:
        args.append("--neg-check")
    return args


def _numbers(flag: str, values: list[float]) -> list[str]:
    return [flag, *(repr(float(v)) for v in values)]


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
            "Each endpoint runs one `sars` command (see the command line documentation) and "
            "returns its JSON envelope: `{sars, command, ok, result | error, warnings}`. "
            "A failed command returns 422 with `ok: false`."
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

    def run(argv_for: Any, request: Any) -> JSONResponse:
        with slots, tempfile.TemporaryDirectory(prefix="sars-server-") as tmp:
            workdir = Path(tmp)
            return _run(config, argv_for(request, workdir), workdir)

    @app.get("/v1/version", tags=["service"])
    def version() -> dict[str, str]:
        """The sars version and the API version."""
        return {"sars": __version__, "api": API_VERSION}

    @app.get("/v1/datasets", tags=["service"], dependencies=guarded)
    def datasets() -> JSONResponse:
        """The bundled datasets (`sars datasets`)."""
        return run(lambda _r, _w: ["datasets"], None)

    @app.post("/v1/fit", tags=["models"], dependencies=guarded)
    def fit(request: FitRequest) -> JSONResponse:
        """Fit one model, or several ranked by AICc (`sars fit`)."""
        return run(
            lambda r, w: ["fit", *_data_args(r, w), *_models(r.models), *_check_args(r)], request
        )

    @app.post("/v1/average", tags=["models"], dependencies=guarded)
    def average(request: AverageRequest) -> JSONResponse:
        """The model average (`sars average`)."""
        return run(
            lambda r, w: [
                "average",
                *_data_args(r, w),
                *_models(r.models),
                f"--ic={r.ic}",
                *_check_args(r),
            ],
            request,
        )

    @app.post("/v1/threshold", tags=["models"], dependencies=guarded)
    def threshold(request: ThresholdRequest) -> JSONResponse:
        """Piecewise (threshold) models (`sars threshold`)."""

        def argv(r: ThresholdRequest, w: Path) -> list[str]:
            args = ["threshold", *_data_args(r, w), *_models(r.models)]
            if r.interval is not None:
                args.append(f"--interval={r.interval!r}")
            if not r.log_area:
                args.append("--no-log-area")
            return args

        return run(argv, request)

    @app.post("/v1/cv", tags=["models"], dependencies=guarded)
    def cv(request: CVRequest) -> JSONResponse:
        """Leave-one-island-out cross-validation (`sars cv`)."""
        return run(
            lambda r, w: [
                "cv",
                *_data_args(r, w, labels=True),
                *_models(r.models),
                f"--ic={r.ic}",
                f"--n-jobs={jobs(r.n_jobs)}",
                *(["--warm"] if r.warm else []),
            ],
            request,
        )

    @app.post("/v1/loss", tags=["models"], dependencies=guarded)
    def loss(request: LossRequest) -> JSONResponse:
        """Species expected to be lost with habitat (`sars loss`)."""

        def argv(r: LossRequest, w: Path) -> list[str]:
            args = ["loss", *_data_args(r, w), *_numbers("--area", r.area)]
            if r.remaining is not None:
                args += _numbers("--remaining", r.remaining)
            else:
                assert r.fraction_lost is not None
                args += _numbers("--fraction-lost", r.fraction_lost)
            args += [*_models(r.models), f"--ic={r.ic}", f"--n-boot={r.n_boot}"]
            args += [f"--conf={r.conf!r}", f"--n-jobs={jobs(r.n_jobs)}"]
            if r.seed is not None:
                args.append(f"--seed={r.seed}")
            return args

        return run(argv, request)

    @app.post("/v1/predict", tags=["models"], dependencies=guarded)
    def predict(request: PredictRequest) -> JSONResponse:
        """Predictions from a saved result (`sars predict`)."""

        def argv(r: PredictRequest, w: Path) -> list[str]:
            (w / "result.json").write_text(json.dumps(r.result), encoding="utf-8")
            return ["predict", "result.json", *(repr(float(a)) for a in r.areas)]

        return run(argv, request)

    return app


def openapi_spec() -> dict[str, Any]:
    """The API's OpenAPI document (the same for every configuration)."""
    return create_app().openapi()


def write_openapi_spec(path: str | Path) -> None:
    """Write the OpenAPI document as JSON (as committed in docs/api/openapi.json)."""
    text = json.dumps(openapi_spec(), indent=2, sort_keys=True, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")
