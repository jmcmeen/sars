"""The HTTP API (sars[server])."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import sars  # noqa: E402
from sars.server import ServerConfig, create_app, openapi_spec  # noqa: E402

SPEC = Path(__file__).resolve().parents[1] / "docs" / "api" / "openapi.json"
INLINE = {
    "area": [0.5, 1, 5, 10, 50, 100, 500, 1000],
    "species": [5, 7, 12, 15, 24, 30, 48, 55],
    "label": ["a", "b", "c, with a comma", "d", "e", "f", "g", "h"],
}


def _inline_ci_lower():
    """bootstrap_ci on the inline data, as the /v1/ci test asks for it."""
    import numpy as np
    import pandas as pd

    data = sars.from_df(pd.DataFrame({"area": INLINE["area"], "species": INLINE["species"]}))
    ci = sars.bootstrap_ci(
        data,
        models=["power"],
        n_boot=20,
        area_grid=np.array([1.0, 100.0]),
        rng=np.random.default_rng(1),
    )
    return list(ci.lower)


def _inline_z_se():
    """lin_pow's slope standard error on the inline data, base-10 logs."""
    import pandas as pd

    data = pd.DataFrame({"area": INLINE["area"], "species": INLINE["species"]})
    return sars.lin_pow(data, log_base="10").z_se


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_version(client):
    assert client.get("/v1/version").json() == {"sars": sars.__version__, "api": "v1"}


def test_fit_a_bundled_dataset(client):
    response = client.post("/v1/fit", json={"dataset": "galap"})
    assert response.status_code == 200
    envelope = response.json()
    assert envelope["ok"] and envelope["command"] == "fit"
    params = envelope["result"]["summary"]["params"]
    assert params["z"] == pytest.approx(sars.sar_power(sars.load_galap()).params["z"])


@pytest.mark.parametrize(
    ("path", "body", "check"),
    [
        ("/v1/fit", {"models": ["power", "loga"]}, lambda s: s["type"] == "MultiSARFit"),
        ("/v1/average", {"models": ["power", "loga"], "ic": "BIC"}, lambda s: s["ic"] == "BIC"),
        ("/v1/threshold", {"models": ["ContOne", "Linear"]}, lambda s: s["type"] == "ThresholdFit"),
        (
            "/v1/cv",
            {"models": ["power", "loga"], "warm": True},
            lambda s: s["models"][0]["worst"] in INLINE["label"],
        ),
        ("/v1/loglog", {"log_base": "10"}, lambda s: s["z_se"] == pytest.approx(_inline_z_se())),
        (
            "/v1/ci",
            {"areas": [1, 100], "models": ["power"], "n_boot": 20, "seed": 1},
            lambda s: [r["lower"] for r in s["intervals"]] == pytest.approx(_inline_ci_lower()),
        ),
        (
            "/v1/loss",
            {"area": [800], "fraction_lost": [0.5], "models": ["power"], "n_boot": 20, "seed": 1},
            lambda s: s["scenarios"][0]["lost_lower"] < s["scenarios"][0]["species_lost"],
        ),
    ],
)
def test_each_command_with_inline_data(client, path, body, check):
    response = client.post(path, json={"data": INLINE, **body})
    assert response.status_code == 200, response.text
    assert check(response.json()["result"]["summary"])


def test_inline_data_match_the_python_api(client):
    response = client.post("/v1/fit", json={"data": INLINE})
    import pandas as pd

    expected = sars.sar_power(pd.DataFrame({"area": INLINE["area"], "species": INLINE["species"]}))
    assert response.json()["result"]["summary"]["params"]["c"] == pytest.approx(
        expected.params["c"]
    )


def test_predict_from_a_saved_result(client):
    saved = client.post("/v1/fit", json={"dataset": "galap"}).json()["result"]["object"]
    response = client.post("/v1/predict", json={"result": saved, "areas": [1, 100]})
    rows = response.json()["result"]["predictions"]
    expected = sars.sar_power(sars.load_galap()).predict([1.0, 100.0])
    assert [r["species"] for r in rows] == pytest.approx(list(expected))


def test_datasets(client):
    names = [d["name"] for d in client.get("/v1/datasets").json()["result"]["datasets"]]
    assert "galap" in names


def test_a_failed_command_is_422_with_its_envelope(client):
    response = client.post("/v1/fit", json={"dataset": "galap", "models": ["nope"]})
    assert response.status_code == 422
    assert response.json()["error"]["message"].startswith("Unknown model: 'nope'")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"dataset": "galap", "data": INLINE},
        {"data": {"area": [1, 2], "species": [1]}},
        {"dataset": "galap", "models": ["-x"]},
    ],
    ids=["no-data", "both", "ragged", "flag-like-model"],
)
def test_bad_requests_are_422(client, body):
    assert client.post("/v1/fit", json=body).status_code == 422


def test_a_token_is_required_when_set():
    client = TestClient(create_app(ServerConfig(token="s3cret")))
    assert client.get("/v1/version").status_code == 200
    assert client.post("/v1/fit", json={"dataset": "galap"}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/v1/fit", json={"dataset": "galap"}, headers=wrong).status_code == 401
    right = {"Authorization": "Bearer s3cret"}
    assert client.post("/v1/fit", json={"dataset": "galap"}, headers=right).status_code == 200


def test_a_command_that_runs_too_long_is_504():
    client = TestClient(create_app(ServerConfig(timeout=0.01)))
    assert client.post("/v1/fit", json={"dataset": "galap"}).status_code == 504


def _shape(spec):
    """The parts of an OpenAPI document that change only when the API does:
    each path's methods and the fields its request body takes. (The rest,
    such as how a nullable list is written, varies with fastapi and pydantic.)"""
    schemas = spec.get("components", {}).get("schemas", {})
    shape = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            ref = (
                op.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
                .get("$ref", "")
            )
            fields = sorted(schemas.get(ref.rsplit("/", 1)[-1], {}).get("properties", {}))
            shape[f"{method.upper()} {path}"] = fields
    return shape


def test_the_committed_openapi_spec_is_current():
    """Regenerate with: python -c "from sars.server import write_openapi_spec;
    write_openapi_spec('docs/api/openapi.json')"."""
    committed = json.loads(SPEC.read_text(encoding="utf-8"))
    assert _shape(committed) == _shape(openapi_spec())
    assert committed["info"]["version"] == "v1"


def test_the_dataset_names_match_the_command_line():
    from typing import get_args

    from sars._operations import DATASETS
    from sars.server.app import Dataset

    assert set(get_args(Dataset)) == set(DATASETS)


@pytest.mark.parametrize("dataset", ["/etc/passwd", "../data.csv", "-h"])
def test_only_bundled_datasets_are_accepted(client, dataset):
    assert client.post("/v1/fit", json={"dataset": dataset}).status_code == 422


def test_labels_with_quotes_commas_and_accents_reach_the_operation_intact():
    from sars._server_handlers import frame
    from sars.server.app import DataRequest, _data

    labels = ['O"Brien, North', "Española", "plain", "tab\there"]
    request = DataRequest(data={"area": [1, 2, 3, 4], "species": [1, 2, 3, 4], "label": labels})
    assert frame(_data(request), labels=True)["label"].tolist() == labels
    assert list(frame(_data(request)).columns) == ["area", "species"]


@pytest.mark.parametrize(("n_jobs", "status"), [(-1, 200), (1, 200), (10_000, 422), (-2, 422)])
def test_n_jobs_is_bounded(client, n_jobs, status):
    body = {"dataset": "galap", "models": ["power"], "warm": True, "n_jobs": n_jobs}
    assert client.post("/v1/cv", json=body).status_code == status


def test_too_many_scenarios_or_models_are_refused(client):
    many = {"dataset": "galap", "area": [10.0] * 1001, "fraction_lost": [0.5], "n_boot": 0}
    assert client.post("/v1/loss", json=many).status_code == 422
    assert (
        client.post("/v1/fit", json={"dataset": "galap", "models": ["power"] * 51}).status_code
        == 422
    )


def test_a_non_ascii_authorization_header_is_401_not_500():
    client = TestClient(create_app(ServerConfig(token="pässwort")))
    ok = {"Authorization": "Bearer pässwort".encode()}  # raw UTF-8 bytes on the wire
    bad = {"Authorization": "Bearer é".encode()}
    assert client.get("/v1/datasets", headers=bad).status_code == 401
    assert client.get("/v1/datasets", headers=ok).status_code == 200


@pytest.fixture
def stand_ins(monkeypatch):
    """Let the worker import tests.server_helpers (it runs with -P, so the
    repository isn't on its path otherwise)."""
    root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
    )
    return "tests.server_helpers"


@pytest.mark.parametrize(("kind", "status"), [("InputError", 422), ("TypeError", 500)])
def test_a_bug_in_the_operation_is_500_not_422(stand_ins, kind, status):
    from sars.server.app import _execute

    response = _execute(ServerConfig(), "fit", "raises", {"kind": kind}, handlers=stand_ins)
    assert response.status_code == status
    assert json.loads(response.body)["error"]["type"] == kind


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_a_timeout_stops_the_operations_worker_processes(stand_ins, tmp_path):
    """The operation's own workers (a stand-in sleeping child here) go too."""
    import time

    from fastapi import HTTPException

    from sars.server.app import _execute

    started, marker = tmp_path / "started", tmp_path / "survived"
    with pytest.raises(HTTPException) as stopped:
        _execute(
            ServerConfig(timeout=4.0),
            "fit",
            "sleeps_with_a_worker",
            {"started": str(started), "marker": str(marker)},
            handlers=stand_ins,
        )
    assert stopped.value.status_code == 504
    assert started.exists()  # the child process was running when time ran out
    time.sleep(6)  # past when the child would have left its marker (8 s in)
    assert not marker.exists()


def test_the_operation_does_not_see_the_token(stand_ins, monkeypatch):
    from sars.server.app import _execute

    monkeypatch.setenv("SARS_SERVER_TOKEN", "s3cret")
    response = _execute(ServerConfig(), "fit", "environment", {}, handlers=stand_ins)
    assert json.loads(response.body)["result"] == {"token": None}


def test_a_worker_that_dies_is_500(stand_ins):
    from fastapi import HTTPException

    from sars.server.app import _execute

    with pytest.raises(HTTPException) as failed:
        _execute(ServerConfig(), "fit", "dies", {}, handlers=stand_ins)
    assert failed.value.status_code == 500
    assert "exit code 3" in failed.value.detail


def test_a_result_json_cant_hold_is_500():
    from sars._operations import envelope_json
    from sars.server.app import _status

    text = envelope_json({"sars": "x", "command": "fit", "ok": True, "result": float("nan")})
    assert _status(json.loads(text)) == 500


def test_the_data_stored_in_a_result_match_the_command_line(client, capsys):
    """Only area and species, as the CLI stores them (gala has more columns)."""
    from sars.cli import main

    served = client.post("/v1/fit", json={"dataset": "gala"}).json()["result"]["object"]
    main(["fit", "gala", "--json"])
    printed = json.loads(capsys.readouterr().out)["result"]["object"]
    assert served == printed


def test_the_server_does_not_run_the_command_line():
    """The server and the command line share the operations; the server never
    goes through the command line (the CLI imports the server only to start
    it, in `sars serve`)."""
    import ast

    for file in sorted(
        (Path(__file__).resolve().parents[1] / "src" / "sars" / "server").glob("*.py")
    ):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m.startswith("sars.cli") for m in imported), file


def test_serve_refuses_a_timeout_that_is_not_positive(capsys):
    from sars.cli import main

    assert main(["serve", "--timeout", "0"]) == 2
    assert "--timeout must be a positive" in capsys.readouterr().err


def test_the_extra_is_named_when_missing(monkeypatch):
    import importlib.util

    from sars import _extras

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "uvicorn" else real(name, *a)
    )
    with pytest.raises(_extras.MissingExtraError, match=r"pip install 'sars\[server\]'"):
        _extras.require_extra("server", "sars.server")
