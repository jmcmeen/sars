# HTTP server

The optional `server` extra serves `sars` over a small HTTP API, so programs in other
languages, notebooks on another machine, or a web page can use it without Python bindings.
It runs the same operations as the [command line](cli.md) (the two share them, and
neither goes through the other), so an endpoint and its command give the same result.

```console
$ pip install "sars[server]"
$ sars serve
INFO:     Uvicorn running on http://127.0.0.1:8765
```

Each endpoint runs one operation and returns the same JSON envelope its command prints
with `--json`: `{sars, command, ok, result | error, warnings}`. The envelope's
`result.object` is the whole result, in the format [`sars.save`](api/io.md) writes.

| Status | Meaning |
|---|---|
| 200 | The operation succeeded (`"ok": true`). |
| 422 with an envelope | The operation refused the request (`"ok": false`): an unknown model, data it couldn't fit, no model passing the checks. |
| 422 with `detail` | The request itself was invalid (a missing field, a value out of range); `detail` says which. |
| 401 | A token is required and wasn't given, or was wrong. |
| 500 | The operation failed in a way that points to a bug (its envelope says how), or its worker ended without a result. |
| 504 | The operation ran out of time and was stopped. |

| Endpoint | Does what this command does |
|---|---|
| `GET /v1/version` | The `sars` and API versions (no token needed) |
| `GET /v1/datasets` | `sars datasets`: the datasets, and the other bundled `tables` |
| `POST /v1/fit` | `sars fit` |
| `POST /v1/average` | `sars average` |
| `POST /v1/threshold` | `sars threshold` |
| `POST /v1/cv` | `sars cv` |
| `POST /v1/loss` | `sars loss` |
| `POST /v1/many` | `sars many` (`datasets`: name to data, or `bundled`: names) |
| `POST /v1/loglog` | `sars loglog` |
| `POST /v1/ci` | `sars ci` |
| `POST /v1/predict` | `sars predict`, from a saved result (`time`: island ages, for a general dynamic model) |
| `POST /v1/gdm` | `sars gdm` |
| `POST /v1/habitat` | `sars habitat` |
| `POST /v1/countryside` | `sars countryside` |
| `POST /v1/coleman` | `sars coleman` |
| `POST /v1/effort` | `sars effort` |

The request bodies are in the [OpenAPI document](api/openapi.json), which a running server
also serves at `/docs` and `/openapi.json`.

## Requests

Data go in as arrays, or as the name of a bundled dataset. The other fields are the
command's options:

```console
$ curl -s localhost:8765/v1/average -H 'Content-Type: application/json' -d '{
    "data": {"area": [0.5, 1, 5, 10, 50, 100, 500, 1000],
             "species": [5, 7, 12, 15, 24, 30, 48, 55]},
    "models": ["power", "loga"], "ic": "AICc"}'
{"sars": "0.12.0", "command": "average", "ok": true, "result": {"summary": {...}, ...}}
```

`data` can also carry `label`, a name per site, which `cv` uses to name the worst-predicted
island. `dataset` must be one of the bundled names. Requests are bounded: up to 100,000
sites, 50 model names, 1,000 loss scenarios and 100,000 bootstrap resamples. To predict from a saved fit, send it back with the areas:

```json
{"result": {"sars_json": 1, "sars_version": "...", "object": {...}}, "areas": [1, 100]}
```

### Other tables

The analyses that read more than area and richness take their own `data` (see the command
line's [Other tables](cli.md#other-tables)):

| Endpoint | `data` | Bundled `dataset` |
|---|---|---|
| `/v1/gdm` | `area`, `species` and `time` (island age), a value per island | none |
| `/v1/habitat` | `area`, `habitats` and `species`, a value per island | `habitat` |
| `/v1/countryside` | `habitats` and `species`: each a mapping of name to a value per site (the area of each habitat, the richness of each species group) | `countryside` |
| `/v1/coleman`, `/v1/effort` | `sites` (names), `counts` (a row per species, a count per site), and optionally `species` (names) and `areas` (a value per site, which `coleman` needs) | `xu_abundance`, with its sites' areas |

```console
$ curl -s localhost:8765/v1/coleman -H 'Content-Type: application/json' -d '{
    "data": {"sites": ["a", "b", "c"], "areas": [1, 4, 0.5],
             "species": ["wren", "robin", "finch"],
             "counts": [[10, 40, 3], [5, 20, 0], [1, 9, 2]]},
    "n_sim": 999, "seed": 1}'
```

Counts go in with species in rows only; the command line's other layouts are for files.
`effort` returns its table as `result.sites`, a row per site. An abundance table holds up
to 1,000,000 counts, and a countryside table up to 200 columns of each kind. To predict
from a saved general dynamic model, send the island ages as `time`: one per area, or one
for all.

## Running it safely

`sars serve` listens on 127.0.0.1 (port 8765) by default, so only this machine can reach
it. To use it from elsewhere, prefer an SSH tunnel (`ssh -L 8765:localhost:8765 host`) over
listening on a public address. If it must listen on the network (`--host 0.0.0.0`), give it
a token:

```console
$ openssl rand -hex 16 > token.txt
$ sars serve --host 0.0.0.0 --token-file token.txt
```

Every request but `/v1/version` then needs `Authorization: Bearer <token>`. The token can
also come from the `SARS_SERVER_TOKEN` environment variable.

Each request runs its operation in a fresh worker process, which gets the request's values
as data: no shell is involved, no file is written, and nothing in a request is run as code.
The worker doesn't see the server's token. An operation that runs longer than `--timeout`
seconds (600 by default) is stopped, with any worker processes it started, and the request
returns 504. Cold cross-validation of every model, or many bootstrap resamples, can take
minutes.

`sars --backend cupy serve` (or `ServerConfig(backend="cupy")`) runs every request's
start-grid screens on the GPU; the server checks CuPy works when it starts.

Four requests run at once; further requests wait their turn. A request's `n_jobs` is
capped at the machine's core count, and `-1` means that many.

## In Python

```python
from sars.server import ServerConfig, create_app

app = create_app(ServerConfig(token="...", timeout=120, max_concurrent=2, max_jobs=4))
# any ASGI server can run it
```
