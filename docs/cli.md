# Command line

Installing `sars` also installs a `sars` command (`python -m sars` runs the same thing).
It needs no extra dependencies.

```console
$ sars fit galap
power: c = 33.1803  z = 0.283176
R2 = 0.491247  AIC = 187.031  AICc = 189.031  BIC = 189.349  n = 16
```

## Data

Every fitting command takes its data as the first argument:

- a CSV file, with columns `area` and `species`, or others named with `--area-col` and
  `--species-col`;
- `-` to read the CSV from standard input;
- the name of a bundled dataset (`sars datasets` lists them).

```console
$ sars fit islands.csv --area-col Area_km2 --species-col Richness
$ sars datasets xu > xu.csv
```

## Commands

| Command | Does |
|---|---|
| `sars fit DATA [-m MODEL ...]` | Fits one model (`power` by default). With several models (`-m power,loga -m koba`) or `-m all`, fits each and ranks them by AICc. |
| `sars average DATA [-m MODEL ...] [--ic AICc\|AIC\|BIC]` | Averages the models (all by default), weighted by the criterion. |
| `sars threshold DATA [-m MODEL ...] [--interval X] [--no-log-area]` | Fits piecewise models (ContOne, ZslopeOne and Linear by default) and picks one by AICc. |
| `sars cv DATA [-m MODEL ...] [--ic AICc\|AIC\|BIC] [--warm] [--label-col COL] [--n-jobs N]` | Ranks the models by leave-one-island-out cross-validation (see [`loo_cv`](api/inference.md)). |
| `sars loss DATA --area A ... (--remaining R ... \| --fraction-lost F ...) [-m MODEL ...] [--n-boot N] [--seed S]` | Species expected to be lost with habitat, with bootstrap intervals (see [`habitat_loss`](api/inference.md)). |
| `sars ci DATA [--area A ...] [-m MODEL ...] [--method fast\|full] [--n-boot N] [--seed S] [--n-jobs N]` | The model-averaged richness at the areas (ten over the data's range by default), with bootstrap intervals (see [`bootstrap_ci`](api/inference.md)). |
| `sars predict FILE AREA ...` | Predicts richness at the areas from a result saved with `--save`. |
| `sars serve [--host H] [--port P] [--token-file F] [--timeout S]` | Serves the same operations over HTTP; needs the `server` extra (see [HTTP server](server.md)). |
| `sars datasets [NAME]` | Lists the bundled datasets, or writes one as CSV. |

`fit`, `average` and `ci` also take the residual checks of [`sar_multi`](api/inference.md):
`--normality shapiro|lillie|kolmogorov`, `--homoscedasticity cor.fitted|cor.area` and
`--neg-check`. `sars COMMAND --help` lists every option.

## Saving and predicting

`--save FILE` writes the result in the JSON format of [`sars.save`](api/io.md), which
`sars.load` reads in Python and `sars predict` reads on the command line:

```console
$ sars average galap --save galap.json
$ sars predict galap.json 1 10 100
1	10.9059
10	42.6951
100	148.098
```

## JSON output

`--json` prints one JSON object, the same for every command:

```json
{
  "sars": "0.12.0",
  "command": "fit",
  "ok": true,
  "result": {
    "summary": {"type": "SARFit", "model": "power", "params": {"c": 33.18, "z": 0.2832}, "...": "..."},
    "saved": null,
    "object": {"sars_json": 1, "...": "..."}
  },
  "warnings": []
}
```

- `result.summary` holds the readable values: parameters and criteria for one fit, a
  ranked table (`models`) for several, the weights for an average.
- `result.object` is the whole result as `sars.save` writes it, so `sars.from_json` can
  rebuild it.
- `warnings` lists the warnings the command raised (a model that didn't converge, one
  left out by a residual check). Without `--json` they go to standard error.
- A number that isn't defined (AICc with too few islands) is `null` in the summary.

On failure, such as a missing file, an unknown model, a missing column or no model
passing the checks asked for, the command exits with status 1. The object has
`"ok": false` and an `error` with a `type` and a `message`. A command line that can't be
parsed (an unknown option, a bad choice) exits with status 2, with `"type": "UsageError"`.
Without `--json`, the message goes to standard error.

With residual checks, `fit` ranks the models that pass them, even when only one model is
asked for.
