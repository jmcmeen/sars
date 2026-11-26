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

Some analyses read more than area and richness; each has its own command and its own
table (see [Other tables](#other-tables)).

## Commands

| Command | Does |
|---|---|
| `sars fit DATA [-m MODEL ...] [--params [--conf C]]` | Fits one model (`power` by default). With several models (`-m power,loga -m koba`) or `-m all`, fits each and ranks them by AICc. `--params` adds each model's parameter table: standard errors, t tests and confidence intervals (see [`param_table`](api/models.md)). |
| `sars average DATA [-m MODEL ...] [--ic AICc\|AIC\|BIC]` | Averages the models (all by default), weighted by the criterion. |
| `sars threshold DATA [-m MODEL ...] [--interval X] [--no-log-area] [--ci [--ci-method boot\|F] [--ci-model M] [--n-boot N]]` | Fits piecewise models (ContOne, ZslopeOne and Linear by default) and picks one by AICc. `--ci` adds confidence intervals for the chosen model's breakpoints, by residual bootstrap or inverted F test (see [`threshold_ci`](api/threshold.md)). |
| `sars cv DATA [-m MODEL ...] [--ic AICc\|AIC\|BIC] [--warm] [--label-col COL] [--n-jobs N]` | Ranks the models by leave-one-island-out cross-validation (see [`loo_cv`](api/inference.md)). |
| `sars loss DATA --area A ... (--remaining R ... \| --fraction-lost F ...) [-m MODEL ...] [--n-boot N] [--seed S]` | Species expected to be lost with habitat, with bootstrap intervals (see [`habitat_loss`](api/inference.md)). |
| `sars many DATA DATA ... \| sars many TABLE --by COL [-m MODEL ...] [--no-threshold] [--n-jobs N]` | Fits the same models to many datasets and summarizes them in one table: best model and weight, log-log z, threshold model (see [`sar_many`](api/inference.md)). |
| `sars loglog DATA [--log-base e\|2\|10] [--con C] [--compare]` | Fits the log-log power model by least squares, with standard errors, t tests and p values (see [`lin_pow`](api/models.md)). |
| `sars ci DATA [--area A ...] [-m MODEL ...] [--method fast\|full] [--n-boot N] [--seed S] [--n-jobs N]` | The model-averaged richness at the areas (ten over the data's range by default), with bootstrap intervals (see [`bootstrap_ci`](api/inference.md)). |
| `sars habitat DATA [--mod-type power_log\|power\|logarithmic] [--habitats-col COL] [--con C] [--log-base e\|2\|10]` | Fits the habitat models (choros, jigsaw, Kallimanis and the curve on area alone) to data with a number of habitats per island, and ranks them by AICc (see [`sar_habitat`](api/habitat.md)). |
| `sars countryside DATA --habitats COL,... --species COL,... [--mod-type power\|logarithmic] [--grid-start partial\|exhaustive\|none] [--z-lower Z] [--n-starts N\|all]` | Fits the countryside model: each species group's richness from the areas of several habitats (see [`sar_countryside`](api/habitat.md)). |
| `sars predict FILE AREA ...` | Predicts richness at the areas from a result saved with `--save`. |
| `sars serve [--host H] [--port P] [--token-file F] [--timeout S]` | Serves the same operations over HTTP; needs the `server` extra (see [HTTP server](server.md)). |
| `sars datasets [NAME]` | Lists the bundled datasets and tables, or writes one as CSV. |

Every command takes `--backend numpy|cupy` before its name (`sars --backend cupy fit
...`), or the `SARS_BACKEND` environment variable, to screen start grids on an NVIDIA GPU
(needs the `gpu` extra; see [Speed and the GPU](getting-started.md#speed-and-the-gpu)).
The JSON envelope records the backend used.

`fit`, `average` and `ci` also take the residual checks of [`sar_multi`](api/inference.md):
`--normality shapiro|lillie|kolmogorov`, `--homoscedasticity cor.fitted|cor.area` and
`--neg-check`. `sars COMMAND --help` lists every option.

## Other tables

These commands read a CSV file, standard input or their own bundled table, and take
`--json` and `--save` like the others.

**Habitat models** (`sars habitat`): one row per island, with a column for the number of
habitats (or another measure of heterogeneity) beside area and richness. The columns are
`area`, `habitats` and `species` unless named with `--area-col`, `--habitats-col` and
`--species-col`. The bundled table is `habitat`.

```console
$ sars habitat habitat
$ sars habitat snails.csv --habitats-col Het --mod-type power
```

**Countryside model** (`sars countryside`): one row per site, a column of area for each
habitat and a column of richness for each species group. Name them with `--habitats` and
`--species`, repeated or comma-separated. The bundled table is `countryside`, whose
columns are used when none are named.

```console
$ sars countryside countryside
$ sars countryside sites.csv --habitats forest,farmland --species forest_spp,farm_spp,generalists
```

`sars predict` doesn't predict from these results, since they need more than areas: load
the saved file in Python and call its `predict`.

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
