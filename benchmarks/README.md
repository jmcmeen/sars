# Benchmarks

`run.py` times sars against R sars on the same jobs:

| Job | Python | R |
| --- | --- | --- |
| One power fit | `sar_power` | `sar_power` |
| All 20 models, weighted | `sar_average` | `sar_average` |
| Threshold models | `sar_threshold` (ContOne, ZslopeOne, Linear) | `sar_threshold` (ContOne, ZslopeOne, with the non-threshold models) |
| Bootstrap intervals | `bootstrap_ci`, fast and full, on one core and on all | `sar_average(confInt = TRUE)` |

The power, average and threshold jobs run on the bundled galap (16 islands)
and aegean (90) datasets, and on synthetic power-law datasets of 50 to 5,000
points. Bootstraps take minutes in R, so they run on galap only.

Two options cover what the defaults don't:

- `--large N` adds the all-models job on a synthetic table of N points shaped
  like a species-accumulation nest of observation records: areas from 10,000
  to 300,000 km² and richness rising toward 200 species, with noise. Large
  areas and a curved, noisy relationship are where the nonlinear models are
  slowest. It's timed once, and not in R (R would take hours); with 100,000
  points, all 20 models took about 3.5 minutes on an AMD Ryzen 7 7700 and
  1.3 minutes on an RTX 5060 Ti.
- `--gpu` times the all-models and bootstrap jobs again with
  `sars.set_backend("cupy")`, as a "Python GPU" series, when CuPy
  (`pip install "sars[gpu]"`) finds a GPU; that includes the `--large`
  table, timed once more on the GPU. The GPU and CuPy version go in
  `machine.csv`.

```sh
uv run python benchmarks/run.py            # Python and, if Rscript and R sars are installed, R
uv run python benchmarks/run.py --quick    # small datasets, one run, 20 resamples (to <machine>-quick/)
uv run python benchmarks/run.py --no-r     # Python only
uv run python benchmarks/run.py --large 100000 --gpu   # add a large table and the GPU series
uv run python benchmarks/run.py --plot-only   # redraw the figure and speed-ups from saved timings
```

To time R too, install R and the sars package from CRAN
(`install.packages("sars")`); without them the script times Python only.

## Fairness

- Both sides use their packages' defaults: no residual checks, and R's
  partial start grid.
- Everything runs on one core except the jobs labeled "all cores" (R sars
  has no parallel bootstrap).
- Each job is timed inside one session, after a warm-up run, as the median of
  three runs, so start-up and imports aren't counted. A bootstrap runs once,
  with no warm-up: it's long enough that one-time costs don't matter.
- Each timing carries a result (the power model's z, the top-weighted model,
  the best threshold model), so the two packages can be seen to have done the
  same work. A bootstrap's result is just the number of resamples asked for.
- Python's fast bootstrap refits each resample from the original fit's
  parameters; its full bootstrap searches the whole start grid, as R does on
  every resample. Compare R with the full method for like with like.

## Results

Each run writes `results/<machine>/`, named after the CPU, so results from
several machines sit side by side and a re-run replaces only its own folder:

- `timings.csv`: one row per job and language, with its time and result;
- `speedup.csv`: R's time over Python's, per job (R's bootstrap against each
  Python bootstrap; `bootstrap_full` is the like-for-like one);
- `machine.csv`: the CPU, system and package versions, repeats and resamples;
- `timings.webp`: the times, one panel per job.

## Results on three machines

sars as of this README (unreleased, after 0.11.11) against R 4.5 with sars 2.1.1, on
one core unless noted, 100 resamples per bootstrap:

| Machine | `results/` |
| --- | --- |
| AMD Ryzen 7 7700 (8 cores), RTX 5060 Ti, Linux | `amd-ryzen-7-7700/` |
| Intel Core i5-12400F (6 cores), RTX 3060, Linux | `12th-gen-intel-core-i5-12400f/` |
| Apple M5 Pro, macOS (no NVIDIA GPU) | `apple-m5-pro/` |

How many times faster sars is than R sars, by job (the range over the datasets):

| Job | Ryzen 7 7700 | i5-12400F | M5 Pro |
| --- | --- | --- | --- |
| One power fit | 33-124× | 14-112× | 16-130× |
| All 20 models | 2.8-37× | 1.8-25× | 1.5-25× |
| Threshold models | 33-765× | 19-596× | 28-704× |
| Bootstrap, full | 29× | 22× | 22× |
| Bootstrap, fast | 95× | 79× | 87× |
| Bootstrap, full, all cores | 174× | 103× | 179× |

All 20 models take 0.2 to 0.4 s on galap (R 5 to 10 s) and 18 to 38 s on 5,000 points
(R 31 to 67 s). The full bootstrap, R's own method, takes 23 to 44 s where R takes 522
to 985 s. The speed-up is smallest on the largest synthetic dataset, where the final
solve of each fit dominates.

The large table (`--large 100000`) and the GPU (`--gpu`), all 20 models:

| Machine | CPU | GPU |
| --- | --- | --- |
| Ryzen 7 7700, RTX 5060 Ti | 207 s | 76 s |
| i5-12400F, RTX 3060 | 471 s | 167 s |
| M5 Pro | 162 s | |

The GPU pays off on large tables, 2.7 to 2.8 times faster here. On the small datasets
and the galap bootstraps it's slower than the CPU (about 1 s against 0.3 s for all 20
models on galap): a few hundred islands don't keep it busy.

Both packages agree on every result but one: on the 1,000-point synthetic dataset R's
chapman fit stops short of the optimum sars finds, so R's best model there is p1 and
sars's is chapman.
