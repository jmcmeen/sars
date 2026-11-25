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

## On an AMD Ryzen 7 7700

sars 0.11.8 against R 4.5.2 with sars 2.1.1, on one core unless noted
(`results/amd-ryzen-7-7700/`):

| Job | sars | R sars | R ÷ sars |
| --- | --- | --- | --- |
| One power fit | 0.04-0.10 s | 0.17-0.64 s | 3-11× |
| All 20 models | 4.2 s (galap) to 24 s (5,000 points) | 9.7 s to 55 s | 1.7-2.3× |
| Threshold models | 1 ms (galap) to 89 ms (5,000 points) | 0.5 s to 1.7 s | 19-720× |
| Bootstrap, 100 resamples, full | 613 s | 983 s | 1.6× |
| Bootstrap, 100 resamples, fast | 17 s | | 58× |
| Bootstrap, 100 resamples, full, all 16 threads | 79 s | | 12× |

Both packages agree on every result but one: on the 1,000-point synthetic
dataset R's chapman fit stops at an AICc of 8194.6, where sars finds 6279.9,
so R's best model there is p1 (6279.97) and sars's is chapman.
