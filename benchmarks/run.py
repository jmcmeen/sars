"""Time sars against R sars on the same jobs, for one machine at a time.

Usage (from the repository root):

    uv run python benchmarks/run.py [--quick] [--no-r] [--name NAME]

Runs each job (one power fit, all 20 models, the threshold models, bootstrap
intervals) on the bundled galap and aegean datasets and on synthetic datasets
of growing size, in Python and, if ``Rscript`` and R sars are installed, in R.
Writes the timings, the machine and a figure to ``benchmarks/results/<machine>/``,
named after the CPU, so runs on different machines sit side by side.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

import sars  # noqa: E402

HERE = Path(__file__).resolve().parent
SIZES = [50, 200, 1000, 5000]  # synthetic datasets, in islands (data points)
TASKS = ["power", "average", "threshold"]  # run on every dataset
BOOTSTRAP_DATASETS = ["galap"]  # bootstraps take minutes in R, so on galap only
WEBP = {"lossless": True, "method": 6}
# One color per series in every panel: Python's in blues, R's in orange.
COLORS = {
    "Python": "#1f77b4",
    "Python full": "#6baed6",
    "Python all cores": "#08306b",
    "Python full all cores": "#17becf",
    "R": "#ff7f0e",
}


def synthetic(n: int) -> pd.DataFrame:
    """``n`` islands with areas drawn uniformly in log from 0.01 to 10,000 km² and
    richness from a power law (c = 10, z = 0.25) with lognormal noise."""
    rng = np.random.default_rng(n)
    area = np.sort(10 ** rng.uniform(-2, 4, n))
    species = np.maximum(1, np.round(10 * area**0.25 * np.exp(rng.normal(0, 0.15, n))))
    return pd.DataFrame({"area": area, "species": species})


def python_job(task: str, data: pd.DataFrame, n_boot: int) -> Callable[[], str]:
    """A function running one job and returning a result to compare with R's."""
    rng = np.random.default_rng(0)
    if task == "power":
        return lambda: f"z={sars.sar_power(data).params['z']:.4f}"
    if task == "average":

        def average() -> str:
            weights = sars.sar_average(data).weights
            return f"best={max(weights, key=weights.get)}"

        return average
    if task == "threshold":
        return lambda: (
            f"best={sars.sar_threshold(data, ['ContOne', 'ZslopeOne', 'Linear']).best_model}"
        )
    method, jobs = {
        "bootstrap": ("fast", 1),
        "bootstrap_full": ("full", 1),
        "bootstrap_all_cores": ("fast", -1),
        "bootstrap_full_all_cores": ("full", -1),
    }[task]

    def bootstrap() -> str:
        sars.bootstrap_ci(data, n_boot=n_boot, method=method, n_jobs=jobs, rng=rng)
        return f"n_boot={n_boot}"  # the resamples asked for, as R reports them

    return bootstrap


def time_python(jobs: pd.DataFrame, datasets: dict, repeats: int, n_boot: int) -> pd.DataFrame:
    rows = []
    for job in jobs.itertuples():
        data = datasets[job.dataset]
        run = python_job(job.task, data, n_boot)
        runs = 1 if job.task.startswith("bootstrap") else repeats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not job.task.startswith("bootstrap"):
                run()  # warm-up (a bootstrap is long enough without)
            times = []
            for _ in range(runs):
                started = time.perf_counter()
                result = run()
                times.append(time.perf_counter() - started)
        seconds = float(np.median(times))
        rows.append(
            {
                "language": "Python",
                "task": job.task,
                "dataset": job.dataset,
                "n": len(data),
                "seconds": seconds,
                "runs": runs,
                "result": result,
            }
        )
        print(f"Python {job.task} {job.dataset}: {seconds:.3f} s", flush=True)
    return pd.DataFrame(rows)


def time_r(jobs: pd.DataFrame, folder: Path, repeats: int, n_boot: int) -> pd.DataFrame:
    """R sars's timings for the jobs R has (no full or parallel bootstrap)."""
    r_jobs = jobs[jobs["task"].isin([*TASKS, "bootstrap"])]
    r_jobs.to_csv(folder / "jobs.csv", index=False)
    out = folder / "r.csv"
    subprocess.run(
        [
            "Rscript",
            str(HERE / "run.R"),
            str(folder / "jobs.csv"),
            str(out),
            str(repeats),
            str(n_boot),
        ],
        check=True,
    )
    return pd.read_csv(out)


def speedups(timings: pd.DataFrame) -> pd.DataFrame:
    """R's time over Python's, per job. R's one bootstrap is set against each
    Python bootstrap; the full method is the like-for-like one."""
    r_boot = timings[(timings["language"] == "R") & (timings["task"] == "bootstrap")]
    variants = timings.loc[
        (timings["language"] == "Python") & timings["task"].str.startswith("bootstrap"), "task"
    ].unique()
    paired = pd.concat(
        [timings[~((timings["language"] == "R") & (timings["task"] == "bootstrap"))]]
        + [r_boot.assign(task=task) for task in variants],
        ignore_index=True,
    )
    wide = paired.pivot_table(
        index=["task", "dataset", "n"], columns="language", values="seconds"
    ).dropna()
    wide["R_over_Python"] = wide["R"] / wide["Python"]
    return wide


def write_results(timings: pd.DataFrame, folder: Path) -> None:
    """The speed-ups (only when R was timed) and the figure, from the timings."""
    if (timings["language"] == "R").any():
        wide = speedups(timings)
        wide.round(4).to_csv(folder / "speedup.csv")
        print(wide.round(3).to_string())
    else:
        (folder / "speedup.csv").unlink(missing_ok=True)  # don't leave an older run's
    figure(timings, folder / "timings.webp")


def r_versions() -> dict:
    if shutil.which("Rscript") is None:
        return {}
    script = 'cat(R.version$major, ".", R.version$minor, " ", as.character(packageVersion("sars")), sep="")'
    try:
        text = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return {}
    r, r_sars = text.split()
    return {"R": r, "R sars": r_sars}


def machine() -> dict:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":  # macOS reports the chip here; Linux in /proc/cpuinfo
        cpu = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
            or cpu
        )
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "cpu": cpu,
        "system": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sars": sars.__version__,
        **r_versions(),
    }


def slug(text: str) -> str:
    """A folder name from a CPU name: "AMD Ryzen 7 7700 8-Core Processor" ->
    "amd-ryzen-7-7700"."""
    text = re.sub(r"\(r\)|\(tm\)|\bcpu\b|@.*$|\d+-core.*$|processor", "", text.lower())
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "machine"


def figure(timings: pd.DataFrame, path: Path) -> None:
    """Seconds per job, Python against R, one panel per task."""
    tasks = [t for t in [*TASKS, "bootstrap"] if t in set(timings["task"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.2), squeeze=False)
    seen: set[str] = set()
    for ax, task in zip(axes[0], tasks, strict=True):
        rows = timings[
            timings["task"].str.startswith(task) if task == "bootstrap" else timings["task"] == task
        ]
        order = rows.drop_duplicates("dataset").sort_values("n")["dataset"].tolist()
        series = rows.assign(
            label=rows["language"] + rows["task"].str.replace(task, "", regex=False)
        )
        series["label"] = series["label"].str.replace("_", " ")
        labels = [label for label in COLORS if label in set(series["label"])]
        seen.update(labels)
        width = 0.8 / len(labels)
        for i, label in enumerate(labels):
            part = series[series["label"] == label].set_index("dataset").reindex(order)
            ax.bar(
                np.arange(len(order)) + i * width,
                part["seconds"],
                width,
                label=label,
                color=COLORS[label],
            )
        ax.set_xticks(np.arange(len(order)) + 0.4 - width / 2)
        names = rows.drop_duplicates("dataset").set_index("dataset")["n"]
        short = {d: d.replace("synthetic_", "synthetic\n") for d in order}
        ax.set_xticklabels(
            [f"{short[d]}\nn={names[d]}" if "\n" not in short[d] else short[d] for d in order],
            fontsize=7,
        )
        ax.set_yscale("log")
        ax.set_ylabel("seconds")
        ax.set_title(task)
    drawn = [label for label in COLORS if label in seen]
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[label]) for label in drawn]
    fig.legend(handles, drawn, loc="lower center", ncol=len(drawn), fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=100, pil_kwargs=WEBP)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="small datasets, one run, 20 resamples"
    )
    parser.add_argument("--no-r", action="store_true", help="time Python only")
    parser.add_argument("--repeats", type=int, default=3, help="runs per job (median; default 3)")
    parser.add_argument("--n-boot", type=int, default=100, help="bootstrap resamples (default 100)")
    parser.add_argument("--name", help="the machine's folder name (default: from the CPU)")
    parser.add_argument(
        "--out", type=Path, default=HERE / "results", help="where machine folders go"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the figure and speed-ups from a run's timings.csv",
    )
    args = parser.parse_args()
    info = machine()
    # A quick run goes to its own folder, so it never replaces a full run's results.
    folder = args.out / (args.name or slug(info["cpu"]) + ("-quick" if args.quick else ""))
    if args.plot_only:
        write_results(pd.read_csv(folder / "timings.csv"), folder)
        print(f"Redrew {folder}")
        return
    sizes, repeats, n_boot = (
        (SIZES[:2], 1, 20) if args.quick else (SIZES, args.repeats, args.n_boot)
    )

    datasets = {"galap": sars.load_galap(), "aegean": sars.load_aegean()}
    datasets |= {f"synthetic_{n}": synthetic(n) for n in sizes}
    jobs = pd.DataFrame(
        [{"task": t, "dataset": d} for d in datasets for t in TASKS]
        + [
            {"task": t, "dataset": d}
            for d in BOOTSTRAP_DATASETS
            for t in (
                "bootstrap",
                "bootstrap_full",
                "bootstrap_all_cores",
                "bootstrap_full_all_cores",
            )
        ]
    )
    use_r = not args.no_r and "R sars" in info
    if not args.no_r and not use_r:
        print("Rscript or the R sars package not found: timing Python only.", file=sys.stderr)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        paths = {}
        for name, data in datasets.items():
            paths[name] = scratch_dir / f"{name}.csv"
            data[["area", "species"]].to_csv(paths[name], index=False)
        jobs["path"] = jobs["dataset"].map(lambda d: str(paths[d]))
        timings = [time_python(jobs, datasets, repeats, n_boot)]
        if use_r:
            try:
                timings.append(time_r(jobs, scratch_dir, repeats, n_boot))
            except subprocess.CalledProcessError as err:
                print(f"R failed ({err}): keeping the Python timings only.", file=sys.stderr)
                info = {k: v for k, v in info.items() if k not in ("R", "R sars")}
    timings_frame = pd.concat(timings, ignore_index=True)

    timings_frame.to_csv(folder / "timings.csv", index=False)
    pd.DataFrame([info | {"repeats": repeats, "n_boot": n_boot}]).to_csv(
        folder / "machine.csv", index=False
    )
    write_results(timings_frame, folder)
    print(f"Written to {folder}")


if __name__ == "__main__":
    main()
