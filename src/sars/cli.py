"""The ``sars`` command line: fit, average, threshold, predict and datasets.

Built on the standard library's argparse, so it needs no extra. The work is
the library's (``sars._operations``, which the HTTP server calls too): this
module reads the arguments and the data, calls it and prints. Every command
can print a JSON envelope (``--json``)::

    {"sars": "<version>", "command": "fit", "ok": true,
     "result": {...}, "warnings": [...]}

or, on failure, ``"ok": false`` and ``"error": {"type": ..., "message": ...}``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sars import _operations as ops
from sars._operations import DATASETS, TABLES, THRESHOLD_MODELS, InputError

__all__ = ["DATASETS", "THRESHOLD_MODELS", "build_parser", "main"]


# ---------------------------------------------------------------------------
# Reading data
# ---------------------------------------------------------------------------


def _read_table(source: str, command: str | None = None) -> Any:
    """The table in a CSV file, standard input (``-``) or a bundled dataset.
    ``command`` names a command that reads more than area and richness: the
    bundled tables are then the ones for it."""
    import pandas as pd

    if source == "-" or Path(source).is_file():
        label = "standard input" if source == "-" else source
        try:
            text = (
                sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8-sig")
            )
            frame = pd.read_csv(io.StringIO(text))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise InputError(f"Couldn't read {label} as CSV: {exc}") from exc
        return frame
    names = (
        list(DATASETS)
        if command is None
        else [name for name, spec in TABLES.items() if command in spec["commands"]]
    )
    if source in names:
        return ops.bundled(source)
    if source in TABLES:
        use = " and ".join(f"`sars {name}`" for name in TABLES[source]["commands"])
        raise InputError(f"The bundled {source} table is for {use}.")
    if source in DATASETS:
        raise InputError(
            f"The bundled {source} dataset holds area and richness only, not what "
            f"`sars {command}` reads."
        )
    known = f"nor a bundled dataset: {', '.join(names)}" if names else "this command has none"
    raise InputError(f"No such file: {source} ({known}).")


def _is_bundled(source: str) -> bool:
    """Whether ``source`` is read as a bundled table: a file of the same name
    comes first."""
    return source != "-" and not Path(source).is_file()


def _read_data(source: str, area_col: str, species_col: str, label_col: str | None = None) -> Any:
    """A data frame of area and species from a CSV file, standard input
    (``-``) or a bundled dataset's name. ``label_col`` keeps a column naming the
    islands too: its name, or "auto" for the first of island, site, locality
    and name found."""
    import sars

    frame = _read_table(source)
    for column in (area_col, species_col):
        if column not in frame.columns:
            raise InputError(
                f"No column {column!r} in {source}; its columns are {list(frame.columns)}. "
                "Name the columns with --area-col and --species-col."
            )
    data = sars.from_df(frame, area_col=area_col, species_col=species_col)
    if label_col is not None:
        from sars._crossval import _LABEL_GUESSES

        if label_col == "auto":
            lower = {str(c).lower(): c for c in frame.columns}
            label_col = next((lower[g] for g in _LABEL_GUESSES if g in lower), None)
        elif label_col not in frame.columns:
            raise InputError(
                f"No column {label_col!r} in {source}; its columns are {list(frame.columns)}."
            )
        if label_col is not None and label_col not in data.columns:
            data[label_col] = frame[label_col].to_numpy()
    return data


# ---------------------------------------------------------------------------
# Printing a result
# ---------------------------------------------------------------------------


def _print_summary(summary: dict[str, Any], out: Any) -> None:
    import pandas as pd

    kind = summary["type"]
    if kind == "GDMFit":
        _print_gdm(summary, out)
        return
    if kind == "GDMMultiFit":
        print(f"Best: {summary['best']} (the general dynamic model, by AICc)", file=out)
        _print_table(summary["models"], out, text=("model", "k", "converged"))
        for fit in summary["fits"].values():
            print("", file=out)
            _print_gdm(fit, out)
        return
    if kind == "HabitatFit":
        forms = {
            "power_log": "power forms, fitted on the logs",
            "power": "power forms, by non-linear least squares",
            "logarithmic": "logarithmic forms",
        }
        print(
            f"Best: {summary['best']} ({forms[summary['mod_type']]}; n = {summary['n']})", file=out
        )
        _print_table(summary["models"], out, text=("model",))
        for model in summary["not_fitted"]:
            print(f"not fitted: {model}", file=out)
        return
    if kind == "CountrysideFit":
        print(
            f"Countryside model ({summary['mod_type']}), {len(summary['groups'])} species groups "
            f"on {len(summary['habitats'])} habitats, n = {summary['n']}:",
            file=out,
        )
        _print_table(summary["params"], out, text=("group",))
        print("Affinity (each h over the group's largest):", file=out)
        _print_table(summary["affinity"], out, text=("group",))
        if summary["rss"]:
            parts = "  ".join(f"{name} = {_fmt(value)}" for name, value in summary["rss"].items())
            print(f"RSS of total richness: {parts}", file=out)
        for group in summary["not_fitted"]:
            print(f"not fitted: {group}", file=out)
        return
    if kind == "ManySARFit":
        table = pd.DataFrame(summary["datasets"])
        shown = ["dataset", "n", "z", "R2", "best_model", "best_weight", "threshold_model"]
        shown += ["breakpoint_area"]
        if table["error"].fillna("").astype(bool).any():
            shown.append("error")
        notes = pd.DataFrame(summary["datasets"])
        table = table[[c for c in shown if c in table.columns]].copy()
        for column in table.columns:
            if column not in ("dataset", "n", "best_model", "threshold_model", "error"):
                table[column] = [_cell(v) for v in table[column]]
        print(table.fillna("").to_string(index=False), file=out)
        for row in notes.to_dict(orient="records"):
            for model in row.get("not_fitted") or []:
                print(f"{row['dataset']}: not fitted: {model}", file=out)
            for text in row.get("warnings") or []:
                print(f"{row['dataset']}: warning: {text}", file=out)
        return
    if kind == "LinPowFit":
        log = "log" if summary["log_base"] == "e" else f"log{summary['log_base']}"
        print(f"{log} S = {log} c + z {log} A, by least squares:", file=out)
        print(
            f"  c = {_fmt(summary['c'])} (SE {_fmt(summary['c_se'])})  "
            f"{log} c = {_fmt(summary['logc'])} (SE {_fmt(summary['logc_se'])}, "
            f"t = {_t(summary['logc_t'], summary['logc_se'])}, p = {_fmt(summary['logc_p'])})",
            file=out,
        )
        print(
            f"  z = {_fmt(summary['z'])} (SE {_fmt(summary['z_se'])}, t = "
            f"{_t(summary['z_t'], summary['z_se'])}, p = {_fmt(summary['z_p'])})",
            file=out,
        )
        print(
            f"R2 = {_fmt(summary['R2'])}  adjusted R2 = {_fmt(summary['adj_R2'])}  "
            f"sigma = {_fmt(summary['sigma'])}  n = {summary['n']}",
            file=out,
        )
        if summary["power"] is not None:
            power = summary["power"]["params"]
            print(
                f"nonlinear power model: c = {_fmt(power['c'])}  z = {_fmt(power['z'])}", file=out
            )
        return
    if kind == "BootstrappedCI":
        table = pd.DataFrame(summary["intervals"])
        print(
            f"Model-averaged richness, {summary['conf']:.0%} bootstrap intervals "
            f"({summary['n_boot']} resamples):",
            file=out,
        )
        for column in table.columns:
            table[column] = [_cell(v) for v in table[column]]
        print(table.to_string(index=False), file=out)
        for reason, count in summary["dropped"].items():
            print(f"dropped: {count} resamples: {reason}", file=out)
        return
    if kind == "HabitatLoss":
        table = pd.DataFrame(summary["scenarios"])
        print(
            f"Species lost, model average by {summary['ic']}, {summary['conf']:.0%} intervals:",
            file=out,
        )
        for column in table.columns:
            table[column] = [_cell(v) for v in table[column]]
        print(table.to_string(index=False), file=out)
        return
    if kind == "SARFit":
        params = "  ".join(f"{k} = {_fmt(v)}" for k, v in summary["params"].items())
        print(f"{summary['model']}: {params}", file=out)
        print(
            f"R2 = {_fmt(summary['R2'])}  AIC = {_fmt(summary['AIC'])}  "
            f"AICc = {_fmt(summary['AICc'])}  BIC = {_fmt(summary['BIC'])}  n = {summary['n']}"
            + ("" if summary["converged"] else "  (did not converge)"),
            file=out,
        )
        return
    if kind == "AveragedSAR":
        print(f"Model average, weights by {summary['ic']}:", file=out)
    elif kind == "LOOCV":
        print(
            f"Leave-one-island-out: best {summary['best']} (by RMSE of left-out islands)", file=out
        )
    elif kind == "ThresholdFit":
        points = ", ".join(_fmt(b) for b in summary["breakpoints"]) or "none"
        scale = "log area" if summary["log_area"] else "area"
        print(f"Best: {summary['best']} (breakpoints on {scale}: {points})", file=out)
    else:
        print(f"Best: {summary['best']}", file=out)
    table = pd.DataFrame(summary["models"])
    columns = [c for c in table.columns if c not in ("shape", "asymptote", "observed_shape")]
    table = table[columns].copy()
    for column in columns:
        if column != "model":
            table[column] = [_cell(v) for v in table[column]]
    print(table.to_string(index=False), file=out)
    for label in ("excluded", "not_fitted"):
        for model, reason in summary.get(label, {}).items():
            print(f"{label.replace('_', ' ')}: {model}: {reason}", file=out)


def _print_gdm(summary: dict[str, Any], out: Any) -> None:
    print(f"{summary['model']}: {summary['formula']}", file=out)
    rows = [
        {
            "term": name,
            "estimate": value,
            "se": summary["std_errors"][name],
            "p": summary["p_values"][name],
        }
        for name, value in summary["params"].items()
    ]
    _print_table(rows, out, text=("term",))
    print(
        f"R2 = {_fmt(summary['R2'])}  RSE = {_fmt(summary['RSE'])}  AIC = {_fmt(summary['AIC'])}  "
        f"AICc = {_fmt(summary['AICc'])}  BIC = {_fmt(summary['BIC'])}  n = {summary['n']}"
        + ("" if summary["converged"] else "  (did not converge)"),
        file=out,
    )
    if summary["comparison"] is not None:
        print("Against simpler models of the same form:", file=out)
        _print_table(summary["comparison"], out, text=("model", "k", "converged"))


def _print_table(rows: list[dict[str, Any]], out: Any, text: Sequence[str] = ()) -> None:
    """A table of records, its numbers to four significant figures; ``text``
    names the columns printed as they are."""
    import pandas as pd

    table = pd.DataFrame(rows)
    for column in table.columns:
        if column not in text:
            table[column] = [_cell(v) for v in table[column]]
    print(table.to_string(index=False), file=out)


def _cell(value: Any) -> str:
    """A table cell: blank where there is no value (a model without a second
    breakpoint), else four significant figures."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value if isinstance(value, str) else f"{value:.4g}"


def _t(t: Any, se: Any) -> str:
    """A t statistic: infinite where the points lie exactly on the line (its
    standard error is 0, and JSON can't hold the infinity)."""
    return "inf" if t is None and se == 0 else _fmt(t)


def _fmt(value: Any) -> str:
    return "undefined" if value is None else f"{value:.6g}"


# ---------------------------------------------------------------------------
# Commands: each returns (result for the envelope, function to print it)
# ---------------------------------------------------------------------------

Printer = Callable[[Any], None]


def _finish(obj: Any, args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import sars

    if args.save:
        sars.save(obj, args.save)
    result = ops.result(obj, include_object=args.json, saved=str(args.save) if args.save else None)
    return result, lambda out: _print_summary(result["summary"], out)


def _checks(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "normality": args.normality,
        "homoscedasticity": args.homoscedasticity,
        "neg_check": args.neg_check,
    }


def _cmd_fit(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    # Checked before anything is fitted or saved.
    if args.conf is not None and not args.params:
        raise InputError("--conf sets the parameter tables' level: give --params too.")
    conf = 0.95 if args.conf is None else args.conf
    if not 0 < conf < 1:
        raise InputError(f"The interval level must be between 0 and 1, got {conf}.")
    data = _read_data(args.data, args.area_col, args.species_col)
    obj = ops.fit(data, ops.model_names(args.model), **_checks(args))
    result, show = _finish(obj, args)
    if not args.params:
        return result, show
    tables = ops.parameters(obj, conf=conf)
    result["summary"]["parameters"] = tables

    def show_all(out: Any) -> None:
        show(out)
        _print_parameters(tables, conf, out)

    return result, show_all


def _print_parameters(tables: dict[str, list[dict[str, Any]]], conf: float, out: Any) -> None:
    import pandas as pd

    for model, rows in tables.items():
        print(f"\n{model}: parameters, {conf:.0%} intervals", file=out)
        table = pd.DataFrame(rows)
        for column in table.columns:
            if column != "parameter":
                # NaN where the data don't identify a parameter: shown, not blank.
                table[column] = ["NaN" if v is None else _cell(v) for v in table[column]]
        print(table.to_string(index=False), file=out)


def _cmd_average(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    data = _read_data(args.data, args.area_col, args.species_col)
    obj = ops.average(data, ops.model_names(args.models), ic=args.ic, **_checks(args))
    return _finish(obj, args)


def _cmd_threshold(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    data = _read_data(args.data, args.area_col, args.species_col)
    obj = ops.threshold(
        data, ops.model_names(args.models), interval=args.interval, log_area=not args.no_log_area
    )
    result, show = _finish(obj, args)
    if not args.ci:
        return result, show
    intervals = ops.summary(
        ops.breakpoint_ci(
            obj,
            model=args.ci_model,
            method=args.ci_method,
            n_boot=args.n_boot,
            conf=args.conf,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
    )
    result["summary"]["ci"] = intervals

    def show_all(out: Any) -> None:
        show(out)
        how = (
            "inverted F test"
            if intervals["method"] == "F"
            else (f"bootstrap of {intervals['n_boot']} resamples")
        )
        print(
            f"\n{intervals['model']} breakpoints, {intervals['conf']:.0%} intervals ({how}):",
            file=out,
        )
        for i, row in enumerate(intervals["breakpoints"], start=1):
            print(
                f"  {i}: area {_fmt(row['estimate_area'])} ({_fmt(row['lower_area'])} to "
                f"{_fmt(row['upper_area'])})",
                file=out,
            )

    return result, show_all


def _cmd_cv(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    label = args.label_col or "auto"
    data = _read_data(args.data, args.area_col, args.species_col, label_col=label)
    obj = ops.cv(
        data,
        ops.model_names(args.models),
        warm=args.warm,
        ic=args.ic,
        label=label,
        n_jobs=args.n_jobs,
    )
    return _finish(obj, args)


def _cmd_loss(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    data = _read_data(args.data, args.area_col, args.species_col)
    obj = ops.loss(
        data,
        area=args.area,
        remaining=args.remaining,
        fraction_lost=args.fraction_lost,
        models=ops.model_names(args.models),
        ic=args.ic,
        n_boot=args.n_boot,
        conf=args.conf,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    return _finish(obj, args)


def _cmd_many(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import sars

    options = {
        "models": ops.model_names(args.models),
        "ic": args.ic,
        "threshold": not args.no_threshold,
        "n_jobs": args.n_jobs,
    }
    if args.by is not None:
        if len(args.data) != 1:
            raise InputError("With --by, give one table holding every dataset.")
        frame = _read_table(args.data[0])
        missing = [c for c in (args.by, args.area_col, args.species_col) if c not in frame.columns]
        if missing:
            raise InputError(
                f"No column {missing[0]!r} in {args.data[0]}; its columns are "
                f"{list(frame.columns)}."
            )
        table = frame[[args.by, args.area_col, args.species_col]].rename(
            columns={args.area_col: "area", args.species_col: "species"}
        )
        return _finish(ops.many(table, by=args.by, **options), args)
    if len(args.data) < 2:
        raise InputError("Give two or more datasets, or one table and --by COLUMN.")
    datasets: dict[str, Any] = {}
    for source in args.data:
        base = source if source in DATASETS else Path(source).stem or source
        name, count = base, 1
        while name in datasets:  # the same name twice (two folders' islands.csv)
            count += 1
            name = f"{base}_{count}"
        datasets[name] = sars.from_df(
            _read_data(source, args.area_col, args.species_col)[["area", "species"]]
        )
    return _finish(ops.many(datasets, **options), args)


def _cmd_loglog(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    data = _read_data(args.data, args.area_col, args.species_col)
    obj = ops.loglog(data, log_base=args.log_base, con=args.con, compare=args.compare)
    return _finish(obj, args)


def _cmd_ci(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    data = _read_data(args.data, args.area_col, args.species_col)
    obj = ops.ci(
        data,
        areas=args.area,
        models=ops.model_names(args.models),
        ic=args.ic,
        method=args.method,
        n_boot=args.n_boot,
        conf=args.conf,
        seed=args.seed,
        n_jobs=args.n_jobs,
        **_checks(args),
    )
    return _finish(obj, args)


def _column(frame: Any, column: str, source: str, option: str) -> None:
    if column not in frame.columns:
        raise InputError(
            f"No column {column!r} in {source}; its columns are {list(frame.columns)}. "
            f"Name it with {option}."
        )


def _cmd_gdm(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    frame = _read_table(args.data, command="gdm")
    for column, option in (
        (args.area_col, "--area-col"),
        (args.species_col, "--species-col"),
        (args.time_col, "--time-col"),
    ):
        _column(frame, column, args.data, option)
    obj = ops.gdm(
        frame,
        model=args.model,
        mod_sel=args.mod_sel,
        area_col=args.area_col,
        species_col=args.species_col,
        time_col=args.time_col,
    )
    return _finish(obj, args)


def _cmd_habitat(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    frame = _read_table(args.data, command="habitat")
    for column, option in (
        (args.area_col, "--area-col"),
        (args.habitats_col, "--habitats-col"),
        (args.species_col, "--species-col"),
    ):
        _column(frame, column, args.data, option)
    obj = ops.habitat(
        frame,
        mod_type=args.mod_type,
        area_col=args.area_col,
        habitats_col=args.habitats_col,
        species_col=args.species_col,
        con=args.con,
        log_base=args.log_base,
    )
    return _finish(obj, args)


def _cmd_countryside(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    frame = _read_table(args.data, command="countryside")
    defaults = TABLES[args.data]["columns"] if _is_bundled(args.data) else {}
    habitats = ops.column_names(args.habitats) or defaults.get("habitats")
    species = ops.column_names(args.species) or defaults.get("species")
    if habitats is None or species is None:
        raise InputError(
            "Name the habitat area columns with --habitats and the species group columns "
            f"with --species; the columns of {args.data} are {list(frame.columns)}."
        )
    for column in habitats:
        _column(frame, column, args.data, "--habitats")
    for column in species:
        _column(frame, column, args.data, "--species")
    obj = ops.countryside(
        frame,
        habitats=habitats,
        species=species,
        mod_type=args.mod_type,
        grid_start=args.grid_start,
        z_lower=args.z_lower,
        n_starts=args.n_starts,
    )
    return _finish(obj, args)


def _n_starts(value: str) -> int | None:
    """--n-starts: a positive whole number, or "all" for the whole grid."""
    if value == "all":
        return None
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"a positive whole number or 'all', not {value!r}")
    return number


def _cmd_predict(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import sars

    try:
        obj = sars.load(args.fit)
    except FileNotFoundError as exc:
        raise InputError(f"No such file: {args.fit}") from exc
    result = ops.predict(obj, args.areas, time=args.time)

    def show(out: Any) -> None:
        for row in result["predictions"]:
            age = f"{row['time']:.6g}\t" if "time" in row else ""
            print(f"{row['area']:.6g}\t{age}{_fmt(row['species'])}", file=out)

    return result, show


def _cmd_datasets(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    if args.name is None:
        result = ops.datasets(tables=True)

        def show(out: Any) -> None:
            for row in result["datasets"]:
                print(
                    f"{row['name']:<12} {row['n']:>4} sites  area {row['area_min']:.4g} to "
                    f"{row['area_max']:.4g}",
                    file=out,
                )
            for row in result["tables"]:
                use = " and ".join(f"`sars {name}`" for name in row["commands"])
                print(f"{row['name']:<12} {row['n']:>4} rows   for {use}", file=out)

        return result, show
    text = ops.bundled(args.name).to_csv(index=False)
    return {"name": args.name, "csv": text}, lambda out: out.write(text)


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "data",
        help="a CSV file, - for standard input, or a bundled dataset's name (see `sars datasets`)",
    )
    parser.add_argument("--area-col", default="area", help="the CSV's area column (default: area)")
    parser.add_argument(
        "--species-col", default="species", help="the CSV's richness column (default: species)"
    )


def _add_check_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--normality",
        choices=["shapiro", "lillie", "lilliefors", "kolmogorov", "kolmo"],
        help="leave out models whose residuals fail this normality test",
    )
    parser.add_argument(
        "--homoscedasticity",
        choices=["cor.fitted", "cor.area"],
        help="leave out models whose residuals fail this homoscedasticity test",
    )
    parser.add_argument(
        "--neg-check",
        action="store_true",
        help="leave out models that predict negative richness at an observed area",
    )


def _add_output_args(parser: argparse.ArgumentParser, save: bool = True) -> None:
    parser.add_argument("--json", action="store_true", help="print a JSON envelope")
    if save:
        parser.add_argument(
            "--save", type=Path, metavar="FILE", help="save the result as JSON (see `sars predict`)"
        )


class _UsageError(Exception):
    """A command line argparse can't parse (an unknown option, a bad choice)."""


class _Parser(argparse.ArgumentParser):
    """An argparse parser whose usage errors `main` can turn into a JSON
    envelope; subcommand parsers are of this class too."""

    def error(self, message: str) -> Any:
        raise _UsageError(f"{self.prog}: {message}")


def _backend_default() -> str:
    """--backend's default: $SARS_BACKEND if it names a backend, else numpy
    (a bad value is reported when the command runs)."""
    return os.environ.get("SARS_BACKEND", "").strip() or "numpy"


def build_parser() -> argparse.ArgumentParser:
    """The ``sars`` argument parser."""
    parser = _Parser(prog="sars", description="Fit species-area relationship models.")
    parser.add_argument("--version", action="version", version=f"sars {ops.version()}")
    parser.add_argument(
        "--backend",
        choices=["numpy", "cupy"],
        default=_backend_default(),
        help="array library for the start-grid screens: numpy, or cupy on an NVIDIA GPU "
        "(needs the [gpu] extra; default: $SARS_BACKEND, else numpy)",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    fit = commands.add_parser(
        "fit",
        help="fit one model, or several and rank them",
        description="Fit one model (power by default). With several models, or `all`, "
        "fit each and rank them by AICc.",
    )
    _add_data_args(fit)
    fit.add_argument(
        "-m",
        "--model",
        action="append",
        metavar="NAME",
        help="a model, or several (repeated or comma-separated), or `all`",
    )
    fit.add_argument(
        "--params",
        action="store_true",
        help="also print each model's parameter table: standard errors, t tests, intervals",
    )
    fit.add_argument("--conf", type=float, help="interval level for --params (default 0.95)")
    _add_check_args(fit)
    _add_output_args(fit)
    fit.set_defaults(handler=_cmd_fit)

    average = commands.add_parser(
        "average",
        help="average models by information-criterion weights",
        description="Fit the models and average them, weighted by AICc (or AIC, BIC).",
    )
    _add_data_args(average)
    average.add_argument(
        "-m", "--models", action="append", metavar="NAME", help="models to average (default: all)"
    )
    average.add_argument(
        "--ic", choices=["AICc", "AIC", "BIC"], default="AICc", help="weights by (default: AICc)"
    )
    _add_check_args(average)
    _add_output_args(average)
    average.set_defaults(handler=_cmd_average)

    threshold = commands.add_parser(
        "threshold",
        help="fit piecewise (threshold) models",
        description="Fit piecewise models with one or two breakpoints, and pick one by AICc.",
    )
    _add_data_args(threshold)
    threshold.add_argument(
        "-m",
        "--models",
        action="append",
        metavar="NAME",
        help="threshold models (default: ContOne, ZslopeOne and Linear); any of "
        f"{', '.join(THRESHOLD_MODELS)} and Linear",
    )
    threshold.add_argument(
        "--interval", type=float, help="search breakpoints on a grid of this step"
    )
    threshold.add_argument(
        "--no-log-area", action="store_true", help="fit on area rather than log area"
    )
    threshold.add_argument(
        "--ci", action="store_true", help="also give confidence intervals for the breakpoints"
    )
    threshold.add_argument(
        "--ci-model", metavar="NAME", help="the model whose breakpoints (default: the chosen one)"
    )
    threshold.add_argument(
        "--ci-method",
        choices=["boot", "F"],
        default="boot",
        help="residual bootstrap, or inverted F test for ContOne and ZslopeOne (default boot)",
    )
    threshold.add_argument(
        "--n-boot", type=int, default=1000, metavar="N", help="resamples for --ci (default 1000)"
    )
    threshold.add_argument("--conf", type=float, default=0.95, help="interval level (default 0.95)")
    threshold.add_argument("--seed", type=int, help="seed for the resamples")
    threshold.add_argument(
        "--n-jobs", type=int, default=1, metavar="N", help="processes to use; -1 for one per core"
    )
    _add_output_args(threshold)
    threshold.set_defaults(handler=_cmd_threshold)

    cv = commands.add_parser(
        "cv",
        help="rank models by leave-one-island-out cross-validation",
        description="Leave each island out in turn, fit the models to the rest, predict it, "
        "and rank the models by the error of those predictions, beside their AICc rank.",
    )
    _add_data_args(cv)
    cv.add_argument(
        "-m", "--models", action="append", metavar="NAME", help="models to compare (default: all)"
    )
    cv.add_argument(
        "--warm",
        action="store_true",
        help="start each fold from the full fit: much faster, but optimistic for flexible models",
    )
    cv.add_argument(
        "--ic",
        choices=["AICc", "AIC", "BIC"],
        default="AICc",
        help="criterion to compare with and average by (default: AICc)",
    )
    cv.add_argument(
        "--label-col",
        metavar="COLUMN",
        help="the column naming the islands (default: island, site, locality or name if present)",
    )
    cv.add_argument(
        "--n-jobs", type=int, default=1, metavar="N", help="processes to use; -1 for one per core"
    )
    _add_output_args(cv)
    cv.set_defaults(handler=_cmd_cv)

    loss = commands.add_parser(
        "loss",
        help="species expected to be lost with habitat",
        description="Species expected to be lost when an area loses habitat, from the model "
        "average, with bootstrap intervals. An eventual loss, and likely an overestimate: see "
        "the documentation.",
    )
    _add_data_args(loss)
    loss.add_argument(
        "--area", type=float, nargs="+", required=True, help="the area before the loss"
    )
    after = loss.add_mutually_exclusive_group(required=True)
    after.add_argument("--remaining", type=float, nargs="+", help="the area left")
    after.add_argument(
        "--fraction-lost", type=float, nargs="+", metavar="F", help="the share lost, 0 to 1"
    )
    loss.add_argument(
        "-m", "--models", action="append", metavar="NAME", help="models to average (default: all)"
    )
    loss.add_argument("--ic", choices=["AICc", "AIC", "BIC"], default="AICc")
    loss.add_argument(
        "--n-boot", type=int, default=1000, metavar="N", help="resamples (default 1000; 0: none)"
    )
    loss.add_argument("--conf", type=float, default=0.95, help="interval level (default 0.95)")
    loss.add_argument("--seed", type=int, help="seed for the resamples, for reproducible intervals")
    loss.add_argument(
        "--n-jobs", type=int, default=1, metavar="N", help="processes to use; -1 for one per core"
    )
    _add_output_args(loss)
    loss.set_defaults(handler=_cmd_loss)

    many = commands.add_parser(
        "many",
        help="the same models on many datasets, in one table",
        description="Fit the same models to many datasets and summarize them in one table: "
        "each dataset's best averaged model and its weight, the log-log fit's z, and the "
        "threshold model chosen (see sar_many). Give several CSVs or bundled datasets, or "
        "one CSV and --by naming the column that says which dataset each row is in.",
    )
    many.add_argument(
        "data", nargs="+", help="CSV files, - for standard input, or bundled datasets' names"
    )
    many.add_argument("--by", metavar="COLUMN", help="the column naming each row's dataset")
    many.add_argument("--area-col", default="area", help="the area column (default: area)")
    many.add_argument(
        "--species-col", default="species", help="the richness column (default: species)"
    )
    many.add_argument(
        "-m", "--models", action="append", metavar="NAME", help="models to average (default: all)"
    )
    many.add_argument("--ic", choices=["AICc", "AIC", "BIC"], default="AICc")
    many.add_argument("--no-threshold", action="store_true", help="skip the threshold models")
    many.add_argument(
        "--n-jobs", type=int, default=1, metavar="N", help="processes to use; -1 for one per core"
    )
    _add_output_args(many)
    many.set_defaults(handler=_cmd_many)

    loglog = commands.add_parser(
        "loglog",
        help="the log-log power model, by least squares",
        description="Fit log S = log c + z log A by ordinary least squares, with standard "
        "errors, t tests and p values (see lin_pow).",
    )
    _add_data_args(loglog)
    loglog.add_argument(
        "--log-base", choices=["e", "2", "10"], default="e", help="base of the logs (default: e)"
    )
    loglog.add_argument(
        "--con",
        type=float,
        default=1.0,
        help="added to every richness when an island has none (default 1)",
    )
    loglog.add_argument(
        "--compare", action="store_true", help="also fit the nonlinear power model, to compare"
    )
    _add_output_args(loglog)
    loglog.set_defaults(handler=_cmd_loglog)

    ci = commands.add_parser(
        "ci",
        help="bootstrap intervals on the model-averaged curve",
        description="The model-averaged richness at the given areas, with bootstrap "
        "intervals (see bootstrap_ci). The fast method refits each resample from the "
        "original fits; the full method searches every model's start grid on each.",
    )
    _add_data_args(ci)
    ci.add_argument(
        "--area",
        type=float,
        nargs="+",
        metavar="A",
        help="areas to predict at (default: ten spread over the data's range)",
    )
    ci.add_argument(
        "-m", "--models", action="append", metavar="NAME", help="models to average (default: all)"
    )
    ci.add_argument("--ic", choices=["AICc", "AIC", "BIC"], default="AICc")
    ci.add_argument(
        "--method", choices=["fast", "full"], default="fast", help="refits per resample"
    )
    ci.add_argument(
        "--n-boot", type=int, default=1000, metavar="N", help="resamples (default 1000)"
    )
    ci.add_argument("--conf", type=float, default=0.95, help="interval level (default 0.95)")
    ci.add_argument("--seed", type=int, help="seed for the resamples, for reproducible intervals")
    ci.add_argument(
        "--n-jobs", type=int, default=1, metavar="N", help="processes to use; -1 for one per core"
    )
    _add_check_args(ci)
    _add_output_args(ci)
    ci.set_defaults(handler=_cmd_ci)

    gdm = commands.add_parser(
        "gdm",
        help="fit the general dynamic model (area and island age)",
        description="Fit the general dynamic model of oceanic island biogeography: richness "
        "from island area and a hump in island age (time and time squared). The data have a "
        "column of island age beside area and richness.",
    )
    gdm.add_argument("data", help="a CSV file, or - for standard input")
    gdm.add_argument("--area-col", default="area", help="the area column (default: area)")
    gdm.add_argument(
        "--species-col", default="species", help="the richness column (default: species)"
    )
    gdm.add_argument("--time-col", default="time", help="the island age column (default: time)")
    gdm.add_argument(
        "-m",
        "--model",
        choices=["loga", "linear", "power_area", "power_area_time", "all"],
        default="loga",
        help="how area and time enter (default: loga); `all` fits and ranks the four",
    )
    gdm.add_argument(
        "--mod-sel",
        action="store_true",
        help="also compare with the simpler models: area and time, area only, intercept only",
    )
    _add_output_args(gdm)
    gdm.set_defaults(handler=_cmd_gdm)

    habitat = commands.add_parser(
        "habitat",
        help="fit the habitat models (area and number of habitats)",
        description="Fit the habitat species-area models (choros, jigsaw, Kallimanis and the "
        "curve on area alone) and rank them by AICc. The data have a column for the number "
        "of habitats (or another measure of heterogeneity) beside area and richness.",
    )
    habitat.add_argument(
        "data",
        help="a CSV file, - for standard input, or the bundled table `habitat`",
    )
    habitat.add_argument("--area-col", default="area", help="the area column (default: area)")
    habitat.add_argument(
        "--habitats-col", default="habitats", help="the habitats column (default: habitats)"
    )
    habitat.add_argument(
        "--species-col", default="species", help="the richness column (default: species)"
    )
    habitat.add_argument(
        "--mod-type",
        choices=["power_log", "power", "logarithmic"],
        default="power_log",
        help="power forms fitted on the logs (default), power forms by non-linear least "
        "squares, or the logarithmic forms",
    )
    habitat.add_argument(
        "--con",
        type=float,
        help="added to every richness before taking logs (power_log, when an island has none)",
    )
    habitat.add_argument(
        "--log-base", choices=["e", "2", "10"], default="e", help="the logarithm (default: e)"
    )
    _add_output_args(habitat)
    habitat.set_defaults(handler=_cmd_habitat)

    countryside = commands.add_parser(
        "countryside",
        help="fit the countryside model (areas of several habitats)",
        description="Fit the countryside species-area model: the richness of each species "
        "group as a curve of a weighted sum of the habitat areas. The data have one row per "
        "site, a column of area for each habitat and a column of richness for each group.",
    )
    countryside.add_argument(
        "data",
        help="a CSV file, - for standard input, or the bundled table `countryside`",
    )
    countryside.add_argument(
        "--habitats",
        action="append",
        metavar="COLUMN",
        help="the habitat area columns (repeated or comma-separated)",
    )
    countryside.add_argument(
        "--species",
        action="append",
        metavar="COLUMN",
        help="the species group richness columns (repeated or comma-separated)",
    )
    countryside.add_argument(
        "--mod-type", choices=["power", "logarithmic"], default="power", help="default: power"
    )
    countryside.add_argument(
        "--grid-start",
        choices=["partial", "exhaustive", "none"],
        default="partial",
        help="how many starting values to try (default: partial)",
    )
    countryside.add_argument(
        "--z-lower", type=float, default=0.0, metavar="Z", help="the smallest z allowed (default 0)"
    )
    countryside.add_argument(
        "--n-starts",
        type=_n_starts,
        default=40,
        metavar="N",
        help="fit from the N most promising starting values (default 40), or `all`",
    )
    _add_output_args(countryside)
    countryside.set_defaults(handler=_cmd_countryside)

    predict = commands.add_parser(
        "predict",
        help="predict richness from a saved result",
        description="Predict species richness at the given areas from a result saved with --save.",
    )
    predict.add_argument("fit", type=Path, help="a result saved with --save")
    predict.add_argument("areas", nargs="+", type=float, metavar="AREA", help="areas to predict at")
    predict.add_argument(
        "--time",
        nargs="+",
        type=float,
        metavar="T",
        help="island ages, for a general dynamic model: one for each area, or one for all",
    )
    _add_output_args(predict, save=False)
    predict.set_defaults(handler=_cmd_predict)

    serve = commands.add_parser(
        "serve",
        help="run the HTTP API (needs the [server] extra)",
        description="Serve the sars commands over HTTP (see the documentation's Server page). "
        "Needs the [server] extra: pip install 'sars[server]'.",
    )
    serve.add_argument(
        "--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)"
    )
    serve.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    serve.add_argument(
        "--token-file",
        type=Path,
        metavar="FILE",
        help="require 'Authorization: Bearer <token>', the token read from this file "
        "(or set SARS_SERVER_TOKEN)",
    )
    serve.add_argument(
        "--timeout", type=float, default=600.0, help="seconds a command may run (default: 600)"
    )
    serve.set_defaults(handler=None)

    datasets = commands.add_parser(
        "datasets",
        help="list the bundled datasets, or write one as CSV",
        description="List the bundled datasets and the tables for the commands that read "
        "more than area and richness, or write the named one as CSV.",
    )
    datasets.add_argument("name", nargs="?", help="a dataset to write as CSV")
    _add_output_args(datasets, save=False)
    datasets.set_defaults(handler=_cmd_datasets)
    return parser


def _serve(args: argparse.Namespace) -> int:
    """Run the HTTP API until interrupted."""
    import os

    try:
        import uvicorn

        from sars.server import ServerConfig, create_app
    except ImportError as exc:
        print(f"sars serve: {exc}", file=sys.stderr)
        return 1
    if not args.timeout > 0:
        print("sars serve: --timeout must be a positive number of seconds.", file=sys.stderr)
        return 2
    token = os.environ.get("SARS_SERVER_TOKEN", "").strip() or None
    if args.token_file is not None:
        try:
            token = args.token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"sars serve: can't read the token file: {exc}", file=sys.stderr)
            return 1
        if not token:
            print(f"sars serve: the token file {args.token_file} is empty.", file=sys.stderr)
            return 1
    if token is None and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: serving on {args.host} without a token: anyone who can reach it can run "
            "fits on this machine. Use --token-file, or reach it through an SSH tunnel.",
            file=sys.stderr,
        )
    app = create_app(ServerConfig(token=token, timeout=args.timeout, backend=args.backend))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``sars`` command; returns the exit code: 0, 1 when the command
    fails, 2 when the command line can't be parsed."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        if "--json" in argv:
            command = next((a for a in argv if not a.startswith("-")), None)
            usage: dict[str, Any] = {
                "sars": ops.version(),
                "command": command,
                "ok": False,
                "error": {"type": "UsageError", "message": str(exc)},
                "warnings": [],
            }
            print(ops.envelope_json(usage))
        else:
            parser.print_usage(sys.stderr)
            print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.backend != "numpy":
        import sars

        try:
            sars.set_backend(args.backend)
        except (ImportError, ValueError) as exc:  # no GPU, or a bad $SARS_BACKEND
            if getattr(args, "json", False):  # serve has no --json
                failed = {
                    "sars": ops.version(),
                    "command": args.command,
                    "backend": args.backend,
                    "ok": False,
                    "error": {"type": "InputError", "message": str(exc)},
                    "warnings": [],
                }
                print(ops.envelope_json(failed))
            else:
                print(f"sars: {exc}", file=sys.stderr)
            return 1
    if args.command == "serve":
        return _serve(args)
    shown: list[Printer] = []

    def call() -> dict[str, Any]:
        result, show = args.handler(args)
        shown.append(show)
        return result

    # Without --json, an error that isn't the input's fault (a bug) keeps its
    # traceback for the report.
    envelope = ops.envelope(args.command, call, reraise_bugs=not args.json)
    if args.json:
        output = ops.envelope_json(envelope)  # may report a value JSON can't hold
        print(output)
        return 0 if json.loads(output)["ok"] else 1
    for text in envelope["warnings"]:
        print(f"warning: {text}", file=sys.stderr)
    if not envelope["ok"]:
        print(f"sars {args.command}: {envelope['error']['message']}", file=sys.stderr)
        return 1
    shown[0](sys.stdout)
    if envelope["result"].get("saved"):
        print(f"saved to {envelope['result']['saved']}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
