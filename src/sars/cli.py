"""The ``sars`` command line: fit, average, threshold, predict and datasets.

Built on the standard library's argparse, so it needs no extra. Every command
can print a JSON envelope (``--json``)::

    {"sars": "<version>", "command": "fit", "ok": true,
     "result": {...}, "warnings": [...]}

or, on failure, ``"ok": false`` and ``"error": {"type": ..., "message": ...}``.
That envelope is the contract scripts and a server use to run the CLI.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# The datasets bundled with the package, by the name the CLI takes, and the
# function that loads each.
DATASETS = {
    "galap": "load_galap",
    "gala": "load_gala",
    "aegean": "load_aegean",
    "aegean2": "load_aegean2",
    "niering": "load_niering",
    "schrader": "load_schrader",
    "xu": "load_xu",
}

THRESHOLD_MODELS = ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo"]


class CLIError(Exception):
    """A problem with what the command was given (a missing file, a column
    that isn't there): reported as an error, without a traceback."""


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("sars")
    except Exception:  # pragma: no cover - only in an uninstalled checkout
        return "0.0.0.dev0"


# ---------------------------------------------------------------------------
# Reading data
# ---------------------------------------------------------------------------


def _read_data(source: str, area_col: str, species_col: str, label_col: str | None = None) -> Any:
    """A data frame of area and species from a CSV file, standard input
    (``-``) or a bundled dataset's name. ``label_col`` keeps a column naming the
    islands too: its name, or "auto" for the first of island, site, locality
    and name found."""
    import pandas as pd

    import sars

    if source == "-" or Path(source).is_file():
        label = "standard input" if source == "-" else source
        try:
            text = (
                sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8-sig")
            )
            frame = pd.read_csv(io.StringIO(text))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise CLIError(f"Couldn't read {label} as CSV: {exc}") from exc
    elif source in DATASETS:
        frame = getattr(sars, DATASETS[source])()
    else:
        raise CLIError(f"No such file: {source} (nor a bundled dataset: {', '.join(DATASETS)}).")
    for column in (area_col, species_col):
        if column not in frame.columns:
            raise CLIError(
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
            raise CLIError(
                f"No column {label_col!r} in {source}; its columns are {list(frame.columns)}."
            )
        if label_col is not None and label_col not in data.columns:
            data[label_col] = frame[label_col].to_numpy()
    return data


def _model_list(values: Sequence[str] | None) -> list[str] | None:
    """Model names from repeated or comma-separated options."""
    if not values:
        return None
    names = [name.strip() for value in values for name in value.split(",") if name.strip()]
    if not names:
        raise CLIError(f"No model names in {' '.join(repr(v) for v in values)}.")
    return names


# ---------------------------------------------------------------------------
# Summaries: what each result looks like in the envelope and on screen
# ---------------------------------------------------------------------------


def _number(value: Any) -> Any:
    """A JSON-safe number: None where it's undefined or infinite (an AICc at
    small n), so the envelope stays standard JSON."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _records(frame: Any) -> list[dict[str, Any]]:
    """A table's rows as plain JSON values: whole numbers stay integers,
    undefined numbers and missing values become None."""
    return [
        {key: _cell_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _cell_value(value: Any) -> Any:
    import numpy as np
    import pandas as pd

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    return value


def _summary(obj: Any) -> dict[str, Any]:
    """The readable part of a result, as plain JSON values."""
    kind = type(obj).__name__
    if kind == "SARFit":
        return {
            "type": kind,
            "model": obj.model,
            "params": {k: _number(v) for k, v in obj.params.items()},
            "R2": _number(obj.r_squared),
            "AIC": _number(obj.aic),
            "AICc": _number(obj.aicc),
            "BIC": _number(obj.bic),
            "n": obj.n,
            "converged": bool(obj.converged),
        }
    if kind == "MultiSARFit":
        return {
            "type": kind,
            "best": obj.summary.iloc[0]["model"] if len(obj.summary) else None,
            "models": _records(obj.summary),
            "excluded": dict(obj.excluded),
            "not_fitted": dict(obj.not_fitted),
        }
    if kind == "AveragedSAR":
        return {
            "type": kind,
            "ic": obj.ic,
            "weights": {k: _number(v) for k, v in obj.weights.items()},
            "models": _records(_averaged_table(obj)),
            "excluded": dict(obj.multi.excluded),
            "not_fitted": dict(obj.multi.not_fitted),
        }
    if kind == "ThresholdFit":
        return {
            "type": kind,
            "best": obj.best_model,
            "breakpoints": [_number(b) for b in obj.best_breakpoints],
            "log_area": obj.log_area,
            "segments": [{k: _number(v) for k, v in s.items()} for s in obj.best_segments],
            "models": _records(obj.summary),
        }
    if kind == "HabitatLoss":
        return {
            "type": kind,
            "conf": obj.conf,
            "ic": obj.average.ic,
            "scenarios": _records(obj.table),
        }
    if kind == "LOOCV":
        return {
            "type": kind,
            "best": obj.summary.iloc[0]["model"],
            "warm_start": obj.warm_start,
            "models": _records(obj.summary),
            "not_fitted": dict(obj.multi.not_fitted),
        }
    raise CLIError(f"Can't summarise a {kind}.")  # pragma: no cover


def _averaged_table(obj: Any) -> Any:
    """The fitted models ranked by the average's criterion, with its
    differences and weights (``multi.summary`` is always by AICc)."""
    table = obj.multi.summary[["model", "R2", "AIC", "AICc", "BIC"]].copy()
    table[f"delta_{obj.ic}"] = table[obj.ic] - table[obj.ic].min()
    table["weight"] = table["model"].map(obj.weights).astype(float)
    return table.sort_values(obj.ic, kind="stable").reset_index(drop=True)


def _print_summary(summary: dict[str, Any], out: Any) -> None:
    import pandas as pd

    kind = summary["type"]
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


def _cell(value: Any) -> str:
    """A table cell: blank where there is no value (a model without a second
    breakpoint), else four significant figures."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value if isinstance(value, str) else f"{value:.4g}"


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
    summary = _summary(obj)
    result = {"summary": summary, "saved": str(args.save) if args.save else None}
    if args.json:
        result["object"] = json.loads(sars.to_json(obj))
    return result, lambda out: _print_summary(summary, out)


def _cmd_fit(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import sars

    data = _read_data(args.data, args.area_col, args.species_col)
    models = _model_list(args.model) or ["power"]
    checks = args.normality or args.homoscedasticity or args.neg_check
    if len(models) == 1 and models[0] != "all" and not checks:
        from sars._models import _model_spec

        try:
            _model_spec(models[0])
        except ValueError as exc:
            raise CLIError(str(exc)) from exc
        obj = getattr(sars, f"sar_{models[0]}")(data)
    else:
        obj = sars.sar_multi(
            data,
            models="all" if models == ["all"] else models,
            normality=args.normality,
            homoscedasticity=args.homoscedasticity,
            neg_check=args.neg_check,
        )
        if obj.summary.empty:
            reasons = {**obj.excluded, **obj.not_fitted}
            raise CLIError(
                "No model was fitted"
                + (" or passed the checks: " if reasons else ".")
                + "; ".join(f"{m}: {r}" for m, r in reasons.items())
            )
    return _finish(obj, args)


def _cmd_average(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import sars

    data = _read_data(args.data, args.area_col, args.species_col)
    models = _model_list(args.models) or ["all"]
    obj = sars.sar_average(
        data,
        models="all" if models == ["all"] else models,
        ic=args.ic,
        normality=args.normality,
        homoscedasticity=args.homoscedasticity,
        neg_check=args.neg_check,
    )
    return _finish(obj, args)


def _cmd_threshold(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import sars

    data = _read_data(args.data, args.area_col, args.species_col)
    obj = sars.sar_threshold(
        data,
        models=_model_list(args.models),
        interval=args.interval,
        log_area=not args.no_log_area,
    )
    return _finish(obj, args)


def _cmd_cv(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import sars

    data = _read_data(
        args.data, args.area_col, args.species_col, label_col=args.label_col or "auto"
    )
    models = _model_list(args.models) or ["all"]
    obj = sars.loo_cv(
        data,
        models="all" if models == ["all"] else models,
        warm_start=args.warm,
        ic=args.ic,
        n_jobs=args.n_jobs,
    )
    return _finish(obj, args)


def _cmd_loss(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import numpy as np

    import sars

    data = _read_data(args.data, args.area_col, args.species_col)
    models = _model_list(args.models) or ["all"]
    obj = sars.habitat_loss(
        data,
        area=args.area,
        remaining=args.remaining,
        fraction_lost=args.fraction_lost,
        models="all" if models == ["all"] else models,
        ic=args.ic,
        n_boot=args.n_boot,
        conf=args.conf,
        rng=np.random.default_rng(args.seed),
        n_jobs=args.n_jobs,
    )
    return _finish(obj, args)


def _cmd_predict(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import numpy as np

    import sars

    try:
        obj = sars.load(args.fit)
    except FileNotFoundError as exc:
        raise CLIError(f"No such file: {args.fit}") from exc
    if not hasattr(obj, "predict"):
        raise CLIError(
            f"A saved {type(obj).__name__} doesn't predict; save a single fit, an "
            "average or a threshold fit instead."
        )
    areas = np.asarray(args.areas, dtype=float)
    if not np.isfinite(areas).all():
        raise CLIError("Areas must be finite numbers.")
    species = np.asarray(obj.predict(areas), dtype=float)
    rows = [{"area": float(a), "species": _number(s)} for a, s in zip(areas, species, strict=True)]
    result = {"type": type(obj).__name__, "predictions": rows}

    def show(out: Any) -> None:
        for row in rows:
            print(f"{row['area']:.6g}\t{_fmt(row['species'])}", file=out)

    return result, show


def _cmd_datasets(args: argparse.Namespace) -> tuple[dict[str, Any], Printer]:
    import sars

    if args.name is None:
        names = []
        for name, loader in DATASETS.items():
            frame = getattr(sars, loader)()
            names.append(
                {
                    "name": name,
                    "n": len(frame),
                    "area_min": float(frame["area"].min()),
                    "area_max": float(frame["area"].max()),
                    "columns": [str(c) for c in frame.columns],
                }
            )

        def show(out: Any) -> None:
            for row in names:
                print(
                    f"{row['name']:<10} {row['n']:>4} sites  area {row['area_min']:.4g} to "
                    f"{row['area_max']:.4g}",
                    file=out,
                )

        return {"datasets": names}, show
    if args.name not in DATASETS:
        raise CLIError(f"No bundled dataset {args.name!r}; choose from {', '.join(DATASETS)}.")
    frame = getattr(sars, DATASETS[args.name])()
    text = frame.to_csv(index=False)
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


def build_parser() -> argparse.ArgumentParser:
    """The ``sars`` argument parser."""
    parser = _Parser(prog="sars", description="Fit species-area relationship models.")
    parser.add_argument("--version", action="version", version=f"sars {_version()}")
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

    predict = commands.add_parser(
        "predict",
        help="predict richness from a saved result",
        description="Predict species richness at the given areas from a result saved with --save.",
    )
    predict.add_argument("fit", type=Path, help="a result saved with --save")
    predict.add_argument("areas", nargs="+", type=float, metavar="AREA", help="areas to predict at")
    _add_output_args(predict, save=False)
    predict.set_defaults(handler=_cmd_predict)

    datasets = commands.add_parser(
        "datasets",
        help="list the bundled datasets, or write one as CSV",
        description="List the bundled datasets, or write the named one as CSV.",
    )
    datasets.add_argument("name", nargs="?", help="a dataset to write as CSV")
    _add_output_args(datasets, save=False)
    datasets.set_defaults(handler=_cmd_datasets)
    return parser


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
                "sars": _version(),
                "command": command,
                "ok": False,
                "error": {"type": "UsageError", "message": str(exc)},
                "warnings": [],
            }
            print(json.dumps(usage, indent=2))
        else:
            parser.print_usage(sys.stderr)
            print(f"error: {exc}", file=sys.stderr)
        return 2
    envelope: dict[str, Any] = {"sars": _version(), "command": args.command}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result, show = args.handler(args)
        except Exception as exc:
            expected = (CLIError, ValueError, KeyError, RuntimeError, OSError)
            if not args.json and not isinstance(exc, expected):
                raise  # a bug: keep the traceback for the report
            message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
            envelope.update(ok=False, error={"type": type(exc).__name__, "message": message})
            result, show = None, None
    if show is not None:
        envelope.update(ok=True, result=result)
    seen: list[str] = []
    for warning in caught:
        text = str(warning.message)
        if text not in seen:
            seen.append(text)
    envelope["warnings"] = seen
    if args.json:
        try:
            output = json.dumps(envelope, indent=2, allow_nan=False)
        except (TypeError, ValueError) as exc:  # a value JSON can't hold: report it
            envelope = {key: envelope[key] for key in ("sars", "command", "warnings")}
            envelope.update(ok=False, error={"type": type(exc).__name__, "message": str(exc)})
            output = json.dumps(envelope, indent=2)
        print(output)
        return 0 if envelope["ok"] else 1
    for text in seen:
        print(f"warning: {text}", file=sys.stderr)
    if show is None:
        print(f"sars {args.command}: {envelope['error']['message']}", file=sys.stderr)
    else:
        show(sys.stdout)
        if result.get("saved"):
            print(f"saved to {result['saved']}", file=sys.stderr)
    return 0 if envelope["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
