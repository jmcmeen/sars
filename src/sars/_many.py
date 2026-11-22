"""Fitting the same models to many datasets in one call."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sars._inference import AveragedSAR, sar_average
from sars._linpow import LinPowFit, lin_pow
from sars._models import _model_spec
from sars._parallel import map_in_order, n_workers
from sars._threshold import THRESHOLD_MODELS, ThresholdFit, sar_threshold

_SUMMARY_COLUMNS = [
    "dataset",
    "n",
    "area_min",
    "area_max",
    "z",
    "R2",
    "best_model",
    "best_weight",
    "threshold_model",
    "breakpoint_area",
    "not_fitted",
    "warnings",
    "error",
]


@dataclass
class ManySARFit:
    """Result of fitting the same models to many datasets with `sar_many`.

    Attributes
    ----------
    summary : pd.DataFrame
        One row per dataset, in the order given: ``dataset`` (its name), ``n``
        (islands), ``area_min`` and ``area_max``, ``z`` and ``R2`` (the log-log
        power fit, `lin_pow`), ``best_model`` and ``best_weight`` (the top
        model of the multi-model average under ``ic`` and its weight), ``threshold_model``
        and ``breakpoint_area`` (the model `sar_threshold` chose and its first
        breakpoint; NaN for Linear, or without the threshold fit),
        ``not_fitted`` (the models the average couldn't fit, each with the
        reason), ``warnings`` (each warning fitting this dataset raised, with
        its category) and ``error`` (the fits that failed and why, or empty;
        the others are kept).
    averages : dict[str, AveragedSAR]
        Each fitted dataset's multi-model average, by name.
    log_log : dict[str, LinPowFit]
        Each fitted dataset's log-log power fit, by name.
    thresholds : dict[str, ThresholdFit]
        Each fitted dataset's threshold fit, by name (empty with
        ``threshold=False``).
    """

    summary: pd.DataFrame = field(repr=False)
    averages: dict[str, AveragedSAR] = field(default_factory=dict, repr=False)
    log_log: dict[str, LinPowFit] = field(default_factory=dict, repr=False)
    thresholds: dict[str, ThresholdFit] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        failed = int((self.summary["error"] != "").sum()) if "error" in self.summary else 0
        return f"ManySARFit({len(self.summary)} datasets, {failed} with errors)"


@dataclass(frozen=True)
class _Outcome:
    """One dataset's fits, or the error that stopped them, and its warnings."""

    average: AveragedSAR | None = None
    log_log: LinPowFit | None = None
    threshold: ThresholdFit | None = None
    error: str = ""
    warned: tuple[str, ...] = ()


def sar_many(
    datasets: Mapping[str, pd.DataFrame] | pd.DataFrame,
    by: str | None = None,
    models: str | list[str] = "all",
    ic: str = "AICc",
    threshold: bool | list[str] = True,
    n_jobs: int = 1,
) -> ManySARFit:
    """Fit the same models to many datasets and summarise them in one table.

    Each dataset gets `sar_average` (with ``models`` and ``ic``), `lin_pow`
    and, unless ``threshold`` is False, `sar_threshold`. A dataset that can't
    be fitted (too few islands, an island with no species, ...) gets its error
    in the summary instead of stopping the others, and one warning at the end
    says how many had errors. Each fit runs on its own, so a dataset too small
    for the threshold models still gets its average and log-log fit. The
    warnings each dataset raises are listed against it, in the summary's
    ``warnings`` column, instead of being shown; the caller's warning filters
    don't apply to them.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame] or pd.DataFrame
        The datasets, each with 'area' and 'species' columns: a mapping of
        name to DataFrame, or one DataFrame holding them all, with ``by``
        naming the column that says which dataset each row is in.
    by : str, optional
        With a single DataFrame, the column that names each row's dataset.
        The datasets are taken in the order they first appear.
    models : str or list[str]
        Models for the multi-model average: "all" (default) or a list.
    ic : str
        Information criterion for the average's weights: "AICc" (default),
        "AIC" or "BIC".
    threshold : bool or list[str]
        Fit the threshold models too (True, the default, for `sar_threshold`'s
        default set), a list of threshold models to fit, or False for none.
    n_jobs : int
        Processes to fit the datasets in: 1 (the default) in this one, -1 one
        per core. The result doesn't depend on it. In a script, run the call
        under ``if __name__ == "__main__":``: each worker process imports the
        script.

    Returns
    -------
    ManySARFit
        The summary table and each dataset's fits.

    Raises
    ------
    TypeError
        If ``datasets`` is neither a mapping nor a DataFrame, or a mapping's
        value isn't a DataFrame.
    ValueError
        If there are no datasets, ``by`` is missing or not a column (or given
        with a mapping), a row has no dataset name, a name is repeated, a model
        in ``models`` or ``threshold`` is unknown (or ``threshold`` is a single
        string), ``ic`` is unknown, or ``n_jobs`` isn't a positive whole number
        or -1.

    Warns
    -----
    UserWarning
        If any fit failed on any dataset.

    Examples
    --------
    >>> many = sars.sar_many({"galap": sars.load_galap(), "xu": sars.load_xu()})
    >>> many.summary[["dataset", "n", "z", "best_model", "threshold_model"]]
    """
    named = _named_datasets(datasets, by)
    if ic not in ("AICc", "AIC", "BIC"):
        raise ValueError(f"Unsupported IC: {ic!r}. Choose from ['AIC', 'AICc', 'BIC'].")
    n_workers(n_jobs)  # a bad n_jobs fails here, before any fitting
    _check_models(models, threshold)

    shared = {"models": models, "ic": ic, "threshold": threshold}
    outcomes = map_in_order(_fit_one, list(named.values()), n_jobs, shared)

    rows = []
    result = ManySARFit(summary=pd.DataFrame())
    for (name, data), outcome in zip(named.items(), outcomes, strict=True):
        rows.append(_summary_row(name, data, outcome))
        if outcome.average is not None:
            result.averages[name] = outcome.average
        if outcome.log_log is not None:
            result.log_log[name] = outcome.log_log
        if outcome.threshold is not None:
            result.thresholds[name] = outcome.threshold
    result.summary = pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    failed = result.summary.loc[result.summary["error"] != "", "dataset"].tolist()
    if failed:
        listed = ", ".join(str(name) for name in failed[:5])
        more = f" and {len(failed) - 5} more" if len(failed) > 5 else ""
        warnings.warn(
            f"{len(failed)} of {len(named)} datasets had a fit that failed ({listed}{more}); "
            "the summary's error column says which and why.",
            stacklevel=2,
        )
    return result


def _check_models(models: str | list[str], threshold: bool | list[str]) -> None:
    """Refuse unknown model names before any dataset is fitted, rather than
    filling the summary with the same error for every one."""
    if isinstance(models, str):
        if models != "all":
            raise ValueError(f'models must be "all" or a list of model names, got {models!r}.')
    else:
        for name in models:
            _model_spec(name)  # raises for an unknown name
    if isinstance(threshold, bool):
        return
    if isinstance(threshold, str):
        raise ValueError(
            f"threshold must be True, False or a list of threshold models, got {threshold!r}; "
            f"for one model, pass [{threshold!r}]."
        )
    unknown = [name for name in threshold if name not in THRESHOLD_MODELS]
    if unknown:
        raise ValueError(f"Unknown threshold model(s) {unknown}; choose from {THRESHOLD_MODELS}.")


def _named_datasets(
    datasets: Mapping[str, pd.DataFrame] | pd.DataFrame, by: str | None
) -> dict[str, pd.DataFrame]:
    """The datasets as an ordered mapping of name to DataFrame."""
    if isinstance(datasets, pd.DataFrame):
        if by is None:
            raise ValueError("Pass by= to name the column that says which dataset each row is in.")
        if by not in datasets.columns:
            raise ValueError(f"by={by!r} is not a column; columns are {list(datasets.columns)}.")
        if datasets[by].isna().any():
            # groupby would drop these rows without a word.
            raise ValueError(f"Some rows have no dataset name in column {by!r}.")
        groups = [
            (str(name), group.drop(columns=by).reset_index(drop=True))
            for name, group in datasets.groupby(by, sort=False, observed=True)
        ]
        names = [name for name, _ in groups]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Dataset names must be unique as strings; repeated: {repeated}.")
        named = dict(groups)
    elif isinstance(datasets, Mapping):
        if by is not None:
            raise ValueError("by= is for a single DataFrame; a mapping names its datasets itself.")
        names = [str(name) for name in datasets]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Dataset names must be unique as strings; repeated: {repeated}.")
        named = dict(zip(names, datasets.values(), strict=True))
        not_frames = [name for name, data in named.items() if not isinstance(data, pd.DataFrame)]
        if not_frames:
            raise TypeError(f"Every dataset must be a DataFrame; not for {not_frames}.")
    else:
        raise TypeError(
            "datasets must be a mapping of name to DataFrame, or a DataFrame with by=, "
            f"got {type(datasets).__name__}."
        )
    if not named:
        raise ValueError("No datasets given.")
    return named


def _fit_one(
    data: pd.DataFrame, models: str | list[str], ic: str, threshold: bool | list[str]
) -> _Outcome:
    """Fit one dataset (``sar_many``'s per-dataset work). Module level, and free
    of shared state, so it can run in a worker process."""
    # Each fit on its own: one that fails (the threshold models need more
    # islands than the average does) mustn't throw away the others.
    fits: dict[str, Any] = {}
    errors = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        steps: list[tuple[str, Callable[[], Any]]] = [
            ("sar_average", lambda: sar_average(data, models=models, ic=ic)),
            ("lin_pow", lambda: lin_pow(data)),
        ]
        if threshold is not False:
            chosen = None if threshold is True else list(threshold)
            steps.append(("sar_threshold", lambda: sar_threshold(data, models=chosen)))
        for step, fit in steps:
            try:
                fits[step] = fit()
            except Exception as exc:
                errors.append(f"{step}: {type(exc).__name__}: {exc}")
    return _Outcome(
        fits.get("sar_average"),
        fits.get("lin_pow"),
        fits.get("sar_threshold"),
        error="; ".join(errors),
        warned=_messages(caught),
    )


def _messages(caught: list[warnings.WarningMessage]) -> tuple[str, ...]:
    """Each distinct warning, with its category, once, in the order first raised."""
    return tuple(dict.fromkeys(f"{w.category.__name__}: {w.message}" for w in caught))


def _summary_row(name: str, data: pd.DataFrame, outcome: _Outcome) -> dict[str, object]:
    area = pd.to_numeric(data["area"], errors="coerce") if "area" in data else pd.Series()
    row: dict[str, object] = {
        "dataset": name,
        "n": len(data),
        "area_min": float(area.min()) if len(area) else np.nan,
        "area_max": float(area.max()) if len(area) else np.nan,
        "z": np.nan,
        "R2": np.nan,
        "best_model": "",
        "best_weight": np.nan,
        "threshold_model": "",
        "breakpoint_area": np.nan,
        "not_fitted": [],
        "warnings": list(outcome.warned),
        "error": outcome.error,
    }
    if outcome.average is not None:
        # The model with the largest weight under the chosen IC (the summary is
        # always in AICc order).
        weights = outcome.average.weights
        best = max(weights, key=lambda model: weights[model])
        row["best_model"] = best
        row["best_weight"] = float(weights[best])
        row["not_fitted"] = [
            f"{model}: {reason}" for model, reason in outcome.average.multi.not_fitted.items()
        ]
    if outcome.log_log is not None:
        row["z"] = outcome.log_log.z
        row["R2"] = outcome.log_log.r_squared
    if outcome.threshold is not None:
        row["threshold_model"] = outcome.threshold.best_model
        breakpoint_area = outcome.threshold.breakpoint_area
        row["breakpoint_area"] = np.nan if breakpoint_area is None else breakpoint_area
    return row
