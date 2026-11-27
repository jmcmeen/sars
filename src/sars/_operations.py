"""The work behind each command, shared by the command line and the HTTP API.

Each operation takes plain arguments (a data frame, model names, options) and
returns its result; ``result`` turns a result into JSON-ready values, and
``envelope`` runs an operation and wraps what it gives, the warnings it raised,
or its error, in the envelope both front ends return::

    {"sars": "<version>", "command": "fit", "ok": true,
     "result": {...}, "warnings": [...]}

or, on failure, ``"ok": false`` and ``"error": {"type": ..., "message": ...}``.

Neither front end imports the other: the command line reads its arguments and
files, calls an operation and prints; the server validates a request, calls
the same operation and responds.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

# The datasets bundled with the package, by the name the front ends take, and
# the function that loads each.
DATASETS = {
    "galap": "load_galap",
    "gala": "load_gala",
    "aegean": "load_aegean",
    "aegean2": "load_aegean2",
    "niering": "load_niering",
    "schrader": "load_schrader",
    "xu": "load_xu",
}

# The bundled tables that aren't area and richness alone, for the commands
# that read more columns: the loader, the commands each is for, and the
# columns a command reads when none are named.
TABLES: dict[str, dict[str, Any]] = {
    "habitat": {"loader": "load_habitat", "commands": ["habitat"], "columns": {}},
    "countryside": {
        "loader": "load_countryside",
        "commands": ["countryside"],
        "columns": {
            "habitats": ["Area_AG", "Area_SH", "Area_QF"],
            "species": ["Spcs_AG", "Spcs_SH", "Spcs_QF", "Spcs_UB"],
        },
    },
    "xu_abundance": {
        "loader": "load_xu_abundance",
        "commands": ["coleman", "effort"],
        "columns": {},
        "areas": "xu",
    },
}

THRESHOLD_MODELS = ["ContOne", "ZslopeOne", "DiscOne", "ContTwo", "ZslopeTwo", "DiscTwo"]


# The most resample indices a bootstrap may hold at once (resamples times
# islands): 200 million, 1.6 GB, so a request can't exhaust the machine.
MAX_RESAMPLE_VALUES = 200_000_000


# The most simulated placements a Coleman test may run.
MAX_SIMULATIONS = 1_000_000
# The most simulated richness values it may hold at once (simulations times
# sites): 50 million, under a gigabyte with the counts drawn beside them.
MAX_SIMULATED_VALUES = 50_000_000


def _check_resamples(n_boot: int, n: int) -> None:
    if n_boot * n > MAX_RESAMPLE_VALUES:
        raise InputError(
            f"{n_boot} resamples of {n} islands is more than the {MAX_RESAMPLE_VALUES:,} "
            "resampled values a bootstrap here may hold; use fewer resamples."
        )


class InputError(Exception):
    """A problem with what an operation was given (a missing column, an
    unknown dataset, no model fitted): reported as an error, without a
    traceback."""


# Errors that mean the request was at fault (bad input, an unknown model, data
# that can't be fitted), not a bug; and their names, as an envelope reports
# them (with the command line's own UsageError, for arguments it can't parse).
INPUT_ERRORS: tuple[type[Exception], ...] = (
    InputError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
)
INPUT_ERROR_NAMES = frozenset(
    {"InputError", "UsageError", "ValueError", "KeyError", "RuntimeError", "OSError"}
)


def version() -> str:
    try:
        from importlib.metadata import version as installed

        return installed("sars")
    except Exception:  # pragma: no cover - only in an uninstalled checkout
        return "0.0.0.dev0"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def bundled(name: str) -> Any:
    """A bundled dataset, or one of the other bundled tables, by its short name."""
    import sars

    if name in TABLES:
        return getattr(sars, TABLES[name]["loader"])()
    if name not in DATASETS:
        raise InputError(
            f"No bundled dataset {name!r}; choose from {', '.join([*DATASETS, *TABLES])}."
        )
    return getattr(sars, DATASETS[name])()


def data_frame(
    area: Sequence[float], species: Sequence[float], label: Sequence[str] | None = None
) -> Any:
    """Species-area data from arrays, with a name per site if given."""
    import pandas as pd

    import sars

    frame = sars.from_df(pd.DataFrame({"area": list(area), "species": list(species)}))
    if label is not None:
        frame["label"] = list(label)
    return frame


def model_names(values: Sequence[str] | None) -> list[str] | None:
    """Model names from repeated or comma-separated options."""
    return _names(values, "model")


def column_names(values: Sequence[str] | None) -> list[str] | None:
    """Column names from repeated or comma-separated options."""
    return _names(values, "column")


def _names(values: Sequence[str] | None, what: str) -> list[str] | None:
    if not values:
        return None
    names = [name.strip() for value in values for name in value.split(",") if name.strip()]
    if not names:
        raise InputError(f"No {what} names in {' '.join(repr(v) for v in values)}.")
    return names


def _models(names: Sequence[str] | None, default: str) -> str | list[str]:
    names = list(names) if names else [default]
    return "all" if names == ["all"] else names


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def datasets(tables: bool = False) -> dict[str, Any]:
    """The bundled datasets (name, size, area range and columns) and, with
    ``tables``, the other bundled tables with the commands each is for."""
    rows = []
    for name in DATASETS:
        frame = bundled(name)
        rows.append(
            {
                "name": name,
                "n": len(frame),
                "area_min": float(frame["area"].min()),
                "area_max": float(frame["area"].max()),
                "columns": [str(c) for c in frame.columns],
            }
        )
    if not tables:
        return {"datasets": rows}
    others = []
    for name, spec in TABLES.items():
        frame = bundled(name)
        others.append(
            {
                "name": name,
                "n": len(frame),
                "columns": [str(c) for c in frame.columns],
                "commands": list(spec["commands"]),
            }
        )
    return {"datasets": rows, "tables": others}


def fit(
    data: Any,
    models: Sequence[str] | None = None,
    normality: str | None = None,
    homoscedasticity: str | None = None,
    neg_check: bool = False,
) -> Any:
    """One model (power by default), or several (or "all") ranked by AICc."""
    import sars

    names = list(models) if models else ["power"]
    checks = normality or homoscedasticity or neg_check
    if len(names) == 1 and names[0] != "all" and not checks:
        from sars._models import _model_spec

        try:
            _model_spec(names[0])
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        return getattr(sars, f"sar_{names[0]}")(data)
    multi = sars.sar_multi(
        data,
        models=_models(names, "power"),
        normality=normality,
        homoscedasticity=homoscedasticity,
        neg_check=neg_check,
    )
    if multi.summary.empty:
        reasons = {**multi.excluded, **multi.not_fitted}
        raise InputError(
            "No model was fitted"
            + (" or passed the checks: " if reasons else ".")
            + "; ".join(f"{m}: {r}" for m, r in reasons.items())
        )
    return multi


def average(
    data: Any,
    models: Sequence[str] | None = None,
    ic: str = "AICc",
    normality: str | None = None,
    homoscedasticity: str | None = None,
    neg_check: bool = False,
) -> Any:
    """The model average, weighted by an information criterion."""
    import sars

    return sars.sar_average(
        data,
        models=_models(models, "all"),
        ic=ic,
        normality=normality,
        homoscedasticity=homoscedasticity,
        neg_check=neg_check,
    )


def threshold(
    data: Any,
    models: Sequence[str] | None = None,
    interval: float | None = None,
    log_area: bool = True,
) -> Any:
    """Piecewise (threshold) models, one chosen by AICc."""
    import sars

    return sars.sar_threshold(
        data, models=list(models) if models else None, interval=interval, log_area=log_area
    )


def breakpoint_ci(
    fit: Any,
    model: str | None = None,
    method: str = "boot",
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int | None = None,
    n_jobs: int = 1,
) -> Any:
    """Confidence intervals for a threshold fit's breakpoints: its chosen
    model's, or ``model``'s."""
    import numpy as np

    import sars

    name = model or fit.best_model
    if name == "Linear":
        which = "Linear has" if model else "The chosen model, Linear, has"
        raise InputError(f"{which} no breakpoint: name a threshold model for the intervals.")
    return sars.threshold_ci(
        fit,
        model=name,
        n_boot=n_boot,
        conf=conf,
        rng=np.random.default_rng(seed),
        method=method,
        n_jobs=n_jobs,
    )


def cv(
    data: Any,
    models: Sequence[str] | None = None,
    warm: bool = False,
    ic: str = "AICc",
    label: str = "auto",
    n_jobs: int = 1,
) -> Any:
    """Leave-one-island-out cross-validation."""
    import sars

    return sars.loo_cv(
        data, models=_models(models, "all"), warm_start=warm, ic=ic, label=label, n_jobs=n_jobs
    )


def loss(
    data: Any,
    area: Sequence[float],
    remaining: Sequence[float] | None = None,
    fraction_lost: Sequence[float] | None = None,
    models: Sequence[str] | None = None,
    ic: str = "AICc",
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int | None = None,
    n_jobs: int = 1,
) -> Any:
    """Species expected to be lost with habitat, with bootstrap intervals."""
    import numpy as np

    import sars

    _check_resamples(n_boot, len(data))

    return sars.habitat_loss(
        data,
        area=list(area),
        remaining=None if remaining is None else list(remaining),
        fraction_lost=None if fraction_lost is None else list(fraction_lost),
        models=_models(models, "all"),
        ic=ic,
        n_boot=n_boot,
        conf=conf,
        rng=np.random.default_rng(seed),
        n_jobs=n_jobs,
    )


def ci(
    data: Any,
    areas: Sequence[float] | None = None,
    models: Sequence[str] | None = None,
    ic: str = "AICc",
    method: str = "fast",
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int | None = None,
    n_jobs: int = 1,
    normality: str | None = None,
    homoscedasticity: str | None = None,
    neg_check: bool = False,
) -> Any:
    """Bootstrap intervals on the model-averaged curve at ``areas`` (by
    default ten areas spread evenly in log over the data's range)."""
    import numpy as np

    import sars
    from sars._models import _validate_data

    if n_boot < 1:
        raise InputError(f"The number of resamples must be at least 1, got {n_boot}.")
    if not 0 < conf < 1:
        raise InputError(f"The interval level must be between 0 and 1, got {conf}.")
    _validate_data(data)  # before the default areas are taken from it
    _check_resamples(n_boot, len(data))
    if areas is None:
        at = np.geomspace(float(data["area"].min()), float(data["area"].max()), 10)
    else:
        at = np.asarray(areas, dtype=float)
        if not (np.isfinite(at).all() and (at > 0).all()):
            raise InputError("Areas must be positive, finite numbers.")
    return sars.bootstrap_ci(
        data,
        models=_models(models, "all"),
        n_boot=n_boot,
        conf=conf,
        area_grid=at,
        rng=np.random.default_rng(seed),
        method=method,
        ic=ic,
        normality=normality,
        homoscedasticity=homoscedasticity,
        neg_check=neg_check,
        n_jobs=n_jobs,
    )


def loglog(data: Any, log_base: str = "e", con: float = 1.0, compare: bool = False) -> Any:
    """The log-log power model, log S = log c + z log A, by least squares."""
    import sars

    return sars.lin_pow(data, con=con, log_base=log_base, compare=compare)


def many(
    datasets: Any,
    by: str | None = None,
    models: Sequence[str] | None = None,
    ic: str = "AICc",
    threshold: bool = True,
    n_jobs: int = 1,
) -> Any:
    """The same models on many datasets, summarized in one table: a mapping of
    name to data, or one table with ``by`` naming each row's dataset."""
    import sars

    return sars.sar_many(
        datasets, by=by, models=_models(models, "all"), ic=ic, threshold=threshold, n_jobs=n_jobs
    )


def habitat(
    data: Any,
    mod_type: str = "power_log",
    area_col: str = "area",
    habitats_col: str = "habitats",
    species_col: str = "species",
    con: float | None = None,
    log_base: str = "e",
) -> Any:
    """The habitat species-area models (choros, jigsaw, Kallimanis and the
    curve on area alone), ranked by AICc."""
    import sars

    columns = [area_col, habitats_col, species_col]
    if len(set(columns)) < 3:
        raise InputError(
            f"Area, habitats and richness must be three different columns, got {columns}."
        )
    if len(data) == 0:
        raise InputError("The table has no rows.")
    return sars.sar_habitat(
        data,
        mod_type=mod_type,
        area_col=area_col,
        habitats_col=habitats_col,
        species_col=species_col,
        con=con,
        log_base=log_base,
    )


def countryside(
    data: Any,
    habitats: Sequence[str],
    species: Sequence[str],
    mod_type: str = "power",
    grid_start: str = "partial",
    z_lower: float = 0.0,
    n_starts: int | None = 40,
) -> Any:
    """The countryside species-area model: each species group's richness
    from the areas of the habitats."""
    import sars

    for label, columns in (("habitat", habitats), ("species", species)):
        repeated = sorted({c for c in columns if list(columns).count(c) > 1})
        if repeated:
            raise InputError(f"A {label} column is named twice: {', '.join(repeated)}.")
    both = sorted(set(habitats) & set(species))
    if both:
        raise InputError(f"A column is named as both a habitat and a species: {', '.join(both)}.")
    if "group" in habitats:
        # The result's tables name each species group in a column `group`.
        raise InputError("A habitat column can't be named 'group'; rename it.")
    if len(data) == 0:
        raise InputError("The table has no rows.")
    return sars.sar_countryside(
        data,
        habitats=list(habitats),
        species=list(species),
        mod_type=mod_type,
        grid_start=grid_start,
        z_lower=z_lower,
        n_starts=n_starts,
    )


def gdm(
    data: Any,
    model: str = "loga",
    mod_sel: bool = False,
    area_col: str = "area",
    species_col: str = "species",
    time_col: str = "time",
) -> Any:
    """The general dynamic model: richness from island area and a hump in
    island age."""
    import sars

    return sars.sar_gdm(
        data,
        model=model,
        mod_sel=mod_sel,
        area_col=area_col,
        species_col=species_col,
        time_col=time_col,
    )


def abundance_table(
    table: Any,
    layout: str = "species",
    site_col: str = "site",
    species_col: str = "species",
    count_col: str | None = "count",
) -> Any:
    """A species-by-site abundance table (species in rows, a column per site)
    from one of three layouts.

    ``layout`` is "species" (already so: the first column names the species),
    "sites" (sites in rows: the first column names the sites) or "long" (one
    row per site and species, in ``site_col`` and ``species_col``, with the
    number of individuals in ``count_col``; with ``count_col`` None each row
    is one individual). A table whose index is already named (a bundled one)
    is taken as it is.
    """
    import numpy as np
    import pandas as pd

    if layout not in ("species", "sites", "long"):
        raise InputError(f"layout must be 'species', 'sites' or 'long', got {layout!r}.")
    if layout == "long":
        needed = [site_col, species_col] + ([] if count_col is None else [count_col])
        for column in needed:
            if column not in table.columns:
                raise InputError(
                    f"No column {column!r} in the long table; its columns are "
                    f"{list(table.columns)}."
                )
        if len(set(needed)) < len(needed):
            raise InputError("The site, species and count columns must be different columns.")
        if table[[site_col, species_col]].isna().any().any():
            raise InputError("The long table has rows with no site or no species.")
        if count_col is None:
            counts = pd.Series(1, index=table.index)
        else:
            if pd.api.types.is_bool_dtype(table[count_col]):
                raise InputError(f"The count column {count_col!r} must hold numbers.")
            counts = pd.to_numeric(table[count_col], errors="coerce")
            # Each row is checked, not only a pair's total: -1 and 1 aren't 0.
            whole = np.isfinite(counts) & (counts >= 0) & (counts == np.round(counts))
            if not whole.all():
                raise InputError(
                    f"The count column {count_col!r} must hold whole numbers, 0 or more, with "
                    f"none missing: see row {int(np.flatnonzero(~whole.to_numpy())[0]) + 1}."
                )
        sites = table[site_col].astype(str).str.strip()
        species = table[species_col].astype(str).str.strip()
        wide = counts.groupby([species, sites], sort=False).sum().unstack(fill_value=0)
        wide = wide.reindex(index=pd.unique(species), columns=pd.unique(sites))
    else:
        named = table.index.name is not None
        if not named and table.shape[1] < 2:
            what = "species" if layout == "species" else "sites"
            raise InputError(
                f"The table needs a first column naming the {what} and a column of counts "
                f"for each {'site' if layout == 'species' else 'species'}."
            )
        wide = table if named else table.set_index(table.columns[0])
        if wide.index.isna().any() or not wide.index.is_unique:
            what = "species" if layout == "species" else "site"
            raise InputError(f"The first column must name each {what} once, with none missing.")
        try:
            wide = wide.apply(pd.to_numeric)
        except (ValueError, TypeError) as exc:
            raise InputError(
                "Every column after the first must hold counts; one holds something that "
                f"isn't a number ({exc})."
            ) from exc
        if layout == "sites":
            wide = wide.T
    wide.index = wide.index.astype(str).str.strip()
    wide.columns = wide.columns.astype(str).str.strip()
    if not wide.index.is_unique:
        repeated = sorted(set(wide.index[wide.index.duplicated()]))
        raise InputError(f"A species is named twice: {', '.join(repeated)}.")
    wide.index.name, wide.columns.name = "species", None
    if not wide.columns.is_unique:
        repeated = sorted(set(wide.columns[wide.columns.duplicated()]))
        raise InputError(f"A site is named twice: {', '.join(repeated)}.")
    return wide


def site_areas(
    sites: Sequence[str],
    table: Any,
    site_col: str | None = None,
    area_col: str = "area",
    names_header: str | None = None,
) -> list[float]:
    """The area of each of ``sites``, in order, looked up by name in a table
    with a column naming the sites (``site_col``, or the first of island,
    site, locality and name found) and a column of areas. Sites of the table
    that aren't among ``sites`` are left out. ``names_header`` is the header
    of the abundance table's first column, which names its rows: if it is
    one of the areas table's sites, that column held counts, not names."""
    import numpy as np
    import pandas as pd

    from sars._crossval import _LABEL_GUESSES

    if site_col is None:
        lower = {str(c).lower(): c for c in table.columns}
        site_col = next((lower[g] for g in _LABEL_GUESSES if g in lower), None)
        if site_col is None:
            raise InputError(
                f"No column naming the sites in the areas table (one of "
                f"{', '.join(_LABEL_GUESSES)}); its columns are {list(table.columns)}."
            )
    for column in (site_col, area_col):
        if column not in table.columns:
            raise InputError(
                f"No column {column!r} in the areas table; its columns are {list(table.columns)}."
            )
    table = table[table[site_col].notna()]
    names = table[site_col].astype(str).str.strip()
    if names_header is not None and names_header in set(names):
        raise InputError(
            f"The abundance table's first column, {names_header!r}, is one of the sites: "
            "the first column must hold the names of the rows, not counts."
        )
    areas = pd.to_numeric(table[area_col], errors="coerce")
    # A long table repeats a site's area on each of its rows: one value each.
    pairs = pd.DataFrame({"site": names, "area": areas}).drop_duplicates()
    wanted = pairs[pairs["site"].isin(list(sites))]
    unusable = sorted(set(wanted["site"][wanted["area"].isna()]))
    if unusable:
        raise InputError(
            f"Sites with an area that is missing or not a number: {', '.join(unusable[:5])}."
        )
    repeated = sorted(set(wanted["site"][wanted["site"].duplicated()]))
    if repeated:
        raise InputError(f"Sites with more than one area: {', '.join(repeated[:5])}.")
    lookup = wanted.set_index("site")["area"]
    missing = [s for s in sites if s not in lookup.index]
    if missing:
        raise InputError(
            f"No area for {len(missing)} of the {len(sites)} sites: {', '.join(missing[:5])}"
            + (", ..." if len(missing) > 5 else "")
            + ". Sites are matched by name."
        )
    out = lookup.reindex(list(sites)).to_numpy(dtype=float)
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        bad = [s for s, a in zip(sites, out, strict=True) if not (np.isfinite(a) and a > 0)]
        raise InputError(f"Site areas must be positive numbers: see {', '.join(bad[:5])}.")
    return [float(a) for a in out]


def coleman(
    abundance: Any, areas: Sequence[float], n_sim: int = 999, seed: int | None = None
) -> tuple[Any, Any]:
    """Coleman's random placement model against observed richness, and its
    simulation test (None with ``n_sim`` 0)."""
    import numpy as np

    import sars

    if not 0 <= n_sim <= MAX_SIMULATIONS:
        raise InputError(
            f"The number of simulations must be from 0 to {MAX_SIMULATIONS:,}, got {n_sim}."
        )
    if seed is not None and seed < 0:
        raise InputError(f"The seed can't be negative, got {seed}.")
    n_sites = len(areas)
    if n_sim * n_sites > MAX_SIMULATED_VALUES:
        raise InputError(
            f"{n_sim} simulations of {n_sites} sites is more than the "
            f"{MAX_SIMULATED_VALUES:,} simulated values a test here may hold; use fewer "
            "simulations."
        )
    fit = sars.coleman(abundance, list(areas))
    test = fit.placement_test(n_sim=n_sim, rng=np.random.default_rng(seed)) if n_sim else None
    return fit, test


def coleman_result(
    fit: Any, test: Any, include_object: bool, saved: str | None = None
) -> dict[str, Any]:
    """A Coleman fit for the envelope, with its test in the summary."""
    out = result(fit, include_object=include_object, saved=saved)
    out["summary"]["test"] = (
        None
        if test is None
        else {
            "statistic": number(test.statistic),
            "p_value": number(test.p_value),
            "n_sim": test.n_sim,
            "rule_rejection_rate": number(test.rule_rejection_rate),
        }
    )
    return out


def effort(
    abundance: Any,
    areas: Sequence[float] | None = None,
    base: str = "coverage",
    level: float | None = None,
) -> dict[str, Any]:
    """Richness standardized for sampling effort, a row per site."""
    import sars

    if level is not None and not math.isfinite(level):
        raise InputError(f"The level must be a finite number, got {level}.")
    if areas is not None and not all(math.isfinite(a) and a > 0 for a in areas):
        raise InputError("Site areas must be positive numbers.")
    table = sars.standardised_richness(
        abundance, area=None if areas is None else list(areas), base=base, level=level
    )
    return {"base": base, "level": number(table["level"].iloc[0]), "sites": records(table)}


def parameters(obj: Any, conf: float = 0.95) -> dict[str, list[dict[str, Any]]]:
    """Each fitted model's parameter table (standard errors, t tests and
    confidence intervals), by model name: one model, or every model of a
    multi-model fit."""
    if not 0 < conf < 1:
        raise InputError(f"The interval level must be between 0 and 1, got {conf}.")
    fits = obj.fits if type(obj).__name__ == "MultiSARFit" else [obj]
    tables = {}
    for fit in fits:
        table = fit.param_table(conf=conf).reset_index()
        tables[fit.model] = records(table)
    return tables


def predict(
    obj: Any, areas: Sequence[float], time: Sequence[float] | None = None
) -> dict[str, Any]:
    """Predicted richness at ``areas`` from a fitted result. A general dynamic
    model needs ``time`` too: an island age for each area, or one for all."""
    import numpy as np

    kind = type(obj).__name__
    if kind == "GDMMultiFit":
        obj, kind = obj.best, "GDMFit"
    if kind == "GDMFit":
        return _predict_gdm(obj, areas, time)

    if not hasattr(obj, "predict"):
        raise InputError(
            f"A saved {type(obj).__name__} doesn't predict; save a single fit, an "
            "average or a threshold fit instead."
        )
    needs = {
        "HabitatFit": "a number of habitats for each area",
        "CountrysideFit": "an area for each habitat",
    }
    if type(obj).__name__ in needs:
        raise InputError(
            f"Predicting from a saved {type(obj).__name__} needs {needs[type(obj).__name__]}, "
            "which this command doesn't take; load it in Python (sars.load) and call its "
            "predict."
        )
    if time is not None:
        raise InputError(f"Only a general dynamic model predicts with island ages, not a {kind}.")
    at = np.asarray(areas, dtype=float)
    if not np.isfinite(at).all():
        raise InputError("Areas must be finite numbers.")
    species = np.asarray(obj.predict(at), dtype=float)
    rows = [{"area": float(a), "species": number(s)} for a, s in zip(at, species, strict=True)]
    return {"type": type(obj).__name__, "predictions": rows}


def _predict_gdm(obj: Any, areas: Sequence[float], time: Sequence[float] | None) -> dict[str, Any]:
    import numpy as np

    at = np.asarray(areas, dtype=float)
    if time is None:
        if len(obj.params) >= 3:  # the fit has a time term
            raise InputError(
                "A general dynamic model predicts from area and island age: give the ages "
                "too, one for each area or one for all."
            )
        species = np.asarray(obj.predict(at), dtype=float)
        rows = [{"area": float(a), "species": number(s)} for a, s in zip(at, species, strict=True)]
        return {"type": "GDMFit", "model": obj.model, "predictions": rows}
    if len(time) not in (1, len(areas)):
        raise InputError(
            f"Give one island age for each area, or one for all: got {len(time)} ages "
            f"for {len(areas)} areas."
        )
    ages = np.broadcast_to(np.asarray(time, dtype=float), at.shape)
    species = np.asarray(obj.predict(at, ages), dtype=float)
    rows = [
        {"area": float(a), "time": float(t), "species": number(s)}
        for a, t, s in zip(at, ages, species, strict=True)
    ]
    return {"type": "GDMFit", "model": obj.model, "predictions": rows}


def from_saved(saved: Any) -> Any:
    """A result from what ``sars.save`` wrote, as parsed JSON."""
    import sars

    try:
        return sars.from_json(json.dumps(saved))
    except (ValueError, KeyError, TypeError) as exc:
        raise InputError(f"Not a saved sars result: {exc}") from exc


# ---------------------------------------------------------------------------
# Results as JSON
# ---------------------------------------------------------------------------


def result(obj: Any, include_object: bool, saved: str | None = None) -> dict[str, Any]:
    """A result for the envelope: its summary, where it was saved, and with
    ``include_object`` the whole result in ``sars.save``'s format."""
    import sars

    out: dict[str, Any] = {"summary": summary(obj), "saved": saved}
    if include_object:
        out["object"] = json.loads(sars.to_json(obj))
    return out


def number(value: Any) -> Any:
    """A JSON-safe number: None where it's undefined or infinite (an AICc at
    small n), so the envelope stays standard JSON."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def records(frame: Any) -> list[dict[str, Any]]:
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
        return number(value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    return value


def summary(obj: Any) -> dict[str, Any]:
    """The readable part of a result, as plain JSON values."""
    kind = type(obj).__name__
    if kind == "SARFit":
        return {
            "type": kind,
            "model": obj.model,
            "params": {k: number(v) for k, v in obj.params.items()},
            "R2": number(obj.r_squared),
            "AIC": number(obj.aic),
            "AICc": number(obj.aicc),
            "BIC": number(obj.bic),
            "n": obj.n,
            "converged": bool(obj.converged),
        }
    if kind == "MultiSARFit":
        return {
            "type": kind,
            "best": obj.summary.iloc[0]["model"] if len(obj.summary) else None,
            "models": records(obj.summary),
            "excluded": dict(obj.excluded),
            "not_fitted": dict(obj.not_fitted),
        }
    if kind == "AveragedSAR":
        return {
            "type": kind,
            "ic": obj.ic,
            "weights": {k: number(v) for k, v in obj.weights.items()},
            "models": records(_averaged_table(obj)),
            "excluded": dict(obj.multi.excluded),
            "not_fitted": dict(obj.multi.not_fitted),
        }
    if kind == "ThresholdFit":
        return {
            "type": kind,
            "best": obj.best_model,
            "breakpoints": [number(b) for b in obj.best_breakpoints],
            "log_area": obj.log_area,
            "segments": [{k: number(v) for k, v in s.items()} for s in obj.best_segments],
            "models": records(obj.summary),
        }
    if kind == "HabitatLoss":
        return {
            "type": kind,
            "conf": obj.conf,
            "ic": obj.average.ic,
            "scenarios": records(obj.table),
        }
    if kind == "ThresholdCI":
        rows = [
            {
                "estimate": number(e),
                "lower": number(lo),
                "upper": number(hi),
                "estimate_area": number(ea),
                "lower_area": number(la),
                "upper_area": number(ua),
            }
            for e, lo, hi, ea, la, ua in zip(
                obj.estimates,
                obj.lower,
                obj.upper,
                obj.estimates_area,
                obj.lower_area,
                obj.upper_area,
                strict=True,
            )
        ]
        return {
            "type": kind,
            "model": obj.model,
            "method": obj.method,
            "conf": obj.conf,
            "n_boot": obj.n_boot,
            "log_area": obj.log_area,
            "breakpoints": rows,
        }
    if kind == "ManySARFit":
        return {"type": kind, "datasets": records(obj.summary)}
    if kind == "LinPowFit":
        names = ("logc", "c", "z", "logc_se", "c_se", "z_se", "logc_t", "z_t", "logc_p", "z_p")
        out = {"type": kind, **{name: number(getattr(obj, name)) for name in names}}
        out.update(
            R2=number(obj.r_squared),
            adj_R2=number(obj.adj_r_squared),
            sigma=number(obj.sigma),
            n=obj.n,
            log_base=obj.log_base,
            con=number(obj.con),
            power=None if obj.power is None else summary(obj.power),
        )
        return out
    if kind == "BootstrappedCI":
        rows = [
            {
                "area": number(a),
                "mean": number(m),
                "median": number(md),
                "lower": number(lo),
                "upper": number(hi),
            }
            for a, m, md, lo, hi in zip(
                obj.area_grid, obj.mean, obj.median, obj.lower, obj.upper, strict=True
            )
        ]
        return {
            "type": kind,
            "conf": obj.conf,
            "n_boot": obj.n_boot,
            "n_models_attempted": obj.n_models_attempted,
            "dropped": dict(obj.dropped),
            "intervals": rows,
        }
    if kind == "GDMFit":
        return {
            "type": kind,
            "model": obj.model,
            "terms": obj.terms,
            "formula": obj.formula,
            "params": {k: number(v) for k, v in obj.params.items()},
            "std_errors": {k: number(v) for k, v in obj.std_errors.items()},
            "p_values": {k: number(v) for k, v in obj.p_values.items()},
            "R2": number(obj.r_squared),
            "RSE": number(obj.rse),
            "AIC": number(obj.aic),
            "AICc": number(obj.aicc),
            "BIC": number(obj.bic),
            "n": obj.n,
            "converged": bool(obj.converged),
            "comparison": None if obj.comparison is None else records(obj.comparison),
        }
    if kind == "GDMMultiFit":
        return {
            "type": kind,
            "best": obj.summary.iloc[0]["model"],
            "models": records(obj.summary),
            "fits": {name: summary(fit) for name, fit in obj.fits.items()},
        }
    if kind == "ColemanFit":
        return {
            "type": kind,
            "n_sites": obj.n_sites,
            "n_species": obj.n_species,
            "total_area": number(obj.total_area),
            "fraction_outside": number(obj.fraction_outside),
            "rejected": bool(obj.rejected),
            "sites": records(obj.sites.rename_axis("site").reset_index()),
        }
    if kind == "HabitatFit":
        return {
            "type": kind,
            "mod_type": obj.mod_type,
            "best": obj.summary.iloc[0]["model"] if len(obj.summary) else None,
            "log_base": obj.log_base,
            "con": number(obj.con),
            "n": len(obj.data),
            "models": records(obj.summary),
            "not_fitted": list(obj.not_fitted),
        }
    if kind == "CountrysideFit":
        return {
            "type": kind,
            "mod_type": obj.mod_type,
            "habitats": list(obj.habitats),
            "groups": list(obj.groups),
            "n": len(obj.data),
            "params": records(obj.params.reset_index()),
            "affinity": records(obj.affinity.reset_index()),
            "c": {str(k): number(v) for k, v in obj.c.items()},
            "rss": {k: number(v) for k, v in obj.rss.items()},
            "not_fitted": list(obj.not_fitted),
        }
    if kind == "LOOCV":
        return {
            "type": kind,
            "best": obj.summary.iloc[0]["model"],
            "warm_start": obj.warm_start,
            "models": records(obj.summary),
            "not_fitted": dict(obj.multi.not_fitted),
        }
    raise InputError(f"Can't summarise a {kind}.")  # pragma: no cover


def _averaged_table(obj: Any) -> Any:
    """The fitted models ranked by the average's criterion, with its
    differences and weights (``multi.summary`` is always by AICc)."""
    table = obj.multi.summary[["model", "R2", "AIC", "AICc", "BIC"]].copy()
    table[f"delta_{obj.ic}"] = table[obj.ic] - table[obj.ic].min()
    table["weight"] = table["model"].map(obj.weights).astype(float)
    return table.sort_values(obj.ic, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def envelope(
    command: str | None, call: Callable[[], dict[str, Any]], reraise_bugs: bool = False
) -> dict[str, Any]:
    """Run ``call`` (an operation, returning its JSON-ready result) and wrap
    its outcome: the result, or the error it raised, and every distinct
    warning. An error that isn't the request's fault (a bug) is re-raised with
    ``reraise_bugs``, to keep its traceback."""
    from sars._batched import get_backend

    out: dict[str, Any] = {"sars": version(), "command": command, "backend": get_backend()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
        except Exception as exc:
            if reraise_bugs and not isinstance(exc, INPUT_ERRORS):
                raise
            message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
            out.update(ok=False, error={"type": type(exc).__name__, "message": message})
        else:
            out.update(ok=True, result=value)
    seen: list[str] = []
    for warning in caught:
        text = str(warning.message)
        if text not in seen:
            seen.append(text)
    out["warnings"] = seen
    return out


def envelope_json(env: dict[str, Any]) -> str:
    """The envelope as standard JSON; a value JSON can't hold (a NaN that
    slipped through) is reported as the envelope's error instead."""
    try:
        return json.dumps(env, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:  # a bug, not the request's fault
        failed = {key: env.get(key) for key in ("sars", "command", "backend", "warnings")}
        message = f"The result can't be written as JSON: {exc}"
        failed.update(ok=False, error={"type": "SerializationError", "message": message})
        return json.dumps(failed, indent=2)
