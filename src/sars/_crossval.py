"""Leave-one-island-out cross-validation of the SAR models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sars._checks import CheckOptions
from sars._inference import MultiSARFit, _sar_multi, akaike_weights
from sars._models import _fit_nls, _validate_data
from sars._parallel import map_in_order

AVERAGE = "average"


@dataclass
class LOOCV:
    """Leave-one-island-out cross-validation of the SAR models.

    Each island in turn is left out, every model is fitted to the others, and
    the fit predicts the island left out. How far those predictions fall from
    the observed richness measures how well a model predicts an island it
    hasn't seen, which, unlike AICc, doesn't rest on the residuals being normal
    and independent, nor on a large-sample approximation.

    Attributes
    ----------
    summary : pd.DataFrame
        One row per model, best first: ``rmse`` and ``mae`` (root mean square
        and mean absolute error of the left-out predictions, in species),
        ``n_failed`` (islands whose fit failed or whose prediction wasn't
        finite; the errors are over the others), ``max_error`` and
        ``worst`` (the largest absolute error and the island it was for),
        ``rmse_common`` (the RMSE over the islands every model predicted, so
        models that failed different folds can be compared on the same
        islands), ``cv_rank``, and the model's information criterion on all
        the data (a column named after ``ic``) and its rank by it
        (``ic_rank``; missing where the criterion isn't finite). The row
        ``"average"`` is the model average weighted by ``ic``, refitted in each
        fold. Models that failed in any fold are ranked after those that never
        did.
    predictions : pd.DataFrame
        The left-out prediction for each island (rows, in the data's order)
        and model (columns); NaN where a fit failed.
    multi : MultiSARFit
        The fit to all the data.
    data : pd.DataFrame
        The data.
    warm_start : bool
        Whether each fold started from the full fit's parameters (rather than
        a full search).
    ic : str
        The criterion the models are ranked by beside the cross-validation
        (``ic_rank``) and the average weights them by.
    """

    summary: pd.DataFrame = field(repr=False)
    predictions: pd.DataFrame = field(repr=False)
    multi: MultiSARFit = field(repr=False)
    data: pd.DataFrame = field(repr=False)
    warm_start: bool = False
    ic: str = "AICc"

    def __repr__(self) -> str:
        best = self.summary.iloc[0]
        return (
            f"LOOCV({len(self.data)} islands, {len(self.summary) - 1} models, "
            f"best={best['model']!r} (RMSE {best['rmse']:.4g}))"
        )


_IC_ATTR = {"AICc": "aicc", "AIC": "aic", "BIC": "bic"}
_LABEL_GUESSES = ("island", "site", "locality", "name")


def _fold(
    i: int,
    data: pd.DataFrame,
    warm_starts: dict[str, dict[str, float]],
    warm_start: bool,
    ic: str,
) -> dict[str, float]:
    """Fit every model without island ``i`` and predict it; module level so
    it can run in a worker process."""
    keep = np.arange(len(data)) != i  # by position: index labels may repeat
    rest = data.iloc[keep].reset_index(drop=True)
    area = float(data["area"].iloc[i])
    out: dict[str, float] = {}
    fits = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for name, params in warm_starts.items():
            # A fit that fails here fails this fold only, whichever the start.
            try:
                fit = _fit_nls(name, rest, start_from=params if warm_start else None)
            except Exception:
                continue
            if fit.converged:
                fits.append(fit)
        for fit in fits:
            pred = float(fit.predict(np.array([area]))[0])
            if np.isfinite(pred):
                out[fit.model] = pred
        usable = [f for f in fits if f.model in out]
        if usable:
            values = np.array([getattr(f, _IC_ATTR[ic]) for f in usable], dtype=float)
            weights = akaike_weights(values)
            if np.all(np.isfinite(weights)):
                out[AVERAGE] = float(
                    sum(w * out[f.model] for w, f in zip(weights, usable, strict=True))
                )
    return out


def _labels(data: pd.DataFrame, label: str | None) -> list[str]:
    """Each island's name for the ``worst`` column."""
    if label == "auto":
        lower = {str(c).lower(): c for c in data.columns}
        column = next((lower[g] for g in _LABEL_GUESSES if g in lower), None)
    elif label is None:
        column = None
    else:
        if label not in data.columns:
            raise ValueError(f"No column {label!r} to label the islands by.")
        column = label
    if column is None:
        return [str(i) for i in range(len(data))]
    return [str(v) for v in data[column]]


def loo_cv(
    data: pd.DataFrame,
    models: str | list[str] = "all",
    warm_start: bool = False,
    ic: str = "AICc",
    label: str | None = "auto",
    n_jobs: int = 1,
) -> LOOCV:
    """Leave-one-island-out cross-validation of the SAR models.

    Fits the models to all the data, then, for each island, to every island
    but that one, and predicts the island left out. The models are ranked by
    the root mean square error of those predictions. For a small archipelago
    this is a check on AICc that doesn't lean on its assumptions: AICc's
    small-sample correction is itself an approximation, and with few islands
    the models it ranks closely can predict quite differently.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    models : str or list[str]
        Model names to compare, or "all" (the default). Models that don't
        converge on all the data are left out (see ``multi.not_fitted``).
    warm_start : bool
        Start each fold's fit from the parameters of the fit to all the data,
        as ``bootstrap_ci``'s fast method does, instead of running the full
        multi-start search in every fold (False, the default). It is 15 to 30
        times faster, but those starting values have seen the island left
        out, and a flexible model can stay near them rather than find the
        best fit to the other islands. Its error then comes out too small: on
        Johnson and Raven's Galápagos data, warm starts move weibull4 from
        15th to 4th. Use ``n_jobs`` to speed up the full search instead.
    ic : str
        The criterion the models are compared with (``ic_rank``) and the
        average weights them by: "AICc" (the default), "AIC" or "BIC". AICc
        is undefined for a model with as many parameters as the islands
        allow, so with very few islands use "AIC".
    label : str or None
        The column that names the islands, for the ``worst`` column. "auto"
        (the default) takes the first of island, site, locality and name
        found (in any case); None, or no such column, uses row numbers.
    n_jobs : int
        Fit the folds in this many processes; -1 for one per core. The result
        doesn't depend on it.

    Returns
    -------
    LOOCV

    Raises
    ------
    ValueError
        With fewer than four islands, an unknown ``ic`` or ``label``, or if no
        model converges on all the data.

    Warns
    -----
    UserWarning
        If the model average couldn't be computed in any fold (the criterion
        undefined for every model on n - 1 islands).

    Examples
    --------
    >>> cv = sars.loo_cv(sars.load_galap(), models=["power", "loga", "negexpo"])
    >>> cv.summary[["model", "rmse", "cv_rank", "ic_rank"]]  # doctest: +SKIP
    """
    _validate_data(data)
    if ic not in _IC_ATTR:
        raise ValueError(f"Unsupported IC: {ic!r}. Choose from {sorted(_IC_ATTR)}.")
    n = len(data)
    if n < 4:
        raise ValueError(f"Cross-validation needs at least 4 islands, not {n}.")
    labels = _labels(data, label)
    multi = _sar_multi(data, models, CheckOptions(), stacklevel=2)
    if not multi.fits:
        raise ValueError("No model converged on the data, so there is nothing to cross-validate.")
    warm_starts = {fit.model: dict(fit.params) for fit in multi.fits}
    shared = {"data": data, "warm_starts": warm_starts, "warm_start": warm_start, "ic": ic}
    folds = map_in_order(_fold, list(range(n)), n_jobs, shared)

    names = [*warm_starts, AVERAGE]
    predictions = pd.DataFrame(
        {name: [fold.get(name, np.nan) for fold in folds] for name in names},
        index=data.index,
    )
    if predictions[AVERAGE].isna().all():
        warnings.warn(
            f"The {ic} model average couldn't be computed in any fold: {ic} is undefined "
            f"for every model on {n - 1} islands. Use ic='AIC'.",
            stacklevel=2,
        )
    observed = data["species"].to_numpy(dtype=float)
    common = predictions[list(warm_starts)].notna().all(axis=1).to_numpy()
    criterion = {f.model: float(getattr(f, _IC_ATTR[ic])) for f in multi.fits}
    finite = sorted((v, m) for m, v in criterion.items() if np.isfinite(v))
    ic_rank = {m: r for r, (_, m) in enumerate(finite, start=1)}
    rows = []
    for name in names:
        pred = predictions[name].to_numpy(dtype=float)
        ok = np.isfinite(pred)
        err = pred[ok] - observed[ok]
        worst = int(np.argmax(np.abs(err))) if ok.any() else None
        shared_ok = ok & common
        common_err = pred[shared_ok] - observed[shared_ok]
        rows.append(
            {
                "model": name,
                "rmse": float(np.sqrt(np.mean(err**2))) if ok.any() else np.nan,
                "mae": float(np.mean(np.abs(err))) if ok.any() else np.nan,
                "max_error": float(np.abs(err[worst])) if worst is not None else np.nan,
                "worst": labels[np.flatnonzero(ok)[worst]] if worst is not None else None,
                "n_failed": int((~ok).sum()),
                "rmse_common": float(np.sqrt(np.mean(common_err**2)))
                if shared_ok.any()
                else np.nan,
                ic: criterion.get(name, np.nan),
                "ic_rank": ic_rank.get(name, pd.NA),
            }
        )
    summary = pd.DataFrame(rows)
    summary["_failed"] = summary["n_failed"] > 0
    summary = summary.sort_values(["_failed", "rmse"], kind="stable").drop(columns="_failed")
    summary = summary.reset_index(drop=True)
    summary.insert(7, "cv_rank", np.arange(1, len(summary) + 1))
    summary["ic_rank"] = summary["ic_rank"].astype("Int64")
    return LOOCV(
        summary=summary,
        predictions=predictions,
        multi=multi,
        data=data,
        warm_start=warm_start,
        ic=ic,
    )
