"""Predicted richness at given areas, from any kind of fit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sars._inference import AveragedSAR, MultiSARFit
from sars._linpow import LinPowFit
from sars._models import SARFit
from sars._threshold import ThresholdFit


def _label(fit: object) -> str:
    """The ``model`` name for a fit that gives one prediction per area, after
    checking that it has a prediction to give."""
    if isinstance(fit, SARFit):
        if not fit.converged:
            raise ValueError(f"The '{fit.model}' fit did not converge, so it has no predictions")
        return fit.model
    if isinstance(fit, AveragedSAR):
        if not fit.multi.fits:
            raise ValueError("The averaged fit holds no models, so there is nothing to average")
        return "average"
    if isinstance(fit, ThresholdFit):
        return fit.best_model
    if isinstance(fit, LinPowFit):
        return "lin_pow"
    raise TypeError(
        "fit must be a SARFit, MultiSARFit, AveragedSAR, ThresholdFit or LinPowFit, "
        f"got {type(fit).__name__}"
    )


def _by_aicc(fits: list[SARFit]) -> list[SARFit]:
    """Fits with the lowest AICc first. Those without a finite AICc come last,
    and ties keep the order the fits are held in."""
    return sorted(
        fits, key=lambda f: (not np.isfinite(f.aicc), f.aicc if np.isfinite(f.aicc) else 0.0)
    )


def _check_areas(area: object, allow_non_positive: bool) -> np.ndarray:
    raw = np.asarray(area)
    # Strings and booleans would convert to numbers without complaint.
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"'area' must be numeric, got {raw.dtype} values")
    areas = np.atleast_1d(raw.astype(float))
    if areas.ndim != 1:
        raise ValueError(f"'area' must be a number or a 1-D array, got {areas.ndim} dimensions")
    if areas.size == 0:
        raise ValueError("'area' is empty")
    if not np.isfinite(areas).all():
        raise ValueError("'area' contains NaN or infinite values")
    if not allow_non_positive and (areas <= 0).any():
        raise ValueError("'area' must contain only positive values")
    return areas


def sar_pred(
    fit: SARFit | MultiSARFit | AveragedSAR | ThresholdFit | LinPowFit,
    area: float | np.ndarray,
) -> pd.DataFrame:
    """Predict species richness at given areas from a fitted model.

    Extrapolating to areas larger than any sampled is one of the main uses of
    a species-area relationship. This puts the predictions of any fit in one
    table, in place of calling each fit's ``predict``.

    Parameters
    ----------
    fit : SARFit, MultiSARFit, AveragedSAR, ThresholdFit or LinPowFit
        A fitted model. A ``MultiSARFit`` gives each of its models' predictions
        separately; an ``AveragedSAR`` gives the weighted average of its
        models' predictions, one row per area.
    area : float or array-like
        Area value(s) to predict for, in the units the model was fitted in.
        They may lie outside the range of the data, and must be positive. The
        exception is a ``ThresholdFit`` fitted with ``log_area=False``, whose
        lines are defined for any area: it takes zero and negative areas too.

    Returns
    -------
    pd.DataFrame
        One row per model and area, with columns:

        - ``model``: the model's name ("power", "loga", ...). "average" for an
          ``AveragedSAR``, the best model's name ("ContOne", ...) for a
          ``ThresholdFit``, and "lin_pow" for a ``LinPowFit``.
        - ``area``: the area predicted for.
        - ``prediction``: predicted species richness.

        Rows are grouped by model, a ``MultiSARFit``'s with the lowest AICc
        first (the order of its ``summary``), with areas in the order given.
        A ``MultiSARFit`` that holds no fits gives a table with no rows.

        The predictions are the models' own, unchecked: far outside the
        fitted range some models give negative or infinite richness.

    Raises
    ------
    TypeError
        If ``fit`` isn't one of the fit objects above, or ``area`` isn't
        numeric (strings and booleans aren't converted).
    ValueError
        If ``area`` is empty, has more than one dimension, or holds a value
        that is NaN, infinite or (except as above) zero or negative; if a
        ``SARFit`` didn't converge; or if an ``AveragedSAR`` holds no models.

    Notes
    -----
    An averaged prediction depends on which models were averaged, so it only
    matches R ``sars``'s ``sar_pred`` for a ``sar_average`` over the same
    models.

    References
    ----------
    Matthews TJ, Aspin TWH (2019) Model averaging fails to improve the
    extrapolation capability of the island species-area relationship. Journal
    of Biogeography 46:1558-1568.
    """
    # The fit is checked first, so that a wrong object is reported as that
    # whatever the areas are.
    sources: list[tuple[str, SARFit | AveragedSAR | ThresholdFit | LinPowFit]]
    if isinstance(fit, MultiSARFit):
        sources = [(f.model, f) for f in _by_aicc(fit.fits)]
    else:
        sources = [(_label(fit), fit)]

    on_raw_area = isinstance(fit, ThresholdFit) and not fit.log_area
    areas = _check_areas(area, allow_non_positive=on_raw_area)

    predictions = [np.asarray(source.predict(areas), dtype=float) for _, source in sources]
    # Built the same way however many models there are, none included, so the
    # columns have the same types.
    return pd.DataFrame(
        {
            "model": pd.Series([name for name, _ in sources for _ in areas], dtype=str),
            "area": np.tile(areas, len(sources)),
            "prediction": np.concatenate([np.empty(0), *predictions]),
        }
    )
