"""The general dynamic model of oceanic island biogeography (GDM).

Whittaker, Triantis & Ladle (2008), Journal of Biogeography 35, 977-994,
predict that an oceanic island's richness rises while the island is being
built and falls as it erodes and subsides, so richness is a function of area
plus a hump-shaped function of island age: area, time and time squared, their
"ATT²" model.

Four forms are fitted, the ones R sars's ``gdm`` offers, written with its
parameter names (Int, A, Ti, Ti2):

- loga: ``S = Int + A*log(area) + Ti*T + Ti2*T^2`` (Whittaker et al.'s ATT²)
- linear: ``S = Int + A*area + Ti*T + Ti2*T^2``
- power_area: ``S = exp(Int + A*log(area) + Ti*T + Ti2*T^2)``
- power_area_time: ``S = exp(Int + A*log(area) + Ti*log(T) + Ti2*log(T)^2)``

The first two are linear in their parameters and are fitted by ordinary least
squares. The power forms are fitted by nonlinear least squares on richness
itself, not on its log, so all four have the same response and their AIC
values can be compared.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import stdtr

from sars._inference import akaike_weights
from sars._models import _compute_ic

GDM_MODELS = ["loga", "linear", "power_area", "power_area_time"]

_PARAM_NAMES = ["Int", "A", "Ti", "Ti2"]

# The GDM and the three simpler models it is compared with, as the number of
# leading parameters each keeps: every candidate is the GDM with its last
# terms dropped.
_CANDIDATES = {"GDM": 4, "A + T": 3, "A": 2, "Intercept": 1}

# Models whose richness is exp(linear predictor), and those that use log(time).
_EXP_MODELS = {"power_area", "power_area_time"}
_LOG_TIME_MODELS = {"power_area_time"}
_LOG_AREA_MODELS = {"loga", "power_area", "power_area_time"}

# exp() overflows a float just above this; a trial step that goes past it is
# held here, where its huge residual already rejects it.
_MAX_EXPONENT = 700.0


def _formula(model: str, n_terms: int) -> str:
    """The fitted equation in words, for the model's first ``n_terms`` terms."""
    area = "log(area)" if model in _LOG_AREA_MODELS else "area"
    time = "log(time)" if model in _LOG_TIME_MODELS else "time"
    terms = ["Int", f"A*{area}", f"Ti*{time}", f"Ti2*{time}^2"][:n_terms]
    rhs = " + ".join(terms)
    return f"S = exp({rhs})" if model in _EXP_MODELS else f"S = {rhs}"


def _design(model: str, area: np.ndarray, time: np.ndarray, n_terms: int) -> np.ndarray:
    """Design matrix of the linear predictor: one column per kept term.

    ``area`` and ``time`` are 1-D and the same length. Each is only read (and
    its log only taken) when a term that uses it is kept.
    """
    cols = [np.ones_like(area)]
    if n_terms >= 2:
        cols.append(np.log(area) if model in _LOG_AREA_MODELS else area)
    if n_terms >= 3:
        t = np.log(time) if model in _LOG_TIME_MODELS else time
        cols.append(t)
        if n_terms >= 4:
            cols.append(t**2)
    return np.column_stack(cols)


def _exp(eta: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(eta, _MAX_EXPONENT))


def _unit_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The matrix with each column divided by its length, and those lengths.

    The columns of a design differ by many orders of magnitude (1 beside a
    squared age in years), which is a matter of units, not of the data. With
    the columns at unit length, rank tests and least-squares solutions don't
    depend on the units of area or time.
    """
    norms = np.linalg.norm(matrix, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return matrix / norms, norms


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class GDMFit:
    """Result of fitting the general dynamic model with one area form.

    Attributes
    ----------
    model : str
        The form fitted: "loga", "linear", "power_area" or "power_area_time".
    terms : str
        Which terms the fit has: "GDM" (area, time and time squared), or one
        of the simpler candidates "A + T", "A" and "Intercept" (the fits in
        ``candidates``).
    params : dict[str, float]
        Coefficients, named as in R sars: ``Int`` (intercept), ``A`` (area),
        ``Ti`` (time) and ``Ti2`` (time squared). A simpler candidate has only
        its own. In the power forms every coefficient, ``Int`` included, is on
        the log scale: richness is the exponential of the linear predictor.
    std_errors : dict[str, float]
        Standard error of each coefficient, from the residual variance and the
        model's gradient at the fit (exact for the forms that are linear in
        their parameters, the usual large-sample approximation for the power
        forms).
    p_values : dict[str, float]
        Two-sided p value of each coefficient's t statistic (coefficient over
        standard error, on ``n`` minus the number of coefficients degrees of
        freedom).
    fitted : np.ndarray
        Fitted richness for each row of ``data``.
    r_squared : float
        1 - RSS / total sum of squares, on richness itself for every form.
    rse : float
        Residual standard error: sqrt(RSS / (n - number of coefficients)).
    aic, aicc, bic : float
        Information criteria from the normal likelihood, counting each
        coefficient and the residual variance as parameters (5 for the GDM).
        AICc is infinite when ``n`` is not above that count plus one. A fit
        that passes through every island has no residual variance to
        estimate; its criteria are computed as if its residuals were at the
        rounding error of the data (see the notes of ``sar_gdm``).
    n : int
        Number of islands.
    converged : bool
        Whether the solver converged. Always True for "loga" and "linear",
        which have a closed-form solution.
    data : pd.DataFrame
        The data fitted (columns: area, species, time).
    comparison : pd.DataFrame | None
        With ``mod_sel=True``, the GDM and its three simpler candidates ranked
        by AICc: columns model, k (number of parameters), RSE, R2, AIC, AICc,
        BIC, delta_AIC, delta_AICc, weight_AIC, weight (the AICc weight) and
        converged. A candidate that did not converge is listed last, with no
        deltas and no weight. None otherwise.
    candidates : dict[str, GDMFit] | None
        With ``mod_sel=True``, the fits behind ``comparison``, keyed "GDM",
        "A + T", "A" and "Intercept". None otherwise.
    """

    model: str
    terms: str
    params: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]
    fitted: np.ndarray
    r_squared: float
    rse: float
    aic: float
    aicc: float
    bic: float
    n: int
    converged: bool
    data: pd.DataFrame
    comparison: pd.DataFrame | None = None
    candidates: dict[str, GDMFit] | None = None

    @property
    def formula(self) -> str:
        """The fitted equation, e.g. ``S = Int + A*log(area) + Ti*time + Ti2*time^2``."""
        return _formula(self.model, len(self.params))

    @property
    def residuals(self) -> np.ndarray:
        """Observed minus fitted richness for each row of ``data``."""
        return self.data["species"].to_numpy(dtype=float) - self.fitted

    def predict(
        self, area: float | np.ndarray, time: float | np.ndarray | None = None
    ) -> np.ndarray:
        """Predict species richness for given area and island age value(s).

        Parameters
        ----------
        area : float or array-like
            Island area(s), in the units of the fitted data.
        time : float or array-like, optional
            Island age(s), in the units of the fitted data. ``area`` and
            ``time`` are broadcast against each other, so one may be a single
            value. Only a candidate without time terms ("A", "Intercept") can
            leave it out; given one anyway, it shapes the result but doesn't
            change the predictions.

        Returns
        -------
        np.ndarray
            Predicted richness, in the broadcast shape of ``area`` and
            ``time``. The forms without an exponential can predict fewer than
            zero species away from the data.

        Raises
        ------
        ValueError
            If ``time`` is missing from a fit that has time terms, a value is
            NaN or infinite, or a value is outside the domain of the fitted
            form (area or time not positive where the fit uses its log).
        """
        n_terms = len(self.params)
        area_arr = np.asarray(area, dtype=float)
        if not np.isfinite(area_arr).all():
            raise ValueError("area contains NaN or infinite values")
        if time is None:
            if n_terms >= 3:
                raise ValueError("time is required: this fit has a time term.")
            # Never read: this fit has no time term.
            time_arr = np.zeros_like(area_arr)
        else:
            time_arr = np.asarray(time, dtype=float)
            if not np.isfinite(time_arr).all():
                raise ValueError("time contains NaN or infinite values")
            area_arr, time_arr = np.broadcast_arrays(area_arr, time_arr)
        if self.model in _LOG_AREA_MODELS and n_terms >= 2 and np.any(area_arr <= 0):
            raise ValueError(f"Areas must be positive: '{self.model}' uses log(area).")
        if self.model in _LOG_TIME_MODELS and n_terms >= 3 and np.any(time_arr <= 0):
            raise ValueError(f"Times must be positive: '{self.model}' uses log(time).")
        design = _design(self.model, area_arr.ravel(), time_arr.ravel(), n_terms)
        eta = design @ np.array(list(self.params.values()))
        out = _exp(eta) if self.model in _EXP_MODELS else eta
        return out.reshape(area_arr.shape)

    def __repr__(self) -> str:
        p = "  ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        terms = "" if self.terms == "GDM" else f", terms='{self.terms}'"
        return (
            f"GDMFit(model='{self.model}'{terms}, {p}, "
            f"R²={self.r_squared:.4f}, AICc={self.aicc:.2f})"
        )


@dataclass(repr=False)
class GDMMultiFit:
    """The general dynamic model fitted with every area form (``model="all"``).

    Attributes
    ----------
    fits : dict[str, GDMFit]
        The fit for each form, keyed "loga", "linear", "power_area" and
        "power_area_time".
    summary : pd.DataFrame
        The four fits ranked by AICc: columns model, k, RSE, R2, AIC, AICc,
        BIC, delta_AIC, delta_AICc, weight_AIC, weight (the AICc weight) and
        converged. The forms share a response (richness) and a parameter
        count, so the ranking is the ranking by residual sum of squares. A
        form whose fit did not converge is listed last, with no deltas and no
        weight.
    data : pd.DataFrame
        The data fitted (columns: area, species, time).
    """

    fits: dict[str, GDMFit]
    summary: pd.DataFrame
    data: pd.DataFrame

    @property
    def best(self) -> GDMFit:
        """The fit ranked first in ``summary``."""
        return self.fits[str(self.summary.iloc[0]["model"])]

    def __repr__(self) -> str:
        return f"GDMMultiFit({len(self.fits)} models, best='{self.summary.iloc[0]['model']}')"


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _fit_ols(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of ``y = design @ beta``, by QR on
    unit-length columns."""
    scaled, norms = _unit_columns(design)
    q, r = np.linalg.qr(scaled)
    return np.linalg.solve(r, q.T @ y) / norms


def _fit_exp(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Least-squares coefficients of ``y = exp(design @ beta)``.

    The columns of the design (time and time squared above all) are nearly
    collinear, which makes the problem badly conditioned in ``beta``. So the
    search is over the coefficients of an orthonormal basis of the same column
    space (the Q of a QR decomposition), and the answer is mapped back.

    The sum of squares can have more than one minimum. Two starts, both taken
    from the data, are tried and the better fit kept: the regression of
    log(richness) on the design (the fit that would be right under
    multiplicative error), and a flat curve at the mean richness.
    """
    scaled, norms = _unit_columns(design)
    q, r = np.linalg.qr(scaled)

    def residuals(gamma: np.ndarray) -> np.ndarray:
        return _exp(q @ gamma) - y

    def jacobian(gamma: np.ndarray) -> np.ndarray:
        return _exp(q @ gamma)[:, None] * q

    # An island with no species has no log; half a species stands in for it,
    # only to place the start.
    log_y = np.log(np.maximum(y, 0.5))
    flat = np.full_like(y, np.log(max(float(np.mean(y)), 0.5)))
    best = None
    for start in (q.T @ log_y, q.T @ flat):
        res = least_squares(
            residuals,
            start,
            jac=jacobian,
            method="trf",
            x_scale="jac",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-12,
            max_nfev=5000,
        )
        if best is None or res.cost < best.cost:
            best = res
    if best is None:  # the loop always runs; this is for the type checker
        raise RuntimeError("no start was tried")
    return np.linalg.solve(r, best.x) / norms, bool(best.success)


def _std_errors(gradient: np.ndarray, sigma2: float) -> np.ndarray:
    """Standard errors from the coefficient covariance ``sigma2 * (G'G)^-1``.

    Taken from the R factor of the gradient with unit-length columns, so the
    nearly collinear time columns aren't squared and the units don't matter.
    """
    scaled, norms = _unit_columns(gradient)
    r_inv = np.linalg.pinv(np.linalg.qr(scaled, mode="r"))
    return np.sqrt(sigma2 * np.sum(r_inv**2, axis=1)) / norms


def _fit_candidate(model: str, terms: str, design: np.ndarray, data: pd.DataFrame) -> GDMFit:
    """Fit one candidate (the GDM or a simpler model) of one form.

    ``design`` is the candidate's own: the leading columns of the GDM's.
    """
    n_terms = design.shape[1]
    y = data["species"].to_numpy(dtype=float)
    n = len(y)

    if model in _EXP_MODELS:
        beta, converged = _fit_exp(design, y)
        fitted = _exp(design @ beta)
        # Gradient of the fitted values with respect to the coefficients.
        gradient = fitted[:, None] * design
    else:
        beta = _fit_ols(design, y)
        converged = True
        fitted = design @ beta
        gradient = design

    rss = float(np.sum((y - fitted) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    dof = n - n_terms
    sigma2 = rss / dof
    se = _std_errors(gradient, sigma2)
    # An exact fit has zero standard errors, and so no t statistics.
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2.0 * stdtr(dof, -np.abs(beta / se))

    # Each coefficient and the residual variance count as a parameter; an
    # exact fit is scored at the data's rounding error (see rss_floor).
    aic, aicc, bic = _compute_ic(n_terms + 1, n, rss, y)
    names = _PARAM_NAMES[:n_terms]
    return GDMFit(
        model=model,
        terms=terms,
        params=dict(zip(names, beta.tolist(), strict=True)),
        std_errors=dict(zip(names, se.tolist(), strict=True)),
        p_values=dict(zip(names, np.asarray(p).tolist(), strict=True)),
        fitted=fitted,
        r_squared=1.0 - rss / tss,
        rse=float(np.sqrt(sigma2)),
        aic=float(aic),
        aicc=float(aicc),
        bic=float(bic),
        n=n,
        converged=converged,
        data=data,
    )


def _comparison_table(fits: dict[str, GDMFit]) -> pd.DataFrame:
    """Rank fits by AICc (AIC breaking ties, and deciding when AICc is undefined).

    The columns shared with ``sar_multi``'s summary (model, R2, AIC, AICc,
    BIC, delta_AICc, weight) have the same names, meaning and order, and the
    table is sorted by AICc as that one is. This table also has AIC's delta
    and weight, which R's ``gdm`` ranks by.

    ``sar_multi`` leaves out a fit that didn't converge. Here every candidate
    is a row, since the table is a fixed set of nested models: one that didn't
    converge is listed last, and has no deltas or weight because its criteria
    are not those of a least-squares fit.
    """
    converged = np.array([f.converged for f in fits.values()])
    df = pd.DataFrame(
        {
            "model": list(fits),
            "k": [len(f.params) + 1 for f in fits.values()],
            "RSE": [f.rse for f in fits.values()],
            "R2": [f.r_squared for f in fits.values()],
            "AIC": [f.aic for f in fits.values()],
            "AICc": [f.aicc for f in fits.values()],
            "BIC": [f.bic for f in fits.values()],
        }
    )
    for ic in ("AIC", "AICc"):
        values = np.where(converged, df[ic].to_numpy(dtype=float), np.inf)
        # inf - inf is NaN: where a criterion is undefined for every fit, so
        # is its delta.
        with np.errstate(invalid="ignore"):
            delta = values - values.min()
        df[f"delta_{ic}"] = np.where(converged, delta, np.nan)
        df["weight_AIC" if ic == "AIC" else "weight"] = akaike_weights(values)
    df["converged"] = converged
    order = ["model", "k", "RSE", "R2", "AIC", "AICc", "BIC"]
    order += ["delta_AIC", "delta_AICc", "weight_AIC", "weight", "converged"]
    return (
        df[order]
        .sort_values(["converged", "AICc", "AIC"], ascending=[False, True, True], kind="stable")
        .reset_index(drop=True)
    )


def _prepare_data(
    data: pd.DataFrame, model: str, area_col: str, species_col: str, time_col: str
) -> pd.DataFrame:
    """Validate the input and return it with columns area, species, time.

    The SAR models' validation isn't reused because the GDM's input differs:
    it has a third column (island age) under caller-chosen names, and an
    island with no species is valid data here, since no form takes the log of
    richness.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    cols = {"area": area_col, "species": species_col, "time": time_col}
    missing = [c for c in cols.values() if c not in data.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. DataFrame has columns: {list(data.columns)}"
        )
    if len(set(cols.values())) < 3:
        raise ValueError(
            f"area_col, species_col and time_col must name three different columns, got {cols}"
        )

    out = pd.DataFrame(index=data.index)
    for name, col in cols.items():
        series = data[col]
        if isinstance(series, pd.DataFrame):
            raise ValueError(f"'{col}' names more than one column")
        # Strings that look like numbers and booleans would convert to float
        # without complaint; they are more likely the wrong column.
        dtype = series.dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise ValueError(
                f"'{col}' must be a numeric column, got dtype {dtype}"
                " (convert it with pandas.to_numeric if it holds numbers)"
            )
        if series.isna().any():
            raise ValueError(f"'{col}' contains missing values")
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"'{col}' contains infinite values")
        out[name] = values

    # The GDM has four coefficients; a fifth island gives the residual
    # variance one degree of freedom.
    if len(out) < 5:
        raise ValueError(f"Need at least 5 islands to fit the GDM, got {len(out)}")
    if (out["area"] <= 0).any():
        raise ValueError(f"'{area_col}' must contain only positive values")
    if (out["species"] < 0).any():
        raise ValueError(f"'{species_col}' must not contain negative values")
    if (out["time"] < 0).any():
        raise ValueError(f"'{time_col}' must not contain negative values")
    if model in ("power_area_time", "all") and (out["time"] <= 0).any():
        raise ValueError(
            f"'{time_col}' must contain only positive values for model "
            "'power_area_time', which uses log(time)"
        )
    # Time and time squared can't be told apart with fewer than three ages,
    # nor area from the intercept with a single area.
    if out["time"].nunique() < 3:
        raise ValueError(
            f"'{time_col}' needs at least 3 different values to fit a hump in island age, "
            f"got {out['time'].nunique()}"
        )
    if out["area"].nunique() < 2:
        raise ValueError(f"'{area_col}' needs at least 2 different values")
    if out["species"].nunique() < 2:
        raise ValueError(f"'{species_col}' is the same on every island: there is nothing to fit")
    return out


def _fit_model(model: str, data: pd.DataFrame, mod_sel: bool) -> GDMFit:
    """Fit the GDM of one form and, with ``mod_sel``, its simpler candidates."""
    design = _design(
        model, data["area"].to_numpy(dtype=float), data["time"].to_numpy(dtype=float), 4
    )
    if np.linalg.matrix_rank(_unit_columns(design)[0]) < 4:
        raise ValueError(
            f"The area and time terms of model '{model}' are collinear in this data, "
            "so their coefficients can't be told apart"
        )
    fit = _fit_candidate(model, "GDM", design, data)
    if mod_sel:
        candidates = {"GDM": fit}
        for terms, n_terms in list(_CANDIDATES.items())[1:]:
            candidates[terms] = _fit_candidate(model, terms, design[:, :n_terms], data)
        fit.candidates = candidates
        fit.comparison = _comparison_table(candidates)
    return fit


def _unconverged(fit: GDMFit) -> list[str]:
    """Names of the fits (the GDM and any candidates) that did not converge."""
    fitted = fit.candidates if fit.candidates is not None else {"GDM": fit}
    return [terms for terms, cand in fitted.items() if not cand.converged]


def sar_gdm(
    data: pd.DataFrame,
    model: str = "loga",
    mod_sel: bool = False,
    area_col: str = "area",
    species_col: str = "species",
    time_col: str = "time",
) -> GDMFit | GDMMultiFit:
    """Fit the general dynamic model of oceanic island biogeography (GDM).

    The GDM (Whittaker, Triantis & Ladle 2008) describes island richness as a
    function of island area plus a hump-shaped function of island age: time
    and time squared, with a negative coefficient on time squared when
    richness rises and then falls over an island's life. ``model`` chooses how
    area (and time) enter, with ``S`` richness and ``T`` island age:

    - ``"loga"``: ``S = Int + A*log(area) + Ti*T + Ti2*T^2``, the "ATT²"
      model of Whittaker et al. (R sars's ``"loga"`` and ``"ATT2"``)
    - ``"linear"``: ``S = Int + A*area + Ti*T + Ti2*T^2``
    - ``"power_area"``: ``S = exp(Int + A*log(area) + Ti*T + Ti2*T^2)``, a
      power law of area (``area^A``) times a hump in time
    - ``"power_area_time"``:
      ``S = exp(Int + A*log(area) + Ti*log(T) + Ti2*log(T)^2)``, with time
      logged as well
    - ``"all"``: all four, with a table comparing them

    These are the forms R sars's ``gdm`` fits. The first two are fitted by
    ordinary least squares. The power forms are fitted by nonlinear least
    squares on richness itself (not by regressing log richness), so every form
    has the same response and their AIC values are comparable.

    With ``mod_sel=True`` the GDM is also compared with three simpler models
    of the same form: area and time (no time squared), area only, and an
    intercept only. A GDM that doesn't beat "A + T" shows no evidence of a
    hump in island age.

    Parameters
    ----------
    data : pd.DataFrame
        One row per island, with numeric columns for area, species richness
        and island age.
    model : str
        "loga" (the default), "linear", "power_area", "power_area_time" or
        "all".
    mod_sel : bool
        Also fit the three simpler candidates and rank all four by AICc, in
        the fit's ``comparison`` table (the fits themselves are in
        ``candidates``). With ``model="all"`` each form gets its own table.
    area_col, species_col, time_col : str
        Names of the columns holding area, richness and island age.

    Returns
    -------
    GDMFit or GDMMultiFit
        A ``GDMFit`` for a single form; for ``model="all"``, a ``GDMMultiFit``
        holding the ``GDMFit`` of every form and a table ranking them.

    Raises
    ------
    TypeError
        If *data* is not a DataFrame.
    KeyError
        If a named column is missing.
    ValueError
        If ``model`` is unknown; a column is not numeric (strings and
        booleans are not converted); there are fewer than 5 islands; a value
        is missing, infinite or out of range (area must be positive, richness
        and time non-negative, and time positive for "power_area_time"); or
        the area and time terms are collinear (fewer than 3 different ages,
        for one).

    Warns
    -----
    UserWarning
        If the solver did not converge for a power form's GDM or, with
        ``mod_sel=True``, for one of its simpler candidates. The warning
        names them. Such a fit has ``converged=False``, and is listed last in
        the comparison table with no deltas and no weight, since its criteria
        are not those of a least-squares fit.

    Notes
    -----
    Every coefficient and the residual variance count as parameters, so the
    GDM has k = 5 and AICc is finite only for 7 or more islands. In the power
    forms the simpler candidates are power forms too: "A" is the power law
    ``exp(Int + A*log(area))`` and "Intercept" is ``exp(Int)``, the mean
    richness, with ``Int`` on the log scale like the others.

    The fit doesn't depend on the units of area or time, except through the
    coefficients that carry those units.

    The normal likelihood grows without bound as the residuals go to zero.
    So that a fit passing through every island still has finite criteria, its
    residual sum of squares is not taken below the rounding error of the data
    (machine epsilon times the sum of squared richness) when AIC, AICc and BIC
    are computed. Exact fits then tie on the likelihood and are ranked by
    their number of parameters. ``r_squared``, ``rse`` and the standard errors
    use the residuals as they are.

    References
    ----------
    Whittaker RJ, Triantis KA, Ladle RJ (2008). A general dynamic theory of
    oceanic island biogeography. Journal of Biogeography 35, 977-994.

    Examples
    --------
    >>> import sars
    >>> islands = sars.load_galap()
    >>> islands["time"] = [4, 1, 13, 16, 15, 2, 6, 4, 5, 11, 3, 9, 8, 10, 12, 7]
    >>> fit = sars.sar_gdm(islands, model="loga", mod_sel=True)
    >>> round(fit.params["Ti2"], 3)
    -2.837
    >>> fit.comparison["model"].tolist()
    ['A', 'GDM', 'A + T', 'Intercept']
    """
    if model != "all" and model not in GDM_MODELS:
        raise ValueError(f"Unknown model: {model!r}. Choose from {[*GDM_MODELS, 'all']}")
    prepared = _prepare_data(data, model, area_col, species_col, time_col)

    # The warnings are raised here, in a plain loop, so that they point at the
    # caller's line whichever model was asked for.
    fits: dict[str, GDMFit] = {}
    for name in GDM_MODELS if model == "all" else [model]:
        fits[name] = _fit_model(name, prepared, mod_sel)
        failed = _unconverged(fits[name])
        if failed:
            warnings.warn(
                f"Model '{name}': the solver did not converge for {', '.join(failed)}. "
                "Coefficients and information criteria of a fit that did not converge "
                "are not reliable; it is ranked last, with no weight.",
                UserWarning,
                stacklevel=2,
            )
    if model != "all":
        return fits[model]
    return GDMMultiFit(fits=fits, summary=_comparison_table(fits), data=prepared)
