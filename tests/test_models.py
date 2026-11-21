import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sars  # noqa: I001
from sars._inference import ALL_MODEL_NAMES
from sars._models import _param_std_errors

# ---------------------------------------------------------------------------
# Input validation (applies to all sar_* functions via _fit_nls)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Validation errors should be clear, not buried inside scipy."""

    def test_not_a_dataframe(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            sars.sar_power({"area": [1, 2], "species": [3, 4]})

    def test_missing_area_column(self):
        df = pd.DataFrame({"A": [1, 2], "species": [3, 4]})
        with pytest.raises(KeyError, match="area"):
            sars.sar_power(df)

    def test_missing_species_column(self):
        df = pd.DataFrame({"area": [1, 2], "S": [3, 4]})
        with pytest.raises(KeyError, match="species"):
            sars.sar_power(df)

    def test_missing_both_columns(self):
        df = pd.DataFrame({"A": [1, 2], "S": [3, 4]})
        with pytest.raises(KeyError, match="area.*species"):
            sars.sar_power(df)

    def test_too_few_observations(self):
        df = pd.DataFrame({"area": [1.0], "species": [5]})
        with pytest.raises(ValueError, match="at least 2"):
            sars.sar_power(df)

    def test_nan_in_area(self):
        df = pd.DataFrame({"area": [1.0, np.nan, 3.0], "species": [5, 10, 15]})
        with pytest.raises(ValueError, match="area.*NaN"):
            sars.sar_power(df)

    def test_nan_in_species(self):
        df = pd.DataFrame({"area": [1.0, 2.0, 3.0], "species": [5, np.nan, 15]})
        with pytest.raises(ValueError, match="species.*NaN"):
            sars.sar_power(df)

    def test_inf_in_area(self):
        df = pd.DataFrame({"area": [1.0, np.inf], "species": [5, 10]})
        with pytest.raises(ValueError, match="area.*infinite"):
            sars.sar_power(df)

    def test_zero_area(self):
        df = pd.DataFrame({"area": [0.0, 1.0, 2.0], "species": [5, 10, 15]})
        with pytest.raises(ValueError, match="area.*positive"):
            sars.sar_power(df)

    def test_negative_species(self):
        df = pd.DataFrame({"area": [1.0, 2.0, 3.0], "species": [-1, 10, 15]})
        with pytest.raises(ValueError, match="species.*positive"):
            sars.sar_power(df)

    def test_validation_applies_to_all_models(self):
        """Every function that fits the models to data should reject bad input identically."""
        bad = pd.DataFrame({"wrong": [1, 2], "columns": [3, 4]})
        names = [f"sar_{model}" for model in ALL_MODEL_NAMES] + ["sar_multi", "sar_average"]
        assert len(names) == 22
        for name in names:
            with pytest.raises(KeyError):
                getattr(sars, name)(bad)


def test_power_law_spot_check():
    """Quick sanity check using confirmed R values (no R script needed)."""
    partial = pd.DataFrame(
        {
            "area": [0.20, 0.90, 1.00, 1.80, 1.87, 4.40, 7.10, 7.50, 18.00, 20.00],
            "species": [48, 7, 52, 14, 42, 22, 103, 48, 79, 119],
        }
    )
    fit = sars.sar_power(partial)
    assert fit.converged is True
    assert 0.0 < fit.params["z"] < 1.0
    assert fit.params["c"] > 0
    assert 0.0 < fit.r_squared <= 1.0


# ---------------------------------------------------------------------------
# Every model on the galap data (uses cached galap_fits); agreement with R is
# in test_r_reference.py
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "power",
    "powerR",
    "epm1",
    "epm2",
    "p1",
    "p2",
    "loga",
    "koba",
    "mmf",
    "monod",
    "negexpo",
    "chapman",
    "weibull3",
    "asymp",
    "ratio",
    "gompertz",
    "weibull4",
    "betap",
    "heleg",
    "linear",
]


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_converges(model_name, galap_fits):
    """Every model should converge on the galap dataset."""
    fit = galap_fits[model_name]
    assert fit.converged is True, f"{model_name} did not converge"
    assert 0.0 < fit.r_squared <= 1.0, f"{model_name} R² out of range: {fit.r_squared}"


@pytest.mark.parametrize("model", ["epm1", "epm2"])
def test_fitting_is_quiet(galap, model):
    """The multi-start search visits parameters where a model overflows; those are
    penalised, not worth a numpy warning each (#82). The extended power models
    overflow most."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = getattr(sars, f"sar_{model}")(galap)
    assert fit.converged
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_predict(model_name, galap, galap_fits):
    """Predict should return values for all models."""
    fit = galap_fits[model_name]
    if not fit.converged:
        pytest.skip(f"{model_name} did not converge")
    pred = fit.predict(galap["area"].values)
    assert len(pred) == len(galap)
    assert all(pd.notna(pred)), f"{model_name} predict returned NaN"


# ---------------------------------------------------------------------------
# Parameters searched on a log scale
# ---------------------------------------------------------------------------


def test_a_log_scale_parameter_reaches_tiny_values(galap_fits):
    """weibull4's optimum on galap has c below 1e-20 (R's, propped up by rounding,
    is around 4e-15); a linear search stops short of it."""
    assert 0 < galap_fits["weibull4"].params["c"] < 1e-10


def test_a_warm_start_takes_natural_scale_values(galap, galap_fits):
    from sars._models import _fit_nls

    fit = galap_fits["weibull4"]
    warm = _fit_nls("weibull4", galap, start_from=fit.params)
    # weibull4 sits on a flat ridge here, and where a solver stops along it varies
    # with the platform's floating point by a few thousandths of an AICc unit. A
    # start misread as log-scale values would be whole units away.
    assert warm.aicc == pytest.approx(fit.aicc, abs=1e-2)


def test_a_zero_lower_bound_becomes_the_log_floor():
    from sars._models import _LOG_FLOOR, _to_natural_scale, _to_search_scale

    assert _to_search_scale([0.0, 2.0, 1.0], [0, 2]) == [_LOG_FLOOR, 2.0, 0.0]
    np.testing.assert_allclose(_to_natural_scale([0.0, 2.0], [0]), [1.0, 2.0])


def test_epm1_fits_areas_in_square_metres(galap):
    """With areas in m², epm1's c falls to about 1e-49. R sars 2.1.1 (seeded as in
    tests/r_reference) reaches AICc 190.356 there."""
    fit = sars.sar_epm1(galap.assign(area=galap["area"] * 1e6))
    assert fit.aicc <= 190.356 + 1e-3
    assert fit.params["c"] < 1e-40


def test_a_warm_start_must_be_positive_for_a_log_scale_parameter(galap):
    from sars._models import _fit_nls

    with pytest.raises(ValueError, match="must be positive"):
        _fit_nls("epm1", galap, start_from={"c": -1.0, "z": 0.3, "d": 0.1})


# Models whose form survives a change of units: rescaling area or richness
# changes their parameters, not their fit. epm1 and epm2's exponents depend on
# the area itself, so their fit changes with the area units.
UNIT_FREE = [m for m in ALL_MODEL_NAMES if m not in {"epm1", "epm2"}]


@pytest.mark.parametrize("model", UNIT_FREE)
def test_a_fit_is_the_same_in_any_units(galap, galap_fits, model):
    """Areas in m² instead of km², richness ten times larger: the same fit. Before
    the fit ran on rescaled data, negexpo, asymp, gompertz and chapman fitted
    nothing (R² 0) in m², and their fixed start values and bounds were the cause."""
    from sars._models import _fit_nls

    in_km2 = galap_fits[model]
    in_m2 = _fit_nls(model, galap.assign(area=galap["area"] * 1e6, species=galap["species"] * 10))
    assert in_m2.converged
    assert in_m2.r_squared == pytest.approx(in_km2.r_squared, abs=1e-6)
    areas = galap["area"].to_numpy()
    np.testing.assert_allclose(in_m2.predict(areas * 1e6), 10 * in_km2.predict(areas), rtol=1e-4)


@pytest.mark.parametrize("model", ALL_MODEL_NAMES)
def test_rescaling_converts_parameters_both_ways(model):
    """Each model's rescaled parameters convert to the data's units and back, and
    the curve is the same: S(A) = s * S'(A / a)."""
    from sars._models import _AREA_UNSCALED, _MODEL_REGISTRY, _RESCALE

    spec = _MODEL_REGISTRY[model]
    to_natural, to_scaled = _RESCALE[model]
    a = 1.0 if model in _AREA_UNSCALED else 37.0
    s, rng = 250.0, np.random.default_rng(0)
    scaled = np.array([np.mean(g) for g in spec.start_grid]) * rng.uniform(
        0.8, 1.2, len(spec.start_grid)
    )
    natural = np.array(to_natural(scaled, a, s))
    np.testing.assert_allclose(to_scaled(natural, a, s), scaled, rtol=1e-10)
    areas = np.geomspace(0.5, 2000, 9)
    np.testing.assert_allclose(
        spec.func(areas, *natural), s * spec.func(areas / a, *scaled), rtol=1e-9
    )


def test_a_fit_whose_parameters_overflow_in_the_data_units_fails(galap, monkeypatch):
    from sars import _models

    to_natural, to_scaled = _models._RESCALE["power"]
    monkeypatch.setitem(_models._RESCALE, "power", (lambda p, a, s: [np.inf, p[1]], to_scaled))
    fit = _models._fit_nls("power", galap)
    assert fit.converged is False and np.isnan(fit.r_squared)


def test_a_warm_start_that_cant_be_rescaled_falls_back_to_the_grid(galap, monkeypatch):
    from sars import _models

    to_natural, _ = _models._RESCALE["power"]
    monkeypatch.setitem(_models._RESCALE, "power", (to_natural, lambda p, a, s: [np.inf, p[1]]))
    fit = _models._fit_nls("power", galap, start_from={"c": 30.0, "z": 0.3})
    assert fit.converged and fit.r_squared == pytest.approx(0.491247, abs=1e-5)


def test_r_squared_is_nan_when_richness_doesnt_vary():
    data = pd.DataFrame({"area": [1.0, 2.0, 5.0, 10.0], "species": [7.0] * 4})
    fit = sars.sar_power(data)
    assert np.isnan(fit.r_squared)


def test_a_model_needs_a_rescaling_entry():
    from sars import _models

    with pytest.raises(ValueError, match="rescaling entry"):
        _models._register(
            _models._ModelSpec(
                name="unknown", func=lambda a, c: c + 0 * a, param_names=["c"],
                bounds_lower=[0.0], bounds_upper=[1.0], start_grid=[[0.5]],
            )
        )  # fmt: skip


class TestObservedShape:
    """Curves whose shape is known, classified on [1, 100]."""

    @pytest.mark.parametrize(
        ("curve", "expected"),
        [
            (lambda a: 3 + 2 * a, "linear"),
            (lambda a: 50 - 0.2 * a, "linear"),
            (lambda a: np.full_like(a, 7.0), "linear"),
            (lambda a: 5 * a**0.3, "convex"),
            (lambda a: 10 * np.log(a + 1), "convex"),
            (lambda a: 80 * (1 - np.exp(-0.05 * a)), "convex"),
            (lambda a: 80 / (1 + np.exp(-0.2 * (a - 40))), "sigmoid"),
            (lambda a: 80 * np.exp(-np.exp(-0.1 * (a - 30))), "sigmoid"),
            # The inflection point is below the range: only the decelerating part shows.
            (lambda a: 80 / (1 + np.exp(-0.2 * (a + 20))), "convex"),
            # ...or above it: only the accelerating part.
            (lambda a: 80 / (1 + np.exp(-0.05 * (a - 300))), "none"),
            (lambda a: 0.01 * a**2, "none"),
            (lambda a: 5 * a**0.5 * np.exp(-0.02 * a), "none"),  # falls after A = 25
            (lambda a: a + 5 * np.sin(a / 5), "none"),  # several inflection points
            (lambda a: np.full_like(a, np.nan), "none"),
            # Flat but for rounding noise: still a line of slope zero.
            (lambda a: 200 + 1e-13 * np.sin(a), "linear"),
            # A bend too small to see doesn't make a convex curve sigmoid.
            (lambda a: 5 * a**0.3 + 1e-4 * 60 / (1 + np.exp(-(a - 3))), "convex"),
        ],
    )
    def test_known_curves(self, curve, expected):
        from sars._models import _curve_shape

        assert _curve_shape(curve, 1.0, 100.0) == expected

    def test_a_bend_among_the_small_islands_is_seen(self):
        """Areas spanning eight orders of magnitude: the sigmoid's rising slope sits
        below a thousandth of the range, where an even grid has no points."""
        from sars._models import _curve_shape

        def curve(a):
            return 80 * a**2 / (1e-4 + a**2)

        assert _curve_shape(curve, 1e-3, 1e5) == "sigmoid"

    @pytest.mark.parametrize(
        ("curve", "expected"),
        [
            # Among the largest islands a slope is tiny beside one among the smallest:
            # a fall of four species there, or a tail that accelerates, must still count.
            (lambda a: 5 * a**0.25 * np.exp(-a / 2e5), "none"),
            (lambda a: 5 * a**0.25 + 1e-8 * a**2, "none"),
            # ...unless it is too small to see: this tail bends up by 0.003 species.
            (lambda a: 5 * a**0.25 + 1e-9 * a**2, "convex"),
            (lambda a: 5 * a**0.25, "convex"),
        ],
    )
    def test_areas_spanning_many_orders_of_magnitude(self, curve, expected):
        from sars._models import _curve_shape

        assert _curve_shape(curve, 1e-3, 1e5) == expected

    def test_a_small_rise_on_a_large_intercept_is_still_convex(self):
        from sars._models import _curve_shape

        def curve(a):
            return 979.316529976624 + 0.6524907527948977 * a**0.11493479713945963

        assert _curve_shape(curve, 0.08536470703959534, 10.669173905822692) == "convex"

    def test_rounding_noise_is_not_a_bend(self):
        from sars._models import _curve_shape

        rng = np.random.default_rng(0)

        def curve(a):
            return 5 * a**0.3 * (1 + 1e-13 * rng.standard_normal(a.shape))

        assert _curve_shape(curve, 1.0, 100.0) == "convex"

    def test_one_distinct_area_has_no_shape(self):
        from sars._models import _curve_shape

        assert _curve_shape(lambda a: a, 5.0, 5.0) == "none"

    def test_fits_recover_the_shape_of_their_data(self):
        area = np.geomspace(1, 1000, 25)
        convex = pd.DataFrame({"area": area, "species": 5 * area**0.25})
        assert sars.sar_power(convex).observed_shape() == "convex"
        line = pd.DataFrame({"area": area, "species": 3 + 0.1 * area})
        assert sars.sar_linear(line).observed_shape() == "linear"
        area = np.linspace(1, 100, 25)
        s_curve = pd.DataFrame({"area": area, "species": 80 * np.exp(-np.exp(-0.1 * (area - 30)))})
        assert sars.sar_gompertz(s_curve).observed_shape() == "sigmoid"

    def test_a_failed_fit_has_no_shape(self):
        from sars._models import _failed_fit

        data = pd.DataFrame({"area": [1.0, 2.0, 3.0], "species": [1.0, 2.0, 3.0]})
        assert _failed_fit("power", ["c", "z"], 3, data).observed_shape() == "none"


TABLE_COLUMNS = ["estimate", "std_error", "t_value", "p_value", "ci_lower", "ci_upper"]


_X = np.linspace(1.0, 10.0, 12)


_Y = 3.0 + 2.0 * _X + np.sin(5.0 * _X)  # a line plus something it can't fit


def _wald_std_errors(jac: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """sqrt(diag(sigma^2 (J'J)^-1)), written out for an exact Jacobian."""
    n, p = jac.shape
    cov = residuals @ residuals / (n - p) * np.linalg.inv(jac.T @ jac)
    return np.sqrt(np.diag(cov))


def _quiet_table(fit: sars.SARFit, **kwargs) -> pd.DataFrame:
    """The table of a fit that must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit.param_table(**kwargs)


@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
def test_param_table_has_a_row_per_parameter(model_name, galap_fits):
    fit = galap_fits[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        table = fit.param_table()
    assert list(table.columns) == TABLE_COLUMNS
    assert list(table.index) == list(fit.params)
    assert table.index.name == "parameter"
    assert table["estimate"].to_dict() == fit.params


def test_param_table_matches_the_analytic_jacobian(galap, galap_fits):
    """The power model's derivatives are known exactly: a^z and c a^z ln(a)."""
    fit = galap_fits["power"]
    area = galap["area"].to_numpy(float)
    c, z = fit.params["c"], fit.params["z"]
    jac = np.column_stack([area**z, c * area**z * np.log(area)])
    expected = _wald_std_errors(jac, galap["species"].to_numpy(float) - fit.predict(area))
    table = _quiet_table(fit)
    np.testing.assert_allclose(table["std_error"], expected, rtol=1e-7)


def test_param_table_of_the_linear_model_is_ordinary_least_squares(galap, galap_fits):
    area = galap["area"].to_numpy(float)
    species = galap["species"].to_numpy(float)
    design = np.column_stack([np.ones_like(area), area])
    coef = np.linalg.lstsq(design, species, rcond=None)[0]
    expected = _wald_std_errors(design, species - design @ coef)
    table = _quiet_table(galap_fits["linear"])
    np.testing.assert_allclose(table["std_error"], expected, rtol=1e-6)


def test_param_table_tests_and_intervals_use_the_t_distribution(galap_fits):
    fit = galap_fits["power"]
    table = _quiet_table(fit)
    dof = fit.n - 2
    np.testing.assert_allclose(table["t_value"], table["estimate"] / table["std_error"])
    np.testing.assert_allclose(table["p_value"], 2 * stats.t.sf(table["t_value"].abs(), dof))
    half = stats.t.ppf(0.975, dof) * table["std_error"]
    np.testing.assert_allclose(table["ci_lower"], table["estimate"] - half)
    np.testing.assert_allclose(table["ci_upper"], table["estimate"] + half)
    # Wider than the estimate ± 2 standard errors R reports as its interval.
    assert np.all(half > 2 * table["std_error"])


def test_param_table_interval_widens_with_the_confidence_level(galap_fits):
    fit = galap_fits["power"]
    narrow, wide = _quiet_table(fit, conf=0.8), _quiet_table(fit, conf=0.99)
    assert np.all(wide["ci_lower"] < narrow["ci_lower"])
    assert np.all(wide["ci_upper"] > narrow["ci_upper"])
    # Only the interval depends on the level.
    pd.testing.assert_frame_equal(wide[TABLE_COLUMNS[:4]], narrow[TABLE_COLUMNS[:4]])
    half = stats.t.ppf(0.9, fit.n - 2) * narrow["std_error"]
    np.testing.assert_allclose(narrow["ci_upper"] - narrow["estimate"], half)


@pytest.mark.parametrize("conf", [0.0, 1.0, 1.5, -0.1, float("nan")])
def test_param_table_rejects_a_level_outside_0_to_1(conf, galap_fits):
    with pytest.raises(ValueError, match="conf must be between 0 and 1"):
        galap_fits["power"].param_table(conf=conf)


def test_param_table_has_no_standard_error_for_a_parameter_on_a_bound(galap_fits):
    """weibull4 trades c toward 0 against a growing z on the Galápagos data, a
    flat ridge that runs to z's upper bound of 10. There the optimum is the
    bound's, and a standard error doesn't apply; c's dwarfs its estimate."""
    fit = galap_fits["weibull4"]
    with pytest.warns(UserWarning, match="'weibull4': z stopped on a bound"):
        table = fit.param_table()
    assert table["estimate"].to_dict() == fit.params
    assert table.loc["z", TABLE_COLUMNS[1:]].isna().all()
    assert table.drop(index="z").notna().all(axis=None)
    assert table.loc["c", "std_error"] > 10 * table.loc["c", "estimate"]


def test_param_table_warns_where_the_curve_cant_be_differentiated(monkeypatch):
    """A curve that is rough in one parameter: the warning names it, and it and
    the parameters whose standard errors depend on it have none."""
    from sars import _models

    def func(a, c, m, q):
        return c + m * a + _rough(q) * a**2

    spec = _models._ModelSpec(
        name="rough",
        func=func,
        param_names=["c", "m", "q"],
        bounds_lower=[-1e6] * 3,
        bounds_upper=[1e6] * 3,
        start_grid=[[1.0]] * 3,
    )
    monkeypatch.setitem(_models._MODEL_REGISTRY, "rough", spec)
    data = pd.DataFrame({"area": _X, "species": _Y})
    fit = _hand_built_fit("rough", data)
    fit.params = {"c": 3.0, "m": 2.0, "q": 0.0}
    with pytest.warns(
        UserWarning, match="'rough': the curve can't be differentiated reliably with respect to q"
    ):
        table = fit.param_table()
    assert np.isnan(table.loc["q", "std_error"])


def test_param_table_names_the_parameters_the_data_dont_identify(monkeypatch):
    """A model in which only c1 + c2 matters: neither has a standard error,
    and the slope keeps its own."""
    from sars import _models

    spec = _models._ModelSpec(
        name="ridge",
        func=lambda a, c1, c2, m: c1 + c2 + m * a,
        param_names=["c1", "c2", "m"],
        bounds_lower=[-1e6] * 3,
        bounds_upper=[1e6] * 3,
        start_grid=[[0.0], [0.0], [1.0]],
    )
    monkeypatch.setitem(_models._MODEL_REGISTRY, "ridge", spec)
    data = pd.DataFrame({"area": _X, "species": _Y})
    fit = sars.SARFit(
        model="ridge",
        params={"c1": 1.0, "c2": 2.0, "m": 2.0},
        r_squared=np.nan,
        aic=np.nan,
        aicc=np.nan,
        bic=np.nan,
        n=len(_X),
        converged=True,
        data=data,
    )
    with pytest.warns(UserWarning, match="'ridge': the data don't identify c1, c2, so"):
        table = fit.param_table()
    assert table.loc[["c1", "c2"], TABLE_COLUMNS[1:]].isna().all(axis=None)
    assert table.loc["m", TABLE_COLUMNS[1:]].notna().all()


def _hand_built_fit(model: str, data: pd.DataFrame, converged: bool = True) -> sars.SARFit:
    return sars.SARFit(
        model=model,
        params={"c": 1.0, "m": 2.0},
        r_squared=np.nan,
        aic=np.nan,
        aicc=np.nan,
        bic=np.nan,
        n=len(data),
        converged=converged,
        data=data,
    )


def test_param_table_of_an_unknown_model_says_so_like_predict():
    data = pd.DataFrame({"area": _X, "species": _Y})
    with pytest.raises(NotImplementedError, match="param_table not yet implemented for 'nope'"):
        _hand_built_fit("nope", data).param_table()
    # A failed fit has nothing to compute, whatever its model.
    table = _quiet_table(_hand_built_fit("nope", data, converged=False))
    assert list(table.index) == ["c", "m"]
    assert table[TABLE_COLUMNS[1:]].isna().all(axis=None)


@pytest.mark.parametrize("change", ["a value", "the rows"])
def test_param_table_refuses_data_changed_since_the_fit(change):
    """The fit keeps the caller's DataFrame, not a copy; a table computed from
    other data than the parameters were fitted to would be silently wrong."""
    data = pd.DataFrame({"area": _X, "species": _Y})
    fit = sars.sar_linear(data)
    _quiet_table(fit)
    if change == "a value":
        data.loc[0, "species"] += 5.0
    else:
        data.drop(index=0, inplace=True)
    with pytest.raises(ValueError, match="data of this 'linear' fit have changed"):
        fit.param_table()


def test_param_table_has_no_standard_errors_without_degrees_of_freedom():
    data = pd.DataFrame({"area": [1.0, 10.0], "species": [5.0, 12.0]})
    fit = sars.sar_power(data)
    assert fit.converged
    table = _quiet_table(fit)
    assert table["estimate"].to_dict() == fit.params
    assert table[TABLE_COLUMNS[1:]].isna().all(axis=None)


def test_param_table_of_a_failed_fit_is_all_nan(galap):
    from sars._models import _failed_fit

    fit = _failed_fit("power", ["c", "z"], len(galap), galap)
    table = _quiet_table(fit)
    assert list(table.index) == ["c", "z"]
    assert table.isna().all(axis=None)


def test_param_table_of_an_exact_fit_has_zero_standard_errors():
    data = pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0], "species": [3.0, 5.0, 7.0, 9.0]})
    fit = sars.SARFit(
        model="linear",
        params={"c": 1.0, "m": 2.0},
        r_squared=1.0,
        aic=np.nan,
        aicc=np.nan,
        bic=np.nan,
        n=4,
        converged=True,
        data=data,
    )
    table = _quiet_table(fit)
    assert (table["std_error"] == 0).all()
    assert np.isinf(table["t_value"]).all()
    assert (table["p_value"] == 0).all()
    assert (table["ci_lower"] == table["estimate"]).all()


def test_an_exponents_standard_error_doesnt_depend_on_the_area_units(galap, galap_fits):
    """The power model's z has no units, so its row is the same in m² as in
    km². (c's isn't: c absorbs the unit change through z, as c * 1e6^-z.)"""
    metres = galap.assign(area=galap["area"] * 1e6)
    km, m2 = _quiet_table(galap_fits["power"]), _quiet_table(sars.sar_power(metres))
    np.testing.assert_allclose(m2.loc["z"], km.loc["z"], rtol=1e-5)


def _line_std_errors() -> np.ndarray:
    design = np.column_stack([np.ones_like(_X), _X])
    return _wald_std_errors(design, _Y - (3.0 + 2.0 * _X))


def test_a_parameter_the_curve_ignores_has_no_standard_error():
    def func(a, c, m, unused):
        return c + m * a

    se, rough = _param_std_errors(func, _X, _Y, np.array([3.0, 2.0, 7.0]))
    assert not rough.any()
    assert np.isnan(se[2])
    # The others are what they'd be with it held fixed (on n - p degrees of
    # freedom, which count it).
    np.testing.assert_allclose(se[:2], _line_std_errors() * np.sqrt(10 / 9), rtol=1e-6)


def test_a_curve_that_ignores_every_parameter_has_no_standard_errors():
    def func(a, c, m):
        return 3.0 + 2.0 * a

    se, rough = _param_std_errors(func, _X, _Y, np.array([3.0, 2.0]))
    assert not rough.any()
    assert np.isnan(se).all()


def test_parameters_on_a_ridge_have_no_standard_error_but_others_do():
    """Only c1 + c2 matters, so neither is identified; the slope still is."""

    def func(a, c1, c2, m):
        return c1 + c2 + m * a

    se, rough = _param_std_errors(func, _X, _Y, np.array([1.0, 2.0, 2.0]))
    assert not rough.any()
    assert np.isnan(se[:2]).all()
    assert se[2] == pytest.approx(_line_std_errors()[1] * np.sqrt(10 / 9), rel=1e-6)


def test_a_curve_that_cant_be_differentiated_has_no_standard_errors():
    def breaks(a, c, m):  # undefined anywhere but at the fitted slope
        return c + m * a if m == 2.0 else np.full_like(a, np.nan)

    se, rough = _param_std_errors(breaks, _X, _Y, np.array([3.0, 2.0]))
    assert rough.tolist() == [False, True]
    assert np.isnan(se).all()

    def undefined(a, c, m):  # not even defined at the fitted values
        return np.full_like(a, np.nan)

    se, rough = _param_std_errors(undefined, _X, _Y, np.array([3.0, 2.0]))
    assert rough.all()
    assert np.isnan(se).all()

    def breaks_nearby(a, c, m):  # defined at the first step tried, not the finer one
        return np.full_like(a, np.nan) if 0 < abs(m - 2.0) < 1e-5 else c + m * a

    se, rough = _param_std_errors(breaks_nearby, _X, _Y, np.array([3.0, 2.0]))
    assert rough.tolist() == [False, True]
    assert np.isnan(se).all()

    rng = np.random.default_rng(0)

    def noisy(a, c, m):  # rounding noise far larger than a finite-difference step
        return c + m * a + rng.normal(scale=1e-3, size=a.shape)

    se, rough = _param_std_errors(noisy, _X, _Y, np.array([3.0, 2.0]))
    assert rough.all()
    assert np.isnan(se).all()


@pytest.mark.parametrize("top", [10.0, 5000.0, 5e9])
@pytest.mark.parametrize("d", [0.0, 1e-42, 1e-310])
def test_a_parameter_at_or_near_zero_still_gets_a_standard_error(d, top):
    """The step has to come from how the curve responds, not from the size of
    the parameter: p1's d reaches 1e-42 on small samples, a log-scale parameter
    1e-300, and the response to d grows with the largest area."""

    def func(a, c, z, d):
        return c * a**z * np.exp(-d * a)

    area = np.geomspace(1.0, top, 12)
    values = np.array([4.0, 0.3, d])
    model = func(area, *values)
    species = model * (1.0 + 0.1 * np.sin(5.0 * _X))
    jac = np.column_stack([model / 4.0, model * np.log(area), -area * model])
    se, rough = _param_std_errors(func, area, species, values)
    assert not rough.any()
    np.testing.assert_allclose(se, _wald_std_errors(jac, species - model), rtol=1e-6)


def test_a_step_that_would_leave_the_models_domain_is_taken_one_sided():
    """q must not be negative, and the curve responds to it so weakly that a
    step big enough to see is bigger than q itself."""

    def func(a, c, m, q):
        return c + m * a + 1e-8 * q * a**2 if q >= 0 else np.full_like(a, np.nan)

    values = np.array([3.0, 2.0, 1e-3])
    jac = np.column_stack([np.ones_like(_X), _X, 1e-8 * _X**2])
    se, rough = _param_std_errors(func, _X, _Y, values)
    assert not rough.any()
    np.testing.assert_allclose(se, _wald_std_errors(jac, _Y - func(_X, *values)), rtol=1e-6)


def _apart_from_the_line(values: np.ndarray) -> np.ndarray:
    """What's left of ``values`` after regressing them on a constant and _X."""
    design = np.column_stack([np.ones_like(_X), _X])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def _rough(q: float) -> float:
    """A function of q whose slope can't be found by differencing."""
    return q + 1e-7 * np.sin(1e9 * q)


def test_a_rough_derivative_blanks_only_the_parameters_that_depend_on_it():
    separate = _apart_from_the_line(_X**2)

    def func(a, c, m, q):  # q's column is orthogonal to the line's
        return c + m * a + _rough(q) * separate

    se, rough = _param_std_errors(func, _X, _Y, np.array([3.0, 2.0, 0.0]))
    assert rough.tolist() == [False, False, True]
    assert np.isnan(se[2])
    np.testing.assert_allclose(se[:2], _line_std_errors() * np.sqrt(10 / 9), rtol=1e-6)

    def tangled(a, c, m, q):  # q's column, a^2, is correlated with the line's
        return c + m * a + _rough(q) * a**2

    se, rough = _param_std_errors(tangled, _X, _Y, np.array([3.0, 2.0, 0.0]))
    assert rough.tolist() == [False, False, True]
    assert np.isnan(se).all()


def test_a_parameter_near_a_ridge_but_not_on_it_keeps_its_standard_error():
    """c1 and c2 are all but interchangeable; the slope is correlated with what
    little tells them apart, so its loading on the ridge isn't exactly zero,
    but it is as well determined as in the model that names that term."""
    wiggle = np.sin(_X)

    def func(a, c1, c2, m):
        return c1 + c2 * (1.0 + 1e-6 * wiggle) + m * a

    values = np.array([1.0, 2.0, 2.0])
    se, rough = _param_std_errors(func, _X, _Y, values)
    assert not rough.any()
    assert np.isnan(se[:2]).all()
    named = np.column_stack([np.ones_like(_X), wiggle, _X])
    expected = _wald_std_errors(named, _Y - func(_X, *values))[2]
    assert se[2] == pytest.approx(expected, rel=1e-4)
    # ... which is not what it would be were c1 and c2 one parameter.
    assert abs(expected / (_line_std_errors()[1] * np.sqrt(10 / 9)) - 1) > 1e-3
