import numpy as np
import pandas as pd
import pytest

import sars  # noqa: I001
from sars._inference import ALL_MODEL_NAMES

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
    """R's weibull4 optimum on galap has c around 4e-15; a linear search stops
    short of it."""
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
    if model == "weibull4":
        # Its likelihood is flat along a ridge: the solver stops at a slightly
        # different point on it, as good a fit to 1e-4 in R², a curve up to 2% apart.
        assert in_m2.r_squared == pytest.approx(in_km2.r_squared, abs=1e-4)
        return
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
