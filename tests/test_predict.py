"""Tests for ``sar_pred``.

The R numbers are in ``tests/r_reference/pred_galap.csv``, written by
``generate_r_reference.R``: R ``sars``'s ``sar_pred`` on the Galápagos data at
areas 50, 500 and 10000, for the power, loga and negexpo models and for the
average of those three.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sars
from sars._models import _failed_fit

REF = Path(__file__).parent / "r_reference"
R_PRED = pd.read_csv(REF / "pred_galap.csv")
AREAS = [50.0, 500.0, 10000.0]
R_MODELS = ["power", "loga", "negexpo"]

# Both packages find each optimum iteratively, and R's starting values are
# partly random, so its predictions move in the fifth figure between seeds.
PRED_RTOL = 1e-3


def _r_predictions(model):
    rows = R_PRED[R_PRED["model"] == model].set_index("area")
    return rows.loc[AREAS, "prediction"].to_numpy()


@pytest.mark.parametrize("model", R_MODELS)
def test_single_model_matches_r(model, galap, galap_fits):
    pred = sars.sar_pred(galap_fits[model], AREAS)
    assert list(pred.columns) == ["model", "area", "prediction"]
    assert list(pred["model"]) == [model] * 3
    assert list(pred["area"]) == AREAS
    np.testing.assert_allclose(pred["prediction"], _r_predictions(model), rtol=PRED_RTOL)


def test_average_matches_r_over_the_same_models(galap):
    """R's averaged prediction is over the models it averaged, so it is compared
    over the same three; an average over all 20 is a different number."""
    avg = sars.sar_average(galap, models=R_MODELS)
    pred = sars.sar_pred(avg, AREAS)
    assert list(pred["model"]) == ["average"] * 3
    assert list(pred["area"]) == AREAS
    np.testing.assert_allclose(pred["prediction"], _r_predictions("average"), rtol=PRED_RTOL)


def test_average_over_other_models_differs_from_r(galap_average):
    pred = sars.sar_pred(galap_average, AREAS)
    assert not np.allclose(pred["prediction"], _r_predictions("average"), rtol=PRED_RTOL)


def test_multi_gives_each_model_in_summary_order(galap):
    multi = sars.sar_multi(galap, models=["linear", "negexpo", "power", "loga"])
    order = list(multi.summary["model"])
    assert order != ["linear", "negexpo", "power", "loga"]  # sorted by AICc, not as given

    pred = sars.sar_pred(multi, AREAS)

    assert len(pred) == len(order) * len(AREAS)
    assert list(pred["model"]) == [m for m in order for _ in AREAS]
    assert list(pred["area"]) == AREAS * len(order)
    assert list(pred.index) == list(range(len(pred)))
    for fit in multi.fits:
        rows = pred[pred["model"] == fit.model]
        np.testing.assert_array_equal(rows["prediction"], fit.predict(np.array(AREAS)))
    for model in R_MODELS:
        rows = pred[pred["model"] == model]
        np.testing.assert_allclose(rows["prediction"], _r_predictions(model), rtol=PRED_RTOL)


def test_average_is_the_weighted_sum_of_the_models(galap):
    avg = sars.sar_average(galap, models=R_MODELS)
    each = sars.sar_pred(avg.multi, AREAS)
    weighted = sum(
        avg.weights[m] * each.loc[each["model"] == m, "prediction"].to_numpy() for m in R_MODELS
    )
    np.testing.assert_allclose(sars.sar_pred(avg, AREAS)["prediction"], weighted)


def test_threshold_fit_uses_its_best_model(galap):
    th = sars.sar_threshold(galap)
    pred = sars.sar_pred(th, AREAS)
    assert list(pred["model"]) == [th.best_model] * 3
    np.testing.assert_array_equal(pred["prediction"], th.predict(np.array(AREAS)))


def test_lin_pow_fit(galap):
    fit = sars.lin_pow(galap)
    pred = sars.sar_pred(fit, AREAS)
    assert list(pred["model"]) == ["lin_pow"] * 3
    np.testing.assert_allclose(pred["prediction"], fit.c * np.array(AREAS) ** fit.z)


def test_scalar_area_gives_one_row(galap_fits):
    pred = sars.sar_pred(galap_fits["power"], 5000)
    assert len(pred) == 1
    assert pred.loc[0, "model"] == "power"
    assert pred.loc[0, "area"] == 5000.0
    assert pred.loc[0, "prediction"] == pytest.approx(float(galap_fits["power"].predict(5000.0)))
    assert pred["area"].dtype == float
    assert pred["prediction"].dtype == float


@pytest.mark.parametrize("area", [np.array([50.0, 500.0]), (50, 500), pd.Series([50, 500])])
def test_array_like_areas(area, galap_fits):
    pred = sars.sar_pred(galap_fits["loga"], area)
    assert list(pred["area"]) == [50.0, 500.0]


def test_areas_keep_their_order_and_repeats(galap_fits):
    pred = sars.sar_pred(galap_fits["power"], [500.0, 50.0, 500.0])
    assert list(pred["area"]) == [500.0, 50.0, 500.0]
    assert pred.loc[0, "prediction"] == pred.loc[2, "prediction"]
    assert pred.loc[1, "prediction"] < pred.loc[0, "prediction"]


@pytest.mark.parametrize("area", [0, -1.0, [50.0, 0.0], [50.0, -3.0]])
def test_non_positive_area_raises(area, galap_fits):
    with pytest.raises(ValueError, match="positive"):
        sars.sar_pred(galap_fits["power"], area)


@pytest.mark.parametrize("area", [np.nan, np.inf, [50.0, np.nan], [50.0, -np.inf]])
def test_non_finite_area_raises(area, galap_fits):
    with pytest.raises(ValueError, match="NaN or infinite"):
        sars.sar_pred(galap_fits["power"], area)


def test_empty_or_2d_area_raises(galap_fits):
    with pytest.raises(ValueError, match="empty"):
        sars.sar_pred(galap_fits["power"], [])
    with pytest.raises(ValueError, match="1-D"):
        sars.sar_pred(galap_fits["power"], [[50.0, 500.0]])


def test_bad_area_raises_for_every_kind_of_fit(galap, galap_average):
    fits = [
        sars.sar_multi(galap, models=["linear"]),
        galap_average,
        sars.sar_threshold(galap),
        sars.lin_pow(galap),
    ]
    for fit in fits:
        with pytest.raises(ValueError, match="positive"):
            sars.sar_pred(fit, [50.0, -1.0])
        with pytest.raises(ValueError, match="NaN or infinite"):
            sars.sar_pred(fit, [50.0, np.nan])


def test_threshold_fit_on_untransformed_area_takes_any_finite_area(galap):
    """Its lines are defined at zero and below, and its own predict takes them."""
    th = sars.sar_threshold(galap, log_area=False)
    areas = np.array([-1.0, 0.0, 50.0])
    pred = sars.sar_pred(th, areas)
    assert list(pred["area"]) == list(areas)
    np.testing.assert_array_equal(pred["prediction"], th.predict(areas))
    with pytest.raises(ValueError, match="NaN or infinite"):
        sars.sar_pred(th, [0.0, np.inf])


@pytest.mark.parametrize(
    "area", ["50", ["50", "500"], True, [True, False], None, [50.0, None], [50.0, "500"]]
)
def test_non_numeric_area_raises(area, galap_fits):
    with pytest.raises(TypeError, match="'area' must be numeric"):
        sars.sar_pred(galap_fits["power"], area)


def test_integer_areas_are_numeric(galap_fits):
    pred = sars.sar_pred(galap_fits["power"], np.array([50, 500], dtype=np.int32))
    assert list(pred["area"]) == [50.0, 500.0]
    assert pred["area"].dtype == float


def test_unknown_fit_type_raises(galap):
    with pytest.raises(TypeError, match="DataFrame"):
        sars.sar_pred(galap, 50.0)
    with pytest.raises(TypeError, match="fit must be"):
        sars.sar_pred(None, 50.0)


def test_wrong_fit_is_reported_before_bad_areas(galap):
    with pytest.raises(TypeError, match="fit must be"):
        sars.sar_pred(galap, -1.0)
    with pytest.raises(TypeError, match="fit must be"):
        sars.sar_pred(galap, "50")


def test_unconverged_fit_raises(galap):
    failed = _failed_fit("power", ["c", "z"], len(galap), galap)
    with pytest.raises(ValueError, match="'power' fit did not converge"):
        sars.sar_pred(failed, AREAS)


def test_an_average_of_no_models_cant_be_built_or_predicted_from(galap):
    empty = sars.sar_multi(galap, models=[])
    with pytest.raises(ValueError, match="nothing to average"):
        sars.AveragedSAR(multi=empty, ic="AICc", weights={})
    avg = sars.sar_average(galap, models=["power"])
    avg.multi.fits.clear()  # emptied after it was built
    with pytest.raises(ValueError, match="nothing to average"):
        sars.sar_pred(avg, AREAS)


def test_multi_order_and_models_come_from_its_fits(galap):
    """Not from the summary table, which a hand-built or edited result may not match."""
    multi = sars.sar_multi(galap, models=["linear", "negexpo", "power", "loga"])
    by_aicc = list(multi.summary["model"])
    shuffled = sars.MultiSARFit(
        fits=multi.fits[::-1],
        data=galap,
        summary=multi.summary.iloc[:2].assign(model=["nonesuch", "power"]),
    )
    pred = sars.sar_pred(shuffled, 50.0)
    assert list(pred["model"]) == by_aicc


def test_multi_puts_fits_without_an_aicc_last(galap):
    multi = sars.sar_multi(galap, models=["power", "loga", "negexpo"])
    best_first = list(multi.summary["model"])
    fits = {f.model: f for f in multi.fits}
    no_aicc = dataclasses.replace(fits[best_first[0]], aicc=np.inf)
    nan_aicc = dataclasses.replace(fits[best_first[1]], aicc=np.nan)
    held = sars.MultiSARFit(
        fits=[no_aicc, nan_aicc, fits[best_first[2]]], data=galap, summary=multi.summary
    )
    pred = sars.sar_pred(held, 50.0)
    # The one finite AICc first; the other two in the order they are held.
    assert list(pred["model"]) == [best_first[2], best_first[0], best_first[1]]


def test_multi_without_fits_gives_no_rows(galap):
    empty = sars.sar_multi(galap, models=[])
    pred = sars.sar_pred(empty, AREAS)
    assert len(pred) == 0
    # The same table as any other, only empty.
    full = sars.sar_pred(sars.sar_multi(galap, models=["power"]), AREAS)
    assert list(pred.columns) == list(full.columns) == ["model", "area", "prediction"]
    assert list(pred.dtypes) == list(full.dtypes)
    single = sars.sar_pred(sars.sar_power(galap), 50.0)
    assert list(single.dtypes) == list(full.dtypes)
