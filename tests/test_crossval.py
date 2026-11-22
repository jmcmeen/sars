"""Leave-one-island-out cross-validation."""

import warnings

import numpy as np
import pandas as pd
import pytest

import sars
from sars import _crossval

FEW = ["power", "loga", "negexpo"]


@pytest.fixture(scope="module")
def galap_cv():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sars.loo_cv(sars.load_galap(), models=FEW)


def test_matches_leaving_each_island_out_by_hand(galap_cv):
    galap = sars.load_galap()
    preds = []
    for i in range(len(galap)):
        rest = galap.drop(index=galap.index[i]).reset_index(drop=True)
        preds.append(float(sars.sar_power(rest).predict([galap["area"].iloc[i]])[0]))
    np.testing.assert_allclose(
        galap_cv.predictions["power"], preds, rtol=1e-4
    )  # warm start vs search
    err = np.array(preds) - galap["species"].to_numpy(dtype=float)
    row = galap_cv.summary.set_index("model").loc["power"]
    assert row["rmse"] == pytest.approx(np.sqrt(np.mean(err**2)), rel=1e-4)
    assert row["mae"] == pytest.approx(np.mean(np.abs(err)), rel=1e-4)
    assert row["max_error"] == pytest.approx(np.max(np.abs(err)), rel=1e-4)
    assert row["worst"] == str(int(np.argmax(np.abs(err))))


def test_summary_is_ranked_and_has_aicc_ranks(galap_cv):
    summary = galap_cv.summary
    assert list(summary["cv_rank"]) == list(range(1, len(summary) + 1))
    assert summary["rmse"].is_monotonic_increasing
    assert set(summary["model"]) == {*FEW, "average"}
    aicc_order = galap_cv.multi.summary["model"].tolist()
    ranks = summary.set_index("model")["ic_rank"]
    assert [ranks[m] for m in aicc_order] == [1, 2, 3]
    assert pd.isna(ranks["average"])
    assert (summary["n_failed"] == 0).all()


def test_the_average_is_the_aicc_average_of_each_fold(galap_cv):
    galap = sars.load_galap()
    rest = galap.iloc[1:].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg = sars.sar_average(rest, models=FEW)
    expected = float(avg.predict([galap["area"].iloc[0]])[0])
    assert galap_cv.predictions["average"].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_warm_starts_agree_with_the_full_search_for_these_models(galap_cv):
    assert not galap_cv.warm_start  # the full search is the default
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = sars.loo_cv(sars.load_galap(), models=FEW, warm_start=True)
    assert warm.warm_start
    pd.testing.assert_frame_equal(
        warm.predictions, galap_cv.predictions, rtol=1e-4, check_exact=False
    )


def test_warm_starts_can_flatter_a_flexible_model():
    """Johnson and Raven's data: warm-started folds of weibull4 stay near the
    full fit, which has seen the island left out, and predict it better than
    the best fit to the other islands does."""
    gala = sars.load_gala()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = sars.loo_cv(gala, models=["weibull4"], warm_start=True)
        full = sars.loo_cv(gala, models=["weibull4"])
    rmse = lambda cv: cv.summary.set_index("model").loc["weibull4", "rmse"]  # noqa: E731
    assert rmse(warm) < 0.95 * rmse(full)


def test_n_jobs_gives_the_same_result(galap_cv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel = sars.loo_cv(sars.load_galap(), models=FEW, n_jobs=2)
    pd.testing.assert_frame_equal(parallel.summary, galap_cv.summary)


def test_island_names_label_the_worst_error():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = sars.loo_cv(sars.load_gala(), models=["power"])
    assert cv.summary.set_index("model").loc["power", "worst"] in set(sars.load_gala()["island"])


@pytest.mark.parametrize("warm_start", [True, False])
def test_failed_folds_are_counted_and_ranked_last(monkeypatch, warm_start):
    real = _crossval._fit_nls

    def flaky(name, data, start_from=None):
        if name == "loga" and len(data) == 15 and data["area"].max() < 1000:
            raise RuntimeError("no fit")  # the fold without the largest island
        return real(name, data, start_from=start_from)

    monkeypatch.setattr(_crossval, "_fit_nls", flaky)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = sars.loo_cv(sars.load_galap(), models=["power", "loga"], warm_start=warm_start)
    summary = cv.summary.set_index("model")
    assert summary.loc["loga", "n_failed"] == 1
    assert cv.summary["model"].iloc[-1] == "loga"
    assert cv.predictions["loga"].isna().sum() == 1
    # rmse_common leaves out the island loga missed, for every model.
    common = cv.predictions["loga"].notna()
    err = cv.predictions.loc[common, "power"] - sars.load_galap().loc[common, "species"]
    assert summary.loc["power", "rmse_common"] == pytest.approx(np.sqrt(np.mean(err**2)))


def test_too_few_islands():
    with pytest.raises(ValueError, match="at least 4 islands"):
        sars.loo_cv(sars.load_galap().head(3))


def test_saves_and_reloads(galap_cv):
    again = sars.from_json(sars.to_json(galap_cv))
    pd.testing.assert_frame_equal(again.summary, galap_cv.summary)
    pd.testing.assert_frame_equal(again.predictions, galap_cv.predictions)


def test_repr(galap_cv):
    assert repr(galap_cv).startswith("LOOCV(16 islands, 3 models, best=")


def test_a_repeated_index_label_leaves_out_one_island(galap_cv):
    galap = sars.load_galap()
    galap.index = [0, 0, *range(1, len(galap) - 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = sars.loo_cv(galap, models=FEW)
    np.testing.assert_allclose(
        cv.predictions.to_numpy(), galap_cv.predictions.to_numpy(), rtol=1e-6
    )


def test_few_islands_warn_and_ic_aic_gives_an_average():
    tiny = sars.load_galap().head(5)
    models = ["power", "loga", "negexpo", "koba"]
    with pytest.warns(UserWarning, match="Use ic='AIC'"):
        cv = sars.loo_cv(tiny, models=models)
    assert cv.summary.set_index("model").loc["average", "n_failed"] == 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aic = sars.loo_cv(tiny, models=models, ic="AIC")
    summary = aic.summary.set_index("model")
    assert aic.ic == "AIC" and "AIC" in aic.summary.columns
    assert summary.loc["average", "n_failed"] == 0
    assert sorted(summary["ic_rank"].dropna()) == [1, 2, 3, 4]


def test_ic_rank_is_missing_where_the_criterion_is_infinite():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = sars.loo_cv(sars.load_galap().head(4), models=["power", "loga"], ic="AICc")
    assert cv.summary["ic_rank"].isna().all()


@pytest.mark.parametrize(
    ("label", "expected"),
    [("auto", "Isabela"), ("island", "Isabela"), (None, "15")],
)
def test_labels(label, expected):
    gala = sars.load_gala().rename(columns={"island": "Island"} if label == "auto" else {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = sars.loo_cv(gala, models=["power"], label=label)
    assert cv.summary.set_index("model").loc["power", "worst"] == expected


def test_bad_arguments():
    with pytest.raises(ValueError, match="Unsupported IC"):
        sars.loo_cv(sars.load_galap(), ic="aic")
    with pytest.raises(ValueError, match="No column 'nope'"):
        sars.loo_cv(sars.load_galap(), label="nope")
