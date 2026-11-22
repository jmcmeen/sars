"""sar_many: the same models fitted to many datasets, one summary row each."""

import numpy as np
import pandas as pd
import pytest

import sars

FEW = ["power", "loga", "negexpo", "monod", "p1"]


@pytest.fixture(scope="module")
def two():
    return {"galap": sars.load_galap(), "xu": sars.load_xu()}


@pytest.fixture(scope="module")
def many(two):
    return sars.sar_many(two, models=FEW)


def test_one_row_per_dataset_matching_the_fits_on_their_own(two, many):
    summary = many.summary
    assert summary["dataset"].tolist() == ["galap", "xu"]
    assert list(summary.columns) == [
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
    for name, data in two.items():
        row = summary.set_index("dataset").loc[name]
        log_log = sars.lin_pow(data)
        average = sars.sar_average(data, models=FEW)
        threshold = sars.sar_threshold(data)
        assert row["n"] == len(data)
        assert (row["area_min"], row["area_max"]) == (data["area"].min(), data["area"].max())
        assert row["z"] == pytest.approx(log_log.z)
        assert row["R2"] == pytest.approx(log_log.r_squared)
        best = max(average.weights, key=average.weights.get)
        assert row["best_model"] == best
        assert row["best_weight"] == pytest.approx(average.weights[best])
        assert row["threshold_model"] == threshold.best_model
        assert row["error"] == ""
    assert set(many.averages) == set(many.log_log) == set(many.thresholds) == {"galap", "xu"}
    assert repr(many) == "ManySARFit(2 datasets, 0 with errors)"


def test_a_long_frame_with_by_gives_the_same_summary(two, many):
    long = pd.concat([d.assign(site=name) for name, d in two.items()], ignore_index=True)
    grouped = sars.sar_many(long, by="site", models=FEW)
    pd.testing.assert_frame_equal(grouped.summary, many.summary)


def test_a_dataset_that_fails_doesnt_stop_the_others(two):
    empty = two["galap"].copy()
    empty.loc[0, "species"] = 0
    with pytest.warns(UserWarning, match=r"1 of 2 datasets had a fit that failed \(empty\)"):
        many = sars.sar_many({"empty": empty, "xu": two["xu"]}, models=FEW)
    row = many.summary.set_index("dataset").loc["empty"]
    # The average and the threshold models refuse an island with no species;
    # lin_pow, which adds a constant before taking logs, fits it.
    assert row["error"].startswith("sar_average: ValueError:") and "positive" in row["error"]
    assert "sar_threshold: ValueError:" in row["error"]
    assert row["best_model"] == "" and np.isfinite(row["z"])
    assert set(many.averages) == {"xu"} and set(many.log_log) == {"empty", "xu"}
    assert repr(many) == "ManySARFit(2 datasets, 1 with errors)"


def test_warnings_are_listed_against_their_dataset(two):
    small = two["galap"].nsmallest(6, "area")  # six islands: the threshold models warn
    many = sars.sar_many({"small": small}, models=FEW)
    listed = many.summary.loc[0, "warnings"]
    assert any(m.startswith("UserWarning: ") and "observations" in m for m in listed)


def test_threshold_false_or_a_list(two):
    off = sars.sar_many(two, models=FEW, threshold=False)
    assert off.thresholds == {}
    assert (off.summary["threshold_model"] == "").all()
    assert off.summary["breakpoint_area"].isna().all()
    chosen = sars.sar_many({"xu": two["xu"]}, models=FEW, threshold=["ZslopeOne", "Linear"])
    assert chosen.summary.loc[0, "threshold_model"] in ("ZslopeOne", "Linear")


def test_best_model_follows_the_ic(two):
    bic = sars.sar_many({"xu": two["xu"]}, models=FEW, ic="BIC", threshold=False)
    average = sars.sar_average(two["xu"], models=FEW, ic="BIC")
    assert bic.summary.loc[0, "best_model"] == max(average.weights, key=average.weights.get)


def test_the_same_in_parallel(two, many):
    parallel = sars.sar_many(two, models=FEW, n_jobs=2)
    pd.testing.assert_frame_equal(parallel.summary, many.summary)


@pytest.mark.parametrize(
    ("args", "error", "match"),
    [
        (({},), ValueError, "No datasets"),
        ((pd.DataFrame({"area": [1.0], "species": [1]}),), ValueError, "Pass by="),
        ((pd.DataFrame({"area": [1.0], "species": [1]}), "site"), ValueError, "not a column"),
        (({"a": sars.load_galap()}, "site"), ValueError, "by= is for a single DataFrame"),
        (({1: sars.load_galap(), "1": sars.load_galap()},), ValueError, "unique"),
        (([sars.load_galap()],), TypeError, "mapping of name to DataFrame"),
    ],
)
def test_bad_arguments(args, error, match):
    with pytest.raises(error, match=match):
        sars.sar_many(*args)


def test_rows_without_a_dataset_name_are_refused():
    long = pd.DataFrame({"area": [1.0, 2.0], "species": [1, 2], "site": ["a", None]})
    with pytest.raises(ValueError, match="no dataset name"):
        sars.sar_many(long, by="site")


def test_bad_ic_and_n_jobs_fail_before_fitting(two):
    with pytest.raises(ValueError, match="Unsupported IC"):
        sars.sar_many(two, ic="WAIC")
    with pytest.raises(ValueError, match="n_jobs"):
        sars.sar_many(two, n_jobs=0)


def test_a_failed_threshold_keeps_the_other_fits(two):
    """Four islands: enough for the average and the log-log fit, not for the
    threshold models."""
    tiny = two["galap"].nsmallest(4, "area")
    with pytest.warns(UserWarning, match="had a fit that failed"):
        many = sars.sar_many({"tiny": tiny}, models=["power", "loga"], ic="AIC")
    row = many.summary.loc[0]
    assert row["error"].startswith("sar_threshold: ")
    assert row["best_model"] in ("power", "loga") and np.isfinite(row["z"])
    assert set(many.averages) == set(many.log_log) == {"tiny"} and many.thresholds == {}


def test_not_fitted_keeps_the_reason(two, monkeypatch):
    import dataclasses

    from sars import _inference

    real = _inference._fit_nls

    def fit(name, data, **kw):
        result = real(name, data, **kw)
        return dataclasses.replace(result, converged=False) if name == "loga" else result

    monkeypatch.setattr(_inference, "_fit_nls", fit)
    many = sars.sar_many({"xu": two["xu"]}, models=["power", "loga"], threshold=False)
    assert many.summary.loc[0, "not_fitted"] == ["loga: did not converge"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"models": ["nope"]}, "Unknown model"),
        ({"models": "power"}, 'models must be "all"'),
        ({"threshold": "ContOne"}, r"pass \['ContOne'\]"),
        ({"threshold": ["ContOne", "Nope"]}, "Unknown threshold model"),
    ],
)
def test_unknown_models_fail_before_fitting(two, kwargs, match):
    with pytest.raises(ValueError, match=match):
        sars.sar_many(two, **kwargs)


def test_every_dataset_must_be_a_dataframe(two):
    with pytest.raises(TypeError, match=r"must be a DataFrame; not for \['bad'\]"):
        sars.sar_many({"xu": two["xu"], "bad": None})


def test_unused_categories_are_not_datasets(two):
    long = two["xu"].assign(site=pd.Categorical(["a"] * len(two["xu"]), categories=["a", "b"]))
    many = sars.sar_many(long, by="site", models=FEW, threshold=False)
    assert many.summary["dataset"].tolist() == ["a"]


def test_repr_of_an_empty_result():
    assert repr(sars.ManySARFit(summary=pd.DataFrame())) == "ManySARFit(0 datasets, 0 with errors)"
