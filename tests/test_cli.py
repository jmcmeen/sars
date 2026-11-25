"""The ``sars`` command line."""

import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import sars
from sars.cli import DATASETS, main


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_json(capsys, *argv):
    code, out, _ = run(capsys, *argv, "--json")

    def refuse(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    return code, json.loads(out, parse_constant=refuse)


@pytest.fixture
def galap_csv(tmp_path):
    path = tmp_path / "galap.csv"
    sars.load_galap().rename(columns={"area": "Area", "species": "Richness"}).to_csv(
        path, index=False
    )
    return path


def test_fit_one_model(capsys):
    code, envelope = run_json(capsys, "fit", "galap")
    assert code == 0 and envelope["ok"] and envelope["command"] == "fit"
    assert envelope["sars"] == sars.__version__
    summary = envelope["result"]["summary"]
    expected = sars.sar_power(sars.load_galap())
    assert summary["model"] == "power"
    assert summary["params"]["z"] == pytest.approx(expected.params["z"])
    assert summary["AICc"] == pytest.approx(expected.aicc)
    again = sars.from_json(json.dumps(envelope["result"]["object"]))
    assert again.params == pytest.approx(expected.params)


def test_fit_several_models_ranks_them(capsys):
    code, envelope = run_json(capsys, "fit", "galap", "-m", "power,loga", "-m", "negexpo")
    summary = envelope["result"]["summary"]
    assert summary["type"] == "MultiSARFit"
    assert {row["model"] for row in summary["models"]} == {"power", "loga", "negexpo"}
    assert summary["best"] == summary["models"][0]["model"]


def test_a_csv_with_other_column_names(capsys, galap_csv):
    code, out, err = run(
        capsys, "fit", str(galap_csv), "--area-col", "Area", "--species-col", "Richness"
    )
    assert code == 0, err
    assert out.startswith("power: c = 33.18")


def test_standard_input(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(sars.load_xu().to_csv(index=False)))
    code, envelope = run_json(capsys, "fit", "-")
    assert code == 0 and envelope["result"]["summary"]["n"] == 42


def test_average_weights_follow_the_criterion(capsys):
    code, envelope = run_json(capsys, "average", "galap", "-m", "power,koba,p1", "--ic", "BIC")
    summary = envelope["result"]["summary"]
    expected = sars.sar_average(sars.load_galap(), models=["power", "koba", "p1"], ic="BIC")
    assert summary["ic"] == "BIC"
    assert summary["weights"] == pytest.approx(expected.weights)
    table = summary["models"]
    assert [row["weight"] for row in table] == pytest.approx(
        [expected.weights[row["model"]] for row in table]
    )
    assert table[0]["delta_BIC"] == 0
    assert [row["BIC"] for row in table] == sorted(row["BIC"] for row in table)


def test_threshold(capsys):
    code, envelope = run_json(capsys, "threshold", "galap", "-m", "ContOne,Linear")
    summary = envelope["result"]["summary"]
    expected = sars.sar_threshold(sars.load_galap(), models=["ContOne", "Linear"])
    assert summary["best"] == expected.best_model
    assert {row["model"] for row in summary["models"]} == {"ContOne", "Linear"}
    code, out, _ = run(capsys, "threshold", "galap", "-m", "ContOne,Linear")
    assert "None" not in out and "nan" not in out


def test_undefined_numbers_are_null(capsys, tmp_path):
    """Four islands: AICc is infinite, and the envelope stays standard JSON."""
    path = tmp_path / "tiny.csv"
    sars.load_galap().head(4).to_csv(path, index=False)
    code, envelope = run_json(capsys, "fit", str(path))
    assert code == 0 and envelope["result"]["summary"]["AICc"] is None
    code, out, _ = run(capsys, "fit", str(path))
    assert "AICc = undefined" in out


@pytest.mark.parametrize(
    ("command", "fit"),
    [
        (["fit", "galap"], lambda g: sars.sar_power(g)),
        (
            ["average", "galap", "-m", "power,loga"],
            lambda g: sars.sar_average(g, ["power", "loga"]),
        ),
        (["threshold", "galap"], lambda g: sars.sar_threshold(g)),
    ],
    ids=["fit", "average", "threshold"],
)
def test_save_then_predict(capsys, tmp_path, command, fit):
    """The pipeline: one command saves a result, `predict` reads it back."""
    path = tmp_path / "result.json"
    code, _, err = run(capsys, *command, "--save", str(path))
    assert code == 0 and "saved to" in err
    code, envelope = run_json(capsys, "predict", str(path), "1", "100")
    predictions = [row["species"] for row in envelope["result"]["predictions"]]
    expected = fit(sars.load_galap()).predict(np.array([1.0, 100.0]))
    assert predictions == pytest.approx(expected)
    code, out, _ = run(capsys, "predict", str(path), "1", "100")
    assert out.splitlines()[0].split("\t")[0] == "1"


def test_predict_refuses_a_result_that_cannot_predict(capsys, tmp_path):
    path = tmp_path / "multi.json"
    run(capsys, "fit", "galap", "-m", "power,loga", "--save", str(path))
    code, out, err = run(capsys, "predict", str(path), "10")
    assert code == 1 and "doesn't predict" in err


def test_datasets(capsys):
    code, envelope = run_json(capsys, "datasets")
    assert [row["name"] for row in envelope["result"]["datasets"]] == list(DATASETS)
    code, out, _ = run(capsys, "datasets", "galap")
    assert out.splitlines()[0] == ",".join(map(str, sars.load_galap().columns))
    assert len(out.splitlines()) == len(sars.load_galap()) + 1


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["fit", "nosuch.csv"], "No such file"),
        (["fit", "galap", "-m", "nope"], "Unknown model"),
        (["fit", "galap", "--species-col", "richness"], "No column 'richness'"),
        (["average", "galap", "-m", "nope"], "Unknown model"),
        (["threshold", "galap", "-m", "Nope"], "Nope"),
        (["predict", "nosuch.json", "1"], "No such file"),
        (["datasets", "nope"], "No bundled dataset"),
    ],
)
def test_errors_exit_1_with_a_message(capsys, argv, message):
    code, out, err = run(capsys, *argv)
    assert code == 1 and message in err and "Traceback" not in err
    code, envelope = run_json(capsys, *argv)
    assert code == 1 and not envelope["ok"] and message in envelope["error"]["message"]


def test_warnings_go_to_the_envelope_and_stderr(capsys):
    # The straight line's residuals on galap fail the Shapiro-Wilk test.
    argv = ["fit", "galap", "-m", "power,linear", "--normality", "shapiro"]
    code, envelope = run_json(capsys, *argv)
    assert code == 0 and "linear: residuals not normal" in envelope["warnings"][0]
    assert envelope["result"]["summary"]["excluded"].keys() == {"linear"}
    code, _, err = run(capsys, *argv)
    assert "warning: 1 of 2 fitted models failed" in err


def test_python_dash_m_and_version():
    out = subprocess.run(
        [sys.executable, "-m", "sars", "--version"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == f"sars {sars.__version__}"


def test_usage_errors_exit_2_with_an_envelope_under_json(capsys):
    code, out, err = run(capsys, "average", "galap", "--ic", "aic", "--json")
    envelope = json.loads(out)
    assert code == 2 and not envelope["ok"] and envelope["command"] == "average"
    assert (
        envelope["error"]["type"] == "UsageError"
        and "invalid choice" in envelope["error"]["message"]
    )
    code, out, err = run(capsys, "average", "galap", "--ic", "aic")
    assert code == 2 and out == "" and "invalid choice" in err


def test_checks_apply_to_a_single_model(capsys):
    code, envelope = run_json(capsys, "fit", "galap", "-m", "linear", "--normality", "shapiro")
    assert code == 1 and "linear: residuals not normal" in envelope["error"]["message"]
    code, envelope = run_json(capsys, "fit", "galap", "--neg-check")
    assert code == 0 and envelope["result"]["summary"]["type"] == "MultiSARFit"


def test_average_all(capsys):
    code, envelope = run_json(capsys, "average", "galap", "-m", "all")
    assert code == 0 and len(envelope["result"]["summary"]["weights"]) > 10


@pytest.mark.parametrize("area", ["nan", "inf"])
def test_predict_refuses_non_finite_areas(capsys, tmp_path, area):
    path = tmp_path / "fit.json"
    run(capsys, "fit", "galap", "--save", str(path))
    code, envelope = run_json(capsys, "predict", str(path), area)
    assert code == 1 and envelope["error"]["message"] == "Areas must be finite numbers."


def test_empty_model_option_is_an_error(capsys):
    code, out, err = run(capsys, "fit", "galap", "-m", " , ")
    assert code == 1 and "No model names" in err


def test_unreadable_standard_input(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code, out, err = run(capsys, "fit", "-")
    assert code == 1 and "Couldn't read standard input as CSV" in err


def test_a_directory_does_not_hide_a_bundled_dataset(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "galap").mkdir()
    code, out, err = run(capsys, "fit", "galap")
    assert code == 0 and out.startswith("power: c = 33.18")


def test_unexpected_errors_are_an_envelope_under_json(capsys, monkeypatch):
    def broken(data):
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(sars, "sar_power", broken)
    code, envelope = run_json(capsys, "fit", "galap")
    assert code == 1 and envelope["error"] == {"type": "ZeroDivisionError", "message": "boom"}
    with pytest.raises(ZeroDivisionError):  # without --json, a bug keeps its traceback
        main(["fit", "galap"])


def test_cv(capsys):
    code, envelope = run_json(capsys, "cv", "galap", "-m", "power,loga")
    summary = envelope["result"]["summary"]
    assert code == 0 and summary["type"] == "LOOCV"
    assert [row["cv_rank"] for row in summary["models"]] == [1, 2, 3]
    assert isinstance(summary["models"][0]["n_failed"], int)
    code, out, _ = run(capsys, "cv", "galap", "-m", "power,loga", "--warm")
    assert code == 0 and out.startswith("Leave-one-island-out: best")
    code, envelope = run_json(capsys, "cv", "gala", "-m", "power", "--ic", "AIC")
    rows = {row["model"]: row for row in envelope["result"]["summary"]["models"]}
    assert rows["power"]["worst"] == "Isabela" and "AIC" in rows["power"]


def test_loss(capsys):
    argv = [
        "loss",
        "gala",
        "--area",
        "900",
        "--fraction-lost",
        "0.5",
        "-m",
        "power",
        "--n-boot",
        "30",
        "--seed",
        "1",
    ]
    code, envelope = run_json(capsys, *argv)
    row = envelope["result"]["summary"]["scenarios"][0]
    expected = sars.habitat_loss(
        sars.load_gala(), area=900, fraction_lost=0.5, models=["power"], n_boot=0
    )
    assert code == 0 and row["species_lost"] == pytest.approx(
        expected.table["species_lost"].iloc[0]
    )
    assert row["lost_lower"] < row["species_lost"] < row["lost_upper"]
    code, out, _ = run(capsys, *argv)
    assert out.startswith("Species lost, model average by AICc")
    code, out, err = run(
        capsys, "loss", "gala", "--area", "900", "--remaining", "450", "--fraction-lost", "0.5"
    )
    assert code == 2 and "not allowed with" in err


def test_the_backend_is_chosen_and_recorded(capsys, monkeypatch):
    code, envelope = run_json(capsys, "fit", "galap")
    assert code == 0 and envelope["backend"] == "numpy"
    chosen = []

    def refuse(name):
        chosen.append(name)
        raise ImportError("no GPU here")

    monkeypatch.setattr(sars, "set_backend", refuse)
    code, envelope = run_json(capsys, "--backend", "cupy", "fit", "galap")
    assert code == 1 and chosen == ["cupy"] and envelope["error"]["message"] == "no GPU here"
    monkeypatch.setenv("SARS_BACKEND", "cupy")
    code, out, err = run(capsys, "fit", "galap")
    assert code == 1 and "no GPU here" in err


def test_threshold_ci(capsys):
    argv = ["threshold", "gala", "--ci", "--n-boot", "40", "--seed", "1"]
    code, envelope = run_json(capsys, *argv)
    ci = envelope["result"]["summary"]["ci"]
    fit = sars.sar_threshold(sars.load_gala())
    expected = sars.threshold_ci(fit, n_boot=40, rng=np.random.default_rng(1))
    assert code == 0 and ci["model"] == fit.best_model == "ZslopeOne"
    row = ci["breakpoints"][0]
    assert row["lower_area"] == pytest.approx(expected.lower_area[0])
    assert row["lower_area"] < row["estimate_area"] < row["upper_area"]
    code, out, _ = run(capsys, "threshold", "gala", "--ci", "--ci-method", "F")
    assert code == 0 and "ZslopeOne breakpoints, 95% intervals (inverted F test)" in out
    code, envelope = run_json(capsys, "threshold", "galap", "--ci", "--n-boot", "10")
    assert code == 1 and "Linear, has no breakpoint" in envelope["error"]["message"]
    code, envelope = run_json(capsys, "threshold", "gala", "--ci", "--ci-model", "Linear")
    assert code == 1 and envelope["error"]["message"].startswith("Linear has no breakpoint")


def test_fit_params(capsys):
    code, envelope = run_json(capsys, "fit", "galap", "-m", "negexpo,power", "--params")
    tables = envelope["result"]["summary"]["parameters"]
    expected = sars.sar_negexpo(sars.load_galap()).param_table()
    assert code == 0 and set(tables) == {"negexpo", "power"}
    rows = {row["parameter"]: row for row in tables["negexpo"]}
    assert rows["d"]["std_error"] == pytest.approx(expected.loc["d", "std_error"])
    code, out, _ = run(capsys, "fit", "galap", "--params", "--conf", "0.9")
    assert code == 0 and "power: parameters, 90% intervals" in out and "ci_lower" in out
    code, envelope = run_json(capsys, "fit", "galap", "--params", "--conf", "1.5")
    assert code == 1 and "between 0 and 1" in envelope["error"]["message"]
    code, envelope = run_json(capsys, "fit", "galap", "--conf", "0.9")
    assert code == 1 and "give --params too" in envelope["error"]["message"]


def test_parameters_the_data_dont_identify_show_nan(capsys, monkeypatch):
    from sars import _operations

    def unidentified(obj, conf=0.95):
        row = {"parameter": "z", "estimate": 0.3, "std_error": None, "t_value": None}
        return {"power": [{**row, "p_value": None, "ci_lower": None, "ci_upper": None}]}

    monkeypatch.setattr(_operations, "parameters", unidentified)
    code, out, _ = run(capsys, "fit", "galap", "--params")
    assert code == 0 and "NaN" in out.split("parameters, 95% intervals")[1]


def test_a_bad_level_saves_nothing(capsys, tmp_path):
    saved = tmp_path / "fit.json"
    code, _ = run_json(capsys, "fit", "galap", "--params", "--conf", "95", "--save", str(saved))
    assert code == 1 and not saved.exists()


def test_many(capsys, tmp_path):
    code, envelope = run_json(capsys, "many", "galap", "gala", "-m", "power,loga", "--no-threshold")
    rows = envelope["result"]["summary"]["datasets"]
    assert code == 0 and [r["dataset"] for r in rows] == ["galap", "gala"]
    expected = sars.sar_many(
        {"galap": sars.load_galap(), "gala": sars.load_gala()},
        models=["power", "loga"],
        threshold=False,
    )
    assert rows[0]["z"] == pytest.approx(expected.summary["z"].iloc[0])
    long = pd.concat(
        [sars.load_galap().assign(group="a"), sars.load_gala().assign(group="b")]
    ).rename(columns={"area": "A", "species": "S"})
    path = tmp_path / "long.csv"
    long.to_csv(path, index=False)
    argv = ["many", str(path), "--by", "group", "--area-col", "A", "--species-col", "S"]
    code, envelope = run_json(capsys, *argv, "-m", "power", "--no-threshold")
    rows = envelope["result"]["summary"]["datasets"]
    assert code == 0 and [r["dataset"] for r in rows] == ["a", "b"]
    code, out, _ = run(capsys, *argv, "-m", "power", "--no-threshold")
    assert code == 0 and out.split()[0] == "dataset"
    code, envelope = run_json(capsys, "many", "galap")
    assert code == 1 and "two or more" in envelope["error"]["message"]
    # Files sharing a name each get one of their own, never another's.
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        sars.load_galap().to_csv(tmp_path / folder / "x.csv", index=False)
    sars.load_gala().to_csv(tmp_path / "x_2.csv", index=False)
    files = [
        str(tmp_path / "x_2.csv"),
        str(tmp_path / "a" / "x.csv"),
        str(tmp_path / "b" / "x.csv"),
    ]
    code, envelope = run_json(capsys, "many", *files, "-m", "power", "--no-threshold")
    names = [r["dataset"] for r in envelope["result"]["summary"]["datasets"]]
    assert code == 0 and len(set(names)) == 3
    code, envelope = run_json(capsys, "many", str(path), "--by", "nope")
    assert code == 1 and "No column 'nope'" in envelope["error"]["message"]


def test_loglog(capsys):
    code, envelope = run_json(capsys, "loglog", "galap", "--compare")
    summary = envelope["result"]["summary"]
    expected = sars.lin_pow(sars.load_galap())
    assert code == 0 and summary["type"] == "LinPowFit"
    assert summary["z"] == pytest.approx(expected.z) and summary["z_se"] == pytest.approx(
        expected.z_se
    )
    assert summary["power"]["params"]["z"] == pytest.approx(
        sars.sar_power(sars.load_galap()).params["z"]
    )
    code, out, _ = run(capsys, "loglog", "galap", "--log-base", "10")
    assert code == 0 and out.startswith("log10 S = log10 c + z log10 A")
    assert out.count("t = ") == 2  # the intercept's t test and the slope's


def test_loglog_on_an_exact_line_shows_an_infinite_t(capsys, tmp_path):
    path = tmp_path / "exact.csv"
    path.write_text("area,species\n1,10\n10,20\n100,40\n1000,80\n")
    code, out, _ = run(capsys, "loglog", str(path))
    assert code == 0 and "t = inf" in out


def test_ci(capsys):
    argv = ["ci", "galap", "--area", "1", "100", "-m", "power", "--n-boot", "40", "--seed", "2"]
    code, envelope = run_json(capsys, *argv)
    summary = envelope["result"]["summary"]
    expected = sars.bootstrap_ci(
        sars.load_galap(),
        models=["power"],
        n_boot=40,
        area_grid=np.array([1.0, 100.0]),
        rng=np.random.default_rng(2),
    )
    assert code == 0 and summary["type"] == "BootstrappedCI" and summary["n_boot"] == 40
    assert [row["lower"] for row in summary["intervals"]] == pytest.approx(list(expected.lower))
    assert [row["upper"] for row in summary["intervals"]] == pytest.approx(list(expected.upper))
    code, out, _ = run(capsys, *argv)
    assert code == 0 and out.startswith("Model-averaged richness, 95% bootstrap intervals")
    code, envelope = run_json(capsys, "ci", "galap", "-m", "power", "--n-boot", "10")
    assert len(envelope["result"]["summary"]["intervals"]) == 10  # the default areas
    code, envelope = run_json(capsys, "ci", "galap", "--area", "-1", "--n-boot", "10")
    assert code == 1 and "positive" in envelope["error"]["message"]
    for bad, message in [("--n-boot=0", "at least 1"), ("--conf=95", "between 0 and 1")]:
        code, envelope = run_json(capsys, "ci", "galap", bad)
        assert code == 1 and message in envelope["error"]["message"]


def test_a_bootstrap_too_big_to_hold_is_refused(capsys, monkeypatch):
    from sars import _operations

    monkeypatch.setattr(_operations, "MAX_RESAMPLE_VALUES", 1000)
    code, envelope = run_json(capsys, "ci", "galap", "--n-boot", "100")
    assert code == 1 and "fewer resamples" in envelope["error"]["message"]
    code, envelope = run_json(
        capsys, "loss", "galap", "--area", "100", "--fraction-lost", "0.5", "--n-boot", "100"
    )
    assert code == 1 and "fewer resamples" in envelope["error"]["message"]


def test_ci_checks_the_data_first(capsys, tmp_path):
    path = tmp_path / "zero.csv"
    path.write_text("area,species\n0,3\n2,5\n4,8\n")
    code, envelope = run_json(capsys, "ci", str(path), "--n-boot", "5")
    assert code == 1 and "positive" in envelope["error"]["message"]


def test_cv_labels_islands_by_a_named_column(capsys, tmp_path):
    path = tmp_path / "gala.csv"
    sars.load_gala().rename(columns={"island": "Site_Name"}).to_csv(path, index=False)
    code, envelope = run_json(
        capsys, "cv", str(path), "-m", "power", "--warm", "--label-col", "Site_Name"
    )
    rows = {row["model"]: row for row in envelope["result"]["summary"]["models"]}
    assert code == 0 and rows["power"]["worst"] == "Isabela"
