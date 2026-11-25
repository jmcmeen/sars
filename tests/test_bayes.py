"""The Bayesian hierarchical log-log model (the bayes extra)."""

import numpy as np
import pandas as pd
import pytest

import sars


def archipelagos(n_groups=8, seed=3):
    """Archipelagos from one log-log model, each with its own c and z."""
    rng = np.random.default_rng(seed)
    out = {}
    for j in range(n_groups):
        area = np.sort(10 ** rng.uniform(-1, 3, 12 + j))
        logc = 2.0 + rng.normal(0, 0.4)
        z = 0.3 + rng.normal(0, 0.06)
        species = np.round(np.exp(logc + z * np.log(area) + rng.normal(0, 0.15, len(area))))
        out[f"arch{j}"] = pd.DataFrame({"area": area, "species": np.maximum(species, 1)})
    return out


def test_a_missing_pymc_is_named(monkeypatch):
    from sars import _extras

    monkeypatch.setitem(_extras.EXTRAS, "bayes", ("no_such_module_for_sars",))
    with pytest.raises(ImportError, match=r"sars\[bayes\]"):
        sars.lin_pow_bayes(archipelagos())


def test_the_inputs_are_checked_as_the_mixed_model_checks_them():
    pytest.importorskip("pymc")
    with pytest.raises(ValueError, match="at least 3 archipelagos"):
        sars.lin_pow_bayes(dict(list(archipelagos().items())[:2]))
    with pytest.raises(ValueError, match="'conf' must be between 0 and 1"):
        sars.lin_pow_bayes(archipelagos(), conf=1.5)
    with pytest.raises(ValueError, match="at least 2"):
        sars.lin_pow_bayes(archipelagos(), chains=1)


@pytest.mark.bayes
def test_the_posterior_agrees_with_reml():
    """With weak priors and enough data, the posterior means of the group's z
    and of its spread are near lin_pow_mixed's REML estimates."""
    pytest.importorskip("pymc")
    data = archipelagos()
    reml = sars.lin_pow_mixed(data)
    fit = sars.lin_pow_bayes(data, draws=600, tune=600, chains=2, seed=1)
    z = fit.summary.loc["z"]
    assert z["mean"] == pytest.approx(reml.z, abs=2 * reml.z_se)
    assert z["lower"] < reml.z < z["upper"]
    assert fit.summary.loc["sigma", "mean"] == pytest.approx(reml.sigma, rel=0.25)
    assert fit.converged, fit.summary
    assert list(fit.groups.index) == list(data)
    assert np.allclose(fit.groups["c"], np.exp(fit.groups["logc"]))
    # Pooled toward the group: the archipelagos' slopes spread less than their own fits'.
    assert np.std(fit.groups["z"]) <= np.std(fit.groups["z_alone"])
    pred = fit.predict([1.0, 100.0], group="arch0")
    assert pred.shape == (2,) and pred[1] > pred[0] > 0
    assert isinstance(fit.predict(10.0), float)
    with pytest.raises(ValueError, match="No archipelago"):
        fit.predict(1.0, group="nowhere")
    with pytest.raises(ValueError, match="positive"):
        fit.predict([0.0, 5.0])


@pytest.mark.bayes
def test_the_units_of_area_dont_change_the_fit():
    """Log area is centered in the model, so areas in m² rather than km² give
    the same slopes and spread (and an intercept moved by z log(1e6))."""
    pytest.importorskip("pymc")
    km2 = archipelagos()
    m2 = {name: d.assign(area=d["area"] * 1e6) for name, d in km2.items()}
    a = sars.lin_pow_bayes(km2, draws=500, tune=500, chains=2, seed=2)
    b = sars.lin_pow_bayes(m2, draws=500, tune=500, chains=2, seed=2)
    for name in ("z", "sd_z", "sigma"):
        assert b.summary.loc[name, "mean"] == pytest.approx(a.summary.loc[name, "mean"], rel=0.1)
    shift = a.summary.loc["z", "mean"] * np.log(1e6)
    assert b.summary.loc["logc", "mean"] == pytest.approx(
        a.summary.loc["logc", "mean"] - shift, abs=0.2
    )
