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


@pytest.mark.slow
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
    assert list(fit.groups["group"]) == list(data)
    # Pooled toward the group: no archipelago's z is further out than its own fit's.
    spread_alone = np.std(fit.groups["z_alone"])
    assert np.std(fit.groups["z"]) <= spread_alone
    pred = fit.predict([1.0, 100.0], group="arch0")
    assert pred.shape == (2,) and pred[1] > pred[0] > 0
    with pytest.raises(ValueError, match="No archipelago"):
        fit.predict(1.0, group="nowhere")
