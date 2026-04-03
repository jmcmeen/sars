import pandas as pd
import pytest

from sars import sar_power


def test_power_law_spot_check():
    """Quick sanity check using confirmed R values (no R script needed)."""
    partial = pd.DataFrame({
        "area":    [0.20, 0.90, 1.00, 1.80, 1.87, 4.40, 7.10,
                    7.50, 18.00, 20.00],
        "species": [  48,    7,   52,   14,   42,   22,  103,
                        48,    79,   119],
    })
    fit = sar_power(partial)
    assert fit.converged is True
    assert 0.0 < fit.params["z"] < 1.0   # z must be in plausible SAR range
    assert fit.params["c"] > 0
    assert 0.0 < fit.r_squared <= 1.0


def test_power_law_matches_r(galap, r_reference):
    """Full validation: must match R sars::sar_power(galap) exactly."""
    fit = sar_power(galap)
    ref = r_reference.loc["power"]
    assert fit.r_squared == pytest.approx(ref["r2"], abs=0.005)
    assert fit.aicc == pytest.approx(ref["aicc"], abs=0.1)
    assert fit.params["c"] == pytest.approx(33.1792, abs=0.05)
    assert fit.params["z"] == pytest.approx(0.2832, abs=0.005)
