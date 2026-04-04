from pathlib import Path

import pandas as pd
import pytest

import sars

REF = Path("tests/r_reference")


@pytest.fixture(scope="session")
def galap():
    path = REF / "galap.csv"
    if not path.exists():
        pytest.skip("Run generate_r_reference.R first")
    return pd.read_csv(path).rename(columns={"a": "area", "s": "species"})


@pytest.fixture(scope="session")
def r_reference():
    path = REF / "all_models_galap.csv"
    if not path.exists():
        pytest.skip("Run generate_r_reference.R first")
    return pd.read_csv(path).set_index("model")


@pytest.fixture(scope="session")
def galap_multi(galap):
    """Fit all 20 models once and share across tests."""
    return sars.sar_multi(galap)


@pytest.fixture(scope="session")
def galap_fits(galap_multi):
    """Dict of model_name -> SARFit from the shared multi-model run."""
    return {fit.model: fit for fit in galap_multi.fits}


@pytest.fixture(scope="session")
def galap_average(galap):
    """Shared sar_average result for the galap dataset."""
    return sars.sar_average(galap)
