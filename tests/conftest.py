import pandas as pd
import pytest
from pathlib import Path

REF = Path("tests/r_reference")


@pytest.fixture(scope="session")
def galap():
    path = REF / "galap.csv"
    if not path.exists():
        pytest.skip("Run generate_r_reference.R first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def r_reference():
    path = REF / "all_models_galap.csv"
    if not path.exists():
        pytest.skip("Run generate_r_reference.R first")
    return pd.read_csv(path).set_index("model")
