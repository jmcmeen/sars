"""Data loading and I/O adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_galap() -> pd.DataFrame:
    """Load the Galapagos plant species-area dataset.

    Returns the Preston (1962) 16-island dataset as shipped in the R sars
    package (Albemarle/Isabela excluded).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.
    """
    path = Path(__file__).resolve().parents[2] / "tests" / "r_reference" / "galap.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"galap.csv not found at {path}. "
            "Run tests/r_reference/generate_r_reference.R first."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={"a": "area", "s": "species"})
    return df
