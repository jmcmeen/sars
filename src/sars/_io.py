"""Data loading and I/O adapters."""

from __future__ import annotations


def load_galap():
    """Load the Galapagos plant species-area dataset.

    Returns the Preston (1962) 16-island dataset as shipped in the R sars
    package (Albemarle/Isabela excluded).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'area' and 'species'.

    Raises
    ------
    NotImplementedError
        Until the R reference data is generated and committed.
    """
    raise NotImplementedError(
        "Run tests/r_reference/generate_r_reference.R first, "
        "then this function will read from galap.csv"
    )
