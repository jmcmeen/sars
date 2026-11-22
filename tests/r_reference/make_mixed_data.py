"""Write mixed_data.csv, the input of generate_r_reference.R's section 17.

Run from the repository root before that script: python tests/r_reference/make_mixed_data.py

The six archipelagos come from the package's bundled data, two of which (gala,
from R faraway, and schrader and xu, from Gooriah et al. 2021) aren't in R sars.
Schrader's has islands with no species, so, as lin_pow_mixed does, one is added
to every richness value (the con column).
"""

from pathlib import Path

import pandas as pd

import sars

DATASETS = {
    "gala": sars.load_gala(),
    "aegean": sars.load_aegean(),
    "aegean2": sars.load_aegean2(),
    "niering": sars.load_niering(),
    "schrader": sars.load_schrader(),
    "xu": sars.load_xu(),
}

if __name__ == "__main__":
    con = 1 if any((d["species"] == 0).any() for d in DATASETS.values()) else 0
    long = pd.concat(
        [d[["area", "species"]].assign(group=name, con=con) for name, d in DATASETS.items()],
        ignore_index=True,
    )
    long.to_csv(Path(__file__).parent / "mixed_data.csv", index=False)
