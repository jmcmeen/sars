"""Rebuild this directory's tables from the Gooriah et al. (2021) deposit.

Download data.zip from https://zenodo.org/records/7311286 (CC0), unzip it, and run::

    python build_tables.py <unzipped>/data

For each study it writes ``<study>.csv`` with one row per island that has an area:
the island's code or name, its area in km² (the deposit's hectares divided by 100,
in decimal arithmetic) and the number of species recorded on it (non-zero columns
of the site-by-species matrix; cells that aren't numbers count as absent).
Islands with no species recorded are kept. Matrix islands are matched to the
environment file's codes ignoring case and surrounding spaces; those with no match,
or no area, are left out and listed on stderr.
"""

import csv
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main(deposit: Path) -> None:
    # The environment file is labelled UTF-8 but is Latin-1.
    env = pd.read_csv(deposit / "env_file_UTF-8.csv", encoding="latin-1", dtype=str)
    for matrix in sorted((deposit / "ISAR_datasets").glob("*_data.csv")):
        study = matrix.name.removesuffix("_data.csv")
        m = pd.read_csv(matrix, encoding="latin-1")
        counts = m.drop(columns="island_name").apply(pd.to_numeric, errors="coerce")
        richness = (counts.fillna(0) > 0).sum(axis=1)
        e = env[env["Study"] == f"{study}_data"].copy()
        e["code"] = e["island_code"].astype(str).str.strip().str.casefold()
        areas = dict(zip(e["code"], e["island_area_ha"], strict=True))
        names = dict(zip(e["code"], e["island_name"], strict=True))
        rows, left_out = [], []
        for island, s in zip(m["island_name"].astype(str).str.strip(), richness, strict=True):
            code = island.casefold()
            ha = areas.get(code)
            if ha is None or pd.isna(ha):
                left_out.append(island)
                continue
            name = names[code] if isinstance(names[code], str) else island
            km2 = (Decimal(ha.strip()) / Decimal(100)).normalize()
            rows.append((name.strip(), format(km2, "f"), int(s)))
        with open(HERE / f"{study}.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["island", "area", "species"])
            w.writerows(rows)
        if left_out:
            print(f"{study}: no area for {', '.join(left_out)}", file=sys.stderr)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
