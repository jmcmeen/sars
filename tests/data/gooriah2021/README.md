# Gooriah et al. (2021) island datasets

Species-area tables for the 34 island and habitat-island datasets compiled by

> Gooriah L, Blowes SA, Sagouis A, Schrader J, Karger DN, Kreft H, Chase JM (2021).
> Synthesis reveals that island species-area relationships emerge from processes
> beyond passive sampling. *Global Ecology and Biogeography*, 30, 2119-2131.

from their data deposit, Zenodo record 7311286, released under CC0.
`tests/test_gooriah.py` fits them all, as a check of the package on data beyond the
bundled datasets.

Each `<study>.csv` has `island` (the deposit's island name, or its code where it has no
name), `area` (km², the deposit's hectares divided by 100) and `species` (the number of
species recorded). Islands with no species recorded are kept. `build_tables.py` rebuilds
the tables from the deposit, matching the species matrices' island names to the
environment file's codes ignoring case.

Islands the deposit gives no area for are left out, and `build_tables.py` lists them:
17 in Evans 2016 and 5 in Rangasamy frogs 2014 (in the environment file, with no area),
and Pierce 2003's Gambier Is, Usher 1986's Byers Peninsula, Dion Island and Faure
Islands, and Zagar 2013's Kamenovo, Peljesac, Lake Skadar and Dojran Lake (not in the
environment file at all). In Werner Zalewski 2006 the matrix's "ful" and "guc" have no
entry, while the environment file has areas for "muszla" and "olchowa", which the matrix
doesn't have; they may be the same islands under other names, but without the source to
confirm it all four are left out.

Richness here counts species recorded in sampled assemblages, not complete island lists,
and the deposit marks sampling effort as not standardised between islands in 27 of the
34 studies. Anyone using these
data should cite the original study as well as the synthesis: `REFERENCES.txt` is the
deposit's list of the original studies.

Two of the datasets also ship with the package: `load_schrader()` (Schrader_2019) and
`load_xu()` (Xu_2017).
