#!/usr/bin/env bash
# The analyses that read more than area and richness: habitats, habitat
# areas, island age, and counts of individuals.
set -euo pipefail

# Habitat models: a number of habitats per island beside area and richness.
sars habitat habitat

# The countryside model: an area column per habitat, a richness column per
# species group. The bundled table's columns are used when none are named.
sars countryside countryside --n-starts 10

# The general dynamic model: a column of island age. No table is bundled, so
# this one is made up for the example (the ages are not real island ages).
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cat > "$work/islands.csv" <<'CSV'
area,species,age_myr
12,31,0.4
85,74,1.1
240,118,2.0
31,82,2.9
640,161,3.6
150,139,4.5
9,66,5.3
410,150,6.2
58,88,7.4
22,57,8.8
310,96,10.1
14,29,11.6
CSV
sars gdm "$work/islands.csv" --time-col age_myr --mod-sel
sars gdm "$work/islands.csv" --time-col age_myr -m all --save "$work/gdm.json" > /dev/null
sars predict "$work/gdm.json" 50 500 --time 3

# Counts of individuals, species by site: random placement, with a
# simulation test. The bundled table takes its site areas from `xu`.
sars coleman xu_abundance --seed 1

# Richness standardized for sampling effort is written as CSV, so it goes
# straight into a fit.
sars effort xu_abundance | sars fit - -m power,loga
