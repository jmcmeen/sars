#!/usr/bin/env bash
# Fit one model, then all 20, and rank them by AICc.
set -euo pipefail

# The power model, S = c A^z, on the bundled Galápagos plants (Preston 1962).
sars fit galap

# All 20 models, ranked by AICc with Akaike weights.
sars fit galap -m all

# The leaders' parameters, with standard errors and 95% intervals.
sars fit galap -m negexpo,monod,koba --params

# Your own data: a CSV with columns named however you like.
tmp=$(mktemp -d)
sars datasets xu > "$tmp/xu.csv"
sars fit "$tmp/xu.csv" -m power,loga --area-col area --species-col species
rm -r "$tmp"
