#!/usr/bin/env bash
# Every command can print a JSON envelope for scripts: {sars, command, ok, result, ...}.
# This one needs jq (https://jqlang.github.io/jq/).
set -euo pipefail

# The best model's name and weight.
sars fit galap -m all --json | jq -r '.result.summary.models[0] | "\(.model) \(.weight)"'

# z and its standard error from the log-log fit.
sars loglog galap --json | jq '.result.summary | {z, z_se, z_p}'

# Save a fit, then predict from it later.
tmp=$(mktemp -d)
sars average galap --save "$tmp/average.json" > /dev/null
sars predict "$tmp/average.json" 5 50 500 --json | jq -c '.result.predictions[]'
rm -r "$tmp"
