#!/usr/bin/env bash
# Species expected to be lost with habitat, from the model average.
# An eventual loss, and likely an overestimate: see the documentation.
set -euo pipefail

# An area of 900 km² losing half, then three-quarters, of its habitat.
sars loss gala --area 900 --fraction-lost 0.5 0.75 --n-boot 200 --seed 1

# The same, given the area left instead.
sars loss gala --area 900 --remaining 450 225 --n-boot 200 --seed 1
