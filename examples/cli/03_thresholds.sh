#!/usr/bin/env bash
# Look for a small-island effect: a breakpoint below which area matters less.
set -euo pipefail

# On 30 Galápagos islands (Johnson & Raven 1973) a flat-then-rising model wins.
sars threshold gala

# With intervals for the breakpoint, by bootstrap and by inverted F test.
sars threshold gala --ci --n-boot 200 --seed 1
sars threshold gala --ci --ci-method F

# All six threshold models, against a straight line.
sars threshold gala -m ContOne,ZslopeOne,DiscOne,ContTwo,ZslopeTwo,DiscTwo,Linear
