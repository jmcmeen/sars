#!/usr/bin/env bash
# Average the models, then put bootstrap intervals on the averaged curve.
set -euo pipefail

# The model average, weighted by AICc.
sars average galap

# Leave out models whose residuals fail a normality test before averaging.
sars average galap --normality shapiro

# The averaged richness at chosen areas (km²), with 95% bootstrap intervals.
# --seed makes the intervals reproducible; use more resamples in real work.
sars ci galap --area 1 10 100 1000 --n-boot 100 --seed 1

# The full method searches every model's start grid on each resample, as R does.
sars ci galap --area 10 100 --method full --n-boot 20 --seed 1 -m power,negexpo,monod
