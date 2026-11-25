#!/usr/bin/env bash
# The log-log fit beside the nonlinear one, and how well each model predicts.
set -euo pipefail

# log S = log c + z log A by least squares, with the nonlinear power model beside it.
sars loglog galap --compare

# Leave each island out and predict it; --warm starts each fold from the full fit.
sars cv galap -m power,loga,negexpo,monod --warm

# Many datasets in one table: best model, log-log z, threshold model.
sars many galap gala xu -m power,loga,negexpo,monod
