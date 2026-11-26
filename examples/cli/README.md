# Command-line examples

Short, commented workflows with the `sars` command, each run on the bundled datasets so
they work as they are. Install sars (`pip install sars`) and run any of them with `bash`:

| Script | Shows |
|---|---|
| `01_fit_and_rank.sh` | One model, all 20 ranked by AICc, parameter tables, your own CSV |
| `02_average_with_intervals.sh` | The model average, residual checks, bootstrap intervals (fast and full) |
| `03_thresholds.sh` | Threshold models and breakpoint intervals (the small-island effect) |
| `04_habitat_loss.sh` | Species expected to be lost with habitat |
| `05_compare_and_validate.sh` | Log-log against nonlinear power, cross-validation, many datasets |
| `06_json_and_jq.sh` | JSON output for scripts, saving and predicting (needs `jq`) |
| `07_other_tables.sh` | Habitat and countryside models, the general dynamic model, random placement and richness standardized for effort |

`sars --help` and `sars COMMAND --help` list every option, and the documentation's
[Command line](https://jmcmeen.github.io/sars/cli/) page describes them. The tests run
each script, so they stay in step with the command.
