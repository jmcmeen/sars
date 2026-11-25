"""``python -m sars``: the command line (see `sars.cli`)."""

import sys

from sars.cli import main

# Guarded: a worker process (n_jobs, or the HTTP server's) started by spawn
# imports the parent's main module, which mustn't run the command again.
if __name__ == "__main__":
    sys.exit(main())
