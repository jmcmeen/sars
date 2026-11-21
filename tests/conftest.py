import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pytest
from filelock import FileLock

import sars

REF = Path(__file__).parent / "r_reference"

T = TypeVar("T")


def shared(tmp_path_factory: pytest.TempPathFactory, name: str, make: Callable[[], T]) -> T:
    """``make()``, computed once per test run and shared by every pytest-xdist
    worker through a pickle in the run's temporary directory.

    Session fixtures are per worker, so without this each worker refits the
    same models, every worker at once at the start of a run: the seven-island
    fits of all 20 models took over two minutes that way, on every core.

    Under xdist each worker gets its own unpickled copy, as it had its own
    object before; without xdist the value is ``make()`` itself. Don't compare
    these values by identity or change them in place.

    If ``make()`` raises, the error is shared too, so the other workers fail
    at once instead of each repeating the computation."""
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        return make()
    path = tmp_path_factory.getbasetemp().parent / f"{name}.pickle"
    with FileLock(f"{path}.lock"):
        if not path.is_file():
            try:
                outcome: tuple[bool, object] = (True, make())
            except Exception as exc:
                outcome = (False, exc)
            # Written whole and then renamed, so a worker that dies mid-write
            # can't leave a truncated file for the others.
            partial = path.with_suffix(f".{os.getpid()}.tmp")
            partial.write_bytes(pickle.dumps(outcome))
            os.replace(partial, path)
        ok, value = pickle.loads(path.read_bytes())
    if not ok:
        raise RuntimeError(f"computing the shared fixture {name!r} failed") from value
    return value  # type: ignore[return-value]


@pytest.fixture(scope="session")
def galap():
    path = REF / "galap.csv"
    if not path.exists():
        pytest.skip("Run generate_r_reference.R first")
    return pd.read_csv(path).rename(columns={"a": "area", "s": "species"})


@pytest.fixture(scope="session")
def galap_multi(galap, tmp_path_factory):
    """All 20 models fitted once, shared across tests and workers."""
    return shared(tmp_path_factory, "galap_multi", lambda: sars.sar_multi(galap))


@pytest.fixture(scope="session")
def galap_fits(galap_multi):
    """Dict of model_name -> SARFit from the shared multi-model run."""
    return {fit.model: fit for fit in galap_multi.fits}


@pytest.fixture(scope="session")
def galap_average(galap_multi):
    """The AICc-weighted average of the shared fits: what sar_average(galap)
    returns, without fitting the 20 models again."""
    weights = sars.akaike_weights(galap_multi.summary["AICc"].to_numpy(dtype=float))
    return sars.AveragedSAR(
        multi=galap_multi,
        ic="AICc",
        weights=dict(zip(galap_multi.summary["model"], weights, strict=True)),
    )
