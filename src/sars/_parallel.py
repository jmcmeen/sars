"""Running independent work (bootstrap resamples) across processes."""

from __future__ import annotations

import multiprocessing
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from operator import index
from typing import Any

_BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def n_workers(n_jobs: int) -> int:
    """The number of processes ``n_jobs`` asks for: itself if positive, and
    for -1 one per core this process may use."""
    given = n_jobs
    try:
        if isinstance(n_jobs, bool):
            raise TypeError
        n_jobs = index(n_jobs)
    except TypeError:
        raise ValueError(f"n_jobs must be a whole number, got {given!r}") from None
    if n_jobs == -1:
        return _usable_cores()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive whole number or -1, got {given!r}")
    return n_jobs


def _usable_cores() -> int:
    """Cores this process may run on (its affinity, or a container's limit where
    Python sees it), not every core in the machine."""
    count = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if count is not None:
        return count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


class Workers:
    """``n_jobs`` worker processes for mapping a function over items in order.

    Use as a context manager, and call ``map`` as often as needed: the pool
    starts once. ``shared`` holds keyword arguments every call to the function
    gets; they go to each worker once, when it starts, not with every item.
    One worker (``n_jobs=1``) runs everything in the calling process, with no
    pool.

    Warnings a worker raises are recorded there and raised again here, in
    order, so the caller's warning filters see them as they would in one
    process. The function and items must be picklable, the function defined at
    module level.
    """

    def __init__(
        self, n_jobs: int, shared: dict[str, Any] | None = None, stacklevel: int = 3
    ) -> None:
        # stacklevel: where a warning from __enter__ points, counted from
        # __enter__ (3: the caller of the function that opened the pool).
        self.stacklevel = stacklevel
        self.workers = n_workers(n_jobs)
        self.shared = shared or {}
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Workers:
        if self.workers > 1 and _main_without_file():
            warnings.warn(
                f"n_jobs={self.workers} needs the main script to be a file, which each "
                "worker process imports; this one isn't a file the workers can open (read "
                "from standard input or a pipe). Running in this process instead (as "
                "n_jobs=1).",
                stacklevel=self.stacklevel,
            )
            self.workers = 1
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=_context(),
                initializer=_start_worker,
                initargs=(self.shared,),
            )
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pool is not None:
            self._pool.shutdown()

    def map(self, fn: Callable[..., Any], items: Sequence[Any]) -> list[Any]:
        """``[fn(item, **shared) for item in items]``, in that order."""
        if self._pool is None or len(items) <= 1:
            return [fn(item, **self.shared) for item in items]
        # A few chunks per worker balance the load without a round trip per item.
        chunksize = max(1, -(-len(items) // (self.workers * 4)))
        calls = ((fn, item) for item in items)
        try:
            outcomes = list(self._pool.map(_call_in_worker, calls, chunksize=chunksize))
        except (BrokenProcessPool, EOFError) as exc:
            # A worker that dies while starting (as when it re-runs an unguarded
            # script) breaks the pool, or the pipe to it, depending on when.
            raise RuntimeError(
                "The worker processes for n_jobs stopped before finishing. A script that "
                "runs this at its top level needs the call under "
                "`if __name__ == '__main__':`, since each worker imports the script; "
                "or run with n_jobs=1."
            ) from exc
        results = []
        for result, caught in outcomes:
            for message, category, filename, lineno in caught:
                warnings.warn_explicit(message, category, filename, lineno)
            results.append(result)
        return results


def map_in_order(
    fn: Callable[..., Any], items: Sequence[Any], n_jobs: int, shared: dict[str, Any] | None = None
) -> list[Any]:
    """``[fn(item, **shared) for item in items]``, in ``n_jobs`` processes."""
    with Workers(n_jobs, shared, stacklevel=4) as workers:
        return workers.map(fn, items)


# What each worker process keeps for the calls it runs: the shared arguments.
_SHARED: dict[str, Any] = {}


def _start_worker(shared: dict[str, Any]) -> None:
    # The pool's workers already run side by side, one per core: a math-library
    # thread pool per worker sized to the whole machine would oversubscribe it.
    # Best effort: it only takes effect where the library starts after this.
    for name in _BLAS_THREADS:
        os.environ.setdefault(name, "1")
    _SHARED.update(shared)


def _call_in_worker(
    call: tuple[Callable[..., Any], Any],
) -> tuple[Any, list[tuple[str, type[Warning], str, int]]]:
    fn, item = call
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(item, **_SHARED)
    return result, [(str(w.message), w.category, w.filename, w.lineno) for w in caught]


def _main_without_file() -> bool:
    """Whether worker processes would fail to import the main script.

    Follows multiprocessing's own choice (``spawn.get_preparation_data``): a
    worker re-runs the main script by path only when ``__main__`` wasn't run
    as a module (``python -m``, a zipapp) and the program isn't frozen. That
    path, made absolute against the directory the process started in, then
    has to be a regular file the worker can open: "<stdin>" (``python -``)
    isn't one, nor is a pipe such as "/dev/fd/63" (``python <(...)``). An
    interactive session or a notebook has no ``__file__``, and is fine.
    """
    import sys
    from multiprocessing import process

    main = sys.modules.get("__main__")
    if getattr(sys, "frozen", False) or getattr(getattr(main, "__spec__", None), "name", None):
        return False
    path = getattr(main, "__file__", None)
    if not isinstance(path, str):
        return False
    original_dir = getattr(process, "ORIGINAL_DIR", None)  # not in the type stubs
    if not os.path.isabs(path) and original_dir is not None:
        path = os.path.join(original_dir, path)
    return not os.path.isfile(os.path.normpath(path))


def _context() -> multiprocessing.context.BaseContext:
    """How to start the workers: forkserver where the platform has it (Python
    3.14's default on Linux), else spawn. Plain fork copies a process that may
    have threads running (numpy's, a notebook's), which can deadlock the child."""
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")


def chunks(rows: Iterable[Any], n: int) -> list[Any]:
    """Split ``rows`` (an array) into at most ``n`` non-empty, nearly equal parts."""
    import numpy as np

    arr = np.asarray(rows)
    return [part for part in np.array_split(arr, max(1, min(n, len(arr)))) if len(part)]
