"""The HTTP server's worker: ``python -m sars._server_worker``.

Reads one request from standard input as JSON, ``{"command", "handlers",
"target", "kwargs"}``, runs ``handlers.target(**kwargs)`` (by default a
function of ``sars._server_handlers``, which call the library's operations),
and prints the envelope as JSON. The server starts one per request, with its
own environment and a time limit; it imports neither the server's web
framework nor the command line.
"""

from __future__ import annotations

import importlib
import json
import sys


def main() -> None:
    from sars import _operations as ops

    request = json.loads(sys.stdin.read())
    backend = request.get("backend", "numpy")
    if backend != "numpy":
        import sars

        sars.set_backend(backend)  # checked when the server started
    handlers = importlib.import_module(request.get("handlers", "sars._server_handlers"))
    function = getattr(handlers, request["target"])
    envelope = ops.envelope(request["command"], lambda: function(**request["kwargs"]))
    sys.stdout.write(ops.envelope_json(envelope))


if __name__ == "__main__":
    main()
