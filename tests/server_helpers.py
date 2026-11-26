"""Stand-ins the HTTP server's worker process can run (``module:function``
targets), for testing how the server handles what an operation does."""

import os
import subprocess
import sys
import time


def raises(kind):
    from sars._operations import InputError

    raise {"InputError": InputError, "TypeError": TypeError}[kind]("x")


def sleeps_with_a_worker(started, marker):
    """Start a process of its own that leaves ``marker`` if it survives, say
    so in ``started``, then sleep past any timeout."""
    code = f"import time, pathlib; time.sleep(8); pathlib.Path({str(marker)!r}).touch()"
    subprocess.Popen([sys.executable, "-c", code])
    open(started, "w").close()
    time.sleep(60)


def environment():
    return {"token": os.environ.get("SARS_SERVER_TOKEN")}


def dies():
    os._exit(3)
