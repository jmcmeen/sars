"""The command-line example scripts (examples/cli/*.sh) run as they are."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "examples" / "cli").glob("*.sh"))


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_the_example_script_runs(script):
    bin_dir = Path(sys.executable).parent
    if not (bin_dir / "sars").exists() and shutil.which("sars") is None:
        pytest.skip("the sars command isn't installed")
    if "jq" in script.read_text() and shutil.which("jq") is None:
        pytest.skip("needs jq")
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    done = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip()


def test_every_script_is_listed_in_the_readme():
    readme = (SCRIPTS[0].parent / "README.md").read_text(encoding="utf-8")
    assert all(f"`{s.name}`" in readme for s in SCRIPTS)
