"""The example notebooks in docs/examples: committed executed (the docs render their stored
outputs), and still running start to finish against the current code."""

import copy
from pathlib import Path

import nbformat
import pytest
from ipykernel.kernelspec import install as install_kernelspec
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


@pytest.fixture(scope="session")
def kernel_dir(tmp_path_factory):
    """A kernelspec for the interpreter running the tests. Looking ``python3`` up by name
    would find whatever kernel the developer registered first, which may not have sars."""
    prefix = tmp_path_factory.mktemp("kernels")
    install_kernelspec(prefix=str(prefix), kernel_name="sars-tests")
    return prefix / "share" / "jupyter" / "kernels"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_notebook_is_committed_executed_and_runs(path, kernel_dir):
    committed = nbformat.read(path, as_version=4)
    code_cells = [c for c in committed.cells if c.cell_type == "code"]
    assert code_cells
    # What the docs render: every cell was run, and none ended in a traceback.
    assert all(c.execution_count is not None for c in code_cells), "notebook has unrun cells"
    assert not [o for c in code_cells for o in c.outputs if o.output_type == "error"]

    km = KernelManager(
        kernel_name="sars-tests",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]),
    )
    client = NotebookClient(
        copy.deepcopy(committed),
        km=km,
        timeout=900,
        resources={"metadata": {"path": str(EXAMPLES)}},
    )
    client.execute()  # raises CellExecutionError on the first failing cell
