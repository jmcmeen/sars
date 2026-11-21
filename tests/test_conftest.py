"""The helper that shares expensive fixtures between pytest-xdist workers."""

import pytest

from tests.conftest import shared


class _Factory:
    """Stands in for pytest's TempPathFactory: one worker's base directory."""

    def __init__(self, base):
        self.base = base

    def getbasetemp(self):
        return self.base


def test_a_value_is_computed_once_and_shared(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    calls = []
    first = shared(_Factory(tmp_path / "gw0"), "value", lambda: calls.append(1) or {"a": 1})
    again = shared(_Factory(tmp_path / "gw1"), "value", lambda: calls.append(1) or {"a": 2})
    assert first == again == {"a": 1}
    assert calls == [1]
    assert not list(tmp_path.glob("*.tmp"))  # written whole, then renamed


def test_a_failure_is_shared_without_computing_again(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")

    def fails():
        raise ValueError("the fit broke")

    with pytest.raises(RuntimeError, match="'broken' failed") as first:
        shared(_Factory(tmp_path / "gw0"), "broken", fails)
    assert isinstance(first.value.__cause__, ValueError)
    calls = []
    with pytest.raises(RuntimeError, match="'broken' failed"):
        shared(_Factory(tmp_path / "gw1"), "broken", lambda: calls.append(1))
    assert calls == []


def test_without_workers_the_value_is_computed_directly(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    value = object()
    assert shared(_Factory(tmp_path), "direct", lambda: value) is value
    assert not list(tmp_path.parent.glob("direct.pickle"))
