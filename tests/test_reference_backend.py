"""A portable correctness oracle must never become a fake NJIT benchmark."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_backend(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("CALMETRICS_TEST_REFERENCE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("CALMETRICS_TEST_REFERENCE_BACKEND", mode)
    spec = importlib.util.spec_from_file_location(
        "_reference_backend_contract", Path(__file__).with_name("reference_backend.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_python_oracle_runs_original_body_without_numba(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)
    backend = load_backend(monkeypatch, "python")

    def body(values):
        return sum(values)

    decorated = backend.njit(cache=True, nogil=True)(body)
    assert decorated is body and decorated([2, 3]) == 5
    with pytest.raises(RuntimeError, match="requires the njit"):
        backend.require_njit_reference()


@pytest.mark.parametrize("mode", [None, "njit"])
def test_missing_numba_never_falls_back(monkeypatch, mode):
    monkeypatch.setitem(sys.modules, "numba", None)
    with pytest.raises(ModuleNotFoundError):
        load_backend(monkeypatch, mode)


def test_disabled_jit_fails_before_reference_execution(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "numba", SimpleNamespace(config=SimpleNamespace(DISABLE_JIT=True), njit=None)
    )
    with pytest.raises(RuntimeError, match="NUMBA_DISABLE_JIT=0"):
        load_backend(monkeypatch, "njit")


@pytest.mark.parametrize("mode", ["", "auto", "fallback"])
def test_unknown_reference_mode_is_rejected(monkeypatch, mode):
    with pytest.raises(RuntimeError, match="must be njit or python"):
        load_backend(monkeypatch, mode)
