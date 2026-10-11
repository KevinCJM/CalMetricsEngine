"""Measurement-boundary regressions without importing or calling the engine."""

import importlib.util
import os
import sys
import weakref
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def tool(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(directory))
    # Loading a standalone benchmark sets these variables. Register their old
    # values with monkeypatch so this test cannot change later suite behavior.
    for key in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS", "MKL_NUM_THREADS",
    ):
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    spec = importlib.util.spec_from_file_location(
        "_large_matrix_performance_tool_test", directory / "check_large_matrix_performance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LazyField:
    def __init__(self, value, index, reads):
        self.value, self.index, self.reads = value, index, reads

    @property
    def values(self):
        self.reads.append(("values", self.index))
        return (self.value,)


class LazyResult:
    def __init__(self, arrays, reads):
        self.fields = tuple(LazyField(array, index, reads) for index, array in enumerate(arrays))
        self.reads = reads
        self.audit = {"source": "original-result"}

    @property
    def outputs(self):
        self.reads.append(("outputs",))
        return self.fields


def test_materialization_reads_all_fields_now_and_keeps_heterogeneous_storage(tool):
    floating = np.arange(12, dtype=np.float64)[::-3]
    floating.flags.writeable = False
    integer = np.array([[2**53 + 1, -3], [4, 5]], dtype=np.int64)
    boolean = np.array(True, dtype=np.bool_)
    original = (floating, integer, boolean)
    reads = []
    raw = LazyResult(original, reads)
    materialized = tool.MaterializedResult(raw)
    assert reads == [("outputs",), ("values", 0), ("values", 1), ("values", 2)]
    assert materialized.raw is raw
    assert tool.value(materialized) is materialized.outputs
    exposed = tool.output_arrays(materialized)
    for actual, expected in zip(exposed, original, strict=True):
        assert actual is expected
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.strides == expected.strides
    assert not exposed[0].flags.writeable
    assert exposed[1][0, 0] == 2**53 + 1
    assert exposed[2].shape == ()
    assert tool.audit({"entry": "ordinary"}, "new", {}, materialized) is raw.audit
    # Verification and diagnostic access must not reopen lazy output fields.
    assert reads == [("outputs",), ("values", 0), ("values", 1), ("values", 2)]


@pytest.mark.parametrize("accessor", ["value", "output_arrays"])
def test_unmaterialized_lazy_result_is_rejected_without_reading_outputs(tool, accessor):
    reads = []
    raw = LazyResult((np.arange(3, dtype=np.float64),), reads)
    with pytest.raises(TypeError, match="inside the measured callable"):
        getattr(tool, accessor)(raw)
    assert reads == []


def test_single_output_keeps_owner_and_does_not_hide_aliases(tool):
    array = np.arange(8, dtype=np.float64)[::2]
    raw = LazyResult((array,), [])
    retained_owner = weakref.ref(raw)
    materialized = tool.MaterializedResult(raw)
    del raw
    assert retained_owner() is materialized.raw
    assert tool.value(materialized) is array
    actual = tool.output_arrays(materialized)
    assert actual[0] is array
    with pytest.raises(AssertionError):
        tool.check_independent_outputs(actual, (array.view(),), {})
    tool.check_independent_outputs(actual, (array.copy(),), {})
    del materialized
    assert retained_owner() is None


def test_permanent_em_tool_uses_repo_root_and_complete_heterogeneous_cases(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    # The EM entry point configures a harness on import. Replace that harness
    # so this metadata-only check neither changes a live benchmark nor imports
    # a numerical engine or prepares/executes any compiled reference.
    harness = ModuleType("check_large_matrix_performance")
    monkeypatch.setitem(sys.modules, "check_large_matrix_performance", harness)
    spec = importlib.util.spec_from_file_location(
        "_em_library_performance_tool_test", root / "tools/check_em_library_performance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == root
    selected = module.cases()
    assert len(selected) == 24
    assert {(case["name"], case["n"], case["entry"]) for case in selected} == {
        (name, rows, entry)
        for name in ("gmm", "hmm")
        for rows in (16, 252, 2048, 32768)
        for entry in ("ordinary", "direct", "prepared")
    }
    assert all(case["iterations"] == 12 for case in selected)
    assert all(case["rtol"] == case["atol"] == 1e-9 for case in selected)
    assert module.MODELS["gmm"][2] == (
        "mixture_weights", "mixture_means", "mixture_variances",
    )
    assert module.MODELS["hmm"][2] == (
        "hmm_initial", "hmm_transition", "hmm_means", "hmm_variances",
    )
    assert module.DIAGNOSTICS == ("em_iterations", "em_log_likelihood", "em_converged")
    # All array parameters plus int64 count, float likelihood and bool state
    # remain separate outputs: six for GMM and seven for HMM.
    assert len(module.MODELS["gmm"][2] + module.DIAGNOSTICS) == 6
    assert len(module.MODELS["hmm"][2] + module.DIAGNOSTICS) == 7
    assert harness.cases is module.cases and harness.setup is module.setup
