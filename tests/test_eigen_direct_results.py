"""Private bounded Jacobi writes independent values and F vector roots."""
import os
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_eigenvalue_projection import exact, symmetric
from .test_matmul_library import settings
from .test_matrix_array_foundation import matrix
from .test_matrix_library_backend import PACKING_PLATFORM, sample

ADMITTED = PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN", "1") == "1"
EXPRESSIONS = {"w": "eigenvalues(symmetric_eigh(a))",
               "v": "eigenvectors(symmetric_eigh(a))"}


def execute(a, *, lane="single", order="auto", expressions=None, isolate=False):
    graph = GraphCompiler({"a": matrix()}).compile(
        EXPRESSIONS if expressions is None else expressions, output_order=order,
        error_policy="isolate" if isolate else "raise")
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        return scheduler.execute(graph, {"a": a}, bounds, bounds)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("layout", ["C", "F", "padded_C", "padded_F", "reverse"])
@pytest.mark.parametrize("order", ["auto", "F"])
def test_private_pair_every_lane_layout_and_ownership(lane, layout, order):
    a = sample(512)
    if layout.startswith("padded"):
        owner = np.full((515, 515), np.nan, order=layout[-1])
        owner[:512, :512] = a
        a = owner[:512, :512]
    else:
        a = np.array(a, order="C" if layout == "reverse" else layout)
        if layout == "reverse":
            a = a[::-1, ::-1]
    a.flags.writeable = False
    before = a.copy()
    record = op.symmetric_eigh(a)
    result = execute(a, lane=lane, order=order)
    for name, expected in (("w", record[:, 0]), ("v", record[:, 1:].T)):
        values = result.named_outputs[name].values
        for value in values:
            exact(value, expected)
            assert not value.flags.writeable and not np.shares_memory(value, a)
        assert not np.shares_memory(*values)
    assert not np.shares_memory(result.named_outputs["w"].values[0],
                                result.named_outputs["v"].values[0])
    if order == "F" or ADMITTED:
        assert result.named_outputs["v"].values[0].flags.f_contiguous
    for chunk in result.audit["native_chunks"]:
        if ADMITTED:
            assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
            assert chunk["direct_output_bytes"] == chunk["rows"] * record.nbytes
            assert chunk["eigen_rotation_calls"] > 0
            assert chunk["algorithm_copy_bytes"] > chunk["rows"] * record.nbytes // 2
        else:
            assert chunk["numeric_arena_bytes"] >= record.nbytes
    exact(a, before)


@pytest.mark.parametrize("n", [0, 1, 128, 256, 511, 512, 513, 768, 769, 1024, 2048])
@pytest.mark.parametrize("reverse_roots", [False, True])
def test_shape_admission_sorting_and_root_order(n, reverse_roots):
    a = np.diag(np.arange(n, 0, -1, dtype=float) % 7)
    a.flags.writeable = False
    record = op.symmetric_eigh(a)
    expressions = dict(reversed(list(EXPRESSIONS.items()))) if reverse_roots else EXPRESSIONS
    result = execute(a, expressions=expressions)
    exact(result.named_outputs["w"].values[0], record[:, 0])
    exact(result.named_outputs["v"].values[0], record[:, 1:].T)
    chunk = result.audit["native_chunks"][0]
    if ADMITTED and n >= 512 and n % 256 == 0:
        assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
    else:
        assert chunk["numeric_arena_bytes"] >= record.nbytes


@pytest.mark.parametrize("reason", ["C", "isolate", "extra_root", "consumer"])
def test_noneligible_graphs_keep_real_record(reason):
    a = sample(512)
    expressions = dict(EXPRESSIONS)
    if reason == "extra_root":
        expressions["independent"] = "7"
    if reason == "consumer":
        expressions["v"] += "+0"
    result = execute(a, expressions=expressions, order="C" if reason == "C" else "F",
                     isolate=reason == "isolate")
    record = op.symmetric_eigh(a)
    exact(result.named_outputs["w"].values[0], record[:, 0])
    expected_vectors = record[:, 1:].T + 0 if reason == "consumer" else record[:, 1:].T
    exact(result.named_outputs["v"].values[0], expected_vectors)
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] >= record.nbytes


@pytest.mark.parametrize("fault,code", [("nan", "NON_FINITE_RESULT"),
                                      ("asymmetry", "DOMAIN_ERROR"),
                                      ("sweeps", "NON_CONVERGENCE"),
                                      ("tolerance", "INVALID_PARAMETER")])
def test_producer_errors_are_not_hidden(fault, code):
    a = sample(512)
    parameters = "1e-12,32"
    if fault == "nan":
        a[-1, -1] = np.nan
    elif fault == "asymmetry":
        a[-1, 0] = 1
    elif fault == "sweeps":
        parameters = "1e-15,1"
    else:
        parameters = "0,32"
    expr = f"symmetric_eigh(a,{parameters})"
    with pytest.raises(ValueError, match=code):
        execute(a, expressions={"w": f"eigenvalues({expr})", "v": f"eigenvectors({expr})"})


def test_dense_reconstruction_and_exact_legacy_results():
    a = symmetric(512)
    expected = op.symmetric_eigh(a)
    result = execute(a)
    w, v = (result.named_outputs[name].values[0] for name in ("w", "v"))
    exact(w, expected[:, 0])
    exact(v, expected[:, 1:].T)
    np.testing.assert_allclose((v * w) @ v.T, a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(v.T @ v, np.eye(512), rtol=1e-12, atol=1e-12)


def test_prepared_serialized_snapshots_geometry_and_budget():
    a = sample(512)
    graph = pickle.loads(pickle.dumps(GraphCompiler({"a": matrix()}).compile(EXPRESSIONS)))
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a}, bounds, bounds)
        first = prepared.run_snapshot()
        snapshots = [root.values[0].copy() for root in first.outputs]
        a += np.eye(512)
        second = prepared.run_snapshot()
        record = op.symmetric_eigh(a)
        for index, expected in enumerate((record[:, 0], record[:, 1:].T)):
            exact(second.outputs[index].values[0], expected)
            exact(first.outputs[index].values[0], snapshots[index])
            assert not np.shares_memory(first.outputs[index].values[0], second.outputs[index].values[0])
        plan = scheduler.plan(graph, {"a": a}, bounds, bounds)
        assert plan.estimated_output_bytes == record.nbytes
        with pytest.raises(MemoryError):
            scheduler.plan(graph, {"a": a}, bounds, bounds,
                           memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        # Same graph changes actual shape and must not retain the larger scratch.
        smaller = scheduler.execute(graph, {"a": sample(128)}, bounds, bounds)
        small_record = op.symmetric_eigh(sample(128))
        exact(smaller.outputs[1].values[0], small_record[:, 1:].T)
        with pytest.raises(ValueError, match="stale|geometry|changed|mismatch"):
            scheduler.execute(graph, {"a": sample(128)}, bounds, bounds, plan=plan)
