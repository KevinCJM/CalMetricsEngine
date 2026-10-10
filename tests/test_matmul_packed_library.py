"""Large GEMM panels: mathematical boundaries, budgets and independent results."""
import gc
import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_matmul_library import assert_ieee_classes_and_close, declarations, settings
from .test_matrix_library_backend import PACKING_PLATFORM

ISA = "eigen-packed-matmul"


def expected_panels(k):
    return (320 * k + 72) * 8


def assert_dispatch(audit, eligible=True):
    enabled = os.environ.get("CME_EXPECT_EIGEN")
    if not eligible or not PACKING_PLATFORM or enabled == "0":
        assert audit["isa"] != ISA
        assert audit["workspace_bytes"] == 0
    elif enabled == "1":
        assert audit["isa"] == ISA


def integer_inputs(m=1025, k=257, n=1027, layout="cc"):
    a = np.array(np.broadcast_to((np.arange(m) % 5 - 2)[:, None], (m, k)), dtype=np.float64, order=layout[0].upper())
    b = np.array(np.broadcast_to((np.arange(n) % 7 - 3)[None, :], (k, n)), dtype=np.float64, order=layout[1].upper())
    # Mixed-sign GEMM has a bounded domain; exact power-of-two scaling keeps
    # this fixture testing the packed fallback without changing its integer math.
    a *= 1024.0
    b *= 1024.0
    # Paired sign flips preserve every product.
    a[:, ::2] *= -1
    b[::2] *= -1
    expected = k * (np.arange(m) % 5 - 2)[:, None] * (np.arange(n) % 7 - 3)[None, :]
    a.flags.writeable = b.flags.writeable = False
    return a, b, expected.astype(np.float64) * 1024.0 ** 2


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("n", [1024, 1025, 1027])
def test_packed_borrows_layout_keeps_tails_canaries_and_old_output(layout, n):
    a, b, expected = integer_inputs(layout=layout, n=n)
    work = op.Workspace()
    bound = op.get("matmul").requirements(a, b)["scratch_doubles"]
    work.reserve(doubles=bound)
    before = (a.copy(), b.copy())
    storage = np.full(expected.size + 2, np.nan)
    storage[[0, -1]] = [71., -71.]
    out = storage[1:-1].reshape(expected.shape)
    actual, audit = op.matmul(a, b, out=out, workspace=work, audit=True)
    assert actual is out
    assert_dispatch(audit)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(storage[[0, -1]], [71., -71.])
    assert audit["input_copy_bytes"] == 0
    if audit["isa"] == ISA:
        assert audit["workspace_bytes"] == expected_panels(a.shape[1])
        assert audit["workspace_capacity_bytes"] == audit["workspace_bytes"]
        prefix, bulk = a.shape[0] // 4 * 4, b.shape[1] // 6 * 6
        assert audit["algorithm_copy_bytes"] == (prefix + ((prefix + 127) // 128) * bulk) * a.shape[1] * 8
    second = op.matmul(a * 2, b, workspace=work)
    np.testing.assert_array_equal(second, expected * 2)
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, second)
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("case", ["cancellation", "overflow", "zero_inf", "signed_tiny"])
def test_packed_ieee_and_comparisons_include_bulk_and_tails(layout, case):
    m, k, n = 1025, 257, 1027
    a = np.zeros((m, k), order=layout[0].upper())
    b = np.ones((k, n), order=layout[1].upper())
    if case == "cancellation":
        a[:, :3] = [1e16, 1., -1e16]
    elif case == "overflow":
        a[:, :3] = [1e308, 1e308, -1e308]
        b[:3] = 2.
    elif case == "zero_inf":
        b[7] = np.inf
    else:
        a.fill(-0.)
        a[0, 0] = a[-1, 0] = np.nextafter(0., 1.)
    a.flags.writeable = b.flags.writeable = False
    actual, audit = op.matmul(a, b, audit=True)
    assert_dispatch(audit)
    # Independent small scalar products exercise both packed bulk and remainder.
    for rows, cols in [(slice(0, 2), slice(0, 2)), (slice(-2, None), slice(-2, None))]:
        reference = op.matmul(a[rows], b[:, cols], simd="scalar")
        assert_ieee_classes_and_close(actual[rows, cols], reference)
        np.testing.assert_array_equal(actual[rows, cols] > 0, reference > 0)
        np.testing.assert_array_equal(actual[rows, cols] == 0, reference == 0)
    if case == "cancellation":
        np.testing.assert_array_equal(actual, 0.)
    elif case in {"overflow", "zero_inf"}:
        assert np.isnan(actual).all()


@pytest.mark.parametrize("mode", ["small", "small_k", "negative", "repeated", "scalar", "padded"])
def test_packed_admission_does_not_allocate_for_other_paths(mode):
    a, b, expected = integer_inputs(m=1023 if mode == "small" else 1024, k=255 if mode == "small_k" else 257, n=1024)
    if mode == "negative":
        a, expected = a[::-1], expected[::-1]
    elif mode == "repeated":
        a = np.broadcast_to(a[:1], a.shape)
        expected = np.broadcast_to(expected[:1], expected.shape)
    elif mode == "padded":
        holder = np.full((b.shape[0] + 3, b.shape[1]), np.nan, order="F")
        holder[:b.shape[0]] = b
        b = holder[:b.shape[0]]
        b.flags.writeable = False
    actual, audit = op.matmul(a, b, audit=True, simd="scalar" if mode == "scalar" else "auto")
    assert_dispatch(audit, mode == "padded")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_packed_graph_worker_ownership_copy_audit_and_budget(lane):
    a, b, expected = integer_inputs(m=1024, k=256, n=1024, layout="ff")
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(2, np.int64)
    values = {"a": a, "b": b}
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        plan = scheduler.plan(graph, values, bounds, bounds)
        if PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN") == "1":
            assert plan.estimated_worker_scratch_bytes >= expected_panels(256)
        with pytest.raises((ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc"):
            scheduler.execute(graph, values, bounds, bounds, memory_budget_bytes=expected_panels(256) - 1)
        result = scheduler.execute(graph, values, bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    for output in result.outputs[0].values:
        np.testing.assert_array_equal(output, expected)
        assert not output.flags.writeable and not np.shares_memory(output, a)
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[0].values[1])


def test_packed_shared_input_release_and_prepared_snapshot():
    a, b, expected = integer_inputs(m=1024, k=256, n=1024, layout="ff")
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
        first = prepared.run_snapshot()
        prepared.run_snapshot()
        np.testing.assert_array_equal(first.outputs[0].values[0], expected)
    with SharedInputBundle.from_inputs({"a": a, "b": b}) as shared:
        with AdaptiveScheduler(cpu_budget=1, config=settings("process_shared")) as scheduler:
            result = scheduler.execute(graph, shared.arrays, bounds, bounds, hard_stop=True)
    del shared
    gc.collect()
    np.testing.assert_array_equal(result.outputs[0].values[0], expected)


def test_packed_shape_change_drops_old_panel_capacity_even_when_arena_size_stays():
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    capacities = []
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        for k in (512, 256):
            a, b, expected = integer_inputs(m=1024, k=k, n=1024)
            result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
            np.testing.assert_array_equal(result.outputs[0].values[0], expected)
            capacities.append(result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"])
    if PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN") == "1":
        assert capacities[0] - capacities[1] == (512 - 256) * 320 * 8


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("lane", ["single", "process_shared"])
def test_packed_nested_scope_includes_panel_budget(nested, lane):
    # Matrix captures are supported by iterate; block/filter scopes deliberately
    # retain their existing one-dimensional capture contract.
    a, b, expected = integer_inputs(m=1024, k=256, n=1024)
    body = "iterate_x+sum(matmul(a,b))"
    if nested:
        body = f"iterate_x+iterate({body},0,0,2)"
    graph = GraphCompiler(declarations()).compile(
        f"iterate({body},0,0,2)", result_format="typed", scope_work_budget=10_000_000_000)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds, hard_stop=lane.startswith("process"))
    assert result.outputs[0].values[0] == (4 if nested else 2) * expected.sum()
    assert result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"] <= result.plan.estimated_worker_scratch_bytes



@pytest.mark.parametrize("lane", ["single", "process_shared"])
def test_packed_graph_preserves_discrete_and_error_consumers(lane):
    a = np.zeros((1024, 256), order="F")
    a[:, :3] = [1e16, 1., -1e16]
    b = np.ones((256, 1024), order="F")
    graph = GraphCompiler(declarations()).compile(
        ["matmul(a,b)", "greater_than(matmul(a,b),0)", "1/sum(matmul(a,b))", "7"], error_policy="isolate")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds, hard_stop=lane.startswith("process"))
    matrix, comparison, broken, healthy = result.outputs
    np.testing.assert_array_equal(matrix.values[0], 0.)
    np.testing.assert_array_equal(comparison.values[0], False)
    assert matrix.root_statuses[0] == comparison.root_statuses[0] == healthy.root_statuses[0] == 0
    assert broken.root_statuses[0] != 0 and healthy.values[0] == 7
