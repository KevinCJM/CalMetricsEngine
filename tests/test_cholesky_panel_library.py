"""Ordered Eigen Cholesky panels: numerical policy, budgets and ownership."""

import gc
import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_matmul_library import settings
from .test_matrix_library_backend import PACKING_PLATFORM

ISA = "eigen-array-panel-cholesky"
DECL = {"a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"])}


def panel_bytes(n):
    return (16 * (n + 17) + 128) * 8


def spd(n):
    rng = np.random.default_rng(663781)
    a = rng.normal(size=(n, n))
    a = (a + a.T) * 0.5
    a[np.diag_indices(n)] += np.abs(a).sum(axis=1) + 1.0
    return a


def assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def assert_dispatch(audit, n, eligible=True):
    enabled = os.environ.get("CME_EXPECT_EIGEN")
    if not eligible or n < 1024 or not PACKING_PLATFORM or enabled == "0":
        assert audit["isa"] != ISA
        assert audit["workspace_bytes"] == 0
    elif enabled == "1":
        assert audit["isa"] == ISA
        assert audit["workspace_bytes"] == panel_bytes(n)
    if audit["isa"] == ISA:
        assert audit["algorithm_copy_bytes"] == n * (n + 1) * 8
    assert audit["input_copy_bytes"] == 0


@pytest.mark.parametrize(
    "n,layout",
    [
        (1023, "c"),
        (1024, "c"),
        (1025, "c"),
        (1031, "f"),
        (1025, "negative"),
        (1025, "padded"),
        (1025, "sliced"),
    ],
)
def test_panel_layout_tails_bits_input_and_output_canaries(n, layout):
    a = spd(n)
    if layout == "f":
        a = np.asfortranarray(a)
    elif layout == "negative":
        a = a[::-1, ::-1]
    elif layout in {"padded", "sliced"}:
        holder = np.full((n * 2, n * 2), np.nan)
        target = holder[:n, :n] if layout == "padded" else holder[::2, ::2]
        target[:] = a
        a = target
    before = a.copy()
    a.flags.writeable = False
    expected = op.cholesky(a, simd="scalar")
    storage = np.full(n * n + 2, -771.0)
    out = storage[1:-1].reshape(n, n)
    work = op.Workspace()
    bound = op.get("cholesky").requirements(a)["scratch_doubles"]
    work.reserve(doubles=bound)
    actual, audit = op.cholesky(a, out=out, workspace=work, audit=True)
    assert actual is out
    assert_bits(actual, expected)
    assert_bits(a, before)
    assert storage[0] == storage[-1] == -771.0
    assert_dispatch(audit, n)
    np.testing.assert_array_equal(np.triu(actual, 1), 0.0)
    second = op.cholesky(a * 4, workspace=work)
    assert not np.shares_memory(actual, second)
    assert_bits(actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        "cancellation",
        "tiny",
        "signed_zero",
        "domain_priority",
        "overflow",
        "upper_nonfinite",
        "asymmetry",
        "zero_stride",
    ],
)
def test_panel_numerical_policy_and_error_priority(case):
    n = 1025
    a = np.eye(n)
    expected_error = None
    if case == "cancellation":
        ids = [0, 1, n - 1]
        a[np.ix_(ids, ids)] = [[1, 0, 2.0**27], [0, 1, 1], [2.0**27, 1, 2.0**54 + 4]]
    elif case == "tiny":
        a[0, 0] = np.nextafter(0.0, 1.0)
    elif case == "signed_zero":
        a[a == 0] = -0.0
    elif case == "domain_priority":
        a[16, 16] = -1.0
        a[0, 0] = np.nextafter(0.0, 1.0)
        a[0, 17] = a[17, 0] = 1e308
        expected_error = "DOMAIN_ERROR"
    elif case == "overflow":
        a[0, 0] = np.nextafter(0.0, 1.0)
        a[0, 17] = a[17, 0] = 1e308
        expected_error = "NON_FINITE_RESULT"
    elif case == "upper_nonfinite":
        a[0, 0] = -1.0
        a[-2, -1] = np.nan
        expected_error = "NON_FINITE_RESULT"
    elif case == "asymmetry":
        a[0, -1] = np.nextafter(0.0, 1.0)
        expected_error = "DOMAIN_ERROR"
    else:
        a = np.broadcast_to(np.ones(1), (n, n))
        expected_error = "DOMAIN_ERROR"
    a.flags.writeable = False
    if expected_error:
        for isa in ("scalar", "auto"):
            with pytest.raises((ValueError, RuntimeError), match=expected_error):
                op.cholesky(a, simd=isa)
    else:
        expected = op.cholesky(a, simd="scalar")
        actual, audit = op.cholesky(a, audit=True)
        assert_bits(actual, expected)
        assert_dispatch(audit, n)
        if case == "cancellation":
            assert actual[-1, -1] == np.sqrt(3.0)


def test_explicit_scalar_keeps_original_allocation_contract():
    a = np.eye(1024)
    actual, audit = op.cholesky(a, audit=True, simd="scalar")
    assert_dispatch(audit, len(a), False)
    assert audit["workspace_capacity_bytes"] == 0
    assert_bits(actual, a)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_graph_lane_budget_retention_and_ownership(lane):
    a = np.eye(1024)
    graph = GraphCompiler(DECL).compile("cholesky(a)")
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        plan = scheduler.plan(graph, {"a": a}, bounds, bounds)
        if PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN") == "1":
            assert plan.estimated_worker_scratch_bytes >= panel_bytes(len(a))
        with pytest.raises(
            (ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc"
        ):
            scheduler.execute(
                graph, {"a": a}, bounds, bounds, memory_budget_bytes=panel_bytes(len(a)) - 1
            )
        result = scheduler.execute(graph, {"a": a}, bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    outputs = result.outputs[0].values
    for output in outputs:
        assert_bits(output, a)
        assert not output.flags.writeable and not np.shares_memory(output, a)
    assert not np.shares_memory(outputs[0], outputs[1])


def test_prepared_snapshots_and_shared_owner_close():
    a = np.eye(1024)
    graph = GraphCompiler(DECL).compile("cholesky(a)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a}, bounds, bounds)
        first, second = prepared.run_snapshot(), prepared.run_snapshot()
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
        assert_bits(first.outputs[0].values[0], a)
    with SharedInputBundle.from_inputs({"a": a}) as shared:
        with AdaptiveScheduler(cpu_budget=1, config=settings("process_shared")) as scheduler:
            result = scheduler.execute(graph, shared.arrays, bounds, bounds, hard_stop=True)
    del shared
    gc.collect()
    assert_bits(result.outputs[0].values[0], a)


def test_shape_shrink_releases_panels_even_when_largest_array_is_unchanged():
    decl = dict(DECL, b=dict(kind="matrix", axes=["asset", "asset"], shape=["M", "M"]))
    graph = GraphCompiler(decl).compile(["cholesky(a)", "sum(b)"])
    bounds = np.zeros(1, np.int64)
    capacities = []
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        for n in (1040, 1024, 1023):
            result = scheduler.execute(
                graph, {"a": np.eye(n), "b": np.ones((1040, 1040))}, bounds, bounds
            )
            capacities.append(result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"])
    if PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN") == "1":
        assert capacities == [panel_bytes(1040), panel_bytes(1024), 0]


@pytest.mark.parametrize("lane", ["single", "process_shared"])
def test_nested_scope_budget_and_isolated_failures(lane):
    bounds = np.zeros(1, np.int64)
    expression = "iterate(iterate_x+sum(cholesky(a)),0,0,2)"
    graph = GraphCompiler(DECL).compile(
        [expression, f"iteration_status({expression})", "cholesky(a)", "7"],
        error_policy="isolate",
        result_format="typed",
        scope_work_budget=10_000_000_000,
    )
    with AdaptiveScheduler(cpu_budget=1, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"a": np.eye(1024)}, bounds, bounds)
        assert result.outputs[0].values[0] == 2048.0
        assert (
            result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"]
            <= result.plan.estimated_worker_scratch_bytes
        )
        bad = np.eye(1024)
        bad[16, 16] = -1.0
        result = scheduler.execute(graph, {"a": bad}, bounds, bounds)
    # Iteration numerical failure returns its last finite state plus status 5;
    # a directly consumed failing operator remains a failed root.
    assert result.outputs[0].root_statuses[0] == 0
    assert result.outputs[0].values[0] == 0.0
    assert result.outputs[1].values[0] == 5
    assert result.outputs[2].root_statuses[0] != 0
    assert result.outputs[3].values[0] == 7.0
