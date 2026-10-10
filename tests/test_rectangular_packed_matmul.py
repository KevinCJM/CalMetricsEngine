"""Existing ordered Eigen packing also covers bounded rectangular row blocks."""

import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import assert_ieee_classes_and_close, declarations, inputs, settings
from .test_matrix_library_backend import PACKING_PLATFORM

SHAPES = [
    (31, 30, 252),
    (32, 1, 128),
    (33, 30, 129),
    (63, 30, 252),
    (64, 30, 2520),
    (65, 128, 253),
    (128, 128, 128),
    (129, 30, 252),
    (64, 129, 252),
    (64, 30, 127),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff", "padded_c_b", "padded_f_b"])
def test_rectangular_packed_edges_borrowing_workspace_and_old_results(shape, layout):
    a, b, owners = inputs(shape, layout)
    saved = [x.copy() for x in (a, b, *owners)]
    a.flags.writeable = b.flags.writeable = False
    expected = op.matmul(a, b, simd="scalar")
    work = op.Workspace()
    storage = np.full(expected.size + 2, 71.0)
    storage[-1] = -71.0
    out = storage[1:-1].reshape(expected.shape)
    actual, audit = op.matmul(a, b, out=out, workspace=work, audit=True)
    assert actual is out
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(storage[[0, -1]], [71.0, -71.0])
    rectangular = 32 <= shape[0] <= 128 and 1 <= shape[1] <= 128 and shape[2] >= 128
    if PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN") == "1":
        assert (audit["isa"] == "eigen-packed-matmul") == rectangular
    if os.environ.get("CME_EXPECT_EIGEN") == "0":
        assert audit["isa"] != "eigen-packed-matmul"
    assert audit["input_copy_bytes"] == 0
    if audit["isa"] == "eigen-packed-matmul":
        assert (
            audit["workspace_bytes"]
            == audit["workspace_capacity_bytes"]
            == (320 * shape[1] + 72) * 8
        )
        prefix = shape[0] // 4 * 4
        bulk = shape[2] // 6 * 6
        assert (
            audit["algorithm_copy_bytes"]
            == (prefix + ((prefix + 127) // 128) * bulk) * shape[1] * 8
        )
    second = op.matmul(a, b, workspace=work)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert not np.shares_memory(actual, second)
    for x, original in zip((a, b, *owners), saved, strict=True):
        np.testing.assert_array_equal(x.view(np.uint64), original.view(np.uint64))


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("case", ["cancellation", "fma", "overflow", "zero_inf", "signed_tiny"])
def test_rectangular_pack_preserves_ieee_discrete_boundaries(layout, case):
    a = np.zeros((65, 30), order=layout[0].upper())
    b = np.ones((30, 253), order=layout[1].upper())
    if case == "cancellation":
        a[:, :3] = [1e16, 1.0, -1e16]
    elif case == "fma":
        a[:, :2] = [-1.0, 1 + 2**-27]
        b[:2] = np.array([1.0, 1 - 2**-27])[:, None]
    elif case == "overflow":
        a[:, :3] = [1e308, 1e308, -1e308]
        b[:3] = 2.0
    elif case == "zero_inf":
        b[7] = np.inf
    else:
        a.fill(-0.0)
        a[0, 0] = a[-1, 0] = np.nextafter(0.0, 1.0)
    a.flags.writeable = b.flags.writeable = False
    expected = op.matmul(a, b, simd="scalar")
    actual = op.matmul(a, b)
    assert_ieee_classes_and_close(actual, expected)
    for comparison in [np.greater, np.less, np.equal]:
        np.testing.assert_array_equal(comparison(actual, 0), comparison(expected, 0))
    if case in ["cancellation", "fma"]:
        np.testing.assert_array_equal(actual, 0.0)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("layout", ["cc", "ff"])
def test_rectangular_graph_keeps_error_isolation_budget_transports_and_owners(lane, layout):
    a = np.zeros((65, 30), order=layout[0].upper())
    a[:, :3] = [1e16, 1.0, -1e16]
    b = np.ones((30, 253), order=layout[1].upper())
    a.flags.writeable = b.flags.writeable = False
    graph = GraphCompiler(declarations()).compile(
        ["matmul(a,b)", "greater_than(matmul(a,b),0)", "1/sum(matmul(a,b))", "7"],
        result_format="typed",
        error_policy="isolate",
    )
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        second = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for index in range(2):
        matrix, mask, broken, healthy = first.outputs
        np.testing.assert_array_equal(matrix.values[index], 0.0)
        np.testing.assert_array_equal(mask.values[index], False)
        assert (
            matrix.root_statuses[index]
            == mask.root_statuses[index]
            == healthy.root_statuses[index]
            == 0
        )
        assert broken.root_statuses[index] != 0 and healthy.values[index] == 7
        assert not matrix.values[index].flags.writeable
        assert not np.shares_memory(matrix.values[index], second.outputs[0].values[index])
        assert not np.shares_memory(matrix.values[index], a) and not np.shares_memory(
            matrix.values[index], b
        )
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in first.audit["native_chunks"]
    )
    assert first.plan.metadata()["estimated_worker_scratch_bytes"] >= observed


@pytest.mark.parametrize("mode", ["negative", "sliced", "broadcast"])
def test_rectangular_nonpacked_left_preserves_the_current_paths(mode):
    a, b, _ = inputs((64, 30, 252), "cc")
    if mode == "negative":
        a = a[::-1]
    elif mode == "sliced":
        owner = np.full((64, 60), np.nan)
        owner[:, ::2] = a
        a = owner[:, ::2]
    else:
        a = np.broadcast_to(a[:1], a.shape)
    expected = op.matmul(a, b, simd="scalar")
    actual, audit = op.matmul(a, b, audit=True)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert audit["isa"] != "eigen-packed-matmul" and audit["input_copy_bytes"] == 0


@pytest.mark.parametrize("ends", [(8192, 2520), (16384, 2520), (8192, 128)])
@pytest.mark.parametrize("k", [30, 128])
@pytest.mark.parametrize("order", ["C", "F"])
def test_shorter_intervals_can_enter_packing_within_the_original_plan(ends, k, order):
    # A nonlinear consumer excludes the existing MatMul summary optimization.
    types = dict(
        w=dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K"]),
        r=dict(kind="matrix", axes=["time", "factor"], shape=["T", "K"]),
    )
    y = "matmul(w,transpose(r))"
    graph = GraphCompiler(types).compile(f"sum_axis(multiply({y},{y}),1)", result_format="typed")
    data = dict(
        w=np.full((193, k), 1 / k, order=order), r=np.full((max(ends), k), 0.01, order=order)
    )
    for x in data.values():
        x.flags.writeable = False
    starts = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, data, starts, np.array(ends, np.int64))
        second = scheduler.execute(graph, data, starts, np.array(ends[::-1], np.int64))
    for result, lengths in [(first, ends), (second, ends[::-1])]:
        observed = max(
            c["numeric_arena_bytes"]
            + c["mask_arena_bytes"]
            + c["operator_workspace_capacity_bytes"]
            for c in result.audit["native_chunks"]
        )
        assert result.plan.metadata()["estimated_worker_scratch_bytes"] >= observed
        for index, length in enumerate(lengths):
            np.testing.assert_allclose(result.outputs[0].values[index], length * 0.0001, rtol=1e-12)
            assert not result.outputs[0].values[index].flags.writeable
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
