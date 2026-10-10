"""Long output lanes with 1..3 rows reuse the ordered Eigen expression."""

import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import assert_ieee_classes_and_close, declarations, inputs, settings
from .test_matrix_library_backend import PACKING_PLATFORM


@pytest.mark.parametrize(
    "shape", [(1, 1, 128), (2, 30, 129), (3, 128, 257), (1, 30, 81921), (1, 30, 200000)]
)
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff", "padded_c_b", "padded_f_b"])
def test_tiny_rows_borrow_dense_right_and_keep_scalar_results(shape, layout):
    a, b, owners = inputs(shape, layout)
    saved = [x.copy() for x in (a, b, *owners)]
    a.flags.writeable = b.flags.writeable = False
    expected = op.matmul(a, b, simd="scalar")
    workspace = op.Workspace()
    storage = np.full(expected.size + 2, 71.0)
    storage[-1] = -71.0
    out = storage[1:-1].reshape(expected.shape)
    actual, audit = op.matmul(a, b, out=out, workspace=workspace, audit=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(storage[[0, -1]], [71.0, -71.0])
    assert actual is out
    eligible = b.strides[0] == 8 and b.strides[1] >= shape[1] * 8
    if PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN") == "1" and eligible:
        assert audit["isa"] == "eigen-array-matmul"
    if not eligible:
        assert audit["isa"] != "eigen-array-matmul"
    if os.environ.get("CME_EXPECT_EIGEN") == "0":
        assert audit["isa"] != "eigen-array-matmul"
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0
    second = op.matmul(a * 2, b, workspace=workspace)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(second, expected * 2)
    assert not np.shares_memory(actual, second)
    for x, before in zip((a, b, *owners), saved, strict=True):
        np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("fault", ["cancellation", "fma", "exceptional", "subnormal"])
def test_short_rows_keep_discrete_ieee_boundaries(m, order, fault):
    a = np.zeros((m, 30))
    b = np.ones((30, 129), order=order)
    if fault == "cancellation":
        a[:, :3] = [1.0, 2**-54, -1.0]
    elif fault == "fma":
        a[:, :2] = [-1.0, 1 + 2**-27]
        b[:2] = np.array([1.0, 1 - 2**-27])[:, None]
    elif fault == "exceptional":
        a[:, :3] = [1e308, 1e308, -1e308]
        b[:3, 0] = 2.0
        b[7, 1] = np.inf
        b[8, 2] = np.nan
    else:
        a[:, :3] = [1.0, 1.0, -1.0]
        b.fill(0.0)
        b[0, :4] = [np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0), 0.0, -0.0]
    before = [a.copy(), b.copy()]
    a.flags.writeable = b.flags.writeable = False
    expected = op.matmul(a, b, simd="scalar")
    actual = op.matmul(a, b)
    assert_ieee_classes_and_close(actual, expected)
    np.testing.assert_array_equal(actual > 0, expected > 0)
    np.testing.assert_array_equal(actual < 0, expected < 0)
    if fault in ["cancellation", "fma"]:
        np.testing.assert_array_equal(actual, 0.0)
    for x, saved in zip((a, b), before, strict=True):
        np.testing.assert_array_equal(x, saved)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("layout", ["cc", "cf"])
def test_short_row_graph_masks_transports_and_old_results(lane, layout):
    a, b, _ = inputs((3, 30, 129), layout)
    a[:, :] = 0.0
    a[:, :3] = [1.0, 2**-54, -1.0]
    b[:, :] = 1.0
    a.flags.writeable = b.flags.writeable = False
    roots = ["matmul(a,b)", "greater_than(matmul(a,b),0)", "7"]
    graph = GraphCompiler(declarations()).compile(roots, result_format="typed", output_order="F")
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        snapshots = [
            x.values[0].copy() if isinstance(x.values[0], np.ndarray) else x.values[0]
            for x in first.outputs
        ]
        second = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for row in range(2):
        np.testing.assert_array_equal(first.outputs[0].values[row], 0.0)
        np.testing.assert_array_equal(first.outputs[1].values[row], False)
        assert first.outputs[2].values[row] == 7
        for i in range(2):
            old = first.outputs[i].values[row]
            new = second.outputs[i].values[row]
            np.testing.assert_array_equal(old, snapshots[i])
            assert not np.shares_memory(old, new)
            assert not old.flags.writeable and not new.flags.writeable
            assert not np.shares_memory(new, a) and not np.shares_memory(new, b)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "layout", ["negative_columns", "negative_rows", "sliced_columns", "broadcast_rows"]
)
def test_short_row_library_borrows_general_left_strides(m, order, layout):
    a, b, _ = inputs((m, 30, 129), "cf" if order == "F" else "cc")
    if layout == "negative_columns":
        a = a[:, ::-1]
    elif layout == "negative_rows":
        a = a[::-1]
    elif layout == "sliced_columns":
        owner = np.full((m, 60), np.nan)
        owner[:, ::2] = a
        a = owner[:, ::2]
    else:
        a = np.broadcast_to(a[:1], a.shape)
    saved_a = a.copy()
    saved_b = b.copy()
    a.flags.writeable = b.flags.writeable = False
    expected = op.matmul(a, b, simd="scalar")
    actual, audit = op.matmul(a, b, audit=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(a, saved_a)
    np.testing.assert_array_equal(b, saved_b)
    assert audit["input_copy_bytes"] == 0 and audit["workspace_capacity_bytes"] == 0
    assert not np.shares_memory(actual, a) and not np.shares_memory(actual, b)
