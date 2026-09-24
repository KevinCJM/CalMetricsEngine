"""Rank-3 protocol through the installed native compiler, planner and workers."""
import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler, PlannerConfig
from calmetrics_engine import operators as op


def tensor(dtype="float64", temporal=True):
    return {"kind": "tensor", "dtype": dtype,
            "axes": ["time" if temporal else "scenario", "path", "asset"],
            "shape": ["T" if temporal else "S", "P", "N"]}


def configuration(lane):
    return PlannerConfig(thread_work_units=1 if lane == "thread" else 1e100,
                         process_work_units=1 if lane.startswith("process") else 1e100,
                         shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
                         min_rows_per_worker=1, max_processes=2)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("dtype", ["float64", "int64", "bool"])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "broadcast", "empty"])
@pytest.mark.parametrize("temporal", [True, False])
def test_tensor_identity_exact_geometry_and_ownership(lane, dtype, layout, temporal):
    base = np.arange(96).reshape(8, 4, 3).astype(dtype)
    if dtype == "int64":
        base += 2**53
    x = base
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "negative":
        x = x[::-1, ::-2, ::-1]
    elif layout == "broadcast":
        x = np.broadcast_to(x[:, :1, :], (8, 4, 3))
    elif layout == "empty":
        x = x[:, :0, :]
    x.flags.writeable = False
    g = GraphCompiler({"x": tensor(dtype, temporal)}).compile("x", error_policy="isolate")
    g = pickle.loads(pickle.dumps(g))
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as scheduler:
        result = scheduler.execute(g, {"x": x}, np.array([0, 2], np.int64), np.array([2, 6], np.int64))
    assert result.plan.lane == ("single" if not x.size else "process" if lane.startswith("process") else lane)
    for row, expected in enumerate([x[:2], x[2:6]] if temporal else [x, x]):
        actual = result.outputs[0].values[row]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable
        assert not np.shares_memory(actual, base)
        assert np.all(result.outputs[0].statuses[row] == 0)
    view = result.outputs[0].values[1]
    reference = view.copy()
    del result, x, base
    gc.collect()
    np.testing.assert_array_equal(view, reference)


@pytest.mark.parametrize("stride", [1, 2, -1])
def test_tensor_math_and_mixed_outputs(stride):
    x = np.arange(120., dtype=np.float64).reshape(10, 4, 3)[::stride, ::-1]
    g = GraphCompiler({"x": tensor()}).compile(
        ["x*2+1", "x>10", "sum(x)", "count_true(x>10)", "finite_mask(x)"], error_policy="isolate")
    starts, ends = np.array([0], np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(g, {"x": x}, starts, ends)
        prepared = scheduler.prepare_execution(g, {"x": x}, starts, ends)
        snap = prepared.run_snapshot()
        x[:] = 0
        next_result = prepared.run()
        assert np.all(next_result.outputs[0].values[0] == 1)
    expected = [np.arange(120.).reshape(10, 4, 3)[::stride, ::-1]]
    expected = [expected[0] * 2 + 1, expected[0] > 10, expected[0].sum(),
                np.count_nonzero(expected[0] > 10), np.ones(x.shape, bool)]
    for index, wanted in enumerate(expected):
        np.testing.assert_array_equal(result.outputs[index].values[0], wanted)
        np.testing.assert_array_equal(snap.outputs[index].values[0], wanted)


def test_tensor_direct_math_keeps_matrix_and_sequence_rank_constraints():
    x = np.arange(24.).reshape(2, 3, 4)[:, ::-1]
    np.testing.assert_array_equal(op.add(x, 2), x + 2)
    np.testing.assert_array_equal(op.finite_mask(x), np.isfinite(x))
    assert op.sum(x) == x.sum()
    for function in [op.transpose, op.diag, op.cumulative_sum, op.sum_time]:
        with pytest.raises((ValueError, RuntimeError), match="RANK"):
            function(x)


def test_tensor_bad_declarations_and_shapes_fail_closed():
    for axes, shape in [(["path", "time", "asset"], ["P", "T", "N"]),
                        (["time", "asset"], ["T", "N"])]:
        with pytest.raises(GraphCompileError, match="AXIS|RANK"):
            GraphCompiler({"x": {"kind": "tensor", "axes": axes, "shape": shape}})
    g = GraphCompiler({"x": tensor(), "y": tensor()}).compile("x+y")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match="symbolic"):
            scheduler.execute(g, {"x": np.zeros((2, 3, 4)), "y": np.zeros((2, 2, 4))},
                              np.array([0], np.int64), np.array([2], np.int64))


@pytest.mark.parametrize("bad", [2, 128, 255])
@pytest.mark.parametrize("shape", [(17,), (3, 4), (2, 3, 4)])
def test_strict_contiguous_boolean_roots_validate_every_byte(shape, bad):
    spec = {1: {"kind": "vector", "axes": ["asset"], "shape": ["N"]},
            2: {"kind": "matrix", "axes": ["model", "asset"], "shape": ["M", "N"]},
            3: tensor(temporal=False)}[len(shape)] | {"dtype": "bool"}
    x = np.zeros(shape, np.uint8)
    x.flat[-1] = bad
    graph = GraphCompiler({"x": spec}).compile({"mask": "x"})
    with AdaptiveScheduler(cpu_budget=1) as engine:
        with pytest.raises((ValueError, RuntimeError), match="INVALID_MASK"):
            engine.execute(graph, {"x": x}, np.array([0], np.int64), np.array([1], np.int64))
