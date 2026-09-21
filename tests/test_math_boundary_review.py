"""Independent regression cases for newly enabled native matrix boundaries."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig


def config(lane):
    return PlannerConfig(
        thread_work_units=1e100 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_borrowed_transpose_keeps_backing_arena_until_last_consumer(lane):
    graph = GraphCompiler({"x": {"kind": "matrix"}}).compile([
        "sum(transpose(x+1)+transpose(x+2))",
        "sum(transpose(transpose(x+3))+x+4)",
    ])
    x = np.arange(1.0, 19.0).reshape(6, 3)
    starts, ends = np.array([0, 3], np.int64), np.array([3, 6], np.int64)
    expected = [[np.sum((x[a:z] + 1).T + (x[a:z] + 2).T),
                 np.sum((x[a:z] + 3).T.T + x[a:z] + 4)]
                for a, z in zip(starts, ends, strict=True)]
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        actual = scheduler.execute(graph, {"x": x}, starts, ends)
    np.testing.assert_allclose(actual.values, expected, rtol=0, atol=0)
    assert actual.audit["lane"] == ("process" if lane.startswith("process") else lane)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_borrowed_diagonal_keeps_backing_arena_until_last_consumer(lane):
    graph = GraphCompiler({
        "x": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["N", "N"]},
    }).compile([
        "sum(diag(x+1)+diag(x+2))",
        "sum(diag(transpose(x+3))+diag(x+4))",
    ])
    x = np.arange(1.0, 10.0).reshape(3, 3)
    starts, ends = np.array([0, 3], np.int64), np.array([3, 6], np.int64)
    expected = [[np.sum(np.diag(x + 1) + np.diag(x + 2)),
                 np.sum(np.diag((x + 3).T) + np.diag(x + 4))]] * 2
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        actual = scheduler.execute(graph, {"x": x}, starts, ends)
    np.testing.assert_allclose(actual.values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_static_vector_matrix_growth_is_planned_before_execution(lane):
    graph = GraphCompiler({"v": {"kind": "vector", "shape": ["N"]}}).compile([
        "sum(outer(v,v))", "sum(diag(v))", "sum(matvec(outer(v,v),v))",
    ])
    v = np.array([1.0, 2.0, 3.0])
    starts, ends = np.array([0, 1], np.int64), np.array([1, 2], np.int64)
    expected = [[np.outer(v, v).sum(), np.diag(v).sum(), (np.outer(v, v) @ v).sum()]] * 2
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(graph, {"v": v}, starts, ends)
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=0)
    assert result.plan.estimated_worker_scratch_bytes >= v.size**2 * 8


@pytest.mark.parametrize("prepared", [True, False])
def test_resized_underlying_owner_invalidates_pinned_matrix_view(prepared):
    graph = GraphCompiler({"x": {"kind": "matrix"}}).compile("mean(sum_asset(x))")
    owner = np.arange(16.0)
    view = owner.reshape(4, 4)
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if prepared:
            call = scheduler.prepare_execution(graph, {"x": view}, starts, ends).run
        else:
            # The ordinary scheduler caches this view binding as well.
            scheduler.execute(graph, {"x": view}, starts, ends)

            def call():
                return scheduler.execute(graph, {"x": view}, starts, ends)
        owner.resize((8,), refcheck=False)
        # View shape, strides, dtype and pointer remain unchanged. Its owner no
        # longer owns the entire input span and must be checked before execution.
        with pytest.raises(ValueError, match="PREPARED_INPUT_CHANGED|INPUT_OUT_OF_BOUNDS"):
            call()
