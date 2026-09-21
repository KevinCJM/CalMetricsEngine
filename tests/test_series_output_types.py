"""Exact public time-series dtypes, validity and native transport ownership."""

import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import (
    AdaptiveScheduler,
    GraphCompileError,
    GraphCompiler,
    PlannerConfig,
    SharedInputBundle,
)


def declaration(dtype):
    return {"kind": "series", "dtype": dtype, "axes": ["time"], "shape": ["T"]}


def config(lane):
    return PlannerConfig(
        thread_work_units=1e99 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )


def sample(dtype):
    if dtype == "bool":
        return np.array([False, True, False, True, True, False, True, False])
    return np.array([0, 2**53 + 1, 2**53 + 2, -(2**63),
                     2**63 - 1, -1, 0, 2**53 + 1], dtype=np.int64)


@pytest.mark.parametrize("dtype", ["bool", "int64"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_exact_series_dtype_across_native_lanes(dtype, lane):
    # Two root columns plus uneven/empty intervals exercise byte offsets, not
    # merely contiguous single-column copies. Negative-stride inputs stay exact.
    values = sample(dtype)[::-1]
    expected_source = values.copy()
    values.setflags(write=False)
    graph = GraphCompiler({"x": declaration(dtype)}).compile(["x", "x"])
    starts, ends = np.array([0, 3, 3], np.int64), np.array([3, 3, 8], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(graph, {"x": values}, starts, ends)
        # Ordinary results must remain independent of later output allocations.
        later = scheduler.execute(graph, {"x": values}, starts, ends)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    assert result.values.dtype == np.dtype(dtype)
    assert result.values.flags.c_contiguous
    assert not np.shares_memory(result.values, values)
    assert not np.shares_memory(result.values, later.values)
    np.testing.assert_array_equal(result.values, np.column_stack([expected_source] * 2))
    np.testing.assert_array_equal(result.offsets, [0, 3, 3, 8])
    assert result.offsets.dtype == np.int64
    assert result.statuses is None
    assert result.audit["output_dtype"] == dtype
    assert graph.metadata()["output_dtype"] == dtype
    assert graph._program.metadata["output_dtype"] == dtype
    assert result.plan.estimated_output_bytes == result.values.nbytes
    assert result.audit["result_lifetime"] == "independent"
    assert result.audit["python_fallback"] == 0
    if lane == "process_shared":
        assert result.audit["output_ownership"] == "native_shared_region"
        assert result.audit["output_copy_bytes"] == 0
    np.testing.assert_array_equal(values, expected_source)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_bool_root_failure_is_distinct_from_valid_false(lane):
    graph = GraphCompiler({"x": "series"}).compile(
        ["greater_than(divide(x,std(x,1)),0)", "greater_than(x,2)"],
        error_policy="isolate",
    )
    x = np.array([1., 1., 1., 2., 3., 4.])
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(graph, {"x": x},
                                   np.array([0, 3], np.int64), np.array([3, 6], np.int64))
    assert result.values.dtype == np.bool_
    np.testing.assert_array_equal(result.values, [
        [False, False], [False, False], [False, False],
        [True, False], [True, True], [True, True],
    ])
    np.testing.assert_array_equal(result.statuses, [[2, 0]] * 3 + [[0, 0]] * 3)
    assert result.plan.metadata()["estimated_status_bytes"] == result.statuses.nbytes
    assert not result.statuses.flags.writeable


@pytest.mark.parametrize("dtype", ["bool", "int64"])
@pytest.mark.parametrize("lane", ["single", "process_shared"])
def test_typed_minimum_sample_status_survives_transport(dtype, lane):
    values = sample(dtype)
    graph = GraphCompiler({"x": declaration(dtype)}).compile(
        ["x"], error_policy="isolate", minimum_observations=2)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(graph, {"x": values},
                                   np.array([0, 0], np.int64), np.array([1, 3], np.int64))
    np.testing.assert_array_equal(result.statuses[:, 0], [1, 0, 0, 0])
    assert result.values[0, 0] == 0
    assert result.values[1, 0] == 0  # Equal bits, different validity.
    np.testing.assert_array_equal(result.values[1:, 0], values[:3])


@pytest.mark.parametrize("dtype", ["bool", "int64"])
@pytest.mark.filterwarnings(
    "ignore:Setting the dtype on a NumPy array has been deprecated:DeprecationWarning"
)
def test_prepared_typed_output_reuse_snapshot_and_dtype_guard(dtype):
    values = sample(dtype)
    original = values.copy()
    graph = GraphCompiler({"x": declaration(dtype)}).compile(["x"])
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(
            graph, {"x": values}, np.array([0], np.int64), np.array([8], np.int64))
        first = prepared.run()
        pointer = first.ctypes.data
        retained = prepared.run_snapshot()
        values[:] = 0
        second = prepared.run()
        assert first is second and second.ctypes.data == pointer
        assert not np.shares_memory(second, retained.values)
        assert retained.values.dtype == np.dtype(dtype)
        assert not retained.values.flags.writeable
        np.testing.assert_array_equal(second, 0)
        np.testing.assert_array_equal(retained.values[:, 0], original)
        assert prepared.run_audit().audit["result_lifetime"] == "borrowed_until_next_run"
        assert retained.audit["result_lifetime"] == "independent"
        # Mutate the bound ndarray itself to exercise the native guard. A new
        # view would not test this hazard; NumPy 2.5 warns about this setter.
        prepared.output.dtype = np.uint8 if dtype == "bool" else np.uint64
        with pytest.raises(ValueError, match="PREPARED_OUTPUT_CHANGED"):
            prepared.run()
    np.testing.assert_array_equal(retained.values[:, 0], original)


@pytest.mark.parametrize("dtype", ["bool", "int64"])
def test_prepared_bare_typed_array_cannot_hide_failed_status(dtype):
    graph = GraphCompiler({"x": declaration(dtype)}).compile(
        "x", error_policy="isolate", minimum_observations=2)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(
            graph, {"x": sample(dtype)}, np.array([0], np.int64), np.array([1], np.int64))
        with pytest.raises(ValueError, match="TYPED_OUTPUT_STATUS_REQUIRED"):
            prepared.run()
        result = prepared.run_audit()
        np.testing.assert_array_equal(result.statuses, [[1]])
        assert result.values.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(prepared.run_snapshot().statuses, [[1]])
    out = np.empty((1, 1), dtype=dtype)
    audit = graph._program.execute((sample(dtype),), np.array([0], np.int64),
                                   np.array([1], np.int64), out)
    np.testing.assert_array_equal(audit["statuses"], [[1]])
    assert audit["output_dtype"] == dtype


@pytest.mark.parametrize("dtype", ["bool", "int64"])
def test_typed_raw_output_contract_and_pickle(dtype):
    graph = GraphCompiler({"x": declaration(dtype)}).compile("x")
    restored = pickle.loads(pickle.dumps(graph))
    assert restored.fingerprint == graph.fingerprint
    assert restored.metadata()["output_dtype"] == dtype
    x = sample(dtype)
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    out = np.empty((8, 1), dtype=dtype)
    restored._program.execute((x,), starts, ends, out)
    np.testing.assert_array_equal(out[:, 0], x)
    with pytest.raises(TypeError):
        restored._program.execute((x,), starts, ends, np.empty((8, 1), np.float64))
    with pytest.raises(ValueError, match="aliases"):
        restored._program.execute((x,), starts, ends, x.reshape(-1, 1))


@pytest.mark.parametrize("dtype", ["bool", "int64"])
def test_shared_typed_results_retain_owner_after_release(dtype):
    values = sample(dtype)
    graph = GraphCompiler({"x": declaration(dtype)}).compile("x")
    bundle = SharedInputBundle.from_inputs({"x": values})
    with AdaptiveScheduler(cpu_budget=2, config=config("process_shared")) as scheduler:
        result = scheduler.execute(graph, bundle,
                                   np.array([0, 4], np.int64), np.array([4, 8], np.int64))
        bundle.release()
    gc.collect()
    np.testing.assert_array_equal(result.values[:, 0], values)
    assert result.audit["output_ownership"] == "native_shared_region"
    assert result.audit["output_copy_bytes"] == 0


@pytest.mark.parametrize("dtype", ["bool", "int64"])
def test_empty_typed_series_preserves_dtype_and_offsets(dtype):
    graph = GraphCompiler({"x": declaration(dtype)}).compile("x")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": np.empty(0, dtype)},
                                   np.array([0, 0], np.int64), np.array([0, 0], np.int64))
    assert result.values.shape == (0, 1)
    assert result.values.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result.offsets, [0, 0, 0])


@pytest.mark.parametrize("dtype", ["bool", "int64"])
@pytest.mark.parametrize("shared", [False, True])
def test_empty_worker_output_uses_zero_payload_without_null_buffer(dtype, shared):
    graph = GraphCompiler({"x": declaration(dtype)}).compile("x")
    settings = PlannerConfig(process_input_threshold_bytes=1, min_rows_per_worker=1,
                             shared_memory_threshold_bytes=1 if shared else 1 << 30)
    with AdaptiveScheduler(cpu_budget=2, config=settings) as scheduler:
        result = scheduler.execute(graph, {"x": sample(dtype)},
                                   np.array([0, 3], np.int64), np.array([0, 3], np.int64))
    assert result.plan.lane == "process"
    assert result.plan.use_shared_memory == shared
    assert result.values.shape == (0, 1)
    assert result.values.dtype == np.dtype(dtype)
    assert result.plan.estimated_output_bytes == 0
    np.testing.assert_array_equal(result.offsets, [0, 0, 0])


@pytest.mark.parametrize("dtype", ["bool", "int64"])
def test_hard_stop_worker_returns_exact_typed_series(dtype):
    values = sample(dtype)
    graph = GraphCompiler({"x": declaration(dtype)}).compile("x")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": values},
                                   np.array([0], np.int64), np.array([8], np.int64),
                                   hard_stop=True, timeout=10)
    assert result.plan.hard_stop and result.plan.lane == "process"
    assert result.values.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result.values[:, 0], values)


def test_mixed_output_dtypes_and_unaligned_integer_vectors_fail_closed():
    with pytest.raises(GraphCompileError, match="MIXED_ROOT_DTYPES"):
        GraphCompiler({"x": "series"}).compile(["x", "greater_than(x,0)"])
    with pytest.raises(GraphCompileError, match="PUBLIC_ROOT_TYPE"):
        GraphCompiler({"x": "series"}).compile("argsort(x)")
    with pytest.raises(GraphCompileError, match="PUBLIC_ROOT_TYPE"):
        GraphCompiler({"x": {"kind": "vector", "dtype": "int64"}}).compile("x")


def test_state_event_planner_accounts_quadratic_and_window_work():
    events = {**declaration("int64"), "semantic_dimension": "event"}
    compiler = GraphCompiler({"x": "series", "events": events})
    ps = compiler.compile("ps_filter(x,events,2,4,0.1)")
    small_window = compiler.compile("local_extrema(x,1,1,0,0)")
    large_window = compiler.compile("local_extrema(x,20,30,0,0)")
    starts = np.array([0], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        work = []
        for length in [64, 128]:
            inputs = {"x": np.arange(1., length + 1), "events": np.zeros(length, np.int64)}
            ends = np.array([length], np.int64)
            plan = scheduler.plan(ps, inputs, starts, ends)
            work.append(plan.estimated_work_units)
            assert plan.estimated_work_units >= length * length
            assert plan.estimated_worker_scratch_bytes >= length * np.dtype(np.int64).itemsize
        assert work[1] > 3 * work[0]
        small = scheduler.plan(small_window, inputs, starts, ends)
        large = scheduler.plan(large_window, inputs, starts, ends)
        assert large.estimated_work_units > 10 * small.estimated_work_units
