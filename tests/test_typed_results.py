"""Real AOT typed-result contracts, including native thread/process transport."""
import gc
import pickle
import warnings

import numpy as np
import pytest

from calmetrics_engine import (
    AdaptiveScheduler,
    GraphCompileError,
    GraphCompiler,
    PlannerConfig,
    SharedInputBundle,
)


def decl(kind, dtype="float64", **extra):
    return {"kind": kind, "dtype": dtype, **extra}


def config(lane):
    return PlannerConfig(thread_work_units=1e100 if lane == "single" else 1,
                         process_work_units=1 if lane.startswith("process") else 1e100,
                         shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
                         min_rows_per_worker=1, max_processes=2)


def execute(graph, inputs, starts=(0,), ends=None, lane="single", **kw):
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        return scheduler.execute(graph, inputs, np.asarray(starts, dtype=np.int64),
                                 np.asarray(ends if ends is not None else [len(next(iter(inputs.values())))], dtype=np.int64), **kw)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("stride", [1, 2, -1])
@pytest.mark.parametrize("isolate", [False, True])
def test_mixed_shapes_dtypes_share_one_native_graph(lane, stride, isolate):
    source = np.random.default_rng(317).normal(size=(32, 6))
    x = source[::stride, ::2]
    x.flags.writeable = False
    ids = np.array([2**53 + 1, -(2**63), 2**63 - 1], dtype=np.int64)
    graph = GraphCompiler({"x": decl("matrix"), "ids": decl("vector", "int64")}).compile(
        ["mean(sum_asset(x))", "mean_time(x)", "covariance(x)", "transpose(x)",
         "greater_than(sum_asset(x),0)", "argsort(sum_asset(x))", "ids", "mean_time(x)"],
        error_policy="isolate" if isolate else "raise")
    starts, ends = [0, 3, 4], [4, 9, 11]
    result = execute(graph, {"x": x, "ids": ids}, starts, ends, lane)
    assert result.output_kind == "typed"
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    outputs = result.outputs
    for row, (a, z) in enumerate(zip(starts, ends, strict=True)):
        m = x[a:z]
        sums = m.sum(axis=1)
        expected = [sums.mean(), m.mean(axis=0), np.cov(m, rowvar=False), m.T,
                    sums > 0, np.argsort(sums, kind="stable"), ids, m.mean(axis=0)]
        for out, value in zip(outputs, expected, strict=True):
            actual = out.values[row]
            assert actual.shape == np.asarray(value).shape
            assert actual.dtype == np.asarray(value).dtype
            assert not actual.flags.writeable
            assert not np.shares_memory(actual, source)
            np.testing.assert_allclose(actual, value, rtol=1e-13, atol=1e-14)
            if actual.dtype == np.int64:
                np.testing.assert_array_equal(actual, value)
            assert out.shape_known[row]
            assert out.root_statuses[row] == 0
            if isolate:
                np.testing.assert_array_equal(out.statuses[row], np.zeros(actual.shape, np.int16))
            else:
                assert out.statuses[row] is None
    assert result.audit["audit_schema"] == "cpp-aot-execution-2"
    assert result.audit["result_protocol"] == "typed-results-1"
    assert result.audit["python_operator_calls"] == 0
    if lane == "process_shared":
        assert result.audit["output_copy_bytes"] == 0
    with pytest.raises(ValueError, match="outputs"):
        _ = result.values


@pytest.mark.parametrize("dtype", ["float64", "bool", "int64"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_matrix_identity_and_exact_scalar_dtype(dtype, lane):
    x = np.arange(12).reshape(4, 3).astype(dtype)
    d = decl("matrix", dtype)
    if dtype == "bool":
        d["semantic_dimension"] = "mask"
    g = GraphCompiler({"x": d}).compile("x")
    result = execute(g, {"x": x}, [0, 2], [2, 4], lane)
    np.testing.assert_array_equal(result.outputs[0].values[0], x[:2])
    np.testing.assert_array_equal(result.outputs[0].values[1], x[2:])
    assert result.outputs[0].dtype == dtype


def test_explicit_typed_scalar_boolean_and_pickle_preserves_operator_dtype():
    g = GraphCompiler({"x": decl("series", "int64")}).compile(["distinct_count(x)", "greater_than(1,0)"], result_format="typed")
    clone = pickle.loads(pickle.dumps(g))
    assert clone.fingerprint == g.fingerprint
    x = np.array([2**53 + 1, 2**53 + 2], np.int64)
    result = execute(clone, {"x": x})
    # distinct_count's existing mathematical contract returns a float64 count.
    assert result.outputs[0].values[0].dtype == np.float64
    assert result.outputs[0].values[0].shape == ()
    assert result.outputs[0].values[0] == 2
    assert result.outputs[1].values[0].dtype == np.bool_
    assert result.outputs[1].values[0]
    auto = GraphCompiler({"x": "series"}).compile("mean(x)")
    typed = GraphCompiler({"x": "series"}).compile("mean(x)", result_format="typed")
    assert auto.fingerprint != typed.fingerprint
    assert execute(auto, {"x": np.ones(2)}).values.shape == (1, 1)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_dynamic_lengths_and_isolated_failures(lane):
    g = GraphCompiler({"x": "series", "k": "scalar"}).compile(
        ["lag(x,k)", "block_apply(mean(x),2)", "divide(x,0)", "mean(x)"], error_policy="isolate")
    x = np.arange(1., 9.)
    result = execute(g, {"x": x}, [0, 2], [6, 8], lane, parameters={"k": 2.})
    for row, start in enumerate([0, 2]):
        np.testing.assert_array_equal(result.outputs[0].values[row], x[start:start + 4])
        assert result.outputs[1].values[row].shape == (3,)
        assert result.outputs[2].root_statuses[row] != 0
        assert np.isnan(result.outputs[2].values[row]).all()
        assert result.outputs[3].values[row] == x[start:start + 6].mean()
        assert result.outputs[3].root_statuses[row] == 0


def test_prepared_borrows_but_snapshots_own_values_and_dynamic_shapes():
    g = GraphCompiler({"x": "series", "k": "scalar"}).compile(["lag(x,k)", "mean(x)"], error_policy="isolate")
    x = np.arange(1., 9.)
    parameters = np.array([2.])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(g, {"x": x}, np.array([0], np.int64), np.array([8], np.int64), parameters=parameters)
        first = prepared.run()
        borrowed = first.outputs[0].values[0]
        snap = prepared.run_snapshot()
        stable = snap.outputs[0].values[0]
        x += 100
        parameters[0] = 4
        second = prepared.run_audit()
        assert second.outputs[0].values[0].shape == (4,)
        assert np.shares_memory(borrowed, second.outputs[0].values[0])
        assert not np.shares_memory(stable, borrowed)
        np.testing.assert_array_equal(stable, np.arange(1., 7.))
        assert snap.audit["result_lifetime"] == "independent"
        assert first.audit["result_lifetime"] == "borrowed_until_next_run"
    del prepared, first, second, snap
    gc.collect()
    np.testing.assert_array_equal(stable, np.arange(1., 7.))


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_empty_shapes_and_failed_root_without_geometry(lane):
    x = np.empty((4, 0), dtype=np.float64)
    g = GraphCompiler({"x": decl("matrix")}).compile(["x", "transpose(x)", "mean_time(x)"], error_policy="isolate")
    result = execute(g, {"x": x}, [0, 1], [0, 4], lane)
    assert result.outputs[0].values[0].shape == (0, 0)
    assert result.outputs[0].values[1].shape == (3, 0)
    assert result.outputs[1].values[1].shape == (0, 3)
    g = GraphCompiler({"x": "series"}).compile(["x", "mean(x)"], error_policy="isolate", minimum_observations=2)
    result = execute(g, {"x": np.array([1.])}, [0, 0], [0, 1], lane)
    for out in result.outputs:
        assert out.shape_known == (False, False)
        np.testing.assert_array_equal(out.root_statuses, [1, 1])
        assert all(value.size == 0 for value in out.values)


def test_shared_result_view_outlives_bundle_result_and_scheduler():
    g = GraphCompiler({"x": decl("matrix")}).compile(["x", "mean_time(x)"])
    x = np.arange(24.).reshape(8, 3)
    with SharedInputBundle.from_inputs({"x": x}) as bundle:
        with AdaptiveScheduler(cpu_budget=2, config=config("process_shared")) as scheduler:
            result = scheduler.execute(g, bundle, np.array([0, 4], np.int64), np.array([4, 8], np.int64))
            view = result.outputs[0].values[1]
    del result, bundle, scheduler
    gc.collect()
    np.testing.assert_array_equal(view, x[4:])


def test_typed_memory_admission_and_stale_plan():
    g = GraphCompiler({"x": decl("matrix")}).compile(["covariance(x)", "x"])
    x = np.arange(48.).reshape(8, 6)
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(g, {"x": x}, starts, ends)
        assert plan.estimated_output_bytes == (36 + 48) * 8
        assert plan.metadata()["estimated_result_metadata_bytes"] > 0
        with pytest.raises(MemoryError):
            scheduler.execute(g, {"x": x}, starts, ends, memory_budget_bytes=plan.estimated_output_bytes - 1)
        with pytest.raises(ValueError, match="stale|geometry"):
            scheduler.execute(g, {"x": x}, starts, np.array([7], np.int64), plan=plan)


def test_hard_stop_typed_and_internal_record_rejection():
    g = GraphCompiler({"x": "series"}).compile(["x", "mean(x)"])
    result = execute(g, {"x": np.arange(4.)}, hard_stop=True, timeout=10)
    assert result.plan.lane == "process"
    np.testing.assert_array_equal(result.outputs[0].values[0], np.arange(4.))
    for expression in ["linear_fit(x)", "rolling_window(x,2)", "scalar_kalman(x,1,1)"]:
        with pytest.raises(GraphCompileError, match="PUBLIC_ROOT_TYPE"):
            GraphCompiler({"x": "series"}).compile(expression, result_format="typed")
    with pytest.raises(GraphCompileError, match="result_format"):
        GraphCompiler({"x": "series"}).compile("x", result_format="coerce")


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_position_errors_stay_with_mixed_dtype_outputs(lane):
    events = decl("series", "int64", semantic_dimension="event")
    segment = "segment_apply(1/std(x,0),between_events(e))"
    g = GraphCompiler({"x": "series", "e": events}).compile(
        [segment, segment + ">0", "state_select(" + segment + ">0,1,2,finite_mask(x))",
         "finite_mask(x)", "mean_where(x,finite_mask(x))"], error_policy="isolate")
    x = np.array([2., 2., 2., np.nan, 2., 3., 4., 5.])
    e = np.array([-1, 0, 1, -2, -1, 0, 1, 0], np.int64)
    result = execute(g, {"x": x, "e": e}, [0, 0], [8, 8], lane)
    for out in result.outputs[:3]:
        for row in range(2):
            assert np.all(out.statuses[row][:2] != 0)
            assert np.all(out.statuses[row][4:6] == 0)
            assert out.root_statuses[row] != 0
    for row in range(2):
        np.testing.assert_array_equal(result.outputs[3].values[row], np.isfinite(x))
        assert np.all(result.outputs[3].statuses[row] == 0)
        assert result.outputs[4].root_statuses[row] == 0
        assert result.outputs[4].values[row] == np.nanmean(x)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_unknown_dynamic_shape_preserves_status_and_healthy_root(lane):
    g = GraphCompiler({"x": "series"}).compile(
        ["lag(x,1/std(x,0))", "greater_than(lag(x,1/std(x,0)),0)", "x"], error_policy="isolate")
    result = execute(g, {"x": np.ones(4)}, [0, 0], [4, 4], lane)
    for out in result.outputs[:2]:
        assert out.shape_known == (False, False)
        assert np.all(out.root_statuses != 0)
        assert all(v.size == 0 for v in out.values)
    for row in range(2):
        np.testing.assert_array_equal(result.outputs[2].values[row], np.ones(4))
        assert result.outputs[2].root_statuses[row] == 0


@pytest.mark.parametrize("mutation", ["shape", "dtype", "readonly"])
def test_prepared_payload_tampering_fails_closed(mutation):
    g = GraphCompiler({"x": "series"}).compile(["x", "mean(x)"])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        p = scheduler.prepare_execution(g, {"x": np.arange(4.)}, np.array([0], np.int64), np.array([4], np.int64))
        output = p.run().outputs[0].values[0]
        owner = output.base
        assert isinstance(owner, np.ndarray)
        # Deliberately mutate the SAME owner: reshape/view would leave the
        # prepared buffer unchanged and would not test its admission guard.
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings("always", category=DeprecationWarning,
                                    message="Setting the (shape|dtype) on a NumPy array has been deprecated")
            if mutation == "shape":
                owner.shape = (1, owner.size)
            elif mutation == "dtype":
                owner.dtype = np.float64
            else:
                owner.flags.writeable = False
        assert len(caught) <= 1
        for warning in caught:
            assert warning.category is DeprecationWarning
            assert str(warning.message).startswith(f"Setting the {mutation} on a NumPy array has been deprecated")
        with pytest.raises(ValueError, match="PREPARED_OUTPUT_CHANGED"):
            p.run()


def test_typed_async_and_concurrent_calls_own_results():
    import asyncio
    g = GraphCompiler({"x": decl("matrix")}).compile(["x", "mean_time(x)"])
    x = np.arange(12.).reshape(4, 3)
    async def run():
        with AdaptiveScheduler(cpu_budget=2) as scheduler:
            job = {"graph": g, "inputs": {"x": x}, "starts": np.array([0], np.int64), "ends": np.array([4], np.int64)}
            return await scheduler.execute_many_async([job, job, job])
    results = asyncio.run(run())
    for r in results:
        np.testing.assert_array_equal(r.outputs[0].values[0], x)
    assert not np.shares_memory(results[0].outputs[0].values[0], results[1].outputs[0].values[0])


def test_typed_zero_intervals_preserve_output_schemas():
    g = GraphCompiler({"x": decl("matrix")}).compile(["x", "mean_time(x)"])
    r = execute(g, {"x": np.ones((4,3))}, [], [])
    assert len(r.outputs) == 2
    assert all(out.values == () and out.shape_known == () for out in r.outputs)
    assert r.outputs[0].axes == ["time", "asset"]


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("typed", [False, True])
def test_full_existing_status_vocabulary_crosses_transport(lane, typed):
    g = GraphCompiler({"x": "series", "dd": "series"}).compile(
        ["value_at(x,-1)", "interval_start(last_drawdown_interval(x))",
         "interval_recovery(last_drawdown_interval(dd))"], error_policy="isolate",
        result_format="typed" if typed else "auto")
    r = execute(g, {"x": np.zeros(3), "dd": np.array([0., -.1, -.2])}, [0, 0], [3, 3], lane)
    if typed:
        for out, status in zip(r.outputs, [7, 9, 10], strict=True):
            np.testing.assert_array_equal(out.root_statuses, [status, status])
            assert all(value.item() == status for value in out.statuses)
    else:
        np.testing.assert_array_equal(r.statuses, [[7, 9, 10], [7, 9, 10]])


def test_typed_worker_timeout_recovers_and_does_not_invalidate_old_output():
    g = GraphCompiler({"x": decl("matrix")}).compile(["x", "mean_time(x)"])
    x = np.arange(300.).reshape(100,3)
    starts, ends = np.array([0], np.int64), np.array([100], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(g, {"x": x}, starts, ends, hard_stop=True, timeout=10)
        saved = result.outputs[0].values[0]
        with pytest.raises(TimeoutError):
            scheduler.execute(g, {"x": x}, starts, ends, hard_stop=True, timeout=1e-9)
        recovered = scheduler.execute(g, {"x": x}, starts, ends, hard_stop=True, timeout=10)
        np.testing.assert_array_equal(recovered.outputs[0].values[0], x)
    np.testing.assert_array_equal(saved, x)


def test_metadata_budget_rejects_large_root_interval_product():
    g = GraphCompiler({"x": "series"}).compile(["x"] * 128, result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(MemoryError):
            scheduler.plan(g, {"x": np.ones(2)}, np.zeros(1000, np.int64), np.ones(1000, np.int64), memory_budget_bytes=1024)


def test_inline_transport_budget_includes_large_typed_payload_copies():
    g = GraphCompiler({"x": decl("matrix")}).compile("x")
    x = np.ones((2000,8))
    starts, ends = np.array([0,0], np.int64), np.array([2000,2000], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config("process_inline")) as scheduler:
        plan = scheduler.plan(g, {"x": x}, starts, ends)
        assert plan.lane == "process" and not plan.use_shared_memory
        assert plan.estimated_total_memory_bytes >= plan.estimated_input_bytes + 5 * plan.estimated_output_bytes
        with pytest.raises(MemoryError):
            scheduler.plan(g, {"x": x}, starts, ends, memory_budget_bytes=4*plan.estimated_output_bytes)


@pytest.mark.parametrize("isolate", [False, True])
@pytest.mark.parametrize("width,workers,shared", [(3600, 1, False), (4000, 1, False), (4096, 1, True),
                                                (4100, 1, True), (4100, 2, False)])
def test_small_input_large_result_transport_accounts_for_complete_chunk(width, workers, shared, isolate):
    shared = shared or (isolate and width == 4000)
    # Planning checks >256 MiB capacities without allocating the matrix results.
    g = GraphCompiler({"x": decl("vector")}).compile("outer(x,x)",
            error_policy="isolate" if isolate else "raise")
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=workers, config=config("process_inline")) as scheduler:
        plan = scheduler.plan(g, {"x": np.ones(width)}, bounds, bounds)
        assert plan.lane == "process"
        assert plan.use_shared_memory == shared
        assert plan.estimated_input_bytes == width * 8
        assert plan.estimated_output_bytes == 2 * width * width * 8
        assert ("process_result_exceeds_ipc_frame_limit" in plan.reason_codes) == shared


def test_reducing_workers_rechecks_frame_size_and_shared_memory_budget():
    g = GraphCompiler({"x": decl("vector")}).compile("outer(x,x)")
    bounds = np.zeros(2, np.int64)
    x = np.ones(4100)
    with AdaptiveScheduler(cpu_budget=1, config=config("process_shared")) as scheduler:
        shared = scheduler.plan(g, {"x": x}, bounds, bounds)
    with AdaptiveScheduler(cpu_budget=2, config=config("process_inline")) as scheduler:
        initial = scheduler.plan(g, {"x": x}, bounds, bounds)
        assert not initial.use_shared_memory
        plan = scheduler.plan(g, {"x": x}, bounds, bounds,
                              memory_budget_bytes=shared.estimated_total_memory_bytes)
        assert plan.process_count == 1 and plan.use_shared_memory
        assert plan.estimated_total_memory_bytes == shared.estimated_total_memory_bytes
        assert "memory_budget_reduced_parallelism" in plan.reason_codes
        with pytest.raises(MemoryError):
            scheduler.plan(g, {"x": x}, bounds, bounds,
                           memory_budget_bytes=shared.estimated_total_memory_bytes - 1)


@pytest.mark.parametrize("hard_stop", [False, True])
def test_status_metadata_over_frame_limit_fails_during_planning(hard_stop):
    g = GraphCompiler({"x": decl("vector")}).compile("outer(x,x)", error_policy="isolate")
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("process_inline")) as scheduler:
        with pytest.raises(ValueError, match="status/shape metadata exceeds IPC frame limit"):
            scheduler.plan(g, {"x": np.ones(8192)}, bounds, bounds, hard_stop=hard_stop)
