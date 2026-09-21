"""Ordinary rolling/group failures retain provenance without a segment source."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig

VARIABLES = {
    "x": "series",
    "key": {"kind": "series", "dtype": "int64", "semantic_dimension": "category"},
}
SOURCES = {"rolling": "rolling_apply(1/std(x,0),2)",
           "group": "group_apply(1/std(x,0),key)"}


def batch(scope):
    first = [2., 2., 3., 4.] if scope == "rolling" else [2., 3., 2., 4.]
    inputs = {"x": np.array(first + [1., 2., 3., 4.]),
              "key": np.tile(np.array([0, 1, 0, 1], np.int64), 2)}
    return inputs, np.array([0, 4], np.int64), np.array([4, 8], np.int64)


def compiler(scope, kind, policy="isolate"):
    condition = "finite_mask(v)" if kind == "finite" else "v>0"
    dependent = f"state_select({condition},1,2,finite_mask(x))" if kind == "state" else condition
    independent = "state_select(x>0,11,12,finite_mask(x))" if kind == "state" else "x>0"
    return GraphCompiler(VARIABLES).compile(
        [dependent, independent, dependent], root_bindings=[{"v": SOURCES[scope]}] * 3,
        error_policy=policy)


def check_result(result, scope, kind):
    bad = [1] if scope == "rolling" else [0, 2]
    healthy = np.ones(8, bool)
    healthy[bad] = False
    assert np.isin(result.statuses[np.ix_(bad, [0, 2])], [2, 4]).all()
    np.testing.assert_array_equal(result.statuses[healthy], 0)
    np.testing.assert_array_equal(result.statuses[:, 1], 0)
    expected = np.ones(8, bool)
    if scope == "rolling":
        expected[[0, 4]] = False  # Warmup is not a failed body execution.
    values = np.where(expected, 1, 2) if kind == "state" else expected
    np.testing.assert_array_equal(result.values[healthy, 0], values[healthy])
    np.testing.assert_array_equal(result.values[:, 1], 11 if kind == "state" else True)
    np.testing.assert_array_equal(result.statuses[:, 0], result.statuses[:, 2])
    np.testing.assert_array_equal(result.offsets, [0, 4, 8])


@pytest.mark.parametrize("scope", SOURCES)
@pytest.mark.parametrize("kind", ["comparison", "finite", "state"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_ordinary_scope_errors_survive_typed_roots_and_worker_transport(scope, kind, lane):
    inputs, starts, ends = batch(scope)
    for value in inputs.values():
        value.setflags(write=False)
    config = PlannerConfig(thread_work_units=1e99 if lane == "single" else 1,
                           process_work_units=1 if lane.startswith("process") else 1e100,
                           shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
                           min_rows_per_worker=1, max_processes=2)
    graph = pickle.loads(pickle.dumps(compiler(scope, kind)))
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    if lane.startswith("process"):
        assert result.plan.use_shared_memory == (lane == "process_shared")
    check_result(result, scope, kind)


@pytest.mark.parametrize("scope", SOURCES)
@pytest.mark.parametrize("kind", ["comparison", "state"])
def test_ordinary_scope_prepared_failure_recovery_and_snapshot(scope, kind):
    inputs, starts, ends = batch(scope)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(compiler(scope, kind), inputs, starts, ends)
        with pytest.raises(ValueError, match="TYPED_OUTPUT_STATUS_REQUIRED"):
            prepared.run()
        saved = prepared.run_snapshot()
        check_result(saved, scope, kind)
        inputs["x"][:4] = [1., 2., 3., 4.]
        fresh = prepared.run_audit()
        np.testing.assert_array_equal(fresh.statuses, 0)
        np.testing.assert_array_equal(prepared.run(), fresh.values)
        assert not np.shares_memory(saved.values, fresh.values)
    check_result(saved, scope, kind)


@pytest.mark.parametrize("scope", SOURCES)
def test_ordinary_scope_provenance_is_in_memory_admission(scope):
    inputs, starts, ends = batch(scope)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        isolated = scheduler.plan(compiler(scope, "comparison"), inputs, starts, ends)
        strict = scheduler.plan(compiler(scope, "comparison", "raise"), inputs, starts, ends)
        assert isolated.estimated_worker_scratch_bytes - strict.estimated_worker_scratch_bytes >= 2 * 4 * 2
        with pytest.raises(MemoryError):
            scheduler.plan(compiler(scope, "comparison"), inputs, starts, ends,
                           memory_budget_bytes=isolated.estimated_total_memory_bytes - 1)


@pytest.mark.parametrize("outer", ["group", "block", "filter", "bisect"])
def test_nested_rolling_exception_reaches_its_isolating_outer_scope(outer):
    body = "count_true(rolling_apply(1/std(x,0),2)>0)"
    expressions = {
        "group": f"group_apply({body},key)>0",
        "block": f"sum(block_apply({body},2))>0",
        "filter": f"filter_apply({body},x>0)>0",
        "bisect": f"bisect(solve_x-{body},-1,10,1e-8,100)>0",
    }
    x = np.array([2., 2., 3., 4.])
    inputs = {"x": x, "key": np.array([0, 0, 1, 1], np.int64)}
    graph = GraphCompiler(VARIABLES).compile(expressions[outer], error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        bad = scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([4], np.int64))
        assert np.all(bad.statuses[:2 if outer == "group" else 1] != 0)
        if outer == "group":
            np.testing.assert_array_equal(bad.statuses[2:], 0)
            np.testing.assert_array_equal(bad.values[2:], True)
        x[:] = [1., 2., 3., 4.]
        healthy = scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([4], np.int64))
        np.testing.assert_array_equal(healthy.statuses, 0)
        np.testing.assert_array_equal(healthy.values, 1)


def test_strict_rolling_compatibility_is_independent_of_position_tracking_metadata():
    # Existing strict rolling produces NaN for a numerical body failure. Adding
    # provenance for isolate mode must not change this, even with a group body.
    expressions = ["rolling_apply(1/std(x,0),2)>0",
                   "rolling_apply(mean(x)+mean(group_apply(1/std(x,0),argsort(x))),2)>0"]
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        for expression in expressions:
            isolated = GraphCompiler({"x": "series"}).compile(expression, error_policy="isolate")
            failed = scheduler.execute(isolated, {"x": np.ones(4)},
                                       np.array([0], np.int64), np.array([4], np.int64))
            assert np.all(failed.statuses[1:] != 0)
            graph = GraphCompiler({"x": "series"}).compile(expression, error_policy="raise")
            result = scheduler.execute(graph, {"x": np.ones(4)},
                                       np.array([0], np.int64), np.array([4], np.int64))
            np.testing.assert_array_equal(result.values, False)
        graph = compiler("group", "comparison", "raise")
        inputs, starts, ends = batch("group")
        with pytest.raises(ValueError, match="DIVIDE_BY_ZERO"):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize("scope", SOURCES)
def test_nonexceptional_missing_values_are_not_misclassified_as_scope_faults(scope):
    graph = GraphCompiler(VARIABLES).compile(
        SOURCES[scope].replace("1/std(x,0)", "mean(x)") + ">0", error_policy="isolate")
    inputs, starts, ends = batch(scope)
    inputs["x"][:] = np.nan
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    np.testing.assert_array_equal(result.values, False)
    np.testing.assert_array_equal(result.statuses, 0)
