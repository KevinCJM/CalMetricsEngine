"""Internal numeric stops cannot hide independent consumed constraints."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_matmul_library import settings

VARIABLES = {
    "x": "series",
    "m": {"kind": "series", "dtype": "bool"},
    "keys": {"kind": "series", "dtype": "int64"},
    "events": {"kind": "series", "dtype": "int64", "semantic_dimension": "event"},
}
SCOPES = [
    "mean(group_apply(mean_where(log(x),m),keys))",
    "first(segment_apply(mean_where(log(x),m),between_events(events)))",
    "mean(block_apply(mean_where(log(x),m),257))",
    "filter_apply(filter_apply(mean_where(log(x),m),finite_mask(x)),finite_mask(x))",
]
LANES = ["single", "thread", "process_inline", "process_shared"]


def data(failed=False, bad_mask=False):
    inputs = {
        "x": np.full(257, -1.0 if failed else 1.0),
        "m": np.ones(257, np.uint8),
        "keys": np.zeros(257, np.int64),
        "events": np.zeros(257, np.int64),
    }
    inputs["events"][[0, 256]] = [-1, 1]
    if bad_mask:
        inputs["m"][128] = 2
    return inputs, np.zeros(2, np.int64), np.full(2, 257, np.int64)


@pytest.mark.parametrize("expression", SCOPES)
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("lane", LANES)
def test_consumed_bad_mask_survives_an_internal_numeric_stop(expression, failed, lane):
    inputs, starts, ends = data(failed, True)
    for value in inputs.values():
        value.flags.writeable = False
    graph = GraphCompiler(VARIABLES).compile(expression, error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        with pytest.raises(
            RuntimeError if lane.startswith("process") else ValueError, match="INVALID_MASK"
        ):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize("expression", SCOPES)
@pytest.mark.parametrize("lane", LANES)
def test_numeric_stop_with_valid_structure_keeps_failure_and_independent_roots(expression, lane):
    inputs, starts, ends = data(True)
    graph = GraphCompiler(VARIABLES).compile([expression, "mean(x)"], error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        failed = scheduler.execute(graph, inputs, starts, ends)
        assert np.all(failed.statuses[:, 0] != 0)
        np.testing.assert_array_equal(failed.statuses[:, 1], 0)
        np.testing.assert_array_equal(failed.values[:, 1], -1.0)
        inputs["x"][:] = 1.0
        healthy = scheduler.execute(graph, inputs, starts, ends)
        np.testing.assert_array_equal(healthy.statuses, 0)
        np.testing.assert_array_equal(healthy.values[:, 0], 0.0)
        np.testing.assert_array_equal(healthy.values[:, 1], 1.0)
        np.testing.assert_array_equal(failed.values[:, 1], -1.0)
        assert np.all(failed.statuses[:, 0] != 0)
        assert not np.shares_memory(failed.values, healthy.values)


@pytest.mark.parametrize("policy", ["raise", "isolate"])
@pytest.mark.parametrize("lane", LANES)
def test_numeric_stop_does_not_consume_an_excluded_bad_mask(policy, lane):
    inputs, starts, ends = data(True, True)
    inputs["x"][128] = -100.0
    expression = "filter_apply(mean_where(log(x),m),x>-2)"
    graph = GraphCompiler(VARIABLES).compile(expression, error_policy=policy)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        if policy == "raise":
            with pytest.raises(
                RuntimeError if lane.startswith("process") else ValueError, match="DOMAIN_ERROR"
            ):
                scheduler.execute(graph, inputs, starts, ends)
        else:
            result = scheduler.execute(graph, inputs, starts, ends)
            assert np.all(result.statuses != 0)


@pytest.mark.parametrize("failed", [False, True])
def test_empty_filter_does_not_run_or_validate_its_body(failed):
    inputs, starts, ends = data(failed, True)
    graph = GraphCompiler(VARIABLES).compile(
        "filter_apply(mean_where(log(x),m),x>2,7)", error_policy="isolate"
    )
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    np.testing.assert_array_equal(result.statuses, 0)
    np.testing.assert_array_equal(result.values, 7.0)


def test_prepared_error_recovery_does_not_publish_or_retain_the_audit_result():
    inputs, starts, ends = data(True, True)
    graph = GraphCompiler(VARIABLES).compile(
        ["filter_apply(mean_where(log(x),m),finite_mask(x))", "mean(x)"], error_policy="isolate"
    )
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, starts, ends)
        with pytest.raises(ValueError, match="^INVALID_MASK$"):
            prepared.run_snapshot()
        inputs["m"][128] = 1
        saved = prepared.run_snapshot()
        assert np.all(saved.statuses[:, 0] != 0)
        np.testing.assert_array_equal(saved.values[:, 1], -1.0)
        inputs["x"][:] = 1.0
        healthy = prepared.run_snapshot()
        np.testing.assert_array_equal(healthy.statuses, 0)
        np.testing.assert_array_equal(healthy.values[:, 0], 0.0)
        np.testing.assert_array_equal(saved.values[:, 1], -1.0)
        assert np.all(saved.statuses[:, 0] != 0)


def test_scope_work_budget_accounts_for_the_failure_audit():
    graph = GraphCompiler(VARIABLES).compile(
        "mean(group_apply(mean(log(x)),keys))", error_policy="isolate", scope_work_budget=1000
    )
    inputs, starts, ends = data()
    starts, ends = starts[:1], ends[:1]
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        healthy = scheduler.execute(graph, inputs, starts, ends)
        np.testing.assert_array_equal(healthy.statuses, 0)
        np.testing.assert_array_equal(healthy.values, 0.0)
        inputs["x"][:] = -1.0
        with pytest.raises(ValueError, match="^SCOPE_COMPUTE_BUDGET_EXCEEDED$"):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("lane", LANES)
def test_rolling_internal_stop_keeps_independent_state_code_validation(failed, lane):
    x = np.full(257, -1.0 if failed else 1.0)
    inputs = {"x": x, "y": np.ones(257)}
    expression = (
        "mean(rolling_apply(mean(log(x))+count_true(state_select(y>0,-2,1,finite_mask(y))==1),257))"
    )
    graph = GraphCompiler({"x": "series", "y": "series"}).compile(
        expression, error_policy="isolate"
    )
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        with pytest.raises(
            RuntimeError if lane.startswith("process") else ValueError, match="INVALID_STATE_CODE"
        ):
            scheduler.execute(graph, inputs, np.zeros(2, np.int64), np.full(2, 257, np.int64))
