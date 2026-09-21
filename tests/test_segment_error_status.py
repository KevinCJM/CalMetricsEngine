"""A failed segment must remain failed after comparisons and state selection."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig

EVENT = {"kind": "series", "dtype": "int64", "semantic_dimension": "event"}
SEGMENT = "segment_apply(1/std(x,0),between_events(events))"
DEPENDENT_COLUMNS = [0, 1, 3]


def batch():
    # In the first interval [1,3] is constant, [3,5] and [5,7] are healthy.
    # Rows 0,7,8 have no complete segment. The second interval is fully healthy
    # wherever a complete segment exists, to detect cross-interval status leaks.
    x = np.array([7., 2., 2., 2., 3., 4., 5., 6., 7.,
                  7., 1., 2., 3., 4., 5., 6., 7., 8.])
    events = np.tile(np.array([0, -1, 0, 1, 0, -1, 0, 1, 0], np.int64), 2)
    return {"x": x, "events": events}, np.array([0, 9], np.int64), np.array([9, 18], np.int64)


def settings(lane):
    return PlannerConfig(
        thread_work_units=1e99 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        min_rows_per_worker=1,
        max_processes=2,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
    )


def compile_graph(kind, *, error_policy="isolate"):
    positive = "segment > 0"
    if kind == "comparison":
        roots = [positive, f"logical_not({positive})", "x > 0", positive]
    else:
        selection = f"state_select({positive},1,2,finite_mask(x))"
        roots = [selection,
                 f"state_select(logical_not({positive}),3,4,finite_mask(x))",
                 "state_select(x>0,11,12,finite_mask(x))", selection]
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile(
        roots, error_policy=error_policy, root_bindings=[{"segment": SEGMENT}] * len(roots))
    assert graph.metadata()["apply_scope_count"] == 1
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    return graph


def expected_values(kind, positive):
    if kind == "comparison":
        return np.column_stack([positive, ~positive, np.ones_like(positive), positive])
    selection = np.where(positive, 1, 2)
    return np.column_stack([selection, np.where(positive, 4, 3),
                            np.full(len(positive), 11, np.int64), selection])


def check_partial_failure(result, kind):
    statuses = result.statuses
    bad_rows = np.array([1, 2])
    healthy_rows = np.ones(18, dtype=bool)
    healthy_rows[bad_rows] = False
    positive = np.zeros(18, dtype=bool)
    positive[3:7] = True
    positive[10:16] = True
    expected = expected_values(kind, positive)

    # Divide-by-zero (2), or the existing unavailable numerical-result code (4),
    # must survive every dependency. A valid False/state code is never evidence
    # that the failed body recovered.
    assert np.isin(statuses[np.ix_(bad_rows, DEPENDENT_COLUMNS)], [2, 4]).all()
    np.testing.assert_array_equal(statuses[healthy_rows], 0)
    np.testing.assert_array_equal(statuses[:, 2], 0)
    np.testing.assert_array_equal(result.values[healthy_rows], expected[healthy_rows])
    np.testing.assert_array_equal(result.values[:, 2], expected[:, 2])
    np.testing.assert_array_equal(statuses[:, 0], statuses[:, 3])
    np.testing.assert_array_equal(result.values[:, 0], result.values[:, 3])
    np.testing.assert_array_equal(result.offsets, [0, 9, 18])
    assert result.values.dtype == (np.bool_ if kind == "comparison" else np.int64)
    assert statuses.dtype == np.int16
    assert not statuses.flags.writeable


@pytest.mark.parametrize("kind", ["comparison", "state_select"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_failed_segment_status_survives_downstream_shared_roots(kind, lane):
    inputs, starts, ends = batch()
    inputs["x"].setflags(write=False)
    inputs["events"].setflags(write=False)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(compile_graph(kind), inputs, starts, ends)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    if lane.startswith("process"):
        assert result.plan.use_shared_memory == (lane == "process_shared")
    check_partial_failure(result, kind)


@pytest.mark.parametrize("kind", ["comparison", "state_select"])
def test_incomplete_segments_keep_normal_nan_comparison_semantics(kind):
    inputs, starts, ends = batch()
    inputs["events"][:] = 0  # No body runs, hence no numerical exception.
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(compile_graph(kind), inputs, starts, ends)
    np.testing.assert_array_equal(result.statuses, 0)
    np.testing.assert_array_equal(result.values, expected_values(kind, np.zeros(18, dtype=bool)))


@pytest.mark.parametrize("kind", ["comparison", "state_select"])
def test_prepared_snapshot_retains_segment_failure_after_successful_reuse(kind):
    inputs, starts, ends = batch()
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        prepared = scheduler.prepare_execution(compile_graph(kind), inputs, starts, ends)
        retained = prepared.run_snapshot()
        retained_statuses = retained.statuses
        retained_values = retained.values.copy()
        inputs["x"][1:4] = [1., 2., 3.]  # Repair the failing segment in-place.
        fresh = prepared.run_audit()
        np.testing.assert_array_equal(fresh.statuses, 0)
        np.testing.assert_array_equal(prepared.run(), fresh.values)
        assert not np.shares_memory(retained.values, fresh.values)
        assert retained.audit["result_lifetime"] == "independent"
        assert fresh.audit["result_lifetime"] == "borrowed_until_next_run"
    check_partial_failure(retained, kind)
    np.testing.assert_array_equal(retained.statuses, retained_statuses)
    np.testing.assert_array_equal(retained.values, retained_values)
    assert not retained.values.flags.writeable


@pytest.mark.parametrize("kind", ["comparison", "state_select"])
def test_bare_prepared_typed_output_cannot_discard_segment_failure(kind):
    inputs, starts, ends = batch()
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        prepared = scheduler.prepare_execution(compile_graph(kind), inputs, starts, ends)
        with pytest.raises(ValueError, match="TYPED_OUTPUT_STATUS_REQUIRED"):
            prepared.run()


@pytest.mark.parametrize("kind", ["comparison", "state_select"])
def test_strict_segment_body_failure_still_raises(kind):
    inputs, starts, ends = batch()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match="DIVIDE_BY_ZERO"):
            scheduler.execute(compile_graph(kind, error_policy="raise"), inputs, starts, ends)


def test_nested_segment_failure_survives_group_broadcast_and_comparison():
    inputs, _, _ = batch()
    inputs["keys"] = np.repeat(np.array([0, 1], np.int64), 9)
    variables = {"x": "series", "events": EVENT,
                 "keys": {"kind": "series", "dtype": "int64", "semantic_dimension": "category"}}
    # The group body is strict: one nested segment exception fails that group's
    # statistic, while the next group remains usable. Ordinary boundary NaNs are
    # explicitly excluded from the mean, without suppressing thrown exceptions.
    body = f"mean_where({SEGMENT},finite_mask({SEGMENT}))"
    graph = GraphCompiler(variables).compile(
        [f"group_apply({body},keys)>0", "x>0"], error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([18], np.int64))
    assert np.isin(result.statuses[:9, 0], [2, 4]).all()
    np.testing.assert_array_equal(result.statuses[9:, 0], 0)
    np.testing.assert_array_equal(result.statuses[:, 1], 0)
    np.testing.assert_array_equal(result.values[9:, 0], True)
    np.testing.assert_array_equal(result.values[:, 1], True)


def test_planner_admits_position_status_workspace_before_any_failure_occurs():
    length = 257
    inputs = {"x": np.arange(1., length + 1), "events": np.zeros(length, np.int64)}
    inputs["events"][[0, -1]] = [-1, 1]
    roots = [f"{SEGMENT}>{threshold}" for threshold in range(10)]
    compiler = GraphCompiler({"x": "series", "events": EVENT})
    isolated = compiler.compile(roots, error_policy="isolate")
    strict = compiler.compile(roots, error_policy="raise")
    starts, ends = np.array([0], np.int64), np.array([length], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        isolated_plan = scheduler.plan(isolated, inputs, starts, ends)
        strict_plan = scheduler.plan(strict, inputs, starts, ends)
        # One segment source and ten separately addressable comparisons can
        # carry failures, even though this particular batch is healthy.
        assert (isolated_plan.estimated_worker_scratch_bytes
                - strict_plan.estimated_worker_scratch_bytes) >= 11 * length * 2
        with pytest.raises(MemoryError):
            scheduler.plan(isolated, inputs, starts, ends,
                           memory_budget_bytes=isolated_plan.estimated_total_memory_bytes - 1)


def test_aligned_shift_moves_failure_positions_and_discards_unused_prefix():
    inputs, starts, ends = batch()
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile(
        ["aligned_shift(segment,1,0)>0", "aligned_shift(segment,9,1)>0"],
        root_bindings=[{"segment": SEGMENT}] * 2, error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    bad = np.array([2, 3])
    healthy = np.ones(18, bool)
    healthy[bad] = False
    expected = np.zeros(18, bool)
    expected[4:8] = True
    expected[11:17] = True
    assert np.isin(result.statuses[bad, 0], [2, 4]).all()
    np.testing.assert_array_equal(result.statuses[healthy, 0], 0)
    np.testing.assert_array_equal(result.values[healthy, 0], expected[healthy])
    # The second shift writes only fill values and reads no failed source row.
    np.testing.assert_array_equal(result.statuses[:, 1], 0)
    np.testing.assert_array_equal(result.values[:, 1], True)


@pytest.mark.parametrize("view", ["lag(segment,1)", "difference(segment,1)", "tail"])
def test_borrowed_and_difference_failures_survive_a_scalar_reduction(view):
    inputs, starts, ends = batch()
    bindings = {"segment": SEGMENT, "tail": "interval_tail(segment)"}
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile(
        f"count_true({view}>0)", root_bindings=[bindings], error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    assert result.statuses[0, 0] != 0
    assert result.statuses[1, 0] == 0
    assert np.isfinite(result.values[1, 0])


@pytest.mark.parametrize("view", ["lag", "tail"])
def test_views_that_remove_all_failed_source_positions_remain_usable(view):
    if view == "lag":
        x = np.array([7., 1., 2., 3., 4., 5., 5., 5., 8.])
        events = np.array([0, -1, 0, 1, 0, -1, 0, 1, 0], np.int64)
        # Failing membership is [5,7); this lag retains only positions [0,5).
        expression, expected = "count_true(lag(segment,4)>0)", 4
    else:
        x = np.array([2., 2., 3., 4.])
        events = np.array([-1, 1, 0, -1], np.int64)
        # [0,1] is the only bad body; its membership is the single row 0.
        expression, expected = "count_true(tail>0)", 2
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile(
        expression, root_bindings=[{"segment": SEGMENT, "tail": "interval_tail(segment)"}],
        error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x, "events": events},
                                   np.array([0], np.int64), np.array([len(x)], np.int64))
    np.testing.assert_array_equal(result.statuses, [[0]])
    np.testing.assert_array_equal(result.values, [[expected]])


@pytest.mark.parametrize("faulty_selector", [False, True])
def test_filter_distinguishes_failed_selector_from_excluded_failed_values(faulty_selector):
    inputs, starts, ends = batch()
    healthy_selection = np.zeros(18, bool)
    healthy_selection[3:7] = True
    healthy_selection[10:16] = True
    inputs["selected"] = healthy_selection
    variables = {"x": "series", "events": EVENT, "selected": {"kind": "series", "dtype": "bool"}}
    selector = "finite_mask(v)" if faulty_selector else "selected"
    graph = GraphCompiler(variables).compile(f"filter_apply(mean(v),{selector})>0",
                                            root_bindings=[{"v": SEGMENT}], error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    assert result.statuses[1, 0] == 0
    assert result.values[1, 0] == 1
    if faulty_selector:
        assert result.statuses[0, 0] != 0
    else:
        # Selection is explicitly independent of v's validity and excludes all
        # failed positions. Those unused positions cannot poison a healthy mean.
        np.testing.assert_array_equal(result.statuses, 0)
        np.testing.assert_array_equal(result.values, 1)


@pytest.mark.parametrize("operator,expected", [("sqrt", 2.), ("log", np.log(4.))])
def test_pointwise_domain_checks_skip_existing_failed_positions(operator, expected):
    inputs, starts, ends = batch()
    # If the segment failure were mistaken for False, the two failing rows
    # would select -1, and sqrt/log would incorrectly fail the healthy rows too.
    argument = "where(segment>0,4,where(x==2,-1,4))"
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile(
        [f"{operator}({argument})", "x"], root_bindings=[{"segment": SEGMENT}] * 2,
        error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    bad = np.array([1, 2])
    healthy = np.ones(18, bool)
    healthy[bad] = False
    assert np.isin(result.statuses[bad, 0], [2, 4]).all()
    np.testing.assert_array_equal(result.statuses[healthy, 0], 0)
    np.testing.assert_allclose(result.values[healthy, 0], expected)
    np.testing.assert_array_equal(result.statuses[:, 1], 0)
    np.testing.assert_array_equal(result.values[:, 1], inputs["x"])


@pytest.mark.parametrize("kind,expression", [
    ("recurrence", "recursive_filter(segment,.2,0,finite_mask(segment),0,0)>0"),
    ("reduction", "mean_where(segment,finite_mask(segment))>0"),
    ("ordering", "sum(gather(x,argsort(segment)))>0"),
])
def test_nonlocal_dependencies_fail_closed_instead_of_guessing_position_mapping(kind, expression):
    inputs, starts, ends = batch()
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile(
        expression, root_bindings=[{"segment": SEGMENT}], error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    boundary = 9 if kind == "recurrence" else 1
    assert np.all(result.statuses[:boundary, 0] != 0)
    np.testing.assert_array_equal(result.statuses[boundary:, 0], 0)


def test_structurally_invalid_mask_is_not_hidden_by_existing_position_failure():
    inputs, starts, ends = batch()
    inputs["valid"] = np.ones(18, np.uint8)
    inputs["valid"][1] = 2  # Invalid bool payload overlaps a failed segment row.
    graph = GraphCompiler({"x": "series", "events": EVENT,
                           "valid": {"kind": "series", "dtype": "bool"}}).compile(
        "state_select(segment>0,1,2,valid)", root_bindings=[{"segment": SEGMENT}],
        error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match="INVALID_MASK"):
            scheduler.execute(graph, inputs, starts, ends)


# Healthy controls define the existing structural-error contract. In particular,
# INVALID_PARAMETER is intentionally not used here: isolate already converts
# that numerical diagnostic into status 4, even without an upstream failure.
STRUCTURAL_ERRORS = [
    pytest.param("recursive_filter(segment,.2,0,valid)", "INVALID_MASK", id="recurrence-mask"),
    pytest.param("recursive_filter_adaptive(segment,.2,valid,finite_mask(x))",
                 "INVALID_MASK", id="adaptive-mask"),
    pytest.param("linear_filter2(segment,.1,.2,.3,.4,valid)",
                 "INVALID_MASK", id="second-order-mask"),
    pytest.param("mean_where(segment,valid)", "INVALID_MASK", id="reduction-mask"),
    pytest.param("recursive_filter(recursive_filter(segment,.2,0,finite_mask(x)),.2,0,valid)",
                 "INVALID_MASK", id="multihop-mask"),
    pytest.param("sum(recursive_filter(lag(segment,1),.2,0,finite_mask(lag(x,2))))",
                 "SHAPE_MISMATCH", id="recurrence-length"),
    pytest.param("sum(recursive_filter(lag(recursive_filter(segment,.2,0,finite_mask(x)),1),"
                 ".2,0,finite_mask(lag(x,2))))", "SHAPE_MISMATCH", id="multihop-length"),
    pytest.param("state_select(recursive_filter(segment,.2,0,finite_mask(x))>0,-2,1,valid)",
                 "INVALID_STATE_CODE", id="multihop-state-parameter"),
    pytest.param("state_select(state_hysteresis(segment,0,1,2,3)==1,-2,1,valid)",
                 "INVALID_STATE_CODE", id="hysteresis-state-parameter"),
    pytest.param("drawdown_cycle_state(segment,dd,valid,.2,.1,0)",
                 "INVALID_DRAWDOWN", id="state-drawdown-domain"),
]


def structural_error_batch(expression, expected_error, *, source_failure=True, bindings=None):
    inputs, starts, ends = batch()
    # A single interval matters: another healthy interval must not be the one
    # that eventually detects a contract skipped in the failing interval.
    inputs = {name: value[:9].copy() for name, value in inputs.items()}
    if not source_failure:
        inputs["x"][:] = np.arange(1., 10.)
    inputs["valid"] = np.ones(9, np.uint8)
    if expected_error == "INVALID_MASK":
        inputs["valid"][1] = 2
    inputs["dd"] = np.full(9, .1)
    inputs["keys"] = np.zeros(9, np.int64)
    graph = GraphCompiler({"x": "series", "events": EVENT, "dd": "series",
                           "keys": {"kind": "series", "dtype": "int64", "semantic_dimension": "category"},
                           "valid": {"kind": "series", "dtype": "bool"}}).compile(
        expression, root_bindings=[{"segment": SEGMENT, **(bindings or {})}], error_policy="isolate")
    return graph, inputs, starts[:1], ends[:1]


@pytest.mark.parametrize("expression,expected_error", STRUCTURAL_ERRORS)
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_structural_contract_survives_nonlocal_and_multihop_failure(
        expression, expected_error, source_failure):
    graph, inputs, starts, ends = structural_error_batch(
        expression, expected_error, source_failure=source_failure)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match=f"^{expected_error}$"):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_inherited_failure_cannot_hide_structural_error_in_worker(lane, source_failure):
    expression = "recursive_filter(recursive_filter(segment,.2,0,finite_mask(x)),.2,0,valid)"
    graph, inputs, starts, ends = structural_error_batch(
        expression, "INVALID_MASK", source_failure=source_failure)
    # Repeat the interval to force at least two native worker chunks.
    starts, ends = np.repeat(starts, 2), np.repeat(ends, 2)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        assert plan.lane == ("process" if lane.startswith("process") else lane)
        if lane.startswith("process"):
            assert plan.use_shared_memory == (lane == "process_shared")
        # The existing process wire wraps native operator errors in RuntimeError;
        # both healthy and failed upstream sources must retain that same contract.
        error_type = RuntimeError if lane.startswith("process") else ValueError
        message = "^native worker: INVALID_MASK$" if lane.startswith("process") else "^INVALID_MASK$"
        with pytest.raises(error_type, match=message):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize("expression,expected_error", [
    pytest.param("recursive_filter(recursive_filter(segment,.2,0,finite_mask(x)),.2,0,valid)",
                 "INVALID_MASK", id="mask"),
    pytest.param("sum(recursive_filter(lag(segment,1),.2,0,finite_mask(lag(x,2))))",
                 "SHAPE_MISMATCH", id="length"),
    pytest.param("state_select(recursive_filter(segment,.2,0,finite_mask(x))>0,-2,1,valid)",
                 "INVALID_STATE_CODE", id="state-parameter"),
])
def test_prepared_snapshot_does_not_convert_structural_error_into_failure_status(
        expression, expected_error):
    graph, inputs, starts, ends = structural_error_batch(expression, expected_error)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, starts, ends)
        with pytest.raises(ValueError, match=f"^{expected_error}$"):
            prepared.run_snapshot()


GLOBAL_FAILURE = "recursive_filter(segment,.2,0,finite_mask(x))"
FAILED_LENGTH = "1+0*count_true(segment>0)"


@pytest.mark.parametrize("expression,bindings,expected_error", [
    pytest.param("filter_apply(mean(v),valid)", {"v": GLOBAL_FAILURE},
                 "INVALID_MASK", id="filter-mask"),
    pytest.param("sum(group_apply(mean(v),key))",
                 {"v": f"lag({GLOBAL_FAILURE},1)",
                  "key": "state_select(finite_mask(lag(x,2)),1,2,finite_mask(lag(x,2)))"},
                 "SCOPE_ALIGNMENT_MISMATCH", id="group-length"),
    pytest.param("rolling_apply(mean(v),2)", {"v": f"lag({GLOBAL_FAILURE},1)"},
                 "ROLLING_ALIGNMENT_MISMATCH", id="rolling-capture-length"),
    pytest.param("mean_where(rolling_apply(mean(x),n),valid)", {"n": FAILED_LENGTH},
                 "INVALID_MASK", id="failed-window-independent-mask"),
    pytest.param("state_select(rolling_apply(mean(x),n)>0,-2,1,valid)", {"n": FAILED_LENGTH},
                 "INVALID_STATE_CODE", id="failed-window-independent-state-code"),
])
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_scope_structure_remains_checked_when_capture_or_window_failed(
        expression, bindings, expected_error, source_failure):
    graph, inputs, starts, ends = structural_error_batch(
        expression, expected_error, source_failure=source_failure, bindings=bindings)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match=f"^{expected_error}$"):
            scheduler.execute(graph, inputs, starts, ends)


UNKNOWN_GEOMETRY = [
    pytest.param("sum(recursive_filter(lag(x,n),.2,0,finite_mask(lag(x,1))))",
                 [[17.31564544]], id="lag-from-failed-scalar"),
    pytest.param("rolling_apply(mean(x),n)", np.arange(1., 10.).reshape(-1, 1),
                 id="rolling-from-failed-scalar"),
    pytest.param("sum(block_apply(mean(x),n))", [[45.]], id="block-from-failed-scalar"),
]


@pytest.mark.parametrize("expression,expected", UNKNOWN_GEOMETRY)
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_failed_length_is_unknown_and_does_not_create_false_shape_errors(
        expression, expected, source_failure):
    graph, inputs, starts, ends = structural_error_batch(
        expression, "", source_failure=source_failure, bindings={"n": FAILED_LENGTH})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    if source_failure:
        assert np.all(result.statuses != 0)
    else:
        np.testing.assert_array_equal(result.statuses, 0)
        np.testing.assert_allclose(result.values, expected)


@pytest.mark.parametrize("expression,expected", UNKNOWN_GEOMETRY)
def test_prepared_recovery_clears_unknown_shape_and_failure_metadata(expression, expected):
    graph, inputs, starts, ends = structural_error_batch(
        expression, "", bindings={"n": FAILED_LENGTH})
    failed_input = inputs["x"].copy()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, starts, ends)
        first_failure = prepared.run_snapshot()
        assert np.all(first_failure.statuses != 0)
        for _ in range(2):
            inputs["x"][:] = np.arange(1., 10.)
            recovered = prepared.run_snapshot()
            np.testing.assert_array_equal(recovered.statuses, 0)
            np.testing.assert_allclose(recovered.values, expected)
            inputs["x"][:] = failed_input
            failed_again = prepared.run_snapshot()
            assert np.all(failed_again.statuses != 0)
            # Neither a new failure nor arena reuse can alter the healthy copy.
            np.testing.assert_array_equal(recovered.statuses, 0)
            np.testing.assert_allclose(recovered.values, expected)
        inputs["x"][:] = np.arange(1., 10.)
        np.testing.assert_allclose(prepared.run(), expected)
    assert np.all(first_failure.statuses != 0)


def test_valid_multihop_dependency_preserves_only_the_numerical_failure():
    expression = "recursive_filter(v,.2,0,finite_mask(x))>0"
    graph, inputs, starts, ends = structural_error_batch(
        expression, "", bindings={"v": GLOBAL_FAILURE})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        failed = scheduler.execute(graph, inputs, starts, ends)
        assert np.all(failed.statuses != 0)
        inputs["x"][:] = np.arange(1., 10.)
        healthy = scheduler.execute(graph, inputs, starts, ends)
    np.testing.assert_array_equal(healthy.statuses, 0)
    assert healthy.values.dtype == np.bool_
    assert np.any(healthy.values)


@pytest.mark.parametrize("expression", [
    pytest.param("group_apply(mean_where(v,valid),keys)", id="group"),
    pytest.param("segment_apply(mean_where(v,valid),between_events(events))", id="segment"),
    pytest.param("filter_apply(mean_where(v,valid),finite_mask(x))", id="filter"),
    pytest.param("sum(block_apply(mean_where(v,valid),3))", id="block"),
])
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_failed_capture_cannot_skip_independent_structure_validation_inside_scope_body(
        expression, source_failure):
    graph, inputs, starts, ends = structural_error_batch(
        expression, "INVALID_MASK", source_failure=source_failure, bindings={"v": SEGMENT})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match="^INVALID_MASK$"):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize("scope", ["filter", "segment", "block"])
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_scope_body_validation_does_not_read_unconsumed_invalid_mask(scope, source_failure):
    body = "mean_where(v,logical_and(valid,finite_mask(v)))"
    if scope == "filter":
        # On the failing source, x>2 selects rows 0,4..8; neither failed row
        # nor its invalid mask at row 1 belongs to this scope invocation.
        expression = f"filter_apply({body},x>2)>0"
        invalid_row = 1
    elif scope == "segment":
        # Row 0 is outside every complete [left,right] body, including endpoints.
        expression = f"segment_apply({body},between_events(events))>0"
        invalid_row = 0
    else:
        # Complete blocks cover [0,8); the last row is deliberately dropped.
        expression = f"count_true(block_apply({body},4)>0)"
        invalid_row = 8
    graph, inputs, starts, ends = structural_error_batch(
        expression, "", source_failure=source_failure, bindings={"v": SEGMENT})
    inputs["valid"][invalid_row] = 2
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    if scope == "filter":
        np.testing.assert_array_equal(result.statuses, 0)
        np.testing.assert_array_equal(result.values, 1)
    elif scope == "segment":
        expected_status = np.zeros(9, np.int16)
        if source_failure:
            expected_status[1:3] = result.statuses[1:3, 0]
            assert np.all(expected_status[1:3] != 0)
        np.testing.assert_array_equal(result.statuses[:, 0], expected_status)
        np.testing.assert_array_equal(result.values[3:7, 0], True)
        np.testing.assert_array_equal(result.values[[0, 7, 8], 0], False)
    elif source_failure:
        assert result.statuses[0, 0] != 0
    else:
        np.testing.assert_array_equal(result.statuses, 0)
        np.testing.assert_array_equal(result.values, [[2.]])


@pytest.mark.parametrize("uses_observation_count", [False, True], ids=["mean", "observed-count"])
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_rolling_returns_minimum_does_not_dereference_failed_capture(
        uses_observation_count, source_failure):
    inputs, starts, ends = batch()
    inputs = {name: value[:9].copy() for name, value in inputs.items()}
    if not source_failure:
        inputs["x"][:] = np.arange(1., 10.)
    body = "mean(returns)/observation_count" if uses_observation_count else "mean(returns)"
    graph = GraphCompiler({
        "x": "series", "events": EVENT,
        "observation_count": {"kind": "scalar", "dtype": "float64", "semantic_dimension": "count"},
    }).compile(f"rolling_apply({body},3,1)",
               root_bindings=[{"segment": SEGMENT, "returns": GLOBAL_FAILURE}],
               error_policy="isolate")
    assert graph.parameter_names == ()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts[:1], ends[:1])
    if source_failure:
        # The returns descriptor carries known geometry but no readable payload.
        # Counting finite observations must not dereference that descriptor.
        assert np.all(result.statuses != 0)
        assert np.isnan(result.values).all()
    else:
        # Each complete segment's inclusive [left,right] sample is [k,k+1,k+2].
        segment_value = 1 / np.sqrt(2 / 3)
        expected_returns = np.zeros(9)
        for i in range(1, 9):
            expected_returns[i] = (0.8 * expected_returns[i - 1] + 0.2 * segment_value
                                   if i <= 6 else expected_returns[i - 1])
        expected = np.array([expected_returns[max(0, i - 2):i + 1].mean() for i in range(9)])
        if uses_observation_count:
            expected /= np.minimum(np.arange(1, 10), 3)
        np.testing.assert_array_equal(result.statuses, 0)
        np.testing.assert_allclose(result.values[:, 0], expected)


BISECT_STRUCTURAL_ERRORS = [
    pytest.param("solve_x-mean_where(v,valid)", "INVALID_MASK", id="mask"),
    pytest.param("solve_x-filter_apply(mean_where(v,valid),finite_mask(x))",
                 "INVALID_MASK", id="nested-filter"),
    pytest.param("solve_x-sum(recursive_filter(lag(v,1),.2,0,finite_mask(lag(x,2))))",
                 "SHAPE_MISMATCH", id="body-length"),
    pytest.param("solve_x-count_true(state_select(v>0,-2,1,finite_mask(x))==1)",
                 "INVALID_STATE_CODE", id="state-code"),
]


@pytest.mark.parametrize("body,expected_error", BISECT_STRUCTURAL_ERRORS)
@pytest.mark.parametrize("capture", [SEGMENT, GLOBAL_FAILURE], ids=["positional", "global"])
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_bisect_failed_capture_still_validates_body(body, expected_error, capture, source_failure):
    graph, inputs, starts, ends = structural_error_batch(
        f"bisect({body},0,10,1e-12,100)", expected_error,
        source_failure=source_failure, bindings={"v": capture})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match=f"^{expected_error}$"):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_bisect_body_structural_error_survives_worker_transport(lane, source_failure):
    graph, inputs, starts, ends = structural_error_batch(
        "bisect(solve_x-mean_where(v,valid),0,10,1e-12,100)", "INVALID_MASK",
        source_failure=source_failure, bindings={"v": GLOBAL_FAILURE})
    starts, ends = np.repeat(starts, 2), np.repeat(ends, 2)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        assert plan.lane == ("process" if lane.startswith("process") else lane)
        if lane.startswith("process"):
            assert plan.use_shared_memory == (lane == "process_shared")
        error_type = RuntimeError if lane.startswith("process") else ValueError
        message = "^native worker: INVALID_MASK$" if lane.startswith("process") else "^INVALID_MASK$"
        with pytest.raises(error_type, match=message):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize("capture", [SEGMENT, GLOBAL_FAILURE], ids=["positional", "global"])
def test_bisect_prepared_recovers_after_body_validation_and_numerical_failure(capture):
    inputs, starts, ends = batch()
    inputs = {name: value[:9].copy() for name, value in inputs.items()}
    inputs["valid"] = np.ones(9, np.uint8)
    failed_source = inputs["x"].copy()
    graph = GraphCompiler({"x": "series", "events": EVENT,
                           "valid": {"kind": "series", "dtype": "bool"}}).compile(
        ["bisect(solve_x-(2+0*mean_where(v,logical_and(valid,finite_mask(v)))),0,10,1e-12,100)",
         "mean(x)"],
        root_bindings=[{"segment": SEGMENT, "v": capture}, {}], error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, starts[:1], ends[:1])
        for _ in range(2):
            inputs["x"][:] = failed_source
            inputs["valid"][1] = 2
            with pytest.raises(ValueError, match="^INVALID_MASK$"):
                prepared.run_snapshot()
            inputs["valid"][1] = 1
            failed = prepared.run_snapshot()
            assert failed.statuses[0, 0] != 0
            assert np.isnan(failed.values[0, 0])
            assert failed.statuses[0, 1] == 0
            assert failed.values[0, 1] == failed_source.mean()
            inputs["x"][:] = np.arange(1., 10.)
            recovered = prepared.run_snapshot()
            np.testing.assert_array_equal(recovered.statuses, 0)
            np.testing.assert_allclose(recovered.values, [[2., 5.]], atol=1e-12)
            # The next run must not mutate either independently owned snapshot.
            prepared.run()
            assert failed.statuses[0, 0] != 0
            np.testing.assert_allclose(recovered.values, [[2., 5.]], atol=1e-12)


@pytest.mark.parametrize("invalid_mask", [False, True])
@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_bisect_captures_keep_their_individual_lengths(invalid_mask, source_failure):
    expression = "bisect(solve_x-(2+0*count_true(v>0)+0*count_true(valid)),0,10,1e-12,100)"
    graph, inputs, starts, ends = structural_error_batch(
        expression, "INVALID_MASK" if invalid_mask else "", source_failure=source_failure,
        bindings={"v": "lag(segment,1)"})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if invalid_mask:
            with pytest.raises(ValueError, match="^INVALID_MASK$"):
                scheduler.execute(graph, inputs, starts, ends)
        else:
            result = scheduler.execute(graph, inputs, starts, ends)
            if source_failure:
                assert result.statuses[0, 0] != 0
                assert np.isnan(result.values[0, 0])
            else:
                np.testing.assert_array_equal(result.statuses, 0)
                np.testing.assert_allclose(result.values, [[2.]], atol=1e-12)


@pytest.mark.parametrize("source_failure", [False, True], ids=["healthy-source", "failed-source"])
def test_bisect_validates_structure_at_both_endpoints(source_failure):
    # State 0 at the low endpoint is valid; state 1.5 at the high endpoint is
    # structurally invalid. The failed capture must not hide the second check.
    body = ("solve_x-count_true(v>0)+"
            "0*count_true(state_select(finite_mask(x),solve_x,1,finite_mask(x))==1)")
    graph, inputs, starts, ends = structural_error_batch(
        f"bisect({body},0,1.5,1e-12,100)", "", source_failure=source_failure,
        bindings={"v": SEGMENT})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match="^INVALID_STATE_CODE$"):
            scheduler.execute(graph, inputs, starts, ends)


FAILED_SCALAR_SCOPES = [
    pytest.param('bisect(solve_x-n-count_true(valid),0,100,1e-8,100)', id='bisect'),
    pytest.param('sum(block_apply(count_true(valid)-n,3))', id='block'),
    pytest.param('filter_apply(count_true(valid)-n,finite_mask(x))', id='filter'),
    pytest.param('sum(group_apply(count_true(valid)-n,keys))', id='group'),
    pytest.param('sum(segment_apply(count_true(valid)-n,between_events(events)))', id='segment'),
    pytest.param('bisect(solve_x-filter_apply(count_true(valid)-n,finite_mask(x)),0,100,1e-8,100)',
                 id='nested-bisect-filter'),
    pytest.param('bisect(solve_x-sum(v)-count_true(valid),0,100,1e-8,100)', id='unknown-bisect-capture'),
]


@pytest.mark.parametrize('expression', FAILED_SCALAR_SCOPES)
@pytest.mark.parametrize('source_failure', [False, True], ids=['healthy-source', 'failed-source'])
def test_failed_scalar_or_unknown_capture_keeps_independent_scope_mask_checks(expression, source_failure):
    graph, inputs, starts, ends = structural_error_batch(
        expression, 'INVALID_MASK', source_failure=source_failure,
        bindings={'n': FAILED_LENGTH, 'v': 'lag(x,n)'})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match='^INVALID_MASK$'):
            scheduler.execute(graph, inputs, starts, ends)


INDEPENDENT_STATE_ERROR = 'count_true(state_select(v>0,-2,1,finite_mask(v))==1)'
UNKNOWN_SCOPE_CASES = [
    pytest.param(f'sum(block_apply({INDEPENDENT_STATE_ERROR},2))', id='unknown-block-capture'),
    pytest.param(f'filter_apply({INDEPENDENT_STATE_ERROR},finite_mask(lag(x,1)))', id='unknown-filter-capture'),
    pytest.param(f'sum(group_apply({INDEPENDENT_STATE_ERROR},state_select(finite_mask(lag(x,1)),1,1,finite_mask(lag(x,1)))))', id='unknown-group-capture'),
    pytest.param('sum(block_apply(count_true(state_select(x>0,-2,1,finite_mask(x))==1),n))', id='failed-block-width'),
    pytest.param('filter_apply(count_true(state_select(x>0,-2,1,finite_mask(x))==1),selector)', id='failed-selector'),
    pytest.param('rolling_apply(n+count_true(state_select(x>0,-2,1,finite_mask(x))==1),2)', id='rolling-parameter'),
    pytest.param('rolling_apply(count_true(state_select(x>0,-2,1,finite_mask(x))==1),n)', id='rolling-width'),
    pytest.param('bisect(solve_x-count_true(valid),n,100,1e-8,100)', id='bisect-bound-mask'),
]


@pytest.mark.parametrize('expression', UNKNOWN_SCOPE_CASES)
@pytest.mark.parametrize('source_failure', [False, True], ids=['healthy-source', 'failed-source'])
def test_unknown_scope_geometry_still_checks_independent_contracts(expression, source_failure):
    error = 'INVALID_MASK' if 'valid' in expression else 'INVALID_STATE_CODE'
    graph, inputs, starts, ends = structural_error_batch(
        expression, error, source_failure=source_failure,
        bindings={'n': FAILED_LENGTH, 'v': 'lag(x,n)', 'selector': GLOBAL_FAILURE + '>0'})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match=f'^{error}$'):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize('expression', FAILED_SCALAR_SCOPES)
def test_partial_scope_prepared_fault_recovery_keeps_snapshot_and_status(expression):
    graph, inputs, starts, ends = structural_error_batch(
        expression, '', bindings={'n': FAILED_LENGTH, 'v': 'lag(x,n)'})
    bad_source = inputs['x'].copy()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, starts, ends)
        for _ in range(2):
            inputs['x'][:] = bad_source
            bad = prepared.run_snapshot()
            assert np.all(bad.statuses != 0)
            assert np.isnan(bad.values).all()
            inputs['valid'][1] = 2
            with pytest.raises(ValueError, match='^INVALID_MASK$'):
                prepared.run_snapshot()
            inputs['valid'][1] = 1
            inputs['x'][:] = np.arange(1., 10.)
            healthy = prepared.run_snapshot()
            # segment sum legitimately includes incomplete-boundary NaNs.
            reference = scheduler.execute(graph, inputs, starts, ends)
            np.testing.assert_allclose(healthy.values, reference.values, equal_nan=True)
            np.testing.assert_array_equal(healthy.statuses, reference.statuses)
            if 'segment_apply' not in expression:
                np.testing.assert_array_equal(healthy.statuses, 0)
            assert np.all(bad.statuses != 0)


@pytest.mark.parametrize('lane', ['thread', 'process_inline', 'process_shared'])
@pytest.mark.parametrize('unknown_capture', [False, True])
def test_partial_scope_structural_error_crosses_all_worker_lanes(lane, unknown_capture):
    body = 'solve_x-sum(v)-count_true(valid)' if unknown_capture else 'solve_x-n-count_true(valid)'
    graph, inputs, starts, ends = structural_error_batch(
        f'bisect({body},0,100,1e-8,100)', 'INVALID_MASK',
        bindings={'n': FAILED_LENGTH, 'v': 'lag(x,n)'})
    starts, ends = np.repeat(starts, 2), np.repeat(ends, 2)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        assert plan.lane == ('process' if lane.startswith('process') else lane)
        if lane.startswith('process'):
            assert plan.use_shared_memory == (lane == 'process_shared')
        error_type = RuntimeError if lane.startswith('process') else ValueError
        message = '^native worker: INVALID_MASK$' if lane.startswith('process') else '^INVALID_MASK$'
        with pytest.raises(error_type, match=message):
            scheduler.execute(graph, inputs, starts, ends)


@pytest.mark.parametrize('expression', [
    'filter_apply(n+count_true(valid),x>2)',
    'sum(block_apply(n+count_true(valid),4))',
    'sum(segment_apply(n+count_true(valid),between_events(events)))',
])
def test_failed_scalar_preserves_actual_scope_membership(expression):
    graph, inputs, starts, ends = structural_error_batch(expression, '', bindings={'n': FAILED_LENGTH})
    excluded_row = 1 if expression.startswith('filter') else 8 if 'block_apply' in expression else 0
    inputs['valid'][excluded_row] = 2
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    assert np.all(result.statuses != 0)


def test_unresolved_membership_does_not_invent_selected_mask_payload():
    graph, inputs, starts, ends = structural_error_batch(
        'filter_apply(count_true(valid),selector)', 'INVALID_MASK',
        bindings={'selector': GLOBAL_FAILURE + '>0'})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
    assert result.statuses[0, 0] != 0
    assert np.isnan(result.values[0, 0])
