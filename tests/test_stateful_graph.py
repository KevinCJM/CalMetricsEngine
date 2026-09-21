"""Stateful compositions through the actual compiler, scopes and native scheduler."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler, PlannerConfig

EVENT = {"kind": "series", "dtype": "int64", "axes": ["time"], "shape": ["T"],
         "semantic_dimension": "event"}
STATE = {**EVENT, "semantic_dimension": "state"}


def evaluate(graph, inputs, *, lane="single", starts=None, ends=None):
    size = len(next(iter(inputs.values())))
    settings = PlannerConfig(thread_work_units=1 if lane == "thread" else 1e99,
                             process_work_units=1 if lane == "process" else 1e100,
                             shared_memory_threshold_bytes=1, min_rows_per_worker=1)
    with AdaptiveScheduler(cpu_budget=2, config=settings) as scheduler:
        return scheduler.execute(graph, inputs,
                                 np.array([0] if starts is None else starts, np.int64),
                                 np.array([size] if ends is None else ends, np.int64))


@pytest.mark.parametrize("lane", ["single", "thread", "process"])
def test_complete_segments_include_both_endpoints_and_broadcast_half_open(lane):
    prices = np.array([11., 10., 12., 14., 13., 9., 10., 15., 12.])
    events = np.array([0, -1, 0, 1, 0, -1, 0, 1, 0], np.int64)
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile([
        "segment_apply(last(x)/first(x)-1,between_events(events))",
        "segment_apply(length(x),between_events(events))",
    ], error_policy="isolate")
    graph = pickle.loads(pickle.dumps(graph))
    result = evaluate(graph, {"x": prices, "events": events}, lane=lane, starts=[0, 0], ends=[9, 9])
    expected = np.array([[np.nan, np.nan], [.4, 3], [.4, 3], [-5/14, 3], [-5/14, 3],
                         [2/3, 3], [2/3, 3], [np.nan, np.nan], [np.nan, np.nan]])
    np.testing.assert_allclose(result.values, np.tile(expected, (2, 1)), equal_nan=True)
    np.testing.assert_array_equal(result.offsets, [0, 9, 18])
    np.testing.assert_array_equal(result.statuses[:, 0], np.tile([4, 0, 0, 0, 0, 0, 0, 4, 4], 2))
    assert result.audit["python_fallback"] == 0
    assert result.plan.lane == lane
    assert all(chunk["input_copy_bytes"] == 0 for chunk in result.audit["native_chunks"])


def test_segment_empty_missing_break_and_scope_failure_isolation():
    events = np.array([-1, 0, 1, -2, -1, 0, 1, 0], np.int64)
    x = np.array([2., 2., 2., np.nan, 2., 3., 4., 5.])
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile([
        "segment_apply(1/std(x,0),between_events(events))",
        "segment_apply(mean(x),between_events(events))",
    ], error_policy="isolate")
    result = evaluate(graph, {"x": x, "events": events})
    np.testing.assert_allclose(result.values[:, 0], [np.nan]*4 + [np.sqrt(1.5)]*2 + [np.nan]*2,
                               equal_nan=True)
    np.testing.assert_allclose(result.values[:, 1], [2, 2, np.nan, np.nan, 3, 3, np.nan, np.nan],
                               equal_nan=True)
    assert np.all(result.statuses[:2, 0] != 0)
    assert np.all(result.statuses[:2, 1] == 0)
    empty = evaluate(graph, {"x": x[:0], "events": events[:0]})
    assert empty.values.shape == (0, 2)


def test_segment_scope_rejects_untyped_boundaries_and_misaligned_captures():
    with pytest.raises(GraphCompileError, match="segment record"):
        GraphCompiler({"x": "series", "events": EVENT}).compile(["segment_apply(mean(x),events)"])
    with pytest.raises(GraphCompileError, match="aligned"):
        GraphCompiler({"x": "series", "events": EVENT}).compile(
            ["segment_apply(mean(y),between_events(events))"], root_bindings=[{"y": "lag(x)"}])


def test_shared_kalman_projection_survives_liveness_reuse_and_nominal_type_check():
    x = np.linspace(10, 30, 24)
    expressions = ["state_estimate(k)", "state_variance(k)", "x*3+2"]
    graph = GraphCompiler({"x": "series"}).compile(expressions,
        root_bindings=[{"k": "scalar_kalman(x,.1,.5)"}] * 3)
    actual = evaluate(graph, {"x": x}).values
    estimate, variance = x[0], 1.
    expected = [[estimate, variance]]
    for observation in x[1:]:
        predicted = variance + .1
        gain = predicted / (predicted + .5)
        estimate += gain * (observation - estimate)
        variance = (1-gain)*predicted
        expected.append([estimate, variance])
    np.testing.assert_allclose(actual[:, :2], expected)
    np.testing.assert_array_equal(actual[:, 2], x*3+2)
    with pytest.raises(GraphCompileError, match="nominal"):
        GraphCompiler({"x": {"kind": "matrix", "axes": ["time", "asset"],
                             "shape": ["T", "2"]}}).compile(["state_estimate(x)"])
    with pytest.raises(GraphCompileError, match="numeric"):
        GraphCompiler({"x": "series"}).compile(["sum(scalar_kalman(x,.1,.5))"])


def test_state_signals_compare_exact_codes_and_keep_boolean_dtype():
    codes = np.array([2**60, 2**60+1, -1, 0], np.int64)
    other = np.array([2**60+1, 2**60+1, -1, 1], np.int64)
    graph = GraphCompiler({"x": STATE, "y": STATE}).compile(["equal(x,y)", "equal(x,-1)"])
    result = evaluate(graph, {"x": codes, "y": other})
    assert result.values.dtype == np.bool_
    np.testing.assert_array_equal(result.values, [[False, False], [True, False], [True, True], [False, False]])
    with pytest.raises(GraphCompileError, match="code"):
        GraphCompiler({"x": "series", "y": STATE}).compile(["equal(x,y)"])


def test_three_way_visible_classification_connects_to_confirmation():
    x = np.array([2., 3., 0., -2., -3., np.nan, 3., 2.])
    candidate = "state_select(x>=1,0,state_select(x<=-1,2,1,finite_mask(x)),finite_mask(x))"
    graph = GraphCompiler({"x": "series"}).compile([candidate, f"state_confirm({candidate},2,1)"])
    result = evaluate(graph, {"x": x})
    np.testing.assert_array_equal(result.values[:, 0], [0, 0, 1, 2, 2, -1, 0, 0])
    np.testing.assert_array_equal(result.values[:, 1], [-1, 0, 0, 0, 2, -1, 2, 0])


def test_kama_is_a_visible_native_composition_with_gap_reset():
    x = np.array([10., 12., 11., 15., 14., 20., np.nan, 8., 9., 7., 10., 12.])
    noise = "sum(abs(difference(x)))"
    # Express the safe denominator visibly, with no eager invalid division.
    efficiency = f"where({noise}>0,abs(last(x)-first(x))/maximum({noise},1e-300),0)"
    alpha = f"rolling_apply(({efficiency}*(2/3-2/31)+2/31)**2,4)"
    expr = "recursive_filter_adaptive(x,a,finite_mask(a),logical_not(finite_mask(x)),0,1,4,1)"
    graph = GraphCompiler({"x": "series"}).compile([expr], root_bindings=[{"a": alpha}])
    actual = evaluate(graph, {"x": x}).values[:, 0]
    expected = np.full(len(x), np.nan)
    start, previous = 0, 0.
    for t, value in enumerate(x):
        if not np.isfinite(value):
            start = t + 1
            continue
        if t == start:
            previous = value
        if t - start < 3:
            continue
        part = x[t-3:t+1]
        noise_value = np.abs(np.diff(part)).sum()
        er = abs(part[-1]-part[0])/noise_value if noise_value > 0 else 0
        gain = (er*(2/3-2/31)+2/31)**2
        previous += gain*(value-previous)
        expected[t] = previous
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_super_smoother_coefficients_and_two_state_recurrence_stay_in_graph():
    x = np.array([10., 11., 9., 12., 15., np.nan, 10., 8., 11., 14., 13.])
    a = "exp(-sqrt(2)*3.141592653589793/4)"
    b = f"2*{a}*cos(sqrt(2)*3.141592653589793/4)"
    c = f"(1-({b})+({a})**2)/2"
    expression = f"linear_filter2(x,{c},{c},{b},-({a})**2,logical_not(finite_mask(x)),4,1)"
    graph = GraphCompiler({"x": "series"}).compile([expression])
    expected = np.full(x.size, np.nan)
    a_value = np.exp(-np.sqrt(2)*np.pi/4)
    b_value = 2*a_value*np.cos(np.sqrt(2)*np.pi/4)
    c_value = (1-b_value+a_value*a_value)/2
    state = []
    for t, value in enumerate(x):
        if not np.isfinite(value):
            state = []
            continue
        y = value if len(state) < 2 else c_value*value+c_value*x[t-1]+b_value*state[-1]-a_value*a_value*state[-2]
        state.append(y)
        if len(state) >= 4:
            expected[t] = y
    np.testing.assert_allclose(evaluate(graph, {"x": x}).values[:, 0], expected, equal_nan=True)


def test_segment_resource_budget_does_not_become_missing_result():
    graph = GraphCompiler({"x": "series", "events": EVENT}).compile(
        ["segment_apply(mean(x),between_events(events))"], error_policy="isolate", scope_work_budget=1)
    with pytest.raises((ValueError, RuntimeError), match="SCOPE_COMPUTE_BUDGET_EXCEEDED"):
        evaluate(graph, {"x": np.array([1., 2., 3.]), "events": np.array([-1, 0, 1], np.int64)})


def test_record_metadata_names_fields_and_variance_unit():
    graph = GraphCompiler({"x": {"kind": "series", "semantic_dimension": "adjusted_nav"}}).compile([
        "state_estimate(scalar_kalman(x,.1,.5))", "state_variance(scalar_kalman(x,.1,.5))"])
    root_types = graph.metadata()["root_types"]
    assert root_types[0]["semantic_dimension"] == "adjusted_nav"
    assert root_types[1]["semantic_dimension"] == "squared:adjusted_nav"
    with pytest.raises(GraphCompileError, match="dimensionless"):
        GraphCompiler({"x": {"kind": "series", "semantic_dimension": "adjusted_nav"}}).compile(["cos(x)"])


def test_reference_cycle_requires_phase_codes_not_business_states():
    compiler = GraphCompiler({"x": "series", "events": EVENT})
    segments = "between_events(events)"
    changes = f"segment_apply(last(x)/first(x)-1,{segments})"
    neutral = "state_hysteresis(x,1000,1000,-1000,-1000)"
    with pytest.raises(GraphCompileError, match="phase direction"):
        compiler.compile([f"drawdown_cycle_reference({neutral},{changes},{segments},.1)"])
    phases = f"phase_direction(events,{segments})"
    phase_graph = compiler.compile([phases])
    assert phase_graph.metadata()["root_types"][0]["semantic_dimension"] == "phase"
    graph = compiler.compile([f"drawdown_cycle_reference({phases},{changes},{segments},.1)"])
    result = evaluate(graph, {"x": np.array([10.,8.,10.]),
                              "events": np.array([1,-1,1],np.int64)})
    np.testing.assert_array_equal(result.values[:,0], [0,2,1])
    # Declaration-bound external phase codes remain usable without pretending to
    # be ordinary business-state codes or a numerical sequence.
    phase_type = {**STATE, "semantic_dimension": "phase"}
    supplied = GraphCompiler({"phase": phase_type, "changes": "series", "events": EVENT}).compile(
        ["drawdown_cycle_reference(phase,changes,between_events(events),.1)"])
    result = evaluate(supplied, {"phase": np.array([1,0,-1],np.int64),
        "changes": np.array([-.2,.25,np.nan]), "events": np.array([1,-1,1],np.int64)})
    np.testing.assert_array_equal(result.values[:,0], [0,2,1])


@pytest.mark.parametrize("lane", ["single", "thread", "process"])
def test_constant_segment_body_nested_array_keeps_full_scope_capacity(lane):
    size = 257
    events = np.zeros(size,np.int64)
    events[0], events[-1] = -1, 1
    expression = "segment_apply(sum(block_apply(1,1)),between_events(events))"
    compiler = GraphCompiler({"events":EVENT})
    graph = compiler.compile([expression])
    result = evaluate(graph, {"events":events}, lane=lane,
                      starts=[0,0], ends=[size,size])
    expected = np.r_[np.full(size-1,size,dtype=float),np.nan]
    np.testing.assert_allclose(result.values[:,0], np.tile(expected,2),equal_nan=True)
    exhausted = compiler.compile([expression],error_policy="isolate",scope_work_budget=10)
    with pytest.raises((ValueError,RuntimeError),match="SCOPE_COMPUTE_BUDGET_EXCEEDED"):
        evaluate(exhausted,{"events":events})


def test_constant_segment_body_array_scratch_is_in_memory_admission():
    size = 4097
    events = np.zeros(size,np.int64)
    events[0],events[-1] = -1,1
    compiler = GraphCompiler({"events":EVENT})
    constant = compiler.compile(["segment_apply(7,between_events(events))"])
    materialized = compiler.compile([
        "segment_apply(sum(block_apply(1,1)),between_events(events))"])
    starts,ends = np.array([0],np.int64),np.array([size],np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plain = scheduler.plan(constant,{"events":events},starts,ends)
        nested = scheduler.plan(materialized,{"events":events},starts,ends)
        assert nested.estimated_worker_scratch_bytes - plain.estimated_worker_scratch_bytes >= size*8
        assert nested.estimated_work_units > plain.estimated_work_units
        with pytest.raises(MemoryError):
            scheduler.plan(materialized,{"events":events},starts,ends,
                           memory_budget_bytes=plain.estimated_total_memory_bytes)
