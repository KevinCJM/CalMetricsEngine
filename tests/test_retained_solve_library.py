"""Terminal graph LU never uses retained input/public/shared A as scratch."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_guarded_lu_graph import COUNTEREXAMPLES, EXPECTED_LIBRARY, check, count, data, graph, run
from .test_solve_library import same_ieee
from .test_typed_results import config


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("mode", ["input", "public", "shared", "two_input_roots"])
def test_retained_coefficients_library_results_and_exact_sources(layout, mode):
    values = data(layout)
    saved = {k: v.copy() for k, v in values.items()}
    for v in values.values():
        v.flags.writeable = False
    expression = "solve_multi(a,b)"
    if mode == "public":
        expression = ["solve_multi(a+0,b)", "a+0"]
    elif mode == "shared":
        expression = ["solve_multi(a+0,b)", "sum_axis(a+0,0)"]
    elif mode == "two_input_roots":
        expression = ["solve_multi(a,b)", "solve_multi(a,b*2)"]
    g = graph(expression)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = run(g, values, scheduler)
        second = run(g, values, scheduler)
    check(first, values)
    check(second, values)
    calls = 2 if mode == "two_input_roots" else 1
    assert count(first, "accelerate_lu_calls") == calls * EXPECTED_LIBRARY
    assert count(first, "accelerate_lu_attempts") == calls * EXPECTED_LIBRARY
    assert count(first, "accelerate_lu_fallbacks") == 0
    assert count(first, "input_copy_bytes") == 0
    assert (
        count(first, "solve_residual_materialization_bytes")
        == calls * EXPECTED_LIBRARY * values["a"].nbytes
    )
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    if mode == "public":
        same_ieee(first.outputs[1].values[0], values["a"])
    if mode == "shared":
        same_ieee(first.outputs[1].values[0], op.sum_axis(values["a"], 0))
    if mode == "two_input_roots":
        np.testing.assert_allclose(
            first.outputs[1].values[0],
            op.solve_multi(values["a"], values["b"] * 2),
            rtol=1e-11,
            atol=4.9e-5,
        )
        assert not np.shares_memory(first.outputs[0].values[0], first.outputs[1].values[0])
    for k, v in values.items():
        same_ieee(v, saved[k])


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize(
    "fault",
    [
        "rhs_nan",
        "a_nan",
        "both_nan",
        "singular",
        "below",
        "equal",
        "above",
        "overflow",
        "zero",
        "pivot",
        "cancel_zero",
    ],
)
def test_original_errors_zero_bits_and_attempt_costs(layout, fault):
    a = np.eye(1024) * 2
    b = np.ones((1024, 2))
    if fault in ["rhs_nan", "both_nan"]:
        b[-1, 0] = np.nan
    if fault in ["a_nan", "both_nan"]:
        a[-1, -1] = np.nan
    if fault == "singular":
        a[-1, -1] = 0
    if fault in ["below", "equal", "above"]:
        a[-1, -1] = {
            "below": np.nextafter(1e-14, 0.0),
            "equal": 1e-14,
            "above": np.nextafter(1e-14, np.inf),
        }[fault]
    if fault == "overflow":
        a[-2:, -2:] = [[1e308, 1e308], [-1e308, 1e308]]
    if fault == "zero":
        b[:] = -0.0
    if fault == "pivot":
        a[:2, :2] = [[1.0, 0.0], [2.0, 3.0]]
    if fault == "cancel_zero":
        a[:2, :2] = [[2.0, 1.0], [0.0, 2.0]]
        b[:2, :] = [[1.0, 1.0], [2.0, 2.0]]
    values = dict(a=np.array(a, order=layout.upper()), b=np.array(b, order=layout.upper()))
    saved = {k: v.copy() for k, v in values.items()}
    for v in values.values():
        v.flags.writeable = False
    try:
        expected = op.solve_multi(values["a"], values["b"])
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            run(graph("solve_multi(a,b)"), values)
        assert str(actual.value) == str(error)
    else:
        result = run(graph("solve_multi(a,b)"), values)
        same_ieee(result.outputs[0].values[0], expected)
        assert count(result, "accelerate_lu_calls") == 0
        if fault in ["pivot", "cancel_zero"]:
            assert count(result, "accelerate_lu_attempts") == EXPECTED_LIBRARY
            assert count(result, "accelerate_lu_fallbacks") == EXPECTED_LIBRARY
            if EXPECTED_LIBRARY:
                assert count(result, "native_library_workspace_reserve_bytes") > 0
                copies = (2 if layout == "c" else 1) * values["a"].nbytes + values["b"].nbytes
                copies += values["a"].nbytes + values["b"].nbytes
                assert count(result, "algorithm_copy_bytes") == copies
    for k, v in values.items():
        same_ieee(v, saved[k])


@pytest.mark.parametrize("case", COUNTEREXAMPLES)
@pytest.mark.parametrize("layout", ["c", "f"])
def test_real_singular_threshold_counterexamples_are_retained(case, layout):
    a = np.eye(1024) * 2
    a[:2, :2] = case["a"]
    values = dict(a=np.array(a, order=layout.upper()), b=np.ones((1024, 1)))
    try:
        expected = op.solve_multi(values["a"], values["b"])
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            run(graph("solve_multi(a,b)"), values)
        assert str(actual.value) == str(error)
    else:
        result = run(graph("solve_multi(a,b)"), values)
        same_ieee(result.outputs[0].values[0], expected)
        assert count(result, "accelerate_lu_calls") == 0


@pytest.mark.parametrize("layout", ["c", "f"])
def test_prepared_live_input_changes_recovery_and_two_snapshots(layout):
    values = data(layout)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph("solve_multi(a,b)"), values, bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        values["a"][np.diag_indices(1024)] += 3
        values["b"] *= 2
        second = prepared.run_audit()
        check(second, values)
        assert count(second, "accelerate_lu_calls") == EXPECTED_LIBRARY
        good = values["a"].copy()
        values["a"][-1, :] = 0
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            prepared.run_audit()
        values["a"][:] = good
        third = prepared.run_snapshot()
        check(third, values)
        same_ieee(first.outputs[0].values[0], saved)
        assert not np.shares_memory(first.outputs[0].values[0], third.outputs[0].values[0])


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_native_transport_library_audit_and_result_owners(layout, lane):
    values = data(layout)
    g = pickle.loads(pickle.dumps(graph("solve_multi(a,b)")))
    saved = {k: v.copy() for k, v in values.items()}
    for v in values.values():
        v.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(values) as bundle:
                first = run(g, bundle, scheduler, 2)
                second = run(g, bundle, scheduler, 2)
        else:
            first = run(g, values, scheduler, 2)
            second = run(g, values, scheduler, 2)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    assert count(first, "accelerate_lu_calls") == 2 * EXPECTED_LIBRARY
    for row in range(2):
        check(first, values, row=row)
        check(second, values, row=row)
        assert not np.shares_memory(first.outputs[0].values[row], second.outputs[0].values[row])
    for k, v in values.items():
        same_ieee(v, saved[k])


@pytest.mark.parametrize("layout", ["c", "f", "negative", "sliced"])
@pytest.mark.parametrize("n,k", [(1023, 2), (1024, 65)])
def test_small_strided_and_wide_rhs_canonical_geometry(layout, n, k):
    values = data(layout, n, k)
    result = run(graph("solve_multi(a,b)"), values)
    same_ieee(result.outputs[0].values[0], op.solve_multi(values["a"], values["b"]))
    assert count(result, "accelerate_lu_calls") == 0


def test_planner_accounts_for_copied_coefficients_and_library_work():
    values = data()
    g = graph("solve_multi(a,b)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = run(g, values, scheduler)
    if EXPECTED_LIBRARY:
        workspace = count(result, "operator_workspace_capacity_bytes")
        assert values["a"].nbytes < workspace < 2 * values["a"].nbytes
        assert result.plan.estimated_worker_scratch_bytes >= workspace
        bounds = np.zeros(1, np.int64)
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            with pytest.raises(MemoryError):
                scheduler.execute(
                    g,
                    values,
                    bounds,
                    bounds,
                    memory_budget_bytes=result.plan.estimated_total_memory_bytes - 1,
                )


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("mode", ["input", "private", "public"])
def test_explicit_f_results_use_the_same_library_and_independent_f_storage(layout, mode):
    values = data(layout)
    saved = {k: v.copy() for k, v in values.items()}
    expression = "solve_multi(a,b)" if mode == "input" else "solve_multi(a+0,b)"
    if mode == "public":
        expression = [expression, "a+0"]
    result = run(graph(expression, output_order="F"), values)
    check(result, values)
    assert count(result, "accelerate_lu_calls") == EXPECTED_LIBRARY
    actual = result.outputs[0].values[0]
    assert result.outputs[0].order == "F" and actual.flags.f_contiguous
    assert not actual.flags.writeable
    if mode == "public":
        same_ieee(result.outputs[1].values[0], values["a"])
    for k, v in values.items():
        same_ieee(v, saved[k])


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("mode", ["input", "private"])
@pytest.mark.parametrize("fault", ["pivot", "zero", "singular", "both_nan"])
def test_explicit_f_failures_replay_original_and_never_overwrite_sources(layout, mode, fault):
    a = np.eye(1024) * 2
    b = np.ones((1024, 2))
    if fault == "pivot":
        a[:2, :2] = [[1.0, 0.0], [2.0, 3.0]]
    if fault == "zero":
        b[:] = -0.0
    if fault == "singular":
        a[-1, -1] = 0
    if fault == "both_nan":
        a[-1, -1] = np.nan
        b[-1, 0] = np.nan
    values = dict(a=np.array(a, order=layout.upper()), b=np.array(b, order=layout.upper()))
    saved = {k: v.copy() for k, v in values.items()}
    for v in values.values():
        v.flags.writeable = False
    coefficient = values["a"] if mode == "input" else op.add(values["a"], 0.0)
    expression = "solve_multi(a,b)" if mode == "input" else "solve_multi(a+0,b)"
    try:
        expected = op.solve_multi(coefficient, values["b"])
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            run(graph(expression, output_order="F"), values)
        assert str(actual.value) == str(error)
    else:
        result = run(graph(expression, output_order="F"), values)
        same_ieee(result.outputs[0].values[0], expected)
        assert result.outputs[0].order == "F"
        assert count(result, "accelerate_lu_calls") == 0
    for k, v in values.items():
        same_ieee(v, saved[k])


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("output_order", ["C", "F"])
@pytest.mark.parametrize("mode", ["input", "private"])
@pytest.mark.parametrize("rhs_count", [1, 64])
def test_partial_transpose_and_residual_panels_with_rhs_extremes(
    layout, output_order, mode, rhs_count
):
    # 1025 leaves one row/column in both the 32-cell transpose tile and the
    # 64-cell residual panel; 1/64 cover both admitted RHS extent boundaries.
    values = data(layout, 1025, rhs_count)
    saved = {k: v.copy() for k, v in values.items()}
    for value in values.values():
        value.flags.writeable = False
    expression = "solve_multi(a,b)" if mode == "input" else "solve_multi(a+0,b)"
    first = run(graph(expression, output_order=output_order), values)
    second = run(graph(expression, output_order=output_order), values)
    check(first, values)
    check(second, values)
    actual = first.outputs[0].values[0]
    assert actual.flags.c_contiguous if output_order == "C" else actual.flags.f_contiguous
    assert count(first, "accelerate_lu_calls") == EXPECTED_LIBRARY
    assert not np.shares_memory(actual, second.outputs[0].values[0])
    for key, value in values.items():
        same_ieee(value, saved[key])
