"""Actual typed graph, guarded library admission, canonical replay and owners."""

import os
import pickle
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_solve_library import residual, same_ieee, sample
from .test_typed_results import config

ACTIVE = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and int(platform.mac_ver()[0].split(".")[0] or 0) >= 15
    and os.environ.get("CME_EXPECT_ACCELERATE") != "0"
    and os.environ.get("CME_EXPECT_EIGEN") != "0"
)
EXPECTED_LIBRARY = int(os.environ.get("CALMETRICS_TEST_EXPECT_ACCELERATE", str(int(ACTIVE))))

TYPES = {
    "a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]),
    "b": dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K"]),
}


def graph(expression="solve_multi(a+0,b)", **kwargs):
    return GraphCompiler(TYPES).compile(expression, result_format="typed", **kwargs)


def count(result, field):
    return sum(chunk[field] for chunk in result.audit["native_chunks"])


def run(g, data, owner=None, rows=1):
    bounds = np.zeros(rows, np.int64)
    if owner is not None:
        return owner.execute(g, data, bounds, bounds)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        return scheduler.execute(g, data, bounds, bounds)


def data(layout="c", n=1024, k=3):
    a, b = sample("solve_multi", n, columns=k, layout=layout)
    return dict(a=a, b=b)


def check(result, inputs, coefficient=None, row=0):
    a = inputs["a"] if coefficient is None else coefficient
    actual = result.outputs[0].values[row]
    expected = op.solve_multi(a, inputs["b"])
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=4.9e-5)
    residual(a, actual, inputs["b"])
    assert result.outputs[0].root_statuses[row] == 0
    assert not actual.flags.writeable
    for value in inputs.values():
        assert not np.shares_memory(actual, value)
    return actual


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("expression", ["a+0", "a-0", "a*1"])
def test_canonical_producers_library_math_and_owners(layout, expression):
    inputs = data(layout)
    saved = {k: v.copy() for k, v in inputs.items()}
    for v in inputs.values():
        v.flags.writeable = False
    result = run(graph(f"solve_multi({expression},b)"), inputs)
    check(result, inputs)
    assert count(result, "accelerate_lu_calls") == EXPECTED_LIBRARY
    assert count(result, "accelerate_lu_attempts") == EXPECTED_LIBRARY
    assert count(result, "accelerate_lu_fallbacks") == 0
    assert count(result, "input_copy_bytes") == 0
    assert (
        count(result, "solve_residual_materialization_bytes")
        == EXPECTED_LIBRARY * inputs["a"].nbytes
    )
    for k, v in inputs.items():
        same_ieee(v, saved[k])


@pytest.mark.parametrize(
    "case",
    [
        "input",
        "public",
        "shared",
        "downstream",
        "nested",
        "negative",
        "sliced",
        "isolate",
        "f_output",
    ],
)
def test_graph_ownership_admission_and_canonical_exclusions(case):
    inputs = data(case if case in ["negative", "sliced"] else "f")
    expression = "solve_multi(a+0,b)"
    kwargs = {}
    if case == "input":
        expression = "solve_multi(a,b)"
    elif case == "public":
        expression = [expression, "a+0"]
    elif case == "shared":
        expression = [expression, "sum_axis(a+0,0)"]
    elif case == "downstream":
        expression += "+0"
    elif case == "nested":
        expression = "solve_multi(a*1+0,b)"
    elif case == "isolate":
        kwargs["error_policy"] = "isolate"
    elif case == "f_output":
        kwargs["output_order"] = "F"
    result = run(graph(expression, **kwargs), inputs)
    if case in ["input", "public", "shared", "f_output"]:
        check(result, inputs)
        assert count(result, "accelerate_lu_calls") == EXPECTED_LIBRARY
    else:
        same_ieee(
            result.outputs[0].values[0], op.solve_multi(op.add(inputs["a"], 0.0), inputs["b"])
        )
        assert count(result, "accelerate_lu_calls") == 0


COUNTEREXAMPLES = [
    {"a": [[1.9248994688602141, 0.5080788550853866], [0.17009577202664072, 0.044896923971490664]]},
    {"a": [[1.452128640053782, 0.7326722651827464], [0.13758180216374685, 0.06941697027304553]]},
    {"a": [[1.0875301670621504, 1.392113783884756], [0.679986121981021, 0.8704292367514931]]},
    {"a": [[1.7562786236179035, 1.070320570242218], [0.4361825612746404, 0.26582067414308574]]},
]


@pytest.mark.parametrize("case", COUNTEREXAMPLES)
@pytest.mark.parametrize("layout", ["c", "f"])
def test_real_near_threshold_counterexamples_embedded_in_large_graph(case, layout):
    a = np.eye(1024) * 2.0
    a[:2, :2] = case["a"]
    inputs = dict(a=np.array(a, order=layout.upper()), b=np.ones((1024, 1), order=layout.upper()))
    try:
        expected = op.solve_multi(op.add(inputs["a"], 0.0), inputs["b"])
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            run(graph(), inputs)
        assert str(actual.value) == str(error)
    else:
        result = run(graph(), inputs)
        same_ieee(result.outputs[0].values[0], expected)
        assert count(result, "accelerate_lu_calls") == 0


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
        "empty",
    ],
)
def test_exact_errors_and_zero_bits(fault):
    inputs = dict(a=np.eye(1024) * 2.0, b=np.ones((1024, 2)))
    if fault in ["rhs_nan", "both_nan"]:
        inputs["b"][-1, 0] = np.nan
    if fault in ["a_nan", "both_nan"]:
        inputs["a"][-1, -1] = np.nan
    if fault == "singular":
        inputs["a"][-1, -1] = 0.0
    if fault in ["below", "equal", "above"]:
        inputs["a"][-1, -1] = {
            "below": np.nextafter(1e-14, 0.0),
            "equal": 1e-14,
            "above": np.nextafter(1e-14, np.inf),
        }[fault]
    if fault == "overflow":
        inputs["a"][-2:, -2:] = [[1e308, 1e308], [-1e308, 1e308]]
    if fault == "zero":
        inputs["b"][:] = -0.0
    if fault == "empty":
        inputs["b"] = np.empty((1024, 0))
    saved = {k: v.copy() for k, v in inputs.items()}
    for value in inputs.values():
        value.flags.writeable = False
    try:
        expected = op.solve_multi(op.add(inputs["a"], 0.0), inputs["b"])
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            run(graph(), inputs)
        assert str(actual.value) == str(error)
    else:
        result = run(graph(), inputs)
        same_ieee(result.outputs[0].values[0], expected)
        assert count(result, "accelerate_lu_calls") == 0
    for key, value in inputs.items():
        same_ieee(value, saved[key])


def test_prepared_live_values_and_recovery():
    inputs = data("f")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph(), inputs, bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        inputs["a"][np.diag_indices(1024)] += 2.0
        inputs["b"] *= 2.0
        second = prepared.run_audit()
        check(second, inputs)
        assert count(second, "accelerate_lu_calls") == EXPECTED_LIBRARY
        good = inputs["a"].copy()
        inputs["a"][-1, :] = 0.0
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            prepared.run_audit()
        inputs["a"][:] = good
        third = prepared.run_snapshot()
        check(third, inputs)
        same_ieee(first.outputs[0].values[0], saved)
        assert not np.shares_memory(first.outputs[0].values[0], third.outputs[0].values[0])


@pytest.mark.parametrize("layout", ["c", "f"])
def test_strict_row_dd_requiring_pivot_replays_original(layout):
    a = np.eye(1024) * 2.0
    a[:2, :2] = [[1.0, 0.0], [2.0, 3.0]]
    inputs = dict(a=np.array(a, order=layout.upper()), b=np.ones((1024, 2), order=layout.upper()))
    result = run(graph(), inputs)
    same_ieee(result.outputs[0].values[0], op.solve_multi(op.add(inputs["a"], 0.0), inputs["b"]))
    assert count(result, "accelerate_lu_calls") == 0
    assert count(result, "accelerate_lu_attempts") == EXPECTED_LIBRARY
    assert count(result, "accelerate_lu_fallbacks") == EXPECTED_LIBRARY


@pytest.mark.parametrize("n,k", [(1023, 2), (1024, 65)])
@pytest.mark.parametrize("layout", ["c", "f"])
def test_geometry_outside_library_domain_preserves_canonical(n, k, layout):
    inputs = data(layout, n, k)
    result = run(graph(), inputs)
    same_ieee(result.outputs[0].values[0], op.solve_multi(op.add(inputs["a"], 0.0), inputs["b"]))
    assert count(result, "accelerate_lu_calls") == 0


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_transport_current_audit_and_two_result_generations(lane):
    inputs = data("f")
    g = pickle.loads(pickle.dumps(graph()))
    for value in inputs.values():
        value.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(inputs) as bundle:
                first = run(g, bundle, scheduler, 2)
                second = run(g, bundle, scheduler, 2)
        else:
            first = run(g, inputs, scheduler, 2)
            second = run(g, inputs, scheduler, 2)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    assert count(first, "accelerate_lu_calls") == 2 * EXPECTED_LIBRARY
    for row in range(2):
        actual = check(first, inputs, row=row)
        check(second, inputs, row=row)
        assert not np.shares_memory(actual, second.outputs[0].values[row])
        assert not np.shares_memory(actual, first.outputs[0].values[1 - row])
