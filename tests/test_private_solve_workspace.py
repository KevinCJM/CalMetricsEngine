"""A dead, materialized coefficient arena is exclusive LU scratch, never an input."""

import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_solve_library import residual, same_ieee, sample
from .test_typed_results import config

DECLARATIONS = {
    "a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]),
    "b": dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K"]),
}


def graph(expression="solve_multi(a+0,b)", **kwargs):
    return GraphCompiler(DECLARATIONS).compile(expression, result_format="typed", **kwargs)


def inputs(n=1024, layout="c", columns=2):
    a, b = sample("solve_multi", n, columns=columns, layout=layout)
    return dict(a=a, b=b)


def count(result, key):
    return sum(chunk[key] for chunk in result.audit["native_chunks"])


def same_result(result, actual, expected):
    # User-authorized floating tail tolerance is confined to admitted terminal
    # library results. Input/retained snapshots and excluded paths stay exact.
    if count(result, "accelerate_lu_calls"):
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=4.9e-5)
    else:
        same_ieee(actual, expected)


def private_copies(result, data, rows=1, layout="c"):
    if count(result, "accelerate_lu_attempts"):
        return rows * (2 * data["b"].nbytes + (data["a"].nbytes if layout == "c" else 0))
    return rows * data["b"].nbytes


def run(g, data, *, scheduler=None, rows=1):
    bounds = np.zeros(rows, np.int64)
    if scheduler is not None:
        return scheduler.execute(g, data, bounds, bounds)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as owner:
        return owner.execute(g, data, bounds, bounds)


def reference(data, expression="add"):
    a = {"add": lambda: op.add(data["a"], 0.),
         "subtract": lambda: op.subtract(data["a"], 0.),
         "multiply": lambda: op.multiply(data["a"], 1.)}[expression]()
    return op.solve_multi(a, data["b"]), a


def check(result, expected, data, *, row=0):
    output = result.outputs[0]
    same_result(result, output.values[row], expected)
    assert output.root_statuses[row] == 0 and output.shape_known[row]
    assert not output.values[row].flags.writeable
    for source in data.values():
        assert not np.shares_memory(output.values[row], source)


@pytest.mark.parametrize("n,layout", [(1023, "c"), (1024, "c"), (1024, "f"),
                                      (1024, "negative"), (1024, "sliced")])
def test_boundary_layout_exact_bits_residual_and_eliminated_copy(n, layout):
    data = inputs(n, layout)
    saved = {key: value.copy() for key, value in data.items()}
    for value in data.values():
        value.flags.writeable = False
    expected, coefficient = reference(data)
    result = run(graph(), data)
    check(result, expected, data)
    residual(coefficient, expected, data["b"])
    copied = (private_copies(result, data, layout=layout) if n >= 1024 else data["b"].nbytes + data["a"].nbytes)
    assert count(result, "algorithm_copy_bytes") == copied
    assert count(result, "input_copy_bytes") == 0
    if n == 1024:
        workspace = count(result, "operator_workspace_capacity_bytes")
        if count(result, "accelerate_lu_attempts"):
            assert 0 < workspace < data["a"].nbytes
        else:
            assert workspace == 0
    for key, value in data.items():
        same_ieee(value, saved[key])


@pytest.mark.parametrize("expression,operation", [("a+0", "add"), ("a-0", "subtract"),
                                                  ("a*1", "multiply")])
def test_materializing_producers_and_downstream_consumer(expression, operation):
    data = inputs()
    expected, _ = reference(data, operation)
    result = run(graph(f"solve_multi({expression},b)+0"), data)
    check(result, op.add(expected, 0.), data)
    assert count(result, "algorithm_copy_bytes") == data["b"].nbytes


def test_interleaved_computed_rhs_does_not_alias_private_coefficients():
    data = inputs()
    rhs = op.add(op.multiply(data["b"], 1.125), 0.25)
    expected = op.solve_multi(op.add(data["a"], 0.), rhs)
    result = run(graph("solve_multi(a+0,b*1.125+0.25)"), data)
    check(result, expected, data)
    assert count(result, "algorithm_copy_bytes") == private_copies(result, dict(a=data["a"], b=rhs))


def test_fused_coefficient_generation_keeps_private_materialized_owner():
    data = inputs()
    coefficient = op.add(op.multiply(data["a"], 1.125), 0.25)
    expected = op.solve_multi(coefficient, data["b"])
    result = run(graph("solve_multi(a*1.125+0.25,b)"), data)
    check(result, expected, data)
    assert count(result, "algorithm_copy_bytes") == data["b"].nbytes


def test_distinct_private_coefficients_rebuild_reused_arena_between_roots():
    data = inputs()
    coefficients = [op.add(data["a"], 0.), op.multiply(data["a"], 1.125)]
    result = run(graph(["solve_multi(a+0,b)", "solve_multi(a*1.125,b)"]), data)
    for index, coefficient in enumerate(coefficients):
        same_result(result, result.outputs[index].values[0], op.solve_multi(coefficient, data["b"]))
        residual(coefficient, result.outputs[index].values[0], data["b"])
    assert count(result, "algorithm_copy_bytes") == private_copies(result, data, rows=2)
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[1].values[0])


@pytest.mark.parametrize("mode", ["input", "view", "public", "shared", "rhs_alias", "isolate", "f_output"])
def test_external_views_public_and_shared_backing_keep_original_copy(mode):
    data = inputs()
    expression = "solve_multi(a+0,b)"
    kwargs = {}
    if mode == "input":
        expression = "solve_multi(a,b)"
    elif mode == "view":
        expression = "solve_multi(transpose(transpose(a+0)),b)"
    elif mode == "public":
        expression = [expression, "a+0"]
    elif mode == "shared":
        expression = [expression, "sum_axis(a+0,0)"]
    elif mode == "rhs_alias":
        expression = "solve_multi(a+0,slice_axis(a+0,1,0,2,1))"
    elif mode == "isolate":
        kwargs["error_policy"] = "isolate"
    else:
        kwargs["output_order"] = "F"
    result = run(graph(expression, **kwargs), data)
    a = op.add(data["a"], 0.)
    rhs = a[:, :2] if mode == "rhs_alias" else data["b"]
    expected = op.solve_multi(data["a"] if mode == "input" else a, rhs)
    check(result, expected, data)
    expected_copies = a.nbytes + rhs.nbytes
    if count(result, "accelerate_lu_attempts"):
        # Retained C coefficients require a private copy and in-place F
        # conversion; RHS and result each cross their actual layout boundary.
        expected_copies = 2 * a.nbytes + 2 * rhs.nbytes
        if not count(result, "accelerate_lu_calls"):
            expected_copies += a.nbytes
        if mode == "f_output" and count(result, "accelerate_lu_calls"):
            expected_copies = a.nbytes + 2 * rhs.nbytes
    elif mode == "f_output" and count(result, "root_copy_bytes"):
        # A backend without a direct canonical F solve computes in C staging.
        # Its exclusive C coefficient arena remains reusable; the independent
        # F result conversion is separately recorded as one root copy.
        assert count(result, "root_copy_bytes") == rhs.nbytes
        expected_copies = rhs.nbytes
    assert count(result, "algorithm_copy_bytes") == expected_copies
    if mode == "public":
        same_ieee(result.outputs[1].values[0], a)
    if mode == "f_output":
        assert result.outputs[0].order == "F"


@pytest.mark.parametrize("case", ["cross_panel_pivot", "equal_pivots", "signed_zero", "empty_rhs"])
def test_pivots_zero_bits_and_empty_rhs(case):
    n = 1024
    data = dict(a=np.eye(n) * 2., b=np.ones((n, 2)))
    if case == "cross_panel_pivot":
        indices = [32, 33, 65]
        data["a"][np.ix_(indices, indices)] = [[4., 1., 2.], [1., 1., 7.], [2., 5., 3.]]
        data["b"][indices] = [[1., 4.], [2., 5.], [3., 6.]]
    elif case == "equal_pivots":
        data["a"][0, 0], data["a"][1, 0], data["a"][65, 0] = 1., -1., 1.
    elif case == "signed_zero":
        data["a"][data["a"] == 0.] = -0.
        data["b"][:] = -0.
    else:
        data["b"] = np.empty((n, 0))
    expected, _ = reference(data)
    result = run(graph(), data)
    check(result, expected, data)
    assert count(result, "algorithm_copy_bytes") == private_copies(result, data)


@pytest.mark.parametrize("fault", ["singular", "matrix_nan", "rhs_nan", "both_nan",
                                   "elimination_overflow", "backsolve_overflow"])
def test_fault_contract_and_readonly_sources(fault):
    data = dict(a=np.eye(1024) * 2., b=np.ones((1024, 2)))
    if fault == "singular":
        data["a"][-1, -1] = 0.
    elif fault == "matrix_nan":
        data["a"][-1, -1] = np.nan
    elif fault == "rhs_nan":
        data["b"][-1, 0] = np.nan
    elif fault == "both_nan":
        data["a"][-1, -1] = data["b"][-1, 0] = np.nan
    elif fault == "elimination_overflow":
        maximum = np.finfo(np.float64).max
        data["a"][-2:, -2:] = [[maximum, maximum], [-maximum, maximum]]
    else:
        data["a"][-1, -1], data["b"][-1, 0] = 2e-14, 1e308
    saved = {key: value.copy() for key, value in data.items()}
    for value in data.values():
        value.flags.writeable = False
    with pytest.raises(ValueError) as want:
        reference(data)
    with pytest.raises(ValueError) as actual:
        run(graph(), data)
    assert str(actual.value) == str(want.value)
    for key, value in data.items():
        same_ieee(value, saved[key])


@pytest.mark.parametrize("position", ["below", "equal", "above"])
def test_exact_pivot_threshold(position):
    data = dict(a=np.eye(1024), b=np.ones((1024, 2)))
    data["a"][-1, -1] = {"below": np.nextafter(1e-14, 0.), "equal": 1e-14,
                            "above": np.nextafter(1e-14, np.inf)}[position]
    if position == "above":
        result = run(graph(), data)
        check(result, reference(data)[0], data)
        assert count(result, "algorithm_copy_bytes") == data["b"].nbytes
    else:
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            run(graph(), data)


def test_prepared_rebuild_live_coefficients_failure_recovery_and_snapshot_ownership():
    data = inputs(layout="f")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph(), data, bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        data["a"][np.diag_indices(1024)] += 3.
        data["b"] *= 2.
        second = prepared.run_audit()
        expected, _ = reference(data)
        check(second, expected, data)
        assert count(second, "algorithm_copy_bytes") == private_copies(second, data, layout="f")
        valid = data["a"].copy()
        data["a"][-1, :] = 0.
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            prepared.run_audit()
        data["a"][:] = valid
        recovered = prepared.run_snapshot()
        check(recovered, expected, data)
        check(first, saved, data)
        assert not np.shares_memory(first.outputs[0].values[0], recovered.outputs[0].values[0])
    same_ieee(first.outputs[0].values[0], saved)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_transport_exclusive_arenas_and_retained_independent_results(lane):
    data = inputs(layout="f")
    saved = {key: value.copy() for key, value in data.items()}
    for value in data.values():
        value.flags.writeable = False
    g = pickle.loads(pickle.dumps(graph()))
    expected, _ = reference(data)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(data) as bundle:
                result = run(g, bundle, scheduler=scheduler, rows=2)
                again = run(g, bundle, scheduler=scheduler, rows=2)
        else:
            result = run(g, data, scheduler=scheduler, rows=2)
            again = run(g, data, scheduler=scheduler, rows=2)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    assert count(result, "algorithm_copy_bytes") == private_copies(result, data, rows=2, layout="f")
    retained = []
    for row in range(2):
        check(result, expected, data, row=row)
        check(again, expected, data, row=row)
        current = result.outputs[0].values[row]
        assert not np.shares_memory(current, result.outputs[0].values[1 - row])
        assert not np.shares_memory(current, again.outputs[0].values[row])
        retained.append((current, current.copy()))
    for key, value in data.items():
        same_ieee(value, saved[key])
    del result, again, scheduler
    gc.collect()
    for value, original_snapshot in retained:
        same_ieee(value, original_snapshot)
        assert not value.flags.writeable
