"""One private coefficient, two mathematical roots, independent owned results."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_guarded_lu_graph import EXPECTED_LIBRARY
from .test_solve_library import residual, same_ieee, sample
from .test_solve_pair_reuse import count, pair_graph
from .test_typed_results import config


def fixture(layout="c", n=1025, k1=2, k2=5):
    a, b1 = sample("solve_multi", n, columns=k1, layout=layout)
    _, b2 = sample("solve_multi", n, columns=k2, layout=layout)
    return dict(a=a, b1=b1, b2=b2)


def graph(producer="a+0", order="C", **kwargs):
    return pair_graph(
        expressions=[f"solve_multi({producer},b1)", f"solve_multi({producer},b2)"],
        result_format="typed",
        output_order=order,
        **kwargs,
    )


def run(g, values, scheduler=None, rows=1):
    bounds = np.zeros(rows, np.int64)
    if scheduler is not None:
        return scheduler.execute(g, values, bounds, bounds)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as owner:
        return owner.execute(g, values, bounds, bounds)


def check(result, values, coefficient=None, row=0, order="C"):
    coefficient = values["a"] if coefficient is None else coefficient
    arrays = []
    for root, key in enumerate(["b1", "b2"]):
        actual = result.outputs[root].values[row]
        expected = op.solve_multi(coefficient, values[key])
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=4.9e-5)
        residual(coefficient, actual, values[key])
        assert result.outputs[root].root_statuses[row] == 0
        assert not actual.flags.writeable
        assert actual.flags.c_contiguous if order == "C" else actual.flags.f_contiguous
        for source in values.values():
            assert not np.shares_memory(source, actual)
        arrays.append(actual)
    assert not np.shares_memory(*arrays)
    return arrays


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("producer", ["a+0", "a-0", "a*1"])
def test_shared_factor_layout_and_owned_generations(layout, order, producer):
    values = fixture(layout)
    saved = {k: v.copy() for k, v in values.items()}
    for v in values.values():
        v.flags.writeable = False
    g = pickle.loads(pickle.dumps(graph(producer, order)))
    first = run(g, values)
    snapshots = [v.copy() for v in check(first, values, order=order)]
    second = run(g, values)
    check(second, values, order=order)
    assert count(first, "accelerate_lu_calls") == EXPECTED_LIBRARY
    assert count(first, "input_copy_bytes") == 0
    if EXPECTED_LIBRARY:
        n = values["a"].shape[0]
        k = values["b1"].shape[1] + values["b2"].shape[1]
        assert count(first, "operator_workspace_capacity_bytes") <= (2 * n * k + 5 * n) * 8
        assert count(first, "solve_residual_materialization_bytes") == values["a"].nbytes
        assert count(first, "algorithm_copy_bytes") == 4 * n * k * 8 + (
            values["a"].nbytes if layout == "c" else 0
        )
    for root, snapshot in enumerate(snapshots):
        same_ieee(first.outputs[root].values[0], snapshot)
        for other in second.outputs:
            assert not np.shares_memory(first.outputs[root].values[0], other.values[0])
    for key, value in values.items():
        same_ieee(value, saved[key])


@pytest.mark.parametrize("stride", ["negative_rows", "negative_columns", "zero_rows", "sliced"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_rhs_views_pack_without_mutation(stride, order):
    values = fixture("f")
    for key in ["b1", "b2"]:
        b = values[key]
        if stride == "negative_rows":
            b = b[::-1]
        elif stride == "negative_columns":
            b = b[:, ::-1]
        elif stride == "zero_rows":
            b = np.broadcast_to(b[:1], b.shape)
        else:
            storage = np.empty((b.shape[0] * 2, b.shape[1] * 2))
            storage[::2, ::2] = b
            b = storage[::2, ::2]
        b.flags.writeable = False
        values[key] = b
    result = run(graph(order=order), values)
    check(result, values, order=order)
    assert count(result, "accelerate_lu_calls") == EXPECTED_LIBRARY


@pytest.mark.parametrize("columns", [(1, 1), (32, 32), (0, 2), (64, 1)])
def test_combined_column_bound_retains_individual_fallback(columns):
    values = fixture("f", k1=columns[0], k2=columns[1])
    result = run(graph(order="F"), values)
    check(result, values, order="F")
    if min(columns) and sum(columns) <= 64:
        assert count(result, "accelerate_lu_calls") == EXPECTED_LIBRARY
        if EXPECTED_LIBRARY:
            assert count(result, "solve_residual_materialization_bytes") == values["a"].nbytes


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize(
    "fault", ["singular_before_later_nan", "first_nan", "later_nan", "pivot_replay"]
)
def test_failure_priority_and_coefficient_reconstruction(layout, fault):
    values = fixture(layout, n=1024)
    if fault == "singular_before_later_nan":
        values["a"][-1, :] = 0.0
        values["b2"][-1, 0] = np.nan
    elif fault == "first_nan":
        values["b1"][-1, 0] = np.nan
    elif fault == "later_nan":
        values["b2"][-1, 0] = np.nan
    else:
        values["a"][:] = np.eye(1024) * 2.0
        values["a"][:2, :2] = [[1.0, 0.0], [2.0, 3.0]]
        values["b1"][:] = 1.0
        values["b2"][:] = 1.0
    saved = {k: v.copy() for k, v in values.items()}
    coefficient = op.add(values["a"], 0.0)
    try:
        expected = [op.solve_multi(coefficient, values[key]) for key in ["b1", "b2"]]
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            run(graph(order="F"), values)
    else:
        result = run(graph(order="F"), values)
        for root, want in enumerate(expected):
            same_ieee(result.outputs[root].values[0], want)
        assert count(result, "accelerate_lu_calls") == 0
        assert count(result, "accelerate_lu_fallbacks") >= EXPECTED_LIBRARY
    for key, value in values.items():
        same_ieee(value, saved[key])


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_transport_rows_and_retained_results(lane):
    values = fixture("f", n=1024)
    g = pickle.loads(pickle.dumps(graph(order="F")))
    for value in values.values():
        value.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(values) as bundle:
                first = run(g, bundle, scheduler, 2)
                second = run(g, bundle, scheduler, 2)
        else:
            first = run(g, values, scheduler, 2)
            second = run(g, values, scheduler, 2)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for row in range(2):
        old = check(first, values, row=row, order="F")
        new = check(second, values, row=row, order="F")
        for a in old:
            for b in new:
                assert not np.shares_memory(a, b)


def test_prepared_live_change_failure_and_recovery():
    values = fixture("f", n=1024)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph(order="F"), values, bounds, bounds)
        first = prepared.run_snapshot()
        snapshots = [v.copy() for v in check(first, values, order="F")]
        values["a"][np.diag_indices(1024)] += 2.0
        values["b1"] *= 2.0
        values["b2"] *= 3.0
        check(prepared.run_audit(), values, order="F")
        saved = values["a"].copy()
        values["a"][-1, :] = 0.0
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            prepared.run_audit()
        values["a"][:] = saved
        check(prepared.run_snapshot(), values, order="F")
    for root, snapshot in enumerate(snapshots):
        same_ieee(first.outputs[root].values[0], snapshot)


@pytest.mark.parametrize("case", ["public", "later_consumer", "nested", "isolate"])
def test_graph_exclusions_keep_later_coefficient_consumers(case):
    values = fixture("f", n=1024)
    expressions = ["solve_multi(a+0,b1)", "solve_multi(a+0,b2)"]
    kwargs = {}
    if case == "public":
        expressions.append("a+0")
    elif case == "later_consumer":
        expressions.append("sum_axis(a+0,0)")
    elif case == "nested":
        expressions = ["solve_multi(a*1+0,b1)", "solve_multi(a*1+0,b2)"]
    else:
        kwargs["error_policy"] = "isolate"
    g = pair_graph(expressions=expressions, result_format="typed", output_order="F", **kwargs)
    result = run(g, values)
    check(result, values, order="F")
    if case == "public":
        same_ieee(result.outputs[2].values[0], op.add(values["a"], 0.0))
    elif case == "later_consumer":
        same_ieee(result.outputs[2].values[0], op.sum_axis(op.add(values["a"], 0.0), 0))
    if EXPECTED_LIBRARY and case != "isolate":
        assert count(result, "solve_residual_materialization_bytes") == 2 * values["a"].nbytes
