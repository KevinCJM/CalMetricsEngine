"""A private MatMul may populate its transposed F root without a matrix copy."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import declarations, settings
from .test_matmul_packed_library import assert_dispatch


def graph(expression="transpose(matmul(a,b))", *, order="auto", isolate=False):
    return GraphCompiler(declarations()).compile(
        expression,
        output_order=order,
        error_policy="isolate" if isolate else "raise",
    )


def matrices(m=33, k=17, n=35, layout="cc"):
    rng = np.random.default_rng(9030)
    a = np.array(rng.normal(size=(m, k)), order=layout[0].upper())
    b = np.array(rng.normal(size=(k, n)), order=layout[1].upper())
    a.flags.writeable = b.flags.writeable = False
    return a, b


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["auto", "F"])
def test_private_transposed_matmul_writes_independent_f_root(layout, lane, order):
    a, b = matrices(layout=layout)
    expected = op.matmul(a, b).T
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        compiled = graph(order=order)
        if lane == "single":
            prepared = scheduler.prepare_execution(compiled, {"a": a, "b": b}, bounds, bounds)
            first = prepared.run_snapshot()
            second = prepared.run_snapshot()
        else:
            first = scheduler.execute(compiled, {"a": a, "b": b}, bounds, bounds)
            second = scheduler.execute(compiled, {"a": a, "b": b}, bounds, bounds)
    for result in (first, second):
        column = result.outputs[0]
        assert column.order == "F"
        for value in column.values:
            assert value.flags.f_contiguous and not value.flags.writeable
            np.testing.assert_array_equal(value, expected)
        for chunk in result.audit["native_chunks"]:
            assert chunk["root_copy_bytes"] == 0
            assert chunk["numeric_arena_bytes"] == 0
            assert chunk["direct_output_bytes"] == chunk["rows"] * expected.nbytes
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    assert not np.shares_memory(first.outputs[0].values[0], a)
    assert not np.shares_memory(first.outputs[0].values[0], b)


def test_large_matmul_transpose_keeps_bounded_product_and_removes_full_arena():
    a, b = matrices(1025, 257, 1027, "ff")
    a = a * 1024.0
    b = b * 1024.0
    a.flags.writeable = b.flags.writeable = False
    product, direct_audit = op.matmul(a, b, audit=True)
    assert_dispatch(direct_audit)
    expected = product.T
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(), {"a": a, "b": b}, bounds, bounds)
    actual = result.outputs[0].values[0]
    assert actual.flags.f_contiguous
    chunk = result.audit["native_chunks"][0]
    if chunk["accelerate_matmul_calls"]:
        assert chunk["accelerate_matmul_calls"] > 1
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.000049)
        with localcontext() as context:
            context.prec = 80
            for i, j in [(0, 0), (17, 23), (a.shape[0] - 1, b.shape[1] - 1)]:
                exact = sum(
                    (
                        Decimal.from_float(float(x)) * Decimal.from_float(float(y))
                        for x, y in zip(a[i], b[:, j], strict=False)
                    ),
                    Decimal(0),
                )
                assert abs(Decimal.from_float(float(actual[j, i])) - exact) <= Decimal.from_float(
                    0.000049
                )
        assert chunk["algorithm_copy_bytes"] == 0
    else:
        np.testing.assert_array_equal(actual, expected)
        assert chunk["algorithm_copy_bytes"] == direct_audit["algorithm_copy_bytes"]
    assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
    assert chunk["direct_output_bytes"] == expected.nbytes


def test_transposed_root_keeps_ieee_bits_and_discrete_comparisons():
    a = np.zeros((33, 35), order="F")
    b = np.ones((35, 37), order="F")
    a[:, :3] = [1e16, 1.0, -1e16]
    a[1, 7] = np.inf
    b[7] = 0.0
    a[2] = -0.0
    a.flags.writeable = b.flags.writeable = False
    expected = op.matmul(a, b).T
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(order="F"), {"a": a, "b": b}, bounds, bounds)
    actual = result.outputs[0].values[0]
    np.testing.assert_array_equal(
        np.asarray(actual, order="C").view(np.uint64),
        np.asarray(expected, order="C").view(np.uint64),
    )
    np.testing.assert_array_equal(actual > 0, expected > 0)


def test_planner_admits_exactly_one_less_matrix_for_f_root():
    a, b = matrices()
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        estimates = {
            order: scheduler.plan(
                graph(order=order), {"a": a, "b": b}, bounds, bounds
            ).estimated_total_memory_bytes
            for order in ("C", "F", "auto")
        }
    assert estimates["F"] == estimates["auto"]
    assert estimates["C"] - estimates["F"] == a.shape[0] * b.shape[1] * 8


@pytest.mark.parametrize(
    "order,isolate,multiple",
    [
        ("C", False, False),
        ("F", True, False),
        ("F", False, True),
    ],
)
def test_nonprivate_or_c_order_keeps_existing_materialization(order, isolate, multiple):
    a, b = matrices()
    expression = ["transpose(matmul(a,b))", "matmul(a,b)"] if multiple else "transpose(matmul(a,b))"
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            graph(expression, order=order, isolate=isolate), {"a": a, "b": b}, bounds, bounds
        )
    np.testing.assert_array_equal(result.outputs[0].values[0], op.matmul(a, b).T)
    chunk = result.audit["native_chunks"][0]
    assert chunk["root_copy_bytes"] >= a.shape[0] * b.shape[1] * 8
    assert chunk["numeric_arena_bytes"] >= a.shape[0] * b.shape[1] * 8
