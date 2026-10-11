"""F-layout Eigen column reductions and mixed-DAG broadcast arithmetic."""

import os
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_typed_results import config


def enabled():
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and int(platform.mac_ver()[0].split(".")[0]) >= 15
    )


def identical(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view("u8"), expected[valid].view("u8"))


@pytest.mark.parametrize("name", ["sum_axis", "mean_axis", "variance_axis"])
@pytest.mark.parametrize("shape", [(31, 8), (32, 8), (65, 9), (257, 65), (128, 128)])
@pytest.mark.parametrize("layout", ["f", "padded", "c", "reverse_rows", "reverse_columns"])
def test_ordered_column_reductions(name, shape, layout):
    x = np.asfortranarray(np.random.default_rng(173).normal(size=shape))
    if layout == "padded":
        store = np.empty((shape[0] + 3, shape[1]), order="F")
        store[: shape[0]] = x
        x = store[: shape[0]]
    elif layout == "c":
        x = np.ascontiguousarray(x)
    elif layout == "reverse_rows":
        x = x[::-1]
    elif layout == "reverse_columns":
        x = x[:, ::-1]
    x.flags.writeable = False
    saved = x.copy()
    args = (x, 0, 1) if name == "variance_axis" else (x, 0)
    expected, scalar = op.call(name, *args, simd="scalar", audit=True)
    work = op.Workspace()
    actual, audit = op.call(name, *args, workspace=work, audit=True)
    identical(actual, expected)
    assert scalar["isa"] == "scalar"
    eligible = (
        enabled() and shape[0] >= 32 and shape[1] >= 8 and layout in ("f", "padded", "reverse_rows")
    )
    if eligible and os.environ.get("CME_EXPECT_EIGEN") == "1":
        assert audit["isa"] == "eigen-array"
    for field in ("workspace_bytes", "algorithm_copy_bytes", "input_copy_bytes"):
        assert audit[field] == scalar[field] == 0
    again = op.call(name, *args, workspace=work)
    identical(actual, expected)
    assert not np.shares_memory(actual, again)
    assert not np.shares_memory(actual, x)
    identical(x, saved)


@pytest.mark.parametrize("name", ["sum_axis", "mean_axis", "variance_axis"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e308])
@pytest.mark.parametrize("reverse", [False, True])
def test_column_nonfinite_contract_distinguishes_reduction(name, bad, reverse):
    x = np.ones((65, 17), order="F")
    if reverse:
        x = x[::-1]
    x[-1, 8] = bad
    if name == "variance_axis":
        with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
            op.call(name, x, 0)
    else:
        identical(op.call(name, x, 0), op.call(name, x, 0, simd="scalar"))


@pytest.mark.parametrize("first", ["add", "subtract", "multiply"])
@pytest.mark.parametrize("second", ["add", "subtract", "multiply", "divide"])
@pytest.mark.parametrize("matrix_left", [False, True])
@pytest.mark.parametrize("middle_left", [False, True])
@pytest.mark.parametrize("shape", [(65, 9), (32, 8), (33, 12), (36, 13), (35, 15)])
@pytest.mark.parametrize("reverse", [False, True])
def test_fused_column_pair_preserves_operand_order(
    first, second, matrix_left, middle_left, shape, reverse
):
    # Exact blocks, row-only/column-only tails and both tails must keep the
    # same IEEE behavior and operand ordering through the public graph.
    x = np.asfortranarray(np.random.default_rng(173).normal(size=shape))
    if reverse:
        x = x[::-1]
    x[0, :6] = [0.0, -0.0, 1.0 - 2.0**-27, np.nan, np.inf, -np.inf]
    a = np.full(shape[1], 1.0 + 2.0**-27)
    b = np.full(shape[1], -1.0)
    # Signed broadcast stride also participates in the library path.
    a = a[::-1]
    arguments = (x, op.broadcast_like(a, x)) if matrix_left else (op.broadcast_like(a, x), x)
    middle = op.call(first, *arguments)
    zero_denominator = second == "divide" and not middle_left and np.any(middle == 0)
    expected = (
        None
        if zero_denominator
        else (
            op.call(second, middle, op.broadcast_like(b, x))
            if middle_left
            else op.call(second, op.broadcast_like(b, x), middle)
        )
    )
    values = dict(x=x, a=a, b=b)
    types = {
        "x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"]),
        "a": dict(kind="vector", axes=["factor"], shape=["D"]),
        "b": dict(kind="vector", axes=["factor"], shape=["D"]),
    }
    fa = f"{first}(x,broadcast_like(a,x))" if matrix_left else f"{first}(broadcast_like(a,x),x)"
    expr = (
        f"{second}({fa},broadcast_like(b,x))"
        if middle_left
        else f"{second}(broadcast_like(b,x),{fa})"
    )
    graph = GraphCompiler(types).compile([expr, "sum_axis(a,0)"], result_format="typed")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        if zero_denominator:
            with pytest.raises((RuntimeError, ValueError), match="DIVIDE_BY_ZERO"):
                engine.execute(graph, values, bounds, bounds)
            return
        result = engine.execute(graph, values, bounds, bounds)
    identical(result.outputs[0].values[0], expected)
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] == 0
    assert result.audit["native_chunks"][0]["fused_pointwise_calls"] > 0


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("isolate", [False, True])
def test_column_standardization_transport_and_owned_outputs(lane, isolate):
    x = np.asfortranarray(np.random.default_rng(518).normal(size=(65, 9)))
    m = "mean_axis(x,0)"
    s = "sqrt(variance_axis(x,0))"
    graph = GraphCompiler(
        {"x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"])}
    ).compile(
        [f"(x-broadcast_like({m},x))/broadcast_like({s},x)", m, s, "7"],
        error_policy="isolate" if isolate else "raise",
        result_format="typed",
    )
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as engine:
        first = engine.execute(graph, {"x": x}, bounds, bounds)
        saved = first.outputs[0].values[0].copy()
        x[:] *= 2
        second = engine.execute(graph, {"x": x}, bounds, bounds)
        np.testing.assert_allclose(
            second.outputs[0].values[0], (x - x.mean(0)) / x.std(0), rtol=1e-12, atol=1e-13
        )
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
        x[:] = 1
        if isolate:
            failed = engine.execute(graph, {"x": x}, bounds, bounds)
            assert np.all(failed.outputs[0].root_statuses != 0)
            assert np.all(failed.outputs[3].root_statuses == 0)
        else:
            with pytest.raises((RuntimeError, ValueError), match="DIVIDE_BY_ZERO"):
                engine.execute(graph, {"x": x}, bounds, bounds)
    identical(first.outputs[0].values[0], saved)


@pytest.mark.parametrize(
    "expression", ["(x*2)+3", "(2-x)/3", "(x+broadcast_like(v,x))*2", "(x-2)/broadcast_like(v,x)"]
)
def test_column_pair_scalar_and_vector_parameters(expression):
    x = np.asfortranarray(np.random.default_rng(92).normal(size=(65, 9)))
    v = np.arange(1.0, 10.0)
    definitions = {
        "x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"]),
        "v": dict(kind="vector", axes=["factor"], shape=["D"]),
    }
    graph = GraphCompiler(definitions).compile(
        [expression, "mean_axis(x,0)"], result_format="typed"
    )
    expected = {
        "(x*2)+3": lambda: (x * 2) + 3,
        "(2-x)/3": lambda: (2 - x) / 3,
        "(x+broadcast_like(v,x))*2": lambda: (x + v) * 2,
        "(x-2)/broadcast_like(v,x)": lambda: (x - 2) / v,
    }[expression]()
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x, "v": v}, bounds, bounds)
    identical(result.outputs[0].values[0], expected)
    assert result.audit["native_chunks"][0]["fused_pointwise_calls"] > 0
