"""Mixed-DAG fusion, shaped arenas and allocation/ownership regressions."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_typed_results import config


def schema(shape):
    return {
        "kind": {1: "vector", 2: "matrix", 3: "tensor"}[len(shape)],
        "axes": ["asset", "factor", "scenario"][: len(shape)],
        "shape": list(shape),
    }


def execute(expressions, data, *, isolate=False, lane="single"):
    definitions = {
        key: dict(schema(value.shape), dtype=str(value.dtype)) for key, value in data.items()
    }
    if "v" in definitions:
        definitions["v"]["axes"] = ["factor"]
    graph = GraphCompiler(definitions).compile(
        expressions, error_policy="isolate" if isolate else "raise"
    )
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        return scheduler.execute(graph, data, np.zeros(2, np.int64), np.zeros(2, np.int64))


@pytest.mark.parametrize("layout", ["c", "reverse", "zero"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("mean", [False, True])
def test_take_reduction_matches_materialization_and_skips_unselected_nan(layout, axis, mean):
    x = np.arange(3 * 5 * 7.0, dtype=np.float64).reshape(3, 5, 7)
    if layout == "reverse":
        x = x[::-1, ::-1, ::-1]
    if layout == "zero":
        x = np.broadcast_to(x[:1], x.shape)
    x = x.copy() if layout == "c" else x
    if layout == "c":
        selected = [slice(None)] * 3
        selected[axis] = 1
        x[tuple(selected)] = np.nan
    ix = np.array([x.shape[axis] - 1, 0, 0], np.int64)
    name = "mean_axis" if mean else "sum_axis"
    expr = f"{name}(take_axis(x,ix,{axis}),{axis})"
    expected = getattr(op, name)(op.take_axis(x, ix, axis), axis)
    result = execute(expr, {"x": x, "ix": ix})
    np.testing.assert_array_equal(result.outputs[0].values[0], expected)
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] == 0
    assert result.audit["native_chunks"][0]["fused_pointwise_calls"] > 0


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_fused_transport_repeated_results_are_independent(lane):
    x = np.arange(6 * 11.0, dtype=np.float64).reshape(6, 11)
    ix = np.array([4, 0, 4], np.int64)
    result = execute("sum_axis(take_axis(x,ix,0),0)", {"x": x, "ix": ix}, lane=lane)
    np.testing.assert_array_equal(result.outputs[0].values[0], x[ix].sum(axis=0))
    assert not np.shares_memory(result.outputs[0].values[0], x)
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[0].values[1])


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("isolate", [False, True])
def test_shared_or_failed_selection_retains_validation(shared, isolate):
    x = np.arange(35.0, dtype=np.float64).reshape(5, 7)
    ix = np.array([4, 2, 0], np.int64)
    take = "take_axis(x,ix,0)"
    expressions = [f"sum_axis({take},0)"] + ([take] if shared else [])
    result = execute(expressions, {"x": x, "ix": ix}, isolate=isolate)
    np.testing.assert_array_equal(result.outputs[0].values[0], x[ix].sum(axis=0))
    if shared:
        np.testing.assert_array_equal(result.outputs[1].values[0], x[ix])
    ix[1] = 5
    with pytest.raises((RuntimeError, ValueError), match="INDEX_OUT_OF_BOUNDS"):
        execute(expressions, {"x": x, "ix": ix}, isolate=isolate)


@pytest.mark.parametrize("first", ["+", "-", "*"])
@pytest.mark.parametrize("second", ["+", "-", "*", "/"])
@pytest.mark.parametrize("left", [False, True])
def test_arithmetic_pairs_preserve_order_strides_and_roots(first, second, left):
    x = np.array([[0.0, -0.0, 1.0 - 2.0**-27, np.inf], [2.0, 3.0, np.nan, -np.inf]])[:, ::-1]
    v = np.array([1.0 + 2.0**-27, 2.0, -1.0, 3.0])[::-1]
    a = f"x{first}broadcast_like(v,x)"
    b = f"({a}){second}broadcast_like(v,x)" if left else f"broadcast_like(v,x){second}({a})"
    funcs = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}
    with np.errstate(all="ignore"):
        middle = funcs[first](x, v)
        expected = funcs[second](middle, v) if left else funcs[second](v, middle)
    # Zero denominators belong to the original throwing domain.
    if second == "/" and not left and np.any(middle == 0):
        with pytest.raises((RuntimeError, ValueError), match="DIVIDE_BY_ZERO"):
            execute([b, "sum_axis(v,0)"], {"x": x, "v": v})
        return
    result = execute([b, "sum_axis(v,0)"], {"x": x, "v": v})
    actual = result.outputs[0].values[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        np.signbit(actual[expected == 0]), np.signbit(expected[expected == 0])
    )
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] == 0


def test_intervening_arena_reuse_cannot_corrupt_deferred_sources():
    x = np.arange(63.0, dtype=np.float64).reshape(7, 9)
    # Backing of broadcast(mean_axis(...)) is an arena slot, not an input/root.
    # The later branches can reuse it; this producer must stay materialized.
    expr = "(x-broadcast_like(mean_axis(x,0),x))/(sqrt(x*x)+1)"
    result = execute(expr, {"x": x})
    np.testing.assert_allclose(result.outputs[0].values[0], (x - x.mean(axis=0)) / (abs(x) + 1))


@pytest.mark.parametrize("width", [1, 7, 8, 9, 17])
def test_blocked_axis_statistics_keep_per_column_order(width):
    x = np.random.default_rng(width).normal(size=(123, width))
    mean = np.zeros(width)
    for row in x:
        mean += row
    mean /= len(x)
    var = np.zeros(width)
    for row in x:
        var += (row - mean) * (row - mean)
    var /= len(x)
    np.testing.assert_array_equal(op.mean_axis(x, 0), mean)
    np.testing.assert_array_equal(op.variance_axis(x, 0), var)
    x[-1, -1] = np.inf
    with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
        op.variance_axis(x, 0)


@pytest.mark.parametrize("indices", [[0, 1, 2], [2, 1, 0], [1, 2, 0]])
def test_unique_index_fast_path_keeps_bounds_and_duplicate_checks(indices):
    x = np.arange(9.0, dtype=np.float64).reshape(3, 3)
    order = np.array(indices, np.int64)
    np.testing.assert_array_equal(op.partition_mean(x, order, 3), x[order])
    order[1] = order[0]
    with pytest.raises(ValueError, match="DUPLICATE_INDEX"):
        op.partition_mean(x, order, 3)
    order[2] = 3
    with pytest.raises(ValueError, match="INDEX_OUT_OF_BOUNDS"):
        op.partition_mean(x, order, 3)


def test_integer_root_capacity_does_not_scale_with_input():
    x = np.arange(80000.0, dtype=np.float64).reshape(10000, 8)
    result = execute("sample_unique_indices(x,4,7)", {"x": x})
    assert result.outputs[0].values[0].dtype == np.int64
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] == 0


def test_same_graph_shape_changes_release_stale_arena_capacity():
    definitions = {"x": {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]}}
    graph = GraphCompiler(definitions).compile("mean_axis(x,0)/sqrt(variance_axis(x,0))")
    results = []
    with AdaptiveScheduler(cpu_budget=1) as engine:
        for shape in [(10000, 8), (80, 1000), (10000, 8), (3, 1)]:
            x = np.random.default_rng(sum(shape)).normal(size=shape)
            result = engine.execute(
                graph, {"x": x}, np.array([0], np.int64), np.array([0], np.int64)
            )
            expected = x.mean(axis=0) / x.std(axis=0)
            np.testing.assert_allclose(
                result.outputs[0].values[0], expected, rtol=1e-12, atol=1e-12
            )
            assert result.audit["native_chunks"][0]["numeric_arena_bytes"] <= shape[1] * 24
            results.append((result, expected.copy()))
    for result, expected in results:
        np.testing.assert_allclose(result.outputs[0].values[0], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("symbol", ["+", "-", "*"])
@pytest.mark.parametrize("layout", ["c", "reverse", "zero", "empty"])
def test_mixed_scalar_first_fusion_uses_array_backing(symbol, layout):
    x = np.arange(1.0, 36.0).reshape(5, 7)
    if layout == "reverse":
        x = x[::-1, ::-1]
    if layout == "zero":
        x = np.broadcast_to(x[:1], x.shape)
    if layout == "empty":
        x = x[:0]
    first = f"2{symbol}x"
    expression = f"({first})/broadcast_like(sum_axis(x,0)+1,x)"
    result = execute(expression, {"x": x})
    function = {"+": np.add, "-": np.subtract, "*": np.multiply}[symbol]
    expected = function(2.0, x) / (x.sum(axis=0) + 1)
    np.testing.assert_array_equal(result.outputs[0].values[0], expected)
    assert result.audit["native_chunks"][0]["fused_pointwise_calls"] > 0
