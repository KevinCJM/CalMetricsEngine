"""Short computed-broadcast snapshots must outlive reused logical arena slots."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_typed_results import config


def schema(time=False):
    return dict(kind="matrix", axes=["time" if time else "asset", "factor"], shape=["N", "D"])


def execute(
    expressions, x, *, order="C", lane="single", isolate=False, time=False, starts=None, ends=None
):
    graph = GraphCompiler({"x": schema(time)}).compile(
        expressions,
        result_format="typed",
        error_policy="isolate" if isolate else "raise",
        output_order=order,
    )
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as engine:
        return engine.execute(
            graph,
            {"x": x},
            bounds if starts is None else np.array(starts, np.int64),
            bounds if ends is None else np.array(ends, np.int64),
        )


@pytest.mark.parametrize("first", ["+", "-", "*"])
@pytest.mark.parametrize("second", ["+", "-", "*", "/"])
@pytest.mark.parametrize("source_left", [False, True])
@pytest.mark.parametrize("pair_left", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "reverse"])
def test_cached_arithmetic_preserves_order_and_reused_backing(
    first, second, source_left, pair_left, layout
):
    x = np.random.default_rng(1548).normal(size=(37, 17)) + 0.25
    if layout == "F":
        x = np.asfortranarray(x)
    if layout == "reverse":
        x = x[::-1, ::-1]
    x.flags.writeable = False
    mean = "broadcast_like(mean_axis(x,0),x)"
    other = "broadcast_like(sqrt(variance_axis(x,0,1))+1,x)"
    middle = f"x{first}{mean}" if source_left else f"{mean}{first}x"
    expression = f"({middle}){second}{other}" if pair_left else f"{other}{second}({middle})"
    funcs = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}
    center = op.mean_axis(x, 0)
    scale = np.sqrt(op.variance_axis(x, 0, 1)) + 1
    value = funcs[first](x, center) if source_left else funcs[first](center, x)
    expected = funcs[second](value, scale) if pair_left else funcs[second](scale, value)
    result = execute(expression, x, order="F" if layout == "F" else "C")
    for actual in result.outputs[0].values:
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable and not np.shares_memory(actual, x)
    assert not np.shares_memory(*result.outputs[0].values)
    chunk = result.audit["native_chunks"][0]
    assert chunk["fused_pointwise_calls"] > 0
    assert chunk["numeric_arena_bytes"] <= x.shape[1] * 32
    assert chunk["algorithm_copy_bytes"] >= x.shape[1] * 8


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_saved_vector_transport_has_independent_results(lane, order):
    x = np.random.default_rng(831).normal(size=(128, 32))
    x.flags.writeable = False
    expr = "(x-broadcast_like(mean_axis(x,0),x))/broadcast_like(sqrt(variance_axis(x,0,1)),x)"
    result = execute(expr, x, order=order, lane=lane)
    expected = (x - x.mean(0)) / x.std(0, ddof=1)
    for actual in result.outputs[0].values:
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        assert not actual.flags.writeable and not np.shares_memory(actual, x)
        assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous


@pytest.mark.parametrize("shape", [(0, 5), (1, 5), (2, 5), (5, 0), (0, 0), (7, 1)])
@pytest.mark.parametrize("isolate", [False, True])
def test_empty_short_and_isolation_keep_canonical_paths(shape, isolate):
    x = np.arange(np.prod(shape), dtype=float).reshape(shape)
    expr = "(x-broadcast_like(sum_axis(x,0),x))/broadcast_like(sum_axis(x*x,0)+1,x)"
    result = execute(expr, x, isolate=isolate)
    expected = (x - x.sum(0)) / (np.sum(x * x, 0) + 1)
    for actual in result.outputs[0].values:
        np.testing.assert_array_equal(actual, expected)


def test_empty_time_interval_does_not_read_a_nonexistent_broadcast_row():
    x = np.arange(32.0, dtype=float).reshape(8, 4)
    expr = "(x-broadcast_like(sum_axis(x,0),x))/broadcast_like(sum_axis(x*x,0)+1,x)"
    result = execute(expr, x, time=True, starts=[0, 0, 7], ends=[0, 8, 8])
    for actual, left, right in zip(result.outputs[0].values, [0, 0, 7], [0, 8, 8], strict=True):
        selected = x[left:right]
        np.testing.assert_array_equal(
            actual, (selected - selected.sum(0)) / (np.sum(selected * selected, 0) + 1)
        )


@pytest.mark.parametrize("shared", [False, True])
def test_live_first_root_is_never_replaced_by_a_short_cache(shared):
    x = np.arange(48.0, dtype=float).reshape(12, 4)
    first = "x-broadcast_like(mean_axis(x,0),x)"
    tail = f"({first})/broadcast_like(sum_axis(x*x,0)+1,x)"
    expressions = [tail, first] if shared else [tail, f"sum_axis({first},0)"]
    result = execute(expressions, x)
    expected_first = x - x.mean(0)
    np.testing.assert_array_equal(
        result.outputs[0].values[0], expected_first / (np.sum(x * x, 0) + 1)
    )
    np.testing.assert_array_equal(
        result.outputs[1].values[0], expected_first if shared else expected_first.sum(0)
    )
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] >= x.nbytes


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("isolate", [False, True])
def test_variance_failure_remains_at_its_logical_node(bad, isolate):
    x = np.arange(64.0, dtype=float).reshape(16, 4)
    x[3, 2] = bad
    expr = "(x-broadcast_like(mean_axis(x,0),x))/broadcast_like(sqrt(variance_axis(x,0,1)),x)"
    if isolate:
        result = execute([expr, "sum_axis(slice_axis(x,1,0,1,1),0)"], x, isolate=True)
        assert result.outputs[0].root_statuses[0] != 0
        assert result.outputs[1].root_statuses[0] == 0
    else:
        with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
            execute(expr, x)


@pytest.mark.parametrize("order", ["C", "F"])
def test_zero_denominator_still_throws_and_independent_isolated_root_returns(order):
    x = np.ones((64, 17))
    expr = "(x-broadcast_like(mean_axis(x,0),x))/broadcast_like(sqrt(variance_axis(x,0,1)),x)"
    with pytest.raises((ValueError, RuntimeError), match="DIVIDE_BY_ZERO"):
        execute(expr, x, order=order)
    result = execute([expr, "sum_axis(x,0)"], x, order=order, isolate=True)
    assert result.outputs[0].root_statuses[0] != 0
    np.testing.assert_array_equal(result.outputs[1].values[0], x.sum(0))


def test_same_graph_input_and_shape_changes_do_not_reuse_numeric_snapshots():
    graph = GraphCompiler({"x": schema()}).compile(
        "(x-broadcast_like(mean_axis(x,0),x))/broadcast_like(sqrt(variance_axis(x,0,1))+1,x)",
        result_format="typed",
    )
    retained = []
    with AdaptiveScheduler(cpu_budget=1) as engine:
        for shape in [(1000, 17), (37, 129), (1000, 17), (3, 1)]:
            x = np.random.default_rng(sum(shape) + len(retained)).normal(size=shape)
            x.flags.writeable = False
            result = engine.execute(graph, {"x": x}, np.zeros(1, np.int64), np.zeros(1, np.int64))
            actual = result.outputs[0].values[0]
            expected = (x - x.mean(0)) / (x.std(0, ddof=1) + 1)
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
            assert result.audit["native_chunks"][0]["numeric_arena_bytes"] <= shape[1] * 32
            retained.append((actual, expected.copy()))
    for actual, expected in retained:
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_unknown_cached_geometry_does_not_shadow_a_proven_other_pair():
    x = np.arange(256 * 32.0, dtype=float).reshape(256, 32)
    expr = "(x+broadcast_like(mean_axis(slice_axis(x,1,0,1,1),0),x))+(x*x)"
    result = execute(expr, x)
    np.testing.assert_array_equal(result.outputs[0].values[0], (x + x[:, :1].mean(0)) + (x * x))
    chunk = result.audit["native_chunks"][0]
    assert chunk["fused_pointwise_calls"] > 0
    assert chunk["numeric_arena_bytes"] <= x.nbytes + 32


@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_snapshots_save_current_vectors_and_keep_old_results(order):
    x = np.array(np.random.default_rng(490).normal(size=(64, 17)), order=order)
    expression = (
        "(x-broadcast_like(mean_axis(x,0),x))/broadcast_like(sqrt(variance_axis(x,0,1))+1,x)"
    )
    graph = GraphCompiler({"x": schema()}).compile(
        expression, result_format="typed", output_order=order
    )
    with AdaptiveScheduler(cpu_budget=1) as engine:
        prepared = engine.prepare_execution(
            graph, {"x": x}, np.zeros(1, np.int64), np.zeros(1, np.int64)
        )
        first = prepared.run_snapshot().outputs[0].values[0]
        saved = first.copy()
        x *= np.linspace(1, 3, x.shape[1])
        x += np.arange(len(x))[:, None] * 0.1
        second = prepared.run_snapshot().outputs[0].values[0]
        np.testing.assert_array_equal(first, saved)
        np.testing.assert_allclose(
            second, (x - x.mean(0)) / (x.std(0, ddof=1) + 1), rtol=1e-13, atol=1e-13
        )
        assert not np.shares_memory(first, second)
