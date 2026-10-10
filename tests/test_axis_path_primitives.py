"""Ordered, seed-explicit axis paths and shared-sort quantile contracts.

These canonical mathematical primitives do not assert the platform's full
Monte Carlo service or its first-fault-prefix/compact-result protocol.
"""

import math

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler
from calmetrics_engine import operators as op

from .test_typed_results import config

PREFIX_NAMES = ("cumulative_product_axis", "cumulative_max_axis")
AXIS_CASES = [
    (shape, axis)
    for shape in ((5,), (3, 4), (2, 3, 4))
    for axis in range(len(shape))
]


def ieee_equal(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.dtype("float64")
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    non_nan = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[non_nan].view(np.uint64), expected[non_nan].view(np.uint64)
    )


def ordered_prefix(values, axis, initial, name):
    """Independent literal recurrence: no cumprod/maximum accumulation shortcut."""
    output = np.empty(values.shape, dtype=np.float64)
    source, target = np.moveaxis(values, axis, -1), np.moveaxis(output, axis, -1)
    with np.errstate(all="ignore"):
        for line in np.ndindex(source.shape[:-1]):
            state = np.float64(initial)
            for index in range(source.shape[-1]):
                current = source[line + (index,)]
                if name == "cumulative_product_axis":
                    state = np.float64(state * current)
                elif current > state:
                    state = current
                target[line + (index,)] = state
    return output


def ordered_quantiles(values, probabilities, axis):
    shape = list(values.shape)
    shape[axis] = len(probabilities)
    output = np.empty(shape, dtype=np.float64)
    source, target = np.moveaxis(values, axis, -1), np.moveaxis(output, axis, -1)
    with np.errstate(all="ignore"):
        for line in np.ndindex(source.shape[:-1]):
            ordered = sorted(source[line], key=lambda value: (bool(np.isnan(value)), float(value)))
            count = len(ordered)
            for q, probability in enumerate(probabilities):
                position = np.float64(count - 1) * probability
                lower, upper = math.floor(position), math.ceil(position)
                if count == 1 or lower == upper:
                    value = ordered[lower]
                else:
                    fraction = np.float64(position - lower)
                    low_product = np.float64(ordered[lower] * (1.0 - fraction))
                    high_product = np.float64(ordered[upper] * fraction)
                    value = np.float64(low_product + high_product)
                target[line + (q,)] = value
    return output


def layout_view(value, layout):
    if layout == "f":
        value = np.array(value, order="F")
    elif layout == "padded":
        owner = np.full(tuple(2 * size for size in value.shape), -991.0)
        slices = (slice(None, None, 2),) * value.ndim
        owner[slices] = value
        value = owner[slices]
    elif layout == "negative":
        reverse = (slice(None, None, -1),) * value.ndim
        value = value[reverse].copy()[reverse]
    value.flags.writeable = False
    return value


def declaration(rank, dtype="float64", semantic="dimensionless"):
    return dict(
        kind={1: "vector", 2: "matrix", 3: "tensor"}[rank],
        dtype=dtype,
        axes={1: ["asset"], 2: ["asset", "factor"], 3: ["scenario", "path", "factor"]}[rank],
        shape=[f"D{axis}" for axis in range(rank)],
        semantic_dimension=semantic,
    )


def q_declaration(dtype="float64", semantic="dimensionless"):
    return dict(kind="vector", dtype=dtype, axes=["component"], shape=["Q"],
                semantic_dimension=semantic)


def graph_for(name, rank, axis, order="C", scalar_parameter=False):
    types = {"values": declaration(rank)}
    if name == "quantiles_axis":
        types["q"] = q_declaration()
        expression = f"quantiles_axis(values,q,{axis})"
    else:
        if scalar_parameter:
            types["seed"] = dict(kind="scalar", semantic_dimension="dimensionless")
        expression = f"{name}(values,{axis},{'seed' if scalar_parameter else '1.25'})"
    return GraphCompiler(types).compile({"path": expression}, result_format="typed", output_order=order)


@pytest.mark.parametrize("name", PREFIX_NAMES)
@pytest.mark.parametrize("shape,axis", AXIS_CASES)
@pytest.mark.parametrize("layout", ["c", "f", "padded", "negative"])
def test_prefix_direct_all_axes_layouts_no_scratch_and_independent_results(name, shape, axis, layout):
    values = layout_view((np.arange(math.prod(shape)).reshape(shape) % 7 - 3.) / 4, layout)
    original = values.copy()
    expected = ordered_prefix(values, axis, 1.25, name)
    workspace = op.Workspace()
    requirements = op.get(name).requirements(values, axis, 1.25)
    assert requirements["shape"] == list(shape)
    assert requirements["scratch_doubles"] == requirements["scratch_indices"] == 0
    first, audit = op.call(name, values, axis, 1.25, workspace=workspace, audit=True)
    second = op.call(name, values, axis, 1.25, workspace=workspace, simd="scalar")
    ieee_equal(first, expected)
    ieee_equal(second, expected)
    ieee_equal(values, original)
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, values)
    assert audit["workspace_bytes"] == audit["algorithm_copy_bytes"] == audit["input_copy_bytes"] == 0
    assert workspace.capacity_bytes == 0


@pytest.mark.parametrize("name", PREFIX_NAMES)
def test_prefix_valid_zero_stride_lines(name):
    values = np.broadcast_to(np.array([[2., 3.], [4., 5.]])[:, None, :], (2, 3, 2))
    result = op.call(name, values, 1, 1.0)
    ieee_equal(result, ordered_prefix(values, 1, 1.0, name))
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize("seed", [0.0, -0.0, np.nan, np.inf, -np.inf])
def test_strict_max_keeps_seed_on_tie_current_nan_and_signed_zero(seed):
    values = np.array([0., -0., np.nan, 1., -1., np.inf, -np.inf])
    actual = op.cumulative_max_axis(values, 0, seed)
    ieee_equal(actual, ordered_prefix(values, 0, seed, "cumulative_max_axis"))
    if seed == 0.0 or np.isnan(seed):
        ieee_equal(actual[:3], np.full(3, seed))


@pytest.mark.parametrize("values,seed", [
    ([-2., .5, -0., np.inf], 3.),
    ([1e308, 2., 0., -1.], 1.),
    ([1e-308, 1e-20, -1., np.inf], 1.),
    ([np.nan, 1., np.inf], -0.),
])
def test_product_preserves_order_seed_overflow_underflow_and_ieee(values, seed):
    source = np.array(values, np.float64)
    ieee_equal(op.cumulative_product_axis(source, 0, seed),
               ordered_prefix(source, 0, seed, "cumulative_product_axis"))


@pytest.mark.parametrize("shape,axis", AXIS_CASES)
@pytest.mark.parametrize("layout", ["c", "f", "padded", "negative"])
def test_quantiles_sort_each_line_once_keep_q_order_and_bounded_scratch(shape, axis, layout):
    values = layout_view((np.arange(math.prod(shape)).reshape(shape) * 7 % 23 - 11.) / 8, layout)
    probabilities = layout_view(np.array([1., .25, 0., .5, .5]), "negative")
    originals = values.copy(), probabilities.copy()
    requirements = op.get("quantiles_axis").requirements(values, probabilities, axis)
    output_shape = list(shape)
    output_shape[axis] = len(probabilities)
    assert requirements["shape"] == output_shape
    assert requirements["scratch_doubles"] == shape[axis]
    assert requirements["scratch_indices"] == 0
    workspace = op.Workspace()
    first, audit = op.quantiles_axis(values, probabilities, axis, workspace=workspace, audit=True)
    second = op.quantiles_axis(values, probabilities, axis, workspace=workspace)
    expected = ordered_quantiles(values, probabilities, axis)
    ieee_equal(first, expected)
    ieee_equal(second, expected)
    ieee_equal(values, originals[0])
    ieee_equal(probabilities, originals[1])
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, values)
    assert not np.shares_memory(first, probabilities)
    assert workspace.capacity_bytes == shape[axis] * 8
    assert audit["workspace_bytes"] == shape[axis] * 8
    assert audit["algorithm_copy_bytes"] == values.nbytes
    assert audit["input_copy_bytes"] == 0


@pytest.mark.parametrize("values,q,expected", [
    ([np.inf, 0., np.inf], [0., .5, 1.], [0., np.inf, np.inf]),
    ([np.nan, -np.inf, np.inf], [0., .5, 1.], [-np.inf, np.inf, np.nan]),
    ([1., np.nan, 3.], [.25, .5, .75, 1.], [2., 3., np.nan, np.nan]),
    ([-np.inf, np.inf], [.25, .75], [np.nan, np.nan]),
    ([np.inf], [0., .125, .9, 1.], [np.inf, np.inf, np.inf, np.inf]),
    ([-0.], [0., .25, 1.], [-0., -0., -0.]),
])
def test_quantiles_integer_ranks_singletons_nan_last_and_no_nan_filter(values, q, expected):
    ieee_equal(op.quantiles_axis(np.array(values), np.array(q), 0), np.array(expected))


@pytest.mark.parametrize("name", PREFIX_NAMES)
@pytest.mark.parametrize("shape,axis", [((0,), 0), ((2, 0), 1), ((0, 3), 1), ((2, 0, 3), 0)])
def test_empty_prefix_preserves_shape_without_workspace(name, shape, axis):
    result, audit = op.call(name, np.empty(shape), axis, np.nan, audit=True)
    assert result.shape == shape
    assert result.size == 0
    assert audit["workspace_bytes"] == audit["algorithm_copy_bytes"] == 0


@pytest.mark.parametrize("shape,axis,q", [
    ((2, 0), 1, []), ((0, 3), 1, [.5]), ((2, 3), 1, []),
    ((0, 3, 2), 1, [.25, .75]), ((2, 0, 3), 2, [.5]),
])
def test_empty_quantile_output_requires_no_workspace(shape, axis, q):
    values, probabilities = np.empty(shape), np.array(q, np.float64)
    result, audit = op.quantiles_axis(values, probabilities, axis, audit=True)
    expected_shape = list(shape)
    expected_shape[axis] = len(q)
    assert result.shape == tuple(expected_shape)
    assert result.size == 0
    assert op.get("quantiles_axis").requirements(values, probabilities, axis)["scratch_doubles"] == 0
    assert audit["workspace_bytes"] == audit["algorithm_copy_bytes"] == 0


@pytest.mark.parametrize("shape,axis", [((0,), 0), ((2, 0), 1), ((2, 0, 3), 1)])
def test_empty_sample_axis_with_nonempty_output_is_insufficient(shape, axis):
    with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
        op.quantiles_axis(np.empty(shape), np.array([.5]), axis)


@pytest.mark.parametrize("probability", [np.nan, np.inf, -np.inf, -.01, 1.01])
@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
def test_invalid_probability_is_checked_even_when_outer_is_empty(probability, shape):
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        op.quantiles_axis(np.empty(shape), np.array([.5, probability]), 1)


@pytest.mark.parametrize("name", [*PREFIX_NAMES, "quantiles_axis"])
@pytest.mark.parametrize("axis", [-1., 2., .5, np.nan, np.inf])
def test_invalid_axes_are_not_silently_normalized(name, axis):
    args = (np.ones((2, 3)), np.array([.5]), axis) if name == "quantiles_axis" else (np.ones((2, 3)), axis, 1.)
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        op.call(name, *args)


@pytest.mark.parametrize("name", [*PREFIX_NAMES, "quantiles_axis"])
@pytest.mark.parametrize("shape", [(), (1, 1, 1, 1)])
def test_values_reject_scalar_and_rank_four(name, shape):
    args = (np.ones(shape), np.array([.5]), 0) if name == "quantiles_axis" else (np.ones(shape), 0, 1.)
    with pytest.raises(ValueError, match="RANK"):
        op.call(name, *args)


@pytest.mark.parametrize("name", [*PREFIX_NAMES, "quantiles_axis"])
def test_exact_float64_inputs_and_independent_seed_or_q_rank(name):
    args = (np.ones(2, np.float32), np.array([.5]), 0) if name == "quantiles_axis" else (np.ones(2, np.float32), 0, 1.)
    with pytest.raises(TypeError):
        op.call(name, *args)
    extra = np.ones((1, 2)) if name == "quantiles_axis" else np.ones(2)
    args = (np.ones(2), extra, 0) if name == "quantiles_axis" else (np.ones(2), 0, extra)
    with pytest.raises(ValueError, match="RANK"):
        op.call(name, *args)
    if name == "quantiles_axis":
        with pytest.raises((TypeError, ValueError)):
            op.call(name, np.ones(2), np.ones(2, np.int64), 0)
    else:
        # Exact small integer scalar configurations follow the common binding
        # contract; float64-only arrays do not forbid a lossless scalar seed.
        ieee_equal(op.call(name, np.ones(2), 0, np.int64(1)),
                   ordered_prefix(np.ones(2), 0, 1., name))


@pytest.mark.parametrize("name", [*PREFIX_NAMES, "quantiles_axis"])
def test_direct_out_c_only_exact_shape_and_no_alias(name):
    values, probabilities = np.ones((2, 3)), np.array([0., .5, 1.])
    args = (values, probabilities, 1) if name == "quantiles_axis" else (values, 1, 1.)
    with pytest.raises((ValueError, RuntimeError), match="ALIAS|overlap"):
        op.call(name, *args, out=values)
    with pytest.raises((ValueError, TypeError), match="contiguous|CONTIGUOUS"):
        op.call(name, *args, out=np.empty((2, 3), order="F"))
    storage = np.full(8, 993.)
    out = storage[1:-1].reshape(2, 3)
    assert op.call(name, *args, out=out) is out
    expected = ordered_quantiles(values, probabilities, 1) if name == "quantiles_axis" else ordered_prefix(values, 1, 1., name)
    ieee_equal(out, expected)
    assert storage[0] == storage[-1] == 993.


@pytest.mark.parametrize("name", [*PREFIX_NAMES, "quantiles_axis"])
@pytest.mark.parametrize("shape,axis", AXIS_CASES)
@pytest.mark.parametrize("mode,order", [("ordinary", "C"), ("ordinary", "F"), ("prepared", "F")])
def test_ast_typed_roots_all_ranks_axes_f_writer_and_retained_ownership(name, shape, axis, mode, order):
    values = layout_view((np.arange(math.prod(shape)).reshape(shape) % 13 - 6.) / 8, "f")
    inputs = {"values": values}
    probabilities = layout_view(np.array([1., .25, 0., .5]), "padded")
    if name == "quantiles_axis":
        inputs["q"] = probabilities
        expected = ordered_quantiles(values, probabilities, axis)
    else:
        expected = ordered_prefix(values, axis, 1.25, name)
    graph = graph_for(name, len(shape), axis, order)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if mode == "prepared":
            prepared = scheduler.prepare_execution(graph, inputs, bounds, bounds)
            call = prepared.run_snapshot
        else:
            def call():
                return scheduler.execute(graph, inputs, bounds, bounds)
        first, second = call(), call()
    actual, later = first.outputs[0].values[0], second.outputs[0].values[0]
    ieee_equal(actual, expected)
    ieee_equal(later, expected)
    assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
    assert first.outputs[0].order == (order if actual.ndim >= 2 else "C")
    axes = declaration(len(shape))["axes"].copy()
    if name == "quantiles_axis":
        axes[axis] = "component"
    assert first.outputs[0].axes == axes
    assert not actual.flags.writeable
    assert not np.shares_memory(actual, later)
    assert all(not np.shares_memory(actual, value) for value in inputs.values())
    np.testing.assert_array_equal(first.outputs[0].root_statuses, [0])
    assert first.audit["result_lifetime"] == "independent"
    assert first.audit["python_operator_calls"] == first.audit["request_time_compilation"] == 0


@pytest.mark.parametrize("name", [*PREFIX_NAMES, "quantiles_axis"])
def test_prepared_reads_mutable_values_q_and_seed_without_reusing_old_result(name):
    values = (np.arange(12.).reshape(3, 4) + 1.) / 16
    inputs = {"values": values}
    probabilities = np.array([.25, .75])
    parameters = np.array([1.25]) if name != "quantiles_axis" else None
    if name == "quantiles_axis":
        inputs["q"] = probabilities
    graph = graph_for(name, 2, 1, "F", scalar_parameter=name != "quantiles_axis")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, bounds, bounds, parameters=parameters)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        values[:] += .25
        probabilities[:] = [.5, 1.]
        if parameters is not None:
            parameters[0] = -1.
        later = prepared.run_snapshot()
        expected = ordered_quantiles(values, probabilities, 1) if name == "quantiles_axis" else ordered_prefix(values, 1, -1., name)
        ieee_equal(later.outputs[0].values[0], expected)
        ieee_equal(first.outputs[0].values[0], saved)
        assert not np.shares_memory(first.outputs[0].values[0], later.outputs[0].values[0])
        values.shape = (2, 6)
        with pytest.raises(ValueError, match="PREPARED_INPUT_CHANGED"):
            prepared.run_snapshot()


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_multiroot_native_transport_keeps_complete_two_generation_ownership(lane, order):
    values = layout_view((np.arange(24.).reshape(2, 3, 4) - 9.) / 16, "negative")
    probabilities = layout_view(np.array([.25, 1., 0.]), "negative")
    inputs = {"values": values, "q": probabilities}
    roots = {"product": "cumulative_product_axis(values,1,1.25)",
             "maximum": "cumulative_max_axis(values,1,1.25)",
             "quantiles": "quantiles_axis(values,q,1)"}
    graph = GraphCompiler({"values": declaration(3), "q": q_declaration()}).compile(
        roots, result_format="typed", output_order=order)
    bounds = np.zeros(2, np.int64)
    expected = [ordered_prefix(values, 1, 1.25, name) for name in PREFIX_NAMES]
    expected.append(ordered_quantiles(values, probabilities, 1))
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(graph, inputs, bounds, bounds)
        second = scheduler.execute(graph, inputs, bounds, bounds)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    assert first.plan.thread_count <= 2
    assert first.plan.process_count * first.plan.threads_per_process <= 2
    for output, later, target in zip(first.outputs, second.outputs, expected, strict=True):
        np.testing.assert_array_equal(output.root_statuses, np.zeros(2, np.int16))
        for value, new in zip(output.values, later.values, strict=True):
            ieee_equal(value, target)
            assert not value.flags.writeable
            assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
            assert all(not np.shares_memory(value, item) for item in inputs.values())
            assert all(not np.shares_memory(value, item) for item in later.values)
            assert not np.shares_memory(value, new)
        assert not np.shares_memory(output.values[0], output.values[1])
    assert first.audit["result_lifetime"] == "independent"
    for chunk in first.audit["native_chunks"]:
        assert chunk["operator_workspace_capacity_bytes"] <= first.plan.estimated_worker_scratch_bytes


@pytest.mark.parametrize("shape,axis,q", [((2, 0), 1, []), ((0, 3), 1, [.5]), ((2, 3), 1, [])])
def test_typed_empty_quantiles_have_zero_sort_scratch(shape, axis, q):
    graph = graph_for("quantiles_axis", 2, axis, "F")
    inputs = {"values": np.empty(shape), "q": np.array(q, np.float64)}
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, bounds, bounds)
    target_shape = list(shape)
    target_shape[axis] = len(q)
    assert result.outputs[0].values[0].shape == tuple(target_shape)
    assert result.outputs[0].values[0].size == 0
    for chunk in result.audit["native_chunks"]:
        assert chunk["operator_workspace_capacity_bytes"] == 0


def test_quantile_exact_total_memory_budget_boundary_and_line_workspace_bound():
    graph = graph_for("quantiles_axis", 2, 1, "F")
    inputs = {"values": np.arange(16 * 32.).reshape(16, 32), "q": np.array([.25, .5, .75])}
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, bounds, bounds)
        assert plan.estimated_output_bytes == 16 * 3 * 8
        assert plan.estimated_worker_scratch_bytes >= 32 * 8
        with pytest.raises(MemoryError):
            scheduler.execute(graph, inputs, bounds, bounds,
                              memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        result = scheduler.execute(graph, inputs, bounds, bounds,
                                   memory_budget_bytes=plan.estimated_total_memory_bytes)
    ieee_equal(result.outputs[0].values[0], ordered_quantiles(inputs["values"], inputs["q"], 1))
    assert all(chunk["operator_workspace_capacity_bytes"] == 32 * 8
               for chunk in result.audit["native_chunks"])


@pytest.mark.parametrize("name", [*PREFIX_NAMES, "quantiles_axis"])
def test_typed_compile_requires_literal_axis_and_float64_math(name):
    types = {"values": declaration(2), "axis": "scalar", "q": q_declaration()}
    expression = "quantiles_axis(values,q,axis)" if name == "quantiles_axis" else f"{name}(values,axis,1)"
    with pytest.raises(GraphCompileError, match="CONSTANT_LAYOUT"):
        GraphCompiler(types).compile(expression)
    types["values"] = declaration(2, dtype="int64")
    expression = "quantiles_axis(values,q,0)" if name == "quantiles_axis" else f"{name}(values,0,1)"
    with pytest.raises(GraphCompileError, match="TYPE"):
        GraphCompiler(types).compile(expression)


@pytest.mark.parametrize("policy", ["raise", "isolate"])
@pytest.mark.parametrize("multihop", [False, True])
def test_failed_values_invalid_probability_preserves_upstream_numeric_error(policy, multihop):
    failed = "stack(block_apply(mean(x),1/z),block_apply(mean(x),1/z),0)"
    values = f"cumulative_product_axis({failed},0,1)" if multihop else failed
    types = {"x": "series", "z": "scalar", "q": q_declaration()}
    roots = {"failed": f"quantiles_axis({values},q,0)", "healthy": "mean(x)"}
    graph = GraphCompiler(types).compile(roots, result_format="typed", error_policy=policy)
    inputs = {"x": np.arange(4.), "q": np.array([np.nan, 1.1])}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        starts, ends = np.array([0], np.int64), np.array([4], np.int64)
        if policy == "raise":
            with pytest.raises((ValueError, RuntimeError), match="DIVIDE_BY_ZERO"):
                scheduler.execute(graph, inputs, starts, ends, parameters={"z": 0.})
        else:
            result = scheduler.execute(graph, inputs, starts, ends, parameters={"z": 0.})
            np.testing.assert_array_equal(result.named_outputs["failed"].root_statuses, [2])
            np.testing.assert_array_equal(result.named_outputs["healthy"].root_statuses, [0])
            assert result.named_outputs["healthy"].values[0] == 1.5


@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_failed_values_do_not_hide_independent_q_shape_contract(policy):
    failed = "stack(block_apply(mean(x),1/z),block_apply(mean(x),1/z),0)"
    graph = GraphCompiler({"x": "series", "z": "scalar", "q": q_declaration()}).compile(
        f"quantiles_axis({failed},q,0)", result_format="typed", error_policy=policy)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises((ValueError, RuntimeError), match="RANK|rank|shape|dimensions"):
            scheduler.execute(graph, {"x": np.arange(4.), "q": np.ones((1, 2))},
                              np.array([0], np.int64), np.array([4], np.int64), parameters={"z": 0.})
