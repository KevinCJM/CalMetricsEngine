"""Independent rows retain the complete canonical graph and owned outputs."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig

from .test_row_batch_execution import TYPES, Y


def data(n, t, k, layout="C"):
    rng = np.random.default_rng(20120)
    w = rng.uniform(0.05, 1.0, size=(n, k))
    r = rng.normal(scale=0.01, size=(t, k))
    if layout == "F":
        w = np.array(w, order="F")
        r = np.array(r, order="F")
    elif layout == "negative":
        w = w[::-1, ::-1]
        r = r[::-1, ::-1]
    elif layout == "padded":
        a = np.full((n, k * 2), np.nan)
        a[:, ::2] = w
        w = a[:, ::2]
        a = np.full((t, k * 2), np.nan)
        a[:, ::2] = r
        r = a[:, ::2]
    result = dict(w=w, r=r, p=np.array([0.05]))
    for x in result.values():
        x.flags.writeable = False
    return result


def config():
    return PlannerConfig(thread_work_units=1.0)


def roots():
    # Keep shared matrix, ordered sum, product, max and nonlinear consumers.
    return [
        f"sum_axis({Y},1)",
        f"sum_axis(multiply({Y},{Y}),1)",
        f"max_axis(cumulative_product_axis(add({Y},1),1,1),1)",
    ]


@pytest.mark.parametrize("n", [128, 257, 517, 4097])
@pytest.mark.parametrize("t,k", [(0, 7), (17, 0), (129, 7), (257, 7)])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
def test_row_ranges_keep_values_metadata_budget_and_retained_outputs(n, t, k, layout):
    inputs = data(n, t, k, layout)
    original = {name: x.copy() for name, x in inputs.items()}
    expressions = roots() if t else roots()[:2]
    graph = GraphCompiler(TYPES).compile(expressions, result_format="typed", output_order="F")
    starts = np.array([0], np.int64)
    ends = np.array([t], np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config()) as single:
        expected = single.execute(graph, inputs, starts, ends)
    with AdaptiveScheduler(cpu_budget=4, config=config()) as parallel:
        first = parallel.execute(graph, inputs, starts, ends)
        saved = [out.values[0].copy() for out in first.outputs]
        second = parallel.execute(graph, inputs, starts, ends)
    workers = min(4, n // 128)
    assert first.audit["cpu_tokens"] <= 4
    if workers >= 2:
        assert first.plan.parallel_dimension == "array_row"
        assert first.plan.thread_count == workers
        chunks = first.plan.metadata()["chunks"]
        assert chunks[0][0] == 0 and chunks[-1][1] == n
        assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:], strict=False))
    for index, out in enumerate(first.outputs):
        actual = out.values[0]
        assert actual.shape == (n,) and not actual.flags.writeable
        np.testing.assert_array_equal(
            actual.view(np.uint64), expected.outputs[index].values[0].view(np.uint64)
        )
        np.testing.assert_array_equal(actual.view(np.uint64), saved[index].view(np.uint64))
        assert not np.shares_memory(actual, second.outputs[index].values[0])
        assert all(not np.shares_memory(actual, x) for x in inputs.values())
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in first.audit["native_chunks"]
    )
    assert first.plan.estimated_worker_scratch_bytes >= observed
    assert sum(c["root_copy_bytes"] for c in first.audit["native_chunks"]) == 0
    for name, x in inputs.items():
        np.testing.assert_array_equal(x.view(np.uint64), original[name].view(np.uint64))


def test_memory_budget_can_recover_the_original_single_thread_admission():
    inputs = data(4097, 257, 7)
    graph = GraphCompiler(TYPES).compile(roots(), result_format="typed")
    starts = np.array([0], np.int64)
    ends = np.array([257], np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config()) as single:
        budget = single.plan(graph, inputs, starts, ends).estimated_total_memory_bytes
        expected = single.execute(graph, inputs, starts, ends)
    with AdaptiveScheduler(cpu_budget=4, config=config()) as parallel:
        result = parallel.execute(graph, inputs, starts, ends, memory_budget_bytes=budget)
    assert result.plan.lane == "single" and result.audit["cpu_tokens"] == 1
    assert result.plan.estimated_total_memory_bytes <= budget
    for a, b in zip(result.outputs, expected.outputs, strict=False):
        np.testing.assert_array_equal(a.values[0], b.values[0])


@pytest.mark.parametrize("reverse", [False, True])
def test_later_row_earlier_node_failure_keeps_serial_error_priority_and_cpu_release(reverse):
    inputs = data(517, 129, 1)
    w = np.full((517, 1), 2.0)
    w[10] = 1.0
    w[-5] = -1.0
    if reverse:
        w = w[::-1]
    w.flags.writeable = False
    r = np.ones((129, 1))
    r.flags.writeable = False
    inputs.update(w=w, r=r)
    expressions = [f"sum_axis(log({Y}),1)", f"sum_axis(divide({Y},subtract({Y},1)),1)"]
    graph = GraphCompiler(TYPES).compile(expressions, result_format="typed")
    starts = np.array([0], np.int64)
    ends = np.array([129], np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config()) as single:
        with pytest.raises(ValueError) as reference:
            single.execute(graph, inputs, starts, ends)
    with AdaptiveScheduler(cpu_budget=4, config=config()) as parallel:
        with pytest.raises(ValueError) as actual:
            parallel.execute(graph, inputs, starts, ends)
        assert str(actual.value) == str(reference.value)
        healthy = data(517, 129, 1)
        result = parallel.execute(
            GraphCompiler(TYPES).compile(roots(), result_format="typed"), healthy, starts, ends
        )
        assert all(np.isfinite(out.values[0]).all() for out in result.outputs)
        assert result.audit["cpu_tokens"] <= 4


def test_multiple_intervals_and_public_matrix_do_not_use_row_ranges():
    inputs = data(517, 257, 7)
    starts = np.array([0, 0], np.int64)
    ends = np.array([257, 257], np.int64)
    compiler = GraphCompiler(TYPES)
    with AdaptiveScheduler(cpu_budget=4, config=config()) as scheduler:
        multiple = scheduler.execute(
            compiler.compile(roots(), result_format="typed"), inputs, starts, ends
        )
        assert multiple.plan.parallel_dimension != "array_row"
        for out in multiple.outputs:
            np.testing.assert_array_equal(out.values[0], out.values[1])
        exposed = scheduler.execute(
            compiler.compile(roots() + [Y], result_format="typed"), inputs, starts[:1], ends[:1]
        )
        assert exposed.plan.parallel_dimension != "array_row"
        assert exposed.outputs[-1].values[0].shape == (517, 257)


def test_timeout_drains_workers_before_next_independent_request():
    inputs = data(65536, 2520, 7)
    compiler = GraphCompiler(TYPES)
    graph = compiler.compile(roots(), result_format="typed")
    starts = np.array([0], np.int64)
    ends = np.array([2520], np.int64)
    with AdaptiveScheduler(cpu_budget=4, config=config()) as scheduler:
        with pytest.raises(TimeoutError):
            scheduler.execute(graph, inputs, starts, ends, timeout=0.001)
        healthy = data(517, 129, 7)
        result = scheduler.execute(graph, healthy, starts, np.array([129], np.int64))
        assert all(np.isfinite(out.values[0]).all() for out in result.outputs)
        assert result.audit["cpu_tokens"] <= 4


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
def test_current_parameters_inputs_and_prepared_single_snapshot_remain_independent(layout):
    inputs = data(517, 129, 7, layout)
    inputs["w"].flags.writeable = True
    types = dict(TYPES, shift="scalar")
    expression = f"sum_axis(add({Y},shift),1)"
    graph = GraphCompiler(types).compile(expression, result_format="typed")
    starts = np.array([0], np.int64)
    ends = np.array([129], np.int64)
    parameters = np.array([0.001])
    with AdaptiveScheduler(cpu_budget=1, config=config()) as single:
        prepared = single.prepare_execution(graph, inputs, starts, ends, parameters=parameters)
        snapshot = prepared.run_snapshot()
        stable = snapshot.outputs[0].values[0].copy()
        inputs["w"] += 0.01
        parameters[0] = 0.003
        expected = prepared.run_snapshot()
    with AdaptiveScheduler(cpu_budget=4, config=config()) as scheduler:
        compatible = scheduler.prepare_execution(graph, inputs, starts, ends, parameters=parameters)
        prepared_result = compatible.run_snapshot()
        assert prepared_result.plan.lane == "single"
        np.testing.assert_array_equal(
            prepared_result.outputs[0].values[0], expected.outputs[0].values[0]
        )
        first = scheduler.execute(graph, inputs, starts, ends, parameters=parameters)
        saved = first.outputs[0].values[0].copy()
        parameters[0] = 0.005
        second = scheduler.execute(graph, inputs, starts, ends, parameters=parameters)
    assert first.plan.parallel_dimension == second.plan.parallel_dimension == "array_row"
    np.testing.assert_array_equal(snapshot.outputs[0].values[0], stable)
    np.testing.assert_array_equal(first.outputs[0].values[0], expected.outputs[0].values[0])
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
    np.testing.assert_allclose(
        second.outputs[0].values[0] - saved, np.full(517, 129 * 0.002), atol=4e-5
    )
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("cpu", [1, 4])
def test_empty_contracting_axis_does_not_offset_extreme_signed_strides(sign, cpu):
    # NumPy and the existing boundary admit any stride for an empty view.
    # It owns no elements; even forming a shifted pointer is unnecessary.
    backing = np.zeros(1)
    step = sign * (np.iinfo(np.intp).max // 8 * 8)
    weights = np.ndarray((517, 0), dtype=np.float64, buffer=backing, strides=(step, 8))
    weights.flags.writeable = False
    inputs = dict(w=weights, r=np.empty((129, 0)), p=np.array([0.05]))
    graph = GraphCompiler(TYPES).compile(roots(), result_format="typed")
    with AdaptiveScheduler(cpu_budget=cpu, config=config()) as scheduler:
        result = scheduler.execute(
            graph, inputs, np.array([0], np.int64), np.array([129], np.int64)
        )
    for out, expected in zip(result.outputs, [0.0, 0.0, 1.0], strict=True):
        np.testing.assert_array_equal(out.values[0], np.full(517, expected))
    if cpu == 4:
        assert result.plan.parallel_dimension == "array_row"
    assert backing[0] == 0
