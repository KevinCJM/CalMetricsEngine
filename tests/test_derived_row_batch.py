"""Derived independent rows retain canonical math, failures and owned results."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig

from .test_array_row_parallel import config, data
from .test_row_batch_execution import NEGATIVE_COUNT, ROOTS, TAIL_COUNT, TYPES, Y, arena

EXPONENTIAL = "-log(maximum(w,1e-15))"
DENOMINATOR = f"transpose(broadcast_like(sum_axis({EXPONENTIAL},1),transpose({EXPONENTIAL})))"
NORMALIZED = f"divide({EXPONENTIAL},{DENOMINATOR})"
TRAJECTORY = f"matmul({NORMALIZED},transpose(r))"


def compile_roots(prefix=TRAJECTORY, expose=False, types=TYPES, order="C"):
    roots = [x.replace(Y, prefix) for x in ROOTS + [TAIL_COUNT, NEGATIVE_COUNT]]
    if expose:
        roots.append(prefix)
    return GraphCompiler(types).compile(roots, result_format="typed", output_order=order)


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("n", [127, 128, 257, 517])
@pytest.mark.parametrize("k", [0, 1, 7])
@pytest.mark.parametrize("cpu", [1, 4])
def test_transformed_normalized_rows_math_budget_and_ownership(layout, n, k, cpu):
    inputs = data(n, 129, k, layout)
    original = {name: value.copy() for name, value in inputs.items()}
    starts = np.array([0], np.int64)
    ends = np.array([129], np.int64)
    with AdaptiveScheduler(cpu_budget=cpu, config=config()) as scheduler:
        first = scheduler.execute(
            compile_roots(order=layout if layout in ["C", "F"] else "C"), inputs, starts, ends
        )
        saved = [out.values[0].copy() for out in first.outputs]
        control = scheduler.execute(compile_roots(expose=True), inputs, starts, ends)
        second = scheduler.execute(compile_roots(), inputs, starts, ends)
    for index, out in enumerate(first.outputs):
        actual = out.values[0]
        np.testing.assert_array_equal(
            actual.view(np.uint64), control.outputs[index].values[0].view(np.uint64)
        )
        np.testing.assert_array_equal(actual.view(np.uint64), saved[index].view(np.uint64))
        assert actual.shape == (n,) and not actual.flags.writeable
        assert out.root_statuses[0] == 0
        assert not np.shares_memory(actual, second.outputs[index].values[0])
        assert all(not np.shares_memory(actual, value) for value in inputs.values())
    for name, value in inputs.items():
        np.testing.assert_array_equal(value.view(np.uint64), original[name].view(np.uint64))
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in first.audit["native_chunks"]
    )
    assert first.plan.estimated_worker_scratch_bytes >= observed
    if n >= 128:
        assert arena(first) < arena(control) / 2
    if cpu == 4 and n >= 256:
        assert first.plan.parallel_dimension == "array_row"
        assert first.plan.thread_count == min(4, n // 128)
    assert sum(c["root_copy_bytes"] for c in first.audit["native_chunks"]) == 0


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_multiple_intervals_transports_and_derived_prefix(layout, lane):
    inputs = data(257, 257, 7, layout)
    starts = np.array([0, 11], np.int64)
    ends = np.array([257, 257], np.int64)
    cfg = PlannerConfig(
        thread_work_units=1e100 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )
    with AdaptiveScheduler(cpu_budget=2, config=cfg) as scheduler:
        result = scheduler.execute(compile_roots(), inputs, starts, ends)
        control = scheduler.execute(compile_roots(expose=True), inputs, starts, ends)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    for a, b in zip(result.outputs, control.outputs[:5], strict=True):
        for row in range(2):
            np.testing.assert_array_equal(
                a.values[row].view(np.uint64), b.values[row].view(np.uint64)
            )
            assert not a.values[row].flags.writeable


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("cpu", [1, 4])
def test_prefix_failure_precedes_later_node_in_an_earlier_batch(layout, cpu):
    inputs = data(517, 129, 1, layout)
    weights = np.full((517, 1), 2.0)
    weights[10] = 1.0
    weights[-5] = -1.0
    if layout == "negative":
        weights = weights[::-1]
    weights.flags.writeable = False
    returns = np.ones((129, 1))
    returns.flags.writeable = False
    inputs.update(w=weights, r=returns)
    y = "matmul(log(w),transpose(r))"
    roots = [f"sum_axis(divide(1,{y}),1)"]
    compiler = GraphCompiler(TYPES)
    starts = np.array([0], np.int64)
    ends = np.array([129], np.int64)
    with AdaptiveScheduler(cpu_budget=cpu, config=config()) as scheduler:
        messages = []
        for expressions in [roots, roots + [y]]:
            with pytest.raises(ValueError) as error:
                scheduler.execute(
                    compiler.compile(expressions, result_format="typed"), inputs, starts, ends
                )
            messages.append(str(error.value))
        assert messages[0] == messages[1] and "DOMAIN_ERROR" in messages[0]
        healthy = scheduler.execute(compile_roots(), data(517, 129, 1), starts, ends)
        assert all(np.isfinite(out.values[0]).all() for out in healthy.outputs)


@pytest.mark.parametrize("prefix", ["cross_row", "second_input", "global_scalar", "matrix_root"])
@pytest.mark.parametrize("cpu", [1, 4])
def test_unproven_dependencies_remain_on_full_canonical_path(prefix, cpu):
    inputs = data(257, 129, 7)
    types = dict(TYPES)
    y = TRAJECTORY
    if prefix == "cross_row":
        left = "divide(w,broadcast_like(sum_axis(w,0),w))"
        y = f"matmul({left},transpose(r))"
    elif prefix == "second_input":
        types["v"] = TYPES["w"]
        inputs["v"] = inputs["w"].copy()
        inputs["v"].flags.writeable = False
        y = "matmul(add(w,v),transpose(r))"
    elif prefix == "global_scalar":
        y = "matmul(divide(w,sum(w)),transpose(r))"
    expressions = [x.replace(Y, y) for x in ROOTS + [TAIL_COUNT, NEGATIVE_COUNT]]
    if prefix == "matrix_root":
        expressions.append(NORMALIZED)
    with AdaptiveScheduler(cpu_budget=cpu, config=config()) as scheduler:
        compiler = GraphCompiler(types)
        result = scheduler.execute(
            compiler.compile(expressions, result_format="typed"),
            inputs,
            np.array([0], np.int64),
            np.array([129], np.int64),
        )
        control = scheduler.execute(
            compiler.compile(expressions + [y], result_format="typed"),
            inputs,
            np.array([0], np.int64),
            np.array([129], np.int64),
        )
    if prefix != "matrix_root":
        assert result.plan.parallel_dimension != "array_row"
    elif cpu == 4:
        assert result.plan.parallel_dimension == "array_row"
    for a, b in zip(result.outputs, control.outputs[: len(expressions)], strict=True):
        np.testing.assert_array_equal(a.values[0].view(np.uint64), b.values[0].view(np.uint64))


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
def test_prepared_content_changes_preserve_prior_snapshot(layout):
    inputs = data(517, 129, 7, layout)
    # Keep addresses/strides stable while changing the live numerical contents.
    inputs["w"].flags.writeable = True
    starts = np.array([0], np.int64)
    ends = np.array([129], np.int64)
    g = compile_roots()
    with AdaptiveScheduler(cpu_budget=4, config=config()) as scheduler:
        prepared = scheduler.prepare_execution(g, inputs, starts, ends)
        first = prepared.run_snapshot()
        saved = [x.values[0].copy() for x in first.outputs]
        inputs["w"] *= 0.5
        current = prepared.run_snapshot()
        ordinary = scheduler.execute(g, inputs, starts, ends)
    for index, out in enumerate(first.outputs):
        np.testing.assert_array_equal(out.values[0], saved[index])
        np.testing.assert_array_equal(
            current.outputs[index].values[0], ordinary.outputs[index].values[0]
        )
        assert not np.shares_memory(out.values[0], current.outputs[index].values[0])
