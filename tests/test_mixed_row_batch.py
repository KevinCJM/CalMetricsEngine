"""Complete mixed outputs retain global geometry, owned storage and failures."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig

from .test_array_row_parallel import config, data
from .test_derived_row_batch import NORMALIZED, TRAJECTORY
from .test_row_batch_execution import NEGATIVE_COUNT, ROOTS, TAIL_COUNT, TYPES, Y, arena

MIXED_TYPES = dict(TYPES, q=dict(kind="vector", axes=["component"], shape=["3"]))


def roots():
    # Different matrix widths surround all three independent risk vectors.
    return (
        [NORMALIZED]
        + [x.replace(Y, TRAJECTORY) for x in ROOTS]
        + [f"quantiles_axis({TRAJECTORY},q,1)"]
        + [x.replace(Y, TRAJECTORY) for x in [TAIL_COUNT, NEGATIVE_COUNT]]
    )


def graph(order="C", expose=False):
    return GraphCompiler(MIXED_TYPES).compile(
        roots() + ([TRAJECTORY] if expose else []), result_format="typed", output_order=order
    )


def inputs(n, t, k, layout="C"):
    result = data(n, t, k, layout)
    result["q"] = np.array([0.05, 0.5, 0.95])
    result["q"].flags.writeable = False
    return result


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("n,k", [(127, 7), (128, 0), (257, 1), (517, 7)])
@pytest.mark.parametrize("cpu", [1, 4])
def test_mixed_roots_full_shape_order_budget_and_owned_generations(layout, order, n, k, cpu):
    args = inputs(n, 129, k, layout)
    original = {key: value.copy() for key, value in args.items()}
    starts = np.array([0], np.int64)
    ends = np.array([129], np.int64)
    with AdaptiveScheduler(cpu_budget=cpu, config=config()) as scheduler:
        first = scheduler.execute(graph(order), args, starts, ends)
        stable = [out.values[0].copy() for out in first.outputs]
        full = scheduler.execute(graph(order, expose=True), args, starts, ends)
        second = scheduler.execute(graph(order), args, starts, ends)
    shapes = [(n, k)] + [(n,)] * 3 + [(n, 3)] + [(n,)] * 2
    for index, out in enumerate(first.outputs):
        value = out.values[0]
        assert value.shape == shapes[index] and not value.flags.writeable
        assert out.root_statuses[0] == 0
        np.testing.assert_array_equal(
            value.view(np.uint64), full.outputs[index].values[0].view(np.uint64)
        )
        np.testing.assert_array_equal(value.view(np.uint64), stable[index].view(np.uint64))
        assert not np.shares_memory(value, second.outputs[index].values[0])
        assert all(not np.shares_memory(value, x) for x in args.values())
        if value.ndim == 2:
            assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
    if k:
        weights = first.outputs[0].values[0]
        assert (weights > 0).all()
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=2e-15, rtol=0)
    for key, value in args.items():
        np.testing.assert_array_equal(value.view(np.uint64), original[key].view(np.uint64))
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in first.audit["native_chunks"]
    )
    assert first.plan.estimated_worker_scratch_bytes >= observed
    if n >= 128:
        assert arena(first) < arena(full) / 2
    if cpu == 4 and n >= 256:
        assert first.plan.parallel_dimension == "array_row"
        assert first.plan.thread_count == min(4, n // 128)
    if order == "F" and n >= 128:
        assert sum(c["root_copy_bytes"] for c in first.audit["native_chunks"]) >= sum(
            x.nbytes for x in stable
        )


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_multi_interval_mixed_outputs_across_transports(layout, order, lane):
    args = inputs(257, 257, 7, layout)
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
        actual = scheduler.execute(graph(order), args, starts, ends)
        full = scheduler.execute(graph(order, expose=True), args, starts, ends)
    assert actual.plan.lane == ("process" if lane.startswith("process") else lane)
    for a, b in zip(actual.outputs, full.outputs[:7], strict=True):
        for row in range(2):
            np.testing.assert_array_equal(
                a.values[row].view(np.uint64), b.values[row].view(np.uint64)
            )
            assert not a.values[row].flags.writeable
            if a.values[row].ndim == 2:
                assert (
                    a.values[row].flags.f_contiguous
                    if order == "F"
                    else a.values[row].flags.c_contiguous
                )


@pytest.mark.parametrize("order", ["C", "F"])
def test_worker_memory_admission_can_use_single_cpu_without_dropping_outputs(order):
    args = inputs(4097, 257, 7)
    starts = np.array([0], np.int64)
    ends = np.array([257], np.int64)
    g = graph(order)
    with AdaptiveScheduler(cpu_budget=1, config=config()) as scheduler:
        budget = scheduler.plan(g, args, starts, ends).estimated_total_memory_bytes
        reference = scheduler.execute(g, args, starts, ends)
    with AdaptiveScheduler(cpu_budget=4, config=config()) as scheduler:
        actual = scheduler.execute(g, args, starts, ends, memory_budget_bytes=budget)
    assert actual.plan.lane == "single" and actual.audit["cpu_tokens"] == 1
    assert actual.plan.estimated_total_memory_bytes <= budget
    for a, b in zip(actual.outputs, reference.outputs, strict=True):
        np.testing.assert_array_equal(a.values[0], b.values[0])


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("cpu", [1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_output_failure_keeps_earliest_logical_node_and_recovers(order, cpu, reverse):
    args = inputs(517, 129, 1)
    w = np.full((517, 1), 2.0)
    w[10] = 1.0
    w[-5] = -1.0
    if reverse:
        w = w[::-1]
    w.flags.writeable = False
    r = np.ones((129, 1))
    r.flags.writeable = False
    args.update(w=w, r=r)
    y = "matmul(log(w),transpose(r))"
    expressions = ["log(w)", f"sum_axis(divide(1,{y}),1)"]
    starts = np.array([0], np.int64)
    ends = np.array([129], np.int64)
    with AdaptiveScheduler(cpu_budget=cpu, config=config()) as scheduler:
        messages = []
        for ex in [expressions, expressions + [y]]:
            with pytest.raises(ValueError) as error:
                scheduler.execute(
                    GraphCompiler(MIXED_TYPES).compile(
                        ex, result_format="typed", output_order=order
                    ),
                    args,
                    starts,
                    ends,
                )
            messages.append(str(error.value))
        assert messages[0] == messages[1] and "DOMAIN_ERROR" in messages[0]
        healthy = scheduler.execute(graph(order), inputs(517, 129, 1), starts, ends)
        assert all(np.isfinite(x.values[0]).all() for x in healthy.outputs)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
def test_mixed_prepared_snapshots_read_current_contents_without_overwriting_prior_results(
    order, layout
):
    args = inputs(517, 129, 7, layout)
    args["w"].flags.writeable = True
    starts = np.array([0], np.int64)
    ends = np.array([129], np.int64)
    with AdaptiveScheduler(cpu_budget=4, config=config()) as scheduler:
        prepared = scheduler.prepare_execution(graph(order), args, starts, ends)
        first = prepared.run_snapshot()
        stable = [x.values[0].copy() for x in first.outputs]
        args["w"] *= 0.5
        current = prepared.run_snapshot()
        ordinary = scheduler.execute(graph(order), args, starts, ends)
    for index, out in enumerate(first.outputs):
        np.testing.assert_array_equal(out.values[0].view(np.uint64), stable[index].view(np.uint64))
        np.testing.assert_array_equal(
            current.outputs[index].values[0].view(np.uint64),
            ordinary.outputs[index].values[0].view(np.uint64),
        )
        assert not np.shares_memory(out.values[0], current.outputs[index].values[0])
