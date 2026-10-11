"""Independent row packets preserve selected folds and full graph ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig

from .test_array_row_parallel import data
from .test_row_batch_execution import TYPES, Y


@pytest.mark.parametrize("rows", [128, 129, 517])
@pytest.mark.parametrize("length", [127, 128, 257])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("cpu", [1, 4])
def test_masked_packets_keep_canonical_steps_tail_and_owned_generations(rows, length, layout, cpu):
    inputs = data(rows, length, 7, layout)
    saved = {name: x.copy() for name, x in inputs.items()}
    predicate = f"less_than({Y},0)"
    delta = f"subtract({Y},.001)"
    selections = [
        f"where({predicate},{Y},-0.)",
        f"where({predicate},0,add(1,{Y}))",
        f"where({predicate},add(1,{Y}),subtract(0,{Y}))",
        f"where({predicate},multiply({delta},{delta}),.1)",
    ]
    roots = [f"sum_axis({x},1)" for x in selections]
    compiler = GraphCompiler(TYPES)
    graph = compiler.compile(roots, result_format="typed")
    # Public selections prevent their removal; the public subtraction also
    # prevents the private square closure. Reference keeps every original node.
    canonical = compiler.compile(roots + selections + [delta, Y], result_format="typed")
    starts = np.array([0], np.int64)
    ends = np.array([length], np.int64)
    with AdaptiveScheduler(
        cpu_budget=cpu, config=PlannerConfig(thread_work_units=1.0)
    ) as scheduler:
        first = scheduler.execute(graph, inputs, starts, ends)
        reference = scheduler.execute(canonical, inputs, starts, ends)
        second = scheduler.execute(graph, inputs, starts, ends)
    for actual, expected, again in zip(
        first.outputs, reference.outputs[:4], second.outputs, strict=True
    ):
        a = actual.values[0]
        b = expected.values[0]
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        np.testing.assert_array_equal(a.view(np.uint64), again.values[0].view(np.uint64))
        assert a.shape == (rows,) and not a.flags.writeable
        assert not np.shares_memory(a, again.values[0])
        assert all(not np.shares_memory(a, x) for x in inputs.values())
        assert actual.root_statuses[0] == 0
    for name, x in inputs.items():
        np.testing.assert_array_equal(x.view(np.uint64), saved[name].view(np.uint64))
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in first.audit["native_chunks"]
    )
    assert first.plan.metadata()["estimated_worker_scratch_bytes"] >= observed


@pytest.mark.parametrize("length", [0, 1, 128, 257])
@pytest.mark.parametrize("mode", ["positive", "negative", "zero"])
def test_masked_packets_keep_empty_and_unselected_nonfinite_values(length, mode):
    inputs = data(129, length, 1)
    inputs["w"] = np.full((129, 1), 0.25)
    inputs["r"] = np.full(
        (length, 1), 1.0 if mode == "positive" else -1.0 if mode == "negative" else -0.0
    )
    for x in inputs.values():
        x.flags.writeable = False
    overflow = f"multiply(1e308,add(2,{Y}))"
    nan = f"subtract({overflow},{overflow})"
    predicate = f"greater_equal({Y},0)" if mode != "negative" else f"less_than({Y},0)"
    selection = f"where({predicate},{Y},{nan})"
    unselected_delta = f"subtract({Y},{nan})"
    square = f"where(logical_not({predicate}),multiply({unselected_delta},{unselected_delta}),.5)"
    roots = [f"sum_axis({selection},1)", f"sum_axis({square},1)"]
    compiler = GraphCompiler(TYPES)
    graph = compiler.compile(roots, result_format="typed")
    canonical = compiler.compile(
        roots + [selection, square, unselected_delta, Y], result_format="typed"
    )
    starts = np.array([0], np.int64)
    ends = np.array([length], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = scheduler.execute(graph, inputs, starts, ends)
        expected = scheduler.execute(canonical, inputs, starts, ends)
    for a, b in zip(actual.outputs, expected.outputs[:2], strict=True):
        np.testing.assert_array_equal(a.values[0].view(np.uint64), b.values[0].view(np.uint64))
    np.testing.assert_array_equal(actual.outputs[1].values[0], np.full(129, length * 0.5))
