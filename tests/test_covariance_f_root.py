"""A symmetric covariance can publish independent F bytes without a C arena."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_covariance_accelerate import MATRIX, compare, counts, inputs
from .test_matmul_library import settings


def compiled(expression="covariance(x)", *, order="F", isolate=False):
    return GraphCompiler({"x": MATRIX}).compile(
        expression, output_order=order,
        error_policy="isolate" if isolate else "raise",
    )


def assert_direct(result, expected):
    assert result.outputs[0].order == "F"
    for value in result.outputs[0].values:
        assert value.flags.f_contiguous and not value.flags.writeable
        np.testing.assert_array_equal(value.view(np.uint64), expected.view(np.uint64))
    for chunk in result.audit["native_chunks"]:
        assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
        assert chunk["direct_output_bytes"] == chunk["rows"] * expected.nbytes


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("pattern", ["normal", "positive_zero", "signed_zero", "nan", "inf", "overflow", "reverse"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_canonical_f_root_bits_transport_and_lifetime(layout, pattern, lane):
    x = inputs(layout, n=17, d=9).copy(order=layout.upper())
    if pattern == "positive_zero":
        x.fill(0)
    elif pattern == "signed_zero":
        x.fill(-0.)
    elif pattern == "nan":
        x[3, -1] = np.array([0x7FF8000000000042], np.uint64).view(np.float64)[0]
    elif pattern == "inf":
        x[3, -1] = np.inf
    elif pattern == "overflow":
        x *= 1e200
    elif pattern == "reverse":
        x = x[::-1]
    x.flags.writeable = False
    before = x.view(np.uint64).copy()
    expected = op.covariance(x)
    np.testing.assert_array_equal(expected.view(np.uint64), expected.T.view(np.uint64))
    starts, ends = np.zeros(2, np.int64), np.full(2, len(x), np.int64)
    # The process decoder reconstructs metadata from the native program; pickle
    # also reconstructs it instead of retaining an old cached F admission.
    graph = pickle.loads(pickle.dumps(compiled()))
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"x": x}, starts, ends)
        again = scheduler.execute(graph, {"x": x}, starts, ends)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    assert_direct(result, expected)
    assert_direct(again, expected)
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[0].values[1])
    assert not np.shares_memory(result.outputs[0].values[0], again.outputs[0].values[0])
    assert not np.shares_memory(result.outputs[0].values[0], x)
    np.testing.assert_array_equal(x.view(np.uint64), before)


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("pattern", ["normal", "scaled", "positive_zero", "signed_zero"])
def test_prepared_f_root_reuses_canonical_writer_and_reads_live_inputs(layout, pattern):
    x = inputs(layout, n=513, d=257)
    x.setflags(write=True)
    if pattern == "scaled":
        x *= 1e3
    elif pattern == "positive_zero":
        x.fill(0)
    elif pattern == "signed_zero":
        x.fill(-0.)
    x.setflags(write=False)
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        c_result = scheduler.execute(compiled(order="C"), {"x": x}, starts, ends)
        prepared = scheduler.prepare_execution(compiled(), {"x": x}, starts, ends)
        first = prepared.run_snapshot()
        expected = c_result.outputs[0].values[0]
        assert_direct(first, expected)
        compare(first.outputs[0].values[0], op.covariance(x))
        saved = first.outputs[0].values[0].copy()
        x.setflags(write=True)
        x.fill(0)
        x.setflags(write=False)
        second = prepared.run_snapshot()
    assert_direct(second, op.covariance(x))
    np.testing.assert_array_equal(first.outputs[0].values[0].view(np.uint64), saved.view(np.uint64))
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    assert counts(first, "accelerate_covariance_attempts") == counts(c_result, "accelerate_covariance_attempts")


@pytest.mark.parametrize("d", [0, 1, 17])
def test_planner_has_no_extra_f_matrix_and_enforces_the_remaining_budget(d):
    x = inputs("f", n=33, d=d)
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        c_plan = scheduler.plan(compiled(order="C"), {"x": x}, starts, ends)
        f_plan = scheduler.plan(compiled(), {"x": x}, starts, ends)
        assert f_plan.estimated_worker_scratch_bytes == c_plan.estimated_worker_scratch_bytes
        assert f_plan.estimated_total_memory_bytes == c_plan.estimated_total_memory_bytes
        with pytest.raises(MemoryError):
            scheduler.plan(compiled(), {"x": x}, starts, ends,
                           memory_budget_bytes=f_plan.estimated_total_memory_bytes - 1)
        result = scheduler.execute(compiled(), {"x": x}, starts, ends)
    assert_direct(result, op.covariance(x))


@pytest.mark.parametrize("mode", ["multiple", "isolate", "iterate"])
def test_unqualified_graphs_keep_the_budgeted_f_materialization(mode):
    x = inputs("f", n=33, d=17)
    expression = "covariance(x)"
    if mode == "multiple":
        expression = [expression, f"trace({expression})"]
    elif mode == "iterate":
        expression = "iterate(covariance(x),covariance(x),0,1)"
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(compiled(expression, isolate=mode == "isolate"),
                                   {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
    np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64), op.covariance(x).view(np.uint64))
    assert result.outputs[0].values[0].flags.f_contiguous
    assert sum(chunk["numeric_arena_bytes"] for chunk in result.audit["native_chunks"]) >= 17 * 17 * 8


@pytest.mark.parametrize("rows", [0, 1])
def test_f_root_failure_keeps_the_original_exception(rows):
    x = inputs("f", n=rows, d=17)
    with AdaptiveScheduler(cpu_budget=1) as scheduler, pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
        scheduler.execute(compiled(), {"x": x}, np.zeros(1, np.int64), np.array([rows], np.int64))
