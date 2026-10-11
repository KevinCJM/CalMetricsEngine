"""A private argmax uses this call's independent root with canonical errors."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig

from .test_row_subview import digest, exact


def fixture(layout="C", n=300013):
    rng = np.random.default_rng(1741)
    matrix = rng.integers(-4, 5, (n, 3)).astype(np.float64)
    if n:
        matrix[: min(4, n)] = [[0.0, -0.0, 0.0], [3, 3, 2], [-1, -1, -1], [0, 1, 1]][: min(4, n)]
    if layout == "F":
        matrix = np.asfortranarray(matrix)
    elif layout == "reverse":
        matrix = matrix[::-1].copy()[::-1]
    elif layout == "column_stride":
        storage = np.empty((n, 6))
        storage[:, ::2] = matrix
        matrix = storage[:, ::2]
    elif layout == "row_stride":
        storage = np.empty((n * 2, 3))
        storage[::2] = matrix
        matrix = storage[::2]
    valid = rng.random(n) > 0.2
    result = dict(matrix=matrix, valid=valid)
    for value in result.values():
        value.flags.writeable = False
    return result


def compile_graph(data, exposure=None, policy="raise", order="C", bad=None):
    n, width = data["matrix"].shape
    types = {
        "matrix": dict(kind="matrix", axes=["time", "factor"], shape=[str(n), str(width)]),
        "valid": dict(kind="series", dtype="bool", axes=["time"], shape=[str(n)]),
    }
    source = "argmax_axis(matrix,1)"
    result = "-1"
    for state in range(3):
        threshold = "1.5" if bad == "threshold" and state == 1 else str(state)
        code = "-2" if bad == "state" and state == 1 else str((state + 2) % 3)
        result = f"state_select({source}=={threshold},{code},{result},valid)"
    roots = [result]
    if exposure == "source":
        roots.append(source)
    elif exposure == "predicate":
        roots.append(source + "==1")
    elif exposure == "consumer":
        roots.append(source + "==4")
    elif exposure == "earlier_private":
        roots.insert(0, "argmax_axis(matrix*2,1)==4")
    return GraphCompiler(types).compile(
        roots, result_format="typed", error_policy=policy, output_order=order
    )


def lane_config(lane):
    return PlannerConfig(
        thread_work_units=1 if lane != "single" else 1e100,
        process_work_units=1 if lane.startswith("process") else 1e100,
        min_rows_per_worker=1,
        max_processes=2,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
    )


def run(data, graph, lane="single", expect_loan=True):
    n = len(data["valid"])
    rows = 1 if lane == "single" else 2
    starts = np.zeros(rows, np.int64)
    ends = np.full(rows, n, np.int64)
    before = digest(data)
    with AdaptiveScheduler(cpu_budget=1) as reference:
        control = reference.prepare_execution(graph, data, starts, ends).run_snapshot()
    with AdaptiveScheduler(cpu_budget=rows, config=lane_config(lane)) as scheduler:
        plan = scheduler.plan(graph, data, starts, ends)
        first = scheduler.execute(
            graph, data, starts, ends, memory_budget_bytes=plan.estimated_total_memory_bytes
        )
        saved = [v.tobytes() for output in first.outputs for v in output.values]
        second = scheduler.execute(graph, data, starts, ends)
        exact(first, control)
        exact(second, control)
        assert saved == [v.tobytes() for output in first.outputs for v in output.values]
        for old, new in zip(first.outputs, second.outputs, strict=True):
            for a, b in zip(old.values, new.values, strict=True):
                assert not a.flags.writeable and not np.shares_memory(a, b)
                assert not any(np.shares_memory(a, x) for x in data.values())
        assert scheduler.peak_active_cpu_tokens <= rows
        for chunk in first.audit["native_chunks"]:
            if expect_loan:
                assert chunk["numeric_arena_bytes"] == 0
                assert chunk["algorithm_copy_bytes"] >= n * 8
                assert chunk["pointwise_workspace_capacity_bytes"] == 4096 + 4096 * 8
            else:
                assert chunk["algorithm_copy_bytes"] == 0
    expected = (np.argmax(data["matrix"], axis=1) + 2) % 3
    expected = np.where(data["valid"], expected, -1).astype(np.int64)
    state_output = next(o for o in first.outputs if o.values[0].dtype == np.int64)
    for actual in state_output.values:
        np.testing.assert_array_equal(actual, expected)
    assert before == digest(data)
    exact(first, control)
    return first


@pytest.mark.parametrize("layout", ["C", "F", "reverse", "column_stride", "row_stride"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_canonical_ties_strides_and_independent_results(layout, order):
    data = fixture(layout)
    run(data, compile_graph(data, order=order))


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
def test_transport_and_original_budget(lane):
    data = fixture("F")
    run(data, compile_graph(data), lane)


@pytest.mark.parametrize("exposure", ["source", "predicate", "consumer", "earlier_private"])
def test_external_consumers_keep_original_storage(exposure):
    data = fixture()
    run(data, compile_graph(data, exposure), expect_loan=False)


@pytest.mark.parametrize("n", [0, 1, 2, 4095, 4096, 4097, 131071, 131072, 131073, 250000])
def test_tile_and_minimum_loan_boundary(n):
    data = fixture(n=n)
    run(data, compile_graph(data), expect_loan=n >= 131072)


@pytest.mark.parametrize("bad", ["threshold", "state", "nan_first", "nan_tile", "inf_last"])
def test_original_exception_and_next_request_recovery(bad):
    data = fixture()
    graph = compile_graph(data, bad=bad)
    if bad.startswith("nan") or bad == "inf_last":
        matrix = data["matrix"].copy()
        row = 0 if bad == "nan_first" else 4096 if bad == "nan_tile" else len(matrix) - 1
        matrix[row, 1] = np.inf if bad == "inf_last" else np.nan
        matrix.flags.writeable = False
        data = dict(data, matrix=matrix)
    starts = np.array([0], np.int64)
    ends = np.array([len(data["valid"])], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(Exception) as reference:
            scheduler.prepare_execution(graph, data, starts, ends).run_snapshot()
        with pytest.raises(type(reference.value)) as actual:
            scheduler.execute(graph, data, starts, ends)
        assert str(actual.value) == str(reference.value)
    healthy = fixture()
    run(healthy, compile_graph(healthy))


def test_isolate_preserves_original_materialization():
    data = fixture()
    run(data, compile_graph(data, policy="isolate"), expect_loan=False)
