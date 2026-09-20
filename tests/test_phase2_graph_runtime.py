from __future__ import annotations

import asyncio
from multiprocessing import shared_memory

import numpy as np
import pytest

from calmetrics_engine import (
    AdaptiveScheduler,
    GraphCompileError,
    GraphCompiler,
    PlannerConfig,
    SharedInputBundle,
)
from calmetrics_engine import (
    operators as op,
)
from calmetrics_engine.runtime import _product_chunks


def returns_graph():
    return GraphCompiler({"nav": "series", "risk_free": "scalar"}).compile(
        [
            "mean(difference(nav, 1) / lag(nav, 1))",
            "std(difference(nav, 1) / lag(nav, 1), 1)",
            "mean(difference(nav, 1) / lag(nav, 1)) - risk_free",
            "-min_value(drawdown_series(nav))",
            "count_true(new_high_mask(nav)) / length(nav)",
        ]
    )


def dataset(products=8, history=96):
    product = np.arange(products, dtype=np.float64)[:, None]
    day = np.arange(history - 1, dtype=np.float64)[None, :]
    returns = 0.0005 + 0.01 * np.sin(day * 0.17 + product * 0.31)
    nav = np.empty((products, history), dtype=np.float64)
    nav[:, 0] = 1.0
    nav[:, 1:] = np.cumprod(1.0 + returns, axis=1)
    flat = np.ascontiguousarray(nav.reshape(-1))
    starts = np.ascontiguousarray(
        [
            product_index * history + offset
            for product_index in range(products)
            for offset in (0, 16)
        ],
        dtype=np.int64,
    )
    ends = np.ascontiguousarray(
        [
            product_index * history + end
            for product_index in range(products)
            for end in (history, history - 8)
        ],
        dtype=np.int64,
    )
    return flat, starts, ends


def numpy_reference(nav, starts, ends, risk_free):
    output = np.empty((starts.size, 5), dtype=np.float64)
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        values = nav[start:end]
        returns = values[1:] / values[:-1] - 1.0
        running_peak = np.maximum.accumulate(values)
        drawdown = values / running_peak - 1.0
        highs = np.empty(values.size, dtype=np.uint8)
        highs[0] = 1
        peak = values[0]
        for index in range(1, values.size):
            highs[index] = values[index] > peak
            if highs[index]:
                peak = values[index]
        output[row] = [
            returns.mean(),
            returns.std(ddof=1),
            returns.mean() - risk_free,
            -drawdown.min(),
            highs.sum() / values.size,
        ]
    return output


def test_restricted_ast_cse_and_liveness():
    graph = returns_graph()
    assert graph.cse_eliminated_nodes > 0
    assert graph.numeric_slots >= 2
    assert graph.mask_slots >= 1
    assert len(graph.roots) == 5
    assert graph.parameter_names == ("risk_free",)

    for expression in (
        "__import__('os')",
        "nav[0]",
        "nav.real",
        "(lambda x: x)(nav)",
        "[x for x in nav]",
    ):
        with pytest.raises(GraphCompileError):
            GraphCompiler({"nav": "series"}).compile(expression)

    # Value-class mistakes fail during graph construction rather than waiting
    # for native prepare/execute.
    for expression in (
        "logical_and(nav, nav)",
        "mean(greater_than(nav, 0))",
        "fit_slope(1)",
        "interval_start(1)",
        "rolling_mean(nav, nav)",
        "value_at(1, 0)",
    ):
        with pytest.raises(GraphCompileError):
            GraphCompiler({"nav": "series"}).compile(expression)


def test_native_graph_fuses_reductions_and_order_stats_once():
    graph = GraphCompiler({"x": "series"}).compile(
        [
            "mean(x)",
            "std(x, 1)",
            "min_value(x)",
            "max_value(x)",
            "root_mean_square(x)",
            "mean_absolute_deviation(x)",
            "median(x)",
            "quantile(x, 0.25)",
            "total_return(x)",
        ]
    )
    values = np.asarray([0.01, -0.02, 0.03, 0.04, -0.01, 0.02], dtype=np.float64)
    starts = np.asarray([0], dtype=np.int64)
    ends = np.asarray([values.size], dtype=np.int64)
    config = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)

    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        result = scheduler.execute(graph, {"x": values}, starts, ends)

    expected = np.asarray(
        [
            values.mean(),
            values.std(ddof=1),
            values.min(),
            values.max(),
            np.sqrt(np.mean(values * values)),
            np.mean(np.abs(values - values.mean())),
            np.median(values),
            np.quantile(values, 0.25),
            np.prod(1.0 + values) - 1.0,
        ]
    )
    np.testing.assert_allclose(result.values[0], expected, rtol=2e-12, atol=2e-14)
    audit = result.audit["native_chunks"][0]
    assert audit["fused_scalar_calls"] == 9
    assert audit["summary_source_scans"] == 1
    assert audit["order_stat_sorts"] == 1


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([0.1, -0.2, 0.3, 0.05, -0.01], dtype=np.float64),
        np.asarray([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        np.asarray([1.0, 2.0, np.nan, 3.0, 4.0], dtype=np.float64),
    ],
)
def test_fused_reduction_edge_semantics_match_canonical_operators(values):
    expressions = [
        "sum(x)",
        "product(x)",
        "mean(x)",
        "min_value(x)",
        "max_value(x)",
        "variance(x, 1)",
        "std(x, 1)",
        "mean_absolute_deviation(x)",
        "root_mean_square(x)",
        "median(x)",
        "quantile(x, 0.25)",
        "total_return(x)",
    ]
    graph = GraphCompiler({"x": "series"}).compile(expressions)
    starts = np.asarray([0], dtype=np.int64)
    ends = np.asarray([values.size], dtype=np.int64)
    config = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)

    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        actual = scheduler.execute(graph, {"x": values}, starts, ends).values[0]

    expected = np.asarray(
        [
            op.sum(values),
            op.product(values),
            op.mean(values),
            op.min_value(values),
            op.max_value(values),
            op.variance(values, 1),
            op.std(values, 1),
            op.mean_absolute_deviation(values),
            op.root_mean_square(values),
            op.median(values),
            op.quantile(values, 0.25),
            op.total_return(values),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_prepared_single_execution_reuses_bound_native_batch():
    graph = returns_graph()
    nav, starts, ends = dataset(products=1, history=96)
    parameters = {"risk_free": 0.0001}
    config = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)

    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        plan = scheduler.plan(graph, {"nav": nav}, starts, ends)
        prepared = scheduler.prepare_execution(
            graph,
            {"nav": nav},
            starts,
            ends,
            parameters=parameters,
            plan=plan,
        )
        first = prepared.run()
        first_pointer = first.__array_interface__["data"][0]
        first_snapshot = first.copy()
        second = prepared.run()

        np.testing.assert_array_equal(second, first_snapshot)
        assert second.__array_interface__["data"][0] == first_pointer
        assert np.shares_memory(first, second)

        audited = prepared.run_audit()
        np.testing.assert_array_equal(audited.values, second)
        assert audited.audit["prepared"]
        assert audited.audit["boundary_copy_bytes"] == 0
        assert audited.audit["native_chunks"][0]["input_copy_bytes"] == 0


def test_cached_single_execute_returns_independent_outputs():
    graph = returns_graph()
    nav, starts, ends = dataset(products=1, history=96)
    config = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)

    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        plan = scheduler.plan(graph, {"nav": nav}, starts, ends)
        first = scheduler.execute(
            graph,
            {"nav": nav},
            starts,
            ends,
            parameters={"risk_free": 0.0001},
            plan=plan,
        )
        snapshot = first.values.copy()
        second = scheduler.execute(
            graph,
            {"nav": nav},
            starts,
            ends,
            parameters={"risk_free": 0.0001},
            plan=plan,
        )

    np.testing.assert_array_equal(first.values, snapshot)
    np.testing.assert_array_equal(second.values, snapshot)
    assert not np.shares_memory(first.values, second.values)
    assert second.audit["prepared_cached"]
    assert second.audit["output_copy_bytes"] == 0


def test_prepare_execution_rejects_parallel_plan():
    graph = returns_graph()
    nav, starts, ends = dataset(products=8, history=96)
    config = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1e100,
        min_rows_per_worker=1,
    )
    with AdaptiveScheduler(cpu_budget=4, config=config) as scheduler:
        plan = scheduler.plan(graph, {"nav": nav}, starts, ends)
        assert plan.lane == "thread"
        with pytest.raises(ValueError, match="single-lane"):
            scheduler.prepare_execution(
                graph,
                {"nav": nav},
                starts,
                ends,
                plan=plan,
            )


def test_native_graph_matches_numpy_and_zero_copy():
    graph = returns_graph()
    nav, starts, ends = dataset()
    config = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)
    with AdaptiveScheduler(cpu_budget=4, config=config) as scheduler:
        result = scheduler.execute(
            graph,
            {"nav": nav},
            starts,
            ends,
            parameters={"risk_free": 0.0001},
        )
    expected = numpy_reference(nav, starts, ends, 0.0001)
    np.testing.assert_allclose(result.values, expected, rtol=2e-12, atol=2e-14)
    assert result.plan.lane == "single"
    assert result.audit["boundary_copy_bytes"] == 0
    assert all(chunk["input_copy_bytes"] == 0 for chunk in result.audit["native_chunks"])
    assert all(chunk["python_operator_calls"] == 0 for chunk in result.audit["native_chunks"])


def test_planner_selects_single_thread_process_shared_and_respects_budget():
    graph = returns_graph()
    nav, starts, ends = dataset(products=16, history=128)
    inputs = {"nav": nav}

    single = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)
    thread = PlannerConfig(thread_work_units=1.0, process_work_units=1e100, min_rows_per_worker=2)
    process = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1.0,
        shared_memory_threshold_bytes=1,
        min_rows_per_worker=2,
        max_processes=8,
    )

    with AdaptiveScheduler(cpu_budget=4, config=single) as scheduler:
        assert scheduler.plan(graph, inputs, starts, ends).lane == "single"
    product_ids = np.ascontiguousarray(starts // 128, dtype=np.int64)
    with AdaptiveScheduler(cpu_budget=4, config=thread) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends, product_ids=product_ids)
        assert plan.lane == "thread"
        assert 1 < plan.thread_count <= 4
        assert plan.process_count == 1
        assert plan.parallel_dimension == "product"
        assert plan.product_count == 16
    with AdaptiveScheduler(cpu_budget=4, config=process) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends, product_ids=product_ids)
        assert plan.lane == "process"
        assert 1 <= plan.process_count <= 4
        assert plan.thread_count == 1
        assert plan.threads_per_process == 1
        assert plan.use_shared_memory
        assert plan.process_count * plan.threads_per_process <= plan.cpu_budget

        one_product = np.zeros(starts.size, dtype=np.int64)
        interval_plan = scheduler.plan(graph, inputs, starts, ends, product_ids=one_product)
        assert interval_plan.parallel_dimension == "interval"
        assert interval_plan.product_count == 1

    no_shared_process = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1.0,
        shared_memory_threshold_bytes=1 << 30,
        min_rows_per_worker=2,
        max_processes=4,
    )
    with AdaptiveScheduler(cpu_budget=4, config=no_shared_process) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        assert plan.lane == "process"
        assert not plan.use_shared_memory


def test_product_chunks_balance_observation_work_without_splitting_products():
    product_ids = np.asarray([0, 0, 1, 2, 2, 2, 3], dtype=np.int64)
    starts = np.zeros(product_ids.size, dtype=np.int64)
    ends = np.asarray([100, 100, 10, 50, 50, 50, 20], dtype=np.int64)

    chunks = _product_chunks(product_ids, starts, ends, workers=2)

    assert chunks == [(0, 2), (2, 7)]
    weights = [int(np.sum(ends[begin:end] - starts[begin:end])) for begin, end in chunks]
    assert weights == [200, 180]
    for begin, end in chunks:
        if begin:
            assert product_ids[begin - 1] != product_ids[begin]
        if end < product_ids.size:
            assert product_ids[end - 1] != product_ids[end]


def test_thread_lane_matches_single_lane():
    graph = returns_graph()
    nav, starts, ends = dataset(products=24, history=128)
    parameters = {"risk_free": 0.0002}

    single_config = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)
    thread_config = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1e100,
        min_rows_per_worker=2,
    )
    with AdaptiveScheduler(cpu_budget=4, config=single_config) as scheduler:
        single = scheduler.execute(graph, {"nav": nav}, starts, ends, parameters=parameters).values
    product_ids = np.ascontiguousarray(starts // 128, dtype=np.int64)
    with AdaptiveScheduler(cpu_budget=4, config=thread_config) as scheduler:
        threaded = scheduler.execute(
            graph,
            {"nav": nav},
            starts,
            ends,
            parameters=parameters,
            product_ids=product_ids,
        )
    np.testing.assert_array_equal(threaded.values, single)
    assert threaded.plan.lane == "thread"
    assert threaded.plan.parallel_dimension == "product"
    assert threaded.audit["cpu_tokens"] == threaded.plan.thread_count
    assert threaded.audit["cpu_tokens"] <= threaded.audit["cpu_budget"]
    assert len(threaded.audit["native_chunks"]) == threaded.plan.thread_count


def test_process_shared_lane_matches_single_and_reusable_shared_input():
    graph = returns_graph()
    nav, starts, ends = dataset(products=24, history=128)
    parameters = {"risk_free": 0.0002}
    single_config = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)
    process_config = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1.0,
        shared_memory_threshold_bytes=1,
        min_rows_per_worker=4,
        max_processes=4,
    )

    with AdaptiveScheduler(cpu_budget=4, config=single_config) as scheduler:
        expected = scheduler.execute(
            graph, {"nav": nav}, starts, ends, parameters=parameters
        ).values

    bundle = SharedInputBundle.from_inputs({"nav": nav})
    descriptor = next(iter(bundle.descriptors.values()))
    try:
        with AdaptiveScheduler(cpu_budget=4, config=process_config) as scheduler:
            result = scheduler.execute(
                graph, bundle, starts, ends, parameters=parameters, timeout=20
            )
        np.testing.assert_array_equal(result.values, expected)
        assert result.plan.lane == "process"
        assert result.plan.use_shared_memory
        assert result.audit["boundary_copy_bytes"] == starts.nbytes + ends.nbytes
        assert result.audit["shared_memory_bytes"] >= nav.nbytes
    finally:
        bundle.release()

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=descriptor.name, create=False)


def test_hard_stop_timeout_terminates_isolated_process():
    graph = returns_graph()
    nav, starts, ends = dataset(products=8, history=96)
    config = PlannerConfig(
        thread_work_units=1e99,
        process_work_units=1e100,
        min_rows_per_worker=2,
        max_processes=2,
    )
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        with pytest.raises(TimeoutError):
            scheduler.execute(
                graph,
                {"nav": nav},
                starts,
                ends,
                parameters={"risk_free": 0.0},
                hard_stop=True,
                timeout=1e-6,
            )


def test_hard_stop_uses_isolated_process_transport():
    graph = returns_graph()
    nav, starts, ends = dataset(products=8, history=96)
    config = PlannerConfig(
        thread_work_units=1e99,
        process_work_units=1e100,
        shared_memory_threshold_bytes=1 << 30,
        min_rows_per_worker=2,
        max_processes=2,
    )
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        plan = scheduler.plan(
            graph,
            {"nav": nav},
            starts,
            ends,
            hard_stop=True,
        )
        assert plan.lane == "process"
        assert plan.hard_stop
        assert plan.use_shared_memory
        result = scheduler.execute(
            graph,
            {"nav": nav},
            starts,
            ends,
            parameters={"risk_free": 0.0},
            plan=plan,
            timeout=20,
        )
    np.testing.assert_allclose(
        result.values,
        numpy_reference(nav, starts, ends, 0.0),
        rtol=2e-12,
        atol=2e-14,
    )


def test_memory_budget_reduces_parallelism():
    graph = returns_graph()
    nav, starts, ends = dataset(products=32, history=256)
    config = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1e100,
        min_rows_per_worker=1,
    )
    with AdaptiveScheduler(cpu_budget=8, config=config) as scheduler:
        unconstrained = scheduler.plan(graph, {"nav": nav}, starts, ends)
        one_worker_memory = (
            nav.nbytes
            + starts.size * len(graph.roots) * 8
            + unconstrained.estimated_worker_scratch_bytes
        )
        constrained = scheduler.plan(
            graph,
            {"nav": nav},
            starts,
            ends,
            memory_budget_bytes=one_worker_memory + 4096,
        )
    assert unconstrained.thread_count > constrained.thread_count
    assert constrained.thread_count == 1
    assert "memory_budget_reduced_parallelism" in constrained.reason_codes


def test_coroutine_orchestration_runs_multiple_independent_graph_jobs():
    graph = returns_graph()
    nav, starts, ends = dataset(products=12, history=96)
    config = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1e100,
        min_rows_per_worker=2,
    )

    async def run_jobs():
        with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
            jobs = [
                {
                    "graph": graph,
                    "inputs": {"nav": nav},
                    "starts": starts,
                    "ends": ends,
                    "parameters": {"risk_free": rate},
                }
                for rate in (0.0, 0.0001, 0.0002)
            ]
            return await scheduler.execute_many_async(jobs)

    results = asyncio.run(run_jobs())
    assert len(results) == 3
    for result, rate in zip(results, (0.0, 0.0001, 0.0002), strict=True):
        assert result.plan.async_orchestration
        assert result.plan.lane == "thread"
        assert result.audit["cpu_tokens"] == 2
        assert result.audit["cpu_budget"] == 2
        assert result.audit["queue_wait_ms"] >= 0.0
        assert "coroutine_orchestration_for_async_boundary" in result.plan.reason_codes
        np.testing.assert_allclose(
            result.values,
            numpy_reference(nav, starts, ends, rate),
            rtol=2e-12,
            atol=2e-14,
        )
