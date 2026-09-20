from __future__ import annotations

import numpy as np

from calmetrics_engine import (
    AdaptiveScheduler,
    GraphCompiler,
    PlannerConfig,
)


def _heavy_branch_formulas() -> tuple[str, ...]:
    returns = "divide(difference(x,1),lag(x,1))"
    return (
        f"mean({returns})",
        f"std({returns},1)",
        f"median({returns})",
        f"quantile({returns},.05)",
        f"min_value({returns})",
        f"max_value({returns})",
        f"mean_absolute_deviation({returns})",
        f"root_mean_square({returns})",
        "-min_value(drawdown_series(x))",
        "linear_slope(x)",
        "linear_r_squared(x)",
        "count_true(new_high_mask(x))/length(x)",
    )


def test_physical_cost_accounts_for_fusion_and_branches():
    formulas = (
        "mean(x)",
        "std(x,1)",
        "min_value(x)",
        "max_value(x)",
        "root_mean_square(x)",
        "mean_absolute_deviation(x)",
        "median(x)",
        "quantile(x,.25)",
        "total_return(x)",
    )
    graph = GraphCompiler({"x": "series"}).compile(formulas)
    metadata = graph.metadata()

    assert metadata["branch_count"] == 2
    assert sorted(len(roots) for roots in metadata["branch_roots"]) == [2, 7]
    for branch in metadata["branch_details"]:
        assert branch["node_count"] <= metadata["node_count"]
        assert branch["numeric_slots"] <= metadata["numeric_slots"]
        assert branch["mask_slots"] <= metadata["mask_slots"]
        assert branch["source_nodes"] == sorted(branch["source_nodes"])
    physical = metadata["physical_cost"]
    assert physical["linear_per_observation"] > 0
    assert physical["sort_nlogn"] == 2.0

    x = np.linspace(1.0, 2.0, 100_000, dtype=np.float64)
    starts = np.asarray([0], dtype=np.int64)
    ends = np.asarray([x.size], dtype=np.int64)
    with AdaptiveScheduler(cpu_budget=4) as scheduler:
        plan = scheduler.plan(graph, {"x": x}, starts, ends)

    assert plan.estimated_work_units < plan.estimated_logical_work_units


def test_interval_partition_uses_physical_work_not_equal_row_count():
    graph = GraphCompiler({"x": "series"}).compile("mean(x)")
    x = np.arange(2_000, dtype=np.float64)
    starts = np.zeros(8, dtype=np.int64)
    ends = np.asarray([1000, 10, 10, 10, 10, 10, 10, 10], dtype=np.int64)
    product_ids = np.zeros(8, dtype=np.int64)
    config = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1e100,
        min_rows_per_worker=1,
    )
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        plan = scheduler.plan(
            graph,
            {"x": x},
            starts,
            ends,
            product_ids=product_ids,
        )

    metadata = plan.metadata()
    assert metadata["lane"] == "thread"
    assert metadata["parallel_dimension"] == "interval"
    assert metadata["chunks"] == [(0, 1), (1, 8)]


def test_single_row_heavy_dag_forks_independent_branches_and_matches_serial():
    graph = GraphCompiler({"x": "series"}).compile(_heavy_branch_formulas())
    x = np.linspace(1.0, 2.0, 200_000, dtype=np.float64)
    starts = np.asarray([0], dtype=np.int64)
    ends = np.asarray([x.size], dtype=np.int64)

    serial_config = PlannerConfig(
        thread_work_units=1e99,
        dag_branch_work_units=1e99,
        process_work_units=1e100,
    )
    with AdaptiveScheduler(cpu_budget=4, config=serial_config) as scheduler:
        serial_plan = scheduler.plan(graph, {"x": x}, starts, ends)
        expected = scheduler.execute(graph, {"x": x}, starts, ends, plan=serial_plan).values

    with AdaptiveScheduler(cpu_budget=4) as scheduler:
        plan = scheduler.plan(graph, {"x": x}, starts, ends)
        result = scheduler.execute(graph, {"x": x}, starts, ends, plan=plan)
        peak = scheduler.peak_active_cpu_tokens

    metadata = plan.metadata()
    assert metadata["lane"] == "thread"
    assert metadata["parallel_dimension"] == "dag_branch"
    assert metadata["branch_task_count"] == graph.metadata()["branch_count"]
    assert 2 <= metadata["thread_count"] <= 4
    assert peak <= 4
    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=0.0)


def test_dag_branch_respects_memory_reduced_cpu_lease():
    graph = GraphCompiler({"x": "series"}).compile(_heavy_branch_formulas())
    x = np.linspace(1.0, 2.0, 200_000, dtype=np.float64)
    starts = np.asarray([0], dtype=np.int64)
    ends = np.asarray([x.size], dtype=np.int64)

    with AdaptiveScheduler(cpu_budget=4) as scheduler:
        unconstrained = scheduler.plan(graph, {"x": x}, starts, ends)
        assert unconstrained.parallel_dimension == "dag_branch"
        fixed = (
            unconstrained.estimated_input_bytes
            + unconstrained.estimated_output_bytes
            + unconstrained.row_count * 16
        )
        budget = fixed + unconstrained.estimated_worker_scratch_bytes * 2 + 4096
        constrained = scheduler.plan(
            graph,
            {"x": x},
            starts,
            ends,
            memory_budget_bytes=budget,
        )
        result = scheduler.execute(graph, {"x": x}, starts, ends, plan=constrained)
        peak = scheduler.peak_active_cpu_tokens

    assert constrained.parallel_dimension == "dag_branch"
    assert constrained.thread_count == 2
    assert constrained.metadata()["branch_task_count"] > constrained.thread_count
    assert "memory_budget_reduced_parallelism" in constrained.reason_codes
    assert peak <= 2
    assert np.isfinite(result.values).all()
