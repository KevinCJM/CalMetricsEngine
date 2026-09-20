"""Phase 2 full-graph benchmark against BetterSaaTaa fused NJIT."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from calmetrics_engine import (
    AdaptiveScheduler,
    GraphCompiler,
    PlannerConfig,
    SharedInputBundle,
)

RETURNS = "divide(difference(adjusted_nav,1),lag(adjusted_nav,1))"
FORMULAS = (
    f"total_return({RETURNS})",
    f"annualized_return({RETURNS},252)",
    f"mean({RETURNS})",
    f"std({RETURNS},1)",
    f"median({RETURNS})",
    f"min_value({RETURNS})",
    f"max_value({RETURNS})",
    f"quantile({RETURNS},0.05)",
    f"mean_absolute_deviation({RETURNS})",
    f"root_mean_square({RETURNS})",
    "-min_value(drawdown_series(adjusted_nav))",
    "count_true(new_high_mask(adjusted_nav))/length(adjusted_nav)",
    "linear_slope(adjusted_nav)",
    "linear_r_squared(adjusted_nav)",
    f"mean_where({RETURNS},greater_than({RETURNS},0))",
    f"std_where({RETURNS},less_than({RETURNS},0))",
)


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def measure(function, repeats: int) -> dict[str, float]:
    wall = []
    cpu = []
    for _ in range(repeats):
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        function()
        wall.append(time.perf_counter_ns() - start)
        cpu.append(time.process_time_ns() - cpu_start)
    return {
        "median_ms": statistics.median(wall) / 1e6,
        "minimum_ms": min(wall) / 1e6,
        "median_cpu_ms": statistics.median(cpu) / 1e6,
    }


def paired_measure(left, right, repeats: int) -> dict[str, object]:
    left_wall = []
    right_wall = []
    for index in range(max(5, repeats)):
        order = ((left, left_wall), (right, right_wall))
        if index % 2:
            order = tuple(reversed(order))
        for function, samples in order:
            started = time.perf_counter_ns()
            function()
            samples.append(time.perf_counter_ns() - started)
    left_ms = statistics.median(left_wall) / 1e6
    right_ms = statistics.median(right_wall) / 1e6
    return {
        "left_median_ms": left_ms,
        "right_median_ms": right_ms,
        "right_over_left": right_ms / left_ms,
        "rounds": max(5, repeats),
        "alternating_first_runner": True,
    }


def data(products: int, history: int, intervals_per_product: int):
    product = np.arange(products, dtype=np.float64)[:, None]
    day = np.arange(history - 1, dtype=np.float64)[None, :]
    returns = (
        0.0002
        + 0.006 * np.sin((day + 1.0) * 0.071 + product * 0.17)
        + 0.003 * np.cos((day + 1.0) * 0.037 + product * 0.11)
    )
    nav = np.empty((products, history), dtype=np.float64)
    nav[:, 0] = 1.0 + product[:, 0] * 0.0001
    nav[:, 1:] = nav[:, :1] * np.cumprod(1.0 + returns, axis=1)
    flat = np.ascontiguousarray(nav.reshape(-1))

    lengths = np.asarray([63, 126, 252, 504, 756, 1008, 1260, 1512], dtype=np.int64)
    starts = []
    ends = []
    product_ids = []
    for product_index in range(products):
        base = product_index * history
        for interval in range(intervals_per_product):
            length = int(min(history, lengths[interval % lengths.size]))
            end_local = history - (interval // lengths.size) * 21
            end_local = max(length, end_local)
            starts.append(base + end_local - length)
            ends.append(base + end_local)
            product_ids.append(product_index)
    return (
        flat,
        np.ascontiguousarray(starts, dtype=np.int64),
        np.ascontiguousarray(ends, dtype=np.int64),
        np.ascontiguousarray(product_ids, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--better-root",
        type=Path,
        default=Path("/Users/chenjunming/Desktop/KevinGit/BetterSaaTaa"),
    )
    parser.add_argument("--products", type=int, default=500)
    parser.add_argument("--history", type=int, default=2520)
    parser.add_argument("--intervals", type=int, default=12)
    parser.add_argument("--cpu-budget", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for path in (args.better_root, args.better_root / "backend"):
        sys.path.insert(0, str(path))

    import numba
    from cal_indicators.typed_dsl import compose_typed_expression
    from cal_indicators.typed_numba_plan import batch_parameter_vector, compile_numba_batch_plan

    nav, starts, ends, product_ids = data(args.products, args.history, args.intervals)
    values = np.ascontiguousarray(nav.reshape(1, -1))
    elapsed = np.ascontiguousarray(ends - starts - 1, dtype=np.float64)

    better_plans = tuple(compose_typed_expression(formula) for formula in FORMULAS)
    definitions = ({},) * len(better_plans)
    rss_before_compile = rss_bytes()
    compile_started = time.perf_counter()
    better = compile_numba_batch_plan(better_plans, definitions, ("adjusted_nav",))
    compile_ms = (time.perf_counter() - compile_started) * 1000
    params = batch_parameter_vector(definitions)
    better_output = np.empty((starts.size, len(FORMULAS)), dtype=np.float64)
    statuses = np.empty_like(better_output, dtype=np.int16)

    numba.set_num_threads(args.cpu_budget)
    better.compute(values, starts, ends, elapsed, better_output, statuses, params, parallel=True)
    if np.any(statuses != 0):
        raise RuntimeError(f"BetterSaaTaa statuses: {np.unique(statuses)}")
    rss_after_warm = rss_bytes()
    better_stats = measure(
        lambda: better.compute(
            values, starts, ends, elapsed, better_output, statuses, params, parallel=True
        ),
        args.repeats,
    )
    reference = better_output.copy()

    graph = GraphCompiler({"adjusted_nav": "series"}).compile(FORMULAS)
    inputs = {"adjusted_nav": nav}
    shared = SharedInputBundle.from_inputs(inputs)

    single_config = PlannerConfig(
        thread_work_units=1e99,
        process_work_units=1e100,
    )
    thread_config = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1e100,
        min_rows_per_worker=8,
    )
    process_config = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1.0,
        shared_memory_threshold_bytes=1,
        min_rows_per_worker=8,
        max_processes=args.cpu_budget,
    )
    auto_config = PlannerConfig()

    paired_plan = AdaptiveScheduler(cpu_budget=args.cpu_budget, config=auto_config)
    try:
        paired_execution_plan = paired_plan.plan(
            graph,
            inputs,
            starts,
            ends,
            product_ids=product_ids,
        )
        paired_plan.execute(
            graph,
            inputs,
            starts,
            ends,
            plan=paired_execution_plan,
            product_ids=product_ids,
        )

        def run_njit():
            better.compute(
                values,
                starts,
                ends,
                elapsed,
                better_output,
                statuses,
                params,
                parallel=True,
            )

        def run_native():
            paired_plan.execute(
                graph,
                inputs,
                starts,
                ends,
                plan=paired_execution_plan,
                product_ids=product_ids,
            )

        paired = paired_measure(run_njit, run_native, args.repeats)
    finally:
        paired_plan.close()

    results = {}
    try:
        for label, config in (
            ("native_single", single_config),
            ("native_thread", thread_config),
            ("native_process", process_config),
            ("native_auto", auto_config),
        ):
            with AdaptiveScheduler(cpu_budget=args.cpu_budget, config=config) as scheduler:
                input_object = shared if label in {"native_process", "native_auto"} else inputs
                plan = scheduler.plan(
                    graph,
                    input_object,
                    starts,
                    ends,
                    product_ids=product_ids,
                )
                scheduler.execute(
                    graph,
                    input_object,
                    starts,
                    ends,
                    plan=plan,
                    product_ids=product_ids,
                    timeout=60,
                )
                holder = {}

                def run(
                    holder=holder,
                    scheduler=scheduler,
                    input_object=input_object,
                    plan=plan,
                ):
                    holder["result"] = scheduler.execute(
                        graph,
                        input_object,
                        starts,
                        ends,
                        plan=plan,
                        product_ids=product_ids,
                        timeout=60,
                    )

                stats = measure(run, args.repeats)
                result = holder["result"]
                np.testing.assert_allclose(
                    result.values,
                    reference,
                    rtol=5e-10,
                    atol=5e-12,
                    equal_nan=True,
                )
                difference = np.abs(result.values - reference)
                results[label] = {
                    "plan": plan.metadata(),
                    "timing": stats,
                    "execution_audit": dict(result.audit),
                    "max_absolute_error": float(np.nanmax(difference)),
                }
    finally:
        shared.release()

    better_meta = better.metadata()
    payload = {
        "workload": {
            "products": args.products,
            "history_per_product": args.history,
            "intervals_per_product": args.intervals,
            "interval_rows": int(starts.size),
            "metrics": len(FORMULAS),
            "result_cells": int(starts.size * len(FORMULAS)),
            "input_bytes": int(nav.nbytes),
            "product_major": True,
        },
        "graph": graph.metadata(),
        "better_saataa": {
            "compile_ms": compile_ms,
            "timing": better_stats,
            "rss_before_compile_bytes": rss_before_compile,
            "rss_after_warm_bytes": rss_after_warm,
            "shared_graph": {
                key: better_meta.get(key)
                for key in (
                    "input_node_count",
                    "shared_node_count",
                    "eliminated_node_count",
                    "metric_count",
                )
            },
        },
        "calmetrics_engine": results,
        "ratios_vs_njit": {
            label: value["timing"]["median_ms"] / better_stats["median_ms"]
            for label, value in results.items()
        },
        "paired_native_auto_vs_njit": paired,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "machine": platform.machine(),
            "cpu_budget": args.cpu_budget,
        },
        "notes": [
            "BetterSaaTaa and CalMetricsEngine compile the same mathematical formulas.",
            "CalMetricsEngine auto process lane reuses a pre-created SharedInputBundle so the one-time input copy is outside repeated timings.",
            "Starts/ends and process output shared segments are created per execution and remain inside measured scheduler latency.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
