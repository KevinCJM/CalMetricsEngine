"""Comprehensive C++-first vs BetterSaaTaa NJIT benchmark matrix.

Time mode runs many product/interval/metric/history combinations in one process
while caching one warmed NJIT plan per metric set. Memory-child mode runs exactly
one backend/scenario in a fresh process so RSS high-water is not polluted by
previous scenarios.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

RETURNS = "divide(difference(adjusted_nav,1),lag(adjusted_nav,1))"
FORMULA_SETS = {
    1: (f"mean({RETURNS})",),
    3: (
        f"mean({RETURNS})",
        f"std({RETURNS},1)",
        "-min_value(drawdown_series(adjusted_nav))",
    ),
    5: (
        f"mean({RETURNS})",
        f"std({RETURNS},1)",
        f"median({RETURNS})",
        "-min_value(drawdown_series(adjusted_nav))",
        "linear_r_squared(adjusted_nav)",
    ),
    8: (
        f"total_return({RETURNS})",
        f"annualized_return({RETURNS},252)",
        f"mean({RETURNS})",
        f"std({RETURNS},1)",
        f"median({RETURNS})",
        f"quantile({RETURNS},0.05)",
        "-min_value(drawdown_series(adjusted_nav))",
        "linear_r_squared(adjusted_nav)",
    ),
    16: (
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
    ),
}


@dataclass(frozen=True)
class Scenario:
    name: str
    products: int
    intervals: int
    metrics: int
    history: int
    rounds: int


SCENARIOS = (
    Scenario("micro_1p_1i_1m_63", 1, 1, 1, 63, 401),
    Scenario("micro_1p_1i_1m_2520", 1, 1, 1, 2520, 201),
    Scenario("micro_1p_1i_5m_63", 1, 1, 5, 63, 401),
    Scenario("small_2p_2i_3m", 2, 2, 3, 252, 201),
    Scenario("small_5p_2i_5m", 5, 2, 5, 504, 101),
    Scenario("product_heavy_100p_1i_3m", 100, 1, 3, 504, 51),
    Scenario("interval_heavy_1p_16i_5m", 1, 16, 5, 2520, 51),
    Scenario("metric_heavy_10p_2i_16m", 10, 2, 16, 756, 51),
    Scenario("dag_branch_1p_1i_16m", 1, 1, 16, 200_000, 7),
    Scenario("mixed_10p_12i_3m", 10, 12, 3, 1512, 31),
    Scenario("mixed_50p_4i_8m", 50, 4, 8, 1008, 21),
    Scenario("mixed_100p_8i_8m", 100, 8, 8, 1512, 11),
    Scenario("many_products_500p_1i_5m", 500, 1, 5, 2520, 9),
    Scenario("large_500p_12i_16m", 500, 12, 16, 2520, 7),
    Scenario("large_1000p_12i_16m", 1000, 12, 16, 2520, 5),
)

MEMORY_SCENARIOS = (
    "micro_1p_1i_1m_63",
    "micro_1p_1i_5m_63",
    "small_5p_2i_5m",
    "product_heavy_100p_1i_3m",
    "interval_heavy_1p_16i_5m",
    "mixed_100p_8i_8m",
    "large_1000p_12i_16m",
)


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def scenario_by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


def make_data(scenario: Scenario):
    products = scenario.products
    history = scenario.history
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

    # One interval means the full available history. Multi-interval scenarios
    # include that full-history interval plus short/medium/long windows.
    lengths = np.asarray(
        [history, 63, 126, 252, 504, 756, 1008, 1260, 1512],
        dtype=np.int64,
    )
    starts: list[int] = []
    ends: list[int] = []
    product_ids: list[int] = []
    for product_index in range(products):
        base = product_index * history
        for interval in range(scenario.intervals):
            length = int(min(history, lengths[interval % lengths.size]))
            end_local = history - (interval // lengths.size) * 21
            end_local = max(length, end_local)
            starts.append(base + end_local - length)
            ends.append(base + end_local)
            product_ids.append(product_index)
    return (
        flat,
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(product_ids, dtype=np.int64),
    )


def paired(left, right, rounds: int) -> dict[str, float | int]:
    left_samples: list[int] = []
    right_samples: list[int] = []
    for index in range(rounds):
        order = ((left, left_samples), (right, right_samples))
        if index & 1:
            order = tuple(reversed(order))
        for function, samples in order:
            start = time.perf_counter_ns()
            function()
            samples.append(time.perf_counter_ns() - start)
    left_ms = statistics.median(left_samples) / 1e6
    right_ms = statistics.median(right_samples) / 1e6
    return {
        "njit_median_ms": left_ms,
        "native_median_ms": right_ms,
        "native_over_njit": right_ms / left_ms,
        "native_speedup": left_ms / right_ms,
        "rounds": rounds,
    }


def prepare_njit(metric_count: int):
    from cal_indicators.typed_dsl import compose_typed_expression
    from cal_indicators.typed_numba_plan import batch_parameter_vector, compile_numba_batch_plan

    formulas = FORMULA_SETS[metric_count]
    plans = tuple(compose_typed_expression(formula) for formula in formulas)
    definitions = ({},) * len(plans)
    started = time.perf_counter_ns()
    plan = compile_numba_batch_plan(plans, definitions, ("adjusted_nav",))
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    params = batch_parameter_vector(definitions)
    return plan, params, compile_ms


def run_time_matrix(cpu_budget: int) -> list[dict]:
    import numba

    njit_cache = {}
    native_graphs = {}
    records = []

    for scenario in SCENARIOS:
        if scenario.metrics not in njit_cache:
            njit_cache[scenario.metrics] = prepare_njit(scenario.metrics)
            native_graphs[scenario.metrics] = GraphCompiler({"adjusted_nav": "series"}).compile(
                FORMULA_SETS[scenario.metrics]
            )

        better, params, compile_ms = njit_cache[scenario.metrics]
        graph = native_graphs[scenario.metrics]
        nav, starts, ends, product_ids = make_data(scenario)
        values = nav.reshape(1, -1)
        elapsed = np.ascontiguousarray(ends - starts - 1, dtype=np.float64)
        njit_output = np.empty((starts.size, scenario.metrics), dtype=np.float64)
        statuses = np.empty_like(njit_output, dtype=np.int16)

        numba.set_num_threads(cpu_budget)
        parallel = starts.size > 1
        better.compute(
            values,
            starts,
            ends,
            elapsed,
            njit_output,
            statuses,
            params,
            parallel=parallel,
        )
        if np.any(statuses != 0):
            raise RuntimeError(f"{scenario.name}: NJIT status failure")
        reference = njit_output.copy()

        with AdaptiveScheduler(cpu_budget=cpu_budget) as scheduler:
            plan = scheduler.plan(
                graph,
                {"adjusted_nav": nav},
                starts,
                ends,
                product_ids=product_ids,
            )
            native_result = scheduler.execute(
                graph,
                {"adjusted_nav": nav},
                starts,
                ends,
                plan=plan,
                product_ids=product_ids,
            )
            np.testing.assert_allclose(
                native_result.values,
                reference,
                rtol=5e-10,
                atol=5e-12,
                equal_nan=True,
            )

            prepared = None
            if plan.lane == "single":
                prepared = scheduler.prepare_execution(
                    graph,
                    {"adjusted_nav": nav},
                    starts,
                    ends,
                    plan=plan,
                    product_ids=product_ids,
                )
                np.testing.assert_allclose(
                    prepared.run(),
                    reference,
                    rtol=5e-10,
                    atol=5e-12,
                    equal_nan=True,
                )

            def run_njit(
                better=better,
                values=values,
                starts=starts,
                ends=ends,
                elapsed=elapsed,
                njit_output=njit_output,
                statuses=statuses,
                params=params,
                parallel=parallel,
            ):
                better.compute(
                    values,
                    starts,
                    ends,
                    elapsed,
                    njit_output,
                    statuses,
                    params,
                    parallel=parallel,
                )

            holder = {}

            def run_native(
                holder=holder,
                scheduler=scheduler,
                graph=graph,
                nav=nav,
                starts=starts,
                ends=ends,
                plan=plan,
                product_ids=product_ids,
            ):
                holder["result"] = scheduler.execute(
                    graph,
                    {"adjusted_nav": nav},
                    starts,
                    ends,
                    plan=plan,
                    product_ids=product_ids,
                )

            for _ in range(8):
                run_njit()
                run_native()
            timing = paired(run_njit, run_native, scenario.rounds)
            last = holder["result"]

            prepared_timing = None
            if prepared is not None:
                prepared_timing = paired(run_njit, prepared.run, scenario.rounds)

            records.append(
                {
                    "scenario": asdict(scenario),
                    "rows": int(starts.size),
                    "result_cells": int(starts.size * scenario.metrics),
                    "input_bytes": int(nav.nbytes),
                    "interval_observations": int(np.sum(ends - starts, dtype=np.int64)),
                    "njit_compile_ms_for_metric_set": compile_ms,
                    "native_plan": plan.metadata(),
                    "timing": timing,
                    "prepared_timing": prepared_timing,
                    "max_absolute_error": float(np.nanmax(np.abs(last.values - reference))),
                    "native_audit": {
                        key: dict(last.audit)[key]
                        for key in (
                            "lane",
                            "shared_memory_bytes",
                            "boundary_copy_bytes",
                            "output_copy_bytes",
                            "native_threads",
                            "native_processes",
                        )
                    },
                }
            )
    return records


def memory_child(backend: str, scenario: Scenario, cpu_budget: int, repeats: int) -> dict:
    nav, starts, ends, product_ids = make_data(scenario)
    formulas = FORMULA_SETS[scenario.metrics]
    baseline_rss = rss_bytes()
    input_bytes = int(nav.nbytes + starts.nbytes + ends.nbytes + product_ids.nbytes)
    output_bytes = int(starts.size * scenario.metrics * 8)

    if backend == "native":
        graph = GraphCompiler({"adjusted_nav": "series"}).compile(formulas)
        compile_rss = rss_bytes()
        with AdaptiveScheduler(cpu_budget=cpu_budget) as scheduler:
            plan = scheduler.plan(
                graph,
                {"adjusted_nav": nav},
                starts,
                ends,
                product_ids=product_ids,
            )
            result = scheduler.execute(
                graph,
                {"adjusted_nav": nav},
                starts,
                ends,
                plan=plan,
                product_ids=product_ids,
            )
            warm_rss = rss_bytes()
            for _ in range(repeats):
                result = scheduler.execute(
                    graph,
                    {"adjusted_nav": nav},
                    starts,
                    ends,
                    plan=plan,
                    product_ids=product_ids,
                )
            final_rss = rss_bytes()
            audit = dict(result.audit)
        chunks = audit.get("native_chunks", [])
        native_scratch = sum(
            chunk.get("numeric_arena_bytes", 0)
            + chunk.get("mask_arena_bytes", 0)
            + chunk.get("operator_workspace_capacity_bytes", 0)
            + chunk.get("order_scratch_capacity_bytes", 0)
            for chunk in chunks
        )
        return {
            "backend": backend,
            "scenario": asdict(scenario),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "rss_baseline_bytes": baseline_rss,
            "rss_after_compile_bytes": compile_rss,
            "rss_after_warm_bytes": warm_rss,
            "rss_after_repeats_bytes": final_rss,
            "compile_rss_growth_bytes": max(0, compile_rss - baseline_rss),
            "warm_rss_growth_bytes": max(0, warm_rss - baseline_rss),
            "steady_state_highwater_growth_bytes": max(0, final_rss - warm_rss),
            "lane": plan.lane,
            "native_scratch_capacity_bytes": native_scratch,
            "shared_memory_bytes": int(audit.get("shared_memory_bytes", 0)),
            "boundary_copy_bytes": int(audit.get("boundary_copy_bytes", 0)),
            "output_copy_bytes": int(audit.get("output_copy_bytes", 0)),
        }

    if backend == "njit":
        import numba

        before_compile = rss_bytes()
        better, params, compile_ms = prepare_njit(scenario.metrics)
        compile_rss = rss_bytes()
        values = nav.reshape(1, -1)
        elapsed = np.ascontiguousarray(ends - starts - 1, dtype=np.float64)
        output = np.empty((starts.size, scenario.metrics), dtype=np.float64)
        statuses = np.empty_like(output, dtype=np.int16)
        numba.set_num_threads(cpu_budget)
        parallel = starts.size > 1
        better.compute(values, starts, ends, elapsed, output, statuses, params, parallel=parallel)
        warm_rss = rss_bytes()
        for _ in range(repeats):
            better.compute(
                values,
                starts,
                ends,
                elapsed,
                output,
                statuses,
                params,
                parallel=parallel,
            )
        final_rss = rss_bytes()
        return {
            "backend": backend,
            "scenario": asdict(scenario),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "rss_baseline_bytes": baseline_rss,
            "rss_before_compile_bytes": before_compile,
            "rss_after_compile_bytes": compile_rss,
            "rss_after_warm_bytes": warm_rss,
            "rss_after_repeats_bytes": final_rss,
            "compile_ms": compile_ms,
            "compile_rss_growth_bytes": max(0, compile_rss - before_compile),
            "warm_rss_growth_bytes": max(0, warm_rss - before_compile),
            "steady_state_highwater_growth_bytes": max(0, final_rss - warm_rss),
        }

    raise ValueError(backend)


def run_memory_matrix(
    script: Path,
    better_root: Path,
    cpu_budget: int,
    repeats: int,
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in MEMORY_SCENARIOS:
        for backend in ("njit", "native"):
            output = output_dir / f"{name}-{backend}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--mode",
                    "memory-child",
                    "--backend",
                    backend,
                    "--scenario",
                    name,
                    "--better-root",
                    str(better_root),
                    "--cpu-budget",
                    str(cpu_budget),
                    "--memory-repeats",
                    str(repeats),
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            records.append(json.loads(output.read_text()))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--better-root",
        type=Path,
        default=Path("/Users/chenjunming/Desktop/KevinGit/BetterSaaTaa"),
    )
    parser.add_argument(
        "--mode",
        choices=("all", "time", "memory", "memory-child"),
        default="all",
    )
    parser.add_argument("--backend", choices=("njit", "native"))
    parser.add_argument("--scenario")
    parser.add_argument("--cpu-budget", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--memory-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory-output-dir", type=Path)
    args = parser.parse_args()

    for path in (args.better_root, args.better_root / "backend"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    if args.mode == "memory-child":
        if not args.backend or not args.scenario:
            parser.error("memory-child requires --backend and --scenario")
        record = memory_child(
            args.backend,
            scenario_by_name(args.scenario),
            args.cpu_budget,
            args.memory_repeats,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return

    payload = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_budget": args.cpu_budget,
        },
        "time_matrix": [],
        "memory_matrix": [],
    }
    if args.mode in {"all", "time"}:
        payload["time_matrix"] = run_time_matrix(args.cpu_budget)
    if args.mode in {"all", "memory"}:
        memory_dir = args.memory_output_dir or Path(".build-benchmark-matrix-memory")
        payload["memory_matrix"] = run_memory_matrix(
            Path(__file__).resolve(),
            args.better_root,
            args.cpu_budget,
            args.memory_repeats,
            memory_dir,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
