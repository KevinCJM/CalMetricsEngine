"""Prepared single-product/single-interval/five-metric C++ vs NJIT benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

RETURNS = "divide(difference(adjusted_nav,1),lag(adjusted_nav,1))"
FORMULAS = (
    f"mean({RETURNS})",
    f"std({RETURNS},1)",
    f"median({RETURNS})",
    "-min_value(drawdown_series(adjusted_nav))",
    "linear_r_squared(adjusted_nav)",
)


def _paired(left, right, rounds: int) -> dict[str, float | int]:
    left_samples = []
    right_samples = []
    for index in range(rounds):
        order = ((left, left_samples), (right, right_samples))
        if index % 2:
            order = tuple(reversed(order))
        for function, samples in order:
            started = time.perf_counter_ns()
            function()
            samples.append(time.perf_counter_ns() - started)
    left_us = statistics.median(left_samples) / 1e3
    right_us = statistics.median(right_samples) / 1e3
    return {
        "njit_median_us": left_us,
        "native_median_us": right_us,
        "native_over_njit": right_us / left_us,
        "rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--better-root",
        type=Path,
        default=Path("/Users/chenjunming/Desktop/KevinGit/BetterSaaTaa"),
    )
    parser.add_argument("--histories", nargs="+", type=int, default=[63, 252, 504, 2520])
    parser.add_argument("--cpu-budget", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=401)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for path in (args.better_root, args.better_root / "backend"):
        sys.path.insert(0, str(path))

    import numba
    from cal_indicators.typed_dsl import compose_typed_expression
    from cal_indicators.typed_numba_plan import (
        batch_parameter_vector,
        compile_numba_batch_plan,
    )

    from calmetrics_engine import AdaptiveScheduler, GraphCompiler

    plans = tuple(compose_typed_expression(formula) for formula in FORMULAS)
    definitions = ({},) * len(FORMULAS)
    compile_started = time.perf_counter_ns()
    batch = compile_numba_batch_plan(plans, definitions, ("adjusted_nav",))
    compile_ms = (time.perf_counter_ns() - compile_started) / 1e6
    params = batch_parameter_vector(definitions)
    graph = GraphCompiler({"adjusted_nav": "series"}).compile(FORMULAS)
    numba.set_num_threads(1)

    records = []
    with AdaptiveScheduler(cpu_budget=args.cpu_budget) as scheduler:
        for history in args.histories:
            day = np.arange(history - 1, dtype=np.float64)
            returns = (
                0.0002 + 0.006 * np.sin((day + 1.0) * 0.071) + 0.003 * np.cos((day + 1.0) * 0.037)
            )
            nav = np.empty(history, dtype=np.float64)
            nav[0] = 1.0
            nav[1:] = np.cumprod(1.0 + returns)
            starts = np.asarray([0], dtype=np.int64)
            ends = np.asarray([history], dtype=np.int64)
            elapsed = np.asarray([history - 1.0], dtype=np.float64)
            values = nav.reshape(1, -1)
            output = np.empty((1, len(FORMULAS)), dtype=np.float64)
            statuses = np.empty_like(output, dtype=np.int16)

            plan = scheduler.plan(graph, {"adjusted_nav": nav}, starts, ends)
            prepared = scheduler.prepare_execution(
                graph,
                {"adjusted_nav": nav},
                starts,
                ends,
                plan=plan,
            )
            batch.compute(
                values,
                starts,
                ends,
                elapsed,
                output,
                statuses,
                params,
                parallel=False,
            )
            reference = output.copy()
            native = prepared.run().copy()
            np.testing.assert_allclose(
                native,
                reference,
                rtol=5e-12,
                atol=5e-14,
                equal_nan=True,
            )

            def run_njit(
                values=values,
                starts=starts,
                ends=ends,
                elapsed=elapsed,
                output=output,
                statuses=statuses,
            ):
                batch.compute(
                    values,
                    starts,
                    ends,
                    elapsed,
                    output,
                    statuses,
                    params,
                    parallel=False,
                )

            def run_native(prepared=prepared):
                prepared.run()

            def run_scheduler(
                scheduler=scheduler,
                nav=nav,
                starts=starts,
                ends=ends,
                plan=plan,
            ):
                scheduler.execute(
                    graph,
                    {"adjusted_nav": nav},
                    starts,
                    ends,
                    plan=plan,
                )

            for _ in range(40):
                run_njit()
                run_native()
                run_scheduler()
            measured = _paired(run_njit, run_native, args.rounds)
            scheduler_measured = _paired(run_njit, run_scheduler, args.rounds)
            records.append(
                {
                    "history": history,
                    "products": 1,
                    "intervals": 1,
                    "metrics": len(FORMULAS),
                    "lane": plan.lane,
                    "max_absolute_error": float(np.max(np.abs(native - reference))),
                    **measured,
                    "scheduler_median_us": scheduler_measured["native_median_us"],
                    "scheduler_over_njit": scheduler_measured["native_over_njit"],
                }
            )

    payload = {
        "metrics": list(FORMULAS),
        "njit_compile_ms": compile_ms,
        "graph": graph.metadata(),
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
