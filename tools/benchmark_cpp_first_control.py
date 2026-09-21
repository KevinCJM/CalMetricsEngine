"""Measure compiler/planner/API overhead using an installed engine build.

Run the same file with baseline and candidate interpreters. No timing compares
planning on one side against a prepared calculation on the other side.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

RETURNS = "divide(difference(nav,1),lag(nav,1))"
FORMULAS = (
    f"mean({RETURNS})",
    f"std({RETURNS},1)",
    f"median({RETURNS})",
    "-min_value(drawdown_series(nav))",
    "linear_r_squared(nav)",
)


def measure(function, rounds):
    for _ in range(10):
        function()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1000.0)
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p10_us": ordered[len(ordered) // 10],
        "p90_us": ordered[len(ordered) * 9 // 10],
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=301)
    parser.add_argument("--history", type=int, default=252)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 11 or args.history < 3:
        parser.error("rounds >= 11 and history >= 3 are required")

    compiler = GraphCompiler({"nav": "series"})
    graph = compiler.compile(FORMULAS)
    t = np.arange(args.history - 1, dtype=np.float64)
    returns = 0.0002 + 0.006 * np.sin((t + 1) * 0.071) + 0.003 * np.cos((t + 1) * 0.037)
    nav = np.empty(args.history)
    nav[0] = 1
    nav[1:] = np.cumprod(1 + returns)
    inputs = {"nav": nav}
    starts, ends = np.array([0], np.int64), np.array([nav.size], np.int64)

    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        prepared = scheduler.prepare_execution(graph, inputs, starts, ends, plan=plan)
        records = {
            "compile": measure(lambda: compiler.compile(FORMULAS), args.rounds),
            "plan": measure(lambda: scheduler.plan(graph, inputs, starts, ends), args.rounds),
            "bind_prepared": measure(
                lambda: scheduler.prepare_execution(graph, inputs, starts, ends, plan=plan),
                args.rounds,
            ),
            "execute_preplanned": measure(
                lambda: scheduler.execute(graph, inputs, starts, ends, plan=plan), args.rounds
            ),
            "plan_and_execute": measure(
                lambda: scheduler.execute(graph, inputs, starts, ends), args.rounds
            ),
            "prepared_run": measure(prepared.run, args.rounds),
            "prepared_run_audit": measure(prepared.run_audit, args.rounds),
            "prepared_run_snapshot": measure(prepared.run_snapshot, args.rounds),
        }
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.machine(),
        "compiler_class_module": GraphCompiler.__module__,
        "products": 1,
        "intervals": 1,
        "history": args.history,
        "metrics": len(FORMULAS),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
