"""Reproducible typed-result end-to-end measurement; not a replacement for legacy gates."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

import numpy as np

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig, build_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=["single", "thread", "process_inline", "process_shared"], default="single")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--assets", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.window - 1, args.assets, args.repeats) < 1:
        parser.error("positive rows/assets/repeats and window >= 2 required")
    x = np.random.default_rng(3791).normal(size=(args.window * 10, args.assets))
    starts = np.linspace(0, len(x) - args.window, args.rows, dtype=np.int64)
    ends = starts + args.window
    compiler = GraphCompiler({"x": {"kind": "matrix"}})
    formulas = ["mean_time(x)", "covariance(x)", "transpose(x)", "sum_asset(x)>0", "argsort(sum_asset(x))"]
    graph = compiler.compile(formulas, error_policy="isolate")
    separate = [compiler.compile(f, result_format="typed", error_policy="isolate") for f in formulas]
    process = args.lane.startswith("process")
    cfg = PlannerConfig(thread_work_units=1e100 if args.lane == "single" else 1,
                        process_work_units=1 if process else 1e100,
                        shared_memory_threshold_bytes=1 if args.lane == "process_shared" else 1 << 30,
                        min_rows_per_worker=1, max_processes=2)
    samples, separate_samples = [], []
    with AdaptiveScheduler(cpu_budget=2, config=cfg) as scheduler:
        def run(g):
            r = scheduler.execute(g, {"x": x}, starts, ends)
            return r, r.outputs  # Include public view/status materialization in timing.
        result, outputs = run(graph)
        for row in [0, args.rows - 1]:
            m = x[starts[row]:ends[row]]
            expected = [m.mean(axis=0), np.cov(m, rowvar=False), m.T,
                        m.sum(axis=1) > 0, np.argsort(m.sum(axis=1), kind="stable")]
            for out, reference in zip(outputs, expected, strict=True):
                np.testing.assert_allclose(out.values[row], reference, rtol=1e-12, atol=1e-13)
                assert out.root_statuses[row] == 0
        for g in separate:
            run(g)
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic warm-cache ordering bias.
            for batched in ([True, False] if repeat % 2 == 0 else [False, True]):
                before = time.perf_counter_ns()
                if batched:
                    result, outputs = run(graph)
                else:
                    _separate_results = [run(g) for g in separate]
                elapsed = (time.perf_counter_ns() - before) / 1e6
                (samples if batched else separate_samples).append(elapsed)
    payload = {"workload": vars(args) | {"output": str(args.output)}, "engine": build_info(),
               "shared_dag_median_ms": median(samples), "separate_graphs_median_ms": median(separate_samples),
               "ratio": median(samples) / median(separate_samples),
               "output_capacity_bytes": result.plan.estimated_output_bytes,
               "estimated_total_memory_bytes": result.plan.estimated_total_memory_bytes,
               "output_copy_bytes": result.audit["output_copy_bytes"],
               "result_protocol": result.audit["result_protocol"], "lane": result.plan.lane,
               "samples_ms": samples, "separate_samples_ms": separate_samples,
               "note": "Local measurement including output views; no pre-existing typed-result baseline or speed guarantee."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{args.lane}: shared DAG {median(samples):.3f} ms, separate {median(separate_samples):.3f} ms; {payload['ratio']:.3f}")


if __name__ == "__main__":
    main()
