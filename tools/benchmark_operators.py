"""Reproducible warm-call operator benchmarks; no pass/fail speed threshold.

Run from an installed wheel, with BLAS budgets set before Python starts:
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tools/benchmark_operators.py --output .build-bench/operators.json

Input preparation, output allocation and workspace growth are outside timed calls.
Python/native boundary validation IS included. No DAG or scheduler exists in this
phase, so these results do not predict complete platform or multi-process speed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from calmetrics_engine import operators as op


def peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def measure(function, repeats):
    for _ in range(4):
        function()
    elapsed, cpu = [], []
    for _ in range(repeats):
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        function()
        elapsed.append(time.perf_counter_ns() - start)
        cpu.append(time.process_time_ns() - cpu_start)
    wall_ns = statistics.median(elapsed)
    return {
        "median_ms": wall_ns / 1e6,
        "minimum_ms": min(elapsed) / 1e6,
        "median_process_cpu_ms": statistics.median(cpu) / 1e6,
        "process_cpu_percent_of_one_core": sum(cpu) / sum(elapsed) * 100,
        "process_peak_rss_bytes": peak_rss_bytes(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 4096, 1_000_000])
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 3:
        parser.error("sizes must be positive and repeats >= 3")
    records = []
    for count in args.sizes:
        for layout in ("contiguous", "strided"):
            step = 1 if layout == "contiguous" else 2
            x = np.linspace(1.0, 2.0, count * step)[::step]
            y = np.linspace(2.0, 3.0, count * step)[::step]
            x.flags.writeable = y.flags.writeable = False
            for name in ("add", "multiply", "finite_mask"):
                handle = op.get(name)
                inputs = (x,) if name == "finite_mask" else (x, y)
                out = np.empty(count, dtype=np.uint8 if name == "finite_mask" else np.float64)
                for policy in ("scalar", "auto"):
                    stats = measure(partial(handle, *inputs, out=out, simd=policy), args.repeats)
                    _, audit = handle(*inputs, out=out, simd=policy, audit=True)
                    row = {
                        "operator": name,
                        "count": count,
                        "layout": layout,
                        "requested_simd": policy,
                        "elements_per_second": count / (stats["median_ms"] / 1000),
                        **stats,
                        "audit": audit,
                    }
                    # Addresses are useful for tests, not portable benchmark artifacts.
                    row["audit"].pop("input_addresses", None)
                    records.append(row)
                numpy_call = {
                    "add": partial(np.add, x, y, out=out),
                    "multiply": partial(np.multiply, x, y, out=out),
                    "finite_mask": partial(np.isfinite, x, out=out),
                }[name]
                stats = measure(numpy_call, args.repeats)
                records.append(
                    {
                        "operator": f"numpy_{name}_reference",
                        "count": count,
                        "layout": layout,
                        "requested_simd": "reference",
                        "elements_per_second": count / (stats["median_ms"] / 1000),
                        **stats,
                    }
                )
    rng = np.random.default_rng(27001)
    for count in (32, 128, 256):
        x, y = rng.normal(size=(count, count)), rng.normal(size=(count, count))
        out = np.empty_like(x)
        expected = x @ y
        for policy in ("scalar", "auto"):
            stats = measure(partial(op.matmul, x, y, out=out, simd=policy), args.repeats)
            np.testing.assert_allclose(out, expected, rtol=2e-12, atol=2e-12)
            _, audit = op.matmul(x, y, out=out, simd=policy, audit=True)
            audit.pop("input_addresses", None)
            records.append(
                {
                    "operator": "matmul",
                    "shape": list(x.shape),
                    "requested_simd": policy,
                    **stats,
                    "audit": audit,
                }
            )
        stats = measure(partial(np.matmul, x, y, out=out), args.repeats)
        records.append({"operator": "numpy_matmul_reference", "shape": list(x.shape), **stats})
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "available_simd": op.available_simd(),
        "registry": op.REGISTRY_VERSION,
        "repeats": args.repeats,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "measurement": "warm calls; input/out allocated before timing; boundary overhead included; process_peak_rss is cumulative high-water mark, not per-call allocation",
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for record in records:
        size = record.get("count", record.get("shape"))
        print(
            record["operator"],
            size,
            record.get("layout", "matrix"),
            record.get("requested_simd", "reference"),
            f"{record['median_ms']:.6f} ms",
        )


if __name__ == "__main__":
    main()
