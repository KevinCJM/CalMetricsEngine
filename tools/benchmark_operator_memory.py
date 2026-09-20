"""Memory and steady-state allocation benchmark for representative canonical operators.

Each scenario runs in a fresh child process so RSS high-water growth is meaningful.
Inputs, outputs, and reusable Workspace capacity are prepared before measurement.
The measured phase therefore exposes hidden per-call growth rather than intentional
caller-owned arrays.
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
import tracemalloc
from pathlib import Path

import numpy as np

from calmetrics_engine import operators as op


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _time_call(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - start)
    return statistics.median(samples) / 1e6


def _scenario(name: str, repeats: int) -> dict:
    rng = np.random.default_rng(20260919)

    if name == "add_1m":
        left = rng.normal(size=1_000_000)
        right = rng.normal(size=1_000_000)
        out = np.empty_like(left)
        workspace = op.Workspace()
        args = (left, right)
        handle = op.add
        kwargs = {"out": out, "workspace": workspace, "simd": "auto"}
    elif name == "rolling_min_1m":
        values = rng.normal(size=1_000_000)
        out = np.empty_like(values)
        workspace = op.Workspace()
        args = (values, 252.0)
        handle = op.rolling_min
        req = op.get("rolling_min").requirements(*args)
        workspace.reserve(indices=req["scratch_indices"], doubles=req["scratch_doubles"])
        kwargs = {"out": out, "workspace": workspace}
    elif name == "quantile_1m":
        values = rng.normal(size=1_000_000)
        out = np.empty((), dtype=np.float64)
        workspace = op.Workspace()
        args = (values, 0.95)
        handle = op.quantile
        req = op.get("quantile").requirements(*args)
        workspace.reserve(indices=req["scratch_indices"], doubles=req["scratch_doubles"])
        kwargs = {"out": out, "workspace": workspace}
    elif name == "solve_256":
        raw = rng.normal(size=(256, 256))
        matrix = raw.T @ raw + np.eye(256) * 0.1
        rhs = rng.normal(size=256)
        out = np.empty(256, dtype=np.float64)
        workspace = op.Workspace()
        args = (matrix, rhs)
        handle = op.solve
        req = op.get("solve").requirements(*args)
        workspace.reserve(indices=req["scratch_indices"], doubles=req["scratch_doubles"])
        kwargs = {"out": out, "workspace": workspace}
    elif name == "covariance_5000x64":
        values = rng.normal(size=(5_000, 64))
        out = np.empty((64, 64), dtype=np.float64)
        workspace = op.Workspace()
        args = (values,)
        handle = op.covariance
        req = op.get("covariance").requirements(*args)
        workspace.reserve(indices=req["scratch_indices"], doubles=req["scratch_doubles"])
        kwargs = {"out": out, "workspace": workspace}
    elif name == "matmul_256":
        left = rng.normal(size=(256, 256))
        right = rng.normal(size=(256, 256))
        out = np.empty((256, 256), dtype=np.float64)
        workspace = op.Workspace()
        args = (left, right)
        handle = op.matmul
        kwargs = {"out": out, "workspace": workspace, "simd": "auto"}
    else:
        raise ValueError(name)

    for array in args:
        if isinstance(array, np.ndarray):
            array.flags.writeable = False

    # Warm once so reusable workspace, dispatch selection and Python import costs
    # are outside the measured steady-state interval.
    handle(*args, **kwargs)
    _, audit = handle(*args, **kwargs, audit=True)
    requirement = op.get(
        name.split("_", 1)[0]
        if name.startswith("add_")
        else {
            "rolling_min_1m": "rolling_min",
            "quantile_1m": "quantile",
            "solve_256": "solve",
            "covariance_5000x64": "covariance",
            "matmul_256": "matmul",
        }.get(name, name)
    ).requirements(*args)

    before_rss = rss_bytes()
    before_workspace = workspace.capacity_bytes
    tracemalloc.start()
    median_ms = _time_call(lambda: handle(*args, **kwargs), repeats)
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_rss = rss_bytes()
    after_workspace = workspace.capacity_bytes

    output = kwargs["out"]
    return {
        "scenario": name,
        "repeats": repeats,
        "median_ms": median_ms,
        "input_bytes": sum(array.nbytes for array in args if isinstance(array, np.ndarray)),
        "output_bytes": output.nbytes,
        "requirement": requirement,
        "audit": {k: v for k, v in audit.items() if k != "input_addresses"},
        "workspace_capacity_before": before_workspace,
        "workspace_capacity_after": after_workspace,
        "python_tracemalloc_peak_bytes": peak_python,
        "rss_high_water_before_bytes": before_rss,
        "rss_high_water_after_bytes": after_rss,
        "rss_high_water_growth_bytes": max(0, after_rss - before_rss),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--child")
    args = parser.parse_args()

    if args.child:
        print(json.dumps(_scenario(args.child, args.repeats)))
        return

    scenarios = [
        "add_1m",
        "rolling_min_1m",
        "quantile_1m",
        "solve_256",
        "covariance_5000x64",
        "matmul_256",
    ]
    records = []
    for scenario in scenarios:
        env = os.environ.copy()
        env.update(
            {
                "VECLIB_MAXIMUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                scenario,
                "--repeats",
                str(args.repeats),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        record = json.loads(completed.stdout)
        records.append(record)
        print(
            scenario,
            f"{record['median_ms']:.6f} ms",
            "rss_growth=",
            record["rss_high_water_growth_bytes"],
            "python_peak=",
            record["python_tracemalloc_peak_bytes"],
            "workspace=",
            record["workspace_capacity_after"],
        )

    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "available_simd": op.available_simd(),
        "registry": op.REGISTRY_VERSION,
        "measurement": (
            "fresh child per scenario; inputs/out/workspace preallocated and one warm call completed "
            "before RSS/tracemalloc measurement; RSS is process high-water growth during steady-state repeats"
        ),
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
