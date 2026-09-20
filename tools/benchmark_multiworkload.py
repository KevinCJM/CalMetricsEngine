"""End-to-end multi-product × multi-interval × multi-metric benchmark.

The NJIT side uses BetterSaaTaa's real CompiledNumbaBatchPlan.  The C++ side
uses a standalone executable linked directly to CalMetricsEngine's canonical
Operator Registry, so Python-per-operator call overhead is not part of C++ time.

This is a benchmark harness, not a production NativeExecutionPlan/Scheduler.
The C++ threaded mode statically partitions interval rows and creates threads per
benchmark iteration; it must not be described as the final persistent thread pool.
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
from pathlib import Path

import numpy as np

FORMULAS = (
    "product(returns + 1) - 1",
    "(product(returns + 1) - 1 + 1) ** (periods_per_year / observation_count) - 1",
    "mean(returns)",
    "std(returns,1)",
    "median(returns)",
    "min_value(returns)",
    "max_value(returns)",
    "quantile(returns,0.05)",
    "mean_absolute_deviation(returns)",
    "root_mean_square(returns)",
    "-min_value(drawdown_series(adjusted_nav))",
    "count_true(new_high_mask(adjusted_nav)) / length(adjusted_nav)",
    "linear_slope(adjusted_nav)",
    "linear_r_squared(adjusted_nav)",
    "mean_where(returns,greater_than(returns,0))",
    "std_where(returns,less_than(returns,0))",
)

METRIC_NAMES = (
    "total_return",
    "annualized_return",
    "mean_return",
    "std_return",
    "median_return",
    "min_return",
    "max_return",
    "var_05",
    "mean_absolute_deviation",
    "root_mean_square",
    "maximum_drawdown",
    "new_high_ratio",
    "nav_slope",
    "nav_r_squared",
    "mean_positive_return",
    "std_negative_return",
)


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def timed(function, repeats: int) -> dict[str, float]:
    values = []
    cpu = []
    for _ in range(repeats):
        cpu_started = time.process_time_ns()
        started = time.perf_counter_ns()
        function()
        values.append(time.perf_counter_ns() - started)
        cpu.append(time.process_time_ns() - cpu_started)
    return {
        "median_ms": statistics.median(values) / 1e6,
        "minimum_ms": min(values) / 1e6,
        "median_cpu_ms": statistics.median(cpu) / 1e6,
    }


def prepare_data(products: int, history: int, intervals: int):
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

    base_lengths = np.asarray([63, 126, 252, 504, 756, 1008, 1260, 1512], dtype=np.int64)
    starts, ends = [], []
    for product_index in range(products):
        base = product_index * history
        for window_index in range(intervals):
            length = int(min(history, base_lengths[window_index % base_lengths.size]))
            end_local = history - (window_index // base_lengths.size) * 21
            if end_local < length:
                end_local = length
            starts.append(base + end_local - length)
            ends.append(base + end_local)
    return (
        flat,
        np.ascontiguousarray(starts, dtype=np.int64),
        np.ascontiguousarray(ends, dtype=np.int64),
    )


def install_better_root(root: Path) -> None:
    for path in (root, root / "backend"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def build_cpp(root: Path, build_dir: Path) -> Path:
    executable = build_dir / "calmetrics_engine_multiworkload_benchmark"
    if sys.platform == "win32":
        executable = build_dir / "Release" / "calmetrics_engine_multiworkload_benchmark.exe"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCALMETRICS_ENGINE_BUILD_PYTHON=OFF",
            "-DCALMETRICS_ENGINE_BUILD_TESTS=OFF",
            "-DCALMETRICS_ENGINE_BUILD_BENCHMARKS=ON",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", "4"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if not executable.exists():
        raise FileNotFoundError(executable)
    return executable


def run_cpp(
    executable: Path,
    directory: Path,
    nav: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    history: int,
    repeats: int,
    threads: int,
):
    nav_path = directory / "nav.f64"
    starts_path = directory / "starts.i64"
    ends_path = directory / "ends.i64"
    output_path = directory / f"cpp-{threads}.f64"
    # Always overwrite: benchmark dimensions are caller-controlled and stale
    # binary inputs from a previous run must never be reused across shapes.
    nav.tofile(nav_path)
    starts.tofile(starts_path)
    ends.tofile(ends_path)

    completed = subprocess.run(
        [
            str(executable),
            "--nav",
            str(nav_path),
            "--starts",
            str(starts_path),
            "--ends",
            str(ends_path),
            "--output",
            str(output_path),
            "--rows",
            str(starts.size),
            "--history",
            str(history),
            "--repeats",
            str(repeats),
            "--threads",
            str(threads),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(completed.stdout)
    output = np.fromfile(output_path, dtype=np.float64).reshape(starts.size, len(FORMULAS))
    return metadata, output


def parity(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), 1e-15)
    np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=5e-12, equal_nan=True)
    return {
        "max_absolute_error": float(np.nanmax(difference)),
        "max_relative_error": float(np.nanmax(difference / scale)),
    }


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
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--thread-grid", nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    thread_grid = sorted(set(args.thread_grid or [args.threads]))
    if (
        min(args.products, args.history, args.intervals, args.threads, args.repeats, *thread_grid)
        < 1
    ):
        parser.error("all dimensions/repeats must be positive")
    if args.history < 64:
        parser.error("history must be >= 64")
    if not args.better_root.is_dir():
        parser.error(f"BetterSaaTaa root does not exist: {args.better_root}")

    root = Path(__file__).resolve().parents[1]
    work = root / ".build-multiworkload-data"
    work.mkdir(exist_ok=True)
    executable = build_cpp(root, root / ".build-multiworkload")

    install_better_root(args.better_root)
    import numba
    from cal_indicators.typed_dsl import compose_typed_expression
    from cal_indicators.typed_numba_plan import batch_parameter_vector, compile_numba_batch_plan
    from custom_indicators.variable_registry import variable_types

    nav, starts, ends = prepare_data(args.products, args.history, args.intervals)
    values = np.ascontiguousarray(nav.reshape(1, -1))
    elapsed = np.ascontiguousarray(ends - starts - 1, dtype=np.float64)

    types = variable_types("single_product")
    plans = tuple(compose_typed_expression(formula, variable_types=types) for formula in FORMULAS)
    definitions = ({},) * len(plans)

    rss_before_compile = peak_rss_bytes()
    compile_started = time.perf_counter()
    batch = compile_numba_batch_plan(plans, definitions, ("adjusted_nav",))
    compile_ms = (time.perf_counter() - compile_started) * 1000
    rss_after_compile = peak_rss_bytes()
    params = batch_parameter_vector(definitions)
    njit_output = np.empty((starts.size, len(plans)), dtype=np.float64)
    statuses = np.empty((starts.size, len(plans)), dtype=np.int16)

    numba.set_num_threads(max(thread_grid))
    batch.compute(values, starts, ends, elapsed, njit_output, statuses, params, parallel=False)
    if np.any(statuses != 0):
        raise RuntimeError(f"NJIT serial statuses contain failures: {np.unique(statuses)}")
    rss_after_compile_warm = peak_rss_bytes()

    serial = timed(
        lambda: batch.compute(
            values, starts, ends, elapsed, njit_output, statuses, params, parallel=False
        ),
        args.repeats,
    )
    njit_serial_output = njit_output.copy()

    njit_parallel = {}
    for thread_count in thread_grid:
        numba.set_num_threads(thread_count)
        batch.compute(values, starts, ends, elapsed, njit_output, statuses, params, parallel=True)
        if np.any(statuses != 0):
            raise RuntimeError(f"NJIT parallel statuses contain failures: {np.unique(statuses)}")
        njit_parallel[str(thread_count)] = timed(
            lambda: batch.compute(
                values, starts, ends, elapsed, njit_output, statuses, params, parallel=True
            ),
            args.repeats,
        )
        np.testing.assert_allclose(
            njit_output, njit_serial_output, rtol=2e-12, atol=2e-14, equal_nan=True
        )
    njit_peak_rss = peak_rss_bytes()

    cpp_serial, cpp_serial_output = run_cpp(
        executable, work, nav, starts, ends, args.history, args.repeats, 1
    )
    cpp_parallel = {}
    cpp_parallel_outputs = {}
    for thread_count in thread_grid:
        stats, current = run_cpp(
            executable, work, nav, starts, ends, args.history, args.repeats, thread_count
        )
        cpp_parallel[str(thread_count)] = stats
        cpp_parallel_outputs[str(thread_count)] = current

    serial_parity = parity(cpp_serial_output, njit_serial_output)
    parallel_parity = {
        str(thread_count): parity(cpp_parallel_outputs[str(thread_count)], njit_serial_output)
        for thread_count in thread_grid
    }

    sharing = batch.metadata()
    rows = starts.size
    cells = rows * len(FORMULAS)
    interval_observations = int(np.sum(ends - starts))
    payload = {
        "workload": {
            "products": args.products,
            "history_per_product": args.history,
            "intervals_per_product": args.intervals,
            "interval_rows": int(rows),
            "metrics": len(FORMULAS),
            "metric_names": METRIC_NAMES,
            "result_cells": int(cells),
            "interval_observations": interval_observations,
            "input_nav_bytes": int(nav.nbytes),
            "output_bytes": int(njit_serial_output.nbytes),
            "status_bytes": int(statuses.nbytes),
        },
        "better_saataa": {
            "backend": "CompiledNumbaBatchPlan",
            "compile_ms": compile_ms,
            "shared_graph": {
                key: sharing.get(key)
                for key in (
                    "metric_count",
                    "input_node_count",
                    "shared_node_count",
                    "eliminated_node_count",
                )
            },
            "thread_grid": thread_grid,
            "serial": serial,
            "parallel_by_threads": njit_parallel,
            "peak_rss_before_compile_bytes": rss_before_compile,
            "peak_rss_after_compile_bytes": rss_after_compile,
            "peak_rss_after_compile_warm_bytes": rss_after_compile_warm,
            "peak_rss_after_benchmark_bytes": njit_peak_rss,
        },
        "calmetrics_engine": {
            "backend": "C++ canonical Operator Registry benchmark harness",
            "serial": cpp_serial,
            "row_parallel_by_threads": cpp_parallel,
            "row_parallel_note": (
                "benchmark-only static interval-row partition; threads are created per iteration; "
                "not the future persistent Scheduler/ThreadPool"
            ),
        },
        "parity": {
            "cpp_serial_vs_njit": serial_parity,
            "cpp_parallel_vs_njit": parallel_parity,
        },
        "ratios": {
            "cpp_serial_vs_njit_serial": cpp_serial["median_ms"] / serial["median_ms"],
            "by_threads": {
                str(thread_count): {
                    "cpp_vs_njit_parallel": cpp_parallel[str(thread_count)]["median_ms"]
                    / njit_parallel[str(thread_count)]["median_ms"],
                    "cpp_vs_njit_serial": cpp_parallel[str(thread_count)]["median_ms"]
                    / serial["median_ms"],
                }
                for thread_count in thread_grid
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "measurement": (
            "warm compute only; synthetic product-major NAV generated once; output preallocated; "
            "BetterSaaTaa uses real compiled shared-DAG batch plan; C++ executable loads input "
            "before its internal timer and directly calls the production canonical registry"
        ),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
