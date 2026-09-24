"""Paired before/after/NJIT checks for segmented pointwise execution.

Workers warm once, then run sequentially in alternating order. IPC and JIT
compilation are outside timers. Heap interposition is a separate execution.
Both C++ versions use ordinary execute and return independent result owners.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from check_execution_scaling import HeapStats
from check_foundation_performance import paired_ratio

CASES = [
    {"id": "long_small", "n": 24, "kernel": "long", "layout": "C"},
    {"id": "long_medium", "n": 32768, "kernel": "long", "layout": "C"},
    {"id": "long_large", "n": 1048576, "kernel": "long", "layout": "C"},
    {"id": "long_reverse", "n": 32768, "kernel": "long", "layout": "reverse"},
    {"id": "long_masked", "n": 32768, "kernel": "long", "layout": "nan_heavy"},
    {"id": "branched_medium", "n": 32768, "kernel": "branched", "layout": "C"},
    {"id": "affine_small", "n": 24, "kernel": "affine", "layout": "C"},
    {"id": "affine_large", "n": 1048576, "kernel": "affine", "layout": "C"},
]


def reference(case):
    from numba import njit, set_num_threads

    set_num_threads(1)

    # Frozen formulas; fastmath is deliberately disabled.
    @njit(nogil=True)
    def long(x):
        values, mask = np.empty(x.size), np.empty(x.size, np.bool_)
        for i in range(x.size):
            v = (((x[i] * 2.0 + 1.0) * 3.0 - 2.0) * 4.0) + 1.0
            values[i], mask[i] = v, v > 0.0
        return values, mask

    @njit(nogil=True)
    def branched(x):
        values, mask = np.empty(x.size), np.empty(x.size, np.bool_)
        for i in range(x.size):
            a, b = ((x[i] * 2.0 + 1.0) * 3.0) - 2.0, ((x[i] * 4.0 - 3.0) * 2.0) + 1.0
            v = (a + b) * 2.0 + 3.0
            values[i], mask[i] = v, v > 0.0
        return values, mask

    @njit(nogil=True)
    def affine(x):
        values, mask = np.empty(x.size), np.empty(x.size, np.bool_)
        for i in range(x.size):
            v = x[i] * 2.0 + 1.0
            values[i], mask[i] = v, v > 0.0
        return values, mask

    return {"long": long, "branched": branched, "affine": affine}[case["kernel"]]


def worker(case, backend, probe_path):
    x = np.linspace(-3.0, 4.0, case["n"])
    if case["layout"] == "reverse":
        x = x[::-1]
    if case["layout"] == "nan_heavy":
        x[::7], x[1::127], x[2::127] = np.nan, np.inf, -np.inf
    x.flags.writeable = False
    expression = {
        "long": "((((x*2+1)*3-2)*4)+1)",
        "branched": "((((x*2+1)*3-2)+((x*4-3)*2+1))*2+3)",
        "affine": "(x*2+1)",
    }[case["kernel"]]
    expected = eval(expression, {"__builtins__": {}}, {"x": x})
    engine = None
    build = None
    if backend == "njit":
        ref = reference(case)
        ref(x)  # Same dtype/layout compiled before tracing or timing.

        def setup():
            return lambda: ref(x)
    else:
        from calmetrics_engine import AdaptiveScheduler, GraphCompiler

        def setup():
            nonlocal engine
            graph = GraphCompiler(
                {"x": {"kind": "vector", "axes": ["asset"], "shape": ["N"]}}
            ).compile({"value": expression, "mask": f"{expression}>0"})
            engine = AdaptiveScheduler(cpu_budget=1)
            inputs = {"x": x}
            starts, ends = np.array([0], np.int64), np.array([1], np.int64)
            return lambda: engine.execute(graph, inputs, starts, ends)

    probe = ctypes.CDLL(probe_path) if probe_path else None

    def start():
        gc.collect()
        if probe:
            probe.cme_heap_begin()

    def stop():
        if not probe:
            return None
        stats = HeapStats()
        probe.cme_heap_end(ctypes.byref(stats))
        record = {name: getattr(stats, name) for name, _ in stats._fields_}
        if stats.table_overflows:
            raise RuntimeError("heap tracking table overflow")
        return record

    start()
    function = setup()
    setup_stats = stop()
    start()
    result = function()
    cold_stats = stop()
    actual = result if backend == "njit" else tuple(field.values[0] for field in result.outputs)
    for a, b in zip(actual, (expected, expected > 0), strict=True):
        np.testing.assert_array_equal(a, b)
    audit = None if backend == "njit" else result.audit
    if audit:
        build = audit["engine_build_id"]
    del actual, result
    for _ in range(5):
        function()
    start()
    retained = None
    for _ in range(11):
        retained = function()
    warm_stats = stop()
    del retained
    print(
        json.dumps(
            {
                "build_id": build,
                "audit": audit,
                "setup_heap": setup_stats,
                "cold_heap": cold_stats,
                "warm_heap": warm_stats,
                "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        ),
        flush=True,
    )
    if probe:
        if engine:
            engine.close()
        return
    loops = 200 if x.size < 100 else 20 if x.size < 100000 else 3
    for line in sys.stdin:
        if line.strip() == "stop":
            break
        result = None
        cpu_start, start_ns = time.process_time_ns(), time.perf_counter_ns()
        for _ in range(loops):
            result = function()
        ns = (time.perf_counter_ns() - start_ns) / loops
        cpu_ns = (time.process_time_ns() - cpu_start) / loops
        del result
        print(json.dumps({"ns": ns, "cpu_ns": cpu_ns}), flush=True)
    if engine:
        engine.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path)
    p.add_argument("--after", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--heap-probe", type=Path)
    p.add_argument("--worker-case")
    p.add_argument("--backend", choices=["before", "after", "njit"])
    p.add_argument("--repeats", type=int, default=31)
    args = p.parse_args()
    if args.worker_case:
        worker(
            json.loads(args.worker_case),
            args.backend,
            str(args.heap_probe) if args.heap_probe else None,
        )
        return
    if not args.before or not args.after or not args.output:
        p.error("--before, --after and --output are required")
    reports = []
    for case in CASES:
        workers, metadata = {}, {}
        timings = {name: [] for name in ["before", "after", "njit"]}
        cpu = {name: [] for name in timings}

        def command(backend, heap=False, case=case):
            env = dict(
                os.environ, PYTHONPATH=str(args.before if backend == "before" else args.after)
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-case",
                json.dumps(case),
                "--backend",
                backend,
            ]
            if heap:
                command += ["--heap-probe", str(args.heap_probe)]
                env["DYLD_INSERT_LIBRARIES"] = str(args.heap_probe)
            return command, env

        try:
            for backend in timings:
                cmd, env = command(backend)
                proc = subprocess.Popen(
                    cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
                )
                workers[backend] = proc
                metadata[backend] = json.loads(proc.stdout.readline())
            for repeat in range(args.repeats):
                order = list(timings)
                if repeat % 2:
                    order.reverse()
                for backend in order:
                    proc = workers[backend]
                    proc.stdin.write("time\n")
                    proc.stdin.flush()
                    stats = json.loads(proc.stdout.readline())
                    timings[backend].append(stats["ns"])
                    cpu[backend].append(stats["cpu_ns"])
        finally:
            for proc in workers.values():
                if proc.poll() is None:
                    proc.stdin.write("stop\n")
                    proc.stdin.flush()
                if proc.wait(timeout=30):
                    raise RuntimeError("measurement worker failed")
        heaps = {}
        if args.heap_probe:
            for backend in timings:
                cmd, env = command(backend, True)
                run = subprocess.run(
                    cmd, env=env, text=True, capture_output=True, check=True, timeout=180
                )
                heaps[backend] = json.loads(run.stdout)
        result = {
            "case": case,
            "metadata": metadata,
            "timing_ns": timings,
            "cpu_ns": cpu,
            "heap": heaps,
            "after_before": paired_ratio(timings["after"], timings["before"]),
            "before_njit": paired_ratio(timings["before"], timings["njit"]),
            "after_njit": paired_ratio(timings["after"], timings["njit"]),
        }
        reports.append(result)
        args.output.write_text(
            json.dumps(
                {
                    "schema": "pointwise-region-performance-1",
                    "cases": CASES,
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "scope": "single CPU; warmed ordinary API; independent values and mask; sequential paired workers; no JIT/IPC inside timer",
                    "memory_limits": [
                        "malloc request bytes; excludes CPython arena subdivisions and mmap",
                        "setup live bytes are retained; warm excludes allocations before its tracing epoch",
                        "RSS is process high water, not incremental allocation; NJIT includes compiler runtime",
                        "C++ includes graph validation and audit metadata absent from specialized NJIT",
                    ],
                    "results": reports,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            case["id"],
            "after/before",
            round(result["after_before"]["median"], 3),
            "after/NJIT",
            round(result["after_njit"]["median"], 3),
            flush=True,
        )


if __name__ == "__main__":
    main()
