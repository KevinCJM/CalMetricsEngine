"""Strict, reproducible M1 formula parity, paired timing and isolated RSS gate.

This developer tool never enters the wheel. NJIT is warmed with the exact input
layout, and both backends retain the latest independent result. A sampled peak
is only a lower bound; sub-page/zero deltas fail as unproven, never pass by equality.
Full platform algorithm equivalence remains a separate M0/M2-M8 inventory gate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def references():
    from numba import njit

    @njit(nogil=True)
    def affine(x):
        values = np.empty(x.shape)
        mask = np.empty(x.shape, np.bool_)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    values[i, j, k] = x[i, j, k] * 2.0 + 1.0
                    mask[i, j, k] = values[i, j, k] > 3.0
        return values, mask

    @njit(nogil=True)
    def relaxation(x):
        current = x.copy()
        residual = np.inf
        count = 0
        status = 1
        for iteration in range(100):
            residual = 0.0
            for i in range(current.size):
                candidate = current[i] * 0.5
                residual = max(residual, abs(candidate - current[i]))
                current[i] = candidate
            count = iteration + 1
            if residual <= 1e-6:
                status = 0
                break
        return current, np.asarray(status, dtype=np.int64), np.asarray(count, dtype=np.int64), np.asarray(residual)

    return {"affine": affine, "relaxation": relaxation}


def data(case):
    # Fixed exact values make the stopping iteration independently checkable.
    x = np.full(case["shape"], 8., np.float64)
    if case["layout"] == "reverse":
        x = x[:, ::-1, ::-1]
    x.flags.writeable = False
    return x


def native(case, x, include_prepared=True):
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    if case["kernel"] == "affine":
        types = {"x": {"kind": "tensor", "axes": ["scenario", "path", "asset"], "shape": ["S", "P", "N"]}}
        expressions = {"value": "x*2+1", "mask": "x*2+1>3"}
    else:
        types = {"x": {"kind": "vector", "axes": ["asset"], "shape": ["N"]}}
        expr = "iterate(iterate_x*0.5,x,1e-6,100)"
        expressions = {"value": expr, "status": f"iteration_status({expr})",
                       "count": f"iteration_count({expr})", "residual": f"iteration_residual({expr})"}
    engine = AdaptiveScheduler(cpu_budget=1)
    graph = GraphCompiler(types).compile(expressions)
    starts, ends = np.array([0], np.int64), np.array([1], np.int64)
    inputs = {"x": x}
    prepared = engine.prepare_execution(graph, inputs, starts, ends) if include_prepared else None
    return engine, lambda: engine.execute(graph, inputs, starts, ends), prepared.run_snapshot if prepared else None


def samples(function, loops, repeats):
    result = None
    values = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(loops):
            result = function()
        values.append((time.perf_counter_ns() - started) / loops)
    return values, result


def paired_ratio(a, b):
    ratios = np.asarray(a) / np.asarray(b)
    rng = np.random.default_rng(20260923)
    indices = rng.integers(0, len(ratios), size=(4000, len(ratios)))
    return {"median": float(np.median(ratios)), "upper95": float(np.quantile(np.median(ratios[indices], axis=1), .95)),
            "cpp_p95_ns": float(np.quantile(a, .95)), "njit_p95_ns": float(np.quantile(b, .95)),
            "cpp_ns": a, "njit_ns": b}


def memory_worker(case, backend):
    import psutil
    x = data(case)
    engine = None
    if backend == "njit":
        reference = references()[case["kernel"]]
        function = lambda: reference(x)
    else:
        engine, ordinary, prepared = native(case, x, include_prepared=backend != "ordinary")
        function = ordinary if backend == "ordinary" else prepared
    warm = function()
    del warm
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    print(json.dumps({"ready": True, "baseline": baseline}), flush=True)
    sys.stdin.readline()
    retained = None
    stop = time.monotonic() + .3
    while time.monotonic() < stop:
        retained = function()
    print(json.dumps({"final_rss": process.memory_info().rss, "retained": retained is not None}), flush=True)
    # Keep the final result alive for a parent sample, then exit cleanly.
    sys.stdin.readline()
    if engine:
        engine.close()


def isolated_memory(script, workload, case, backend):
    import psutil
    proc = subprocess.Popen([sys.executable, str(script), "--workloads", str(workload), "--case", case["id"],
                             "--memory-worker", backend], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        ready = json.loads(proc.stdout.readline())
        baseline = ready["baseline"]
        watched = psutil.Process(proc.pid)
        peak = baseline
        proc.stdin.write("run\n"); proc.stdin.flush()
        # select avoids a polling thread with its own allocator overhead.
        import select
        while not select.select([proc.stdout], [], [], .001)[0]:
            peak = max(peak, watched.memory_info().rss + sum(p.memory_info().rss for p in watched.children(recursive=True)))
        finished = json.loads(proc.stdout.readline())
        peak = max(peak, finished["final_rss"])
        proc.stdin.write("exit\n"); proc.stdin.flush()
        _, error = proc.communicate(timeout=30)
        if proc.returncode:
            raise RuntimeError(error)
        return {"baseline": baseline, "peak": peak, "increment": max(0, peak-baseline)}
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, default=Path(__file__).resolve().parents[1] / "docs/platform-foundation-workloads.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument("--memory-worker", choices=["njit", "ordinary", "prepared"])
    args = parser.parse_args()
    workloads = json.loads(args.workloads.read_text())
    cases = [c for c in workloads["cases"] if not args.case or c["id"] == args.case]
    if not cases or args.repeats < 5 or args.memory_repeats < 3:
        parser.error("valid cases, >=5 timing pairs and >=3 isolated memory repeats are required")
    if args.memory_worker:
        memory_worker(cases[0], args.memory_worker)
        return
    refs = references()
    from calmetrics_engine import build_info
    report = {"build_info": build_info(), "schema": "foundation-performance-2", "workload_sha256": hashlib.sha256(args.workloads.read_bytes()).hexdigest(),
              "python": sys.version, "platform": sys.platform, "cpu_budget": 1, "cases": [],
              "memory_method": "isolated warm process RSS sampled at 1ms; ordinary does not retain an unused prepared buffer; positive deltas below 2 pages are unproven"}
    for case in cases:
        x = data(case)
        reference = refs[case["kernel"]]
        expected = reference(x)
        engine, ordinary, prepared = native(case, x)
        try:
            for result in [ordinary(), prepared()]:
                for field, wanted in zip(result.outputs, expected, strict=True):
                    actual = field.values[0]
                    assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
                    np.testing.assert_array_equal(actual, wanted)
                report["engine_build_id"] = result.audit["engine_build_id"]
            if case["kernel"] == "relaxation":
                assert expected[1] == 0 and expected[2] == 23 and expected[3] <= 1e-6
            loops = max(1, min(200, int(200000 / max(x.size, 1))))
            row = {"id": case["id"], "shape": case["shape"], "correct": True, "timing": {}, "memory": {}}
            for name, function in [("ordinary", ordinary), ("prepared", prepared)]:
                a, b = [], []
                for repeat in range(args.repeats):
                    order = [(a, function), (b, lambda: reference(x))]
                    if repeat % 2:
                        order.reverse()
                    for target, run in order:
                        measured, _ = samples(run, loops, 1)
                        target.extend(measured)
                row["timing"][name] = paired_ratio(a, b)
            row["memory"] = {name: [] for name in ["njit", "ordinary", "prepared"]}
            for repeat in range(args.memory_repeats):
                order = ["njit", "ordinary", "prepared"]
                if repeat % 2: order.reverse()
                for backend in order:
                    row["memory"][backend].append(isolated_memory(Path(__file__), args.workloads, case, backend))
            rss, delta = {}, {}
            for name in ["ordinary", "prepared"]:
                rss[name] = paired_ratio([m["peak"] for m in row["memory"][name]], [m["peak"] for m in row["memory"]["njit"]])
                increments = [m["increment"] for m in row["memory"][name]]
                baseline = [m["increment"] for m in row["memory"]["njit"]]
                delta[name] = paired_ratio(increments, baseline) if min(increments + baseline) >= 2*os.sysconf("SC_PAGE_SIZE") else None
            row["rss_ratios"], row["increment_ratios"] = rss, delta
            row["passed"] = all(row["timing"][n]["upper95"] < 1 and rss[n]["upper95"] < 1 and
                                  delta[n] is not None and delta[n]["upper95"] < 1 for n in ["ordinary", "prepared"])
            report["cases"].append(row)
            print(json.dumps({"case": case["id"], "passed": row["passed"],
                  "time_upper95": {n: row["timing"][n]["upper95"] for n in row["timing"]}}), flush=True)
        finally:
            engine.close()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if all(row["passed"] for row in report["cases"]) else 1)


if __name__ == "__main__":
    main()
