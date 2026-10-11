"""Isolated numerical-library screening; never a production Scheduler gate."""
# ruff: noqa: E402
from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

_reference = None


def candidate_setup(library, x):
    lib = ctypes.CDLL(library)
    pointer = ctypes.POINTER(ctypes.c_double)
    function = lib.covariance_syrk_screen
    function.argtypes = (pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         ctypes.c_ssize_t, ctypes.c_int, pointer)
    function.restype = ctypes.c_int
    threads = lib.covariance_syrk_threading
    threads.restype = ctypes.c_int
    setter = lib.covariance_syrk_set_threading
    setter.argtypes = (ctypes.c_int,)
    setter.restype = ctypes.c_int
    layout = int(x.strides[0] == 8)
    leading = x.strides[1 if layout else 0] // 8
    n, d = x.shape
    source = x.ctypes.data_as(pointer)

    def raw(out, block=512):
        return function(source, n, d, layout, leading, block, out.ctypes.data_as(pointer))

    def run():
        out = np.empty((d, d), np.float64)
        status = raw(out)
        if status:
            raise ValueError(f"screen status {status}")
        return out
    return run, raw, threads, setter


def inputs(case):
    x = np.random.default_rng(20261001).normal(size=(case["n"], case["d"]))
    x = np.array(x, order=case["layout"].upper())
    x.flags.writeable = False
    return x


def setup(args, backend, x):
    global _reference
    if backend == "candidate":
        return None, candidate_setup(args.library, x)[0]
    if backend == "njit":
        if _reference is None:
            from numba import njit
            @njit(nogil=True)
            def reference(x):
                return np.cov(x.T)
            _reference = reference
        _reference(x)
        return None, lambda: _reference(x)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    graph = GraphCompiler({"x": {"kind": "matrix", "axes": ["time", "asset"],
                                 "shape": ["T", "D"]}}).compile("covariance(x)")
    scheduler = AdaptiveScheduler(cpu_budget=1)
    start, end = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    return scheduler, lambda: scheduler.execute(graph, {"x": x}, start, end)


def array(result):
    return result if isinstance(result, np.ndarray) else result.outputs[0].values[0]


def checks(args):
    from calmetrics_engine import build_info, operators
    rows = []
    rng = np.random.default_rng(9117)
    for n, d in ((31, 8), (129, 17), (513, 64), (1041, 128)):
        base = rng.normal(size=(n, d))
        for pattern in ("normal", "constant", "zeros", "duplicate", "offset",
                        "scaled", "opposed", "orthogonal", "extreme", "nan", "inf"):
            source = base.copy()
            if pattern == "constant":
                source[:, -1] = 7.0
            elif pattern == "zeros":
                source[:] = 0.0
                source[::2, ::2] = -0.0
            elif pattern == "duplicate":
                source[:, -1] = source[:, 0]
            elif pattern == "offset":
                source += 1e3
            elif pattern == "scaled":
                source *= 1e3
            elif pattern == "opposed":
                source[:, -1] = -source[:, 0]
            elif pattern == "orthogonal":
                q, _ = np.linalg.qr(np.column_stack([np.ones(n), source[:, :min(d, n - 1)]]))
                source[:, :q.shape[1] - 1] = q[:, 1:]
            elif pattern == "extreme":
                source *= 1e8
            elif pattern in ("nan", "inf"):
                source[-2, -1] = np.nan if pattern == "nan" else np.inf
            for layout in ("c", "f", "padded_c", "padded_f"):
                if layout == "padded_c":
                    owner = np.full((n, d + 3), -937.25)
                    x = owner[:, :d]
                    x[:] = source
                elif layout == "padded_f":
                    owner = np.full((n + 4, d), -937.25, order="F")
                    x = owner[1:n + 1]
                    x[:] = source
                else:
                    owner = x = np.array(source, order=layout.upper())
                x.flags.writeable = False
                before = owner.view(np.uint64).copy()
                run, raw, threads, setter = candidate_setup(args.library, x)
                original_threading = threads()
                guarded = np.full(d * d + 2, -193.5)
                output = guarded[1:-1].reshape(d, d)
                expected = operators.call("covariance", x)
                code = raw(output)
                record = dict(n=n, d=d, pattern=pattern, layout=layout, status=code)
                np.testing.assert_array_equal(owner.view(np.uint64), before)
                assert guarded[0] == guarded[-1] == -193.5
                assert threads() == original_threading
                if code:
                    assert pattern in ("extreme", "nan", "inf")
                    np.testing.assert_array_equal(output, -193.5)
                else:
                    good = np.isclose(output, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
                    record.update(
                        outside_tolerance=int(np.count_nonzero(~good)),
                        max_abs=float(np.max(np.abs(output - expected))),
                        positive_flip=int(np.count_nonzero((output > 0) != (expected > 0))),
                        zero_flip=int(np.count_nonzero((output == 0) != (expected == 0))),
                        bits_changed=int(np.count_nonzero(output.view(np.uint64) != expected.view(np.uint64))),
                    )
                    np.testing.assert_array_equal(output, output.T)
                    for classify in (np.isnan, np.isposinf, np.isneginf):
                        np.testing.assert_array_equal(classify(output), classify(expected))
                    saved = output.copy()
                    next_output = run()
                    assert not np.shares_memory(output, next_output)
                    np.testing.assert_array_equal(output, saved)
                # Exercise caller policy restoration after both success/rejection.
                assert setter(0) == 0
                policy = threads()
                raw(output)
                assert threads() == policy
                assert setter(original_threading) == 0
                rows.append(record)
    report = dict(build=build_info(), cases=rows,
                  observed_tolerance_failures=sum(r.get("outside_tolerance", 0) > 0 for r in rows),
                  observed_sign_cases=sum(r.get("positive_flip", 0) > 0 for r in rows))
    (args.output / "correctness.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
    if report["observed_tolerance_failures"] or report["observed_sign_cases"]:
        raise SystemExit(2)  # A diagnostic with numerical failures is not acceptance.


def worker(args):
    from calmetrics_engine import build_info
    owner = None
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command["op"] == "init":
                if owner:
                    owner.close()
                case = command["case"]
                x = inputs(case)
                owner, function = setup(args, args.backend, x)
                result = function()
                np.save(args.output / f"{args.backend}.npy", array(result))
                response = dict(build=build_info() if args.backend == "native" else args.backend)
            else:
                wall, cpu = [], []
                for _ in range(command["repeat"]):
                    begin = time.perf_counter_ns()
                    cpu_begin = time.process_time_ns()
                    result = function()
                    cpu.append(time.process_time_ns() - cpu_begin)
                    wall.append(time.perf_counter_ns() - begin)
                response = dict(ns=wall, cpu_ns=cpu)
            print(json.dumps(response), flush=True)
    finally:
        if owner:
            owner.close()


def timing(args):
    processes, files, rows = {}, [], []
    try:
        for backend in ("native", "candidate", "njit"):
            err = (args.output / f"{backend}.log").open("w")
            files.append(err)
            env = dict(os.environ, PYTHONPATH=args.install)
            env.pop("DYLD_INSERT_LIBRARIES", None)
            processes[backend] = subprocess.Popen(
                [sys.executable, __file__, "--mode", "worker", "--backend", backend,
                 "--library", args.library, "--output", str(args.output)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, env=env)

        def send(backend, message):
            p = processes[backend]
            p.stdin.write(json.dumps(message) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"failed {backend}; see log")
            return json.loads(line)

        cases = [dict(n=n, d=d, layout=layout) for n, d in
                 ((63, 8), (32768, 64), (512, 256), (2048, 2048), (4096, 4096))
                 for layout in ("c", "f")]
        for case in cases:
            builds = {k: send(k, dict(op="init", case=case)) for k in processes}
            arrays = {k: np.load(args.output / f"{k}.npy") for k in processes}
            comparison = {}
            for backend in ("candidate", "njit"):
                good = np.isclose(arrays[backend], arrays["native"], rtol=1e-12, atol=1e-12)
                comparison[backend] = dict(outside_tolerance=int(np.count_nonzero(~good)),
                                          max_abs=float(np.max(np.abs(arrays[backend] - arrays["native"]))))
            samples = {k: [] for k in processes}
            cpu = {k: [] for k in processes}
            for iteration in range(6):
                keys = list(processes)
                for key in keys[iteration % 3:] + keys[:iteration % 3]:
                    response = send(key, dict(op="run", repeat=7 if case["n"] < 100 else 1))
                    samples[key].extend(response["ns"])
                    cpu[key].extend(response["cpu_ns"])
            report = dict(case=case, builds=builds, comparison=comparison,
                          samples_ns=samples, cpu_ns=cpu,
                          median_ms={k: statistics.median(v) / 1e6 for k, v in samples.items()},
                          median_cpu_wall_ratio={k: statistics.median(a / b for a, b in zip(cpu[k], samples[k], strict=True)) for k in processes})
            rows.append(report)
            (args.output / "timing.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({k: report[k] for k in ("case", "median_ms", "comparison")}), flush=True)
    finally:
        for p in processes.values():
            p.stdin.close()
            p.wait(timeout=30)
        for f in files:
            f.close()


def memory(args):
    # Input, imports and compilation precede the continuous malloc epoch.
    import calmetrics_engine  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from benchmark_memory import measure_heap, measure_rss
    x = inputs(json.loads(args.case))
    if args.backend == "njit":
        _, function = setup(args, "njit", x)
        function()
        del function
    if args.measure == "heap":
        owner, _, report = measure_heap(lambda: setup(args, args.backend, x), args.probe, repeats=2)
    else:
        owner, function = setup(args, args.backend, x)
        function()
        report = measure_rss(function, repeats=2)
    if owner:
        owner.close()
    print(json.dumps(report), flush=True)


def memory_sweep(args):
    reports = []
    for layout in ("c", "f"):
        case = dict(n=4096, d=4096, layout=layout)
        for backend in ("native", "candidate", "njit"):
            for measure in ("heap", "rss"):
                env = dict(os.environ, PYTHONPATH=args.install)
                env.pop("DYLD_INSERT_LIBRARIES", None)
                if measure == "heap":
                    env["DYLD_INSERT_LIBRARIES"] = args.probe
                command = [sys.executable, __file__, "--mode", "memory",
                           "--backend", backend, "--measure", measure,
                           "--library", args.library, "--probe", args.probe,
                           "--case", json.dumps(case), "--output", str(args.output)]
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
                prefix = f"memory-{backend}-{layout}-{measure}"
                (args.output / f"{prefix}.log").write_text(run.stderr)
                if run.returncode:
                    raise RuntimeError(f"{prefix} failed: {run.returncode}; see log")
                report = json.loads(run.stdout)
                if measure == "heap":
                    assert report["cold_peak_bytes"] >= case["d"] ** 2 * 8
                    assert report["warm_peak_bytes"] >= 2 * case["d"] ** 2 * 8
                    assert report["warm_heap"]["table_overflows"] == 0
                reports.append(dict(case=case, backend=backend, measure=measure, report=report))
                (args.output / "memory.json").write_text(json.dumps(reports, indent=2))
                summary = ({k: report[k] for k in ("cold_peak_bytes", "warm_peak_bytes")}
                           if measure == "heap" else report)
                print(json.dumps(dict(case=case, backend=backend, measure=measure, summary=summary)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "timing", "worker", "memory", "memory-sweep"), required=True)
    parser.add_argument("--backend", choices=("native", "candidate", "njit"))
    parser.add_argument("--library", required=True)
    parser.add_argument("--install")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure", choices=("heap", "rss"))
    parser.add_argument("--probe")
    parser.add_argument("--case")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        {"check": checks, "timing": timing, "worker": worker,
         "memory": memory, "memory-sweep": memory_sweep}[args.mode](args)


if __name__ == "__main__":
    main()
