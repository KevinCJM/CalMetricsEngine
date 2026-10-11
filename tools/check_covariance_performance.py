"""Compare two installed AOT builds and warmed NJIT with retained independent results.

Paired timing children are idle except for the currently measured implementation.
Heap/RSS run in separate children; input arrays pre-exist the allocation epoch,
while all graph preparation and retained workspaces are included. Fixed CPU=1.
"""

# Thread limits must precede NumPy/Numba imports.
# ruff: noqa: E402
from __future__ import annotations

import os

for _key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_key] = "1"

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from benchmark_memory import measure_heap, measure_rss


def inputs(case):
    x = np.random.default_rng(20260929).normal(size=(case["n"], case["d"]))
    if case["layout"] == "f":
        x = np.asfortranarray(x)
    elif case["layout"] == "reverse":
        x = x[::-1, ::-1]
    x.flags.writeable = False
    return x


_reference = None


def setup(case, backend, x):
    global _reference
    if backend == "njit":
        if _reference is None:
            from numba import njit

            @njit(nogil=True)
            def reference(x, correlation):
                if correlation:
                    return np.corrcoef(x.T)
                return np.cov(x.T)

            _reference = reference
        correlation = case["name"] == "correlation"
        _reference(x, correlation)  # compile before any warm timer
        return None, lambda: _reference(x, correlation)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler, operators

    if case["entry"] == "direct":
        return None, lambda: operators.call(case["name"], x)
    g = GraphCompiler(
        {"x": {"kind": "matrix", "axes": ["time", "asset"], "shape": ["T", "D"]}}
    ).compile(f"{case['name']}(x)")
    bounds = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    scheduler = AdaptiveScheduler(cpu_budget=1)
    data = {"x": x}
    if case["entry"] == "prepared":
        prepared = scheduler.prepare_execution(g, data, *bounds)
        return scheduler, prepared.run_snapshot
    return scheduler, lambda: scheduler.execute(g, data, *bounds)


def value(result):
    return result if isinstance(result, np.ndarray) else result.outputs[0].values[0]


def worker(backend, output):
    owner = None
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command["op"] == "init":
                if owner:
                    owner.close()
                case = command["case"]
                x = inputs(case)
                owner, function = setup(case, backend, x)
                result = function()
                np.save(output / f"{backend}.npy", value(result))
                from calmetrics_engine import build_info

                response = {"build": build_info() if backend != "njit" else "numba-numpy-linalg"}
            else:
                elapsed = []
                for _ in range(command["repeat"]):
                    start = time.perf_counter_ns()
                    result = function()
                    elapsed.append(time.perf_counter_ns() - start)
                response = {"ns": elapsed, "audit": getattr(result, "audit", {})}
            print(json.dumps(response), flush=True)
    finally:
        if owner:
            owner.close()


def memory(args):
    case = json.loads(args.case)
    x = inputs(case)
    # Never warm a native graph before the epoch: its thread-local workspace
    # survives Scheduler.close() and must remain visible to malloc accounting.
    if args.backend == "njit":
        owner, function = setup(case, args.backend, x)
        function()
        del owner, function
    else:
        import calmetrics_engine  # noqa: F401 -- import initialization is not request work
    if args.memory == "heap":
        owner, function, report = measure_heap(
            lambda: setup(case, args.backend, x), args.heap_probe, repeats=11
        )
    else:
        owner, function = setup(case, args.backend, x)
        function()
        report = measure_rss(function, repeats=41)
    if owner:
        owner.close()
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-install")
    parser.add_argument("--new-install")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("old", "new", "njit"))
    parser.add_argument("--memory", choices=("heap", "rss"))
    parser.add_argument("--backend", choices=("old", "new", "njit"))
    parser.add_argument("--case")
    parser.add_argument("--heap-probe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, args.output)
        return
    if args.memory:
        memory(args)
        return
    process, errors, records = {}, [], []
    try:
        for backend, install in (
            ("old", args.old_install),
            ("new", args.new_install),
            ("njit", args.new_install),
        ):
            err = (args.output / f"{backend}.log").open("w")
            errors.append(err)
            env = dict(os.environ, PYTHONPATH=install)
            env.pop("DYLD_INSERT_LIBRARIES", None)
            process[backend] = subprocess.Popen(
                [sys.executable, __file__, "--worker", backend, "--output", str(args.output)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                env=env,
            )

        def send(backend, command):
            p = process[backend]
            p.stdin.write(json.dumps(command) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"{backend} worker failed; inspect its log")
            return json.loads(line)

        cases = [
            dict(name=name, n=n, d=d, layout=layout, entry="ordinary")
            for name in ("covariance", "correlation")
            for n, d in ((63, 8), (2048, 8), (32768, 8), (2048, 16), (32768, 16), (32768, 64))
            for layout in ("c", "f")
        ]
        cases += [dict(name="covariance", n=2048, d=16, layout="reverse", entry="ordinary")]
        cases += [
            dict(name="covariance", n=32768, d=64, layout="c", entry=entry)
            for entry in ("direct", "prepared")
        ]
        for case in cases:
            identities = {k: send(k, {"op": "init", "case": case}) for k in process}
            arrays = {k: np.load(args.output / f"{k}.npy") for k in process}
            for k in ("new", "njit"):
                np.testing.assert_allclose(arrays[k], arrays["old"], rtol=1e-11, atol=1e-11)
            samples = {k: [] for k in process}
            last = {}
            for iteration in range(11):
                keys = list(process)
                for k in keys[iteration % 3 :] + keys[: iteration % 3]:
                    last[k] = send(k, {"op": "run", "repeat": 5 if case["n"] < 2048 else 1})
                    samples[k].extend(last[k]["ns"])
            medians = {k: statistics.median(v) / 1e6 for k, v in samples.items()}
            records.append(
                dict(
                    case=case,
                    samples_ns=samples,
                    median_ms=medians,
                    builds=identities,
                    audits={k: v["audit"] for k, v in last.items()},
                    correctness=True,
                )
            )
            (args.output / "timing.json").write_text(json.dumps(records, indent=2))
            print(case, medians, flush=True)
    finally:
        for p in process.values():
            p.stdin.close()
            p.wait(timeout=30)
        for err in errors:
            err.close()


if __name__ == "__main__":
    main()
