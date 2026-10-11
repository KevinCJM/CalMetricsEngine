"""Evidence-only real-wheel, owned-result covariance comparisons.

Every child uses one CPU; compilation/import/input construction precede timings.
Old/new ordinary and prepared snapshots retain the preceding result while the
next result is produced. Heap measurement includes live setup and workspace.
No private ABI candidate is measured here; no production Python math callback.
"""
# ruff: noqa: E402
from __future__ import annotations

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

REFERENCE = None
BACKENDS = ("old_ordinary", "new_ordinary", "old_prepared", "new_prepared", "njit")


def inputs(case):
    x = np.random.default_rng(20261001).normal(size=(case["n"], case["d"]))
    if case.get("pattern") == "zeros":
        x.fill(0)
    elif case.get("pattern") == "scaled":
        x *= 1e3
    elif case.get("pattern") == "correlated":
        x[:] = 0.1 * x + x[:, :1]
    x = np.array(x, order=case["layout"].upper())
    x.flags.writeable = False
    return x


def setup(backend, x):
    global REFERENCE
    if backend == "njit":
        if REFERENCE is None:
            from numba import njit

            @njit(nogil=True)
            def covariance_reference(x):
                return np.cov(x.T)

            REFERENCE = covariance_reference
        REFERENCE(x)
        return None, lambda: REFERENCE(x)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    graph = GraphCompiler({"x": dict(kind="matrix", axes=["time", "asset"],
                                    shape=["T", "D"])}).compile("covariance(x)")
    scheduler = AdaptiveScheduler(cpu_budget=1)
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    if backend.endswith("prepared"):
        prepared = scheduler.prepare_execution(graph, {"x": x}, starts, ends)
        return scheduler, prepared.run_snapshot
    return scheduler, lambda: scheduler.execute(graph, {"x": x}, starts, ends)


def values(result):
    return result if isinstance(result, np.ndarray) else result.outputs[0].values[0]


def diagnostics(result):
    if isinstance(result, np.ndarray):
        return {"backend": "warmed NJIT np.cov", "owned": result.flags.owndata}
    names = ("accelerate_covariance_attempts", "accelerate_covariance_calls",
             "covariance_syrk_calls", "covariance_replayed_cells", "covariance_fallbacks",
             "covariance_panel_materialization_bytes", "covariance_zero_calls",
             "input_copy_bytes", "algorithm_copy_bytes")
    return {
        "build_id": result.audit["engine_build_id"],
        "counts": {name: sum(chunk.get(name, 0) for chunk in result.audit["native_chunks"])
                   for name in names},
        "chunks": result.audit["native_chunks"],
        "readonly": not values(result).flags.writeable,
    }


def worker(args):
    owner = result = None
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command["op"] == "init":
                if owner:
                    owner.close()
                result = None
                x = inputs(command["case"])
                owner, function = setup(args.backend, x)
                for _ in range(3):
                    result = function()
                np.save(args.snapshots / f"{args.backend}.npy", values(result))
                response = diagnostics(result)
            else:
                wall, cpu = [], []
                for _ in range(command["repeat"]):
                    begin, cpu_begin = time.perf_counter_ns(), time.process_time_ns()
                    result = function()
                    cpu.append(time.process_time_ns() - cpu_begin)
                    wall.append(time.perf_counter_ns() - begin)
                response = {"ns": wall, "cpu_ns": cpu}
            print(json.dumps(response), flush=True)
    finally:
        if owner:
            owner.close()


def child_env(args, backend):
    env = dict(os.environ, PYTHONPATH=args.old_install if backend.startswith("old") else args.new_install)
    env.pop("DYLD_INSERT_LIBRARIES", None)
    return env


def timing(args):
    processes, files = {}, []
    rows = json.loads((args.output / "timing.json").read_text()) if args.resume else []
    try:
        for backend in BACKENDS:
            error = (args.output / f"timing-{backend}.log").open("w")
            files.append(error)
            processes[backend] = subprocess.Popen(
                [sys.executable, __file__, "--mode", "worker", "--backend", backend,
                 "--output", str(args.output), "--snapshots", str(args.snapshots)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, text=True,
                env=child_env(args, backend))

        def send(backend, command):
            p = processes[backend]
            p.stdin.write(json.dumps(command) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"failed {backend}; see timing log")
            return json.loads(line)

        cases = [dict(n=n, d=d, layout=layout, pattern="normal") for n, d in
                 ((63, 8), (512, 256), (2048, 2048), (4096, 4096)) for layout in ("c", "f")]
        cases += [dict(n=n, d=d, layout=layout, pattern=pattern) for n, d, pattern in
                  ((2048, 2048, "zeros"), (2048, 2048, "scaled"),
                   (32769, 256, "correlated")) for layout in ("c", "f")]
        for case in cases:
            if any(row["case"] == case for row in rows):
                continue
            audits = {backend: send(backend, {"op": "init", "case": case}) for backend in BACKENDS}
            original = np.load(args.snapshots / "old_ordinary.npy", mmap_mode="r")
            comparisons = {}
            for backend in BACKENDS[1:]:
                actual = np.load(args.snapshots / f"{backend}.npy", mmap_mode="r")
                good = np.isclose(actual, original, rtol=1e-12, atol=1e-12, equal_nan=True)
                row = {"outside_tolerance": int(np.count_nonzero(~good)),
                       "max_abs": float(np.max(np.abs(actual - original)))}
                if backend != "njit":
                    assert row["outside_tolerance"] == 0, (case, backend, row)
                    row["sign_changed"] = int(np.count_nonzero((actual > 0) != (original > 0)))
                    row["zero_changed"] = int(np.count_nonzero((actual == 0) != (original == 0)))
                    row["diagonal_bits_changed"] = int(np.count_nonzero(
                        np.diag(actual).view(np.uint64) != np.diag(original).view(np.uint64)))
                    assert row["sign_changed"] == row["zero_changed"] == row["diagonal_bits_changed"] == 0
                else:
                    # Some scaled inputs already differ between canonical C++
                    # and np.cov at the unchanged tolerance. Preserve that fact;
                    # performance measurement cannot redefine the contract.
                    row["status"] = "PASS" if row["outside_tolerance"] == 0 else "PREEXISTING_REFERENCE_DIFFERENCE"
                comparisons[backend] = row
            samples, cpus = {k: [] for k in BACKENDS}, {k: [] for k in BACKENDS}
            # Ten rotations balance all five children twice. Keep raw observations.
            for rotation in range(10):
                order = BACKENDS[rotation % 5:] + BACKENDS[:rotation % 5]
                for backend in order:
                    response = send(backend, {"op": "run", "repeat": 21 if case["n"] < 100 else 1})
                    samples[backend].extend(response["ns"])
                    cpus[backend].extend(response["cpu_ns"])
            report = {"case": case, "audits": audits, "comparisons": comparisons,
                      "samples_ns": samples, "cpu_ns": cpus,
                      "median_ms": {k: statistics.median(v) / 1e6 for k, v in samples.items()},
                      "median_cpu_wall_ratio": {k: statistics.median(c / w for c, w in
                          zip(cpus[k], samples[k], strict=True)) for k in BACKENDS}}
            rows.append(report)
            (args.output / "timing.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({"case": case, "median_ms": report["median_ms"],
                              "new_counts": audits["new_ordinary"]["counts"]}), flush=True)
    finally:
        for p in processes.values():
            p.stdin.close()
            p.wait(timeout=30)
        for f in files:
            f.close()


def memory(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from benchmark_memory import measure_heap, measure_rss

    import calmetrics_engine  # noqa: F401
    x = inputs(json.loads(args.case))
    if args.backend == "njit":
        setup(args.backend, x)[1]()  # Warm compilation before the allocation epoch.
    if args.measure == "heap":
        owner, _, report = measure_heap(lambda: setup(args.backend, x), args.probe, repeats=2)
    else:
        owner, function = setup(args.backend, x)
        function()
        report = measure_rss(function, repeats=2)
    if owner:
        owner.close()
    report["build"] = calmetrics_engine.build_info() if args.backend != "njit" else "warmed NJIT np.cov"
    print(json.dumps(report), flush=True)


def memory_sweep(args):
    rows = []
    shapes = ((2048, 2048),) if args.memory_pattern == "zeros" else ((2048, 2048), (4096, 4096))
    for n, d in shapes:
        for layout in ("c", "f"):
            case = dict(n=n, d=d, layout=layout, pattern=args.memory_pattern)
            for backend in BACKENDS:
                for measure in ("heap", "rss"):
                    env = child_env(args, backend)
                    if measure == "heap":
                        env["DYLD_INSERT_LIBRARIES"] = args.probe
                    run = subprocess.run(
                        [sys.executable, __file__, "--mode", "memory", "--backend", backend,
                         "--measure", measure, "--case", json.dumps(case), "--probe", args.probe,
                         "--output", str(args.output), "--snapshots", str(args.snapshots)],
                        env=env, capture_output=True, text=True, timeout=240)
                    prefix = f"memory-{n}-{layout}-{backend}-{measure}"
                    (args.output / f"{prefix}.log").write_text(run.stderr)
                    if run.returncode:
                        raise RuntimeError(f"{prefix} failed ({run.returncode}); see log")
                    report = json.loads(run.stdout)
                    if measure == "heap":
                        assert report["cold_peak_bytes"] >= d * d * 8
                        assert report["warm_peak_bytes"] >= 2 * d * d * 8
                        assert all(report[k]["table_overflows"] == 0 for k in report if k.endswith("_heap"))
                    rows.append({"case": case, "backend": backend, "measure": measure, "report": report})
                    (args.output / "memory.json").write_text(json.dumps(rows, indent=2))
                    summary = ({k: report[k] for k in ("cold_peak_bytes", "warm_peak_bytes")}
                               if measure == "heap" else report)
                    print(json.dumps({"case": case, "backend": backend, "measure": measure,
                                      "summary": summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "timing", "memory", "memory-sweep"), required=True)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--old-install")
    parser.add_argument("--new-install")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, default=Path("/private/tmp/calmetrics-covariance-integration-snapshots"))
    parser.add_argument("--measure", choices=("heap", "rss"))
    parser.add_argument("--probe")
    parser.add_argument("--case")
    parser.add_argument("--resume", action="store_true", help="Keep completed rows and measure only remaining cases")
    parser.add_argument("--memory-pattern", choices=("normal", "zeros"), default="normal")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.snapshots.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        {"worker": worker, "timing": timing, "memory": memory,
         "memory-sweep": memory_sweep}[args.mode](args)


if __name__ == "__main__":
    main()
