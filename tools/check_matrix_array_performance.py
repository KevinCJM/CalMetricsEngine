"""Frozen M2 foundation workloads. CPU=1; fresh retained results; warm NJIT/LAPACK.

Timing is paired, with the malloc interposer disabled. RSS and allocator evidence
run in separate children. A zero RSS delta is unproven, never memory superiority.
The NJIT array reference fuses selection/reduction, not a deliberately slow clone.
"""

# Thread limits must precede NumPy/Numba loading.
# ruff: noqa: E402
from __future__ import annotations

import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from check_foundation_performance import paired_ratio
from numba import njit


@njit(nogil=True)
def matrix_reference(a, b, mode):
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            if not np.isfinite(a[i, j]) or (mode == 0 and a[i, j] != a[j, i]):
                raise ValueError("invalid matrix")
    if mode == 0:
        return np.linalg.cholesky(a)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if not np.isfinite(b[i, j]):
                raise ValueError("invalid RHS")
    if mode == 2:
        return np.linalg.solve(a, b)
    result = b.copy()
    for i in range(n):
        if abs(a[i, i]) <= 1e-14:
            raise ValueError("singular")
        for k in range(i):
            for j in range(b.shape[1]):
                result[i, j] -= a[i, k] * result[k, j]
        for j in range(b.shape[1]):
            result[i, j] /= a[i, i]
    return result


@njit(nogil=True)
def eigen_reference(a):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if not np.isfinite(a[i, j]) or a[i, j] != a[j, i]:
                raise ValueError("invalid matrix")
    w, v = np.linalg.eigh(a)
    vectors = np.empty(v.shape)
    for j in range(v.shape[1]):
        pivot = 0
        for i in range(1, v.shape[0]):
            if abs(v[i, j]) > abs(v[pivot, j]):
                pivot = i
        sign = -1.0 if v[pivot, j] < 0 else 1.0
        for i in range(v.shape[0]):
            vectors[i, j] = sign * v[i, j]
    return w, vectors


@njit(nogil=True)
def array_reference(x, indices):
    result = np.empty((x.shape[0], x.shape[2]))
    for k in range(indices.size):
        if indices[k] < 0 or indices[k] >= x.shape[1]:
            raise ValueError("index")
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            value = 0.0
            for k in range(indices.size):
                value += x[i, indices[k], j]
            result[i, j] = value
    return result


def cases(extended=False):
    return [
        {"name": f"{kernel}_{n}", "kernel": kernel, "n": n, "rhs": 8}
        for kernel in ("cholesky", "symmetric_eigh", "solve_multi", "triangular_solve")
        for n in ([3, 16, 64] + ([128] if extended else []))
    ] + [
        {"name": f"array_{scale}_{layout}", "kernel": "array", "shape": shape, "layout": layout}
        for scale, shape in [("small", (4, 16, 8)), ("large", (16, 128, 32))]
        for layout in ("contiguous", "reverse")
    ]


def inputs(case):
    rng = np.random.default_rng(20260925)
    if case["kernel"] == "array":
        x = rng.normal(size=case["shape"])
        if case["layout"] == "reverse":
            x = x[:, ::-1, ::-1]
        ix = np.arange(0, x.shape[1], 2, dtype=np.int64)
        x.flags.writeable = False
        ix.flags.writeable = False
        return {"x": x, "ix": ix}
    n = case["n"]
    source = rng.normal(size=(n, n))
    a = source @ source.T + np.eye(n) * n
    b = rng.normal(size=(n, case["rhs"]))
    a.flags.writeable = False
    b.flags.writeable = False
    return {"a": a, "b": b}


def setup(case, backend, data):
    if backend == "njit":
        if case["kernel"] == "array":
            return None, lambda: array_reference(data["x"], data["ix"])
        if case["kernel"] == "symmetric_eigh":
            return None, lambda: eigen_reference(data["a"])
        mode = ("cholesky", "symmetric_eigh", "solve_multi", "triangular_solve").index(
            case["kernel"]
        )
        return None, lambda: matrix_reference(data["a"], data["b"], mode)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler, operators

    if case["kernel"] == "array":
        definitions = {
            "x": {
                "kind": "tensor",
                "axes": ["scenario", "path", "asset"],
                "shape": ["S", "P", "N"],
            },
            "ix": {"kind": "vector", "dtype": "int64", "axes": ["path"], "shape": ["K"]},
        }
        expression = "sum_axis(take_axis(swap_axes(x,1,2),ix,2),2)"
    else:
        definitions = {"a": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["N", "N"]}}
        arguments = "a"
        if case["kernel"] in ("solve_multi", "triangular_solve"):
            arguments = "a,b"
            definitions["b"] = {"kind": "matrix", "axes": ["asset", "factor"], "shape": ["N", "K"]}
        expression = f"{case['kernel']}({arguments})"
        if backend == "direct":
            args = [data[k] for k in arguments.split(",")]
            if case["kernel"] == "symmetric_eigh":

                def eigen_call():
                    pair = operators.symmetric_eigh(*args)
                    return operators.eigenvalues(pair), operators.eigenvectors(pair)

                return None, eigen_call
            return None, lambda: getattr(operators, case["kernel"])(*args)
        if case["kernel"] == "symmetric_eigh":
            expression = [f"eigenvalues({expression})", f"eigenvectors({expression})"]
    graph = GraphCompiler(definitions).compile(expression)
    engine = AdaptiveScheduler(cpu_budget=1)
    starts = ends = np.array([0], np.int64)
    selected = {k: data[k] for k in definitions}
    if backend == "prepared":
        prepared = engine.prepare_execution(graph, selected, starts, ends)
        return engine, prepared.run_snapshot
    return engine, lambda: engine.execute(graph, selected, starts, ends)


def value(result):
    if hasattr(result, "outputs"):
        return tuple(out.values[0] for out in result.outputs)
    return result if isinstance(result, tuple) else (result,)


def samples(function, loops):
    last = None
    start = time.perf_counter_ns()
    for _ in range(loops):
        last = function()
    return (time.perf_counter_ns() - start) / loops, last


def memory_child(backend, library, extended=False):
    from benchmark_memory import measure_heap, measure_rss

    from calmetrics_engine import operators  # noqa: F401

    reports = []
    for case in cases(extended):
        if backend == "direct" and case["kernel"] == "array":
            continue
        data = inputs(case)
        if backend == "njit":
            _, warm = setup(case, backend, data)
            warm()
            del warm
        if library:
            owner, function, report = measure_heap(
                lambda case=case, data=data: setup(case, backend, data), library
            )
        else:
            owner, function = setup(case, backend, data)
            for _ in range(6):
                function()
            report = dict(setup_heap=None, cold_heap=None, warm_heap=None)
        try:
            report.update(measure_rss(function))
            result = function()
            report.update(case=case, backend=backend,
                          audit=result.audit if hasattr(result, "audit") else None)
            del result
            reports.append(report)
        finally:
            del function
            if owner:
                owner.close()
    print(json.dumps(reports))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heap-probe", type=Path)
    parser.add_argument("--extended", action="store_true", help="Retain original cases and add 128x128 matrices")
    parser.add_argument("--memory-child", choices=["njit", "ordinary", "prepared", "direct"])
    args = parser.parse_args()
    if args.memory_child:
        memory_child(args.memory_child, str(args.heap_probe) if args.heap_probe else None, args.extended)
        return
    if os.environ.get("DYLD_INSERT_LIBRARIES"):
        raise RuntimeError("timing must not use allocator instrumentation")
    from threadpoolctl import threadpool_info, threadpool_limits

    from calmetrics_engine import build_info

    output = {"build": build_info(), "cpu_budget": 1, "extended": args.extended, "repeats": 31, "cases": [], "memory": {}}
    with threadpool_limits(limits=1):
        for case in cases(args.extended):
            data = inputs(case)
            _, reference = setup(case, "njit", data)
            expected = reference()
            for backend in ["ordinary", "prepared"] + (
                [] if case["kernel"] == "array" else ["direct"]
            ):
                owner, function = setup(case, backend, data)
                result = function()
                parity_error = None
                try:
                    for actual, target in zip(value(result), value(expected), strict=True):
                        np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-9)
                except AssertionError as error:
                    # Preserve the failed comparison and finish the remaining
                    # workloads. Timings of a mismatched result cannot pass.
                    parity_error = str(error)
                del result
                pilot, _ = samples(reference, 5)
                loops = max(5, min(1000, int(2e6 / max(pilot, 1))))
                a = []
                b = []
                for i in range(31):
                    functions = [function, reference] if i % 2 == 0 else [reference, function]
                    first, _ = samples(functions[0], loops)
                    second, _ = samples(functions[1], loops)
                    a.append(first if i % 2 == 0 else second)
                    b.append(second if i % 2 == 0 else first)
                ratio = paired_ratio(a, b)
                ratio.update({"case": case, "backend": backend,
                              "parity_pass": parity_error is None,
                              "parity_error": parity_error,
                              "time_pass": parity_error is None and ratio["upper95"] < 1})
                output["cases"].append(ratio)
                if owner:
                    owner.close()
            print(case["name"], flush=True)
        output["threadpools"] = threadpool_info()
    for backend in ("njit", "ordinary", "prepared", "direct"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--memory-child",
            backend,
            "--output",
            str(args.output),
            *(["--extended"] if args.extended else []),
        ]
        env = dict(os.environ)
        if args.heap_probe:
            command += ["--heap-probe", str(args.heap_probe.resolve())]
            env["DYLD_INSERT_LIBRARIES"] = str(args.heap_probe.resolve())
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        output["memory"][backend] = json.loads(result.stdout)
    baseline = {v["case"]["name"]: v for v in output["memory"]["njit"]}
    for row in output["cases"]:
        name = row["case"]["name"]
        native = next(v for v in output["memory"][row["backend"]] if v["case"]["name"] == name)
        njit = baseline[name]
        row["rss_pass"] = 0 < native["rss_delta"] < njit["rss_delta"]
        row["heap_pass"] = bool(
            args.heap_probe
            and native["cold_peak_bytes"] < njit["cold_peak_bytes"]
            and native["warm_peak_bytes"] < njit["warm_peak_bytes"]
        )
        row["joint_pass"] = row["time_pass"] and row["rss_pass"] and row["heap_pass"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "comparisons": len(output["cases"]),
                "parity_pass": sum(r["parity_pass"] for r in output["cases"]),
                "time_pass": sum(r["time_pass"] for r in output["cases"]),
                "joint_pass": sum(r["joint_pass"] for r in output["cases"]),
            }
        )
    )
    if any(not r["parity_pass"] for r in output["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
