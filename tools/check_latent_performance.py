"""Warm single-core latent-kernel parity, paired timing and independent heap/RSS.

Explicit initial parameters on BOTH sides; no platform initialization charged
only to NJIT. Full platform pipeline parity is covered in test_latent_operators.
Direct, ordinary DAG and prepared snapshot keep independent result ownership.
The benchmark reuses the existing strict upper95<1 time and heap/RSS gates.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[key] = "1"
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from check_foundation_performance import paired_ratio
from check_matrix_array_performance import samples, value
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import latent_reference as ref


@njit(cache=True, nogil=True)
def filter_reference(em, initial, transition):
    n, k = em.shape
    result = np.empty_like(em)
    for i in range(n):
        maximum = np.max(em[i])
        total = 0.0
        for t in range(k):
            predicted = initial[t] if i == 0 else 0.0
            if i:
                for s in range(k):
                    predicted += result[i - 1, s] * transition[s, t]
            result[i, t] = predicted * np.exp(em[i, t] - maximum)
            total += result[i, t]
        for t in range(k):
            result[i, t] /= max(total, 1e-300)
    return result


@njit(cache=True, nogil=True)
def axis_reference(x, mode):
    n, d = x.shape
    out = np.empty(d if mode == 0 else n)
    for i in range(out.size):
        if mode == 0:
            mean = 0.0
            for j in range(n):
                mean += x[j, i]
            mean /= n
            total = 0.0
            for j in range(n):
                delta = x[j, i] - mean
                total += delta * delta
            out[i] = total / n
        else:
            maximum = x[i, 0]
            for j in range(1, d):
                maximum = max(maximum, x[i, j])
            out[i] = maximum
            if mode == 2:
                total = 0.0
                for j in range(d):
                    total += np.exp(x[i, j] - maximum)
                out[i] += np.log(total)
    return out


@njit(cache=True, nogil=True)
def argmax_reference(x):
    out = np.empty(x.shape[0], np.int64)
    for i in range(x.shape[0]):
        winner = 0
        for j in range(1, x.shape[1]):
            if x[i, j] > x[i, winner]:
                winner = j
        out[i] = winner
    return out


@njit(cache=True, nogil=True)
def softmax_reference(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        z = ref._logsumexp_vector(x[i])
        for j in range(x.shape[1]):
            out[i, j] = np.exp(x[i, j] - z)
    return out


@njit(cache=True, nogil=True)
def code_reference(codes, k, one_hot):
    out = np.zeros((codes.size if one_hot else k, k))
    for i in range(codes.size):
        if codes[i] < 0:
            continue
        if one_hot:
            out[i, codes[i]] = 1.0
        elif i > 0 and codes[i - 1] >= 0:
            out[codes[i - 1], codes[i]] += 1.0
    return out


@njit(cache=True, nogil=True)
def partition_reference(x, order, k):
    n, d = x.shape
    out = np.empty((k, d))
    for c in range(k):
        start, end = n * c // k, n * (c + 1) // k
        for j in range(d):
            total = 0.0
            for i in range(start, end):
                total += x[order[i], j]
            out[c, j] = total / (end - start)
    return out


@njit(cache=True, nogil=True)
def sample_reference(x, k, seed):
    state = seed % 2147483646 + 1
    out = np.empty(k, np.int64)
    for c in range(k):
        state = state * 48271 % 2147483647
        candidate = state % x.shape[0]
        duplicate = True
        while duplicate:
            duplicate = False
            for prior in range(c):
                if out[prior] == candidate:
                    candidate = (candidate + 1) % x.shape[0]
                    duplicate = True
                    break
        out[c] = candidate
    return out


@njit(cache=True, nogil=True)
def scatter_reference(x, indices, length):
    out = np.full((length, x.shape[1]), np.nan)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[indices[i], j] = x[i, j]
    return out


PRIMITIVES = (
    "variance_axis",
    "max_axis",
    "argmax_axis",
    "logsumexp_axis",
    "softmax_axis",
    "transition_counts",
    "one_hot",
    "partition_mean",
    "sample_unique_indices",
    "scatter_rows",
    "standardize",
)


def cases(suite="all", extended=False):
    kernels = ("density", "gmm", "hmm", "filter", "smooth")
    if suite == "primitives":
        kernels = PRIMITIVES
    elif suite == "all":
        kernels += PRIMITIVES
    return [
        {
            "name": f"{kernel}_{n}",
            "kernel": kernel,
            "n": n,
            "d": d,
            "k": k,
            "iterations": 12,
        }
        for kernel in kernels
        for n, d, k in ([(16, 2, 2), (252, 5, 3), (2048, 8, 4)]
                         + ([(32768, 8, 4)] if extended else []))
    ]


def inputs(case):
    rng = np.random.default_rng(20260927)
    n, d, k = case["n"], case["d"], case["k"]
    x = rng.normal(size=(n, d)) + (np.arange(n) % k)[:, None] * 2
    m = x[:k].copy()
    v = np.tile(x.var(axis=0), (k, 1))
    w = np.full(k, 1 / k)
    a = np.full((k, k), 0.1 / (k - 1))
    np.fill_diagonal(a, 0.9)
    em = ref._gaussian_log_density_impl(x, m, v)
    result = {"x": x, "m": m, "v": v, "w": w, "a": a, "em": em}
    result["order"] = np.arange(n - 1, -1, -1, dtype=np.int64)
    result["indices"] = np.arange(n, dtype=np.int64) * 2
    result["codes"] = np.arange(n, dtype=np.int64) % k
    result["codes"][7::11] = -1
    for array in result.values():
        array.flags.writeable = False
    return result


def primitive_setup(case, backend, data):
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    from calmetrics_engine import operators as op

    name, k, n = case["kernel"], case["k"], case["n"]
    x = data["x"]
    if backend == "njit":
        if name in ("variance_axis", "max_axis", "logsumexp_axis"):
            mode = ("variance_axis", "max_axis", "logsumexp_axis").index(name)
            return None, lambda: axis_reference(x, mode)
        if name == "argmax_axis":
            return None, lambda: argmax_reference(x)
        if name == "softmax_axis":
            return None, lambda: softmax_reference(x)
        if name in ("transition_counts", "one_hot"):
            return None, lambda: code_reference(data["codes"], k, name == "one_hot")
        if name == "partition_mean":
            return None, lambda: partition_reference(x, data["order"], k)
        if name == "sample_unique_indices":
            return None, lambda: sample_reference(x, k, -3)
        if name == "scatter_rows":
            return None, lambda: scatter_reference(x, data["indices"], n * 2)
        return None, lambda: ref.standardize_fit_kernel(x)

    if name == "standardize":

        def direct():
            mean = op.mean_axis(x, 0)
            raw = op.sqrt(op.variance_axis(x, 0))
            scale = op.where(op.less_than(raw, 1e-10), 1.0, raw)
            out = op.divide(
                op.subtract(x, op.broadcast_like(mean, x)), op.broadcast_like(scale, x)
            )
            return out, mean, scale

        args, keys = [], ["x"]
        mean = "mean_axis(x,0)"
        raw = "sqrt(variance_axis(x,0))"
        scale = f"where({raw}<1e-10,1,{raw})"
        expression = [
            f"(x-broadcast_like({mean},x))/broadcast_like({scale},x)",
            mean,
            scale,
        ]
    else:
        keys = ["codes"] if name in ("transition_counts", "one_hot") else ["x"]
        if name == "partition_mean":
            keys += ["order"]
        if name == "scatter_rows":
            keys += ["indices"]
        options = {
            "variance_axis": [0],
            "max_axis": [1],
            "argmax_axis": [1],
            "logsumexp_axis": [1],
            "softmax_axis": [1],
            "transition_counts": [k],
            "one_hot": [k],
            "partition_mean": [k],
            "sample_unique_indices": [k, -3],
            "scatter_rows": [n * 2],
        }[name]
        args = [data[key] for key in keys] + options
        expression = f"{name}({','.join(keys + [str(v) for v in options])})"

        def direct():
            return getattr(op, name)(*args)

    if backend == "direct":
        return None, direct
    types = {
        key: (
            {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]}
            if key == "x"
            else {
                "kind": "vector",
                "dtype": "int64",
                "axes": ["scenario"],
                "shape": ["N"],
            }
        )
        for key in keys
    }
    graph = GraphCompiler(types).compile(expression)
    engine = AdaptiveScheduler(cpu_budget=1)
    starts = ends = np.zeros(1, np.int64)
    selected = {key: data[key] for key in keys}
    if backend == "prepared":
        prepared = engine.prepare_execution(graph, selected, starts, ends)
        return engine, prepared.run_snapshot
    return engine, lambda: engine.execute(graph, selected, starts, ends)


def setup(case, backend, data):
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    from calmetrics_engine import operators as op

    kernel = case["kernel"]
    if kernel in PRIMITIVES:
        return primitive_setup(case, backend, data)
    iterations = case["iterations"]
    if backend == "njit":
        if kernel == "density":
            return None, lambda: ref._gaussian_log_density_impl(
                data["x"], data["m"], data["v"]
            )
        if kernel == "gmm":
            return None, lambda: ref.gmm_em_reference(
                data["x"], data["w"], data["m"], data["v"], iterations, 0.0
            )
        if kernel == "hmm":
            return None, lambda: ref.hmm_em_reference(
                data["x"], data["w"], data["a"], data["m"], data["v"], iterations, 0.0
            )
        if kernel == "filter":
            return None, lambda: filter_reference(data["em"], data["w"], data["a"])
        return None, lambda: ref.smooth_emission_reference(
            data["em"], data["w"], data["a"]
        )
    definitions = {
        "x": {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]},
        "m": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "v": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "w": {"kind": "vector", "axes": ["component"], "shape": ["K"]},
        "a": {
            "kind": "matrix",
            "axes": ["component", "component"],
            "shape": ["K", "K"],
        },
        "em": {
            "kind": "matrix",
            "axes": ["scenario", "component"],
            "shape": ["N", "K"],
        },
    }
    names = {
        "density": ("gaussian_log_density", ["x", "m", "v"]),
        "gmm": ("gaussian_mixture_em", ["x", "w", "m", "v"]),
        "hmm": ("gaussian_hmm_em", ["x", "w", "a", "m", "v"]),
        "filter": ("hmm_filter", ["em", "w", "a"]),
        "smooth": ("hmm_smooth", ["em", "w", "a"]),
    }
    name, keys = names[kernel]
    args = [data[k] for k in keys]
    projections = (
        ["mixture_weights", "mixture_means", "mixture_variances"]
        if kernel == "gmm"
        else ["hmm_initial", "hmm_transition", "hmm_means", "hmm_variances"]
    )
    projections += ["em_iterations", "em_log_likelihood"]
    if backend == "direct":
        if kernel not in ("gmm", "hmm"):
            return None, lambda: getattr(op, name)(*args)

        def call():
            model = getattr(op, name)(*args, iterations, 0.0)
            return tuple(getattr(op, p)(model) for p in projections)

        return None, call
    expression = f"{name}({','.join(keys)}" + (
        f",{iterations},0)" if kernel in ("gmm", "hmm") else ")"
    )
    if kernel in ("gmm", "hmm"):
        expression = [f"{p}({expression})" for p in projections]
    graph = GraphCompiler({k: definitions[k] for k in keys}).compile(expression)
    engine = AdaptiveScheduler(cpu_budget=1)
    starts = ends = np.zeros(1, np.int64)
    selected = {k: data[k] for k in keys}
    if backend == "prepared":
        prepared = engine.prepare_execution(graph, selected, starts, ends)
        return engine, prepared.run_snapshot
    return engine, lambda: engine.execute(graph, selected, starts, ends)


def memory(backend, library, suite, extended=False):
    from benchmark_memory import measure_heap, measure_rss

    from calmetrics_engine import operators  # noqa: F401

    reports = []
    for case in cases(suite, extended):
        data = inputs(case)
        if backend == "njit":
            _, warm = setup(case, backend, data)
            warm()
            del warm
        owner, function, report = measure_heap(
            lambda case=case, data=data: setup(case, backend, data), library
        )
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
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heap-probe", type=Path, required=True)
    parser.add_argument("--extended", action="store_true", help="Retain original cases and add 32768-row workloads")
    parser.add_argument(
        "--suite", choices=["all", "models", "primitives"], default="all"
    )
    parser.add_argument(
        "--memory-child", choices=["njit", "ordinary", "prepared", "direct"]
    )
    args = parser.parse_args()
    if args.memory_child:
        print(json.dumps(memory(args.memory_child, str(args.heap_probe), args.suite, args.extended)))
        return
    if os.environ.get("DYLD_INSERT_LIBRARIES"):
        raise RuntimeError("timing cannot use allocation instrumentation")
    from threadpoolctl import threadpool_info, threadpool_limits

    from calmetrics_engine import build_info

    report = {
        "build": build_info(),
        "cpu_budget": 1,
        "repeats": 31,
        "initialization": "explicit on both sides",
        "suite": args.suite,
        "extended": args.extended,
        "cases": [],
        "memory": {},
    }
    with threadpool_limits(limits=1):
        for case in cases(args.suite, args.extended):
            data = inputs(case)
            _, reference = setup(case, "njit", data)
            expected = reference()
            for backend in ("direct", "ordinary", "prepared"):
                owner, function = setup(case, backend, data)
                result = function()
                for actual, target in zip(value(result), value(expected), strict=True):
                    np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-9)
                del result
                pilot, _ = samples(reference, 3)
                loops = max(2, min(1000, int(2e6 / max(pilot, 1))))
                native = []
                baseline = []
                for i in range(31):
                    functions = (
                        [function, reference] if i % 2 == 0 else [reference, function]
                    )
                    first, _ = samples(functions[0], loops)
                    second, _ = samples(functions[1], loops)
                    native.append(first if i % 2 == 0 else second)
                    baseline.append(second if i % 2 == 0 else first)
                row = paired_ratio(native, baseline)
                row.update(case=case, backend=backend, time_pass=row["upper95"] < 1)
                report["cases"].append(row)
                if owner:
                    owner.close()
            print(case["name"], flush=True)
        report["threadpools"] = threadpool_info()
    for backend in ("njit", "direct", "ordinary", "prepared"):
        env = dict(os.environ, DYLD_INSERT_LIBRARIES=str(args.heap_probe.resolve()))
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--memory-child",
                backend,
                "--heap-probe",
                str(args.heap_probe.resolve()),
                "--suite",
                args.suite,
                "--output",
                str(args.output),
                *(["--extended"] if args.extended else []),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        report["memory"][backend] = json.loads(child.stdout)
    for row in report["cases"]:
        name = row["case"]["name"]
        native = next(
            r for r in report["memory"][row["backend"]] if r["case"]["name"] == name
        )
        njit = next(r for r in report["memory"]["njit"] if r["case"]["name"] == name)
        row["rss_pass"] = 0 < native["rss_delta"] < njit["rss_delta"]

        def cold(r):
            return r["cold_peak_bytes"]

        def warm(r):
            return r["warm_peak_bytes"]

        row.update(
            heap_pass=cold(native) < cold(njit) and warm(native) < warm(njit),
            cold_heap_ratio=cold(native) / max(cold(njit), 1),
            warm_heap_ratio=warm(native) / max(warm(njit), 1),
        )
        row["joint_pass"] = row["time_pass"] and row["heap_pass"] and row["rss_pass"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "comparisons": len(report["cases"]),
                "time_pass": sum(r["time_pass"] for r in report["cases"]),
                "heap_pass": sum(r["heap_pass"] for r in report["cases"]),
                "joint_pass": sum(r["joint_pass"] for r in report["cases"]),
            }
        )
    )


if __name__ == "__main__":
    main()
