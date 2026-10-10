"""Compare installed AOT matrix builds with warmed, single-CPU NJIT.

Timing runs in separate, sequentially activated children, rotating first backend
for 11 rounds. Each call returns an independent result. Memory children include
graph setup and retained workspace; their inputs and NJIT compilation precede
the measurement epoch. Ordinary/prepared calls materialize every public output
field inside the measured callable, including ndarray/scalar wrapping. A bounded
holder retains the raw result and the exact projected values; verification never
reopens lazy result fields or flattens/copies arrays to establish ownership.
RSS is supplementary and is not an allocation counter.
"""

# Set limits before importing any numerical package.
# ruff: noqa: E402
from __future__ import annotations

import os

THREAD_ENV = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS", "MKL_NUM_THREADS",
)
for _key in THREAD_ENV:
    os.environ[_key] = "1"

import argparse
import ctypes
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from benchmark_memory import measure_heap, measure_rss

BACKENDS = ("old", "new", "njit")
ROUNDS = 11
_references = {}


def cases():
    result = [
        dict(name="cholesky", n=n, layout="c", entry="ordinary")
        for n in (63, 127, 128, 129, 256, 512, 1024)
    ]
    result += [
        dict(name="cholesky", n=512, layout=layout, entry="ordinary")
        for layout in ("f", "reverse")
    ]
    result += [
        dict(name="matvec", n=n, layout=layout, entry="ordinary")
        for n in (128, 512, 1024, 2048) for layout in ("c", "f")
    ]
    result += [
        dict(name="matvec", n=512, layout=layout, entry="ordinary")
        for layout in ("reverse", "sliced")
    ]
    result += [
        dict(name=name, n=n, layout="c", entry=entry)
        for name, n in (("cholesky", 512), ("matvec", 2048))
        for entry in ("direct", "prepared")
    ]
    result += [
        dict(name="matvec", n=32768, d=d, layout=layout, entry="ordinary")
        for d in (64, 128) for layout in ("c", "f")
    ]
    result += [
        dict(name="matvec", n=32768, d=64, layout="f", entry=entry)
        for entry in ("direct", "prepared")
    ]
    return result


def case_id(case):
    dimensions = f"{case['n']}x{case['d']}" if "d" in case else str(case["n"])
    return f"{case['name']}_{dimensions}_{case['layout']}_{case['entry']}"


def inputs(case):
    rng = np.random.default_rng(20260929)
    n, d = case["n"], case.get("d", case["n"])
    a = rng.normal(size=(n, d))
    if case["name"] == "cholesky":
        a = (a + a.T) * 0.5
        # Strict symmetric diagonal dominance produces a reproducible SPD input
        # without warming BLAS or allocating its internal cache before the epoch.
        a[np.diag_indices(n)] += np.abs(a).sum(axis=1) + 1.0
    if case["layout"] == "f":
        a = np.asfortranarray(a)
    elif case["layout"] == "reverse":
        a = a[::-1, ::-1]
    elif case["layout"] == "sliced":
        storage = np.zeros((n * 2, d * 2))
        storage[::2, ::2] = a
        a = storage[::2, ::2]
    data = {"a": a}
    if case["name"] == "matvec":
        b = rng.normal(size=d)
        data["b"] = b[::-1] if case["layout"] == "reverse" else b
    for array in data.values():
        array.flags.writeable = False
    return data


def input_identity(data):
    return {
        name: dict(shape=list(a.shape), strides=list(a.strides), dtype=str(a.dtype),
                   sha256=hashlib.sha256(a.tobytes(order="C")).hexdigest(),
                   writeable=a.flags.writeable)
        for name, a in data.items()
    }


@contextmanager
def baseline_threads(backend):
    """Constrain only the reference; native must implement its own CPU budget."""
    metadata = {"environment": {key: os.environ[key] for key in THREAD_ENV},
                "accelerate_thread_local": "not_applied_to_native"}
    if backend != "njit":
        yield metadata
        return
    from threadpoolctl import threadpool_info, threadpool_limits

    library = None
    prior = None
    with threadpool_limits(limits=1):
        try:
            if sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0]) >= 15:
                library = ctypes.CDLL(
                    "/System/Library/Frameworks/Accelerate.framework/Accelerate"
                )
                library.BLASGetThreading.argtypes = []
                library.BLASGetThreading.restype = ctypes.c_uint
                library.BLASSetThreading.argtypes = [ctypes.c_uint]
                library.BLASSetThreading.restype = ctypes.c_int
                prior = library.BLASGetThreading()
                metadata["accelerate_thread_local"] = {"prior": prior}
                if library.BLASSetThreading(1) != 0:
                    raise RuntimeError("Accelerate rejected single-threaded baseline")
                metadata["accelerate_thread_local"] = {
                    "prior": prior, "active": library.BLASGetThreading(),
                }
                if metadata["accelerate_thread_local"]["active"] != 1:
                    raise RuntimeError("Accelerate did not select single-threaded baseline")
            else:
                metadata["accelerate_thread_local"] = "unavailable_on_this_platform"
            # SciPy is explicitly loaded before threadpool introspection and NJIT.
            import scipy.linalg  # noqa: F401

            # Import may expose another pool. Apply the cap to that pool as well.
            with threadpool_limits(limits=1):
                metadata["pools"] = threadpool_info()
                yield metadata
        finally:
            if library is not None and prior is not None:
                restored = library.BLASSetThreading(prior)
                metadata["accelerate_thread_local"]["restored"] = library.BLASGetThreading()
                if restored != 0 or library.BLASGetThreading() != prior:
                    raise RuntimeError("Accelerate baseline thread setting was not restored")


def setup(case, backend, data):
    if backend == "njit":
        from numba import typeof

        if not _references:
            from numba import njit

            @njit(nogil=True)
            def cholesky(a):
                return np.linalg.cholesky(a)

            @njit(nogil=True)
            def matvec(a, b):
                return a @ b

            _references.update(cholesky=cholesky, matvec=matvec)
        reference = _references[case["name"]]
        args = tuple(data.values())
        signature = tuple(typeof(arg) for arg in args)
        if signature not in reference.signatures:
            # Compile the exact readonly/layout signature without executing a
            # numerical function: first-use BLAS allocations belong to the
            # memory epoch on both native and NJIT sides.
            reference.compile(signature)
        return None, lambda: reference(*args)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler, operators

    args = tuple(data.values())
    if case["entry"] == "direct":
        return None, lambda: operators.call(case["name"], *args)
    if case["name"] == "cholesky":
        types = {"a": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["N", "N"]}}
    else:
        types = {
            "a": {"kind": "matrix", "axes": ["time", "asset"], "shape": ["T", "D"]},
            "b": {"kind": "vector", "axes": ["asset"], "shape": ["D"]},
        }
    # A matvec over the declared time axis can select the legacy homogeneous
    # series result by default. Request the same independent typed result
    # contract explicitly on both installed native builds and both DAG entries.
    graph = GraphCompiler(types).compile(
        f"{case['name']}({','.join(data)})", result_format="typed"
    )
    bounds = np.zeros(1, np.int64), np.array([case["n"]], np.int64)
    scheduler = AdaptiveScheduler(cpu_budget=1)
    try:
        if case["entry"] == "prepared":
            prepared = scheduler.prepare_execution(graph, data, *bounds)
            return scheduler, lambda: MaterializedResult(prepared.run_snapshot())
        return scheduler, lambda: MaterializedResult(scheduler.execute(graph, data, *bounds))
    except BaseException:
        scheduler.close()
        raise


class MaterializedResult:
    """Keep one raw result and its declared outputs, with no numeric copies."""

    __slots__ = ("raw", "outputs")

    def __init__(self, raw):
        self.raw = raw
        # Executed inside the timed/heap/RSS callable. Accessing raw.outputs and
        # field.values creates the public wrappers that otherwise remain lazy.
        self.outputs = tuple(field.values[0] for field in raw.outputs)


def value(result):
    if isinstance(result, MaterializedResult):
        outputs = result.outputs
        return outputs[0] if len(outputs) == 1 else outputs
    if isinstance(result, np.ndarray | np.generic | int | float | bool):
        return result
    if isinstance(result, tuple | list):
        return tuple(result)
    raise TypeError("Scheduler outputs must be materialized inside the measured callable")


def output_arrays(result):
    """Expose each actual output without flattening or copying array storage."""
    outputs = value(result)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    # Immutable Python/NumPy scalars become rank-zero arrays only for external
    # verification. Existing ndarray views keep their real owner and strides.
    return tuple(np.asarray(output) for output in outputs)


def compare_outputs(actual, expected, case, *, exact=False):
    if len(actual) != len(expected):
        raise AssertionError("result output counts differ")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"output {index} shape/dtype differ")
        if exact or left.dtype.kind in "biu":
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(
                left, right, rtol=case.get("rtol", 1e-11), atol=case.get("atol", 1e-11)
            )


def load_outputs(path):
    with np.load(path, allow_pickle=False) as archive:
        names = [f"output_{index}" for index in range(len(archive.files))]
        if archive.files != names:
            raise AssertionError("unexpected saved output schema")
        return tuple(archive[name] for name in names)


def check_independent_outputs(previous, current, inputs):
    for before in previous:
        for after in current:
            assert not np.shares_memory(before, after)
        for source in inputs.values():
            assert not np.shares_memory(before, source)
    for after in current:
        for source in inputs.values():
            assert not np.shares_memory(after, source)


def audit(case, backend, data, result):
    if backend == "njit":
        return {}
    if case["entry"] == "direct":
        from calmetrics_engine import operators

        # Diagnostic call is outside the timed/allocation epoch, keeping the
        # ordinary direct API comparable to the other independent-result APIs.
        _, report = operators.call(case["name"], *data.values(), audit=True)
        return report
    return result.raw.audit


def build_identity(backend):
    if backend != "njit":
        from calmetrics_engine import build_info

        return build_info()
    import numba
    import scipy

    return dict(backend="numba-numpy-linalg-scipy-blas", numba=numba.__version__,
                numpy=np.__version__, scipy=scipy.__version__)


def invariants(case, data, actual):
    if case["name"] == "cholesky":
        np.testing.assert_array_equal(np.triu(actual, 1), 0.0)
        assert np.all(np.diag(actual) > 0)
        reconstructed = actual @ actual.T
        target = data["a"]
    else:
        reconstructed = actual
        target = data["a"] @ data["b"]
    np.testing.assert_allclose(reconstructed, target, rtol=1e-11, atol=1e-11)
    return {"max_absolute_residual": float(np.max(np.abs(reconstructed - target))),
            "relative_frobenius_residual": float(
                np.linalg.norm(reconstructed - target) / max(np.linalg.norm(target), 1e-300)
            )}


def worker(backend, output):
    owner = result = function = None
    with baseline_threads(backend) as threads:
        try:
            for line in sys.stdin:
                command = json.loads(line)
                if command["op"] == "stop":
                    break
                if command["op"] == "init":
                    result = function = None
                    if owner:
                        owner.close()
                    owner = None
                    case = command["case"]
                    data = inputs(case)
                    identity = input_identity(data)
                    owner, function = setup(case, backend, data)
                    first = function()  # Warm numerical execution outside timer.
                    first_outputs = output_arrays(first)
                    retained = tuple(array.copy() for array in first_outputs)
                    result = function()
                    actual = output_arrays(result)
                    compare_outputs(first_outputs, retained, case, exact=True)
                    compare_outputs(actual, retained, case, exact=True)
                    check_independent_outputs(first_outputs, actual, data)
                    del first, first_outputs, retained
                    assert input_identity(data) == identity
                    residuals = invariants(case, data, value(result))
                    np.savez(
                        output / f"{backend}.npz",
                        **{f"output_{index}": array for index, array in enumerate(actual)},
                    )
                    response = dict(build=build_identity(backend), input=identity,
                                    threads=threads, residuals=residuals,
                                    independent_results=True)
                    del actual
                elif command["op"] == "run":
                    wall, cpu = [], []
                    for _ in range(command["repeat"]):
                        start_cpu = time.process_time_ns()
                        start_wall = time.perf_counter_ns()
                        result = function()
                        wall.append(time.perf_counter_ns() - start_wall)
                        cpu.append(time.process_time_ns() - start_cpu)
                    response = dict(wall_ns=wall, cpu_ns=cpu,
                                    audit=audit(case, backend, data, result))
                else:
                    raise ValueError(f"Unknown worker command: {command['op']}")
                print(json.dumps(response), flush=True)
        finally:
            result = function = None
            if owner:
                owner.close()
    print(json.dumps({"stopped": True, "threads": threads}), flush=True)


def memory_child(args):
    case = json.loads(args.case)
    data = inputs(case)
    identity = input_identity(data)
    owner = function = None
    with baseline_threads(args.backend) as threads:
        if args.backend == "njit":
            _, compiled = setup(case, args.backend, data)
            # setup compiles the actual signature but makes no numerical call.
            del compiled
        else:
            import calmetrics_engine  # noqa: F401 -- imports are outside request work
        try:
            if args.memory == "heap":
                owner, function, report = measure_heap(
                    lambda: setup(case, args.backend, data), args.heap_probe, repeats=11
                )
            else:
                # The first measured call performs setup. Warm native execution
                # before this point could leave an uncounted TLS workspace.
                def cold_through_warm():
                    nonlocal owner, function
                    if function is None:
                        owner, function = setup(case, args.backend, data)
                    return function()

                report = measure_rss(cold_through_warm, repeats=11)
                report["rss_schema"] = "setup-through-warm-monitor-control-1"
            result = function()
            report.update(case=case, backend=args.backend, input=identity,
                          build=build_identity(args.backend), threads=threads,
                          audit=audit(case, args.backend, data, result))
            del result
            assert input_identity(data) == identity
        finally:
            function = None
            if owner:
                owner.close()
    print(json.dumps(report))


def child_env(args, backend, heap=False):
    install = args.old_install if backend == "old" else args.new_install
    env = dict(os.environ, PYTHONPATH=str(Path(install).resolve()))
    env.pop("DYLD_INSERT_LIBRARIES", None)
    env.pop("LD_PRELOAD", None)
    if heap:
        env["DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"] = str(
            Path(args.heap_probe).resolve()
        )
    return env


def timing(args, selected):
    processes, errors, records, restored = {}, [], [], {}

    def send(backend, command):
        child = processes[backend]
        child.stdin.write(json.dumps(command) + "\n")
        child.stdin.flush()
        line = child.stdout.readline()
        if not line:
            raise RuntimeError(f"{backend} worker failed; inspect {backend}.log")
        return json.loads(line)

    try:
        for backend in BACKENDS:
            error = (args.output / f"{backend}.log").open("w")
            errors.append(error)
            processes[backend] = subprocess.Popen(
                [sys.executable, __file__, "--worker", backend, "--output", str(args.output)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error,
                text=True, env=child_env(args, backend),
            )
        for case in selected:
            identities = {key: send(key, {"op": "init", "case": case}) for key in BACKENDS}
            arrays = {key: load_outputs(args.output / f"{key}.npz") for key in BACKENDS}
            for key in ("new", "njit"):
                assert identities[key]["input"] == identities["old"]["input"]
                compare_outputs(arrays[key], arrays["old"], case)
            samples = {key: {"wall_ns": [], "cpu_ns": []} for key in BACKENDS}
            rounds, last = [], {}
            for iteration in range(ROUNDS):
                offset = iteration % len(BACKENDS)
                order = BACKENDS[offset:] + BACKENDS[:offset]
                current = {"order": list(order), "samples": {}}
                for key in order:
                    last[key] = send(key, {"op": "run", "repeat": 5 if case["n"] < 256 else 1})
                    current["samples"][key] = {name: last[key][name] for name in samples[key]}
                    for name in samples[key]:
                        samples[key][name].extend(last[key][name])
                rounds.append(current)
            medians = {key: statistics.median(v["wall_ns"]) / 1e6 for key, v in samples.items()}
            records.append(dict(case=case, rounds=rounds, samples=samples,
                                median_ms=medians, identities=identities,
                                audits={key: v["audit"] for key, v in last.items()},
                                correctness=True, cpu_budget=1))
            (args.output / "timing.json").write_text(json.dumps(records, indent=2))
            print(case_id(case), medians, flush=True)
    finally:
        for backend, child in processes.items():
            try:
                if child.poll() is None:
                    restored[backend] = send(backend, {"op": "stop"})
            except (BrokenPipeError, OSError, RuntimeError, ValueError) as error:
                restored[backend] = {"cleanup_error": str(error)}
            finally:
                try:
                    child.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    restored[backend] = {"cleanup_error": "worker required termination"}
                child.stdout.close()
        for error in errors:
            error.close()
        (args.output / "timing-worker-cleanup.json").write_text(json.dumps(restored, indent=2))


def memory_all(args, selected):
    reports = []
    for case in selected:
        for backend in BACKENDS:
            for kind in ("heap", "rss"):
                child = subprocess.run(
                    [sys.executable, __file__, "--memory", kind, "--backend", backend,
                     "--case", json.dumps(case), "--heap-probe", args.heap_probe,
                     "--output", str(args.output)],
                    env=child_env(args, backend, heap=kind == "heap"),
                    capture_output=True, text=True,
                )
                label = f"{case_id(case)}-{backend}-{kind}"
                (args.output / f"{label}.log").write_text(child.stderr)
                child.check_returncode()
                report = json.loads(child.stdout)
                reports.append(dict(kind=kind, **report))
                (args.output / "memory.json").write_text(json.dumps(reports, indent=2))
                print(label, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-install", default="/private/tmp/calmetrics-covariance-install")
    parser.add_argument("--new-install", default="/private/tmp/calmetrics-large-install")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("timing", "memory", "all"), default="timing")
    parser.add_argument("--heap-probe")
    parser.add_argument("--case-filter", help="Optional substring of a documented case id")
    parser.add_argument("--worker", choices=BACKENDS)
    parser.add_argument("--memory", choices=("heap", "rss"))
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--case", help="JSON case for a dedicated memory child")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, args.output)
        return
    if args.memory:
        if not args.backend or not args.case or (args.memory == "heap" and not args.heap_probe):
            parser.error("memory child requires --backend, --case and --heap-probe for heap")
        memory_child(args)
        return
    if args.mode in ("memory", "all") and not args.heap_probe:
        parser.error("--mode memory/all requires --heap-probe")
    if os.environ.get("DYLD_INSERT_LIBRARIES") or os.environ.get("LD_PRELOAD"):
        parser.error("run the parent without allocation instrumentation")
    selected = [case for case in cases() if not args.case_filter or args.case_filter in case_id(case)]
    if not selected:
        parser.error("--case-filter selected no cases")
    if args.mode in ("timing", "all"):
        timing(args, selected)
    if args.mode in ("memory", "all"):
        memory_all(args, selected)


if __name__ == "__main__":
    main()
