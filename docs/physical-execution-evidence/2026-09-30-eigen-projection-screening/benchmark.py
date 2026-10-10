"""Full Jacobi projection roots: exact old/new, warmed same-core NJIT/LAPACK.

Six balanced rounds retain previous independent results. Library eigenvalues
are a performance reference; LAPACK does not implement bounded Jacobi errors.
Heap and RSS run in separate fresh children, including setup and retained results.
"""

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

ROOT = Path("/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine")
INSTALLS = {
    "old": "/private/tmp/calmetrics-nonnegative-final-install",
    "new": "/private/tmp/calmetrics-eigen-final-install",
    "njit": "/private/tmp/calmetrics-nonnegative-final-install",
}
IDENTITIES = {
    "old": "4df196c9a2c64c984bb940754de488099b9ab2ce1c51253b5d3a8dfdce476f1e",
    "new": "33c2f2b95e374d396dc283c78d2ca5a43ad51a7a97995c3534ffdbae61ff7d39",
}
BACKENDS = ("old", "new", "njit")
PROBE = "/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib"
REFERENCE = None


def cases():
    return [dict(name=name, shape=[n, n, n], layout=layout, full=full,
                 entry=entry, domain="jacobi")
            for name, n, layout, full, entry in (
                ("values_512_c", 512, "C", False, "ordinary"),
                ("values_1024_c", 1024, "C", False, "ordinary"),
                ("values_1024_f", 1024, "F", False, "ordinary"),
                ("full_1024_c", 1024, "C", True, "ordinary"),
                ("values_1024_prepared", 1024, "C", False, "prepared"),
                ("descending_full_2048", 2048, "C", True, "ordinary"),
                ("small_values_16", 16, "C", False, "ordinary"),
            )]


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def inputs(case):
    n = case["shape"][0]
    x = np.random.default_rng(903035 + n).normal(size=(n, n))
    a = np.diag(np.arange(n, 0, -1, dtype=float)) if "descending" in case["name"] else x + x.T
    a = np.array(a, order=case["layout"])
    a.flags.writeable = False
    return a, np.empty(0)


def setup(case, backend, a, b):
    global REFERENCE
    if backend == "njit":
        from numba import njit, set_num_threads, typeof
        set_num_threads(1)
        if REFERENCE is None:
            @njit(nogil=True)
            def reference(a, full):
                for i in range(len(a)):
                    for j in range(len(a)):
                        if not np.isfinite(a[i, j]) or a[i, j] != a[j, i]:
                            raise ValueError("invalid matrix")
                if full:
                    w, v = np.linalg.eigh(a)
                    vectors = np.empty(v.shape)
                    for j in range(len(a)):
                        pivot = 0
                        for i in range(1, len(a)):
                            if abs(v[i, j]) > abs(v[pivot, j]):
                                pivot = i
                        sign = -1.0 if v[pivot, j] < 0 else 1.0
                        for i in range(len(a)):
                            vectors[i, j] = sign * v[i, j]
                    return w, vectors
                return np.linalg.eigvalsh(a), np.empty((0, 0))
            REFERENCE = reference
        signature = (typeof(a), typeof(case["full"]))
        if signature not in REFERENCE.signatures:
            REFERENCE.compile(signature)

        def run():
            w, v = REFERENCE(a, case["full"])
            w.flags.writeable = v.flags.writeable = False
            return None, w, v
        return None, run
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    decl = {"a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"])}
    expression = ["eigenvalues(symmetric_eigh(a))"]
    if case["full"]:
        expression.append("eigenvectors(symmetric_eigh(a))")
    graph = GraphCompiler(decl).compile(expression)
    scheduler = AdaptiveScheduler(cpu_budget=1)
    bound = {"a": a}
    zeros = np.zeros(1, np.int64)
    if case["entry"] == "prepared":
        prepared = scheduler.prepare_execution(graph, bound, zeros, zeros)
        function = prepared.run_snapshot
    else:
        def function():
            return scheduler.execute(graph, bound, zeros, zeros)

    def run():
        result = function()
        return result, result.outputs[0].values[0], result.outputs[1].values[0] if case["full"] else np.empty((0, 0))
    return scheduler, run


def verify(case, backend, result, a, b):
    matrix = result[1]
    n = case["shape"][0]
    assert matrix.shape == (n,)
    assert matrix.dtype == np.float64 and not matrix.flags.writeable
    assert not np.shares_memory(matrix, a)
    assert matrix.flags.c_contiguous
    if case["full"]:
        vectors = result[2]
        assert vectors.shape == (n, n) and not vectors.flags.writeable
        assert not np.shares_memory(vectors, a) and not np.shares_memory(vectors, matrix)
    pools = threadpool_info()
    assert all(pool["num_threads"] == 1 for pool in pools)
    identity = dict(threadpools=pools)
    if backend != "njit":
        identity["engine_build_id"] = result[0].audit["engine_build_id"]
        assert identity["engine_build_id"] == IDENTITIES[backend]
    else:
        import numba
        identity["numba_version"] = numba.__version__
    return dict(sha256=digest(matrix), identity=identity,
                vectors_sha256=digest(result[2]) if case["full"] else None,
                c_contiguous=bool(matrix.flags.c_contiguous),
                f_contiguous=bool(matrix.flags.f_contiguous))


def environment(backend, heap=False):
    env = dict(os.environ, PYTHONPATH=INSTALLS[backend])
    env.pop("DYLD_INSERT_LIBRARIES", None)
    if heap:
        env["DYLD_INSERT_LIBRARIES"] = PROBE
    return env


def worker(args):
    owner = retained = None
    with threadpool_limits(1):
        try:
            for line in sys.stdin:
                command = json.loads(line)
                if command["op"] == "stop":
                    print(json.dumps(dict(stopped=True)), flush=True)
                    break
                if command["op"] == "init":
                    if owner:
                        owner.close()
                    retained = None
                    case = command["case"]
                    a, b = inputs(case)
                    before = [digest(a), digest(b)]
                    owner, function = setup(case, args.worker, a, b)
                    retained = function()
                    response = verify(case, args.worker, retained, a, b)
                    assert before == [digest(a), digest(b)]
                    response["input_sha256"] = before
                    np.save(args.arrays / (args.worker + ".npy"), retained[1])
                else:
                    wall, cpu = [], []
                    for _ in range(command["repeat"]):
                        started = time.perf_counter_ns()
                        cpu_started = time.process_time_ns()
                        retained = function()
                        cpu.append(time.process_time_ns() - cpu_started)
                        wall.append(time.perf_counter_ns() - started)
                    response = dict(wall_ns=wall, cpu_ns=cpu,
                                    audit=retained[0].audit if retained[0] else {})
                print(json.dumps(response), flush=True)
        finally:
            if owner:
                owner.close()


def timing(args):
    children, logs, records = {}, [], []

    def send(key, command):
        child = children[key]
        child.stdin.write(json.dumps(command) + "\n")
        child.stdin.flush()
        line = child.stdout.readline()
        if not line:
            raise RuntimeError(f"{key} worker failed; inspect stderr log")
        return json.loads(line)

    with tempfile.TemporaryDirectory(prefix="cme-eigen-evidence-") as arrays_dir:
        try:
            for backend in BACKENDS:
                log = (args.output / (backend + ".log")).open("x")
                logs.append(log)
                children[backend] = subprocess.Popen(
                    [sys.executable, __file__, "--worker", backend, "--arrays", arrays_dir],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                    env=environment(backend))
            for case in cases():
                identities = {key: send(key, dict(op="init", case=case)) for key in BACKENDS}
                assert len({tuple(v["input_sha256"]) for v in identities.values()}) == 1
                values = {key: np.load(Path(arrays_dir) / (key + ".npy")) for key in BACKENDS}
                np.testing.assert_array_equal(values["old"].view(np.uint64), values["new"].view(np.uint64))
                if case["full"]:
                    assert identities["old"]["vectors_sha256"] == identities["new"]["vectors_sha256"]
                njit_close = np.isclose(values["njit"], values["old"], rtol=1e-9, atol=1e-9)
                parity = dict(new_old_bitwise_passed=True,
                              njit_within_tolerance=bool(njit_close.all()),
                              njit_outside_tolerance=int(np.count_nonzero(~njit_close)),
                              njit_max_abs=float(np.max(np.abs(values["njit"] - values["old"]))))
                del values
                samples = {key: dict(wall_ns=[], cpu_ns=[]) for key in BACKENDS}
                rounds, last = [], {}
                for turn in range(6):
                    order = BACKENDS[turn % 3:] + BACKENDS[:turn % 3]
                    row = dict(order=order, samples={})
                    for key in order:
                        last[key] = send(key, dict(op="run", repeat=15 if case["shape"][0] < 512 else 1))
                        row["samples"][key] = {field: last[key][field] for field in samples[key]}
                        for field in samples[key]:
                            samples[key][field].extend(last[key][field])
                    rounds.append(row)
                    print(case["name"], turn + 1, flush=True)
                medians = {key: statistics.median(v["wall_ns"]) / 1e6 for key, v in samples.items()}
                records.append(dict(case=case, identities=identities, parity=parity,
                                    samples=samples, rounds=rounds, median_ms=medians,
                                    audits={key: last[key]["audit"] for key in BACKENDS}))
                (args.output / "timing.json").write_text(json.dumps(records, indent=2))
                print(medians, flush=True)
        finally:
            for key, child in children.items():
                if child.poll() is None:
                    try:
                        send(key, dict(op="stop"))
                    except (BrokenPipeError, RuntimeError):
                        pass
                child.stdin.close()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                child.stdout.close()
            for log in logs:
                log.close()


def memory_child(args):
    sys.path.insert(0, str(ROOT / "tools"))
    from benchmark_memory import measure_heap, measure_rss
    case = json.loads(args.case)
    a, b = inputs(case)
    before = [digest(a), digest(b)]
    with threadpool_limits(1):
        if args.backend == "njit":
            _, warmed = setup(case, args.backend, a, b)
            warmed()  # compile and warm the exact readonly specialization before the epoch
        else:
            import calmetrics_engine  # noqa: F401
        owner = None
        try:
            if args.memory == "heap":
                owner, function, measured = measure_heap(
                    lambda: setup(case, args.backend, a, b), PROBE, repeats=2)
                assert measured["cold_peak_bytes"] >= (case["shape"][0] + (case["shape"][0]**2 if case["full"] else 0)) * 8, \
                    "result allocation not observed; interposition was not enabled"
            else:
                owner, function = setup(case, args.backend, a, b)
                measured = measure_rss(function, repeats=2)
            result = function()
            verified = verify(case, args.backend, result, a, b)
            assert before == [digest(a), digest(b)]
            report = dict(case=case, backend=args.backend, kind=args.memory,
                          measurement=measured, verification=verified,
                          audit=result[0].audit if result[0] else {})
        finally:
            if owner:
                owner.close()
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("timing", "memory"))
    parser.add_argument("--worker", choices=BACKENDS)
    parser.add_argument("--arrays", type=Path)
    parser.add_argument("--memory", choices=("heap", "rss"))
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.memory:
        return memory_child(args)
    args.output.mkdir(parents=True, exist_ok=True)
    design = dict(cases=cases(), cpu=1, rounds=6, identities=IDENTITIES,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  timing="full independent result, retained previous, same readonly signature",
                  heap="fresh process, setup+cold+warm, library allocations included; excludes stack/mmap",
                  rss="separate fresh child, monitor warmup and two complete calls; supplementary only")
    (args.output / "design.json").write_text(json.dumps(design, indent=2))
    if args.mode == "timing":
        return timing(args)
    records = []
    selected = [case for case in cases() if case["name"] in {
        "values_1024_c", "full_1024_c", "descending_full_2048"}]
    for case in selected:
        for backend in BACKENDS:
            for kind in ("heap", "rss"):
                command = [sys.executable, __file__, "--memory", kind, "--backend", backend,
                           "--case", json.dumps(case)]
                completed = subprocess.run(command, env=environment(backend, kind == "heap"),
                                           capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(completed.stderr)
                records.append(json.loads(completed.stdout))
                (args.output / "memory.json").write_text(json.dumps(records, indent=2))
                print(case["name"], backend, kind, flush=True)


if __name__ == "__main__":
    main()
