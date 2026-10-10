"""Full independent Scheduler roots, baseline wheel and warmed same-core NJIT.

Six balanced round-robin timing rounds retain the preceding result. Heap and
RSS use separate fresh children; setup and two retained warm results are in the
heap epoch. This is a local evidence harness, not a production benchmark API.
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
    "old": "/private/tmp/calmetrics-transpose-wheel-install",
    "new": "/private/tmp/calmetrics-nonnegative-final-install",
    "njit": "/private/tmp/calmetrics-nonnegative-final-install",
}
IDENTITIES = {
    "old": "f8509c481644af890b14c20936092d781d8f3dda388f02c203436a47e3a98264",
    "new": "4df196c9a2c64c984bb940754de488099b9ab2ce1c51253b5d3a8dfdce476f1e",
}
BACKENDS = ("old", "new", "njit")
PROBE = "/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib"
REFERENCE = None


def cases():
    def case(name, shape, layout, transpose=False, output="auto", entry="ordinary",
             domain="positive"):
        return dict(name=name, shape=shape, layout=layout, transpose=transpose,
                    output=output, entry=entry, domain=domain)
    return [
        case("square_cc", [2048, 2048, 2048], "cc"),
        case("square_ff_transpose", [2048, 2048, 2048], "ff", True),
        case("square_ff_explicit_f", [2048, 2048, 2048], "ff", output="F"),
        case("wide_ff_transpose", [4096, 512, 4096], "ff", True),
        case("high_k_cf", [513, 4096, 515], "cf"),
        case("zero_cc", [2048, 2048, 2048], "cc", domain="zero"),
        case("mixed_ff_transpose", [2048, 2048, 2048], "ff", True, domain="mixed"),
        case("small_cc", [32, 32, 32], "cc"),
        case("prepared_ff_transpose", [2048, 2048, 2048], "ff", True, "F", "prepared"),
    ]


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def inputs(case):
    m, k, n = case["shape"]
    rng = np.random.default_rng(903021 + m + k + n)
    generate = rng.normal if case["domain"] == "mixed" else rng.random
    a = np.array(generate(size=(m, k)), order=case["layout"][0].upper())
    b = np.array(generate(size=(k, n)), order=case["layout"][1].upper())
    if case["domain"] == "zero":
        a.fill(0.0)
    a.flags.writeable = b.flags.writeable = False
    return a, b


def setup(case, backend, a, b):
    global REFERENCE
    if backend == "njit":
        from numba import njit, set_num_threads, typeof
        set_num_threads(1)
        if REFERENCE is None:
            @njit(nogil=True)
            def reference(a, b):
                return a @ b
            REFERENCE = reference
        signature = (typeof(a), typeof(b))
        if signature not in REFERENCE.signatures:
            REFERENCE.compile(signature)

        def run():
            value = REFERENCE(a, b)
            if case["transpose"]:
                value = value.T
            order = case["output"]
            if order == "auto":
                order = "F" if case["transpose"] else "C"
            value = np.asfortranarray(value) if order == "F" else np.ascontiguousarray(value)
            value.flags.writeable = False
            return None, value
        return None, run
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    decl = {
        "a": dict(kind="matrix", axes=["asset", "asset"], shape=["M", "K"]),
        "b": dict(kind="matrix", axes=["asset", "asset"], shape=["K", "N"]),
    }
    expression = "transpose(matmul(a,b))" if case["transpose"] else "matmul(a,b)"
    graph = GraphCompiler(decl).compile(expression, output_order=case["output"])
    scheduler = AdaptiveScheduler(cpu_budget=1)
    bound = {"a": a, "b": b}
    zeros = np.zeros(1, np.int64)
    if case["entry"] == "prepared":
        prepared = scheduler.prepare_execution(graph, bound, zeros, zeros)
        function = prepared.run_snapshot
    else:
        def function():
            return scheduler.execute(graph, bound, zeros, zeros)

    def run():
        result = function()
        return result, result.outputs[0].values[0]
    return scheduler, run


def verify(case, backend, result, a, b):
    matrix = result[1]
    m, _, n = case["shape"]
    assert matrix.shape == ((n, m) if case["transpose"] else (m, n))
    assert matrix.dtype == np.float64 and not matrix.flags.writeable
    assert not np.shares_memory(matrix, a) and not np.shares_memory(matrix, b)
    expected_f = case["output"] == "F" or (case["output"] == "auto" and case["transpose"])
    assert matrix.flags.f_contiguous if expected_f else matrix.flags.c_contiguous
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

    with tempfile.TemporaryDirectory(prefix="cme-matmul-evidence-") as arrays_dir:
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
                if case["domain"] == "mixed" or case["shape"][0] < 512:
                    np.testing.assert_array_equal(values["old"].view(np.uint64),
                                                  values["new"].view(np.uint64))
                else:
                    np.testing.assert_allclose(values["new"], values["old"], rtol=2e-12, atol=2e-14)
                njit_close = np.isclose(values["njit"], values["old"], rtol=2e-12, atol=2e-14)
                parity = dict(new_old_tolerance_passed=True,
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
                assert measured["cold_peak_bytes"] >= case["shape"][0] * case["shape"][2] * 8, \
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
        "square_cc", "square_ff_transpose", "wide_ff_transpose", "zero_cc",
        "mixed_ff_transpose"}]
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
