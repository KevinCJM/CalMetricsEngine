import hashlib
import json
import os
import statistics
import sys
import time

for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[name] = "1"
import numpy as np  # noqa: E402

from calmetrics_engine import AdaptiveScheduler, GraphCompiler  # noqa: E402

n = int(sys.argv[1])
mode = sys.argv[2]
rng = np.random.default_rng(903000 + n)
a = np.asfortranarray(rng.normal(size=(n, n)))
b = np.asfortranarray(rng.normal(size=(n, n)))
a.flags.writeable = b.flags.writeable = False
declarations = {
    "a": dict(kind="matrix", axes=["asset", "asset"], shape=["M", "K"]),
    "b": dict(kind="matrix", axes=["asset", "asset"], shape=["K", "N"]),
}
zeros = np.zeros(1, np.int64)


def setup():
    graph = GraphCompiler(declarations).compile("transpose(matmul(a,b))", output_order="F")
    engine = AdaptiveScheduler(cpu_budget=1)
    return engine, lambda: engine.execute(graph, {"a": a, "b": b}, zeros, zeros)


if mode == "time":
    engine, call = setup()
    try:
        previous = call()
        values = []
        for _ in range(7):
            t0 = time.perf_counter_ns()
            current = call()
            values.append((time.perf_counter_ns() - t0) / 1e6)
            previous = current
        matrix = previous.outputs[0].values[0]
        audit = previous.audit["native_chunks"][0]
        print(
            json.dumps(
                {
                    "n": n,
                    "times_ms": values,
                    "median_ms": statistics.median(values),
                    "sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                    "order": previous.outputs[0].order,
                    "audit": {
                        k: audit[k]
                        for k in (
                            "numeric_arena_bytes",
                            "root_copy_bytes",
                            "direct_output_bytes",
                            "algorithm_copy_bytes",
                            "operator_workspace_capacity_bytes",
                        )
                    },
                }
            )
        )
    finally:
        engine.close()
elif mode == "memory":
    sys.path.insert(0, "/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine/tools")
    from benchmark_memory import measure_heap, measure_rss

    engine, call, heap = measure_heap(
        setup, "/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib", repeats=2
    )
    try:
        rss = measure_rss(call, repeats=2)
        print(json.dumps({"n": n, "heap": heap, "rss": rss}))
    finally:
        engine.close()
else:
    raise ValueError(mode)
