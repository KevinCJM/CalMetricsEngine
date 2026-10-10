"""Fresh-process full direct-call allocation/RSS screening."""
import ctypes
import json
import os
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402
from numba import njit, set_num_threads, typeof  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from calmetrics_engine import operators as op  # noqa: E402

sys.path.insert(0, "/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine/tools")
from benchmark_memory import measure_heap, measure_rss  # noqa: E402

backend, case, mode = sys.argv[1:]
m, k, n = (2048, 2048, 2048) if case == "square" else (4096, 512, 4096)
rng = np.random.default_rng(903011 + m)
a = np.asfortranarray(rng.random((m, k)))
b = np.asfortranarray(rng.random((k, n)))
a.flags.writeable = b.flags.writeable = False
set_num_threads(1)
lib = ctypes.CDLL("/private/tmp/calmetrics-nonnegative-gemm/guard.dylib")
ptr = ctypes.POINTER(ctypes.c_double)
fn = lib.cme_nonnegative_gemm
fn.argtypes = [ctypes.c_int, ptr, ctypes.c_ssize_t, ctypes.c_ssize_t, ptr,
               ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_size_t,
               ctypes.c_size_t, ctypes.c_size_t, ptr]
fn.restype = ctypes.c_int


@njit(nogil=True)
def reference(a, b):
    return a @ b


reference.compile((typeof(a), typeof(b)))


def candidate():
    out = np.full((m, n), -1234.0)
    status = fn(0, a.ctypes.data_as(ptr), a.strides[0] // 8, a.strides[1] // 8,
                b.ctypes.data_as(ptr), b.strides[0] // 8, b.strides[1] // 8,
                m, k, n, out.ctypes.data_as(ptr))
    if status:
        raise RuntimeError(status)
    return out


call = {"old": lambda: op.matmul(a, b), "candidate": candidate,
        "njit": lambda: reference(a, b)}[backend]
with threadpool_limits(1):
    if mode == "heap":
        _, _, result = measure_heap(lambda: (None, call),
            "/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib", repeats=2)
        if result["cold_peak_bytes"] < m * n * 8:
            raise RuntimeError("heap probe not interposed: result allocation was not observed")
    elif mode == "rss":
        result = measure_rss(call, repeats=2)
    else:
        raise ValueError(mode)
print(json.dumps({"backend": backend, "shape": [m, k, n], "mode": mode,
                  "result": result}, indent=2))
