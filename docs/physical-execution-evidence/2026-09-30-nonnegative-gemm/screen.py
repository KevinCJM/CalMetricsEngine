"""Screen positive-domain BLAS admission; not a production/graph benchmark."""
import ctypes
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np  # noqa: E402
from numba import njit, set_num_threads  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from calmetrics_engine import operators as op  # noqa: E402

BASE = Path("/private/tmp/calmetrics-nonnegative-gemm")
lib = ctypes.CDLL(str(BASE / "guard.dylib"))
ptr = ctypes.POINTER(ctypes.c_double)
call = lib.cme_nonnegative_gemm
call.argtypes = [ctypes.c_int, ptr, ctypes.c_ssize_t, ctypes.c_ssize_t, ptr,
                 ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_size_t,
                 ctypes.c_size_t, ctypes.c_size_t, ptr]
call.restype = ctypes.c_int
state = lib.cme_blas_thread_state
state.restype = ctypes.c_int
set_num_threads(1)


def candidate(a, b, variant=0):
    out = np.full((a.shape[0], b.shape[1]), -1234.0)
    previous = state()
    status = call(variant, a.ctypes.data_as(ptr), a.strides[0] // 8,
                  a.strides[1] // 8, b.ctypes.data_as(ptr), b.strides[0] // 8,
                  b.strides[1] // 8, a.shape[0], a.shape[1], b.shape[1],
                  out.ctypes.data_as(ptr))
    assert state() == previous
    return status, out


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def generate(m, k, n, family, layout, seed):
    rng = np.random.default_rng(seed)
    if family == "uniform":
        a, b = rng.random((m, k)), rng.random((k, n))
    elif family == "powers":
        a = np.exp2(rng.integers(-128, 129, size=(m, k)))
        b = np.exp2(rng.integers(-128, 129, size=(k, n)))
    elif family == "sparse":
        a, b = rng.random((m, k)), rng.random((k, n))
        a[rng.random(a.shape) < 0.95] = 0.0
        b[rng.random(b.shape) < 0.95] = 0.0
    elif family == "ones":
        a, b = np.ones((m, k)), np.ones((k, n))
    elif family == "tiny":
        a = np.full((m, k), 2.0 ** -128)
        b = np.full((k, n), 2.0 ** -128)
    elif family == "large":
        a = np.full((m, k), 2.0 ** 128)
        b = np.full((k, n), 2.0 ** 128)
    elif family == "zeros":
        a, b = np.zeros((m, k)), np.zeros((k, n))
    else:
        raise ValueError(family)
    a, b = np.array(a, order=layout[0]), np.array(b, order=layout[1])
    a.flags.writeable = b.flags.writeable = False
    return a, b


@njit(nogil=True)
def reference(a, b):
    return a @ b


results = {"correctness": [], "rejections": [], "timing": []}
with threadpool_limits(1):
    for index, (m, k, n) in enumerate(((33, 32, 35), (65, 257, 67),
                                     (64, 4096, 64), (1024, 1024, 1024))):
        for layout in ("CC", "CF", "FC", "FF"):
            for family in ("uniform", "powers", "sparse", "ones", "tiny", "large", "zeros"):
                a, b = generate(m, k, n, family, layout, 903001 + index)
                before = digest(a), digest(b)
                old = op.matmul(a, b)
                for variant in (0, 3):
                    status, actual = candidate(a, b, variant)
                    assert status == 0
                    tolerance = 2e-14 + 2e-12 * np.abs(old)
                    error = np.abs(actual - old)
                    mismatches = int(np.count_nonzero(error > tolerance))
                    assert mismatches == 0
                    assert np.array_equal(actual > 0, old > 0)
                    assert np.array_equal(np.signbit(actual), np.signbit(old))
                    assert before == (digest(a), digest(b))
                    results["correctness"].append({"shape": [m, k, n], "layout": layout,
                        "family": family, "variant": variant, "mismatches": mismatches,
                        "max_relative_error": float(np.max(np.divide(error, np.abs(old),
                            out=np.zeros_like(error), where=old != 0)))})
    for value in (-1.0, -0.0, np.nan, np.inf, 2.0 ** -129, 2.0 ** 129):
        a, b = generate(33, 32, 35, "ones", "CC", 1)
        a = a.copy()
        a[1, 2] = value
        status, out = candidate(a, b)
        assert status == 1 and np.all(out == -1234.0)
        results["rejections"].append({"value": str(value), "status": status})
    for m, k, n, layout in ((2048, 2048, 2048, "CC"), (2048, 2048, 2048, "FF"),
                            (4096, 512, 4096, "FF")):
        a, b = generate(m, k, n, "uniform", layout, 903011 + m)
        expected = op.matmul(a, b)
        previous = reference(a, b)
        njit_order = "F" if previous.flags.f_contiguous else "C"
        methods = {
            "old": lambda a=a, b=b: op.matmul(a, b),
            "whole": lambda a=a, b=b: candidate(a, b, 0)[1],
            "tile256": lambda a=a, b=b: candidate(a, b, 3)[1],
            "njit": lambda a=a, b=b: reference(a, b),
        }
        samples = {name: [] for name in methods}
        for repeat in range(8):
            order = list(methods)
            order = order[repeat % 4:] + order[:repeat % 4]
            for name in order:
                start = time.perf_counter_ns()
                current = methods[name]()
                samples[name].append((time.perf_counter_ns() - start) / 1e6)
                assert np.all(np.abs(current - expected) <= 2e-14 + 2e-12 * np.abs(expected))
                previous = current
        results["timing"].append({"shape": [m, k, n], "layout": layout,
            "samples_ms": samples,
            "medians_ms": {name: statistics.median(values) for name, values in samples.items()},
            "njit_order": njit_order})
        (BASE / "screen.json").write_text(json.dumps(results, indent=2))
(BASE / "screen.json").write_text(json.dumps(results, indent=2))
print(json.dumps({"correctness": len(results["correctness"]),
                  "rejections": len(results["rejections"]), "timing": results["timing"]}, indent=2))
