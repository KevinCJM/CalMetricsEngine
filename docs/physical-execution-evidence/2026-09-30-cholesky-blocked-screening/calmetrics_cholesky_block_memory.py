import ctypes
import gc
import json
import os
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from numba import njit, typeof
from threadpoolctl import threadpool_limits

sys.path.insert(0, '/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine/tools')
from benchmark_memory import HeapStats
sys.path.insert(0, '/private/tmp/calmetrics-covchol-next/install-final')
from calmetrics_engine import operators as ops

backend = sys.argv[1]
n = int(sys.argv[2])
lib = ctypes.CDLL('/private/tmp/calmetrics_cholesky_block_probe.dylib')
fn = lib.cholesky_block_probe
pointer = ctypes.POINTER(ctypes.c_double)
fn.argtypes = (pointer, ctypes.c_int, ctypes.c_int, pointer)
fn.restype = ctypes.c_int
probe = ctypes.CDLL('/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib')

@njit(nogil=True)
def njit_cholesky(a):
    return np.linalg.cholesky(a)

rng = np.random.default_rng(20260929)
x = rng.normal(size=(n, n))
x = (x + x.T) * 0.5
x[np.diag_indices(n)] += np.abs(x).sum(axis=1) + 1.0
x.flags.writeable = False
if backend == 'njit':
    njit_cholesky.compile((typeof(x),))

def call():
    if backend == 'cpp':
        return ops.cholesky(x)
    if backend == 'njit':
        return njit_cholesky(x)
    block = int(backend)
    result = np.empty_like(x)
    code = fn(x.ctypes.data_as(pointer), n, block,
              result.ctypes.data_as(pointer))
    if code:
        raise RuntimeError(code)
    return result

def snapshot():
    stats = HeapStats()
    probe.cme_heap_snapshot(ctypes.byref(stats))
    return {name: getattr(stats, name) for name, _ in stats._fields_}

gc.collect()
with threadpool_limits(1):
    measure = {}
    probe.cme_heap_begin()
    try:
        measure['setup'] = snapshot()
        probe.cme_heap_reset_peak()
        retained = call()
        measure['cold'] = snapshot()
        retained = None
        gc.collect()
        measure['released'] = snapshot()
        probe.cme_heap_reset_peak()
        first = call()
        retained = call()
        measure['warm_two_results'] = snapshot()
    finally:
        final = HeapStats()
        probe.cme_heap_end(ctypes.byref(final))
measure['final'] = {name: getattr(final, name)
                    for name, _ in final._fields_}
measure['backend'] = backend
measure['n'] = n
print(json.dumps(measure))
