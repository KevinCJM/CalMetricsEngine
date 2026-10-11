"""Isolated near-zero covariance comparison; this is not production math."""
import ctypes
import json
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from threadpoolctl import threadpool_limits
from calmetrics_engine import operators as op

library = ctypes.CDLL('/private/tmp/calmetrics-covchol-next/libcovariance_blas_probe.dylib')
pointer = ctypes.POINTER(ctypes.c_double)
candidate = library.covariance_blas_probe
candidate.argtypes = (pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, pointer)
candidate.restype = ctypes.c_int

reports = []
with threadpool_limits(1):
    for layout in ('C', 'F'):
        n, d = 529, 512
        random = np.random.default_rng(123)
        x = random.normal(size=(n, d))
        x -= x.mean(axis=0)
        first = x[:, 0]
        denominator = float(first @ first)
        for column in range(1, d):
            x[:, column] -= float(first @ x[:, column]) / denominator * first
        x = np.array(x, order=layout)
        x.flags.writeable = False
        baseline = op.covariance(x)
        alternate = np.empty((d, d), dtype=np.float64)
        code = candidate(x.ctypes.data_as(pointer), n, d, int(x.flags.f_contiguous),
                         512, alternate.ctypes.data_as(pointer))
        assert code == 0
        upper = np.triu_indices(d, 1)
        sign_changed = ((np.signbit(alternate) != np.signbit(baseline)) &
                        (alternate != 0) & (baseline != 0))
        reports.append(dict(layout=layout, rows=n, columns=d,
                            sign_flips=int(np.count_nonzero(sign_changed[upper])),
                            near_zero=int(np.count_nonzero(np.abs(baseline[upper]) < 1e-12)),
                            max_abs=float(np.max(np.abs(alternate - baseline)))))

print(json.dumps(reports, indent=2))
