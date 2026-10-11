"""Isolated ABI correctness/timing screen, never a whole-engine acceptance."""
# ruff: noqa: E402
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import ast
import ctypes as ct
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numba import njit
from calmetrics_engine import operators as op

HERE = Path(__file__).resolve().parent
LIB = ct.CDLL(str(HERE / 'chol-probe.dylib'))
LIB.chol_probe_workspace.argtypes = [ct.c_size_t]
LIB.chol_probe_workspace.restype = ct.c_void_p
LIB.chol_probe_workspace_free.argtypes = [ct.c_void_p]
LIB.chol_probe_workspace_bytes.argtypes = [ct.c_void_p]
LIB.chol_probe_workspace_bytes.restype = ct.c_size_t
LIB.chol_probe.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_ssize_t, ct.c_ssize_t,
                          ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_size_t]
LIB.chol_probe.restype = ct.c_int


def sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def spd(n, layout='C'):
    rng = np.random.default_rng(20261001)
    a = np.empty((n, n), order=layout)
    rng.standard_normal(out=a)
    for i in range(n):
        a[i, :i] = a[:i, i]
        a[i, i] += np.abs(a[i]).sum() + 1.
    a.flags.writeable = False
    return a


def call(a, workspace, grouped, f_order=False, check_prefix=False):
    storage = np.full(a.size + 2, -917.) if check_prefix else np.empty(a.size + 2)
    storage[0] = storage[-1] = -917.
    y = storage[1:-1].reshape(a.shape, order='F' if f_order else 'C')
    error = ct.create_string_buffer(128)
    status = LIB.chol_probe(a.ctypes.data, len(a), a.strides[0] // 8, a.strides[1] // 8,
                           y.ctypes.data, workspace, grouped, f_order, error, len(error))
    assert storage[0] == storage[-1] == -917.
    return ('OK' if status == 0 else error.value.decode()), y


def correctness():
    source = HERE / 'historical/prototype-fixtures/check_cholesky_probe.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cases')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    fixtures = list(namespace['cases']())
    for n in (1023, 1024, 1025, 1026, 1027, 1028, 1030, 1031, 1032, 1035, 1039, 1040, 1041):
        a = spd(n)
        fixtures.append((f'spd_{n}_C', a, 'production threshold and 4-row/eight-column/16-row tails'))
        if n in (1025, 1031, 1041):
            fixtures.append((f'spd_{n}_F', np.asfortranarray(a), 'C/F logical parity'))
    a = spd(1031)
    fixtures.append(('negative_1031', a[::-1, ::-1], 'negative strides'))
    holder = np.full((1031, 2062), np.nan)
    holder[:, ::2] = a
    fixtures.append(('padded_1031', holder[:, ::2], 'padding never consumed'))
    for case in ('cancellation', 'tiny', 'signed_zero', 'domain_priority', 'shared_tile_overflow_priority', 'overflow', 'nan', 'asymmetry'):
        a = np.eye(1025)
        if case == 'cancellation':
            ids = [0, 1, 1024]
            a[np.ix_(ids, ids)] = [[1, 0, 2.**27], [0, 1, 1], [2.**27, 1, 2.**54+4]]
        elif case == 'tiny':
            a[0, 0] = np.nextafter(0., 1.)
        elif case == 'signed_zero':
            a[a == 0] = -0.
        elif case == 'domain_priority':
            a[16, 16] = -1.
            a[0, 0] = np.nextafter(0., 1.)
            a[0, 17] = a[17, 0] = 1e308
        elif case == 'overflow':
            a[0, 0] = np.nextafter(0., 1.)
            a[0, 17] = a[17, 0] = 1e308
        elif case == 'shared_tile_overflow_priority':
            a[8, 8] = 2e300
            a[0, 8] = a[8, 0] = 1e150
            a[16, 16] = -1.
            a[0, 20] = a[20, 0] = 1e200
        elif case == 'nan':
            a[-1, -1] = np.nan
        else:
            a[0, -1] = np.nextafter(0., 1.)
        fixtures.append((case, a, 'original arithmetic and error priority'))
    rows = []
    calls = 0
    for name, a, purpose in fixtures:
        a.flags.writeable = False
        before = sha(a)
        try:
            reference = op.cholesky(a, simd='scalar')
            reference_status = 'OK'
        except (ValueError, RuntimeError) as error:
            reference, reference_status = None, str(error)
        workspace = LIB.chol_probe_workspace(len(a))
        try:
            for f_order in (False, True):
                baseline_status, baseline = call(a, workspace, 0, f_order, True)
                calls += 1
                assert baseline_status == reference_status, (name, baseline_status, reference_status)
                if reference is not None:
                    np.testing.assert_array_equal(baseline.view(np.uint64), reference.view(np.uint64))
                for grouped in (8, 16, 88):
                    candidate_status, candidate = call(a, workspace, grouped, f_order, True)
                    calls += 1
                    assert candidate_status == baseline_status, (name, grouped, baseline_status, candidate_status)
                    np.testing.assert_array_equal(candidate.view(np.uint64), baseline.view(np.uint64))
                    assert candidate.flags.f_contiguous if f_order else candidate.flags.c_contiguous
                    assert sha(a) == before
                    rows.append(dict(case=name, candidate=grouped, purpose=purpose, n=len(a), strides=list(a.strides),
                                     output_order='F' if f_order else 'C', status=candidate_status,
                                     old_candidate_canonical_equal=reference is not None,
                                     old_candidate_full_buffer_equal=True, input_canary_intact=True,
                                     workspace_bytes=LIB.chol_probe_workspace_bytes(workspace)))
        finally:
            LIB.chol_probe_workspace_free(workspace)
    (HERE / 'screen.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(dict(fixtures=len(fixtures), calls=calls, rows=len(rows), failures=0)))


@njit(cache=False)
def ni(a):
    return np.asfortranarray(np.linalg.cholesky(a))


def timing_impl(threads):
    records = []
    for n, layout in ((63, 'C'), (1024, 'C'), (2048, 'C'), (2048, 'F'), (4096, 'C'), (4096, 'F'), (1031, 'F')):
        a = spd(n, layout)
        workspace = LIB.chol_probe_workspace(n)
        try:
            functions = {'old': lambda: call(a, workspace, 0, True)[1],
                         'g8w4': lambda: call(a, workspace, 8, True)[1],
                         'g16w2': lambda: call(a, workspace, 16, True)[1],
                         'g8w8': lambda: call(a, workspace, 88, True)[1], 'njit': lambda: ni(a)}
            outputs = {b: f() for b, f in functions.items()}
            for candidate in ('g8w4', 'g16w2', 'g8w8'):
                np.testing.assert_array_equal(outputs['old'].view(np.uint64), outputs[candidate].view(np.uint64))
                np.testing.assert_allclose(outputs[candidate], outputs['njit'], rtol=1e-11, atol=1e-11)
            assert all(v.flags.f_contiguous for v in outputs.values())
            samples = {b: [] for b in functions}
            names = list(functions)
            for round_index in range(10):
                order = names[round_index % len(names):] + names[:round_index % len(names)]
                if round_index % 2:
                    order = order[::-1]
                for backend in order:
                    start = time.perf_counter_ns()
                    result = functions[backend]()
                    samples[backend].append((time.perf_counter_ns() - start) / 1e6)
                    outputs[backend] = result
            record = dict(n=n, input_layout=layout, output_order='F', cpu_budget=1,
                          scope='isolated ABI including input scan/output allocation, not Scheduler acceptance',
                          threads=threads,
                          samples_ms=samples, median_ms={b:float(np.median(v)) for b,v in samples.items()},
                          input_sha256=sha(a), output_sha256={b:sha(v) for b,v in outputs.items()},
                          bit_parity_old_candidate=True, workspace_bytes=LIB.chol_probe_workspace_bytes(workspace))
            records.append(record)
            (HERE / 'timing.json').write_text(json.dumps(records, indent=2))
            print(json.dumps({'n':n, 'layout':layout, 'median_ms':record['median_ms']}), flush=True)
        finally:
            LIB.chol_probe_workspace_free(workspace)


def timing():
    import sys
    sys.path.insert(0, '/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine/tools')
    from check_large_matrix_performance import baseline_threads
    with baseline_threads('njit') as threads:
        timing_impl(threads)
    (HERE / 'threads-restored.json').write_text(json.dumps(threads, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['correctness', 'timing'], required=True)
    args = parser.parse_args()
    (correctness if args.mode == 'correctness' else timing)()
