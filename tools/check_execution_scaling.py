"""Equal CPU budgets, warmed NJIT, independent results; optional isolated heap probe.

Run timing without the malloc interposer. --heap-probe executes separate children;
its overhead must never contaminate performance numbers. A measured failure stays
in the report. This supplements, never replaces, the fixed single-core gates.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from check_foundation_performance import data, paired_ratio, references


def reference(case):
    from numba import njit, prange, set_num_threads
    set_num_threads(case['cpu'])
    if case['cpu'] == 1 or case.get('reference_lane') == 'serial':
        return references()[case['kernel']]

    @njit(nogil=True, parallel=True)
    def affine(x):
        values = np.empty(x.shape)
        mask = np.empty(x.shape, np.bool_)
        for i in prange(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    v = x[i, j, k] * 2. + 1.
                    values[i, j, k] = v
                    mask[i, j, k] = v > 3.
        return values, mask

    @njit(nogil=True, parallel=True)
    def relaxation(x):
        current = x.copy()
        residual = np.inf
        count, status = 0, 1
        for iteration in range(100):
            residual = 0.
            for i in prange(current.size):
                value = current[i] * .5
                residual = max(residual, abs(value-current[i]))
                current[i] = value
            count = iteration+1
            if residual <= 1e-6:
                status = 0
                break
        return current, np.asarray(status, np.int64), np.asarray(count, np.int64), np.asarray(residual)
    return affine if case['kernel'] == 'affine' else relaxation


def native(case, x):
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    if case['kernel'] == 'affine':
        variables = {'x': {'kind': 'tensor', 'axes': ['scenario','path','asset'], 'shape': ['S','P','N']}}
        expressions = {'value': 'x*2+1', 'mask': 'x*2+1>3'}
    else:
        variables = {'x': {'kind': 'vector', 'axes': ['asset'], 'shape': ['N']}}
        expr = 'iterate(iterate_x*0.5,x,1e-6,100)'
        expressions = {'value': expr, 'status': f'iteration_status({expr})',
                       'count': f'iteration_count({expr})', 'residual': f'iteration_residual({expr})'}
    graph = GraphCompiler(variables).compile(expressions)
    engine = AdaptiveScheduler(cpu_budget=case['cpu'])
    starts, ends = np.array([0], np.int64), np.array([1], np.int64)
    inputs = {'x': x}
    return engine, lambda: engine.execute(graph, inputs, starts, ends)


def measure(function, loops):
    result = None
    cpu_begin = time.process_time_ns()
    begin = time.perf_counter_ns()
    for _ in range(loops):
        result = function()
    return (time.perf_counter_ns()-begin)/loops, result, (time.process_time_ns()-cpu_begin)/loops


class HeapStats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        'allocations','frees','allocated_bytes','live_bytes','peak_bytes','untracked_frees','table_overflows')]


def heap_worker(case, backend, library):
    x = data(case)
    # JIT compilation is excluded: both timing and memory use compiled NJIT.
    ref = reference(case)
    ref(x)
    engine, cpp = native(case, x)
    function = cpp if backend == 'cpp' else lambda: ref(x)
    probe = ctypes.CDLL(library)
    def capture(loops):
        stats = HeapStats()
        gc.collect()
        probe.cme_heap_begin()
        result = None
        for _ in range(loops):
            result = function()
        probe.cme_heap_end(ctypes.byref(stats))
        record = {name: getattr(stats, name) for name, _ in stats._fields_}
        record['counter_valid'] = stats.table_overflows == 0 and stats.allocations > 0
        return record, result
    cold, result = capture(1)
    del result
    # The cold report includes new engine TLS/pool buffers. Warm counters exclude
    # retained allocations created earlier; cold and warm peaks are NOT additive.
    warm, result = capture(11)
    audit = result.audit if backend == 'cpp' else None
    engine.close()
    print(json.dumps({'cold_request': cold, 'warm_requests': warm, 'audit': audit}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--heap-probe', type=Path)
    parser.add_argument('--heap-worker', choices=['cpp','njit'])
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.heap_worker:
        heap_worker(json.loads(args.case), args.heap_worker, str(args.heap_probe))
        return
    cases = []
    for name, kernel, shape, layout in [
        ('tensor_small','affine',[8,2,3],'C'), ('tensor_medium','affine',[252,16,8],'C'),
        ('tensor_large','affine',[2520,64,8],'C'), ('tensor_reverse','affine',[252,16,8],'reverse'),
        ('iteration_medium','relaxation',[4096],'C'), ('iteration_large','relaxation',[262144],'C')]:
        for cpu in [1,2,4]:
            cases.append(dict(name=name, kernel=kernel, shape=shape, layout=layout, cpu=cpu))
    results = []
    for case in cases:
        x = data(case)
        ref = reference(case)
        engine, cpp = native(case, x)
        njit = lambda: ref(x)
        serial_ref = references()[case['kernel']] if case['cpu'] > 1 else ref
        serial = lambda: serial_ref(x)
        serial()
        for _ in range(5):
            left, right = cpp(), njit()
        for actual, expected in zip(left.outputs, right):
            np.testing.assert_array_equal(actual.values[0], expected)
        loops = 100 if x.size < 100 else 8 if x.size < 100000 else 2
        ca, na, sa, ccpu, ncpu, scpu = [], [], [], [], [], []
        for repeat in range(21):
            order = [('cpp',cpp),('njit',njit)]
            if case['cpu'] > 1: order.append(('serial',serial))
            order = order[repeat%len(order):] + order[:repeat%len(order)]
            for name, function in order:
                value, _, cpu_time = measure(function, loops)
                {'cpp': ccpu, 'njit': ncpu, 'serial': scpu}[name].append(cpu_time)
                {'cpp': ca, 'njit': na, 'serial': sa}[name].append(value)
        # Same CPU budget is an upper bound for both engines. Do not force
        # NJIT into a slower parallel lane just because C++ chose single.
        serial_wins = bool(sa) and np.median(sa) < np.median(na)
        lane = 'serial' if case['cpu'] == 1 or serial_wins else 'parallel'
        result = {'case': case, 'timing': paired_ratio(ca, sa if serial_wins else na),
                  'njit_variant': lane, 'njit_parallel_ns': na if case['cpu'] > 1 else [],
                  'njit_serial_ns': sa if sa else na, 'audit': left.audit,
                  'plan': left.plan.metadata(), 'cpp_cpu_ns': ccpu,
                  'njit_cpu_ns': scpu if serial_wins else ncpu}
        result['speed_pass'] = result['timing']['upper95'] < 1
        engine.close()
        results.append(result)
        print(case['name'],case['cpu'],round(result['timing']['upper95'],3),left.plan.lane,flush=True)
    # Heap runs follow all timed measurements. Injection also records Python /
    # NumPy malloc calls in the measured API boundary, not only C++ scratch.
    if args.heap_probe:
        for result in results:
            case = dict(result['case'], reference_lane=result['njit_variant'])
            if case['cpu'] not in [1,4]:
                continue
            result['heap'] = {}
            for backend in ['cpp','njit']:
                env = dict(os.environ, DYLD_INSERT_LIBRARIES=str(args.heap_probe.resolve()))
                proc = subprocess.run([sys.executable,__file__,'--heap-worker',backend,'--case',json.dumps(case),
                    '--heap-probe',str(args.heap_probe.resolve())], env=env, capture_output=True, text=True, timeout=180, check=True)
                result['heap'][backend] = json.loads(proc.stdout)
    report = {'schema':'physical-execution-scaling-1', 'results':results,
        'heap_scope':'macOS malloc/calloc/realloc/free/posix_memalign/aligned_alloc requested bytes in isolated process',
        'heap_limits':['CPython arena suballocations are not individual malloc calls',
                       'retained pre-measurement allocations are excluded from warm counters',
                       'mmap/shared regions and child processes require separate transport/RSS evidence',
                       'trace instrumentation is excluded from timing'],
        'reference_selection':'fastest warmed serial or parallel NJIT within the same CPU budget',
        'all_speed_pass':all(r['speed_pass'] for r in results)}
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
