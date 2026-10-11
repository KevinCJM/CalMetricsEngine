"""Complete existing MatMul request, controlled warmed BLAS and independent outputs."""
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['baseline', 'candidate', 'njit'], required=True)
parser.add_argument('--cpu', type=int, choices=[1,2,4], required=True)
parser.add_argument('--block', type=int, choices=[64,512], default=64)
parser.add_argument('--order', choices=['C','F'], required=True)
parser.add_argument('--pattern', choices=['random','rejected_scale'], default='random')
parser.add_argument('--n', type=int, default=16384)
parser.add_argument('--t', type=int, default=2520)
parser.add_argument('--k', type=int, default=30)
parser.add_argument('--layout', choices=['CC','CF','FC','FF'], default='CC')
parser.add_argument('--rounds', type=int, default=6)
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--metric', choices=['heap','rss'])
args = parser.parse_args()
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['NUMBA_NUM_THREADS'] = str(args.cpu)

import argparse
import ctypes
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import numba
from numba import njit, prange

from threadpoolctl import threadpool_info
REFERENCE_BLAS_POOLS = []
REFERENCE_SIGNATURES = ()

ROOT = Path(__file__).resolve().parent / ('source-v21' if args.backend == 'baseline' else 'source-v22')
STAGE = Path(__file__).resolve().parent
PRODUCTION = STAGE / ('candidate-v21' if args.backend == 'baseline' else 'candidate-v22')
PROBE = '/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib'
SCOPE = ('owned readonly input preparation + ordinary Scheduler/native or warmed NJIT '
         '+ existing general-signed MatMul; CPU1 and actual single-thread BLAS; '
         'two independent readonly result generations; compilation excluded; '
         f'input={args.layout}, output={args.order}, pattern={args.pattern}; '
         'matrix request, not complete platform service')


@njit(nogil=True)
def product(a, b, f_output):
    result = a @ b
    return np.asfortranarray(result) if f_output else np.ascontiguousarray(result)


def fixture(n, t, k, layout):
    # CLI n=M, t=K, k=N, retained for the already verified orchestration protocol.
    rng = np.random.default_rng(20261005)
    scale = 1024.0 if args.pattern == 'rejected_scale' else 1.0
    return dict(a=np.array(rng.uniform(-scale, scale, (n, t)), order=layout[0]),
                b=np.array(rng.uniform(-scale, scale, (t, k)), order=layout[1]))


def owned(data, layout):
    result = {name: np.array(a, copy=True, order=layout[0 if name == 'a' else 1]) for name, a in data.items()}
    for a in result.values():
        a.flags.writeable = False
    return result


def prepare(args, data):
    if args.backend == 'njit':
        global REFERENCE_BLAS_POOLS, REFERENCE_SIGNATURES
        numba.set_num_threads(args.cpu)
        function = product
        function(**owned(fixture(2, 3, 2, args.layout), args.layout), f_output=args.order == "F")
        REFERENCE_SIGNATURES = tuple(function.signatures)
        REFERENCE_BLAS_POOLS = [p for p in threadpool_info() if p['user_api'] == 'blas']
        assert REFERENCE_BLAS_POOLS and all(p['num_threads'] == 1 for p in REFERENCE_BLAS_POOLS), REFERENCE_BLAS_POOLS
        def call():
            values = (function(**owned(data, args.layout), f_output=args.order == "F"),)
            assert tuple(function.signatures) == REFERENCE_SIGNATURES, "request-time NJIT compilation"
            for a in values:
                a.flags.writeable = False
            return values, None
        return None, call, None
    identity = json.loads((PRODUCTION / 'identity.json').read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in identity['source'].items())
    sys.path.insert(0, str(PRODUCTION / 'install'))
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    graph = GraphCompiler({
        'a':dict(kind='matrix',axes=['asset','asset'],shape=['M','K']),
        'b':dict(kind='matrix',axes=['asset','asset'],shape=['K','N'])
    }).compile('matmul(a,b)', output_order=args.order)
    scheduler = AdaptiveScheduler(cpu_budget=args.cpu)
    starts = np.zeros(1, np.int64); ends = np.zeros(1, np.int64)
    def call():
        result = scheduler.execute(graph, owned(data, args.layout), starts, ends)
        assert result.audit['engine_build_id'] == identity['build_id']
        return tuple(out.values[0] for out in result.outputs), result.audit
    return scheduler, call, identity['build_id']


def independent(data, actual):
    from decimal import Decimal, localcontext
    value = actual[0]; a, b = data['a'], data['b']
    assert value.shape == (a.shape[0], b.shape[1]) and np.isfinite(value).all()
    assert value.flags.f_contiguous if args.order == 'F' else value.flags.c_contiguous
    errors = []
    for row in [0, a.shape[0] // 2, a.shape[0] - 1]:
        for col in [0, b.shape[1] // 2, b.shape[1] - 1]:
            with localcontext() as context:
                context.prec = 80
                reference = sum((Decimal.from_float(float(v)) * Decimal.from_float(float(w))
                                 for v, w in zip(a[row], b[:, col])), Decimal(0))
            errors.append(abs(float(value[row,col]) - float(reference)))
    maximum = max(errors)
    assert maximum < .000049, maximum
    left = np.full(a.shape[0], 1 / a.shape[0]); right = np.full(b.shape[1], 1 / b.shape[1])
    lhs = float(left @ value @ right); rhs = float((left @ a) @ (b @ right))
    residual = abs(lhs-rhs)
    assert residual < .000049, residual
    return dict(decimal_maximum_absolute_error=maximum, normalized_bilinear_residual=residual)


def time_request(args):
    data = fixture(args.n, args.t, args.k, args.layout)
    owner, call, build = prepare(args, data); samples = []; cpus = []
    try:
        retained, audit = call(); snapshots = [a.copy() for a in retained]
        if args.interactive:
            print('READY', flush=True)
        for _ in range(args.rounds):
            if args.interactive:
                assert sys.stdin.readline().strip() == 'RUN'
            cpu = time.process_time_ns(); start = time.perf_counter_ns()
            current, audit = call()
            samples.append((time.perf_counter_ns() - start) / 1e9)
            cpus.append((time.process_time_ns() - cpu) / 1e9)
            for a, b, snapshot in zip(current, retained, snapshots, strict=True):
                np.testing.assert_array_equal(b, snapshot)
                np.testing.assert_array_equal(a, b)
                assert not a.flags.writeable and not np.shares_memory(a, b)
                assert all(not np.shares_memory(a, source) for source in data.values())
            del a, b, snapshot
            retained = current; snapshots = [a.copy() for a in retained]
            if args.interactive:
                print('SAMPLE ' + json.dumps(dict(wall=samples[-1], cpu=cpus[-1])), flush=True)
        report = dict(backend=args.backend, shape=[args.n, args.t, args.k], layout=args.layout,
                      samples_seconds=samples, samples_cpu_seconds=cpus,
                      median_seconds=statistics.median(samples),
                      maximum_absolute_errors=independent(data, retained), build_id=build,
                      input_hashes={name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in data.items()},
                      audit=audit, scope=SCOPE, cpu_budget=args.cpu, output_order=args.order, pattern=args.pattern,
                      reference_execution=dict(threads=numba.get_num_threads(), parallel=False, request_time_compilation=0, blas_threads_requested=1, actual_blas_pools=REFERENCE_BLAS_POOLS) if args.backend == 'njit' else None, whole_goal_complete=False)
        print(json.dumps(report), flush=True)
    finally:
        if owner:
            owner.close()


def peak_request(args):
    # Warm imports/JIT before accounting; fresh real fixtures and graph below.
    small = fixture(2, 3, 2, args.layout)
    owner, _, _ = prepare(args, small)
    if owner:
        owner.close()
    sys.path.insert(0, str(ROOT / 'tools'))
    from benchmark_memory import HeapStats, measure_rss
    state = {}
    def setup():
        state['data'] = fixture(args.n, args.t, args.k, args.layout)
        state['owner'], state['call'], state['build'] = prepare(args, state['data'])
    def call():
        if 'call' not in state:
            setup()
        values, state['audit'] = state['call']()
        return values
    try:
        if args.metric == 'rss':
            measurement = measure_rss(call, repeats=3)
        else:
            assert os.environ.get('DYLD_INSERT_LIBRARIES') == PROBE
            probe = ctypes.CDLL(PROBE); gc.collect(); stages = {}
            probe.cme_heap_begin()
            try:
                setup(); snapshot = HeapStats(); probe.cme_heap_snapshot(ctypes.byref(snapshot))
                stages['setup'] = {name: getattr(snapshot, name) for name, _ in snapshot._fields_}
                current = None
                for repeat in range(3):
                    last = current; probe.cme_heap_reset_peak(); current = call()
                    if last:
                        assert all(not np.shares_memory(a, b) for a, b in zip(current, last, strict=True))
                    snapshot = HeapStats(); probe.cme_heap_snapshot(ctypes.byref(snapshot))
                    stages[str(repeat)] = {name: getattr(snapshot, name) for name, _ in snapshot._fields_}
                measurement = dict(stages=stages, warm_peak_bytes=stages['2']['peak_bytes'],
                                   cold_peak_bytes=stages['0']['peak_bytes'],
                                   schema='continuous-live-1; call 3 retains one old independent generation')
            finally:
                final = HeapStats(); probe.cme_heap_end(ctypes.byref(final)); assert final.table_overflows == 0
        verification_values = call()
        verification = independent(state['data'], verification_values)
        print(json.dumps(dict(backend=args.backend, metric=args.metric, shape=[args.n, args.t, args.k],
                              layout=args.layout, output_order=args.order, pattern=args.pattern, measurement=measurement, audit=state['audit'],
                              post_measurement_mathematical_verification=verification,
                              build_id=state['build'], scope=SCOPE, cpu_budget=args.cpu,
                              reference_execution=dict(threads=numba.get_num_threads(), parallel=False, request_time_compilation=0, blas_threads_requested=1, actual_blas_pools=REFERENCE_BLAS_POOLS) if args.backend == 'njit' else None,
                              input_hashes={name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in state['data'].items()},
                              blind_spots=f'heap excludes Python arena suballocation/mmap; RSS samples may miss short peaks; CPU{args.cpu}; single process native/NJIT workers included',
                              whole_goal_complete=False)), flush=True)
    finally:
        if state.get('owner'):
            state['owner'].close()


if __name__ == '__main__':
    assert args.cpu == 1 and args.n > 0 and args.t > 1 and args.k > 0
    (peak_request if args.metric else time_request)(args)
