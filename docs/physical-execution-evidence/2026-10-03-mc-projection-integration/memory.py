"""Fresh-child complete projection allocation and RSS evidence; root runs serially."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STAGE = Path(__file__).resolve().parent
ENGINE = Path('/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine')
IDENTITY_HELPER = Path('/private/tmp/calmetrics-platform-latent-closure/memory.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class NoOwner:
    def close(self):
        pass

def child(args):
    import bench
    np, package, service, shared, install = bench.load_runtime(args.install, 1)
    sys.path.insert(0, str(ENGINE / 'tools'))
    from benchmark_memory import measure_heap, measure_rss
    from threadpoolctl import threadpool_info, threadpool_limits
    specification = importlib.util.spec_from_file_location('identity_check', IDENTITY_HELPER)
    identity_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(identity_module)
    before = identity_module.identity(args.identity_manifest, package)
    if not before['build_id_verified']:
        raise RuntimeError('source/wheel identity mismatch')
    reference = service if args.reference == 'service' else shared
    data = bench.inputs(np, args.n, args.factors, args.assets, args.combinations, args.steps)
    tiny = bench.inputs(np, 12, args.factors, args.assets, args.combinations, 2)
    started = time.perf_counter_ns()
    reference(tiny['x'], tiny['beta'], tiny['initial'], tiny['weights'])
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    del tiny
    observed = {}
    prepare_ms = [0.0]
    def capture(call):
        def wrapped():
            result = call()
            if not observed:
                if args.backend == 'njit':
                    values = bench.reference_values(np, result)
                else:
                    values = [output.values[0] for output in result.outputs]
                    observed['native_audit'] = dict(result.audit)
                    observed['root_statuses'] = [list(map(int, o.root_statuses)) for o in result.outputs]
                    chunks = result.audit['native_chunks']
                    output_bytes = sum(value.nbytes for value in values)
                    if (not chunks or not all(chunk['fused_pointwise_calls'] > 0 and
                            chunk['pointwise_tiles'] > 0 and chunk['numeric_arena_bytes'] == 0
                            for chunk in chunks) or
                            sum(chunk['direct_output_bytes'] for chunk in chunks) != output_bytes):
                        raise AssertionError('bounded whole-projection execution did not occur')
                    if any(any(status for status in output.root_statuses) for output in result.outputs):
                        raise AssertionError('unexpected failed projection output')
                    if any(result.audit[key] for key in
                           ('python_fallback','python_operator_calls','request_time_compilation')):
                        raise AssertionError('non-native projection execution')
                observed.update(output_count=len(values), output_bytes=sum(v.nbytes for v in values),
                                output_shapes=[list(v.shape) for v in values],
                                output_dtypes=[str(v.dtype) for v in values])
            return result
        return wrapped
    def construct():
        started = time.perf_counter_ns()
        if args.backend == 'njit':
            owner = NoOwner()
            call = lambda: reference(data['x'], data['beta'], data['initial'], data['weights'])
        else:
            owner, call, _, _, _ = bench.setup(np, package, data, args.backend, 1, args.order)
        prepare_ms[0] = (time.perf_counter_ns() - started) / 1e6
        return owner, capture(call)
    owner = None
    with threadpool_limits(limits=1):
        try:
            if args.metric == 'heap':
                if os.environ.get('DYLD_INSERT_LIBRARIES') != str(args.heap_probe.resolve()):
                    raise RuntimeError('heap child requires exact selected interposition library')
                owner, call, measurement = measure_heap(construct, str(args.heap_probe), args.repeats)
            else:
                owner, call = construct()
                measurement = measure_rss(call, args.repeats)
            after = identity_module.identity(args.identity_manifest, package)
            if before != after:
                raise RuntimeError('source changed during observation')
            report = dict(backend=args.backend, metric=args.metric, scope=bench.SCOPE,
                          remaining_gaps=bench.GAPS, reference=args.reference, cpu_budget=1,
                          shape={k: data[k] for k in ('n','steps','paths','factors','assets','combinations')},
                          fresh_pid=os.getpid(), repeats=args.repeats, output_order=args.order,
                          measurement=measurement, outputs=observed, identity=before,
                          source_unchanged=True, build_info=package.build_info(), threadpools=threadpool_info(),
                          tiny_compile_and_warm_ms_excluded=compile_ms, prepare_ms=prepare_ms[0],
                          probe_sha256=sha(args.heap_probe) if args.metric == 'heap' else None,
                          reference_sha256=sha(STAGE/'reference.py'), driver_sha256=sha(__file__),
                          helper_sha256=sha(ENGINE/'tools/benchmark_memory.py'),
                          identity_helper_sha256=sha(IDENTITY_HELPER),
                          notes=['Identical complete period/contribution outputs and independent previous/new replacement.',
                                 'Common readonly inputs and tiny NJIT compilation are outside both observation epochs.',
                                 'Heap includes graph construction, first run, five warmups and two retained-result replacements.',
                                 'RSS owner/graph is constructed before baseline; first full-size execution and replacement are inside.',
                                 'RSS sampled at 0.5ms; idle page control is reported without subtraction.',
                                 'Heap interposition can miss Python arena suballocations, mmap and nonintercepted allocators.',
                                 'CPU 1: no worker process or shared-page aggregation is claimed.',
                                 'Fixed transformed tape projection is not complete MC/NAV/fan acceptance.'])
            print(json.dumps(report, indent=2, default=str), flush=True)
        finally:
            if owner:
                owner.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('ordinary','prepared','njit'), required=True)
    parser.add_argument('--reference', choices=('service','shared'), default='service')
    parser.add_argument('--install', type=Path, required=True)
    parser.add_argument('--identity-manifest', type=Path, required=True)
    parser.add_argument('--heap-probe', type=Path, required=True)
    parser.add_argument('--n', type=int, default=2000000)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--factors', type=int, default=4)
    parser.add_argument('--assets', type=int, default=32)
    parser.add_argument('--combinations', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--order', choices=('C','F'), default='C')
    parser.add_argument('--metric', choices=('heap','rss','all'), default='all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('two independent retained results required')
    if args.child:
        child(args)
        return
    if os.environ.get('DYLD_INSERT_LIBRARIES'):
        parser.error('parent must not inherit heap instrumentation')
    reports = []
    for metric in (('heap','rss') if args.metric == 'all' else (args.metric,)):
        command = [sys.executable, str(Path(__file__).resolve()), '--child']
        for name in ('backend','reference','install','identity_manifest','heap_probe','n','steps',
                     'factors','assets','combinations','repeats','order'):
            command += ['--'+name.replace('_','-'), str(getattr(args, name))]
        command += ['--metric', metric]
        env = dict(os.environ)
        if metric == 'heap':
            env['DYLD_INSERT_LIBRARIES'] = str(args.heap_probe.resolve())
        done = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        reports.append(json.loads(done.stdout))
    text = json.dumps(reports, indent=2, default=str)
    if args.output:
        args.output.write_text(text+'\n')
    print(text, flush=True)

if __name__ == '__main__':
    main()
