"""Actual old/new ordinary Scheduler GEMM roots vs one-CPU warmed NJIT.

Nine balanced rounds; no native threadpool override. Reuse the frozen complete-
request harness with owned/materialized results and fresh cold/warm heap/RSS.
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
STAGE = Path(__file__).resolve().parent
for filename, expected in {
    'check_large_matrix_performance.py':'2f99bf41adac0926e30aa2045fa0f6bb67ea938826bf6a4c69ed6cf278c10518',
    'benchmark_memory.py':'b3862bb22e58f01f874bad0fd8466a3eb63ac4f3803186846184aafb47f4d7f8',
}.items():
    assert hashlib.sha256((STAGE/'tools'/filename).read_bytes()).hexdigest() == expected
sys.path.insert(0,str(STAGE/'tools'))
import check_large_matrix_performance as bench
import numpy as np
OLD_BUILD = 'c9f75aed3f4d38fd44caf893b9faba83619c1a094f006870ce6062f814a238ec'
NEW_BUILD = os.environ['CME_OPT_NEW_BUILD']
assert len(NEW_BUILD) == 64
_backend = None
_input_snapshot = None
_reference = None
_base_audit = bench.audit
_base_heap, _base_rss = bench.measure_heap, bench.measure_rss


def cases():
    def case(label, shape, layout='cc', transposed=False, domain='signed', order='auto'):
        return dict(name='matmul',label=label,shape=list(shape),n=shape[0],layout=layout,
                    transposed=transposed,domain=domain,order=order,entry='ordinary',rtol=2e-12,atol=2e-14)
    selected = [case('small',(32,32,32)),
                case('positive',(2048,2048,2048),domain='positive'),
                case('signed_cc',(2048,2048,2048)),
                case('signed_ff_transpose',(2048,2048,2048),'ff',True),
                case('large_ff_transpose',(4096,4096,4096),'ff',True),
                case('wide_ff_transpose',(4096,512,4096),'ff',True),
                case('sparse_ff_transpose',(2048,2048,2048),'ff',True,domain='sparse'),
                case('mixed_sign_rejected',(2048,2048,2048),'cf',domain='rejected'),
                case('explicit_f',(2048,2048,2048),'ff',order='F')]
    return [c for c in selected if c['label'] in {'signed_cc','large_ff_transpose'}] if os.environ.get('CME_GEMM_MEMORY')=='1' else selected


def inputs(case):
    global _input_snapshot
    m,k,n = case['shape']
    rng = np.random.default_rng(20261002+m+k+n)
    a = np.array(rng.random((m,k)),order=case['layout'][0].upper())
    b = np.array(rng.random((k,n)),order=case['layout'][1].upper())
    if case['domain'] != 'positive':
        a[::2] *= -1
        b[:,::2] *= -1
    if case['domain'] == 'rejected':
        b[-1,-1] *= -1
    if case['domain'] == 'sparse':
        a[:,::3] = b[::5,:] = 0.0
        a[0] = b[:,0] = 0.0
        a[2,k//2:] = b[:k//2,2] = 0.0
    a.flags.writeable = b.flags.writeable = False
    data = dict(a=a,b=b)
    _input_snapshot = bench.input_identity(data)
    return data


def setup(case,backend,data):
    global _backend,_reference
    _backend = backend
    if backend == 'njit':
        from numba import njit,typeof,set_num_threads
        set_num_threads(1)
        if _reference is None:
            @njit(nogil=True)
            def reference(a,b):
                return a @ b
            _reference = reference
        args = (data['a'],data['b'])
        signature = tuple(typeof(x) for x in args)
        if signature not in _reference.signatures:
            _reference.compile(signature)
        def run():
            result = _reference(*args)
            if case['transposed']:
                result = result.T
            result = np.asfortranarray(result) if output_f(case) else np.ascontiguousarray(result)
            result.flags.writeable = False
            return result
        return None,run
    from calmetrics_engine import AdaptiveScheduler,GraphCompiler
    decl = {'a':dict(kind='matrix',axes=['asset','asset'],shape=['M','K']),
            'b':dict(kind='matrix',axes=['asset','asset'],shape=['K','N'])}
    expr = 'transpose(matmul(a,b))' if case['transposed'] else 'matmul(a,b)'
    graph = GraphCompiler(decl).compile(expr,output_order=case['order'])
    scheduler = AdaptiveScheduler(cpu_budget=1)
    bounds = np.zeros(1,np.int64)
    return scheduler,lambda: bench.MaterializedResult(scheduler.execute(graph,data,bounds,bounds))


def output_f(case):
    return case['order']=='F' or (case['order']=='auto' and case['transposed'])


def invariants(case,data,actual):
    m,_,n=case['shape']
    assert actual.shape == ((n,m) if case['transposed'] else (m,n))
    assert actual.dtype == np.float64 and not actual.flags.writeable
    assert actual.flags.f_contiguous if output_f(case) else actual.flags.c_contiguous
    report={}
    if _backend != 'old':
        output=Path(sys.argv[sys.argv.index('--output')+1])
        expected=bench.load_outputs(output/'old.npz')[0]
        for predicate in (np.isfinite,np.isnan,np.signbit,lambda v:v==0):
            np.testing.assert_array_equal(predicate(actual),predicate(expected))
        error=np.abs(actual-expected)
        allowed=2e-14+2e-12*np.abs(expected)
        report=dict(max_absolute_error=float(error.max()),max_tolerance_fraction=float((error/allowed).max()),
                    changed_bits=int(np.count_nonzero(actual.view(np.uint64)!=expected.view(np.uint64))))
        assert np.all(error<=allowed),report
    (STAGE/'timing'/f"validated-{case['label']}-{_backend}.json").write_text(json.dumps(dict(input=_input_snapshot,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),numerical_checks=report),indent=2))
    return report


def audit(case,backend,data,result):
    assert bench.input_identity(data)==_input_snapshot
    report=_base_audit(case,backend,data,result)
    if backend!='njit':
        assert report['engine_build_id']==(OLD_BUILD if backend=='old' else NEW_BUILD)
        count=sum(c['accelerate_matmul_calls'] for c in report['native_chunks'])
        expected=int(case['n']>=512 and (case['domain']=='positive' or (backend=='new' and case['domain']!='rejected')))
        assert count==expected,(case,backend,count)
    actual=np.asarray(bench.value(result))
    assert not actual.flags.writeable
    assert actual.flags.f_contiguous if output_f(case) else actual.flags.c_contiguous
    report.update(input_unchanged=True,output_shape=list(actual.shape),output_order='F' if output_f(case) else 'C',
                  output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
    validated=json.loads((STAGE/'timing'/f"validated-{case['label']}-{backend}.json").read_text())
    assert validated['input']==_input_snapshot
    assert validated['output_sha256']==report['output_sha256']
    report['matches_numerically_validated_timing_output']=True
    return report


bench.__file__=str(Path(__file__).resolve())
bench.cases,bench.case_id,bench.inputs,bench.setup=cases,lambda case:case['label'],inputs,setup
bench.invariants,bench.audit=invariants,audit
bench.ROUNDS=9
bench.measure_heap=lambda setup,lib,repeats=11:_base_heap(setup,lib,repeats=3)
bench.measure_rss=lambda function,repeats=11:_base_rss(function,repeats=3)
if __name__=='__main__':
    for option,path in (('--old-install','/private/tmp/calmetrics-highway-integration-next/install'),('--new-install',str(STAGE/'install'))):
        if not any(arg==option or arg.startswith(option+'=') for arg in sys.argv[1:]):sys.argv.extend((option,path))
    bench.main()
