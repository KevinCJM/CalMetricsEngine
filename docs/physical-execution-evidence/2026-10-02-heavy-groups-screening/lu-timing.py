import os
for name in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import ctypes
import json
import platform
import statistics
import time
from pathlib import Path
import numpy as np
from numba import njit
import sys
REPOSITORY=Path(os.environ.get('CME_REPO','/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine'))
sys.path.insert(0,str(REPOSITORY/'tools'))
from check_large_matrix_performance import baseline_threads
import importlib.util
screen_path=Path(__file__).with_name('lu-screen.py')
spec=importlib.util.spec_from_file_location('lu_screen',screen_path)
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
fn,ROOT,same,VARIANT,MODES=(getattr(screen,name) for name in ('fn','ROOT','same','VARIANT','MODES'))

@njit(nogil=True)
def solve(a,b):
    return np.linalg.solve(a,b)

def setup(a,b,mode):
    arguments=(a.ctypes.data,*[s//8 for s in a.strides],b.ctypes.data,
               *[s//8 for s in b.strides],a.shape[0],b.shape[1])
    def run():
        y=np.empty(b.shape)
        err=ctypes.create_string_buffer(128)
        code=fn(*arguments,y.ctypes.data,mode,err)
        if code:raise AssertionError(err.value)
        return y
    return run

def main():
    records=[]
    cases=[(n,d,layout) for n,d in ((8,1),(128,3),(512,8),(1024,8),(2048,1),(2048,32),(4096,8))
           for layout in ('c','f')]
    cases=json.loads(os.environ.get('CME_LU_TIMING_CASES',json.dumps(cases)))
    for n,d,layout in cases:
        rng=np.random.default_rng(20261001)
        a=rng.normal(size=(n,n));a[np.diag_indices(n)]+=np.abs(a).sum(axis=1)+1
        b=rng.normal(size=(n,d))
        if layout=='f':a,b=np.asfortranarray(a),np.asfortranarray(b)
        a.flags.writeable=b.flags.writeable=False
        calls={str(mode):setup(a,b,mode) for mode in MODES}
        calls['njit']=lambda:solve(a,b)
        ref=setup(a,b,0)()
        holders={};thread_events=[]
        for key,run in calls.items():
            with baseline_threads('njit' if key=='njit' else 'new') as threads:
                holders[key]=run()
                if key!='njit':assert same(ref,holders[key]),key
                else:np.testing.assert_allclose(holders[key],ref,rtol=1e-11,atol=1e-11)
            # The helper mutates metadata on exit to include TLS restoration.
            thread_events.append(dict(phase='warm',backend=key,metadata=threads))
        del ref
        labels=list(calls);raw={k:[] for k in labels};inner=50 if n==8 else 1
        rounds=int(os.environ.get('CME_LU_ROUNDS',2*len(labels)))
        assert rounds>=2*len(labels) and rounds%(2*len(labels))==0
        orders=[]
        for round_ in range(rounds):
            start=round_ % len(labels)
            order=labels[start:]+labels[:start]
            if (round_//len(labels))%2:order=order[::-1]
            orders.append(order)
            for key in order:
                with baseline_threads('njit' if key=='njit' else 'new') as threads:
                    before=time.perf_counter_ns()
                    for _ in range(inner):
                        # Keep this backend's old independent result alive until
                        # its new call completes, then normally replace it.
                        holders[key]=calls[key]()
                    elapsed=time.perf_counter_ns()-before
                    raw[key].append(elapsed/inner)
                thread_events.append(dict(phase='timing',round=round_,backend=key,metadata=threads))
                assert holders[key].shape==b.shape
        for key in labels:
            if key!='njit':assert same(holders['0'],holders[key]),key
            else:np.testing.assert_allclose(holders[key],holders['0'],rtol=1e-11,atol=1e-11)
        median={k:statistics.median(v)/1e6 for k,v in raw.items()}
        record=dict(n=n,d=d,layout=layout,median_ms=median,samples_ns=raw,
                    candidate_output_bytes=b.nbytes,coefficient_workspace_bytes=a.nbytes,
                    schedule=orders,threads=thread_events,
                    ownership='one retained independent previous result per backend; replacement after call',
                    status_checked_every_native_call=True,parity_scope='warmup and final timed outputs',
                    output_layouts={key:dict(c_contiguous=value.flags.c_contiguous,
                                            f_contiguous=value.flags.f_contiguous,
                                            strides=list(value.strides))
                                    for key,value in holders.items()})
        records.append(record)
        (ROOT/f'{VARIANT}-timing.json').write_text(json.dumps(dict(
            scope='isolated C ABI + owned output + input copy; no Scheduler/Planner acceptance',
            cpu_budget=1,warmed_njit=True,platform=platform.platform(),
            thread_policy='baseline_threads(njit) outside each NJIT timed block; native context does not set TLS',
            records=records),indent=2))
        print(n,d,layout,{k:round(v,3) for k,v in median.items()},flush=True)
if __name__=='__main__':main()
