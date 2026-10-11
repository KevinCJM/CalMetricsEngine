"""Complete Scheduler covariance/Cholesky versus the preceding wheel and warm NJIT.

All timed results are independent, fully materialized matrices. Six rotated
rounds retain the preceding result until the next call returns. Fresh memory
children include preparation, first call and two warm retained-result calls.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,ctypes,gc,hashlib,json,statistics,subprocess,sys,threading,time
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
ROOT=Path('/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine')
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_memory import HeapStats
BACKENDS=('old','new','njit')
OLD=Path('/private/tmp/calmetrics-covchol-reuse/final/install-new')
OLD_ID='a63f72ba69ac969b2a60f90254ad15e8075a05ae15639cc9fe7561b229efa699'
NEW_ID='91267c23ac014c947e67ae31054dddc8aaa102fadd12f2ce4331c87752c63758'
REFERENCE=None

def cases():
    result=[dict(n=n,d=d,layout=order,entry='ordinary',output_order='C')
            for n,d in ((63,8),(529,512),(1041,1024),(2065,2048),(4113,4096)) for order in ('c','f')]
    result.append(dict(n=2065,d=2048,layout='f',entry='prepared',output_order='C'))
    result.append(dict(n=2065,d=2048,layout='f',entry='ordinary',output_order='F'))
    return result

def data(case):
    x=np.random.default_rng(20260930).normal(size=(case['n'],case['d']))
    if case['layout']=='f':x=np.asfortranarray(x)
    x.flags.writeable=False
    return x

def setup(case,backend,x):
    global REFERENCE
    if backend=='njit':
        if REFERENCE is None:
            from numba import njit,set_num_threads
            set_num_threads(1)
            @njit(nogil=True)
            def reference(a):return np.linalg.cholesky(np.cov(a.T))
            REFERENCE=reference
        from numba import typeof
        if (typeof(x),) not in REFERENCE.signatures:
            REFERENCE(x) # compile this actual readonly signature before the epoch
        def run():
            value=REFERENCE(x)
            if case['output_order']=='F':value=np.asfortranarray(value)
            return None,value
        return None,run
    from calmetrics_engine import AdaptiveScheduler,GraphCompiler
    g=GraphCompiler({'x':dict(kind='matrix',axes=['time','asset'],shape=['T','D'])}).compile('cholesky(covariance(x))',output_order=case['output_order'])
    scheduler=AdaptiveScheduler(cpu_budget=1)
    bound={'x':x};start=np.zeros(1,np.int64);end=np.array([len(x)],np.int64)
    if case['entry']=='prepared':
        prepared=scheduler.prepare_execution(g,bound,start,end)
        function=prepared.run_snapshot
    else:function=lambda:scheduler.execute(g,bound,start,end)
    def run():
        result=function()
        return result,np.asarray(result.outputs[0].values[0])
    return scheduler,run

def digest(x):return hashlib.sha256(x.tobytes()).hexdigest()
def identity(backend,result):
    pools=threadpool_info()
    assert all(p.get('num_threads')==1 for p in pools),pools
    if backend=='njit':
        import numba
        return dict(backend='numba-numpy-lapack',numba=numba.__version__,threads=pools)
    actual=result[0].audit['engine_build_id']
    expected=OLD_ID if backend=='old' else NEW_ID
    assert actual==expected,(actual,expected)
    return dict(backend=backend,engine_build_id=actual,threads=pools)

def validate(x,result):
    y=result[1]
    assert y.shape==(x.shape[1],x.shape[1]) and y.dtype==np.float64
    assert not np.shares_memory(x,y)
    assert np.isfinite(y).all() and np.all(np.diag(y)>0)
    assert not np.any(np.triu(y,1))
    return dict(shape=y.shape,dtype=str(y.dtype),sha256=digest(y),independent=True)

def worker(args):
    owner=None;retained=None
    with threadpool_limits(1):
        try:
            for line in sys.stdin:
                command=json.loads(line)
                if command['op']=='stop':
                    print(json.dumps(dict(stopped=True)),flush=True);break
                if command['op']=='init':
                    if owner:owner.close()
                    retained=None
                    case=command['case'];x=data(case);before=digest(x)
                    owner,function=setup(case,args.worker,x);retained=function()
                    verification=validate(x,retained)
                    assert digest(x)==before
                    np.save(args.output/(args.worker+'.npy'),retained[1])
                    response=dict(input_sha256=before,identity=identity(args.worker,retained),verification=verification)
                else:
                    wall=[];cpu=[]
                    for _ in range(command['repeat']):
                        w=time.perf_counter_ns();c=time.process_time_ns()
                        retained=function()
                        cpu.append(time.process_time_ns()-c);wall.append(time.perf_counter_ns()-w)
                    response=dict(wall_ns=wall,cpu_ns=cpu,audit=retained[0].audit if retained[0] is not None else {})
                print(json.dumps(response),flush=True)
        finally:
            if owner:owner.close()

def environment(backend,heap=False):
    env=dict(os.environ,PYTHONPATH=str(OLD if backend=='old' else HERE/'install-final'))
    env.pop('DYLD_INSERT_LIBRARIES',None)
    if heap:env['DYLD_INSERT_LIBRARIES']='/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib'
    return env

def timing(args, selected):
    children={};logs=[];records=[];cleanup={}
    def send(key,command):
        child=children[key];child.stdin.write(json.dumps(command)+'\n');child.stdin.flush()
        response=child.stdout.readline()
        if not response:raise RuntimeError(f'{key} worker failed')
        return json.loads(response)
    try:
        for key in BACKENDS:
            log=(args.output/(key+'.log')).open('x');logs.append(log)
            children[key]=subprocess.Popen([sys.executable,__file__,'--worker',key,'--output',str(args.output)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=environment(key))
        for case in selected:
            ids={k:send(k,dict(op='init',case=case)) for k in BACKENDS}
            assert len({v['input_sha256'] for v in ids.values()})==1
            arrays={k:np.load(args.output/(k+'.npy')) for k in BACKENDS}
            if case['d'] >= 2048:
                np.testing.assert_allclose(arrays['old'],arrays['new'],rtol=1e-12,atol=1e-12)
            else:
                np.testing.assert_array_equal(arrays['old'].view(np.uint64),arrays['new'].view(np.uint64))
            close=np.isclose(arrays['njit'],arrays['old'],rtol=1e-11,atol=1e-11)
            parity=dict(native='rtol=atol=1e-12 for large C roots; bitwise otherwise',njit_rtol=1e-11,njit_atol=1e-11,
                        njit_within_tolerance=bool(close.all()),njit_outside_tolerance=int(np.count_nonzero(~close)),
                        njit_max_abs=float(np.max(np.abs(arrays['njit']-arrays['old']))))
            del arrays
            samples={k:dict(wall_ns=[],cpu_ns=[]) for k in BACKENDS};rounds=[];last={}
            for turn in range(6):
                order=BACKENDS[turn%3:]+BACKENDS[:turn%3];row=dict(order=order,samples={})
                for k in order:
                    last[k]=send(k,dict(op='run',repeat=7 if case['d']<512 else 1))
                    row['samples'][k]={field:last[k][field] for field in samples[k]}
                    for field in samples[k]:samples[k][field].extend(last[k][field])
                rounds.append(row)
                (args.output/'progress.json').write_text(json.dumps(dict(case=case,completed_round=turn+1),indent=2))
                print(case,'round',turn+1,flush=True)
            medians={k:statistics.median(v['wall_ns'])/1e6 for k,v in samples.items()}
            records.append(dict(case=case,identities=ids,parity=parity,samples=samples,rounds=rounds,median_ms=medians,audits={k:v['audit'] for k,v in last.items()}))
            (args.output/'timing.json').write_text(json.dumps(records,indent=2));print(medians,flush=True)
    finally:
        for k,child in children.items():
            try:
                if child.poll() is None:cleanup[k]=send(k,dict(op='stop'))
            finally:
                child.stdin.close()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait();cleanup[k]=dict(terminated=True)
                child.stdout.close()
        for log in logs:log.close()
        (args.output/'cleanup.json').write_text(json.dumps(cleanup,indent=2))

def memory_child(args):
    case=json.loads(args.case);x=data(case);before=digest(x);owner=None
    with threadpool_limits(1):
        if args.backend=='njit':
            _,function=setup(case,args.backend,x);del function
        else:import calmetrics_engine
        # Avoid an unmeasured native warm call; its retained arena must be in epoch.
        gc.collect()
        try:
            if args.memory=='heap':
                probe=ctypes.CDLL('/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib')
                stats={k:HeapStats() for k in ('setup','cold','cold_released','warm_two_results')};end=HeapStats()
                probe.cme_heap_begin()
                try:
                    owner,function=setup(case,args.backend,x);probe.cme_heap_snapshot(ctypes.byref(stats['setup']))
                    probe.cme_heap_reset_peak();result=function();probe.cme_heap_snapshot(ctypes.byref(stats['cold']))
                    result=None;probe.cme_heap_snapshot(ctypes.byref(stats['cold_released']))
                    probe.cme_heap_reset_peak()
                    for _ in range(2):result=function()
                    probe.cme_heap_snapshot(ctypes.byref(stats['warm_two_results']))
                finally:probe.cme_heap_end(ctypes.byref(end))
                assert not end.table_overflows
                measured={k:{name:getattr(v,name) for name,_ in v._fields_} for k,v in stats.items()}
            else:
                import psutil
                process=psutil.Process();stop=threading.Event();ready=threading.Event();peak=[0]
                def watch():
                    ready.set()
                    while not stop.wait(.001):peak[0]=max(peak[0],process.memory_info().rss)
                monitor=threading.Thread(target=watch);monitor.start();ready.wait();time.sleep(.02)
                try:
                    base=process.memory_info().rss;peak[0]=base
                    owner,function=setup(case,args.backend,x);result=function()
                    cold=max(peak[0],process.memory_info().rss);result=None;gc.collect()
                    warmbase=process.memory_info().rss;peak[0]=warmbase
                    for _ in range(2):result=function()
                    warm=max(peak[0],process.memory_info().rss)
                    measured=dict(cold_baseline=base,cold_peak=cold,cold_delta=cold-base,warm_baseline=warmbase,warm_peak=warm,warm_delta=warm-warmbase)
                finally:stop.set();monitor.join()
            verification=validate(x,result);assert digest(x)==before
            report=dict(case=case,backend=args.backend,kind=args.memory,measurement=measured,verification=verification,identity=identity(args.backend,result),audit=result[0].audit if result[0] is not None else {})
        finally:
            if owner:owner.close()
    print(json.dumps(report))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',choices=BACKENDS);parser.add_argument('--memory',choices=('heap','rss'))
    parser.add_argument('--backend',choices=BACKENDS);parser.add_argument('--case');parser.add_argument('--mode',choices=('timing','memory'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.worker:return worker(args)
    if args.memory:return memory_child(args)
    selected=[c for c in cases() if c['d'] in (8,2048,4096)] if args.mode=='timing' else [c for c in cases() if c['d'] in (2048,4096) and c['entry']=='ordinary' and c['output_order']=='C']
    design=dict(cases=selected,cpu=1,rounds=6,script_sha256=digest(np.frombuffer(Path(__file__).read_bytes(),np.uint8)),old=OLD_ID,new=NEW_ID,scope='complete independent result; setup included in memory, prepared compared with compiled NJIT',heap_excludes='stack/mmap/Python arenas; RSS supplementary; one fresh process per cell')
    with (args.output/'design.json').open('x') as f:json.dump(design,f,indent=2)
    if args.mode=='timing':return timing(args, selected)
    records=[]
    for index,case in enumerate(selected):
        for backend in BACKENDS:
            for kind in ('heap','rss'):
                command=[sys.executable,__file__,'--memory',kind,'--backend',backend,'--case',json.dumps(case),'--output',str(args.output)]
                result=subprocess.run(command,env=environment(backend,kind=='heap'),capture_output=True,text=True)
                label=f'{index}-{backend}-{kind}'
                (args.output/(label+'.log')).write_text(result.stderr);(args.output/(label+'.json')).write_text(result.stdout)
                result.check_returncode();records.append(json.loads(result.stdout))
                (args.output/'memory.json').write_text(json.dumps(records,indent=2));print(label,'complete',flush=True)
    for case in selected:
        assert all(v['verification']['independent'] for v in records if v['case']==case)

if __name__=='__main__':main()
