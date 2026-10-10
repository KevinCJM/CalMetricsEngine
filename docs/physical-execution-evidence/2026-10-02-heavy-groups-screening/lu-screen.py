import os
for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,os.environ.get("CME_NATIVE_INSTALL","/private/tmp/calmetrics-mask-consumed-next/install"))
from calmetrics_engine import operators as op

ROOT=Path(__file__).parent
VARIANT=os.environ.get('CME_LU_VARIANT','lu-probe')
MODES=json.loads(os.environ.get('CME_LU_MODES','[0,1,2,3]'))
lib=ctypes.CDLL(str(ROOT/f'{VARIANT}.dylib'))
fn=lib.cme_lu_register
fn.argtypes=[ctypes.c_void_p,ctypes.c_ssize_t,ctypes.c_ssize_t,ctypes.c_void_p,
             ctypes.c_ssize_t,ctypes.c_ssize_t,ctypes.c_size_t,ctypes.c_size_t,
             ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
fn.restype=ctypes.c_int
lib.cme_lu_rounding.argtypes=[ctypes.c_int]
SENTINEL=-913.25
def call(a,b,mode):
    storage=np.full(b.size+8,SENTINEL);y=storage[4:-4].reshape(b.shape)
    err=ctypes.create_string_buffer(128)
    code=fn(a.ctypes.data,*[s//8 for s in a.strides],b.ctypes.data,
            *[s//8 for s in b.strides],a.shape[0],b.shape[1],y.ctypes.data,mode,err)
    assert np.all(storage[:4]==SENTINEL) and np.all(storage[-4:]==SENTINEL)
    return y,err.value.decode() if code else None
def same(a,b):
    # Preserve all bits except a NaN payload, matching canonical test policy.
    return (np.array_equal(np.isnan(a),np.isnan(b)) and
            np.array_equal(a[~np.isnan(a)].view(np.uint64),b[~np.isnan(b)].view(np.uint64)))
def native(a,b):
    out=np.full(b.shape,SENTINEL)
    try: op.solve_multi(a,b,out=out);return out,None
    except ValueError as e:return out,str(e)
def layouts(a,b):
    yield 'c',a,b
    yield 'f',np.asfortranarray(a),np.asfortranarray(b)
    yield 'reverse',a[::-1,::-1],b[::-1]
    aa=np.zeros((a.shape[0]*2,a.shape[1]*2));aa[::2,::2]=a
    bb=np.zeros((b.shape[0]*2,b.shape[1]*2));bb[::2,::2]=b
    yield 'sliced',aa[::2,::2],bb[::2,::2]
    yield 'broadcast',a,np.broadcast_to(b[:1,:1],b.shape)
def legacy_fixtures():
    rng=np.random.default_rng(20261001)
    for n in (0,1,16,31,32,127,128,129,257):
        for d in (0,1,3,8):
            a=rng.normal(size=(n,n));b=rng.normal(size=(n,d))
            if n:a[np.diag_indices(n)]+=np.abs(a).sum(axis=1)+1
            for lay,x,y in layouts(a,b):yield f'dense-{n}-{d}-{lay}',x,y
    n=129;b=rng.normal(size=(n,3))
    for kind in ('swap','tie','singular','threshold','above','below','hilbert','overflow','nan-a','inf-a','nan-b','inf-b','backsolve-overflow','negative-zero'):
        a=np.eye(n);bb=b.copy()
        if kind=='swap':a=a[::-1].copy()
        if kind=='tie':a[:2,:2]=[[1,2],[-1,3]]
        if kind=='singular':a[-1]=a[-2]
        if kind=='threshold':a[-1,-1]=1e-14
        if kind=='above':a[-1,-1]=np.nextafter(1e-14,np.inf)
        if kind=='below':a[-1,-1]=np.nextafter(1e-14,0)
        if kind=='hilbert':a=1./(np.arange(n)[:,None]+np.arange(n)+1)
        if kind=='overflow':a[:2,:2]=[[1e308,1e308],[1e308,-1e308]]
        if kind=='nan-a':a[-1,-1]=np.nan
        if kind=='inf-a':a[-1,-1]=np.inf
        if kind=='nan-b':bb[-1,1]=np.nan
        if kind=='inf-b':bb[-1,1]=np.inf
        if kind=='backsolve-overflow':a[-1,-1]=2e-14;bb[-1,1]=1e308
        if kind=='negative-zero':a[a==0]=-0.;bb[0]=[-0.,0.,-0.]
        yield kind,a,bb
    for n in (128,129,257):
        for seed in range(6):
            a=np.random.default_rng(seed).normal(size=(n,n));bb=rng.normal(size=(n,3))
            yield f'pivots-{n}-{seed}',a,bb
def group_fixtures():
    # Actual n>=1024 admission, distinct group and column-tail boundaries.
    rng=np.random.default_rng(20261002)
    sizes=json.loads(os.environ.get('CME_LU_SCREEN_SIZES','[1024,1025,1031,1032]'))
    for n in sizes:
        a=rng.normal(size=(n,n));b=rng.normal(size=(n,3))
        a[np.diag_indices(n)]+=np.abs(a).sum(axis=1)+1
        for lay,x,y in layouts(a,b):
            if lay!='broadcast':yield f'group-dense-{n}-{lay}',x,y
    n=1024;b=rng.normal(size=(n,3))
    for kind in ('swaps','tie','singular','overflow','negative-zero'):
        a=np.eye(n);bb=b.copy()
        if kind=='swaps':a=a[::-1].copy()
        if kind=='tie':a[:2,:2]=[[1,2],[-1,3]]
        if kind=='singular':a[32]=a[31]
        if kind=='overflow':a[:2,:2]=[[1e308,1e308],[1e308,-1e308]]
        if kind=='negative-zero':a[a==0]=-0.;bb[0]=[-0.,0.,-0.]
        yield f'group-{kind}',a,bb
    for seed in (0,1):
        a=np.random.default_rng(seed).normal(size=(n,n))
        yield f'group-pivots-{n}-{seed}',a,rng.normal(size=(n,3))

def rhs_prefix_fixtures():
    n=129;d=3
    payload=np.array([0x7ff8000000001234],dtype=np.uint64).view(np.float64)[0]
    for position in (0,(n*d)//2,n*d-1):
        for kind,value in (('nan-payload',payload),('infinity',np.inf)):
            a=np.eye(n);b=np.arange(n*d,dtype=np.float64).reshape(n,d)+1
            b.flat[position]=value
            for lay,x,y in layouts(a,b):
                if lay!='broadcast':yield f'rhs-prefix-{kind}-{position}-{lay}',x,y

def fixtures():
    scope=os.environ.get('CME_LU_SCOPE','all')
    assert scope in ('all','legacy','group','prefix')
    if scope in ('all','legacy'):yield from legacy_fixtures()
    if scope in ('all','group'):yield from group_fixtures()
    if scope in ('all','prefix'):yield from rhs_prefix_fixtures()

def copied_bits(a,b):
    return np.array_equal(np.ascontiguousarray(a).view(np.uint64),
                          np.ascontiguousarray(b).view(np.uint64))

def run():
    records=[]
    try:
        for rounding in json.loads(os.environ.get('CME_LU_ROUNDINGS','[0,1,2,3]')):
            assert lib.cme_lu_rounding(rounding)==0
            for name,a,b in fixtures():
                before=[x.copy() for x in (a,b)];a.flags.writeable=b.flags.writeable=False
                ref,error=native(a,b)
                if name.startswith("rhs-prefix-"):
                    assert error=="NON_FINITE_RESULT" and copied_bits(ref,b),name
                for mode in MODES:
                    actual,err=call(a,b,mode)
                    records.append(dict(case=name,rounding=rounding,mode=mode,error=err,
                        expected_error=error,pass_=error==err and same(actual,ref) and
                        (not name.startswith("rhs-prefix-") or copied_bits(actual,b))))
                assert copied_bits(a,before[0]) and copied_bits(b,before[1])
    finally:lib.cme_lu_rounding(0)
    out=dict(scope='isolated ABI candidate versus installed production output/error prefix',
             modes=MODES,scope_selection=os.environ.get('CME_LU_SCOPE','all'),
             library_sha256=hashlib.sha256((ROOT/f'{VARIANT}.dylib').read_bytes()).hexdigest(),
             records=records,passed=sum(x['pass_'] for x in records),total=len(records))
    (ROOT/f'{VARIANT}-screen.json').write_text(json.dumps(out,indent=2))
    bad=[x for x in records if not x['pass_']]
    print(dict(passed=out['passed'],total=out['total'],failures=bad[:12]),flush=True)
    return bool(bad)
if __name__=='__main__':raise SystemExit(run())
