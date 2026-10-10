"""Root serial adjacent fresh-PID whole-owner RSS diagnostic; no gate retuning."""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time
BASE=Path('/private/tmp/calmetrics-mc-compact-integration')
OUT=Path(__file__).resolve().parent
CURRENT=Path('/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine')
DRIVER=BASE/'memory-owner-rss-production.py'
assert hashlib.sha256(DRIVER.read_bytes()).hexdigest()=='5a6c0797fb47db8d622aafea8afdd03424e46a573362ef60e7c0d99f281865b9'
versions={'old':(BASE/'parent-slots-final',OUT/'parent-slots-source'),
          'new':(BASE/'node-types-final',CURRENT)}
plan=[(i+1,v,'ordinary') for i,v in enumerate(('old','new','new','old','old','new'))]
plan += [(7,'old','njit'),(8,'new','njit')]
records=[]
assert not (OUT/'rss-owner-pair-status.json').exists()
for trial,version,backend in plan:
    stage,root=versions[version]
    identity=json.loads((stage/'identity.json').read_text())
    for n,h in identity['source'].items(): assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
    env=dict(os.environ,PYTHONPATH=str(stage/'install')+':/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps',
        CME_MC_SOURCE_ROOT=str(root),CME_MC_EXPECT_BUILD_ID=identity['build_id'],
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    assert not env.get('DYLD_INSERT_LIBRARIES')
    name=f'rss-owner-{trial:02d}-{version}-{backend}'
    output=OUT/(name+'.json')
    command=[sys.executable,str(DRIVER),'--install',str(stage/'install'),
        '--identity-manifest',str(stage/'identity.json'),'--heap-probe','/private/tmp/calmetrics-mc-projection-integration/heap_probe.dylib',
        '--backend',backend,'--case','short_many','--repeats','2','--metric','rss','--output',str(output)]
    print('START '+name,flush=True); started=time.time()
    with (OUT/(name+'.log')).open('x') as stream:
        stream.write(json.dumps(dict(command=command,build_id=identity['build_id'],source_root=str(root)))+'\n'); stream.flush()
        completed=subprocess.run(command,env=env,cwd=CURRENT,stdout=stream,stderr=subprocess.STDOUT)
    records.append(dict(trial=trial,version=version,backend=backend,exit_code=completed.returncode,
        elapsed_seconds=time.time()-started,output=output.name,build_id=identity['build_id']))
    (OUT/'rss-owner-pair-status.json').write_text(json.dumps(records,indent=2)+'\n')
    if completed.returncode:
        print((OUT/(name+'.log')).read_text()[-3000:],flush=True); raise SystemExit(completed.returncode)
    report=json.loads(output.read_text())[0]
    print('PASS '+name+' '+json.dumps(report['measurement']),flush=True)
assert len({json.loads((OUT/r['output']).read_text())[0]['fresh_pid'] for r in records})==8
print('ALL 8 fresh-PID observations completed',flush=True)
