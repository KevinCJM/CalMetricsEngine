import os
from pathlib import Path
import subprocess
import sys
stage=Path(__file__).resolve().parent
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMBA_NUM_THREADS='1')
for name,backend,reference in (
 ('memory-ordinary-dense','ordinary','service'),
 ('memory-prepared-dense','prepared','service'),
 ('memory-service-njit-dense','njit','service'),
 ('memory-shared-njit-dense','njit','shared')):
 print('START '+name,flush=True)
 command=[sys.executable,str(stage/'memory.py'),'--install',str(stage/'install-dense'),
          '--identity-manifest',str(stage/'identity-dense.json'),'--heap-probe',str(stage/'heap_probe.dylib'),
          '--backend',backend,'--reference',reference,'--output',str(stage/(name+'.json'))]
 with (stage/(name+'.log')).open('w') as log:
  run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 if run.returncode:
  print((stage/(name+'.log')).read_text()[-4000:],flush=True)
  raise SystemExit(run.returncode)
 print('DONE '+name,flush=True)
