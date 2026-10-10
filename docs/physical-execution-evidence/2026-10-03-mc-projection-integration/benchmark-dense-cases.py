import os
from pathlib import Path
import subprocess
import sys
stage=Path(__file__).resolve().parent
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMBA_NUM_THREADS='1')
cases=[
 ('large-service-dense',['--n','2000000','--steps','200','--assets','32','--combinations','8','--reference','service']),
 ('large-shared-dense-f',['--n','2000000','--steps','200','--assets','32','--combinations','8','--reference','shared','--order','F']),
 ('medium-shared-dense',['--n','200000','--steps','200','--assets','16','--combinations','4','--reference','shared']),
 ('large-single-shared-dense',['--n','2000000','--steps','200','--assets','32','--combinations','1','--reference','shared']),
 ('small-single-shared-dense',['--n','252','--steps','252','--assets','1','--combinations','1','--reference','shared']),
 ('large-prepared-shared-dense',['--n','2000000','--steps','200','--assets','32','--combinations','8','--reference','shared','--mode','prepared']),
]
for name,options in cases:
 print('START '+name,flush=True)
 command=[sys.executable,str(stage/'bench.py'),'--install',str(stage/'install-dense'),
          '--identity-manifest',str(stage/'identity-dense.json'),'--rounds','6','--output',str(stage/(name+'.json'))]+options
 with (stage/(name+'.log')).open('w') as log:
  run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 if run.returncode:
  print((stage/(name+'.log')).read_text()[-4000:],flush=True)
  raise SystemExit(run.returncode)
 print('DONE '+name,flush=True)
