import os
from pathlib import Path
import subprocess
import sys
root=Path('/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine')
stage=Path(__file__).resolve().parent
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMBA_NUM_THREADS='1')
commands=[
 ('native-dense',['/opt/homebrew/bin/ctest','--test-dir','/private/tmp/calmetrics-covchol-next/wheel-build','--output-on-failure']),
 ('asan-build-dense',['/opt/homebrew/bin/cmake','--build','/private/tmp/calmetrics-covchol-next/asan-build','--parallel','3']),
 ('asan-dense',['/opt/homebrew/bin/ctest','--test-dir','/private/tmp/calmetrics-covchol-next/asan-build','--output-on-failure']),
 ('off-build-dense',['/opt/homebrew/bin/cmake','--build','/private/tmp/calmetrics-covchol-next/off-build','--parallel','3']),
 ('off-native-dense',['/opt/homebrew/bin/ctest','--test-dir','/private/tmp/calmetrics-covchol-next/off-build','--output-on-failure']),
]
for name,command in commands:
 print('START '+name,flush=True)
 with (stage/(name+'.log')).open('w') as log:
  run=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 if run.returncode:
  print((stage/(name+'.log')).read_text()[-4000:],flush=True)
  raise SystemExit(run.returncode)
 print('DONE '+name,flush=True)
