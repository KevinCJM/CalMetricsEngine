import os,subprocess,sys
from pathlib import Path
stage=Path(__file__).resolve().parent
root=Path('/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine')
env=dict(os.environ,PATH='/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin',
         PYTHONPATH='/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps',
         PIP_NO_INDEX='1',CMAKE_BUILD_PARALLEL_LEVEL='3',MACOSX_DEPLOYMENT_TARGET='15.0',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMBA_NUM_THREADS='1')
commands=[('build-dense',[sys.executable,'-m','pip','wheel','.','--no-deps','--no-build-isolation',
                  '-Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build','-w',str(stage/'wheels-dense')])]
for name,command in commands:
    print('START '+name,flush=True)
    with (stage/(name+'.log')).open('w') as log:
        run=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:
        print((stage/(name+'.log')).read_text()[-5000:],flush=True);raise SystemExit(run.returncode)
wheels=list((stage/'wheels-dense').glob('*.whl'));assert len(wheels)==1
with (stage/'install-dense.log').open('w') as log:
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(stage/'install-dense'),str(wheels[0])],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('BUILD INSTALLED',flush=True)
