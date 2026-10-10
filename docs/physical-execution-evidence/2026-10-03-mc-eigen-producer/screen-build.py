"""Root-only serial complete-wheel screening; production source stays frozen."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

stage = Path(__file__).resolve().parent
root = stage / "source"
baseline = json.loads((stage.parent / "identity-revalidated.json").read_text())
for name, digest in baseline["source"].items():
    if name != "cpp/affine_projection.cpp":
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
assert hashlib.sha256((root / "cpp/affine_projection.cpp").read_bytes()).hexdigest() == "ff0119105ae53af372cb74f4426ef517f588a3404bf5eb481def15f451011003"
assert not (stage / "install").exists()
environment = dict(os.environ, PATH="/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin",
                   PYTHONPATH="/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps",
                   PIP_NO_INDEX="1", CMAKE_BUILD_PARALLEL_LEVEL="3", MACOSX_DEPLOYMENT_TARGET="15.0",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMBA_NUM_THREADS="1")
with (stage / "build.log").open("w") as log:
    subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation",
                    "-Cbuild-dir=" + str(stage / "build"), "-w", str(stage / "wheels")],
                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
wheels = list((stage / "wheels").glob("*.whl"))
assert len(wheels) == 1
with (stage / "install.log").open("w") as log:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(stage / "install"), str(wheels[0])],
                   env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
print("PASS: mirror wheel built and installed", flush=True)
