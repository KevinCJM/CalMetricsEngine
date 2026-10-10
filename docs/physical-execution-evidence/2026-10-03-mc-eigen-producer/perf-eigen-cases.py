"""Serial production-wheel acceptance for the adopted ordered Eigen producer."""
import json
import os
from pathlib import Path
import subprocess
import sys

stage = Path(__file__).resolve().parent
environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", NUMBA_NUM_THREADS="1")
cases = (
    ("large-single-shared-eigen", 2_000_000, 200, 4, 32, 1, "C", "ordinary"),
    ("large-shared-eigen", 2_000_000, 200, 4, 32, 8, "C", "ordinary"),
    ("wide-shared-eigen", 250_000, 5, 32, 32, 8, "C", "ordinary"),
    ("large-f-shared-eigen", 2_000_000, 200, 4, 32, 8, "F", "ordinary"),
    ("large-prepared-shared-eigen", 2_000_000, 200, 4, 32, 8, "C", "prepared"),
    ("small-shared-eigen", 252, 1, 4, 32, 1, "C", "ordinary"),
)
for name, n, steps, factors, assets, combinations, order, mode in cases:
    output = stage / (name + ".json")
    if output.exists():
        raise RuntimeError("refusing to overwrite existing evidence: " + str(output))
    command = [sys.executable, str(stage / "bench.py"), "--install", str(stage / "install-eigen"),
               "--identity-manifest", str(stage / "identity-eigen.json"), "--reference", "shared",
               "--cpu", "1", "--rounds", "6", "--n", str(n), "--steps", str(steps),
               "--factors", str(factors), "--assets", str(assets), "--combinations", str(combinations),
               "--order", order, "--mode", mode, "--output", str(output)]
    print("START " + name, flush=True)
    with (stage / (name + ".log")).open("w") as log:
        run = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        print((stage / (name + ".log")).read_text()[-4000:], flush=True)
        raise SystemExit(run.returncode)
    report = json.loads(output.read_text())
    print("DONE " + name, flush=True)
