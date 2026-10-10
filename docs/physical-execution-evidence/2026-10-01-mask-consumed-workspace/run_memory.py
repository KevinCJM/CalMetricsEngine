"""Serial fresh-process heap/RSS evidence for 12 complete owned requests.

All reports, including regressions, are retained. Existing directories are never
reused/overwritten. No numerical work executes merely by importing this driver.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_INSTALL = Path("/private/tmp/calmetrics-chol-group-next/install")
NEW_INSTALL = HERE / "install"
NEW_BUILD = "ccca6b67a5220ea2134a4743d281b80c5af4c096e7d41c3d5b747bb8de70e331"
HEAP_PROBE = Path(os.environ.get("CME_HEAP_PROBE", "/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib"))


def filters():
    result = []
    for name in ("median", "quantile"):
        result.append((name, f"{name}_4194304_c_auto_ordinary_random_default_densityna"))
    for name in ("median_where", "quantile_where"):
        for density in ("0.55", "0.01", "1"):
            result.append((f"{name}_density{density}",
                           f"{name}_4194304_c_auto_ordinary_random_default_density{density}"))
        result.append((f"{name}_16M_sparse", f"{name}_16777216_c_auto_ordinary_random_default_density0.01"))
        result.append((f"{name}_prepared_sparse", f"{name}_4194304_c_auto_prepared_random_default_density0.01"))
    return result


def main():
    if os.environ.get("CME_OPT_NEW_BUILD", NEW_BUILD) != NEW_BUILD:
        raise RuntimeError("CME_OPT_NEW_BUILD disagrees with this frozen candidate")
    if not HEAP_PROBE.is_file():
        raise FileNotFoundError(HEAP_PROBE)
    env = dict(os.environ, CME_OPT_NEW_BUILD=NEW_BUILD)
    for label, selected in filters():
        target = HERE / "memory" / label
        if target.exists():
            raise RuntimeError(f"preserve prior memory evidence: {target}")
        print(f"starting {label}", flush=True)
        subprocess.run([sys.executable, str(HERE / "benchmark.py"),
                        "--old-install", str(OLD_INSTALL),
                        "--new-install", str(NEW_INSTALL),
                        "--mode", "memory", "--heap-probe", str(HEAP_PROBE),
                        "--output", str(target), "--case-filter", selected],
                       env=env, check=True)


if __name__ == "__main__":
    main()
