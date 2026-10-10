"""Fresh-child memory baseline for the actual compact fixed-tape numerical chain."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine")
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(ROOT / "tools"))
import benchmark_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("heap", "rss"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "preserve terminal evidence"
    path = HERE / "profile-complete-reference.py"
    spec = importlib.util.spec_from_file_location("complete_profile", path)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    profile.set_num_threads(1)
    module, reference_sha = profile.freeze_reference()
    tiny = profile.bench.inputs(profile.np, n=24, steps=3, factors=4, assets=32, combinations=1)
    profile.run(tiny, module)
    data = profile.bench.inputs(profile.np, n=2_000_000, steps=200, factors=4, assets=32, combinations=8)
    before = profile.fingerprints(data)
    call = lambda: profile.run(data, module)
    probe = HERE / "heap_probe.dylib"
    if args.kind == "heap":
        _, _, report = benchmark_memory.measure_heap(lambda: (None, call), str(probe), repeats=2)
    else:
        report = benchmark_memory.measure_rss(call, repeats=2)
    assert before == profile.fingerprints(data)
    report.update(scope="healthy_fixed_transformed_tape_compact_projection_nav_fan_numeric_samples_contributions",
                  backend="controlled_frozen_njit_reference", source_sha256=profile.FROZEN_SOURCE_SHA256,
                  frozen_reference_sha256=reference_sha, cpu_budget=1,
                  n=2_000_000, steps=200, paths=10_000, factors=4, assets=32, combinations=8,
                  common_input_bytes=sum(data[key].nbytes for key in before),
                  common_inputs_and_tiny_signature_compilation="outside_incremental_epoch",
                  compact_result_contract="fan, 14 statistics, first 12 sample paths, asset contributions",
                  replacement_lifetime="previous and current compact independent results coexist",
                  input_unchanged=True, input_sha256=before,
                  probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  profile_driver_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  excluded=["coverage", "old_random_stream", "covariance_root", "factor_transform", "limits", "json_packaging", "first_fault_cases"],
                  blind_spots=["requested allocator bytes do not equal total RSS", "Python arenas and mmap are not completely tracked", "no child workers in this serial oracle"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", args.kind, report.get("warm_peak_bytes"), report.get("rss_delta_bytes"), flush=True)


if __name__ == "__main__":
    main()
