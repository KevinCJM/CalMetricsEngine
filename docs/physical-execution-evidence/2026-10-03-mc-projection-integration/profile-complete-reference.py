"""Source-shaped complete fixed-tape MC profile, not native acceptance.

Controlled frozen NJIT oracle only. Ordinary engine execution never imports it.
Per-portfolio projection/NAV/fan calls match the readable source order. Data
preparation, compilation and input hashes are outside timing. Coverage, random
generation, factor transformation, limits and JSON packaging are excluded.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from numba import set_num_threads
import bench
from reference import frozen_one, FROZEN_SOURCE_SHA256

HERE = Path(__file__).resolve().parent
SOURCE = Path("/Users/chenjunming/Desktop/KevinGit/CppCalAST/FundInvestmentResearchPlatform/backend/scenario_stress/numba_kernels.py")


def freeze_reference():
    payload = SOURCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FROZEN_SOURCE_SHA256
    tree = ast.parse(payload)
    names = {"returns_to_nav_kernel", "quantile_sorted_kernel", "fan_statistics_kernel"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    for node in functions:
        node.decorator_list = [ast.parse("@njit(cache=True, nogil=True, fastmath=False)\ndef x(): pass").body[0].decorator_list[0]]
    text = "# Frozen mathematical bodies; test-only, lazy compilation.\nimport numpy as np\nfrom numba import njit\n\n" + "\n\n".join(ast.unparse(node) for node in functions) + "\n"
    destination = HERE / "complete_reference.py"
    if destination.exists():
        assert destination.read_text() == text
    else:
        destination.write_text(text)
    spec = importlib.util.spec_from_file_location("complete_reference", destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(text.encode()).hexdigest()


def run(data, module):
    available = np.ones(data["assets"], np.int64)
    output, phases = [], dict(projection_ms=0., nav_ms=0., fan_ms=0., samples_ms=0.)
    for combination in range(data["combinations"]):
        begin = time.perf_counter_ns()
        period, contribution, status, _, _, _ = frozen_one(data["x"], data["beta"], data["initial"], data["weights"][combination], available)
        phases["projection_ms"] += (time.perf_counter_ns() - begin) / 1e6
        assert status == 0
        begin = time.perf_counter_ns()
        nav, terminal, drawdowns, status, _, _, _ = module.returns_to_nav_kernel(period, 1.)
        phases["nav_ms"] += (time.perf_counter_ns() - begin) / 1e6
        assert status == 0
        begin = time.perf_counter_ns()
        fan, metrics = module.fan_statistics_kernel(nav, terminal, drawdowns, .1)
        phases["fan_ms"] += (time.perf_counter_ns() - begin) / 1e6
        begin = time.perf_counter_ns()
        # Numerical sample payload; production JSON formatting is excluded.
        samples = nav[:min(12, data["paths"])].copy()
        output.append((fan, metrics, samples, contribution))
        phases["samples_ms"] += (time.perf_counter_ns() - begin) / 1e6
    return output, phases


def fingerprints(data):
    return {name: hashlib.sha256(data[name].tobytes()).hexdigest() for name in ("x", "beta", "initial", "weights")}


def main():
    set_num_threads(1)
    module, reference_sha = freeze_reference()
    # Compile exact readonly tape/model/weights signatures before measurement;
    # NAV and fan receive newly allocated mutable C arrays on every call.
    tiny = bench.inputs(np, n=24, steps=3, factors=4, assets=32, combinations=1)
    run(tiny, module)
    reports = []
    for combinations in (1, 8):
        data = bench.inputs(np, n=2_000_000, steps=200, factors=4, assets=32, combinations=combinations)
        before = fingerprints(data)
        records = []
        previous = None
        for index in range(3):
            start = time.perf_counter_ns()
            result, phases = run(data, module)
            total = (time.perf_counter_ns() - start) / 1e6
            # Two compact independent results are retained during replacement.
            if previous is not None:
                for left, right in zip(previous, result):
                    for a, b in zip(left, right):
                        np.testing.assert_array_equal(a, b)
                        assert not np.shares_memory(a, b)
            previous = result
            records.append(dict(round=index, total_ms=total, **phases))
            print("ROUND", combinations, index, total, phases, flush=True)
        assert before == fingerprints(data)
        output_bytes = sum(array.nbytes for row in result for array in row)
        reports.append(dict(n=data["n"], steps=data["steps"], paths=data["paths"], factors=4, assets=32,
                            combinations=combinations, records=records,
                            medians_ms={key: statistics.median(row[key] for row in records) for key in records[0] if key != "round"},
                            compact_output_bytes=output_bytes, independent_two_result_ownership=True,
                            input_unchanged=True, input_sha256=before))
        (HERE / "complete-reference-profile.json").write_text(json.dumps(dict(
            scope="healthy_fixed_transformed_tape_projection_nav_fan_numeric_samples_contributions",
            source_sha256=FROZEN_SOURCE_SHA256, frozen_reference_sha256=reference_sha,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cpu_budget=1,
            excluded=["coverage", "old_random_stream", "covariance_root", "factor_transform", "limits", "json_packaging", "first_fault_cases"],
            timing_layer="per-portfolio Python numerical orchestration; compilation/preparation outside",
            cases=reports), indent=2) + "\n")
    print("PASS: fixed-tape complete-reference profile only", flush=True)


if __name__ == "__main__":
    main()
