"""Reuse the original paired harness; add guarded-path provenance and fixtures."""
# ruff: noqa: E402
from __future__ import annotations

import ctypes
import json
import sys
import time

import numpy as np
import screen

_check = screen.checks
_libraries = {}


def counters(library):
    if library not in _libraries:
        lib = ctypes.CDLL(library)
        function = lib.covariance_syrk_counters
        function.argtypes = (ctypes.POINTER(ctypes.c_uint64),)
        _libraries[library] = lib, function
    values = (ctypes.c_uint64 * 5)()
    _libraries[library][1](values)
    return dict(zip(("dsyrk_calls", "replayed_cells", "full_native_fallback",
                     "scratch_bytes", "centered_bytes"), values, strict=True))


def checks(args):
    _check(args)
    from calmetrics_engine import operators as op
    records = []
    for n, d in ((65, 8), (4101, 17), (8193, 65), (32769, 64), (1024, 128),
                 (513, 257), (1041, 512)):
        for pattern in ("normal", "constant", "sparse_orthogonal", "correlated", "scaled", "tiny"):
            source = np.random.default_rng(10019).normal(size=(n, d))
            if pattern == "constant":
                source[:, -1] = 7.0
            elif pattern == "sparse_orthogonal":
                source[:, 0] = np.tile([1., 1., -1., -1.], (n + 3) // 4)[:n]
                source[:, 1] = np.tile([1., -1., 1., -1.], (n + 3) // 4)[:n]
            elif pattern == "correlated":
                source = source * .1 + source[:, :1]
            elif pattern == "scaled":
                source *= 1e3
            elif pattern == "tiny":
                source *= 2. ** -450
            for layout in ("c", "f"):
                x = np.array(source, order=layout.upper())
                x.flags.writeable = False
                before = x.view(np.uint64).copy()
                run, _, _, _ = screen.candidate_setup(args.library, x)
                expected = op.call("covariance", x)
                value = run()
                stats = counters(args.library)
                np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(value > 0, expected > 0)
                np.testing.assert_array_equal(value == 0, expected == 0)
                np.testing.assert_array_equal(np.signbit(value), np.signbit(expected))
                np.testing.assert_array_equal(np.diag(value).view(np.uint64),
                                              np.diag(expected).view(np.uint64))
                np.testing.assert_array_equal(x.view(np.uint64), before)
                np.testing.assert_array_equal(value, value.T)
                if stats["full_native_fallback"]:
                    np.testing.assert_array_equal(value.view(np.uint64), expected.view(np.uint64))
                if pattern == "sparse_orthogonal" and n == 1024:
                    assert not stats["full_native_fallback"]
                    assert stats["replayed_cells"] > 0
                records.append(dict(n=n, d=d, pattern=pattern, layout=layout,
                                    counters=stats, max_abs=float(np.max(np.abs(value - expected)))))
    (args.output / "guarded-correctness.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(additional_cases=len(records), passed=True)), flush=True)


def worker(args):
    from calmetrics_engine import build_info
    owner = None
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command["op"] == "init":
                if owner:
                    owner.close()
                x = screen.inputs(command["case"])
                owner, function = screen.setup(args, args.backend, x)
                result = function()
                np.save(args.output / f"{args.backend}.npy", screen.array(result))
                response = dict(build=build_info() if args.backend == "native" else args.backend)
                if args.backend == "native":
                    response["engine_build_id"] = result.audit["engine_build_id"]
            else:
                wall, cpu = [], []
                for _ in range(command["repeat"]):
                    begin, cpu_begin = time.perf_counter_ns(), time.process_time_ns()
                    result = function()
                    cpu.append(time.process_time_ns() - cpu_begin)
                    wall.append(time.perf_counter_ns() - begin)
                response = dict(ns=wall, cpu_ns=cpu)
            if args.backend == "candidate":
                response["counters"] = counters(args.library)
                with (args.output / "candidate-counters.jsonl").open("a") as log:
                    log.write(json.dumps(response) + "\n")
            print(json.dumps(response), flush=True)
    finally:
        if owner:
            owner.close()


if __name__ == "__main__":
    screen.__file__ = __file__  # Controllers start this worker, not the raw worker.
    screen.worker = worker
    screen.checks = checks
    screen.main()
