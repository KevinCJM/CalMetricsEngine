"""Complete-request solve/solve_multi comparison using the shared matrix harness.

Pass --old-install and --new-install, or set CME_SOLVE_OLD_INSTALL and
CME_SOLVE_NEW_INSTALL. The old wheel's canonical output is frozen before either
timing or memory children start. Native outputs must match its bits; NJIT uses
NumPy solve with the declared tolerance. Residual checks are independent of
either solve algorithm. No production files or shared harness are modified.
"""

# Import the harness before NumPy so its one-CPU environment is set first.
# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Repository-relative after installation in tools/. The explicit override only
# supports reviewing/running this temporary prototype before it is copied there.
ROOT = Path(os.environ.get("CME_SOLVE_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / "tools"))
import check_large_matrix_performance as bench
import numpy as np

BASELINE_BUILD = "75229d1773803cf459a8dd7a63931223ad182ed7344e43fbd3548bd5d9b85c9a"
REFERENCE_ENV = "CME_SOLVE_BENCH_REFERENCE_DIR"
_refs = {}
_active_backend = None
_memory_verification = False
_prepared_reference_root = None
_original_timing = bench.timing
_original_memory_all = bench.memory_all
_original_memory_child = bench.memory_child
_original_audit = bench.audit
_original_build_identity = bench.build_identity


def cases():
    kernels = [("solve", 1), ("solve_multi", 1), ("solve_multi", 8), ("solve_multi", 32)]
    # Every size and RHS count retains C/F/noncontiguous ordinary-request coverage.
    result = [
        dict(name=name, n=n, d=d, layout=layout, entry="ordinary", rtol=1e-11, atol=1e-11)
        for n in (8, 32, 128, 256, 512, 1024)
        for name, d in kernels
        for layout in ("c", "f", "sliced")
    ]
    result += [
        dict(name=name, n=n, d=d, layout=layout, entry=entry, rtol=1e-11, atol=1e-11)
        for n in (32, 512)
        for name, d in kernels
        for layout in ("c", "f")
        for entry in ("direct", "prepared")
    ]
    return result


def reference_key(case):
    return f"{case['name']}_{case['n']}x{case['d']}_{case['layout']}"


def inputs(case):
    rng = np.random.default_rng(20260929)
    n, d = case["n"], case["d"]
    a = rng.normal(size=(n, n))
    # Nonsymmetric and strictly row-diagonally dominant, without BLAS setup.
    # This benchmark deliberately does not exercise pathological pivots.
    a[np.diag_indices(n)] += np.abs(a).sum(axis=1) + 1.0
    b = rng.normal(size=n if case["name"] == "solve" else (n, d))
    if case["layout"] == "f":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif case["layout"] == "sliced":
        matrix_storage = np.zeros((2 * n, 2 * n), dtype=np.float64)
        matrix_storage[::2, ::2] = a
        a = matrix_storage[::2, ::2]
        rhs_storage = np.zeros(tuple(2 * size for size in b.shape), dtype=np.float64)
        selection = (slice(None, None, 2),) * b.ndim
        rhs_storage[selection] = b
        b = rhs_storage[selection]
    for array in (a, b):
        array.flags.writeable = False
    return dict(a=a, b=b)


def setup(case, backend, data):
    global _active_backend
    _active_backend = backend
    a, b = data["a"], data["b"]
    if backend == "njit":
        from numba import njit, typeof

        if "solve" not in _refs:

            @njit(nogil=True)
            def solve(a, b):
                return np.linalg.solve(a, b)

            _refs["solve"] = solve
        reference = _refs["solve"]
        signature = (typeof(a), typeof(b))
        if signature not in reference.signatures:
            # Actual readonly/rank/layout specialization, compile only. The
            # first numerical LAPACK call belongs to the memory epoch.
            reference.compile(signature)
        return None, lambda: reference(a, b)

    from calmetrics_engine import AdaptiveScheduler, GraphCompiler, operators

    if case["entry"] == "direct":
        return None, lambda: operators.call(case["name"], a, b)
    types = {
        "a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]),
        "b": (
            dict(kind="vector", axes=["asset"], shape=["N"])
            if case["name"] == "solve"
            else dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K"])
        ),
    }
    graph = GraphCompiler(types).compile(f"{case['name']}(a,b)", result_format="typed")
    bounds = np.zeros(1, dtype=np.int64)
    scheduler = AdaptiveScheduler(cpu_budget=1)
    try:
        if case["entry"] == "prepared":
            prepared = scheduler.prepare_execution(graph, data, bounds, bounds)
            return scheduler, lambda: bench.MaterializedResult(prepared.run_snapshot())
        return scheduler, lambda: bench.MaterializedResult(
            scheduler.execute(graph, data, bounds, bounds)
        )
    except BaseException:
        scheduler.close()
        raise


def build_identity(backend):
    identity = _original_build_identity(backend)
    if backend != "njit":
        # build_info deliberately has no source fingerprint. Obtain it from a
        # real result audit, after the timer/allocation epoch has ended. This
        # diagnostic mean is never used to warm the measured solve path.
        from calmetrics_engine import AdaptiveScheduler, GraphCompiler

        graph = GraphCompiler({"x": dict(kind="series", axes=["time"], shape=["T"])}).compile(
            "mean(x)", result_format="typed"
        )
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            result = scheduler.execute(
                graph, {"x": np.ones(1)}, np.zeros(1, np.int64), np.ones(1, np.int64)
            )
            identity["engine_build_id"] = result.audit["engine_build_id"]
    if backend == "old" and identity.get("engine_build_id") != BASELINE_BUILD:
        raise AssertionError(f"old wheel is not the frozen 75229 baseline: {identity}")
    return identity


def residuals(case, data, actual):
    a, b = data["a"], data["b"]
    actual = np.asarray(actual)
    if actual.shape != b.shape or actual.dtype != np.dtype("float64"):
        raise AssertionError("solve changed result shape/dtype")
    if not np.isfinite(actual).all():
        raise AssertionError("well-conditioned solve produced non-finite output")
    # optimize=False is a direct contraction, independent of the solve kernel
    # and without a BLAS solve/condition-estimation call. Never in a timer/epoch.
    reconstructed = np.einsum("ij,j...->i...", a, actual, optimize=False)
    delta = reconstructed - b
    np.testing.assert_allclose(reconstructed, b, rtol=case["rtol"], atol=case["atol"])
    norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
    if actual.ndim == 1:
        norm_x, norm_b = float(np.max(np.abs(actual))), float(np.max(np.abs(b)))
        norm_r = float(np.max(np.abs(delta)))
    else:
        norm_x = float(np.max(np.sum(np.abs(actual), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        norm_r = float(np.max(np.sum(np.abs(delta), axis=1)))
    backward_error = norm_r / max(norm_a * norm_x + norm_b, np.finfo(np.float64).tiny)
    if backward_error > 1e-12:
        raise AssertionError(f"solve backward error too large: {backward_error}")
    return dict(
        max_absolute_residual=float(np.max(np.abs(delta))),
        relative_backward_error_inf=backward_error,
        relative_backward_error_limit=1e-12,
        residual_method="direct-einsum-contraction-optimize-false",
    )


def invariants(case, data, actual):
    root = Path(os.environ[REFERENCE_ENV])
    key = reference_key(case)
    metadata = json.loads((root / f"{key}.json").read_text())
    if metadata["build"]["engine_build_id"] != BASELINE_BUILD:
        raise AssertionError("reference build identity changed")
    if metadata["input"] != bench.input_identity(data):
        raise AssertionError("reference input identity changed")
    path = root / f"{key}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["output_sha256"]:
        raise AssertionError("reference output artifact changed")
    expected = bench.load_outputs(path)[0]
    actual = np.asarray(actual)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("result differs from canonical shape/dtype")
    if _active_backend in ("old", "new"):
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        comparison = "bitwise"
    else:
        np.testing.assert_allclose(actual, expected, rtol=case["rtol"], atol=case["atol"])
        comparison = "rtol=atol=1e-11"
    return dict(
        **residuals(case, data, actual),
        canonical_build=BASELINE_BUILD,
        canonical_comparison=comparison,
        canonical_max_absolute_difference=float(np.max(np.abs(actual - expected))),
    )


def audit(case, backend, data, result):
    if backend != "njit" and case["entry"] == "direct":
        # The plain direct API returns no audit. A second diagnostic solve after
        # every timing round would add unmatched native warm/thermal work.
        # Actual library adoption is verified by the dedicated regression and
        # static binary evidence; do not invent counters or an ISA here.
        report = {
            "audit_collection": "not-collected",
            "reason": "plain direct result; no additional diagnostic solve",
        }
    else:
        report = dict(_original_audit(case, backend, data, result))
    if _memory_verification:
        # Shared memory_child invokes audit only after measure_heap/measure_rss
        # has ended; no reference read or residual pollutes allocation evidence.
        report["benchmark_post_epoch_verification"] = invariants(case, data, bench.value(result))
    return report


def scalar_reference_child():
    from calmetrics_engine import operators

    request = json.load(sys.stdin)
    destination = Path(request["destination"])
    identity = build_identity("old")
    records = []
    for case in request["cases"]:
        data = inputs(case)
        before = bench.input_identity(data)
        actual, report = operators.call(case["name"], *data.values(), simd="scalar", audit=True)
        checks = residuals(case, data, actual)
        if bench.input_identity(data) != before:
            raise AssertionError("canonical solve mutated input")
        key = reference_key(case)
        path = destination / f"{key}.npz"
        np.savez(path, output_0=actual)
        record = dict(
            key=key, case=case, build=identity, input=before, audit=report, residuals=checks,
            output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            requested_simd="scalar",
            reference_role="frozen-old-canonical; solve_multi had no separate Isa branch",
        )
        (destination / f"{key}.json").write_text(json.dumps(record, indent=2))
        records.append(record)
    print(json.dumps(dict(build=identity, cases=len(records), records=records)))


def prepare_references(args, selected):
    global _prepared_reference_root
    root = args.output.resolve() / "canonical-reference"
    if _prepared_reference_root == root:
        return
    root.mkdir(parents=True, exist_ok=True)
    os.environ[REFERENCE_ENV] = str(root)
    unique = {reference_key(case): case for case in selected}
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--scalar-reference-child"],
        input=json.dumps(dict(destination=str(root), cases=list(unique.values()))),
        env=bench.child_env(args, "old"), capture_output=True, text=True,
    )
    (args.output / "canonical-reference.log").write_text(child.stderr)
    child.check_returncode()
    reference_report = json.loads(child.stdout)
    manifest = dict(
        baseline_build=BASELINE_BUILD, reference_report=reference_report,
        cases=selected, cpu_budget=1, rounds=bench.ROUNDS,
        install_roots={"old": str(Path(args.old_install).resolve()),
                       "new": str(Path(args.new_install).resolve())},
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__), ROOT / "tools/check_large_matrix_performance.py",
                ROOT / "tools/benchmark_memory.py",
            )
        },
        contracts={
            "timing": "warm complete call; graph construction excluded; public outputs materialized",
            "memory": "inputs+imports+NJIT compile only precede epoch; graph/setup/first solve included",
            "retention": "previous independent result alive until next returns",
            "canonical": "old-install frozen output; bitwise native comparison",
            "njit": "numpy.linalg.solve via Numba/SciPy LAPACK; numerical tolerance",
            "direct_audit": "not-collected; no extra solve outside timed/heap/RSS call",
            "scope": "single product/single interval/single solve; no process/shared-memory speed claim",
        },
    )
    (args.output / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=2))
    _prepared_reference_root = root


def timing(args, selected):
    prepare_references(args, selected)
    return _original_timing(args, selected)


def memory_all(args, selected):
    prepare_references(args, selected)
    return _original_memory_all(args, selected)


def memory_child(args):
    global _memory_verification
    if REFERENCE_ENV not in os.environ:
        raise RuntimeError("invoke --mode memory/all so the canonical reference is prepared")
    _memory_verification = True
    try:
        return _original_memory_child(args)
    finally:
        _memory_verification = False


bench.cases = cases
bench.inputs = inputs
bench.setup = setup
bench.invariants = invariants
bench.audit = audit
bench.build_identity = build_identity
bench.timing = timing
bench.memory_all = memory_all
bench.memory_child = memory_child
bench.__file__ = __file__


def supply_install_arguments():
    """Require explicit installation roots only in the orchestrating parent."""
    arguments = sys.argv[1:]
    internal = any(arg in ("--worker", "--memory") or
                   arg.startswith(("--worker=", "--memory=")) for arg in arguments)
    help_only = "--help" in arguments or "-h" in arguments
    for option, environment in (
        ("--old-install", "CME_SOLVE_OLD_INSTALL"),
        ("--new-install", "CME_SOLVE_NEW_INSTALL"),
    ):
        if any(arg == option or arg.startswith(option + "=") for arg in arguments):
            continue
        if value := os.environ.get(environment):
            sys.argv.extend((option, value))
        elif not internal and not help_only:
            raise SystemExit(f"pass {option} or set {environment}; no implicit installation path")


if __name__ == "__main__":
    if sys.argv[1:] == ["--scalar-reference-child"]:
        scalar_reference_child()
    elif sys.argv[1:] == ["--list-cases"]:
        print(json.dumps([bench.case_id(case) for case in cases()], indent=2))
    else:
        supply_install_arguments()
        bench.main()
