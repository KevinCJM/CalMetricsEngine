"""Complete-request matmul comparison using the shared matrix harness.

Pass --old-install and --new-install, or set CME_MATMUL_OLD_INSTALL and
CME_MATMUL_NEW_INSTALL. The old wheel's canonical output is frozen before either
timing or memory children start. Native outputs must match its bits; NJIT uses
NumPy matmul with the declared tolerance. Sampled high-accuracy products are
independent of either GEMM implementation; this is not a solver residual. No production files or shared harness are modified.
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
ROOT = Path(os.environ.get("CME_MATMUL_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / "tools"))
import check_large_matrix_performance as bench
import numpy as np

BASELINE_BUILD = "4e12b6bb2276be316adc975be2e417d092b961b15d0d83514fed7030cde86d0a"
REFERENCE_ENV = "CME_MATMUL_BENCH_REFERENCE_DIR"
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
    def case(m, k, n, layout, entry="ordinary"):
        return dict(name="matmul", n=m, k=k, d=n, layout=layout, entry=entry,
                    rtol=1e-11, atol=1e-11)
    layouts = ("cc", "cf", "fc", "ff")
    result = [case(n, n, n, layout) for n in (32, 128, 256, 512, 1024)
              for layout in layouts]
    result += [case(n, n, n, layout) for n in (127, 129, 257)
               for layout in ("cc", "cf")]
    rectangles = ((32768, 64, 32), (64, 32768, 64), (64, 64, 32768), (4097, 64, 17))
    result += [case(m, k, n, layout) for m, k, n in rectangles for layout in layouts]
    result += [case(n, n, n, layout) for n in (128, 512)
               for layout in ("sliced", "negative", "padded")]
    result += [case(512, 512, 512, layout) for layout in
               ("padded_f_b", "negative_a_f_b", "sliced_a_f_b", "zero_a_f_b")]
    result += [case(m, k, n, layout) for m, k, n in ((1, 8, 1), (3, 9, 3), (4, 8, 4))
               for layout in ("cc", "cf")]
    result += [case(m, k, n, layout, entry)
               for m, k, n in ((32, 32, 32), (512, 512, 512),
                               (1024, 1024, 1024), (32768, 64, 32))
               for layout in layouts for entry in ("direct", "prepared")]
    return result


def case_id(case):
    return f"matmul_{case['n']}x{case['k']}x{case['d']}_{case['layout']}_{case['entry']}"


def reference_key(case):
    return f"matmul_{case['n']}x{case['k']}x{case['d']}_{case['layout']}"


def inputs(case):
    rng = np.random.default_rng(20260930)
    m, k, n = case["n"], case["k"], case["d"]
    data = dict(a=rng.normal(size=(m, k)), b=rng.normal(size=(k, n)))
    layout = case["layout"]
    for index, (name, array) in enumerate(data.items()):
        if layout in ("cc", "cf", "fc", "ff"):
            array = np.array(array, order=layout[index].upper(), copy=True)
        elif layout.endswith("_f_b"):
            if index == 1:
                if layout == "padded_f_b":
                    storage = np.zeros((array.shape[0] + 3, array.shape[1]), order="F")
                    storage[:array.shape[0]] = array
                    array = storage[:array.shape[0]]
                else:
                    array = np.asfortranarray(array)
            elif layout == "negative_a_f_b":
                array = array[:, ::-1]
            elif layout == "sliced_a_f_b":
                storage = np.zeros((array.shape[0], 2 * array.shape[1]))
                storage[:, ::2] = array
                array = storage[:, ::2]
            elif layout == "zero_a_f_b":
                array = np.broadcast_to(array[:1], array.shape)
        elif layout == "sliced":
            storage = np.zeros(tuple(2 * size for size in array.shape), dtype=np.float64)
            storage[::2, ::2] = array
            array = storage[::2, ::2]
        elif layout == "negative":
            storage = array[::-1, ::-1].copy()
            array = storage[::-1, ::-1]
        elif layout == "padded":
            storage = np.zeros((array.shape[0], array.shape[1] + 3), dtype=np.float64)
            storage[:, :array.shape[1]] = array
            array = storage[:, :array.shape[1]]
        else:
            raise ValueError(f"unknown layout: {layout}")
        array.flags.writeable = False
        data[name] = array
    return data


def setup(case, backend, data):
    global _active_backend
    _active_backend = backend
    a, b = data["a"], data["b"]
    if backend == "njit":
        from numba import njit, typeof

        if "matmul" not in _refs:
            @njit(nogil=True)
            def matmul(a, b):
                return a @ b
            _refs["matmul"] = matmul
        reference = _refs["matmul"]
        signature = (typeof(a), typeof(b))
        if signature not in reference.signatures:
            # Compile actual readonly/layout signature; first numerical BLAS
            # call and its work buffers remain in the memory epoch.
            reference.compile(signature)
        return None, lambda: reference(a, b)

    from calmetrics_engine import AdaptiveScheduler, GraphCompiler, operators

    if case["entry"] == "direct":
        return None, lambda: operators.call("matmul", a, b)
    types = {
        "a": dict(kind="matrix", axes=["asset", "factor"], shape=["M", "K"]),
        "b": dict(kind="matrix", axes=["factor", "component"], shape=["K", "N"]),
    }
    graph = GraphCompiler(types).compile("matmul(a,b)", result_format="typed")
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
        # diagnostic mean is never used to warm the measured matmul path.
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
        raise AssertionError(f"old wheel is not the frozen 4e12 baseline: {identity}")
    return identity


def residuals(case, data, actual):
    """Independent sampled high-accuracy products, outside timers/epochs.

    Full output parity uses the frozen scalar-fold reference separately. fsum
    checks at most 25 deterministic cells without a second BLAS call or a large
    product tensor. This is sampled arithmetic evidence, not a solve residual.
    """
    import math

    a, b = data["a"], data["b"]
    actual = np.asarray(actual)
    if actual.shape != (case["n"], case["d"]) or actual.dtype != np.dtype("float64"):
        raise AssertionError("matmul changed output shape/dtype")
    if not actual.flags.c_contiguous:
        raise AssertionError("matmul output must be C-contiguous")
    if not np.isfinite(actual).all():
        raise AssertionError("finite benchmark input produced non-finite output")
    rows = sorted({0, case["n"] // 4, case["n"] // 2, 3 * case["n"] // 4, case["n"] - 1})
    cols = sorted({0, case["d"] // 4, case["d"] // 2, 3 * case["d"] // 4, case["d"] - 1})
    errors = []
    for row in rows:
        for col in cols:
            expected = math.fsum(float(a[row, k]) * float(b[k, col])
                                 for k in range(case["k"]))
            error = abs(float(actual[row, col]) - expected)
            limit = case["atol"] + case["rtol"] * abs(expected)
            if error > limit:
                raise AssertionError(f"sampled matmul arithmetic failed at {row, col}: {error}")
            errors.append(error)
    return dict(sampled_cells=len(errors), sampled_max_absolute_error=max(errors),
                sampled_reference="math.fsum of float64 rounded products; no BLAS",
                full_output_comparison="frozen old scalar reference, checked separately")


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
        # The plain direct API returns no audit. A second diagnostic matmul after
        # every timing round would add unmatched native warm/thermal work.
        # Actual library adoption is verified by the dedicated regression and
        # static binary evidence; do not invent counters or an ISA here.
        report = {
            "audit_collection": "not-collected",
            "reason": "plain direct result; no additional diagnostic matmul",
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
            raise AssertionError("canonical matmul mutated input")
        key = reference_key(case)
        path = destination / f"{key}.npz"
        np.savez(path, output_0=actual)
        record = dict(
            key=key, case=case, build=identity, input=before, audit=report, residuals=checks,
            output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            requested_simd="scalar",
            reference_role="frozen-old-canonical scalar k-left-fold; multiplication then addition",
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
            "memory": "inputs+imports+NJIT compile only precede epoch; graph/setup/first matmul included",
            "retention": "previous independent result alive until next returns",
            "canonical": "old-install frozen output; bitwise native comparison",
            "njit": "numpy matmul via Numba/SciPy BLAS; numerical tolerance",
            "direct_audit": "not-collected; no extra matmul outside timed/heap/RSS call",
            "scope": "single product/single interval/single matmul; no process/shared-memory speed claim",
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


bench.case_id = case_id
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
        ("--old-install", "CME_MATMUL_OLD_INSTALL"),
        ("--new-install", "CME_MATMUL_NEW_INSTALL"),
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
