"""Full ordinary-Scheduler Highway selection comparisons, prepared not executed.

Reuses the byte-identical short-tail common harness: nine serial balanced
rounds, independently owned/materialized outputs, and fresh heap/RSS children.
All processes start with thread environment 1. Only NJIT receives threadpoolctl
and Accelerate TLS controls; native honors AdaptiveScheduler(cpu_budget=1).
Set CME_OPT_NEW_BUILD to the verified candidate's full build id before running.
"""
# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
FROZEN_TOOLS = {
    "check_large_matrix_performance.py": "2f99bf41adac0926e30aa2045fa0f6bb67ea938826bf6a4c69ed6cf278c10518",
    "benchmark_memory.py": "b3862bb22e58f01f874bad0fd8466a3eb63ac4f3803186846184aafb47f4d7f8",
}
for filename, expected_sha in FROZEN_TOOLS.items():
    assert hashlib.sha256((STAGE / "tools" / filename).read_bytes()).hexdigest() == expected_sha
sys.path.insert(0, str(STAGE / "tools"))
import check_large_matrix_performance as bench
import numpy as np

BASELINE_BUILD = "5b31418d52e9804122fe901bb645f5291bf181a1a2fba17e77cb0fee0489a1c9"
NEW_BUILD = os.environ.get("CME_OPT_NEW_BUILD", "")
if len(NEW_BUILD) != 64 or any(char not in "0123456789abcdef" for char in NEW_BUILD):
    raise RuntimeError("Set CME_OPT_NEW_BUILD to the verified candidate's full build id")
OLD_INSTALL = Path("/private/tmp/calmetrics-cov-centered-integration-next/install")
NEW_INSTALL = STAGE / "install"
_input_snapshot = None
_expected_output_bits = None
_backend = None
_references = {}
_base_audit, _base_memory_all = bench.audit, bench.memory_all
_base_heap, _base_rss = bench.measure_heap, bench.measure_rss


def cases():
    def item(name, n, **kwargs):
        return dict(name=name, n=n, layout="c", entry="ordinary", pattern="random",
                    rtol=1e-12, atol=1e-12, **kwargs)

    result = [item("quantile", 16777216), item("median", 16777216),
              item("quantile", 16777216, probability=.05),
              item("quantile", 4194304),
              item("quantile_where", 16777216, density=.5)]
    for n in (63, 1023, 1024, 1025):
        for name in ("quantile", "median"):
            result.append(item(name, n))
    for pattern in ("sorted", "reverse", "equal"):
        case = item("quantile", 4194304)
        case["pattern"] = pattern
        result.append(case)
    if os.environ.get("CME_ORDER_MEMORY_CASES") == "1":
        return result[:2]
    return result


def case_id(case):
    density = format(case.get("density", .55), ".6g") if case["name"].endswith("_where") else "na"
    return (f"{case['name']}_{case['n']}_{case['layout']}_{case.get('order','auto')}_"
            f"{case['entry']}_{case.get('pattern','spd')}_default_density{density}_q{case.get('probability', .95):g}")


def inputs(case):
    global _input_snapshot
    rng = np.random.default_rng(20261001)
    n = case["n"]
    x = rng.normal(size=n)
    pattern = case.get("pattern")
    if pattern == "sorted":
        x.sort()
    elif pattern == "reverse":
        x.sort()
        x = x[::-1].copy()
    elif pattern == "equal":
        x.fill(3.25)
    data = {"x": x}
    if case["name"].endswith("_where"):
        density = case.get("density", .55)
        if not 0. <= density <= 1.:
            raise ValueError(f"invalid mask density: {density}")
        mask = np.empty(n, bool)
        # Choose density at generation; never repair/change a mask after
        # inserting excluded NaNs. All temporary mask/RNG arrays <=8192.
        for first in range(0, n, 8192):
            last = min(n, first + 8192)
            if density == 1.:
                mask[first:last] = True
            elif density == 0.:
                mask[first:last] = False
            else:
                threshold = .45 if density == .55 else 1. - density
                mask[first:last] = rng.random(last - first) > threshold
            np.copyto(x[first:last], np.nan, where=~mask[first:last])
        data["m"] = mask
    for value in data.values():
        value.flags.writeable = False
    _input_snapshot = input_identity(data)
    return data


def input_identity(data):
    result = {}
    for name, array in data.items():
        digest = hashlib.sha256()
        if array.flags.c_contiguous:
            digest.update(memoryview(array))
        else:
            # The benchmark only has contiguous C/F matrices. Preserve the
            # original logical-C digest with at most one bounded row copy.
            assert array.ndim == 2 and array.flags.f_contiguous
            assert array.shape[1] * array.itemsize <= 65536
            for row in array:
                digest.update(memoryview(np.ascontiguousarray(row)))
        result[name] = dict(shape=list(array.shape), strides=list(array.strides),
                            dtype=str(array.dtype), sha256=digest.hexdigest(),
                            writeable=bool(array.flags.writeable))
    return result


def setup(case, backend, data):
    global _backend, _expected_output_bits
    _backend = backend
    _expected_output_bits = None
    name = case["name"]
    if backend == "njit":
        from numba import njit, typeof
        if not _references:
            @njit(nogil=True)
            def order(x, mask, masked, quantile, probability):
                selected = x[mask] if masked else x
                return np.quantile(selected, probability) if quantile else np.median(selected)

            _references.update(order=order)
        reference = _references["order"]
        args = (data["x"], data.get("m", np.empty(0, bool)),
                name.endswith("_where"), name.startswith("quantile"),
                case.get("probability", .95))
        signature = tuple(typeof(value) for value in args)
        if signature not in reference.signatures:
            # Compile actual dtype/layout without numerical warmup before a
            # fresh memory epoch; timing worker warms it outside the timer.
            reference.compile(signature)
        return None, lambda: reference(*args)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    declarations = {"x": "series"}
    expression, length = f"{name}(x)", len(data["x"])
    if name.endswith("_where"):
        declarations["m"] = dict(kind="series", dtype="bool", axes=["time"],
                                  shape=["T"], semantic_dimension="mask")
        probability = f",{case.get('probability', .95)!r}" if name.startswith("quantile") else ""
        expression = f"{name}(x,m{probability})"
    elif name == "quantile":
        expression = f"quantile(x,{case.get('probability', .95)!r})"
    graph = GraphCompiler(declarations).compile(expression, result_format="typed", output_order=case.get("order", "auto"))
    scheduler = AdaptiveScheduler(cpu_budget=1)
    bounds = np.array([0], np.int64), np.array([length], np.int64)
    try:
        return scheduler, lambda: bench.MaterializedResult(scheduler.execute(graph, data, *bounds))
    except BaseException:
        scheduler.close()
        raise


def invariants(case, data, actual):
    global _expected_output_bits
    actual = np.asarray(actual)
    assert actual.shape == () and actual.dtype == np.float64
    _expected_output_bits = int(actual.view(np.uint64))
    report = dict(output_shape=list(actual.shape), output_order="C",
                  output_uint64=_expected_output_bits,
                  output_sha256=hashlib.sha256(memoryview(actual.ravel(order="K"))).hexdigest())
    if _backend != "old":
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        expected = bench.load_outputs(output / "old.npz")[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        for predicate in (np.isfinite, np.isnan, np.signbit, lambda v: v == 0):
            np.testing.assert_array_equal(predicate(actual), predicate(expected))
        different = int(np.count_nonzero(actual.view(np.uint64) != expected.view(np.uint64)))
        if _backend == "new":
            assert different == 0, (case, different)
        report["canonical_changed_bits"] = different
    return report


def audit(case, backend, data, result):
    # Runs outside the timer/heap epoch. Recheck readonly inputs and the result
    # after each measured round rather than trusting the initialization alone.
    assert input_identity(data) == _input_snapshot
    report = _base_audit(case, backend, data, result)
    if backend != "njit":
        expected = BASELINE_BUILD if backend == "old" else NEW_BUILD
        assert report["engine_build_id"] == expected, report.get("engine_build_id")
    actual = np.asarray(bench.value(result))
    assert actual.shape == () and actual.dtype == np.float64
    output_bits = int(actual.view(np.uint64))
    if _expected_output_bits is not None:
        assert output_bits == _expected_output_bits, (case, backend, output_bits)
    report.update(output_uint64=output_bits, input_unchanged=True)
    return report


def memory_all(args, selected):
    # Every backend/kind still runs in its own fresh child via the frozen tool.
    # Compare the recorded post-epoch scalar; never perform an oracle warmup.
    _base_memory_all(args, selected)
    reports = json.loads((args.output / "memory.json").read_text())
    checks = []
    for case in selected:
        for kind in ("heap", "rss"):
            records = {r["backend"]: r for r in reports if r["case"] == case and r["kind"] == kind}
            assert set(records) == set(bench.BACKENDS)
            old = records["old"]
            expected_bits = old["audit"]["output_uint64"]
            expected = np.array(expected_bits, dtype=np.uint64).view(np.float64)
            for backend in ("new", "njit"):
                actual = records[backend]
                assert actual["input"] == old["input"]
                bits = actual["audit"]["output_uint64"]
                if backend == "new":
                    assert bits == expected_bits, (case, kind, backend)
                else:
                    scalar = np.array(bits, dtype=np.uint64).view(np.float64)
                    np.testing.assert_allclose(scalar, expected, rtol=1e-12, atol=1e-12)
            checks.append(dict(case=case, kind=kind, old_new_exact=True,
                               njit_tolerance=dict(rtol=1e-12, atol=1e-12), input_match=True))
    (args.output / "memory-validation.json").write_text(json.dumps(checks, indent=2))


bench.__file__ = str(Path(__file__).resolve())
bench.cases, bench.case_id, bench.inputs, bench.setup = cases, case_id, inputs, setup
bench.invariants, bench.audit, bench.input_identity = invariants, audit, input_identity
bench.memory_all = memory_all
bench.ROUNDS = 9
# Keep the preceding full-request stage's complete epoch: one cold call,
# five warmups, three warm measurements; RSS has three calls including setup.
bench.measure_heap = lambda setup, lib, repeats=11: _base_heap(setup, lib, repeats=3)
bench.measure_rss = lambda function, repeats=11: _base_rss(function, repeats=3)

if __name__ == "__main__":
    # The common tool stays byte-identical; only this stage supplies its installs.
    for option, install in (("--old-install", OLD_INSTALL), ("--new-install", NEW_INSTALL)):
        if not any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:]):
            sys.argv.extend((option, str(install)))
    bench.main()
