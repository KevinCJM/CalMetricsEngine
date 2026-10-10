"""Full owned-result comparisons for linear selection packing and grouped Cholesky.

The frozen common harness supplies serial rotated timing, bounded retained
results, continuous setup-through-warm heap and separate fresh RSS children.
No baseline runs numerical work before the memory epoch. Scripts are evidence,
never a production calculation or runtime dependency.
"""
# ruff: noqa: E402
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE / "tools"))
import check_large_matrix_performance as bench
import numpy as np

# Frozen common benchmark source SHA-256: 9d5fe138e832d1fe30208e653de72e035c325c70ecdc9047662303d4719c3e37
BASELINE_BUILD = "2ebc5d22f827d240a99a0507b8153213b9fd9216290c22e0bec01f96285567a0"
NEW_BUILD = "850871ebcff4f6f2e089ae96029f736ccc0c94282cd06824193f839a410906b3"
OLD_INSTALL = Path("/private/tmp/calmetrics-order-vforce-fchol-next/install-final")
NEW_INSTALL = STAGE / "install"
if os.environ.get("CME_OPT_NEW_BUILD", NEW_BUILD) != NEW_BUILD:
    raise RuntimeError("CME_OPT_NEW_BUILD disagrees with this frozen candidate")
os.environ.setdefault("CME_OPT_NEW_BUILD", NEW_BUILD)
_backend = None
_references = {}
_base_audit = bench.audit
_base_heap, _base_rss = bench.measure_heap, bench.measure_rss


def cases():
    result = []
    for name in ("median", "quantile", "median_where", "quantile_where"):
        for n in (63, 1048576, 4194304, 16777216):
            case = dict(name=name, n=n, layout="c", entry="ordinary", pattern="random")
            if name.endswith("_where"):
                case["density"] = .55
            result.append(case)
    for pattern in ("sorted", "reverse", "equal"):
        result.append(dict(name="quantile", n=1048576, layout="c", entry="ordinary", pattern=pattern))
    result.append(dict(name="quantile", n=4194304, layout="c", entry="prepared", pattern="random"))
    for name in ("median_where", "quantile_where"):
        for density in (.01, 1.):
            result.append(dict(name=name, n=4194304, layout="c", entry="ordinary",
                               pattern="random", density=density))
    for n in (63, 512, 1024, 2048, 4096):
        for layout in ("c", "f"):
            result.append(dict(name="cholesky", n=n, layout=layout, order="F", entry="ordinary"))
    result.append(dict(name="cholesky", n=4096, layout="c", order="F", entry="prepared"))
    return result


def case_id(case):
    density = format(case.get("density", .55), ".6g") if case["name"].endswith("_where") else "na"
    return (f"{case['name']}_{case['n']}_{case['layout']}_{case.get('order','auto')}_"
            f"{case['entry']}_{case.get('pattern','spd')}_default_density{density}")


def inputs(case):
    rng = np.random.default_rng(20261001)
    n = case["n"]
    if case["name"] == "cholesky":
        # One matrix and only O(N) row temporaries: no released N-squared
        # generator allocation whose resident pages could hide request RSS.
        a = np.empty((n, n), order=case["layout"].upper())
        rng.standard_normal(out=a)
        for row in range(n):
            a[row, :row] = a[:row, row]
        for row in range(n):
            a[row, row] += np.abs(a[row]).sum() + 1.
        data = {"a": a}
    else:
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
    global _backend
    _backend = backend
    name = case["name"]
    if backend == "njit":
        from numba import njit, typeof
        if not _references:
            @njit(nogil=True)
            def order(x, mask, masked, quantile):
                selected = x[mask] if masked else x
                return np.quantile(selected, .95) if quantile else np.median(selected)

            @njit(nogil=True)
            def cholesky(a):
                # Compare the requested owned F result for both C/F inputs.
                return np.asfortranarray(np.linalg.cholesky(a))
            _references.update(order=order, cholesky=cholesky)
        if name == "cholesky":
            reference, args = _references["cholesky"], (data["a"],)
        else:
            reference = _references["order"]
            args = (data["x"], data.get("m", np.empty(0, bool)),
                    name.endswith("_where"), name.startswith("quantile"))
        signature = tuple(typeof(value) for value in args)
        if signature not in reference.signatures:
            # Compile actual dtype/layout without numerical warmup before a
            # fresh memory epoch; timing worker warms it outside the timer.
            reference.compile(signature)
        return None, lambda: reference(*args)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    if name == "cholesky":
        declarations = {"a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"])}
        expression, length = "cholesky(a)", 0
    else:
        declarations = {"x": "series"}
        expression, length = f"{name}(x)", len(data["x"])
        if name.endswith("_where"):
            declarations["m"] = dict(kind="series", dtype="bool", axes=["time"], shape=["T"], semantic_dimension="mask")
            expression = f"{name}(x,m{',.95' if name.startswith('quantile') else ''})"
        elif name == "quantile":
            expression = "quantile(x,.95)"
    graph = GraphCompiler(declarations).compile(expression, result_format="typed", output_order=case.get("order", "auto"))
    scheduler = AdaptiveScheduler(cpu_budget=1)
    bounds = np.array([0], np.int64), np.array([length], np.int64)
    try:
        if case["entry"] == "prepared":
            prepared = scheduler.prepare_execution(graph, data, *bounds)
            return scheduler, lambda: bench.MaterializedResult(prepared.run_snapshot())
        return scheduler, lambda: bench.MaterializedResult(scheduler.execute(graph, data, *bounds))
    except BaseException:
        scheduler.close()
        raise


def invariants(case, data, actual):
    actual = np.asarray(actual)
    report = dict(output_shape=list(actual.shape),
                  output_order="F" if actual.ndim > 1 and actual.flags.f_contiguous else "C",
                  output_sha256=hashlib.sha256(memoryview(actual.ravel(order="K"))).hexdigest())
    if _backend != "old":
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        expected = bench.load_outputs(output / "old.npz")[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14, equal_nan=True)
        for predicate in (np.isfinite, np.isnan, np.signbit, lambda v: v == 0):
            np.testing.assert_array_equal(predicate(actual), predicate(expected))
        different = int(np.count_nonzero(actual.view(np.uint64) != expected.view(np.uint64)))
        if _backend == "new":
            assert different == 0, (case, different)
        report["canonical_changed_bits"] = different
    if case["name"] == "cholesky":
        assert actual.flags.f_contiguous
        assert np.all(np.diag(actual) > 0)
        for row in range(len(actual)):
            assert not np.any(actual[row, row + 1:])
        maximum = 0.
        for i in np.linspace(0, len(actual)-1, 9, dtype=int):
            for j in np.linspace(0, i, 9, dtype=int):
                reconstructed = np.dot(actual[i, :j+1], actual[j, :j+1])
                expected = data["a"][i, j]
                np.testing.assert_allclose(reconstructed, expected, rtol=1e-11, atol=1e-11)
                maximum = max(maximum, abs(reconstructed - expected))
        report["max_sampled_reconstruction_residual"] = maximum
        report["reconstruction_scope"] = "81 deterministic coefficients, full triangular/diagonal and full old-new bits"
    return report


def audit(case, backend, data, result):
    report = _base_audit(case, backend, data, result)
    if backend != "njit":
        expected = BASELINE_BUILD if backend == "old" else NEW_BUILD
        assert expected and report["engine_build_id"] == expected, report.get("engine_build_id")
    return report


bench.__file__ = str(Path(__file__).resolve())
bench.cases, bench.case_id, bench.inputs, bench.setup = cases, case_id, inputs, setup
bench.invariants, bench.audit, bench.input_identity = invariants, audit, input_identity
bench.ROUNDS = 9
# Preserve bounded complete epochs while limiting repeated multi-second factors.
bench.measure_heap = lambda setup, lib, repeats=11: _base_heap(setup, lib, repeats=3)
bench.measure_rss = lambda function, repeats=11: _base_rss(function, repeats=3)

if __name__ == "__main__":
    # The common tool stays byte-identical; only this stage supplies its installs.
    for option, install in (("--old-install", OLD_INSTALL), ("--new-install", NEW_INSTALL)):
        if not any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:]):
            sys.argv.extend((option, str(install)))
    bench.main()
