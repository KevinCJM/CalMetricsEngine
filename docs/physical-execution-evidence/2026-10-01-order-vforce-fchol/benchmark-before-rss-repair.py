"""Actual owned-result wheel/NJIT comparisons for three physical optimizations.

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

BASELINE_BUILD = "e619f8cd39317fc9d36356050393b715e0932874593f689b89dbc1eabff65cdd"
_backend = None
_references = {}
_base_audit = bench.audit
_base_heap, _base_rss = bench.measure_heap, bench.measure_rss


def cases():
    result = []
    for name in ("median", "quantile", "median_where", "quantile_where"):
        for n in (63, 1048576, 4194304):
            result.append(dict(name=name, n=n, layout="c", entry="ordinary", pattern="random"))
    for pattern in ("sorted", "reverse", "equal"):
        result.append(dict(name="quantile", n=1048576, layout="c", entry="ordinary", pattern=pattern))
    result.append(dict(name="quantile", n=4194304, layout="c", entry="prepared", pattern="random"))
    for name in ("exp", "log"):
        result.append(dict(name=name, n=63, layout="c", entry="ordinary", pattern="normal"))
        for pattern in ("normal", "late_outlier", "periodic_outlier", "all_outlier"):
            result.append(dict(name=name, n=16777216, layout="c", entry="ordinary", pattern=pattern))
        for layout, order in (("f", "F"), ("f", "C"), ("c", "F")):
            for baseline in ("vector", "tiled"):
                result.append(dict(name=name, n=4096, d=4096, layout=layout, order=order,
                                   entry="ordinary", pattern="late_outlier", baseline=baseline))
        result.append(dict(name=name, n=16777216, layout="c", entry="prepared", pattern="late_outlier"))
    for n in (63, 512, 2048, 4096):
        for layout in ("c", "f"):
            result.append(dict(name="cholesky", n=n, layout=layout, order="F", entry="ordinary"))
    result.append(dict(name="cholesky", n=4096, layout="c", order="F", entry="prepared"))
    return result


def case_id(case):
    dims = f"{case['n']}x{case['d']}" if "d" in case else str(case["n"])
    return f"{case['name']}_{dims}_{case['layout']}_{case.get('order','auto')}_{case['entry']}_{case.get('pattern','spd')}_{case.get('baseline','default')}"


def inputs(case):
    rng = np.random.default_rng(20261001)
    n = case["n"]
    if case["name"] == "cholesky":
        # One matrix, one O(N) row temporary, no released N squared generator
        # temporary that could mask request RSS by recycling resident pages.
        a = np.empty((n, n), order=case["layout"].upper())
        rng.standard_normal(out=a)
        for row in range(n):
            a[row, :row] = a[:row, row]
        for row in range(n):
            a[row, row] += np.abs(a[row]).sum() + 1.
        data = {"a": a}
    else:
        shape = (n, case["d"]) if "d" in case else (n,)
        x = rng.normal(size=shape) if case["name"].startswith(("median", "quantile")) else rng.uniform(.1, 10., shape)
        pattern = case.get("pattern")
        if case["name"] == "exp":
            x -= 5.
        if pattern == "sorted":
            x.sort()
        elif pattern == "reverse":
            x.sort()
            x = x[::-1].copy()
        elif pattern == "equal":
            x.fill(3.25)
        elif pattern == "nan":
            x[-1] = np.nan
        elif pattern == "mixed_zero":
            x[::2], x[1::2] = -0., 0.
        elif pattern in ("late_outlier", "periodic_outlier", "all_outlier"):
            special = 701. if case["name"] == "exp" else np.nextafter(0., 1.)
            if pattern == "late_outlier":
                x.flat[-1] = special
            elif pattern == "periodic_outlier":
                x.flat[::1048576] = special
            else:
                x.fill(special)
        x = np.asarray(x, order=case["layout"].upper())
        data = {"x": x}
        if case["name"].endswith("_where"):
            mask = rng.random(n) > .45
            x[~mask] = np.nan
            data["m"] = mask
    for value in data.values():
        value.flags.writeable = False
    return data


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
            def unary(x, logarithm, f_output):
                result = np.log(x) if logarithm else np.exp(x)
                return np.asfortranarray(result) if f_output else np.ascontiguousarray(result)

            @njit(nogil=True)
            def unary_tiled(x, logarithm, f_output):
                rows, columns = x.shape
                y = np.empty((columns, rows)).T if f_output else np.empty(x.shape)
                for first in range(0, rows, 128):
                    for col in range(0, columns, 32):
                        for i in range(first, min(rows, first + 128)):
                            for j in range(col, min(columns, col + 32)):
                                y[i, j] = np.log(x[i, j]) if logarithm else np.exp(x[i, j])
                return y

            @njit(nogil=True)
            def cholesky(a):
                # Required F output, not a C-result competitor to an F request.
                return np.asfortranarray(np.linalg.cholesky(a))
            _references.update(order=order, unary=unary, unary_tiled=unary_tiled, cholesky=cholesky)
        if name == "cholesky":
            reference, args = _references["cholesky"], (data["a"],)
        elif name.startswith(("median", "quantile")):
            reference = _references["order"]
            args = (data["x"], data.get("m", np.empty(0, bool)), name.endswith("_where"), name.startswith("quantile"))
        else:
            x = data["x"]
            reference = _references["unary_tiled" if case.get("baseline") == "tiled" and x.ndim == 2 else "unary"]
            args = (x, name == "log", case.get("order") == "F")
        signature = tuple(typeof(value) for value in args)
        if signature not in reference.signatures:
            reference.compile(signature)
        return None, lambda: reference(*args)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    if name == "cholesky":
        declarations = {"a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"])}
        expression, length = "cholesky(a)", 0
    else:
        x = data["x"]
        declarations = {"x": "series" if x.ndim == 1 else dict(kind="matrix", axes=["time", "asset"], shape=["T", "D"])}
        expression, length = f"{name}(x)", len(x)
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
        if _backend == "new" and case["name"] not in ("exp", "log"):
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
        expected = BASELINE_BUILD if backend == "old" else os.environ.get("CME_OPT_NEW_BUILD")
        assert expected and report["engine_build_id"] == expected, report.get("engine_build_id")
    return report


bench.__file__ = str(Path(__file__).resolve())
bench.cases, bench.case_id, bench.inputs, bench.setup = cases, case_id, inputs, setup
bench.invariants, bench.audit = invariants, audit
bench.ROUNDS = 9
# Preserve bounded complete epochs while limiting repeated multi-second factors.
bench.measure_heap = lambda setup, lib, repeats=11: _base_heap(setup, lib, repeats=3)
bench.measure_rss = lambda function, repeats=11: _base_rss(function, repeats=3)

if __name__ == "__main__":
    bench.main()
