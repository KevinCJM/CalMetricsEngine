"""Complete scalar quadratic_form old/new/NJIT comparison; same one-CPU budget.
Both native layouts preserve A.T @ x then dot(projected,x). NJIT calls NumPy
native BLAS after compile-only preparation. No symmetry assumptions. Readonly
inputs, full public result materialization and request heap epoch via shared harness.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_large_matrix_performance as bench  # noqa: E402
import numpy as np  # noqa: E402

_refs = {}


def cases():
    return (
        [
            dict(name="quadratic_form", n=n, layout=layout, entry="ordinary")
            for n in (16, 127, 128, 129, 256, 512, 1024, 2048)
            for layout in ("c", "f")
        ]
        + [
            dict(name="quadratic_form", n=512, layout=layout, entry="ordinary")
            for layout in ("reverse", "sliced")
        ]
        + [
            dict(name="quadratic_form", n=2048, layout=layout, entry=entry)
            for layout in ("c", "f")
            for entry in ("direct", "prepared")
        ]
    )


def inputs(case):
    rng = np.random.default_rng(20260929)
    n = case["n"]
    a = rng.normal(size=(n, n))
    x = rng.normal(size=n)
    if case["layout"] == "f":
        a = np.asfortranarray(a)
    elif case["layout"] == "reverse":
        a = a[::-1, ::-1]
        x = x[::-1]
    elif case["layout"] == "sliced":
        base = np.zeros((n * 2, n * 2))
        base[::2, ::2] = a
        a = base[::2, ::2]
    a.flags.writeable = False
    x.flags.writeable = False
    return dict(x=x, a=a)


def setup(case, backend, data):
    x, a = data["x"], data["a"]
    if backend == "njit":
        from numba import njit, typeof

        if "q" not in _refs:

            @njit(nogil=True)
            def q(x, a):
                return (a.T @ x) @ x

            _refs["q"] = q
        q = _refs["q"]
        signature = (typeof(x), typeof(a))
        if signature not in q.signatures:
            q.compile(signature)
        return None, lambda: q(x, a)
    from calmetrics_engine import AdaptiveScheduler, GraphCompiler, operators

    if case["entry"] == "direct":
        return None, lambda: operators.call("quadratic_form", x, a)
    graph = GraphCompiler(
        {
            "x": dict(kind="vector", axes=["asset"], shape=["N"]),
            "a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]),
        }
    ).compile("quadratic_form(x,a)", result_format="typed")
    bounds = np.zeros(1, np.int64)
    owner = AdaptiveScheduler(cpu_budget=1)
    try:
        if case["entry"] == "prepared":
            prepared = owner.prepare_execution(graph, data, bounds, bounds)
            return owner, lambda: bench.MaterializedResult(prepared.run_snapshot())
        return owner, lambda: bench.MaterializedResult(owner.execute(graph, data, bounds, bounds))
    except BaseException:
        owner.close()
        raise


def invariants(case, data, value):
    expected = (data["a"].T @ data["x"]) @ data["x"]
    np.testing.assert_allclose(value, expected, rtol=1e-11, atol=1e-11)
    return dict(max_absolute_residual=float(abs(value - expected)))


bench.cases = cases
bench.inputs = inputs
bench.setup = setup
bench.invariants = invariants
bench.__file__ = __file__
if __name__ == "__main__":
    bench.main()
