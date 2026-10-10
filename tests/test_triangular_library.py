"""Independent ordered oracle, failure-prefix and layout/lifetime regressions."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op
from tests.test_solve_library import eigen_backend_available as eigen_backend_available
from tests.test_solve_library import same_ieee
from tests.test_tensor_protocol import configuration


def reference(a, b, lower, unit):
    y = np.array(b, order="C", copy=True)
    if not np.isfinite(y).all():
        return y, "NON_FINITE_RESULT"
    n = len(a)
    with np.errstate(all="ignore"):
        for i in range(n) if lower else range(n - 1, -1, -1):
            diagonal = 1.0 if unit else a[i, i]
            if not np.isfinite(diagonal):
                return y, "NON_FINITE_RESULT"
            if abs(diagonal) <= 1e-14:
                return y, "SINGULAR_MATRIX"
            for k in range(i) if lower else range(i + 1, n):
                coefficient = a[i, k]
                if not np.isfinite(coefficient):
                    return y, "NON_FINITE_RESULT"
                y[i] -= coefficient * y[k]
            for j in range(y.shape[1]):
                y[i, j] /= diagonal
                if not np.isfinite(y[i, j]):
                    return y, "NON_FINITE_RESULT"
    return y, None


def sample(n, d, layout="C"):
    rng = np.random.default_rng(925 + n + d)
    a = rng.normal(size=(n, n)) / max(n, 1)
    a[np.diag_indices(n)] += 2
    b = rng.normal(size=(n, d))
    if layout == "F":
        return np.asfortranarray(a), np.asfortranarray(b)
    if layout == "reverse":
        return a[::-1, ::-1], b[::-1, ::-1]
    if layout == "sliced":
        aa, bb = np.zeros((n * 2, n * 2)), np.zeros((n * 2, d * 2))
        aa[::2, ::2], bb[::2, ::2] = a, b
        return aa[::2, ::2], bb[::2, ::2]
    return a, b


def check(a, b, lower, unit, simd="auto"):
    expected, error = reference(a, b, lower, unit)
    old_a, old_b = a.copy(), b.copy()
    a.flags.writeable = b.flags.writeable = False
    out = np.full(b.shape, -913.25)
    if error:
        with pytest.raises(ValueError, match=error):
            op.triangular_solve(a, b, int(lower), int(unit), out=out, simd=simd)
        audit = None
    else:
        value, audit = op.triangular_solve(
            a, b, int(lower), int(unit), out=out, simd=simd, audit=True
        )
        assert out.size == 0 or np.shares_memory(out, value)
    same_ieee(out, expected)
    same_ieee(a, old_a)
    same_ieee(b, old_b)
    return out, audit, error


@pytest.mark.parametrize("lower,unit", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("layout", ["C", "F", "reverse", "sliced"])
@pytest.mark.parametrize(
    "n,d",
    [
        (0, 0),
        (3, 0),
        (16, 8),
        (31, 8),
        (32, 8),
        (65, 9),
        (65, 15),
        (65, 16),
        (65, 17),
        (65, 31),
        (65, 32),
        (65, 33),
        (65, 64),
    ],
)
def test_ordered_rhs_blocks_and_ignored_cells(n, d, layout, lower, unit):
    a, b = sample(n, d, layout)
    a[np.triu_indices(n, 1) if lower else np.tril_indices(n, -1)] = np.nan
    if unit:
        np.fill_diagonal(a, np.inf)
    check(a, b, lower, unit)


@pytest.mark.parametrize("lower", [0, 1])
@pytest.mark.parametrize("simd", ["auto", "scalar"])
@pytest.mark.parametrize(
    "fault",
    [
        "rhs",
        "diagonal",
        "first",
        "middle",
        "last",
        "singular",
        "threshold_above",
        "overflow",
        "division_overflow",
        "zero_inf",
    ],
)
def test_fault_order_and_exact_partial_writes(lower, simd, fault):
    n, d = 65, 17
    a, b = sample(n, d)
    i = 40 if lower else 20
    ks = list(range(i) if lower else range(i + 1, n))
    if fault == "rhs":
        b[-1, -1], a[0, 0] = np.nan, 0
    elif fault == "diagonal":
        a[i, i] = np.nan
    elif fault in ("first", "middle", "last"):
        k = {"first": ks[0], "middle": ks[len(ks) // 2], "last": ks[-1]}[fault]
        a[i, k] = np.inf
    elif fault in ("singular", "threshold_above"):
        a[i, i] = 1e-14 if fault == "singular" else np.nextafter(1e-14, np.inf)
    else:
        a = np.eye(n) * 2
        b = np.ones((n, d))
        if fault == "division_overflow":
            a[i, i], b[i, 1] = 1e-13, np.finfo(float).max
        else:
            a[i, ks[0]], b[ks[0], 1] = np.finfo(float).max, 8
            if fault == "zero_inf":
                # Overflow in the accumulator, then a zero multiplication:
                # finite checks still occur only after all row subtractions.
                a[i, ks[1]] = 0
    check(a, b, lower, False, simd)


@pytest.mark.parametrize("lower", [0, 1])
@pytest.mark.parametrize("d", [8, 17, 32])
def test_cancellation_does_not_become_dot_product_or_fma(lower, d):
    n = 65
    a, b = np.eye(n), np.zeros((n, d))
    i, ks = (n - 1, [0, 1, 2]) if lower else (0, [1, 2, 3])
    a[i, ks] = 1
    b[ks[0]], b[ks[1]], b[ks[2]], b[i] = 1e16, 1, -1e16, 1
    y, _, error = check(a, b, lower, False)
    assert error is None and np.all(y[i] == 0)


@pytest.mark.parametrize(
    "n,d,layout",
    [(16, 8, "C"), (32, 7, "C"), (32, 8, "C"), (65, 32, "F"), (65, 33, "F"), (65, 64, "C")],
)
def test_dispatch_and_no_extra_numerical_workspace(n, d, layout, eigen_backend_available):
    a, b = sample(n, d, layout)
    _, audit, _ = check(a, b, True, False)
    expected = eigen_backend_available and n >= 32 and d >= 8 and (layout == "C" or d <= 32)
    assert (audit["isa"] == "eigen-array-triangular") == expected
    assert audit["workspace_bytes"] == 0
    assert audit["algorithm_copy_bytes"] == b.nbytes
    _, audit, _ = check(a, b, True, False, "scalar")
    assert audit["isa"] != "eigen-array-triangular"


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("isolate", [False, True])
def test_graph_layout_transport_and_retained_result(lane, order, isolate):
    a, b = sample(65, 17, "F")
    types = {
        "a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]),
        "b": dict(kind="matrix", axes=["asset", "factor"], shape=["N", "D"]),
    }
    graph = GraphCompiler(types).compile(
        {"x": "triangular_solve(a,b)", "independent": "b+1"},
        result_format="typed",
        output_order=order,
        error_policy="isolate" if isolate else "raise",
    )
    bounds = np.zeros(2, np.int64)
    expected, error = reference(a, b, True, False)
    assert error is None
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as scheduler:
        if lane == "single":
            prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
            run = prepared.run_snapshot
        else:

            def run():
                return scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)

        first, second = run(), run()
    for index in range(2):
        value = first.outputs[0].values[index]
        same_ieee(value, expected)
        same_ieee(first.outputs[1].values[index], b + 1)
        assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
        assert not value.flags.writeable
        assert not np.shares_memory(value, b)
        assert not np.shares_memory(value, second.outputs[0].values[index])
