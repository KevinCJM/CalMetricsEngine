"""Large single-RHS LU reuses Eigen without adopting multi-RHS NaN semantics."""
import numpy as np
import pytest

from calmetrics_engine import operators as op

from .test_solve_library import residual, same_ieee, sample


@pytest.mark.parametrize("n", [1024, 1025])
@pytest.mark.parametrize("layout", ["c", "f", "negative"])
def test_single_rhs_grouped_pivots_and_owned_result(n, layout):
    a, b = sample("solve", n, layout=layout)
    indices = np.array([0, 32, 65, n - 1])
    a[indices] = a[np.roll(indices, 1)]
    before_a, before_b = a.copy(), b.copy()
    a.flags.writeable = b.flags.writeable = False
    expected = op.solve(a, b, simd="scalar")
    storage = np.full(n + 2, -913.25)
    actual, audit = op.solve(a, b, out=storage[1:-1], audit=True)
    same_ieee(actual, expected)
    residual(a, actual, b)
    same_ieee(a, before_a)
    same_ieee(b, before_b)
    np.testing.assert_array_equal(storage[[0, -1]], -913.25)
    assert audit["workspace_bytes"] == (n * n + n) * 8
    assert audit["algorithm_copy_bytes"] == (n * n + n) * 8


@pytest.mark.parametrize("fault", ["nan_pivot", "overflow", "zero_times_inf"])
def test_single_rhs_nonfinite_propagation_across_grouped_panels(fault):
    n = 1024
    a, b = np.eye(n) * 2, np.ones(n)
    maximum = np.finfo(np.float64).max
    if fault == "nan_pivot":
        a[64, 64] = np.nan
    elif fault == "overflow":
        a[64:66, 64:66] = [[maximum, maximum], [-maximum, maximum]]
    else:
        a[0, 0], a[1, 0] = 1, -1
        a[0, -1] = a[1, -1] = maximum
    expected = op.solve(a, b, simd="scalar")
    actual = op.solve(a, b)
    same_ieee(actual, expected)
    if fault == "overflow":
        # The historical solver permits the infinite pivot: division yields
        # zero rather than an exception or necessarily a NaN result.
        assert np.isfinite(actual).all()
        assert actual[64] == 1.0 / maximum and actual[65] == 0.0
    else:
        assert np.isnan(actual).any()
