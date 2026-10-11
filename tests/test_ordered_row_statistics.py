"""Row-major library reductions retain each column's sequence and error prefix."""

import numpy as np
import pytest

from calmetrics_engine import operators as op


@pytest.mark.parametrize("width", [1, 7, 8, 9, 17, 30, 32, 64, 129, 255, 256, 257])
@pytest.mark.parametrize("rows", [1, 31, 32, 65])
@pytest.mark.parametrize("name", ["sum_axis", "mean_axis", "variance_axis"])
def test_row_statistics_match_explicit_scalar_order(rows, width, name):
    x = np.random.default_rng(rows + width).normal(size=(rows, width))
    x.flags.writeable = False
    function = getattr(op, name)
    expected = function(x, 0, simd="scalar")
    actual = function(x, 0)
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, x) and not np.shares_memory(actual, expected)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "square_overflow", "sum_overflow"])
@pytest.mark.parametrize("column", [0, 4, 8, 17, 31])
def test_variance_replay_preserves_supplied_output_failure_prefix(bad, column):
    x = np.random.default_rng(195).normal(size=(64, 32))
    if bad == "square_overflow":
        x[:, column] = np.tile([1e308, -1e308], 32)
    elif bad == "sum_overflow":
        x[:, column] = 1e308
    else:
        x[20, column] = bad
    scalar = np.full(32, 179.25)
    automatic = scalar.copy()
    with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
        op.variance_axis(x, 0, out=scalar, simd="scalar")
    with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
        op.variance_axis(x, 0, out=automatic)
    np.testing.assert_array_equal(automatic, scalar)


@pytest.mark.parametrize("name", ["sum_axis", "mean_axis", "variance_axis"])
def test_signed_zero_cancellation_and_reversed_layouts(name):
    x = np.zeros((64, 32))
    x[:, ::2] = -0.0
    x[:, 1] = np.tile([1e15, 1, -1e15, 1], 16)
    function = getattr(op, name)
    for data in [x, x[::-1], x[:, ::-1], np.asfortranarray(x)]:
        actual = function(data, 0)
        expected = function(data, 0, simd="scalar")
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
