"""Many short lines select two ranks with the same canonical numeric contract."""

import numpy as np
import pytest

from calmetrics_engine import operators as op


@pytest.mark.parametrize("count", [127, 128, 129, 252, 253, 1023, 1024])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("probability", [0.0, 0.05, 0.5, 0.999, 1.0])
def test_short_single_probability_rank_selection_keeps_exact_math_and_owners(
    count, axis, layout, probability
):
    rng = np.random.default_rng(8515)
    x = rng.normal(size=(3, count))
    x[:, ::3] = 0.01
    x[0, 1] = 0.0
    if layout == "F":
        x = np.array(x, order="F")
    elif layout == "negative":
        x = x[::-1, ::-1]
    elif layout == "padded":
        owner = np.full((3, count * 2), np.nan)
        owner[:, ::2] = x
        x = owner[:, ::2]
    if axis == 0:
        x = x.T
    x.flags.writeable = False
    saved = x.copy()
    p = np.array([probability])
    p.flags.writeable = False
    expected = op.quantiles_axis(x, p, axis, simd="scalar")
    workspace = op.Workspace()
    actual, audit = op.quantiles_axis(x, p, axis, workspace=workspace, audit=True)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    if count >= 128:
        assert audit["isa"] in ["std-nth-element-quantile", "highway_vqselect"]
    else:
        assert audit["isa"] not in ["std-nth-element-quantile", "highway_vqselect"]
    assert audit["highway_vqsort_calls"] == 0
    assert audit["algorithm_copy_bytes"] == x.size * 8 and audit["input_copy_bytes"] == 0
    assert audit["workspace_bytes"] == count * 8
    second = op.quantiles_axis(x, np.array([1 - probability]), axis, workspace=workspace)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(x, saved)
    assert not np.shares_memory(actual, second) and not np.shares_memory(actual, x)


@pytest.mark.parametrize("count", [128, 252, 1023])
@pytest.mark.parametrize("special", ["nan", "negative_zero", "subnormal", "infinities"])
@pytest.mark.parametrize("probability", [0.0, 0.05, 0.5, 0.999, 1.0])
def test_short_selection_preserves_exceptional_bits_and_canonical_interpolation(
    count, special, probability
):
    x = np.arange(count, dtype=float)
    if special == "nan":
        x[count // 2] = np.nan
    elif special == "negative_zero":
        x[count // 2] = -0.0
    elif special == "subnormal":
        x[count // 2] = np.nextafter(0.0, 1.0)
    else:
        x[[0, -1]] = [-np.inf, np.inf]
    x.flags.writeable = False
    saved = x.copy()
    p = np.array([probability])
    expected = op.quantiles_axis(x, p, 0, simd="scalar")
    actual, audit = op.quantiles_axis(x, p, 0, audit=True)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(x.view(np.uint64), saved.view(np.uint64))
    if special != "infinities":
        assert audit["isa"] not in ["std-nth-element-quantile", "highway_vqselect"]
    else:
        assert audit["isa"] in ["std-nth-element-quantile", "highway_vqselect"]


@pytest.mark.parametrize("count", [128, 252, 1023])
@pytest.mark.parametrize("axis", [0, 1])
def test_multiple_short_probabilities_keep_one_full_canonical_sort(count, axis):
    x = np.random.default_rng(8615).normal(size=(3, count))
    p = np.array([0.05, 0.5, 0.95])
    if axis == 0:
        x = x.T
    expected = op.quantiles_axis(x, p, axis, simd="scalar")
    actual, audit = op.quantiles_axis(x, p, axis, audit=True)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert audit["isa"] not in ["std-nth-element-quantile", "highway_vqselect"]
