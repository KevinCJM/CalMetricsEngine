"""Adjacent ranks are sufficient; retain scalar IEEE/error/ownership contracts."""

import numpy as np
import pytest

from calmetrics_engine import operators as op


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
def test_one_quantile_reuses_rank_selection_without_sorting(layout):
    rng = np.random.default_rng(516)
    x = rng.normal(size=(3, 2049))
    x[:, ::3] = 0.01
    x = np.array(x, order="F" if layout == "F" else "C")
    if layout == "negative":
        x = x[::-1, ::-1]
    if layout == "padded":
        owner = np.full((3, 4098), np.nan)
        owner[:, ::2] = x
        x = owner[:, ::2]
    x.flags.writeable = False
    saved = x.copy()
    for probability in [0.0, 0.05, 0.5, 0.999, 1.0]:
        q = np.array([probability])
        q.flags.writeable = False
        expected = op.quantiles_axis(x, q, 1, simd="scalar")
        result, audit = op.quantiles_axis(x, q, 1, audit=True)
        np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
        assert audit["isa"] in ["highway_vqselect", "std-nth-element-quantile"]
        assert audit["highway_vqsort_calls"] == 0
        assert audit["algorithm_copy_bytes"] == x.size * 8
        assert audit["input_copy_bytes"] == 0
        assert audit["workspace_bytes"] == 2049 * 8
        assert not np.shares_memory(result, x)
        second = op.quantiles_axis(x, np.array([1 - probability]), 1)
        np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
        assert not np.shares_memory(result, second)
    np.testing.assert_array_equal(x, saved)


def test_single_quantile_special_bits_use_canonical_sort():
    for special in [np.nan, -0.0, np.nextafter(0.0, 1.0)]:
        x = np.arange(2049, dtype=float)
        x[17] = special
        for p in [0.0, 0.05, 0.5, 1.0]:
            expected = op.quantiles_axis(x, np.array([p]), 0, simd="scalar")
            actual, audit = op.quantiles_axis(x, np.array([p]), 0, audit=True)
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            assert audit["isa"] not in ["highway_vqselect", "std-nth-element-quantile"]
    x = np.arange(2049, dtype=float)
    x[[0, -1]] = [-np.inf, np.inf]
    for p in [0.0, 0.05, 0.5, 1.0]:
        expected = op.quantiles_axis(x, np.array([p]), 0, simd="scalar")
        actual = op.quantiles_axis(x, np.array([p]), 0)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
def test_mask_structure_check_keeps_invalid_bytes_and_empty_domains(layout):
    x = np.ones((3, 2049), order="F" if layout == "F" else "C")
    mask = np.ones(x.shape, dtype=bool, order="F" if layout == "F" else "C")
    if layout == "padded":
        owner = np.zeros((3, 4098), dtype=bool)
        owner[:, ::2] = mask
        mask = owner[:, ::2]
    if layout == "negative":
        mask = mask[::-1, ::-1]
    mask.view(np.uint8).flat[-1] = 2
    target = np.full(x.shape, -777.0)
    with pytest.raises(ValueError, match="INVALID_MASK"):
        op.where(mask, x, 0.0, out=target)
    np.testing.assert_array_equal(target, -777.0)
    np.testing.assert_array_equal(
        op.where(np.empty((0, 2049), bool), np.empty((0, 2049)), 0.0), np.empty((0, 2049))
    )
