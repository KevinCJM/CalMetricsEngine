"""Independent native primitive contracts; no external finance project dependency."""

from __future__ import annotations

import math

import numpy as np
import pytest

from calmetrics_engine import operators as op

EXTENSIONS = [
    "normal_cdf", "aligned_shift", "recursive_filter", "argsort", "gather",
    "distinct_count", "floor",
]


def test_append_only_registry_contract():
    assert [(s["opcode"], s["id"]) for s in op.catalog()[118:]] == list(
        enumerate(EXTENSIONS, start=119)
    )
    for name in EXTENSIONS:
        spec = op.get(name).spec
        assert spec["execution_backend"] == "pybind11_aot"
        assert spec["input_policy"] == "exact_native_dtype_readonly_strided_no_copy"


@pytest.mark.parametrize("simd", ["scalar", "auto"])
def test_normal_cdf_tails_nan_scalar_and_matrix(simd):
    x = np.array([-np.inf, -10.0, -2.0, -0.0, 0.0, 2.0, 10.0, np.inf, np.nan])
    expected = np.array([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in x])
    np.testing.assert_allclose(op.normal_cdf(x, simd=simd), expected, rtol=2e-15)
    assert op.normal_cdf(0.0, simd=simd) == 0.5
    assert op.normal_cdf(-10.0, simd=simd) > 0.0
    np.testing.assert_allclose(op.normal_cdf(x.reshape(3, 3), simd=simd), expected.reshape(3, 3))


def test_floor_is_numeric_and_retains_ieee_boundaries():
    x = np.array([-np.inf, -2.1, -0.0, 0.0, 2.9, np.inf, np.nan])
    actual = op.floor(x)
    np.testing.assert_array_equal(actual, np.floor(x))
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(np.floor(x)))


@pytest.mark.parametrize("periods", [0, 1, 4, 9])
@pytest.mark.parametrize("stride", [1, 2, -1])
def test_aligned_shift_preserves_axis_and_readonly_input(periods, stride):
    original = np.array([1.0, np.nan, np.inf, 3.0, -np.inf, 6.0, 7.0, 8.0])
    x = original[::stride]
    x.flags.writeable = False
    expected = np.full(x.shape, np.nan)
    if periods < x.size:
        expected[periods:] = x[: x.size - periods]
    actual = op.aligned_shift(x, periods)
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, original)
    np.testing.assert_array_equal(original, [1, np.nan, np.inf, 3, -np.inf, 6, 7, 8])


def test_aligned_shift_defaults_empty_fill_and_old_lag():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(op.aligned_shift(x), [np.nan, 1, 2])
    np.testing.assert_array_equal(op.aligned_shift(x, 2, -7.0), [-7, -7, 1])
    assert op.aligned_shift(np.array([], dtype=np.float64)).shape == (0,)
    legacy = op.lag(x, 1)
    np.testing.assert_array_equal(legacy, [1, 2])
    assert np.shares_memory(legacy, x)


@pytest.mark.parametrize("periods,fill", [(-1, 0), (0.5, 0), (np.inf, 0), (1, np.inf)])
def test_aligned_shift_invalid_parameters(periods, fill):
    with pytest.raises((ValueError, TypeError), match="INVALID_PARAMETER"):
        op.aligned_shift(np.array([1.0]), periods, fill)


@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("emit", [0, 1])
def test_recursive_filter_explicit_state_contract(alpha, seed, emit):
    x = np.array([np.nan, 2.0, 7.0, np.inf, 4.0, 10.0])
    allowed = np.array([True, True, False, True, True, True])
    previous = 50.0
    initialized = seed != 2
    expected = []
    for i, value in enumerate(x):
        if seed == 1 and i == 0:
            expected.append(previous)
        elif not allowed[i] or not math.isfinite(value):
            expected.append(previous if initialized and emit == 0 else np.nan)
        else:
            if initialized:
                previous = (1 - alpha) * previous + alpha * value
            else:
                previous, initialized = value, True
            expected.append(previous)
    x.flags.writeable = False
    actual = op.recursive_filter(x, alpha, 50.0, allowed, seed, emit)
    np.testing.assert_allclose(actual, expected, rtol=1e-15)


def test_even_span_ema_and_coupled_kdj_mask_use_reusable_filter():
    x = np.array([2.0, 4.0, np.nan, 8.0, 10.0])
    ema = op.recursive_filter(x, 2.0 / 5.0, 0.0, np.isfinite(x), 2, 0)
    np.testing.assert_allclose(ema, [2, 2.8, 2.8, 4.88, 6.928])
    rsv = np.array([80.0, 80.0, np.nan, 80.0])
    updates = np.isfinite(rsv)
    k = op.recursive_filter(rsv, 1 / 3, 50.0, updates, 1, 0)
    d = op.recursive_filter(k, 1 / 3, 50.0, updates, 1, 0)
    np.testing.assert_allclose(k, [50, 60, 60, 200 / 3])
    np.testing.assert_allclose(d, [50, 160 / 3, 160 / 3, 520 / 9])


def test_recursive_filter_empty_and_noncontiguous_masks():
    x = np.arange(8, dtype=np.float64)[::-2]
    flags = np.array([True, False, True, True, False, True, True, True])[::-2]
    actual = op.recursive_filter(x, 1.0, 0.0, flags)
    np.testing.assert_array_equal(actual, [7, 5, 3, 3])
    assert op.recursive_filter(x[:0], 0.4, 0.0, flags[:0]).shape == (0,)


@pytest.mark.parametrize("alpha,initial,seed,emit", [
    (-0.1, 0, 0, 0), (1.1, 0, 0, 0), (np.nan, 0, 0, 0),
    (0.5, np.nan, 0, 0), (0.5, 0, 3, 0), (0.5, 0, 0, 2),
    (0.5, 0, 0.5, 0),
])
def test_recursive_filter_rejects_invalid_policies(alpha, initial, seed, emit):
    with pytest.raises((ValueError, TypeError), match="INVALID_PARAMETER"):
        op.recursive_filter(np.array([1.0]), alpha, initial, np.array([True]), seed, emit)


def test_recursive_filter_mask_shape_and_dtype_are_not_guessed():
    x = np.array([1.0, 2.0])
    with pytest.raises((ValueError, TypeError), match="SHAPE_MISMATCH"):
        op.recursive_filter(x, 0.5, 0.0, np.array([True]))
    with pytest.raises((ValueError, TypeError)):
        op.recursive_filter(x, 0.5, 0.0, np.array([1.0, 0.0]))
    with pytest.raises((ValueError, TypeError), match="INVALID_MASK"):
        op.recursive_filter(x, 0.5, 0.0, np.array([1, 2], dtype=np.uint8))


@pytest.mark.parametrize("stride", [1, 2, -1])
def test_argsort_is_stable_and_gather_preserves_payload(stride):
    x = np.array([3.0, np.nan, -0.0, np.inf, 3.0, 0.0, -np.inf, np.nan])[::stride]
    x.flags.writeable = False
    snapshot = x.copy()
    indices = op.argsort(x)
    expected = np.argsort(x, kind="stable")
    assert indices.dtype == np.int64
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(op.gather(x, indices), x[expected])
    np.testing.assert_array_equal(x, snapshot)
    np.testing.assert_array_equal(np.signbit(op.gather(x, indices)), np.signbit(x[expected]))


def test_integer_order_gather_and_count_never_pass_through_float():
    x = np.array([2**63 - 1, 2**53 + 1, -(2**63), 2**53, 2**53 + 1, -1], dtype=np.int64)
    x.flags.writeable = False
    expected = np.argsort(x, kind="stable")
    np.testing.assert_array_equal(op.argsort(x), expected)
    actual = op.gather(x, op.argsort(x))
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, x[expected])
    assert op.distinct_count(x) == 5
    assert isinstance(op.distinct_count(x), float)
    flags = np.array([True, False, False, True, True, False])
    assert op.distinct_count(x, flags) == 3
    assert op.distinct_count(x, np.zeros(6, dtype=bool)) == 0
    # -1 is an ordinary category until the schema owner explicitly excludes it.
    assert op.distinct_count(np.array([-1, -1, 0], dtype=np.int64)) == 2


def test_gather_boolean_payload_and_integer_indices_with_stride():
    flags = np.array([True, False, True, False])
    indices = np.array([3, 2, 1, 0], dtype=np.int64)[::2]
    np.testing.assert_array_equal(op.gather(flags, indices), flags[indices])
    assert op.gather(flags, indices).dtype in (np.dtype(bool), np.dtype(np.uint8))


@pytest.mark.parametrize("indices", [
    np.array([-1], dtype=np.int64), np.array([4], dtype=np.int64),
    np.array([0, 2**63 - 1], dtype=np.int64),
])
def test_gather_checks_all_indices_before_output_writes(indices):
    out = np.full(indices.size, 123.0)
    with pytest.raises(ValueError, match="INDEX_OUT_OF_BOUNDS"):
        op.gather(np.arange(4, dtype=np.float64), indices, out=out)
    np.testing.assert_array_equal(out, np.full(indices.size, 123.0))


def test_selection_empty_and_wrong_dtype_fail_closed():
    empty = np.array([], dtype=np.int64)
    assert op.distinct_count(empty) == 0
    assert op.argsort(empty).dtype == np.int64
    assert op.argsort(empty).shape == (0,)
    assert op.gather(np.array([], dtype=np.float64), empty).shape == (0,)
    for bad in [np.array([0.0]), np.array([0], dtype=np.int32)]:
        with pytest.raises((ValueError, TypeError)):
            op.gather(np.array([1.0]), bad)
        with pytest.raises((ValueError, TypeError)):
            op.distinct_count(bad)


def test_integer_output_workspace_reuse_and_no_aliasing():
    x = np.array([3.0, 1.0, 2.0])
    workspace = op.Workspace()
    out = np.empty(3, dtype=np.int64)
    assert op.argsort(x, out=out, workspace=workspace) is out
    np.testing.assert_array_equal(out, [1, 2, 0])
    assert op.argsort(x[::-1], out=out, workspace=workspace) is out
    np.testing.assert_array_equal(out, [1, 0, 2])
    ids = np.array([3, 1, 2], dtype=np.int64)
    with pytest.raises((ValueError, TypeError), match="ALIAS"):
        op.gather(ids, np.array([2, 0, 1], dtype=np.int64), out=ids)


@pytest.mark.parametrize("name", ["argsort", "distinct_count"])
def test_index_scratch_is_explicit_and_inputs_are_zero_copy(name):
    x = np.arange(20, dtype=np.int64)[::-2]
    x.flags.writeable = False
    requirements = op.get(name).requirements(x)
    assert requirements["scratch_indices"] == len(x)
    assert requirements["scratch_doubles"] == 0
    _, audit = op.call(name, x, audit=True)
    assert audit["input_addresses"] == [x.__array_interface__["data"][0]]
    assert audit["input_copy_bytes"] == 0
    assert audit["algorithm_copy_bytes"] == 0
    assert audit["workspace_bytes"] == x.size * np.dtype(np.intp).itemsize
