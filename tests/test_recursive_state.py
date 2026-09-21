"""Generic native recurrences: initialization, gaps, projections and exact states."""

from __future__ import annotations

import gc
import math

import numpy as np
import pytest

from calmetrics_engine import operators as op


def test_new_recurrence_registry_is_append_only_and_declares_semantics():
    names = ["cos", "recursive_filter_adaptive", "linear_filter2", "scalar_kalman",
             "state_estimate", "state_variance"]
    assert [(s["opcode"], s["id"]) for s in op.catalog()[125:131]] == list(
        enumerate(names, 126)
    )
    for name in names:
        spec = op.get(name).spec
        assert spec["temporal_dependency"] == "causal"
        assert spec["granularity"] in {"primitive", "coupled_kernel"}


def test_cos_radians_scalar_strided_and_ieee_missing():
    x = np.array([0.0, math.pi / 2, math.pi, -math.pi, np.nan, np.inf, -np.inf])
    x.flags.writeable = False
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(op.cos(x[::-1]), np.cos(x[::-1]), atol=1e-16)
    assert op.cos(0.0) == 1.0
    assert op.cos(x[:0]).shape == (0,)


def test_adaptive_seed_is_independent_of_updates_and_output_warmup():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    gain = np.array([np.nan, np.nan, 0.5, 1.0])
    update = np.array([False, False, True, True])
    reset = np.zeros(4, dtype=bool)
    # Seed x[0], hold internally at x[1], then use the original seed at x[2].
    actual = op.recursive_filter_adaptive(x, gain, update, reset, min_periods=3)
    np.testing.assert_array_equal(actual, [np.nan, np.nan, 20.0, 40.0])
    eligible = op.recursive_filter_adaptive(x, gain, update, reset, 0, 2, 3, 1)
    np.testing.assert_array_equal(eligible, [np.nan, np.nan, 30.0, 40.0])


@pytest.mark.parametrize("emit", [0, 1])
def test_adaptive_explicit_reset_and_gap_hold_are_independent(emit):
    x = np.array([2.0, 6.0, np.nan, 10.0, 14.0, 18.0])
    update = np.isfinite(x)
    keep = op.recursive_filter_adaptive(x, 0.5, update, np.zeros(x.size, bool),
                                       0, 1, 2, emit)
    np.testing.assert_array_equal(keep, [np.nan, 4, 4 if emit == 0 else np.nan, 7, 10.5, 14.25])
    reset = op.recursive_filter_adaptive(x, 0.5, update, ~update, 0, 1, 2, emit)
    np.testing.assert_array_equal(reset, [np.nan, 4, np.nan, np.nan, 12, 15])


def test_adaptive_explicit_initial_is_consumed_and_reset_before_row():
    x = np.array([10.0, 20.0, 30.0])
    actual = op.recursive_filter_adaptive(
        x, 0.5, np.ones(3, bool), np.array([False, False, True]), 2.0, 0, 1, 0
    )
    np.testing.assert_array_equal(actual, [6.0, 13.0, 16.0])


@pytest.mark.parametrize("gain", [np.nan, np.inf, -0.1, 1.01])
def test_adaptive_validates_only_consumed_alpha(gain):
    x = np.array([3.0, 8.0, 10.0])
    alpha = np.array([gain, gain, 0.5])
    update = np.array([True, False, True])
    np.testing.assert_array_equal(
        op.recursive_filter_adaptive(x, alpha, update, np.zeros(3, bool)),
        [3.0, np.nan, 6.5],
    )
    update[1] = True
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        op.recursive_filter_adaptive(x, alpha, update, np.zeros(3, bool))


def test_adaptive_readonly_negative_strides_and_no_input_copy():
    source = np.arange(12, dtype=np.float64)
    x = source[::-2]
    gain = np.linspace(0.1, 0.8, 12)[::-2]
    update = np.ones(12, dtype=bool)[::-2]
    reset = np.zeros(12, dtype=bool)[::-2]
    x.flags.writeable = False
    expected = [x[0]]
    for value, alpha in zip(x[1:], gain[1:], strict=True):
        expected.append(expected[-1] + alpha * (value - expected[-1]))
    result, audit = op.recursive_filter_adaptive(x, gain, update, reset, audit=True)
    np.testing.assert_allclose(result, expected)
    assert audit["input_copy_bytes"] == 0
    np.testing.assert_array_equal(source, np.arange(12, dtype=np.float64))
    assert not np.shares_memory(result, x)


def _kama_reference(x, window, fast, slow):
    result = np.full(x.size, np.nan)
    count, previous = 0, 0.0
    for t, value in enumerate(x):
        if not np.isfinite(value):
            count = 0
            continue
        if count == 0:
            previous = value
        count += 1
        if count <= window:
            continue
        noise = sum(abs(x[j] - x[j - 1]) for j in range(t - window + 1, t + 1))
        efficiency = abs(value - x[t - window]) / noise if noise > 0 else 0.0
        gain = (efficiency * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
        previous += gain * (value - previous)
        result[t] = previous
    return result


@pytest.mark.parametrize("flat", [False, True])
def test_kama_is_composition_with_true_seed_freeze_and_gap_rewarm(flat):
    x = np.full(55, 10.0) if flat else np.random.default_rng(71).normal(size=55).cumsum() + 100
    x[[0, 19, 38]] = [np.nan, np.inf, np.nan]
    window, fast, slow = 5, 2, 20
    differences = op.absolute(op.subtract(x, op.aligned_shift(x)))
    noise = op.multiply(op.rolling_mean(differences, window, window), window)
    movement = op.absolute(op.subtract(x, op.aligned_shift(x, window)))
    positive = op.greater_than(noise, 0.0)
    efficiency = op.where(positive, op.divide(movement, op.where(positive, noise, 1.0)), 0.0)
    gain = op.power(op.add(op.multiply(efficiency, 2 / (fast + 1) - 2 / (slow + 1)),
                           2 / (slow + 1)), 2.0)
    update = op.logical_and(op.finite_mask(noise), op.finite_mask(x))
    actual = op.recursive_filter_adaptive(x, gain, update, op.logical_not(op.finite_mask(x)),
                                         min_periods=window + 1)
    np.testing.assert_allclose(actual, _kama_reference(x, window, fast, slow),
                               rtol=3e-14, atol=1e-12, equal_nan=True)
    assert np.isnan(actual[20:25]).all()
    assert np.isfinite(actual[25])


def _super_smoother_reference(x, period):
    theta = math.sqrt(2) * math.pi / period
    decay = math.exp(-theta)
    c2, c3 = 2 * decay * math.cos(theta), -decay * decay
    c1 = 1 - c2 - c3
    result = np.full(x.size, np.nan)
    count, previous_input, previous, older = 0, 0.0, 0.0, 0.0
    for t, value in enumerate(x):
        if not np.isfinite(value):
            count = 0
            continue
        current = value if count < 2 else c1 * (value + previous_input) / 2 + c2 * previous + c3 * older
        if not np.isfinite(current):
            count = 0
            continue
        older, previous, previous_input = previous, current, value
        count += 1
        if count >= period:
            result[t] = current
    return result


@pytest.mark.parametrize("period", [3, 10, 32])
def test_super_smoother_composes_coefficients_and_two_state_filter(period):
    x = np.random.default_rng(8).normal(size=180).cumsum() + 100
    x[[0, 70, 110]] = [np.nan, np.inf, -np.inf]
    theta = op.divide(op.multiply(op.sqrt(2.0), math.pi), float(period))
    decay = op.exp(op.negate(theta))
    c2, c3 = op.multiply(op.multiply(2.0, decay), op.cos(theta)), op.negate(op.power(decay, 2.0))
    c1 = op.subtract(op.subtract(1.0, c2), c3)
    actual = op.linear_filter2(x, c1 / 2, c1 / 2, c2, c3,
                              op.logical_not(op.finite_mask(x)), period)
    np.testing.assert_allclose(actual, _super_smoother_reference(x, period), atol=1e-10)
    assert np.isnan(actual[71:71 + period - 1]).all()
    assert np.isfinite(actual[71 + period - 1])


def test_second_order_zero_state_impulse_and_explicit_reset():
    x = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    reset = np.array([False, False, False, True, False, False])
    actual = op.linear_filter2(x, 1.0, 0.0, 0.5, 0.25, reset, 1, 0)
    np.testing.assert_array_equal(actual, [1, 0.5, 0.5, 1, 0.5, 0.5])


def test_second_order_arithmetic_failure_resets_but_raw_gap_is_explicit():
    x = np.array([1e308, 1e308, 1e308, 1.0, 2.0, 3.0])
    actual = op.linear_filter2(x, 2.0, 0.0, 1.0, 0.0, np.zeros(6, bool), 3)
    np.testing.assert_array_equal(actual, [np.nan, np.nan, np.nan, np.nan, np.nan, 8])
    x = np.array([1.0, 2.0, np.nan, 3.0])
    held = op.linear_filter2(x, 1.0, 0.0, 1.0, 0.0, np.zeros(4, bool))
    reset = op.linear_filter2(x, 1.0, 0.0, 1.0, 0.0, ~np.isfinite(x))
    np.testing.assert_array_equal(held, [1, 2, np.nan, 5])
    np.testing.assert_array_equal(reset, [1, 2, np.nan, 3])


def test_kalman_no_process_noise_matches_closed_form_posterior():
    x = np.array([10.0, 12.0, 8.0, 16.0, 15.0])
    measurement, initial_variance = 2.0, 3.0
    state = op.scalar_kalman(x, 0.0, measurement, initial_variance)
    precision = 1 / initial_variance + np.arange(x.size) / measurement
    expected_variance = 1 / precision
    expected_estimate = (x[0] / initial_variance + np.r_[0.0, np.cumsum(x[1:]) / measurement]) / precision
    np.testing.assert_allclose(op.state_estimate(state), expected_estimate, rtol=1e-14)
    np.testing.assert_allclose(op.state_variance(state), expected_variance, rtol=1e-14)


def test_kalman_gaps_never_reset_or_advance_variance():
    x = np.array([np.nan, 10.0, 12.0, np.inf, -np.inf, np.nan, 8.0, 9.0])
    state = op.scalar_kalman(x, 0.25, 2.0)
    finite = np.isfinite(x)
    np.testing.assert_array_equal(state[finite], op.scalar_kalman(x[finite], 0.25, 2.0))
    assert np.isnan(state[~finite]).all()
    np.testing.assert_array_equal(state[1], [10, 1])


def test_kalman_projections_share_one_result_and_retain_owner():
    state, audit = op.scalar_kalman(np.arange(10, dtype=float), 0.1, 0.2, audit=True)
    estimate = op.state_estimate(state)
    variance = op.state_variance(state)
    assert np.shares_memory(state, estimate) and np.shares_memory(state, variance)
    assert estimate.strides == variance.strides == (16,)
    assert not estimate.flags.writeable and not variance.flags.writeable
    copied = np.empty(10)
    assert op.state_variance(state, out=copied) is copied
    np.testing.assert_array_equal(copied, variance)
    before = estimate.copy()
    del state
    gc.collect()
    np.testing.assert_array_equal(estimate, before)
    assert audit["input_copy_bytes"] == 0


@pytest.mark.parametrize("name", ["recursive_filter_adaptive", "linear_filter2", "scalar_kalman"])
def test_new_recurrences_empty_input(name):
    x = np.empty(0)
    mask = np.empty(0, bool)
    if name == "recursive_filter_adaptive":
        actual = op.recursive_filter_adaptive(x, x, mask, mask)
    elif name == "linear_filter2":
        actual = op.linear_filter2(x, 1, 0, 0, 0, mask)
    else:
        actual = op.scalar_kalman(x, 0, 1)
        assert op.state_estimate(actual).shape == (0,)
        assert op.state_variance(actual).shape == (0,)
    assert actual.shape[0] == 0


@pytest.mark.parametrize("q,r,p", [(-1, 1, 1), (0, 0, 1), (1, -1, 1),
                                    (np.nan, 1, 1), (0, np.inf, 1), (0, 1, -1)])
def test_kalman_rejects_invalid_variances(q, r, p):
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        op.scalar_kalman(np.array([1.0, 2.0]), q, r, p)


def test_recurrence_shape_mask_and_policy_fail_closed():
    x, mask = np.ones(3), np.ones(3, bool)
    with pytest.raises(ValueError, match="SHAPE_MISMATCH"):
        op.recursive_filter_adaptive(x, np.ones(2), mask, mask)
    with pytest.raises(ValueError, match="INVALID_MASK"):
        op.recursive_filter_adaptive(x, 0.5, np.array([1, 2, 0], np.uint8), mask)
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        op.recursive_filter_adaptive(x, 0.5, mask, mask, min_periods=0)
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        op.linear_filter2(x, 1, 0, np.inf, 0, mask)
    with pytest.raises(ValueError, match="SHAPE_MISMATCH"):
        op.state_variance(np.ones((3, 3)))


def test_exact_integer_equality_preserves_large_codes_and_mask_composition():
    x = np.array([-(2**63), 2**53, 2**53 + 1, 2**63 - 1, -1], np.int64)
    y = np.array([-(2**63), 2**53 + 1, 2**53, 2**63 - 1, 0], np.int64)
    np.testing.assert_array_equal(op.equal(x, y), x == y)
    np.testing.assert_array_equal(op.not_equal(x, y), x != y)
    np.testing.assert_array_equal(op.equal(x, -1.0), x == -1)
    np.testing.assert_array_equal(op.not_equal(-1.0, x), x != -1)
    np.testing.assert_array_equal(op.equal(x[::-1], y[::-1]), (x == y)[::-1])
    for scalar in [0.5, np.nan, np.inf, float(2**53)]:
        with pytest.raises(ValueError, match="INVALID_INTEGER_SCALAR"):
            op.equal(x, scalar)
    with pytest.raises((ValueError, TypeError)):
        op.equal(x, x.astype(float))


def test_state_select_nested_mapping_connects_to_confirmation():
    x = np.array([3.0, 4.0, -3.0, -4.0, 0.0, np.nan, 5.0])
    valid = op.finite_mask(x)
    low_or_neutral = op.state_select(op.less_equal(x, -2.0), 2, 1, valid)
    state = op.state_select(op.greater_equal(x, 2.0), 0, low_or_neutral, valid)
    expected = np.array([0, 0, 2, 2, 1, -1, 0], np.int64)
    assert state.dtype == np.int64
    np.testing.assert_array_equal(state, expected)
    np.testing.assert_array_equal(op.state_confirm(state, 2, 1),
                                  op.state_confirm(expected, 2, 1))
    spec = op.get("state_select").spec
    assert spec["opcode"] == 146
    assert spec["granularity"] == "primitive"
    assert spec["temporal_dependency"] == "causal"


def test_state_select_exact_large_codes_valid_zero_and_no_proposal():
    condition = np.array([True, False, True, False, True])
    valid = np.array([True, True, False, True, True])
    high = np.array([2**63 - 1, 0, 2**53 + 1, 0, -1], np.int64)
    high.flags.writeable = False
    result = op.state_select(condition[::-1], high[::-1], 0, valid[::-1])
    np.testing.assert_array_equal(result, [-1, 0, -1, 0, 2**63 - 1])
    no_proposal = op.state_select(condition, -1, 0, valid)
    np.testing.assert_array_equal(no_proposal, [-1, 0, -1, 0, -1])
    assert op.state_select(condition[:0], high[:0], 0, valid[:0]).shape == (0,)


@pytest.mark.parametrize("code", [-2.0, 0.5, np.nan, np.inf, float(2**53)])
def test_state_select_rejects_invalid_scalar_codes(code):
    with pytest.raises(ValueError, match="INVALID_STATE_CODE"):
        op.state_select(np.ones(2, bool), code, 0, np.ones(2, bool))


def test_state_select_rejects_float_code_arrays_bad_codes_and_masks_before_writes():
    condition, valid = np.array([True, False]), np.ones(2, bool)
    with pytest.raises((ValueError, TypeError)):
        op.state_select(condition, np.array([0.0, 1.0]), 0, valid)
    with pytest.raises(ValueError, match="SHAPE_MISMATCH"):
        op.state_select(condition, np.zeros(3, np.int64), 0, valid)
    with pytest.raises(ValueError, match="SHAPE_MISMATCH"):
        op.state_select(condition, 0, 1, valid[:1])
    output = np.full(2, 99, np.int64)
    with pytest.raises(ValueError, match="INVALID_STATE_CODE"):
        op.state_select(condition, np.array([0, -2], np.int64), 1, valid, out=output)
    np.testing.assert_array_equal(output, [99, 99])
    with pytest.raises(ValueError, match="INVALID_MASK"):
        op.state_select(np.array([1, 2], np.uint8), 0, 1, valid, out=output)
    np.testing.assert_array_equal(output, [99, 99])
