"""Native state/event contracts, with fixed platform-oracle cases and local references.

No platform import, NJIT compilation, or numerical Python fallback is used by the engine.
"""

import numpy as np
import pytest

from calmetrics_engine import operators as op


def ints(values):
    return np.asarray(values, dtype=np.int64)


def test_hysteresis_inclusive_thresholds_gap_retention_and_jump_precedence():
    x = np.array([0., 2., np.nan, 1.5, 1., -2., np.inf, -1.5, -1., 2., -2.])
    actual = op.state_hysteresis(x, 2., 1., -2., -1.)
    np.testing.assert_array_equal(actual, [1, 0, -1, 0, 1, 2, -1, 2, 1, 0, 2])
    assert actual.dtype == np.int64
    # The native kernel preserves the platform's explicit comparison order even
    # if a caller intentionally supplies overlapping thresholds.
    np.testing.assert_array_equal(op.state_hysteresis(np.array([0., -1.]), 0, 0, 0, 0), [0, 2])


def test_confirmation_minimum_hold_is_one_joint_recurrence():
    codes = ints([0, 0, 1, 1, 1, -1, 0, 0, 1])
    np.testing.assert_array_equal(op.state_confirm(codes, 2, 3), [-1, 0, 0, 0, 1, -1, 1, 0, 0])
    # Missing observations count toward active duration, reset the candidate
    # streak, and emit unknown without deleting the retained active state.
    np.testing.assert_array_equal(op.state_confirm(ints([0, -1, -1, 1]), 1, 3), [0, -1, -1, 1])
    np.testing.assert_array_equal(op.state_confirm(ints([0, 1, 1, 0]), 1, 1), [0, 1, 1, 0])
    # Category identities above the float64 exact-integer range remain exact.
    large = 2**60
    np.testing.assert_array_equal(op.state_confirm(ints([large, large + 1]), 1, 1), [large, large + 1])


def test_continuous_candidate_absence_is_not_missing_observation():
    candidates = ints([-1, 1, -1, 1, 1, -1, 1, 0])
    initial = ints([0, -1, -1, -1, -1, -1, -1, -1])
    price = np.ones(8)
    actual = op.state_continuous(candidates, initial, price, 2, 3)
    expected = ints([[0, 0, 0], [0, 3, 1], [0, 0, 0], [0, 3, 1],
                     [1, 1, 0], [1, 2, 0], [1, 1, 0], [1, 3, 1]])
    np.testing.assert_array_equal(actual, expected)
    for name, column in [("continuous_state_values", 0), ("continuous_state_evidence", 1),
                         ("continuous_state_pending", 2)]:
        projected = op.call(name, actual)
        np.testing.assert_array_equal(projected, expected[:, column])
        assert projected.dtype == np.int64
        assert np.shares_memory(projected, actual)
        assert projected.strides == (3 * np.dtype(np.int64).itemsize,)
        explicit = np.empty(len(actual), dtype=np.int64)
        op.call(name, actual, out=explicit)
        np.testing.assert_array_equal(explicit, expected[:, column])


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0., -1.])
def test_continuous_rejects_invalid_observation_before_writing_output(bad):
    price = np.array([1., bad, 2.])
    output = np.full((3, 3), 99, dtype=np.int64)
    with pytest.raises(ValueError, match="CONTINUOUS_STATE_OBSERVATION_MISSING"):
        op.state_continuous(ints([-1, 1, 1]), ints([0, 0, 0]), price, 2, 3, out=output)
    np.testing.assert_array_equal(output, np.full((3, 3), 99))


def test_continuous_validates_unused_initial_codes_and_requires_first_state():
    with pytest.raises(ValueError, match="CONTINUOUS_STATE_INITIAL_REQUIRED"):
        op.state_continuous(ints([-1, -1]), ints([-1, 0]), np.ones(2), 1, 3)
    with pytest.raises(ValueError, match="CONTINUOUS_STATE_CODE_INVALID"):
        op.state_continuous(ints([-1, -1]), ints([0, 7]), np.ones(2), 1, 3)


def test_realtime_drawdown_transitions_gap_reset_and_one_step_per_observation():
    price = np.array([100., 85., 80., 100., 100., 79., 90., 90., 95., 100.])
    drawdown = np.array([0., -.15, -.2, 0., 0., -.21, -.1, -.1, -.05, 0.])
    valid = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.uint8)
    # A recovery jump at row 3 enters Recovery, then row 4 exits to Normal.
    # An invalid window resets; the dead band at row 7 remains unknown.
    np.testing.assert_array_equal(
        op.drawdown_cycle_state(price, drawdown, valid, .12, .05, .08),
        [0, 2, 2, 1, 0, 2, -1, -1, 0, 0],
    )


def test_realtime_drawdown_prefix_invariance_and_explicit_rolling_input():
    price = np.array([100., 110., 90., 92., 98., 80., 86., 103., 101.])
    peak = np.array([price[max(0, t - 3):t + 1].max() for t in range(len(price))])
    drawdown = price / peak - 1
    valid = np.ones(price.size, dtype=bool)
    expected = op.drawdown_cycle_state(price, drawdown, valid, .12, .05, .08)
    for end in range(1, price.size + 1):
        actual = op.drawdown_cycle_state(price[:end], drawdown[:end], valid[:end], .12, .05, .08)
        np.testing.assert_array_equal(actual, expected[:end])


def test_extrema_earliest_plateau_split_missing_and_endpoint_exclusion():
    x = np.array([1., 3., 3., 1., 2., 1., np.nan, 2., 4., 1., 3.])
    expected = ints([0, 1, 0, -1, 1, 0, -2, 0, 1, -1, 0])
    np.testing.assert_array_equal(op.local_extrema(x, 1, 1, 0, 0), expected)
    np.testing.assert_array_equal(op.local_extrema(x, 1, 1, 2, 2),
                                  [0, 0, 0, -1, 0, 0, -2, 0, 0, 0, 0])
    np.testing.assert_array_equal(op.local_extrema(np.ones(6), 1, 1, 0, 0), np.zeros(6))


def test_extrema_and_ps_preserve_gaps_as_missing_events():
    prices = np.array([1., 3., 1., 0., 2., 5., 2., np.inf, 2., 4., 1.])
    events = op.local_extrema(prices, 1, 1, 0, 0)
    np.testing.assert_array_equal(events, [0, 1, 0, -2, 0, 1, 0, -2, 0, 1, 0])
    selected = op.ps_filter(prices, events, 1, 2, .2)
    np.testing.assert_array_equal(selected, events)


def test_ps_strict_amplitude_exception_and_full_cycle_constraint():
    prices = np.array([100., 80., 100.])
    events = ints([1, -1, 1])
    # A movement exactly equal to the exception does not exempt a short phase.
    np.testing.assert_array_equal(op.ps_filter(prices, events, 2, 2, .25), [0, -1, 0])
    np.testing.assert_array_equal(op.ps_filter(prices, events, 2, 2, .19), events)
    # Equal same-kind endpoints keep the earlier turn when a cycle is too short.
    np.testing.assert_array_equal(op.ps_filter(prices, events, 1, 3, 0.), [1, 0, 0])


def test_ps_endpoint_censoring_can_change_an_earlier_result():
    prices = np.array([1., 3., 2.])
    events = ints([0, 1, 0])
    np.testing.assert_array_equal(op.ps_filter(prices, events, 1, 2, .2), events)
    extended = op.ps_filter(np.array([1., 3., 2., 4.]), ints([0, 1, 0, 0]), 1, 2, .2)
    np.testing.assert_array_equal(extended, [0, 0, 0, 0])


def test_between_events_complete_half_open_intervals_only():
    events = ints([0, 1, 0, -1, 0, 1, -2, -1, 0, 1, 0])
    segments = op.between_events(events)
    expected = ints([[-1, -1], [1, 3], [1, 3], [3, 5], [3, 5], [-1, -1],
                     [-1, -1], [7, 9], [7, 9], [-1, -1], [-1, -1]])
    np.testing.assert_array_equal(segments, expected)
    for name, column in [("segment_starts", 0), ("segment_ends", 1)]:
        values = op.call(name, segments)
        np.testing.assert_array_equal(values, expected[:, column])
        assert np.shares_memory(values, segments)
    # Repeated same-kind events replace the pending boundary; no false phase.
    np.testing.assert_array_equal(op.between_events(ints([1, 1, -1])), [[-1, -1], [1, 2], [-1, -1]])


@pytest.mark.parametrize("stride", [1, 2, -1])
def test_readonly_strided_inputs_output_ownership_and_workspace(stride):
    prices = np.array([1., 3., 1., 4., 2., 5., 2., 6.])[::stride]
    prices.flags.writeable = False
    snapshot = prices.copy()
    events, audit = op.call("local_extrema", prices, 1, 1, 0, 0, audit=True)
    assert audit["input_copy_bytes"] == 0
    assert audit["input_addresses"][0] == prices.__array_interface__["data"][0]
    workspace = op.Workspace()
    output = np.empty(len(prices), dtype=np.int64)
    assert op.ps_filter(prices, events, 1, 2, .2, workspace=workspace, out=output) is output
    saved = output.copy()
    independent = op.ps_filter(prices, events, 1, 2, .2)
    op.ps_filter(prices, ints(np.zeros(prices.size)), 1, 2, .2, workspace=workspace, out=output)
    np.testing.assert_array_equal(independent, saved)
    np.testing.assert_array_equal(prices, snapshot)
    requirements = op.get("ps_filter").requirements(prices, events, 1, 2, .2)
    assert requirements["scratch_indices"] == prices.size
    assert requirements["scratch_doubles"] == 0


@pytest.mark.parametrize("name,args", [
    ("state_hysteresis", (np.array([]), 2, 1, -2, -1)),
    ("state_confirm", (ints([]), 1, 1)),
    ("state_continuous", (ints([]), ints([]), np.array([]), 1, 3)),
    ("drawdown_cycle_state", (np.array([]), np.array([]), np.array([], dtype=bool), .12, .05, .08)),
    ("local_extrema", (np.array([]), 1, 1, 0, 0)),
    ("ps_filter", (np.array([]), ints([]), 1, 2, .2)),
    ("between_events", (ints([]),)),
])
def test_empty_is_well_shaped_int64(name, args):
    result = op.call(name, *args)
    assert result.dtype == np.int64
    assert result.shape[0] == 0


@pytest.mark.parametrize("name,args,error", [
    ("state_confirm", (ints([0]), 0, 1), "INVALID_PARAMETER"),
    ("state_confirm", (ints([-2]), 1, 1), "INVALID_STATE_CODE"),
    ("state_confirm", (np.array([0.]), 1, 1), "DTYPE"),
    ("state_continuous", (ints([0]), ints([0]), np.ones(1), 253, 3), "INVALID_PARAMETER"),
    ("local_extrema", (np.ones(1), 0, 1, 0, 0), "INVALID_PARAMETER"),
    ("ps_filter", (np.ones(1), ints([2]), 1, 2, .2), "INVALID_EVENT_CODE"),
    ("between_events", (ints([-3]),), "INVALID_EVENT_CODE"),
    ("segment_starts", (np.zeros((2, 3), np.int64),), "SHAPE_MISMATCH"),
    ("drawdown_cycle_state", (np.ones(1), np.array([.1]), np.ones(1, bool), .12, .05, .08), "INVALID_DRAWDOWN"),
    ("drawdown_cycle_state", (np.ones(1), np.array([-.1]), np.array([2], np.uint8), .12, .05, .08), "INVALID_MASK"),
])
def test_invalid_contracts_fail_closed(name, args, error):
    with pytest.raises((ValueError, TypeError), match=error):
        op.call(name, *args)


def test_phase_and_reference_cycle_keep_shared_pivot_ownership():
    events = ints([0, 1, 0, -1, 0, 1, 0, -1, 0])
    segments = op.between_events(events)
    phases = op.phase_direction(events, segments)
    np.testing.assert_array_equal(phases, [-1, 1, 1, 0, 0, 1, 1, -1, -1])
    changes = np.array([np.nan, -.2, -.2, .25, .25, -.1, -.1, np.nan, np.nan])
    states = op.drawdown_cycle_reference(phases, changes, segments, .12)
    # Peak 5 keeps the previous Recovery label; terminal trough 7 completes the
    # last Normal decline although phase_direction itself is half-open.
    np.testing.assert_array_equal(states, [-1, 0, 2, 2, 1, 1, 0, 0, -1])


def test_reference_cycle_missing_change_resets_memory_but_empty_positions_do_not():
    events = ints([1, 0, -1, -2, -1, 0, 1])
    segments = op.between_events(events)
    phases = op.phase_direction(events, segments)
    changes = np.array([-.2, -.2, np.nan, np.nan, .25, .25, np.nan])
    # This is the existing platform contract: absent segment positions do not
    # reset cross-segment pressure. Changing it requires a new algorithm version.
    np.testing.assert_array_equal(op.drawdown_cycle_reference(phases, changes, segments, .12),
                                  [0, 2, 2, -1, 2, 1, 1])
    changes[:2] = np.nan
    np.testing.assert_array_equal(op.drawdown_cycle_reference(phases, changes, segments, .12),
                                  [-1, -1, -1, -1, 0, 0, 0])


def test_segment_geometry_is_validated_before_phase_or_reference_access():
    events = ints([1, 0, -1])
    broken = ints([[0, 3], [0, 3], [-1, -1]])
    with pytest.raises(ValueError, match="INVALID_SEGMENT_BOUNDARY"):
        op.phase_direction(events, broken)
    broken = ints([[0, 2], [-1, -1], [-1, -1]])
    with pytest.raises(ValueError, match="INVALID_SEGMENT_BOUNDARY"):
        op.drawdown_cycle_reference(ints([1, 1, -1]), np.array([-.2, -.2, np.nan]), broken, .12)


@pytest.mark.parametrize("name", ["segment_starts", "segment_ends"])
@pytest.mark.parametrize("access", ["borrow", "out", "requirements"])
@pytest.mark.parametrize("boundaries", [
    [[0, 3], [0, 3], [-1, -1]],  # Endpoint outside the observation axis.
    [[0, 0], [-1, -1], [-1, -1]],  # Nonpositive segment length.
    [[-1, 2], [-1, -1], [-1, -1]],  # Only one unknown boundary.
    [[0, 2], [-1, -1], [-1, -1]],  # Missing interior membership.
    [[1, 2], [1, 2], [-1, -1]],  # Left boundary differs from its first row.
    [[-2, -2], [-1, -1], [-1, -1]],  # Unsupported negative sentinel.
])
def test_segment_projections_reject_invalid_payload_before_output(name, access, boundaries):
    segments = ints(boundaries)
    segments.flags.writeable = False
    out = np.full(segments.shape[0], 123, dtype=np.int64)
    with pytest.raises(ValueError, match="INVALID_SEGMENT_BOUNDARY"):
        if access == "requirements":
            op.get(name).requirements(segments)
        elif access == "out":
            op.call(name, segments, out=out)
        else:
            op.call(name, segments)
    np.testing.assert_array_equal(out, 123)
    np.testing.assert_array_equal(segments, boundaries)


@pytest.mark.parametrize("name,column", [("segment_starts", 0), ("segment_ends", 1)])
@pytest.mark.parametrize("stride", [1, 2, -2])
@pytest.mark.parametrize("boundaries", [
    [], [[-1, -1]], [[0, 2], [0, 2], [2, 4], [2, 4], [-1, -1]],
])
def test_segment_projections_preserve_valid_strided_readonly_views(name, column, stride, boundaries):
    expected = ints(boundaries).reshape(-1, 2)
    storage = np.empty((expected.shape[0] * abs(stride), 2 * abs(stride)), dtype=np.int64)
    segments = storage[::stride, ::stride]
    segments[:] = expected
    segments.flags.writeable = False
    result, audit = op.call(name, segments, audit=True)
    np.testing.assert_array_equal(result, expected[:, column])
    assert result.dtype == np.int64
    assert audit["input_copy_bytes"] == 0
    if segments.size:
        assert np.shares_memory(result, segments)
    out = np.full(expected.shape[0], 123, dtype=np.int64)
    assert op.call(name, segments, out=out) is out
    np.testing.assert_array_equal(out, expected[:, column])
    np.testing.assert_array_equal(segments, expected)


# Pinned from current platform PS kernels, independent of the installed platform.
_PS_PLATFORM_ORACLE = [
    ([90.69, 98.48, 89.92, 92.56, 122.07, 120.26, 102.95, 83.21, 76.59, 76.68, 63.4, 50.85, 54.56, 72.77, 77.41, 80.76, 72.06, 86.0], [0, 1, -1, 0, 1, 0, 0, 0, -1, 1, 0, -1, 0, 0, 0, 1, -1, 0], 2, 4, 0.1, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, -1, 0]),
    ([85.9, 93.32, 81.36, 74.47, 76.46, 62.96, 60.31, 74.15, 69.86, 68.57, np.nan, 43.15, 52.22, 40.16, 43.87, 48.85, 54.57, 59.92, 53.13, 62.19], [0, 1, 0, -1, 1, 0, -1, 1, 0, 0, -2, 0, 1, -1, 0, 0, 0, 1, -1, 0], 3, 5, 0.2, [0, 1, 0, 0, 0, 0, -1, 1, 0, 0, -2, 0, 1, -1, 0, 0, 0, 0, 0, 0]),
    ([102.58, 110.08, 120.95, 129.9, 120.9, 118.54, 96.52, 89.49, 66.89, 64.9, 74.46, 89.79, 112.45, 130.73, 158.31, 186.96, 232.79, 234.31, 181.34, 167.86, 188.9, 125.07], [0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 1, 0], 4, 6, 0.30000000000000004, [0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ([119.86, 88.45, 103.35, 93.55, 130.45, 119.38, 143.83, 179.27, 207.22, 210.16, 260.23, 293.86, 308.95, 458.21, 455.05, 494.96, 452.39, 494.13, 511.28, 419.52, 434.35, 386.21, 508.47, 513.21], [0, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 1, -1, 1, -1, 0, 0], 5, 7, 0.1, [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0]),
    ([76.2, 64.46, 78.08, 76.12, 100.51, 147.03, 172.23, 158.72, 130.52, 111.21, 91.13, 90.94, 104.74, np.nan, 124.79, 114.19, 88.91, 63.26, 62.91, 84.05, 79.21, 80.42, 85.95, 89.92, 92.74, 107.18], [0, -1, 1, -1, 0, 0, 1, 0, 0, 0, 0, -1, 0, -2, 0, 0, 0, 0, -1, 1, -1, 0, 0, 0, 0, 0], 2, 8, 0.2, [0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, -2, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]),
    ([86.12, 108.5, 104.41, 105.36, 112.04, 129.62, 118.54, 134.97, 143.89, 120.21, 132.59, 234.72, 232.69, 226.45, 236.06, 251.79, 270.57, 249.48, 244.85, 213.14, 273.82, 343.83, 309.53, 305.54, 402.23, 373.18, 385.75, 377.81], [0, 1, -1, 0, 0, 1, -1, 0, 1, -1, 0, 1, 0, -1, 0, 0, 1, 0, 0, -1, 0, 1, 0, -1, 1, -1, 1, 0], 3, 4, 0.30000000000000004, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0]),
    ([107.8, 116.7, 115.58, 119.16, 117.78, 99.87, 80.37, 70.42, 91.38, 73.23, 87.32, 86.9, 103.42, 55.96, 57.92, 55.06, 47.36, 47.85, 37.58, 40.68, 49.79, 43.48, 29.76, 34.22, 37.59, 25.59, 27.2, 28.75, 29.91, 32.59], [0, 1, -1, 1, 0, 0, 0, -1, 1, -1, 1, -1, 1, -1, 1, 0, -1, 1, -1, 0, 1, 0, -1, 0, 1, -1, 0, 0, 0, 0], 4, 5, 0.1, [0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0]),
    ([116.87, 102.43, 118.42, 146.22, 129.72, 175.91, 191.93, 221.11, 228.79, 198.42, 205.28, 206.33, 217.67, 158.5, 121.95, 95.13, np.nan, 117.43, 121.07, 123.43, 136.02, 153.86, 124.63, 149.26, 122.1, 133.58, 118.1, 127.51, 149.77, 173.66, 174.01, 171.33], [0, -1, 0, 1, -1, 0, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, -2, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, 0], 5, 6, 0.2, [0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0]),
]

@pytest.mark.parametrize("prices,events,phase,cycle,amplitude,expected", _PS_PLATFORM_ORACLE)
def test_ps_fixed_platform_oracle_joint_iterations(prices, events, phase, cycle, amplitude, expected):
    np.testing.assert_array_equal(
        op.ps_filter(np.array(prices), ints(events), phase, cycle, amplitude), expected)
