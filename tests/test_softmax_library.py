"""Guarded Eigen exponentials preserve softmax rows, errors and result owners.

The scalar oracle specifies this fixture's ascending fold, not a new global
bitwise contract. Tiny probabilities also use relative error with zero atol.
"""

import math
import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matrix_library_backend import PACKING_PLATFORM
from .test_typed_results import config

RTOL, ATOL = 1e-12, 1e-15
LIBRARY_ISA = "eigen-array"
MATRIX = {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]}


def reference(x, axis):
    result = np.empty(x.shape, dtype=np.float64)
    if not x.size:
        return result
    rows = np.moveaxis(x, axis, -1)
    targets = np.moveaxis(result, axis, -1)
    for index in np.ndindex(rows.shape[:-1]):
        row = [float(value) for value in rows[index]]
        maximum = max(row)
        exponentials = [math.exp(value - maximum) for value in row]
        total = 0.0
        for value in exponentials:
            total += value
        targets[index] = [value / total for value in exponentials]
    return result


def assert_probabilities(actual, expected, axis):
    assert actual.shape == expected.shape and actual.dtype == np.float64
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    tiny = (expected > 0) & (expected < ATOL)
    np.testing.assert_allclose(actual[tiny], expected[tiny], rtol=RTOL, atol=0)
    np.testing.assert_array_equal(actual == 0, expected == 0)
    assert not np.signbit(actual).any()
    if actual.size:
        np.testing.assert_allclose(actual.sum(axis=axis), 1, rtol=RTOL, atol=ATOL)


def assert_dispatch(audit, eligible):
    enabled = os.environ.get("CME_EXPECT_EIGEN")
    if not eligible or not PACKING_PLATFORM or enabled == "0":
        assert audit["isa"] != LIBRARY_ISA
    elif enabled == "1":
        assert audit["isa"] == LIBRARY_ISA
    else:
        assert audit["isa"] in {"scalar", LIBRARY_ISA}


def assert_no_scratch(audit):
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0


@pytest.mark.parametrize("shape,axis,eligible", [
    ((255,), 0, False), ((256,), 0, True), ((257,), 0, True),
    ((37, 7), 1, False), ((31, 8), 1, False), ((32, 8), 1, True),
    ((33, 9), 1, True), ((256, 1), 1, False),
    ((4, 8, 8), 2, True), ((32, 8, 1), 1, True),
    ((2, 16, 8), 1, False), ((32, 1, 8), 1, False),
])
def test_size_axis_thresholds_and_packet_tails(shape, axis, eligible):
    x = np.random.default_rng(502).normal(size=shape)
    x.flags.writeable = False
    before = x.copy()
    actual, audit = op.softmax_axis(x, axis, audit=True)
    assert_probabilities(actual, reference(x, axis), axis)
    assert_probabilities(actual, op.softmax_axis(x, axis, simd="scalar"), axis)
    np.testing.assert_array_equal(np.argmax(actual, axis=axis), np.argmax(x, axis=axis))
    assert_dispatch(audit, eligible)
    assert_no_scratch(audit)
    assert op.get("softmax_axis").requirements(x, axis)["scratch_doubles"] == 0
    np.testing.assert_array_equal(x, before)
    assert not np.shares_memory(actual, x)


@pytest.mark.parametrize("layout,axis,eligible", [
    ("padded", 1, True), ("negative_rows", 1, True), ("zero_rows", 1, True),
    ("overlapping_rows", 1, True), ("negative_axis", 1, False),
    ("sliced_axis", 1, False), ("zero_axis", 1, False),
    ("f", 1, False), ("f", 0, False),
])
def test_borrowed_strides_padding_and_output_canaries(layout, axis, eligible):
    base = np.random.default_rng(503).normal(size=(32, 9))
    owner = np.full((32, 12), np.nan)
    x = owner[:, 1:10]  # Misaligned and padded, with poisoned unused columns.
    x[:] = base
    if layout == "negative_rows":
        x = x[::-1]
    elif layout == "zero_rows":
        x = np.broadcast_to(x[:1], x.shape)
    elif layout == "overlapping_rows":
        owner = np.linspace(-2, 2, 31 * 5 + 9)
        x = np.ndarray((32, 9), dtype=np.float64, buffer=owner, strides=(40, 8))
    elif layout == "negative_axis":
        x = x[:, ::-1]
    elif layout == "sliced_axis":
        owner = np.repeat(base, 2, axis=1)
        x = owner[:, ::2]
    elif layout == "zero_axis":
        x = np.broadcast_to(x[:, :1], x.shape)
    elif layout == "f":
        owner = np.asfortranarray(base)
        x = owner
    before = owner.copy()
    x.flags.writeable = False
    storage = np.full(x.size + 2, np.nan)
    storage[[0, -1]] = [1729.0, -1729.0]
    out = storage[1:-1].reshape(x.shape)
    actual, audit = op.softmax_axis(x, axis, out=out, audit=True)
    assert actual is out
    assert_probabilities(actual, reference(x, axis), axis)
    assert_dispatch(audit, eligible)
    assert_no_scratch(audit)
    np.testing.assert_array_equal(storage[[0, -1]], [1729.0, -1729.0])
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("side,eligible", [("below", False), ("equal", True), ("above", True)])
def test_guard_neighbors_keep_tiny_probabilities(side, eligible):
    threshold = math.log(9.0) - 700.0
    minimum = {"below": np.nextafter(threshold, -np.inf), "equal": threshold,
               "above": np.nextafter(threshold, np.inf)}[side]
    x = np.zeros((32, 9))
    x[:, -1] = minimum
    actual, audit = op.softmax_axis(x, 1, audit=True)
    assert_probabilities(actual, reference(x, 1), 1)
    assert_dispatch(audit, eligible)
    assert np.all(actual[:, -1] > np.finfo(float).tiny)


@pytest.mark.parametrize("shift", [
    -708.0, np.nextafter(-709.784, np.inf), -709.784,
    np.nextafter(-709.784, -np.inf), -710.0, -744.0, -745.0, -746.0, -1000.0,
])
def test_clamp_underflow_and_zero_boundary_use_original_math(shift):
    x = np.full((32, 9), -1000.0)
    x[:, 4] = 0.0  # Sum rounds to one, so exp(-745) remains a positive subnormal.
    # The same rare probability appears in a packet lane and the odd tail.
    x[:, [0, 8]] = shift
    actual, audit = op.softmax_axis(x, 1, audit=True)
    assert_probabilities(actual, reference(x, 1), 1)
    np.testing.assert_array_equal(actual[:, 0], actual[:, 8])
    assert_dispatch(audit, False)


def test_normal_exponential_can_become_subnormal_after_normalization():
    x = np.zeros((1, 8192))
    x[0, -1] = -700.0
    expected = reference(x, 1)
    assert 0 < expected[0, -1] < np.finfo(float).tiny
    actual, audit = op.softmax_axis(x, 1, audit=True)
    assert_probabilities(actual, expected, 1)
    assert_dispatch(audit, False)


@pytest.mark.parametrize("value", [0.0, -0.0, np.finfo(float).max, -np.finfo(float).max])
def test_constant_rows_preserve_exact_uniform_result(value):
    x = np.full((32, 9), value)
    actual, audit = op.softmax_axis(x, 1, audit=True)
    np.testing.assert_array_equal(actual, np.full_like(x, 1.0 / 9.0))
    np.testing.assert_array_equal(op.argmax_axis(actual, 1), np.zeros(32, np.int64))
    assert not np.signbit(actual).any()
    assert_dispatch(audit, True)


def test_mixed_guard_rows_signed_zero_extremes_and_equal_maxima():
    x = np.zeros((32, 9))
    x[0, ::2] = -0.0
    x[1] = -np.finfo(float).max
    x[1, [1, 5]] = np.finfo(float).max
    x[2] = [-1000, 0, -744, 0, -745, -746, -709.784, -700, -710]
    x[3] = np.linspace(-3, 3, 9)
    actual, audit = op.softmax_axis(x, 1, audit=True)
    assert_probabilities(actual, reference(x, 1), 1)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), np.argmax(x, axis=1))
    np.testing.assert_array_equal(actual[1, [1, 5]], [0.5, 0.5])
    np.testing.assert_array_equal(actual[1, [0, 2, 3, 4, 6, 7, 8]], 0)
    assert_dispatch(audit, True)  # Other ordinary rows still use the library.


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", [0, 8])
def test_bad_row_is_rejected_before_its_first_output_write(bad, column):
    x = np.tile(np.linspace(-2, 2, 9), (32, 1))
    x[1, column] = bad
    for isa in ("scalar", "auto"):
        out = np.full_like(x, -1234.5)
        with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
            op.softmax_axis(x, 1, out=out, simd=isa)
        assert_probabilities(out[:1], reference(x[:1], 1), 1)
        np.testing.assert_array_equal(out[1:], -1234.5)


@pytest.mark.parametrize("shape,axis", [((0,), 0), ((0, 9), 1), ((32, 0), 1),
                                       ((2, 0, 9), 0), ((2, 0, 9), 1)])
def test_empty_softmax_returns_original_empty_shape(shape, axis):
    x = np.empty(shape)
    actual, audit = op.softmax_axis(x, axis, audit=True)
    assert actual.shape == shape and actual.dtype == np.float64
    assert_no_scratch(audit)
    assert_dispatch(audit, False)


def test_scalar_and_unavailable_isa_remain_explicit():
    x = np.tile(np.linspace(-3, 3, 9), (32, 1))
    scalar, audit = op.softmax_axis(x, 1, simd="scalar", audit=True)
    assert audit["isa"] == "scalar"
    assert_probabilities(scalar, reference(x, 1), 1)
    for isa in ["auto", *op.available_simd()]:
        actual, audit = op.softmax_axis(x, 1, simd=isa, audit=True)
        assert_probabilities(actual, scalar, 1)
        assert_dispatch(audit, isa != "scalar")
    for isa in {"sse2", "avx2", "neon"} - set(op.available_simd()):
        with pytest.raises(ValueError, match="UNSUPPORTED_ISA"):
            op.softmax_axis(x, 1, simd=isa)


@pytest.mark.parametrize("isa", ["scalar", "auto"])
def test_reject_output_alias_and_invalid_axis_without_mutation(isa):
    x = np.tile(np.linspace(-3, 3, 9), (32, 1))
    before = x.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIASES_INPUT"):
        op.softmax_axis(x, 1, out=x, simd=isa)
    np.testing.assert_array_equal(x, before)
    out = np.full_like(x, -1234.5)
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        op.softmax_axis(x, 2, out=out, simd=isa)
    np.testing.assert_array_equal(out, -1234.5)


def test_workspace_reuse_does_not_overwrite_independent_results():
    x = np.tile(np.linspace(-3, 3, 9), (32, 1))
    workspace = op.Workspace()
    first, audit = op.softmax_axis(x, 1, workspace=workspace, audit=True)
    saved = first.copy()
    x[:, 0] = 5.0
    second, later_audit = op.softmax_axis(x, 1, workspace=workspace, audit=True)
    assert_probabilities(second, reference(x, 1), 1)
    np.testing.assert_array_equal(first, saved)
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, x) and not np.shares_memory(second, x)
    assert_no_scratch(audit)
    assert_no_scratch(later_audit)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_graph_cse_boolean_roots_worker_transport_and_isolation(lane):
    x = np.tile(np.linspace(-3, 3, 9), (32, 1))
    x[1] = [-1000, 0, -744, 0, -745, -746, -709.784, -700, -710]
    z = "softmax_axis(x,1)"
    graph = GraphCompiler({"x": MATRIX}).compile(
        [z, f"equal({z},0)", f"argmax_axis({z},1)", z, "7"],
        result_format="typed", error_policy="isolate",
    )
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    bounds = np.zeros(2, np.int64)
    expected = reference(x, 1)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(graph, {"x": x}, bounds, bounds)
        for interval in range(2):
            for root in (0, 3):
                assert first.outputs[root].root_statuses[interval] == 0
                value = first.outputs[root].values[interval]
                assert_probabilities(value, expected, 1)
                assert not np.shares_memory(value, x)
            np.testing.assert_array_equal(first.outputs[1].values[interval], expected == 0)
            np.testing.assert_array_equal(first.outputs[2].values[interval],
                                          np.argmax(expected, axis=1))
            assert first.outputs[4].values[interval] == 7
        saved = first.outputs[0].values[0].copy()
        assert not np.shares_memory(first.outputs[0].values[0], first.outputs[3].values[0])
        x[-1, -1] = np.nan
        failed = scheduler.execute(graph, {"x": x}, bounds, bounds)
        for root in range(4):
            assert np.all(failed.outputs[root].root_statuses != 0)
            np.testing.assert_array_equal(failed.outputs[root].root_statuses,
                                          failed.outputs[0].root_statuses)
        np.testing.assert_array_equal(failed.outputs[4].root_statuses, [0, 0])
        np.testing.assert_array_equal(failed.outputs[4].values, [7, 7])
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)


def test_prepared_snapshots_reread_mutated_input_and_keep_previous_owner():
    x = np.tile(np.linspace(-2, 2, 9), (32, 1))
    graph = GraphCompiler({"x": MATRIX}).compile(
        ["softmax_axis(x,1)", "equal(softmax_axis(x,1),0)"], result_format="typed"
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": x}, bounds, bounds)
        first = prepared.run_snapshot()
        saved = [item.values[0].copy() for item in first.outputs]
        x[:, 0] = -1000.0
        second = prepared.run_snapshot()
        expected = reference(x, 1)
        assert_probabilities(second.outputs[0].values[0], expected, 1)
        np.testing.assert_array_equal(second.outputs[1].values[0], expected == 0)
        rebound = scheduler.prepare_execution(graph, {"x": x.copy()}, bounds, bounds)
        third = rebound.run_snapshot()
        assert_probabilities(third.outputs[0].values[0], expected, 1)
    for root in range(2):
        np.testing.assert_array_equal(first.outputs[root].values[0], saved[root])
        for result in (first, second, third):
            value = result.outputs[root].values[0]
            assert not value.flags.writeable and not np.shares_memory(value, x)
        assert not np.shares_memory(first.outputs[root].values[0], second.outputs[root].values[0])
        assert not np.shares_memory(second.outputs[root].values[0], third.outputs[root].values[0])
