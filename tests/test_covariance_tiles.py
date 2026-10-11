"""Covariance scan boundaries, borrowed layouts and mean-storage lifetime."""

import math
import os
import re

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_covariance_library import MATRIX, PACKING_PLATFORM, data
from .test_covariance_reuse import _check_pair, _count
from .test_typed_results import config, execute

RTOL = ATOL = 1e-12  # Existing covariance contract; no new accuracy threshold.


def _padded_f(n, d):
    # One-double offset and three trailing rows exercise unaligned, padded columns.
    owner = np.full((n + 4, d), -9876.5, order="F")
    x = owner[1:n + 1]
    x[:] = data(n, d, "c")
    return owner, x


def _same_numbers(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)


def _direct_matches_scalar(name, x, workspace):
    try:
        expected = op.call(name, x, simd="scalar")
    except ValueError as error:
        with pytest.raises(ValueError, match=re.escape(str(error))):
            op.call(name, x, workspace=workspace)
        return None
    actual, audit = op.call(name, x, workspace=workspace, audit=True)
    _same_numbers(actual, expected)
    np.testing.assert_array_equal(actual, actual.T)
    assert actual.shape == (x.shape[1], x.shape[1])
    assert not np.shares_memory(actual, x)
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0
    assert op.get(name).requirements(x)["scratch_doubles"] == 0
    if (PACKING_PLATFORM and x.shape[0] >= 32 and x.shape[1] >= 8
            and np.isfinite(x).all() and np.max(np.abs(x)) < 1e100
            and (x.strides[0] == 8 or x.strides[1] == 8)
            and os.environ.get("CME_EXPECT_EIGEN")):
        assert audit["eigen_covariance_calls"] == int(os.environ["CME_EXPECT_EIGEN"])
    return actual


def _fsum_cell(x, j, k):
    """Independent sample covariance oracle for selected finite, ordinary cells."""
    left, right = x[:, j], x[:, k]
    lm = math.fsum(map(float, left)) / len(x)
    rm = math.fsum(map(float, right)) / len(x)
    return math.fsum(
        (float(a) - lm) * (float(b) - rm)
        for a, b in zip(left, right, strict=True)
    ) / (len(x) - 1)


@pytest.mark.parametrize("name", ["covariance", "correlation"])
@pytest.mark.parametrize("n,d", [(31, 8), (32, 7), (32, 8), (34, 10), (35, 11), (36, 12),
                              (33, 15), (127, 9), (128, 17), (129, 65)])
@pytest.mark.parametrize("descending", [False, True])
def test_padded_f_scan_thresholds_and_last_mean_slot(name, n, d, descending):
    owner, x = _padded_f(n, d)
    if descending:
        x = x[::-1]
    before = owner.copy()
    x.flags.writeable = False
    workspace = op.Workspace()
    actual = _direct_matches_scalar(name, x, workspace)
    assert actual is not None and np.isfinite(actual).all()
    if descending:
        scalar = op.call(name, x, simd="scalar")
        np.testing.assert_array_equal(actual.view(np.uint64), scalar.view(np.uint64))
    for j, k in ((0, 0), (0, d - 1), (d // 2, d - 1), (d - 1, d - 1)):
        expected = _fsum_cell(x, j, k)
        if name == "correlation":
            expected /= math.sqrt(_fsum_cell(x, j, j) * _fsum_cell(x, k, k))
        np.testing.assert_allclose(actual[j, k], expected, rtol=RTOL, atol=ATOL)
    assert np.all(np.diag(actual) >= 0)
    if name == "correlation":
        np.testing.assert_allclose(np.diag(actual), 1, rtol=RTOL, atol=ATOL)
        assert np.max(np.abs(actual)) <= 1 + ATOL
    saved = actual.copy()
    later = op.call(name, x * 2, workspace=workspace)
    assert not np.shares_memory(actual, later)
    np.testing.assert_array_equal(actual, saved)
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("name", ["covariance", "correlation"])
@pytest.mark.parametrize("layout", ["negative_columns", "zero_columns", "overlapping_columns"])
def test_f_column_stride_fallback_keeps_original_math(name, layout):
    n, d = 129, 9
    if layout == "negative_columns":
        owner, x = _padded_f(n, d)
        x = x[:, ::-1]
    elif layout == "zero_columns":
        owner = data(n, 1, "f")
        x = np.broadcast_to(owner, (n, d))
    else:
        step = n // 2
        # This explicit buffer is large enough for every overlapping column.
        owner = np.random.default_rng(372).normal(size=n + (d - 1) * step)
        x = np.ndarray((n, d), dtype=np.float64, buffer=owner, strides=(8, step * 8))
    before = owner.copy()
    x.flags.writeable = False
    result = _direct_matches_scalar(name, x, op.Workspace())
    assert result is not None and np.isfinite(result).all()
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("n,d", [(127, 9), (128, 17), (129, 65)])
@pytest.mark.parametrize("layout", ["padded_rows", "negative_rows", "zero_rows"])
def test_c_row_strides_and_tile_tails_keep_scalar_contract(n, d, layout):
    if layout == "zero_rows":
        owner = data(1, d, "c")
        x = np.broadcast_to(owner, (n, d))
    else:
        owner = np.full((n, d + 3), -2468.0)
        x = owner[:, :d]
        x[:] = data(n, d, "c")
        if layout == "negative_rows":
            x = x[::-1]
    before = owner.copy()
    x.flags.writeable = False
    workspace = op.Workspace()
    covariance = _direct_matches_scalar("covariance", x, workspace)
    correlation = _direct_matches_scalar("correlation", x, workspace)
    assert covariance is not None
    if layout == "zero_rows":
        np.testing.assert_array_equal(covariance, np.zeros((d, d)))
        assert correlation is None
    else:
        assert correlation is not None
        assert not np.shares_memory(covariance, correlation)
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("fault", [np.nan, np.inf, 1e200], ids=["nan", "inf", "extreme"])
@pytest.mark.parametrize("descending", [False, True])
def test_padded_f_last_column_fault_preserves_healthy_submatrix(fault, descending):
    owner, x = _padded_f(129, 17)
    if descending:
        x = x[::-1]
    x[91, -1] = fault
    before = owner.copy()
    x.flags.writeable = False
    workspace = op.Workspace()
    for name in ("covariance", "correlation"):
        actual = _direct_matches_scalar(name, x, workspace)
        assert actual is not None
        healthy = op.call(name, x[:, :-1], simd="scalar")
        assert np.isfinite(actual[:-1, :-1]).all()
        np.testing.assert_allclose(actual[:-1, :-1], healthy, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("descending", [False, True])
def test_padded_f_last_constant_column_keeps_covariance_and_isolates_correlation(descending):
    owner, x = _padded_f(129, 17)
    if descending:
        x = x[::-1]
    x[:, -1] = 7.0
    before = owner.copy()
    x.flags.writeable = False
    covariance = _direct_matches_scalar("covariance", x, op.Workspace())
    assert _direct_matches_scalar("correlation", x, op.Workspace()) is None
    np.testing.assert_array_equal(covariance[-1], np.zeros(17))
    graph = GraphCompiler({"x": MATRIX}).compile(
        ["covariance(x)", "correlation(x)", "greater_than(correlation(x),0)", "7"],
        error_policy="isolate",
    )
    result = execute(graph, {"x": x})
    cov, corr, comparison, healthy = result.outputs
    assert cov.root_statuses[0] == 0
    _same_numbers(cov.values[0], covariance)
    assert corr.root_statuses[0] != 0
    assert comparison.root_statuses[0] == corr.root_statuses[0]
    assert healthy.root_statuses[0] == 0 and healthy.values[0] == 7
    assert not np.shares_memory(cov.values[0], corr.values[0])
    assert _count(result, "algorithm_copy_bytes") == 17 * 17 * 8
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("n,d", [(129, 17), (34, 12)])
def test_padded_f_prepared_pair_refreshes_input_without_overwriting_snapshots(n, d):
    owner, x = _padded_f(n, d)
    graph = GraphCompiler({"x": MATRIX}).compile(
        ["covariance(x)", "correlation(x)", "greater_than(correlation(x),0)"]
    )
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(
            graph, {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64)
        )
        first = prepared.run_snapshot()
        saved = [out.values[0].copy() for out in first.outputs]
        _check_pair(first, x)
        x *= np.linspace(1.0, 2.0, x.shape[1])
        x[:, -1] += x[:, 0] * 0.25
        second = prepared.run_snapshot()
        _check_pair(second, x)
        np.testing.assert_array_equal(second.outputs[2].values[0], op.correlation(x) > 0)
    for i, expected in enumerate(saved):
        np.testing.assert_array_equal(first.outputs[i].values[0], expected)
        a, b = first.outputs[i].values[0], second.outputs[i].values[0]
        assert not a.flags.writeable and not b.flags.writeable
        assert not np.shares_memory(a, b)
        assert not np.shares_memory(a, owner) and not np.shares_memory(b, owner)
    assert not np.allclose(first.outputs[0].values[0], second.outputs[0].values[0])
    np.testing.assert_array_equal(owner[0], np.full(d, -9876.5))
    np.testing.assert_array_equal(owner[-3:], np.full((3, d), -9876.5))


@pytest.mark.parametrize("pattern", [
    [1e16, 1.0, -1e16],
    [-0.0, 0.0, -0.0],
    [np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0), np.finfo(float).tiny],
    [1 + 2.0**-27, 1 - 2.0**-27, -1.0],
], ids=["cancellation", "signed_zero", "subnormal", "fma_sensitive"])
def test_f_packet_tails_preserve_discrete_consumers(pattern):
    owner, x = _padded_f(34, 10)
    values = np.resize(np.asarray(pattern), len(x))
    x[:] = np.column_stack([np.roll(values, j) * (-1 if j % 2 else 1)
                            for j in range(x.shape[1])])
    before = owner.copy()
    x.flags.writeable = False
    scalar, audit = op.call("covariance", x, simd="scalar", audit=True)
    assert audit["eigen_covariance_calls"] == 0
    actual = _direct_matches_scalar("covariance", x, op.Workspace())
    for compare in (np.greater, np.less, np.equal):
        np.testing.assert_array_equal(compare(actual, 0), compare(scalar, 0))
    np.testing.assert_array_equal(np.signbit(actual[actual == 0]),
                                  np.signbit(scalar[actual == 0]))
    _direct_matches_scalar("correlation", x, op.Workspace())
    np.testing.assert_array_equal(owner, before)
