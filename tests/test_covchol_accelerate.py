"""Guarded large covariance/Cholesky graph path and original-path replay."""

import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_covariance_cholesky_reuse import DECL, EXPR, data
from .test_typed_results import config, execute

ACCELERATE = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and int(platform.mac_ver()[0].split(".")[0]) >= 15
)


@pytest.mark.skipif(not ACCELERATE, reason="macOS 15 arm64 Accelerate path")
@pytest.mark.parametrize("order", ["c", "f"])
@pytest.mark.parametrize("entry", ["ordinary", "prepared", "process"])
@pytest.mark.parametrize("output_order", ["C", "F"])
def test_large_independent_root_matches_direct_operator(order, entry, output_order):
    x = data(2065, 2048, order)
    x.flags.writeable = False
    expected = op.cholesky(op.covariance(x))
    graph = GraphCompiler(DECL).compile(EXPR, output_order=output_order)
    if entry == "process":
        result = execute(
            graph, {"x": x}, starts=[0, 0], ends=[len(x), len(x)], lane="process_inline"
        )
        assert result.plan.lane == "process"
    elif entry == "prepared":
        with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
            prepared = scheduler.prepare_execution(
                graph, {"x": x}, np.array([0], np.int64), np.array([len(x)], np.int64)
            )
            result = prepared.run_snapshot()
            previous = result.outputs[0].values[0].copy()
            again = prepared.run_snapshot()
            np.testing.assert_array_equal(result.outputs[0].values[0], previous)
            assert not np.shares_memory(result.outputs[0].values[0], again.outputs[0].values[0])
    else:
        result = execute(graph, {"x": x})
    actual = result.outputs[0].values[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert not np.shares_memory(actual, x)
    assert not actual.flags.writeable
    assert actual.flags.c_contiguous if output_order == "C" else actual.flags.f_contiguous
    assert sum(chunk["accelerate_covchol_calls"] for chunk in result.audit["native_chunks"]) >= 1


@pytest.mark.skipif(not ACCELERATE, reason="macOS 15 arm64 Accelerate path")
@pytest.mark.parametrize("fault", ["duplicate", "nan", "extreme", "near_zero"])
def test_large_guard_replays_existing_chain(fault):
    x = data(2065, 2048)
    if fault == "duplicate":
        x[:, -1] = x[:, -2]
    elif fault == "nan":
        x[2, 1] = np.nan
    elif fault == "extreme":
        x *= 1e8
    else:
        x *= 1e-10
    graph = GraphCompiler(DECL).compile(EXPR)
    try:
        expected = op.cholesky(op.covariance(x))
    except (ValueError, RuntimeError) as error:
        with pytest.raises((ValueError, RuntimeError)) as actual:
            execute(graph, {"x": x})
        assert str(actual.value) == str(error)
    else:
        result = execute(graph, {"x": x})
        np.testing.assert_array_equal(
            result.outputs[0].values[0].view(np.uint64), expected.view(np.uint64)
        )
        assert result.audit["native_chunks"][0]["accelerate_covchol_calls"] == 0


@pytest.mark.skipif(not ACCELERATE, reason="macOS 15 arm64 Accelerate path")
def test_large_shared_covariance_keeps_original_path():
    x = data(2065, 2048)
    graph = GraphCompiler(DECL).compile([EXPR, "trace(covariance(x))"])
    result = execute(graph, {"x": x})
    expected = op.cholesky(op.covariance(x))
    np.testing.assert_array_equal(
        result.outputs[0].values[0].view(np.uint64), expected.view(np.uint64)
    )
    assert result.audit["native_chunks"][0]["accelerate_covchol_calls"] == 0


@pytest.mark.skipif(not ACCELERATE, reason="macOS 15 arm64 Accelerate path")
def test_near_zero_covariance_replays_ordered_path():
    x = data(2065, 2048)
    x[:, :2] -= x[:, :2].mean(axis=0)
    first = x[:, 0]
    x[:, 1] -= (first @ x[:, 1]) / (first @ first) * first
    expected = op.cholesky(op.covariance(x))
    graph = GraphCompiler(DECL).compile(EXPR)
    result = execute(graph, {"x": x})
    np.testing.assert_array_equal(
        result.outputs[0].values[0].view(np.uint64), expected.view(np.uint64)
    )
    assert result.audit["native_chunks"][0]["accelerate_covchol_calls"] == 0
