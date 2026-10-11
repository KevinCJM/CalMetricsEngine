"""Ordered column variance: Eigen lanes, failure isolation and owned results."""

import os
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_typed_results import config


@pytest.mark.parametrize(
    "shape", [(16, 8), (31, 64), (32, 7), (32, 8), (65, 9), (65, 64), (257, 129)]
)
@pytest.mark.parametrize("layout", ["c", "f", "reverse", "sliced", "broadcast"])
@pytest.mark.parametrize("ddof", [0, 1])
def test_column_variance_preserves_order_workspace_and_ownership(shape, layout, ddof):
    x = np.random.default_rng(813).normal(size=shape)
    if layout == "f":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1]
    elif layout == "sliced":
        x = np.repeat(x, 2, axis=1)[:, ::2]
    elif layout == "broadcast":
        x = np.broadcast_to(x[:1], shape)
    x.flags.writeable = False
    saved = x.copy()
    expected, scalar = op.call("variance_axis", x, 0, ddof, simd="scalar", audit=True)
    workspace = op.Workspace()
    actual, audit = op.call("variance_axis", x, 0, ddof, workspace=workspace, audit=True)
    np.testing.assert_array_equal(actual.view("u8"), expected.view("u8"))
    np.testing.assert_allclose(actual, np.var(x, axis=0, ddof=ddof), rtol=1e-12, atol=1e-14)
    assert scalar["isa"] == "scalar"
    # Contiguous row packets are portable; strided column packets are Apple-only.
    row_eligible = x.flags.c_contiguous and shape[0] >= 32 and 8 <= shape[1] <= 256
    column_eligible = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and int(platform.mac_ver()[0].split(".")[0]) >= 15
        and shape[0] >= 32
        and shape[1] >= 8
        and abs(x.strides[0]) == 8
        and x.strides[1] >= shape[0] * 8
    )
    eligible = row_eligible or column_eligible
    enabled = os.environ.get("CME_EXPECT_EIGEN", os.environ.get("CALMETRICS_TEST_EXPECT_EIGEN"))
    if not eligible:
        assert audit["isa"] == "scalar"
    elif enabled is not None:
        assert audit["isa"] == ("eigen-array" if enabled == "1" else "scalar")
    for key in ("workspace_bytes", "algorithm_copy_bytes", "input_copy_bytes"):
        assert audit[key] == scalar[key] == 0
    frozen = actual.copy()
    later = op.call("variance_axis", x, 0, ddof, workspace=workspace)
    assert not np.shares_memory(actual, later)
    assert not np.shares_memory(actual, x)
    np.testing.assert_array_equal(actual.view("u8"), frozen.view("u8"))
    np.testing.assert_array_equal(x, saved)


@pytest.mark.parametrize("column", [0, 7, 8, 15, 16])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e308])
def test_column_variance_nonfinite_and_overflow_preserve_status(column, value):
    x = np.ones((33, 17))
    x[-1, column] = value
    for simd in ("auto", "scalar"):
        with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
            op.call("variance_axis", x, 0, simd=simd)


@pytest.mark.parametrize("value", [0.0, -0.0, 1e140, 1e-150])
def test_column_variance_extreme_scales(value):
    x = np.full((65, 17), value)
    x[::2] *= -1
    auto = op.call("variance_axis", x, 0)
    scalar = op.call("variance_axis", x, 0, simd="scalar")
    np.testing.assert_array_equal(auto.view("u8"), scalar.view("u8"))


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_variance_graph_transport_and_isolation(lane):
    x = np.random.default_rng(312).normal(size=(65, 17))
    graph = GraphCompiler(
        {"x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"])}
    ).compile(
        ["variance_axis(x,0)", "sqrt(variance_axis(x,0))", "7"],
        error_policy="isolate",
        result_format="typed",
    )
    bounds = np.zeros(2, np.int64)
    expected = op.call("variance_axis", x, 0, simd="scalar")
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        previous = scheduler.execute(graph, {"x": x}, bounds, bounds)
        saved = previous.outputs[0].values[0].copy()
        for index in range(2):
            np.testing.assert_array_equal(
                previous.outputs[0].values[index].view("u8"), expected.view("u8")
            )
        x[-1, -1] = np.nan
        failed = scheduler.execute(graph, {"x": x}, bounds, bounds)
        for index in (0, 1):
            assert np.all(failed.outputs[index].root_statuses != 0)
        assert np.all(failed.outputs[2].root_statuses == 0)
        np.testing.assert_array_equal(failed.outputs[2].values, [7.0, 7.0])
    np.testing.assert_array_equal(previous.outputs[0].values[0], saved)


def test_prepared_variance_rereads_data_and_preserves_prior_output():
    x = np.random.default_rng(32).normal(size=(65, 64))
    graph = GraphCompiler(
        {"x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"])}
    ).compile(["variance_axis(x,0)"], result_format="typed")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": x}, bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        x[:] *= 2
        second = prepared.run_snapshot()
        np.testing.assert_array_equal(
            second.outputs[0].values[0], op.call("variance_axis", x, 0, simd="scalar")
        )
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
