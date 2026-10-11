"""Covariance backend: alias-safe output reuse, mathematical and graph contracts."""

import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matrix_library_backend import PACKING_PLATFORM
from .test_typed_results import config

MATRIX = {"kind": "matrix", "axes": ["time", "asset"], "shape": ["N", "D"]}


def run(graph, inputs, lane):
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        return scheduler.execute(
            graph, inputs, np.zeros(2, np.int64), np.full(2, len(inputs["x"]), np.int64)
        )


def data(n, d, layout):
    x = np.random.default_rng(716).normal(size=(n, d))
    if layout in {"f", "reverse_f"}:
        x = np.asfortranarray(x)
        if layout == "reverse_f":
            x = x[::-1]
    elif layout == "reverse_rows":
        x = x[::-1]
    elif layout == "reverse_columns":
        x = x[:, ::-1]
    elif layout == "slice":
        x = np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)[::2, ::2]
    elif layout == "broadcast":
        x = np.broadcast_to(x[:, :1], (n, d))
    return x


@pytest.mark.parametrize("name", ["covariance", "correlation"])
@pytest.mark.parametrize(
    "n,d", [(2, 0), (2, 1), (31, 8), (32, 8), (64, 1), (64, 7), (65, 9), (128, 64)]
)
@pytest.mark.parametrize(
    "layout", ["c", "f", "reverse_rows", "reverse_columns", "slice", "broadcast", "reverse_f"]
)
def test_covariance_layout_and_independent_ownership(name, n, d, layout):
    x = data(n, d, layout)
    before = x.copy()
    x.flags.writeable = False
    workspace = op.Workspace()
    scalar = op.call(name, x, simd="scalar")
    actual, audit = op.call(name, x, workspace=workspace, audit=True)
    np.testing.assert_allclose(actual, scalar, atol=1e-12, rtol=1e-12)
    if d:
        reference = (
            np.cov(x, rowvar=False) if name == "covariance" else np.corrcoef(x, rowvar=False)
        )
        np.testing.assert_allclose(actual, np.atleast_2d(reference), atol=1e-12, rtol=1e-12)
    assert actual.shape == (d, d)
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0
    assert op.get(name).requirements(x)["scratch_doubles"] == 0
    eligible = (
        PACKING_PLATFORM
        and n >= 32
        and d > 0
        and ((x.strides[1] == 8 and d >= 8) or x.strides[0] == 8
             or (x.strides[0] == -8 and x.strides[1] >= n*8 and d >= 8))
    )
    if layout == "reverse_f":
        np.testing.assert_array_equal(actual.view(np.uint64), scalar.view(np.uint64))
    if not eligible:
        assert audit["eigen_covariance_calls"] == 0
    elif os.environ.get("CME_EXPECT_EIGEN"):
        assert audit["eigen_covariance_calls"] == int(os.environ["CME_EXPECT_EIGEN"])
    np.testing.assert_array_equal(actual, actual.T)
    assert not np.shares_memory(x, actual)
    frozen = actual.copy()
    op.call(name, x * 2, workspace=workspace)
    np.testing.assert_array_equal(actual, frozen)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("name", ["covariance", "correlation"])
@pytest.mark.parametrize("layout", ["c", "f", "reverse_f"])
@pytest.mark.parametrize(
    "fault", ["nan", "inf", "zero", "constant", "huge", "tiny", "near_constant"]
)
def test_covariance_extremes_keep_scalar_contract(name, layout, fault):
    x = data(65, 9, layout)
    if fault == "nan":
        x[21, 2] = np.nan
    elif fault == "inf":
        x[21, 2] = np.inf
    elif fault == "zero":
        x[:, -1] = 0
    elif fault == "constant":
        x[:, 0] = 7
    elif fault == "huge":
        x *= 1e200
    elif fault == "tiny":
        x *= 1e-162
    elif fault == "near_constant":
        x[:] = x * 1e-14 + 7
    try:
        expected = op.call(name, x, simd="scalar")
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error)):
            op.call(name, x)
    else:
        actual = op.call(name, x)
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        if layout == "reverse_f":
            finite = np.isfinite(expected)
            np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))


@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize("d", [0, 8])
def test_insufficient_sample_even_with_no_columns(n, d):
    for name in ("covariance", "correlation"):
        with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
            op.call(name, np.empty((n, d)))


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
@pytest.mark.parametrize("layout", ["c", "reverse_f"])
def test_covariance_cse_transport_and_failure_isolation(lane, layout):
    x = data(65, 9, layout)
    expected, audit = op.covariance(x, audit=True)
    graph = GraphCompiler({"x": MATRIX}).compile(
        ["covariance(x)", "trace(covariance(x))", "correlation(x)"]
    )
    result = run(graph, {"x": x}, lane)
    assert (
        sum(c["eigen_covariance_calls"] for c in result.audit["native_chunks"])
        == 2 * audit["eigen_covariance_calls"]
    )
    for i in range(2):
        np.testing.assert_allclose(result.outputs[0].values[i], expected, atol=1e-12)
        np.testing.assert_allclose(result.outputs[1].values[i], np.trace(expected), atol=1e-12)
    x[:, 0] = 7
    isolated = GraphCompiler({"x": MATRIX}).compile(
        ["correlation(x)", "covariance(x)", "7"], error_policy="isolate"
    )
    bad = run(isolated, {"x": x}, lane)
    assert np.all(bad.outputs[0].root_statuses != 0)
    assert np.all(bad.outputs[1].root_statuses == 0)
    assert np.all(np.asarray(bad.outputs[2].values) == 7)


@pytest.mark.parametrize("layout", ["c", "reverse_f"])
def test_covariance_prepared_rereads_input_and_keeps_snapshot(layout):
    x = data(64, 8, layout)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(
            graph, {"x": x}, bounds, np.array([len(x)], np.int64)
        )
        first = prepared.run_snapshot()
        frozen = first.outputs[0].values[0].copy()
        x *= 2
        second = prepared.run_snapshot()
        np.testing.assert_allclose(second.outputs[0].values[0], frozen * 4, atol=1e-12)
        np.testing.assert_array_equal(first.outputs[0].values[0], frozen)


@pytest.mark.parametrize("layout", ["c", "reverse_f"])
def test_covariance_nested_iteration_and_memory_admission(layout):
    x = data(64, 8, layout)
    _, direct = op.correlation(x, audit=True)
    seed = np.zeros((8, 8))
    graph = GraphCompiler(
        {
            "x": MATRIX,
            "seed": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["D", "D"]},
        }
    ).compile("iterate(correlation(x)+iterate_x*0,seed,1e-12,2)")
    bounds = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x, "seed": seed}, *bounds)
        calls = sum(c["eigen_covariance_calls"] for c in result.audit["native_chunks"])
        assert calls == 2 * direct["eigen_covariance_calls"]
        np.testing.assert_allclose(result.outputs[0].values[0], op.correlation(x), atol=1e-12)
        plan = scheduler.plan(graph, {"x": x, "seed": seed}, *bounds)
        with pytest.raises(MemoryError):
            scheduler.plan(
                graph,
                {"x": x, "seed": seed},
                *bounds,
                memory_budget_bytes=plan.estimated_total_memory_bytes - 1,
            )
