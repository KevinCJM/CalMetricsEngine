"""Compatibility matvec: ordered IEEE results, unchanged inputs and ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matrix_vector_library import (
    assert_ieee_equal,
    assert_library_calls,
    assert_no_workspace,
)
from .test_typed_results import config


def make_matrix(rows, columns, layout):
    x = np.random.default_rng(764).normal(size=(rows, columns))
    if layout == "F":
        return np.asfortranarray(x)
    if layout == "pitch_C":
        padded = np.full((rows, columns + 3), np.nan)
        padded[:, :columns] = x
        return padded[:, :columns]
    if layout == "pitch_F":
        padded = np.full((rows + 3, columns), np.nan, order="F")
        padded[:rows] = x
        return padded[:rows]
    if layout == "reverse":
        return np.asfortranarray(x)[::-1, ::-1]
    return x


@pytest.mark.parametrize("shape", [(127, 9), (128, 7), (128, 9), (4097, 9)])
@pytest.mark.parametrize("layout", ["C", "F", "pitch_C", "pitch_F", "reverse"])
@pytest.mark.parametrize("stride", [1, -1, 0, 2])
def test_portfolio_ordered_layout_threshold_and_independent_results(shape, layout, stride):
    a = make_matrix(*shape, layout)
    weights = np.random.default_rng(7641).normal(size=shape[1])
    b = (
        weights[::-1]
        if stride == -1
        else np.broadcast_to(weights[:1], weights.shape)
        if stride == 0
        else np.repeat(weights, 2)[::2]
        if stride == 2
        else weights
    )
    a.flags.writeable = b.flags.writeable = False
    before_a, before_b = a.copy(), b.copy()
    expected = op.portfolio_returns(a, b, simd="scalar")
    work = op.Workspace()
    actual, audit = op.portfolio_returns(a, b, workspace=work, audit=True)
    assert_ieee_equal(actual, expected)
    assert_ieee_equal(actual, op.matvec(a, b))
    eligible = shape[0] >= 128 and shape[1] >= 8 and layout != "reverse"
    assert_library_calls(audit, eligible)
    assert_no_workspace(audit)
    assert audit["algorithm_copy_bytes"] == 0
    assert op.get("portfolio_returns").requirements(a, b)["scratch_doubles"] == 0
    assert not np.shares_memory(actual, a) and not np.shares_memory(actual, b)
    saved = actual.copy()
    newer = op.portfolio_returns(a, b * 2, workspace=work)
    assert not np.shares_memory(actual, newer)
    assert_ieee_equal(actual, saved)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("fault", ["nan_zero", "inf_zero", "overflow", "cancel", "tiny", "fma"])
def test_portfolio_extremes_and_separate_multiply_add(order, fault):
    a = np.zeros((4097, 8), order=order)
    b = np.ones(8)
    a[:, 0] = -1
    if fault in ("nan_zero", "inf_zero"):
        a[9, 4] = np.nan if fault == "nan_zero" else np.inf
        b[4] = 0
    elif fault == "overflow":
        a[9, :3] = [1e308, 1e308, -1e308]
    elif fault == "cancel":
        a[9, :3] = [1e200, 1, -1e200]
    elif fault == "tiny":
        a *= 1e-300
        b *= 1e-20
    else:
        a[:, 1] = 1 + 2.0**-27
        b[1] = 1 - 2.0**-27
    expected = op.portfolio_returns(a, b, simd="scalar")
    actual, audit = op.portfolio_returns(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, True)
    assert_no_workspace(audit)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("alias", ["matrix", "vector"])
@pytest.mark.parametrize("simd", ["scalar", "auto"])
def test_portfolio_explicit_output_alias_rejected_before_writes(order, alias, simd):
    a = np.array(np.eye(128), order=order)
    b = np.ones(128)
    out = b if alias == "vector" else (a[0] if order == "C" else a[:, 0])
    before_a, before_b = a.copy(), b.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIAS"):
        op.portfolio_returns(a, b, out=out, simd=simd)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


def scalar_rows(result, result_format):
    """Capture the original storage before another execution or owner close."""
    if result_format == "auto":
        assert result.statuses is None
        values = result.values
        return values, (values,)
    outputs = result.outputs
    arrays = []
    for output in outputs:
        assert output.kind == "scalar" and output.dtype == "float64"
        assert all(output.shape_known)
        np.testing.assert_array_equal(output.root_statuses, 0)
        assert not output.root_statuses.flags.writeable
        assert all(status is None for status in output.statuses)
        for value in output.values:
            assert value.shape == () and not value.flags.writeable
            arrays.append(value)
    values = np.array(
        [[output.values[row].item() for output in outputs] for row in range(len(outputs[0].values))]
    )
    return values, tuple(arrays)


def assert_generations(first_arrays, saved, second_arrays):
    for actual, snapshot in zip(first_arrays, saved, strict=True):
        np.testing.assert_array_equal(actual, snapshot)
        assert not any(np.shares_memory(actual, newer) for newer in second_arrays)


@pytest.mark.parametrize("result_format", ["auto", "typed"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_portfolio_complete_multi_root_risk_graph_owns_generations(order, result_format):
    rows, columns, products = 4097, 9, 3
    declarations = {"r": {"kind": "matrix", "axes": ["time", "asset"], "shape": ["T", "N"]}}
    declarations.update(
        {f"w{i}": {"kind": "vector", "axes": ["asset"], "shape": ["N"]} for i in range(products)}
    )
    graph = GraphCompiler(declarations).compile(
        [f"std(portfolio_returns(r,w{i}),1)*sqrt(252)" for i in range(products)],
        result_format=result_format,
    )
    a = make_matrix(rows, columns, order)
    w = np.random.default_rng(7642).normal(size=(products, columns))
    a.flags.writeable = w.flags.writeable = False
    inputs = {"r": a, **{f"w{i}": w[i] for i in range(products)}}
    expected = np.array(
        [
            [
                op.std(op.portfolio_returns(a, w[i], simd="scalar"), 1) * np.sqrt(252)
                for i in range(products)
            ]
        ]
    )
    before = a.copy()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(
            graph, inputs, np.array([0], np.int64), np.array([rows], np.int64)
        )
        first_values, first_arrays = scalar_rows(first, result_format)
        saved = tuple(value.copy() for value in first_arrays)
        second = scheduler.execute(
            graph, inputs, np.array([0], np.int64), np.array([rows], np.int64)
        )
        _, second_arrays = scalar_rows(second, result_format)
    np.testing.assert_allclose(first_values, expected, rtol=1e-12, atol=1e-12)
    assert_generations(first_arrays, saved, second_arrays)
    calls = sum(chunk["eigen_matvec_calls"] for chunk in first.audit["native_chunks"])
    assert_library_calls({"eigen_matvec_calls": calls // products}, True)
    assert calls % products == 0
    assert first.audit["python_worker_callbacks"] == 0
    np.testing.assert_array_equal(a, before)


@pytest.mark.parametrize("result_format", ["auto", "typed"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_portfolio_windowed_risk_graph_transport_and_ownership(lane, order, result_format):
    rows, columns = 4097, 9
    graph = GraphCompiler(
        {
            "r": {"kind": "matrix", "axes": ["time", "asset"], "shape": ["T", "N"]},
            "w": {"kind": "vector", "axes": ["asset"], "shape": ["N"]},
        }
    ).compile(["std(portfolio_returns(r,w),1)*sqrt(252)", "sum(w)"], result_format=result_format)
    a = make_matrix(rows, columns, order)
    w = np.linspace(-0.1, 0.2, columns)
    a.flags.writeable = w.flags.writeable = False
    before_a, before_w = a.copy(), w.copy()
    starts, ends = np.array([0, 2048], np.int64), np.array([2048, rows], np.int64)
    expected = np.array(
        [
            [
                op.std(op.portfolio_returns(a[start:end], w, simd="scalar"), 1) * np.sqrt(252),
                w.sum(),
            ]
            for start, end in zip(starts, ends, strict=False)
        ]
    )
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(graph, {"r": a, "w": w}, starts, ends)
        first_values, first_arrays = scalar_rows(first, result_format)
        saved = tuple(value.copy() for value in first_arrays)
        second = scheduler.execute(graph, {"r": a, "w": w}, starts, ends)
        _, second_arrays = scalar_rows(second, result_format)
    np.testing.assert_allclose(first_values, expected, rtol=1e-12, atol=1e-12)
    assert_generations(first_arrays, saved, second_arrays)
    assert first.audit["lane"] == ("process" if lane.startswith("process") else lane)
    assert first.audit["python_worker_callbacks"] == 0
    calls = sum(chunk["eigen_matvec_calls"] for chunk in first.audit["native_chunks"])
    assert_library_calls({"eigen_matvec_calls": calls // 2}, True)
    assert calls % 2 == 0
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(w, before_w)
