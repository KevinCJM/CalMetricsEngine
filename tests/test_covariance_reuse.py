"""Same-source covariance/correlation reuse: results, failures and ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_covariance_library import MATRIX, data
from .test_typed_results import config, execute

RTOL = ATOL = 1e-12  # Existing covariance-library contract, unchanged.


def _count(result, field):
    return sum(chunk[field] for chunk in result.audit["native_chunks"])


def _pair_graph(*, error_policy="raise", distinct=False):
    declarations = {"x": MATRIX, **({"y": MATRIX} if distinct else {})}
    source = "y" if distinct else "x"
    return GraphCompiler(declarations).compile(
        ["covariance(x)", f"correlation({source})"], error_policy=error_policy
    )


def _compare_output(actual, expected):
    np.testing.assert_array_equal(actual.root_statuses, expected.root_statuses)
    np.testing.assert_array_equal(actual.shape_known, expected.shape_known)
    for row, (got, want) in enumerate(zip(actual.values, expected.values, strict=True)):
        got, want = np.asarray(got), np.asarray(want)
        assert got.shape == want.shape and got.dtype == want.dtype
        if got.dtype == np.float64:
            np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
            np.testing.assert_array_equal(np.isposinf(got), np.isposinf(want))
            np.testing.assert_array_equal(np.isneginf(got), np.isneginf(want))
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            np.testing.assert_array_equal(got, want)
        if expected.statuses[row] is None:
            assert actual.statuses[row] is None
        else:
            np.testing.assert_array_equal(actual.statuses[row], expected.statuses[row])


def _check_pair(result, x, row=0):
    expected_cov = op.covariance(x)
    expected_corr = op.correlation(x)
    covariance, correlation = result.outputs[:2]
    for output, expected in ((covariance, expected_cov), (correlation, expected_corr)):
        value = output.values[row]
        assert output.root_statuses[row] == 0 and output.shape_known[row]
        np.testing.assert_allclose(value, expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(value, value.T)
        assert not value.flags.writeable and not np.shares_memory(value, x)
    assert not np.shares_memory(covariance.values[row], correlation.values[row])


@pytest.mark.parametrize("layout", ["c", "f", "padded", "reverse_columns"])
def test_covariance_pair_reuses_one_core_for_finite_readonly_source(layout):
    if layout == "padded":
        owner = np.full((65, 12), np.nan)
        owner[:, :9] = data(65, 9, "c")
        x = owner[:, :9]
    else:
        owner = x = data(65, 9, layout)
    before = owner.copy()
    x.flags.writeable = False
    _, direct = op.covariance(x, audit=True)
    result = execute(_pair_graph(), {"x": x})
    _check_pair(result, x)
    assert _count(result, "eigen_covariance_calls") == direct["eigen_covariance_calls"]
    assert _count(result, "algorithm_copy_bytes") == 9 * 9 * 8
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("n,d", [(2, 0), (2, 1), (31, 8), (32, 8)])
def test_covariance_pair_geometry_keeps_empty_and_scalar_backend_contract(n, d):
    x = data(n, d, "c")
    _, direct = op.covariance(x, audit=True)
    result = execute(_pair_graph(), {"x": x})
    _check_pair(result, x)
    assert _count(result, "eigen_covariance_calls") == direct["eigen_covariance_calls"]
    assert _count(result, "algorithm_copy_bytes") == d * d * 8
    assert result.outputs[0].values[0].shape == result.outputs[1].values[0].shape == (d, d)


@pytest.mark.parametrize("n,d", [(0, 0), (0, 8), (1, 0), (1, 8)])
def test_covariance_pair_insufficient_rows_do_not_hide_behind_empty_columns(n, d):
    x = np.zeros((n, d))
    for name in ("covariance", "correlation"):
        with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
            op.call(name, x)
    actual = execute(_pair_graph(error_policy="isolate"), {"x": x})
    # Keep the scheduler's existing empty-window/status rules independently of
    # the direct operator error; different source node IDs disable reuse.
    reference = execute(
        _pair_graph(error_policy="isolate", distinct=True), {"x": x, "y": x}
    )
    for got, want in zip(actual.outputs, reference.outputs, strict=True):
        _compare_output(got, want)
        assert np.all(got.root_statuses != 0)
    assert _count(actual, "algorithm_copy_bytes") == 0


@pytest.mark.parametrize("case", ["reverse_roots", "different_source", "nonroot_covariance"])
def test_covariance_pair_does_not_reuse_outside_proven_dependency(case):
    x = data(65, 9, "c")
    covariance, direct = op.covariance(x, audit=True)
    correlation = op.correlation(x)
    declarations, bound = {"x": MATRIX}, {"x": x}
    if case == "reverse_roots":
        expressions, expected = ["correlation(x)", "covariance(x)"], [correlation, covariance]
    elif case == "different_source":
        declarations["y"], bound["y"] = MATRIX, x  # Same storage is not source identity.
        expressions, expected = ["covariance(x)", "correlation(y)"], [covariance, correlation]
    else:
        expressions = ["trace(covariance(x))", "correlation(x)"]
        expected = [np.trace(covariance), correlation]
    result = execute(GraphCompiler(declarations).compile(expressions), bound)
    for output, want in zip(result.outputs, expected, strict=True):
        assert output.root_statuses[0] == 0
        np.testing.assert_allclose(output.values[0], want, rtol=RTOL, atol=ATOL)
    assert _count(result, "eigen_covariance_calls") == 2 * direct["eigen_covariance_calls"]
    assert _count(result, "algorithm_copy_bytes") == 0


@pytest.mark.parametrize("fault", ["nan", "inf", "overflow"])
def test_covariance_pair_nonfinite_falls_back_with_identical_root_and_position_statuses(fault):
    x = data(65, 9, "c")
    if fault == "nan":
        x[17, 2] = np.nan
    elif fault == "inf":
        x[17, 2] = np.inf
    else:
        x *= 1e200
    expressions = ["covariance(x)", "correlation(x)",
                   "greater_than(covariance(x),0)", "greater_than(correlation(x),0)", "7"]
    reference_expressions = ["covariance(x)", "correlation(y)",
                             "greater_than(covariance(x),0)", "greater_than(correlation(y),0)", "7"]
    graph = GraphCompiler({"x": MATRIX}).compile(expressions, error_policy="isolate")
    reference_graph = GraphCompiler({"x": MATRIX, "y": MATRIX}).compile(
        reference_expressions, error_policy="isolate"
    )
    result = execute(graph, {"x": x})
    reference = execute(reference_graph, {"x": x, "y": x})
    for got, want in zip(result.outputs, reference.outputs, strict=True):
        _compare_output(got, want)
    assert result.outputs[-1].root_statuses[0] == 0
    assert result.outputs[-1].values[0] == 7
    assert _count(result, "algorithm_copy_bytes") == 0
    assert _count(result, "eigen_covariance_calls") == _count(reference, "eigen_covariance_calls")


@pytest.mark.parametrize("error_policy", ["raise", "isolate"])
def test_covariance_pair_zero_variance_keeps_covariance_healthy_and_error_order(error_policy):
    x = data(65, 9, "c")
    x[:, 0] = 7.0
    before = x.copy()
    expressions = ["covariance(x)", "correlation(x)",
                   "greater_than(covariance(x),0)", "greater_than(correlation(x),0)", "7"]
    graph = GraphCompiler({"x": MATRIX}).compile(expressions, error_policy=error_policy)
    if error_policy == "raise":
        with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
            execute(graph, {"x": x})
    else:
        result = execute(graph, {"x": x})
        covariance, correlation, cov_bool, corr_bool, healthy = result.outputs
        assert covariance.root_statuses[0] == cov_bool.root_statuses[0] == 0
        assert correlation.root_statuses[0] != 0
        assert corr_bool.root_statuses[0] == correlation.root_statuses[0]
        assert healthy.root_statuses[0] == 0 and healthy.values[0] == 7
        expected = op.covariance(x)
        np.testing.assert_allclose(covariance.values[0], expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(cov_bool.values[0], expected > 0)
        assert not np.shares_memory(covariance.values[0], correlation.values[0])
        # The copy occurred before normalization failed, and must stay audited.
        assert _count(result, "algorithm_copy_bytes") == 9 * 9 * 8
    np.testing.assert_array_equal(x, before)


def test_covariance_pair_prepared_snapshots_refresh_values_and_keep_previous_owners():
    x = data(65, 9, "c")
    graph = _pair_graph()
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": x}, starts, ends)
        first = prepared.run_snapshot()
        saved = [out.values[0].copy() for out in first.outputs]
        _check_pair(first, x)
        x *= np.linspace(1.0, 2.0, x.shape[1])
        x[:, 0] += x[:, 1] * 0.25
        second = prepared.run_snapshot()
        _check_pair(second, x)
        different = x.copy()
        different[:, 2] += different[:, 3] * 0.5
        rebound = scheduler.prepare_execution(graph, {"x": different}, starts, ends).run_snapshot()
        _check_pair(rebound, different)
    for i, snapshot in enumerate(saved):
        np.testing.assert_array_equal(first.outputs[i].values[0], snapshot)
        values = [result.outputs[i].values[0] for result in (first, second, rebound)]
        for j, value in enumerate(values):
            assert not value.flags.writeable
            assert not np.shares_memory(value, x) and not np.shares_memory(value, different)
            assert all(not np.shares_memory(value, old) for old in values[:j])
    for result in (first, second, rebound):
        assert _count(result, "algorithm_copy_bytes") == 9 * 9 * 8


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_covariance_pair_reuse_is_per_interval_with_native_transport_and_independent_roots(lane):
    x = data(65, 9, "c")
    x.flags.writeable = False
    starts, ends = [0, 3], [65, 63]
    graph = GraphCompiler({"x": MATRIX}).compile(
        ["covariance(x)", "correlation(x)", "greater_than(correlation(x),0)", "7"],
        error_policy="isolate",
    )
    result = execute(graph, {"x": x}, starts, ends, lane)
    expected_calls = 0
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        view = x[start:end]
        _check_pair(result, view, row)
        _, audit = op.covariance(view, audit=True)
        expected_calls += audit["eigen_covariance_calls"]
        np.testing.assert_array_equal(result.outputs[2].values[row], op.correlation(view) > 0)
        assert result.outputs[2].root_statuses[row] == 0
        assert result.outputs[3].root_statuses[row] == 0 and result.outputs[3].values[row] == 7
    assert _count(result, "eigen_covariance_calls") == expected_calls
    assert _count(result, "algorithm_copy_bytes") == 2 * 9 * 9 * 8
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    assert result.audit["cpu_tokens"] <= result.audit["cpu_budget"] == 2
    for output in result.outputs[:2]:
        assert not np.shares_memory(output.values[0], output.values[1])
    if lane.startswith("process"):
        assert result.plan.use_shared_memory == (lane == "process_shared")
    if lane == "process_shared":
        assert result.audit["output_copy_bytes"] == 0


def test_covariance_pair_does_not_reuse_outer_root_inside_iteration_scope():
    x = data(64, 8, "c")
    seed = np.zeros((8, 8))
    _, audit = op.covariance(x, audit=True)
    graph = GraphCompiler({
        "x": MATRIX,
        "seed": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["D", "D"]},
    }).compile(["covariance(x)", "iterate(correlation(x)+iterate_x*0,seed,1e-12,2)"])
    result = execute(graph, {"x": x, "seed": seed})
    _check_pair(result, x)
    assert _count(result, "eigen_covariance_calls") == 3 * audit["eigen_covariance_calls"]
