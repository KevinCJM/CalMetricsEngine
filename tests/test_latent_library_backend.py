"""Ordered Eigen EM feature lanes preserve model, stopping and ownership rules."""

import os
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from . import latent_reference as ref
from .test_latent_operators import fixture
from .test_typed_results import config


def feature_stride_view(x):
    """Same values with non-unit feature stride select the active scalar path."""
    storage = np.empty((x.shape[0], x.shape[1] * 2))
    storage[:, ::2] = x
    result = storage[:, ::2]
    result.flags.writeable = False
    return result


def arguments(hmm, x, w, a, m, v, iterations=5, tolerance=0.0):
    return (x, w, a, m, v, iterations, tolerance) if hmm else (
        x, w, m, v, iterations, tolerance
    )


def execute(hmm, values, **kwargs):
    return op.call("gaussian_hmm_em" if hmm else "gaussian_mixture_em", *values, **kwargs)


def assert_identical(actual, expected):
    # Include all diagnostic fields and signed-zero bits, not just projections.
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("d", [16, 32, 64, 128, 256])
def test_explicit_scalar_disables_eigen_for_large_eligible_input(hmm, d):
    x, w, a, m, v = fixture(257, d, 3)
    values = arguments(hmm, x, w, a, m, v)
    scalar, scalar_audit = execute(hmm, values, simd="scalar", audit=True)
    automatic, auto_audit = execute(hmm, values, audit=True)
    assert scalar_audit["isa"] == "scalar"
    assert np.isfinite(scalar).all()
    assert_identical(automatic, scalar)
    assert scalar_audit["workspace_bytes"] == auto_audit["workspace_bytes"]
    assert scalar_audit["algorithm_copy_bytes"] == auto_audit["algorithm_copy_bytes"]
    assert scalar_audit["input_copy_bytes"] == auto_audit["input_copy_bytes"] == 0


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("n", [32, 2048])
def test_unselected_feature_width_uses_scalar_without_changing_result(hmm, n):
    x, w, a, m, v = fixture(n, 6, 4)
    values = arguments(hmm, x, w, a, m, v)
    expected, scalar_audit = execute(hmm, values, simd="scalar", audit=True)
    actual, audit = execute(hmm, values, audit=True)
    assert audit["isa"] == scalar_audit["isa"] == "scalar"
    assert np.isfinite(actual).all()
    assert_identical(actual, expected)
    assert audit["workspace_bytes"] == scalar_audit["workspace_bytes"]
    assert audit["algorithm_copy_bytes"] == scalar_audit["algorithm_copy_bytes"]
    assert audit["input_copy_bytes"] == scalar_audit["input_copy_bytes"] == 0


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize(
    "n,d", [(31, 4), (32, 3), (32, 4), (33, 5), (65, 6), (65, 8), (65, 9), (65, 16), (31, 64), (32, 32), (33, 64), (65, 128), (65, 256)]
)
@pytest.mark.parametrize("layout", ["c", "f", "reverse_rows", "slice_rows", "broadcast_rows"])
def test_em_order_layout_workspace_and_ownership(hmm, n, d, layout):
    x, w, a, m, v = fixture(n, d, 2)
    if layout == "f":
        x = np.asfortranarray(x)
    elif layout == "reverse_rows":
        x = x[::-1]
    elif layout == "slice_rows":
        x = np.repeat(x, 2, axis=0)[::2]
    elif layout == "broadcast_rows":
        x = np.broadcast_to(x[:1], (n, d))
    arrays = (x, w, a, m, v)
    before = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    values = arguments(hmm, *arrays)
    expected, scalar = execute(
        hmm, arguments(hmm, feature_stride_view(x), w, a, m, v), audit=True
    )
    workspace = op.Workspace()
    actual, audit = execute(hmm, values, workspace=workspace, audit=True)
    assert_identical(actual, expected)
    assert scalar["isa"] == "scalar"
    eligible_platform = (
        platform.system() == "Darwin" and platform.machine() == "arm64"
        and int(platform.mac_ver()[0].split(".")[0]) >= 15
        and n >= 32
    )
    expected_isa = "scalar"
    if eligible_platform:
        if d in {4, 5, 8, 9, 16, 32, 64, 128, 256} and x.strides[1] == 8:
            expected_isa = "eigen-array"
        elif d >= 8 and x.strides[0] == 8 and x.strides[1] >= n * 8:
            # Width nine uses the column density lane, then the scalar M-step;
            # complete feature blocks also use the column M-step.
            expected_isa = "eigen-array-column-em" if d % 8 == 0 else "eigen-array-column-density"
    if os.environ.get("CME_EXPECT_EIGEN"):
        assert audit["isa"] == (expected_isa if os.environ["CME_EXPECT_EIGEN"] == "1" else "scalar")
    else:
        assert audit["isa"] in {"scalar", expected_isa}
    expected_scratch = (3 * n * 2 + 2 * 2 * 2 + 4 * 2) if hmm else n * 2 + 3 * 2
    assert audit["workspace_bytes"] == scalar["workspace_bytes"] == 8 * expected_scratch
    assert audit["input_copy_bytes"] == scalar["input_copy_bytes"] == 0
    assert audit["algorithm_copy_bytes"] == scalar["algorithm_copy_bytes"]
    frozen = actual.copy()
    later = execute(hmm, values, workspace=workspace)
    assert not np.shares_memory(actual, later)
    assert_identical(actual, frozen)
    for array, saved in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(array, saved)
        assert not np.shares_memory(actual, array)


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("d", [4, 9, 32, 64, 128, 256])
def test_em_eigen_matches_independent_explicit_reference(hmm, d):
    x, w, a, m, v = fixture(65, d, 3)
    values = arguments(hmm, x, w, a, m, v)
    expected = (ref.hmm_em_reference if hmm else ref.gmm_em_reference)(*values)
    actual = execute(hmm, values)
    projections = (
        (op.hmm_initial, op.hmm_transition, op.hmm_means, op.hmm_variances)
        if hmm else (op.mixture_weights, op.mixture_means, op.mixture_variances)
    )
    for project, target in zip(projections, expected[:-2], strict=True):
        np.testing.assert_allclose(project(actual), target, rtol=1e-10, atol=1e-10)
    assert op.em_iterations(actual) == expected[-2]
    assert not op.em_converged(actual)
    np.testing.assert_allclose(op.em_log_likelihood(actual), expected[-1], rtol=1e-12)


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("case", ["zero_weight", "zero_posterior", "constant", "large_finite"])
@pytest.mark.parametrize("d", [8, 32, 64, 128, 256])
def test_em_floors_zero_weights_and_product_order(hmm, case, d):
    x, w, a, m, v = fixture(65, d, 2)
    if case == "zero_weight":
        w[:] = [1.0, 0.0]
    elif case == "zero_posterior":
        m[1] = 1000.0
        v[:] = 1.0
    elif case == "constant":
        x[:] = 2.0
        m[:] = 2.0
        v[:] = 0.0
    elif case == "large_finite":
        x *= 1e140
        m *= 1e140
        v *= 1e280
        w[:] = [1.0 - 1e-12, 1e-12]
    expected = execute(hmm, arguments(hmm, feature_stride_view(x), w, a, m, v))
    actual = execute(hmm, arguments(hmm, x, w, a, m, v))
    assert_identical(actual, expected)
    variances = op.hmm_variances(actual) if hmm else op.mixture_variances(actual)
    assert np.all(variances >= 1e-5)


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("d", [8, 32, 64, 128, 256])
def test_em_stopping_at_tolerance_neighbors_is_exact(hmm, d):
    x, w, a, m, v = fixture(65, d, 2)
    scalar_x = feature_stride_view(x)
    first = execute(hmm, arguments(hmm, scalar_x, w, a, m, v, 1))
    second = execute(hmm, arguments(hmm, scalar_x, w, a, m, v, 2))
    difference = abs(op.em_log_likelihood(second) - op.em_log_likelihood(first))
    assert difference > 0
    for tolerance in (np.nextafter(difference, 0.0), difference,
                      np.nextafter(difference, np.inf)):
        expected = execute(hmm, arguments(hmm, scalar_x, w, a, m, v, 3, tolerance))
        actual = execute(hmm, arguments(hmm, x, w, a, m, v, 3, tolerance))
        assert_identical(actual, expected)
        if tolerance > difference:
            assert op.em_iterations(actual) == 2
        else:
            assert op.em_iterations(actual) == 3


@pytest.mark.parametrize("hmm,n,d,k,step", [(False, 65, 8, 8, 5), (True, 32, 3, 2, 3)])
def test_em_later_stopping_boundary_keeps_discrete_diagnostics(hmm, n, d, k, step):
    # Vector exp can pass fixed-iteration tolerances yet change these EM stops.
    x, w, a, m, v = fixture(n, d, k)
    scalar_x = feature_stride_view(x)
    before = execute(hmm, arguments(hmm, scalar_x, w, a, m, v, step - 1))
    after = execute(hmm, arguments(hmm, scalar_x, w, a, m, v, step))
    difference = abs(op.em_log_likelihood(after) - op.em_log_likelihood(before))
    assert difference > 0
    for tolerance in (np.nextafter(difference, 0.0), difference,
                      np.nextafter(difference, np.inf)):
        expected = execute(hmm, arguments(hmm, scalar_x, w, a, m, v, step + 1, tolerance))
        actual = execute(hmm, arguments(hmm, x, w, a, m, v, step + 1, tolerance))
        assert_identical(actual, expected)


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("d", [8, 32, 64, 128, 256])
def test_em_nonfinite_failure_keeps_scalar_status(hmm, bad, d):
    x, w, a, m, v = fixture(65, d, 2)
    x[7, 3] = bad
    for data in (x, feature_stride_view(x)):
        with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
            execute(hmm, arguments(hmm, data, w, a, m, v))


def graph_fixture(hmm, d=8):
    x, w, a, m, v = fixture(65, d, 2)
    types = {
        "x": {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]},
        "w": {"kind": "vector", "axes": ["component"], "shape": ["K"]},
        "a": {"kind": "matrix", "axes": ["component", "component"], "shape": ["K", "K"]},
        "m": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "v": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
    }
    data = dict(x=x, w=w, a=a, m=m, v=v)
    expression = "gaussian_hmm_em(x,w,a,m,v,5,0)" if hmm else "gaussian_mixture_em(x,w,m,v,5,0)"
    mean = "hmm_means" if hmm else "mixture_means"
    graph = GraphCompiler(types).compile(
        [f"{mean}({expression})", f"em_iterations({expression})",
         f"em_converged({expression})", "7"], result_format="typed", error_policy="isolate"
    )
    return graph, data


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("d", [8, 64])
def test_em_graph_cse_transport_and_failure_isolation(hmm, lane, d):
    graph, data = graph_fixture(hmm, d)
    expected = execute(hmm, arguments(hmm, *[data[k] for k in ("x", "w", "a", "m", "v")]))
    mean = op.hmm_means(expected) if hmm else op.mixture_means(expected)
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(graph, data, bounds, bounds)
        for row in range(2):
            np.testing.assert_array_equal(result.outputs[0].values[row], mean)
            assert result.outputs[1].values[row] == op.em_iterations(expected)
            assert result.outputs[2].values[row] == op.em_converged(expected)
        data["x"][7, 3] = np.nan
        failed = scheduler.execute(graph, data, bounds, bounds)
        assert np.all(failed.outputs[0].root_statuses != 0)
        assert np.all(failed.outputs[3].root_statuses == 0)
        np.testing.assert_array_equal(failed.outputs[3].values, [7.0, 7.0])


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("d", [8, 64])
def test_em_prepared_snapshot_rereads_input_and_retains_results(hmm, d):
    graph, data = graph_fixture(hmm, d)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, data, bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        data["x"][:] += 0.25
        second = prepared.run_snapshot()
        expected = execute(hmm, arguments(hmm, *[data[k] for k in ("x", "w", "a", "m", "v")]))
        mean = op.hmm_means(expected) if hmm else op.mixture_means(expected)
        np.testing.assert_array_equal(second.outputs[0].values[0], mean)
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
