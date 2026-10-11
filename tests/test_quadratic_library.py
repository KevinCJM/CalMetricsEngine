"""Ordered quadratic projection reuses the native matvec backend and workspace."""

import hashlib

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matrix_array_foundation import matrix, run
from .test_matrix_vector_library import assert_ieee_equal, assert_library_calls
from .test_typed_results import config


def sample(n, layout="c"):
    rng = np.random.default_rng(291041 + n)
    a = rng.normal(size=(n, n))
    x = rng.normal(size=n)
    if layout == "f":
        a = np.asfortranarray(a)
    elif layout in {"padded_c", "padded_f"}:
        if layout == "padded_c":
            storage = np.full((n, n + 3), np.nan)
            view = storage[:, :n]
        else:
            storage = np.full((n + 3, n), np.nan, order="F")
            view = storage[:n, :]
        view[:] = a
        a = view
    elif layout == "reverse":
        a = a[::-1, ::-1]
    elif layout == "slice":
        a = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)[::2, ::2]
    return x, a


def declarations():
    return {
        "a": matrix(),
        "x": {"kind": "vector", "axes": ["asset"], "shape": ["N"]},
    }


def cancellation(n=129, order="C"):
    a = np.zeros((n, n), order=order)
    a[0, :3] = [1e16, 1, -1e16]
    a[1, 0] = 1
    return np.ones(n), a


@pytest.mark.parametrize("n", [0, 1, 127, 128, 129, 257])
@pytest.mark.parametrize("layout", ["c", "f", "padded_c", "padded_f", "reverse", "slice"])
def test_quadratic_layout_boundaries_workspace_and_scalar_bypass(n, layout):
    x, a = sample(n, layout)
    saved_x, saved_a = x.copy(), a.copy()
    x.flags.writeable = a.flags.writeable = False
    workspace = op.Workspace()
    expected, scalar = op.quadratic_form(x, a, workspace=workspace, simd="scalar", audit=True)
    actual, audit = op.quadratic_form(x, a, workspace=workspace, audit=True)
    assert_ieee_equal(actual, expected)
    assert scalar["isa"] == "scalar"
    assert scalar["eigen_matvec_calls"] == 0
    assert_library_calls(audit, n >= 128 and layout not in {"reverse", "slice"})
    assert op.get("quadratic_form").requirements(x, a)["scratch_doubles"] == n
    assert audit["workspace_bytes"] == scalar["workspace_bytes"] == n * 8
    assert audit["workspace_capacity_bytes"] == scalar["workspace_capacity_bytes"]
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0
    np.testing.assert_array_equal(x, saved_x)
    np.testing.assert_array_equal(a, saved_a)
    if layout == "padded_c":
        assert np.isnan(a.base[:, n:]).all()
    elif layout == "padded_f":
        assert np.isnan(a.base[n:, :]).all()


def test_quadratic_c_layout_crosses_projection_cache_block_without_extra_matrix_copy():
    # A.T uses the F backend: 4096 projected elements plus one tail element.
    # Keep just one 128 MiB matrix; sparse writes, streamed hashes and scalar
    # references avoid a second matrix, a transpose copy or a BLAS temporary.
    n = 4097
    a = np.zeros((n, n))
    x = np.ones(n)
    a[0, :3] = [1e16, 1, -1e16]
    a[1, 0] = 1
    a[0, 4095], a[1, 4095] = 2, 3
    a[4096, 4096] = 7
    x[4096] = 2
    assert a.flags.c_contiguous
    assert a.T.strides[0] == 8
    x.flags.writeable = a.flags.writeable = False
    before = hashlib.sha256(memoryview(a)).digest()
    workspace = op.Workspace()
    expected, scalar = op.quadratic_form(x, a, workspace=workspace, simd="scalar", audit=True)
    actual, audit = op.quadratic_form(x, a, workspace=workspace, audit=True)
    # The first three columns cancel to 0; the block boundary contributes 5
    # and the final cache block contributes (7*2)*2 = 28.
    assert_ieee_equal(expected, 33.0)
    assert_ieee_equal(actual, expected)
    assert scalar["eigen_matvec_calls"] == 0
    assert_library_calls(audit, True)
    assert audit["workspace_bytes"] == scalar["workspace_bytes"] == n * 8
    assert audit["workspace_capacity_bytes"] == scalar["workspace_capacity_bytes"]
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0
    assert hashlib.sha256(memoryview(a)).digest() == before
    assert not a.flags.writeable and not x.flags.writeable
    assert np.count_nonzero(x != 1.0) == 1 and x[-1] == 2.0


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("stride", [-2, -1, 0, 1, 2])
def test_quadratic_signed_and_broadcast_vector_strides(order, stride):
    x, a = sample(129, order.lower())
    if stride == 0:
        x = np.broadcast_to(np.array(0.5), x.shape)
    elif abs(stride) == 2:
        x = np.repeat(x, 2)[::stride]
    elif stride == -1:
        x = x[::-1]
    x.flags.writeable = a.flags.writeable = False
    expected = op.quadratic_form(x, a, simd="scalar")
    actual, audit = op.quadratic_form(x, a, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, True)


@pytest.mark.parametrize("order", ["C", "F"])
def test_quadratic_keeps_transposed_projection_before_final_dot(order):
    x, a = cancellation(order=order)
    # Column projection then dot gives 0; the algebraically equivalent A@x
    # then dot gives 1 for this nonsymmetric, cancellation-sensitive fixture.
    assert op.dot(op.matvec(a.T, x, simd="scalar"), x, simd="scalar") == 0.0
    assert op.dot(op.matvec(a, x, simd="scalar"), x, simd="scalar") == 1.0
    actual, audit = op.quadratic_form(x, a, audit=True)
    assert_ieee_equal(actual, 0.0)
    assert_library_calls(audit, True)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "fault", ["nan", "positive_inf", "negative_inf", "zero_inf", "overflow",
              "overflow_cancel", "subnormal", "signed_zero", "fma"]
)
def test_quadratic_ieee_classification_and_separate_multiply_add(order, fault):
    n = 129  # Includes the R8 tail after a transposed F input.
    a = np.zeros((n, n), order=order)
    x = np.ones(n)
    if fault in {"nan", "positive_inf", "negative_inf", "zero_inf"}:
        a[-1, -1] = {
            "nan": np.nan, "positive_inf": np.inf,
            "negative_inf": -np.inf, "zero_inf": np.inf,
        }[fault]
        if fault == "zero_inf":
            x[-1] = 0.0
    elif fault == "overflow":
        a[:2, -1] = 1e308
    elif fault == "overflow_cancel":
        a[:2, -1] = [np.finfo(float).max, -np.finfo(float).max]
        x[:2] = 2.0
    elif fault == "subnormal":
        a[-1, -1] = np.nextafter(0.0, 1.0)
    elif fault == "signed_zero":
        a[::2] = -0.0
        x[::2] = -0.0
    else:
        a[0, -1] = -1
        a[1, -1] = 1 + 2.0**-27
        x[1] = 1 - 2.0**-27
    expected = op.quadratic_form(x, a, simd="scalar")
    actual, audit = op.quadratic_form(x, a, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, True)
    if fault == "fma":
        assert_ieee_equal(actual, 0.0)
    elif fault == "zero_inf":
        assert np.isnan(actual)


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("view", ["row", "column", "reverse_row", "broadcast"])
def test_quadratic_readonly_inputs_may_share_storage(layout, view):
    _, a = sample(129, layout)
    if view == "row":
        x = a[3]
    elif view == "column":
        x = a[:, 3]
    elif view == "reverse_row":
        x = a[3, ::-1]
    else:
        x = np.broadcast_to(a[3, :1], (129,))
    saved = a.copy()
    assert np.shares_memory(x, a)
    x.flags.writeable = a.flags.writeable = False
    expected = op.quadratic_form(x, a, simd="scalar")
    actual, audit = op.quadratic_form(x, a, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, True)
    np.testing.assert_array_equal(a, saved)


@pytest.mark.parametrize("alias", ["matrix", "vector"])
@pytest.mark.parametrize("simd", ["scalar", "auto"])
def test_quadratic_rejects_output_alias_before_writes(alias, simd):
    x, a = sample(129)
    out = (a[:1, :1] if alias == "matrix" else x[:1]).reshape(())
    saved_a, saved_x = a.copy(), x.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIASES_INPUT"):
        op.quadratic_form(x, a, out=out, simd=simd)
    np.testing.assert_array_equal(a, saved_a)
    np.testing.assert_array_equal(x, saved_x)


@pytest.mark.parametrize("shape", [(129, 128), (128, 129), (128, 128)])
def test_quadratic_shape_checks_precede_execution(shape):
    x = np.ones(129)
    a = np.ones(shape)
    out = np.array(123.0)
    for simd in ("scalar", "auto"):
        with pytest.raises(ValueError, match="SHAPE_MISMATCH"):
            op.quadratic_form(x, a, out=out, simd=simd)
        assert out == 123.0


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_quadratic_cse_audit_boolean_and_independent_failure_transport(lane):
    x, a = cancellation()
    _, direct = op.quadratic_form(x, a, audit=True)
    graph = GraphCompiler({**declarations(), "bad": matrix(symbol="B")}).compile(
        ["quadratic_form(x,a)", "quadratic_form(x,a)>0", "cholesky(bad)", "x"],
        error_policy="isolate",
    )
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    result = run(graph, {"a": a, "x": x, "bad": np.zeros((2, 2))}, lane)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    calls = sum(c["eigen_matvec_calls"] for c in result.audit["native_chunks"])
    assert calls == 2 * direct["eigen_matvec_calls"]
    for row in range(2):
        assert_ieee_equal(result.outputs[0].values[row], 0.0)
        assert not result.outputs[1].values[row]
        assert result.outputs[1].values[row].dtype == np.bool_
        assert result.outputs[0].root_statuses[row] == 0
        assert result.outputs[1].root_statuses[row] == 0
        assert result.outputs[2].root_statuses[row] != 0
        np.testing.assert_array_equal(result.outputs[3].values[row], x)
        assert not np.shares_memory(result.outputs[3].values[row], x)


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_quadratic_overflow_isolates_consumers_but_preserves_healthy_roots(lane):
    x = np.ones(129)
    a = np.zeros((129, 129))
    a[:2, -1] = 1e308
    assert np.isinf(op.quadratic_form(x, a, simd="scalar"))
    graph = GraphCompiler(declarations()).compile(
        ["quadratic_form(x,a)", "quadratic_form(x,a)>0", "sum(x)", "x"],
        error_policy="isolate",
    )
    result = run(graph, {"a": a, "x": x}, lane)
    for row in range(2):
        assert result.outputs[0].root_statuses[row] != 0
        assert result.outputs[1].root_statuses[row] != 0
        assert result.outputs[2].root_statuses[row] == 0
        assert result.outputs[3].root_statuses[row] == 0
        assert result.outputs[2].values[row] == 129.0
        np.testing.assert_array_equal(result.outputs[3].values[row], x)


@pytest.mark.parametrize("order", ["C", "F"])
def test_quadratic_prepared_rereads_inputs_and_retains_independent_snapshots(order):
    x, a = sample(129, order.lower())
    graph = GraphCompiler(declarations()).compile(
        ["quadratic_form(x,a)", "quadratic_form(x,a)>0", "x"]
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "x": x}, bounds, bounds)
        first = prepared.run_snapshot()
        saved = [output.values[0].copy() for output in first.outputs]
        assert_ieee_equal(saved[0], op.quadratic_form(x, a, simd="scalar"))
        _, direct = op.quadratic_form(x, a, audit=True)
        x *= 2
        a *= -0.5
        second = prepared.run_snapshot()
        expected = op.quadratic_form(x, a, simd="scalar")
        assert_ieee_equal(second.outputs[0].values[0], expected)
        assert second.outputs[1].values[0] == (expected > 0)
        np.testing.assert_array_equal(second.outputs[2].values[0], x)
        for original, later, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
            np.testing.assert_array_equal(original.values[0], snapshot)
            assert not np.shares_memory(original.values[0], later.values[0])
        for result in (first, second):
            calls = sum(c["eigen_matvec_calls"] for c in result.audit["native_chunks"])
            assert calls == direct["eigen_matvec_calls"]
