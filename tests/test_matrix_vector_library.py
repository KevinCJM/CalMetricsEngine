"""Ordered Eigen matvec: IEEE parity, dispatch, ownership and native transport."""

import ctypes
import os
import platform
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matrix_array_foundation import matrix, run
from .test_matrix_library_backend import PACKING_PLATFORM
from .test_typed_results import config


def assert_library_calls(audit, eligible):
    calls = audit["eigen_matvec_calls"]
    expected = os.environ.get("CME_EXPECT_EIGEN")
    if not eligible or not PACKING_PLATFORM:
        assert calls == 0
    elif expected is not None:
        assert calls == int(expected)
    else:
        assert calls in (0, 1)


def assert_ieee_equal(actual, expected):
    """Match finite bits, infinity signs and NaN positions, not NaN payloads."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    numeric = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[numeric].view(np.uint64), expected[numeric].view(np.uint64)
    )


def layout_array(x, layout):
    if layout == "f":
        return np.asfortranarray(x)
    if layout == "reverse":
        return x[::-1, ::-1]
    if layout == "slice":
        return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)[::2, ::2]
    return x


def positive_matrix(n, layout="c"):
    # Diagonal dominance gives a dense SPD matrix without a separate BLAS setup.
    rng = np.random.default_rng(508 + n)
    x = rng.uniform(-0.05, 0.05, (n, n))
    x = x + x.T
    x[np.diag_indices(n)] += n + 1
    return layout_array(x, layout)


def assert_no_workspace(audit):
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0
    assert audit["input_copy_bytes"] == 0


def rectangular_declarations():
    return {
        "a": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["N", "D"]},
        "b": {"kind": "vector", "axes": ["asset"], "shape": ["D"]},
    }


def large_matvec_inputs(rows=4096, columns=32):
    rng = np.random.default_rng(749)
    return np.asfortranarray(rng.normal(size=(rows, columns))), rng.normal(size=columns)


@pytest.mark.parametrize("n", [0, 1, 127, 128, 129])
@pytest.mark.parametrize("layout", ["c", "f", "reverse", "slice"])
def test_cholesky_layout_scalar_parity_and_independent_results(n, layout):
    a = positive_matrix(n, layout)
    frozen_input = a.copy()
    a.flags.writeable = False
    work = op.Workspace()
    expected, scalar_audit = op.cholesky(a, simd="scalar", audit=True)
    actual, audit = op.cholesky(a, workspace=work, audit=True)
    assert scalar_audit["eigen_matvec_calls"] == 0
    assert_library_calls(audit, False)
    assert_no_workspace(audit)
    assert op.get("cholesky").requirements(a)["scratch_doubles"] == 0
    assert audit["algorithm_copy_bytes"] == n * (n + 1) // 2 * 8
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual @ actual.T, a, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual, np.tril(actual))
    assert not np.shares_memory(a, actual)
    retained = actual.copy()
    op.cholesky(a * 2, workspace=work)
    np.testing.assert_array_equal(actual, retained)
    np.testing.assert_array_equal(a, frozen_input)


@pytest.mark.parametrize("n", [128, 129])
@pytest.mark.parametrize("small_diagonal", [1e-14, 1e-24, np.finfo(float).tiny])
def test_cholesky_small_positive_pivot_is_not_rejected(n, small_diagonal):
    # Cholesky permits positive pivots below the solve operator's 1e-14 threshold.
    a = np.eye(n)
    a[-1, -1] = small_diagonal
    expected = op.cholesky(a, simd="scalar")
    actual, audit = op.cholesky(a, audit=True)
    np.testing.assert_array_equal(actual, expected)
    assert_library_calls(audit, False)
    assert audit["algorithm_copy_bytes"] == n * (n + 1) // 2 * 8
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0


@pytest.mark.parametrize("direction", [0.0, 1.0, np.inf])
def test_cholesky_near_degenerate_positive_block_keeps_status(direction):
    a = np.eye(128)
    a[-2, -1] = a[-1, -2] = 1
    a[-1, -1] = np.nextafter(1.0, direction) if direction != 1.0 else 1.0
    if direction == np.inf:
        expected = op.cholesky(a, simd="scalar")
        actual, audit = op.cholesky(a, audit=True)
        np.testing.assert_array_equal(actual, expected)
        assert_library_calls(audit, False)
    else:
        for simd in ("scalar", "auto"):
            with pytest.raises(ValueError, match="DOMAIN_ERROR"):
                op.cholesky(a, simd=simd)


@pytest.mark.parametrize(
    "fault,error",
    [
        ("negative", "DOMAIN_ERROR"),
        ("asymmetric", "DOMAIN_ERROR"),
        ("nan", "NON_FINITE_RESULT"),
        ("inf", "NON_FINITE_RESULT"),
        ("nan_and_asymmetric", "NON_FINITE_RESULT"),
        ("overflow", "NON_FINITE_RESULT"),
    ],
)
def test_cholesky_error_class_and_priority(fault, error):
    a = np.eye(128)
    if fault == "negative":
        a[-1, -1] = -1
    elif fault == "asymmetric":
        a[0, -1] = 0.25
    elif fault == "nan":
        a[-1, -1] = np.nan
    elif fault == "inf":
        a[-1, -1] = np.inf
    elif fault == "nan_and_asymmetric":
        a[0, -1] = 0.25
        a[-1, -1] = np.nan
    else:
        a[0, 0] = np.finfo(float).tiny
        a[0, 1] = a[1, 0] = 1e200
    before = a.copy()
    for simd in ("scalar", "auto"):
        with pytest.raises(ValueError, match=error):
            op.cholesky(a, simd=simd)
    np.testing.assert_array_equal(a, before)


@pytest.mark.parametrize("scale", [1e-310, 1e308])
def test_cholesky_extreme_scale_uses_original_numerical_path(scale):
    a = np.eye(128) * scale
    expected = op.cholesky(a, simd="scalar")
    actual, audit = op.cholesky(a, audit=True)
    np.testing.assert_array_equal(actual, expected)
    assert_library_calls(audit, False)
    assert audit["algorithm_copy_bytes"] == 128 * 129 // 2 * 8


@pytest.mark.parametrize("rows,columns", [(511, 32), (512, 32), (128, 128), (256, 31)])
@pytest.mark.parametrize("layout", ["c", "f", "reverse", "slice"])
def test_matvec_layout_scalar_parity_and_independent_results(rows, columns, layout):
    rng = np.random.default_rng(910 + columns)
    a = layout_array(rng.normal(size=(rows, columns)), layout)
    b = rng.normal(size=columns)
    before_a, before_b = a.copy(), b.copy()
    a.flags.writeable = b.flags.writeable = False
    work = op.Workspace()
    expected, scalar_audit = op.matvec(a, b, simd="scalar", audit=True)
    actual, audit = op.matvec(a, b, workspace=work, audit=True)
    assert scalar_audit["eigen_matvec_calls"] == 0
    assert_library_calls(audit, layout in ("c", "f"))
    assert_ieee_equal(actual, expected)
    np.testing.assert_allclose(actual, a @ b, rtol=1e-12, atol=1e-12)
    assert_no_workspace(audit)
    assert audit["algorithm_copy_bytes"] == 0
    assert op.get("matvec").requirements(a, b)["scratch_doubles"] == 0
    assert not np.shares_memory(actual, a) and not np.shares_memory(actual, b)
    retained = actual.copy()
    op.matvec(a, b * 2, workspace=work)
    np.testing.assert_array_equal(actual, retained)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("shape", [(4096, 32), (4097, 64)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_large_matvec_dispatch_padding_and_output_ownership(shape, order, padded, offset):
    rows, columns = shape
    values, b = large_matvec_inputs(rows, columns)
    extra = 3 if padded else 0
    stored_shape = (
        (rows + extra + offset, columns)
        if order == "F" else (rows, columns + extra + offset)
    )
    storage = np.full(stored_shape, np.nan, order=order)
    a = (
        storage[offset:offset + rows, :]
        if order == "F" else storage[:, offset:offset + columns]
    )
    a[:] = values
    a.flags.writeable = b.flags.writeable = False
    before_a, before_b = a.copy(), b.copy()
    expected, scalar_audit = op.matvec(a, b, simd="scalar", audit=True)
    work = op.Workspace()
    out = np.full(rows, np.nan)
    actual, audit = op.matvec(a, b, out=out, workspace=work, audit=True)
    assert scalar_audit["eigen_matvec_calls"] == 0
    assert_library_calls(audit, True)
    assert_no_workspace(audit)
    assert audit["algorithm_copy_bytes"] == 0
    assert np.shares_memory(actual, out)
    assert_ieee_equal(actual, expected)
    np.testing.assert_allclose(actual, a @ b, rtol=1e-12, atol=1e-12)
    retained = actual.copy()
    op.matvec(a, b * 2, workspace=work)
    np.testing.assert_array_equal(actual, retained)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)
    for padding in (
        (storage[:offset, :], storage[offset + rows:, :])
        if order == "F" else (storage[:, :offset], storage[:, offset + columns:])
    ):
        assert np.all(np.isnan(padding))


@pytest.mark.parametrize(
    "shape",
    [(0, 8), (1, 8), (127, 8), (128, 0), (128, 7), (128, 8), (129, 9),
     (4095, 32), (4096, 31), (4096, 129), (2048, 2048)],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_matvec_dispatch_shape_boundaries_and_wide_matrix(shape, order):
    a, b = large_matvec_inputs(*shape)
    a = np.array(a, order=order, copy=True)
    actual, audit = op.matvec(a, b, audit=True)
    expected = op.matvec(a, b, simd="scalar")
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, shape[0] >= 128 and shape[1] >= 8)
    assert_no_workspace(audit)
    assert audit["algorithm_copy_bytes"] == 0


@pytest.mark.parametrize("stride", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("layout", ["c", "f"])
def test_matvec_signed_and_broadcast_vector_strides(layout, stride):
    a = positive_matrix(128, layout)
    original = np.arange(128, dtype=float)
    if stride == 0:
        b = np.broadcast_to(np.array(2.0), (128,))
    elif abs(stride) == 2:
        b = np.repeat(original, 2)[::stride]
    elif stride == 1:
        b = original
    else:
        b = original[::-1]
    before_a, before_b = a.copy(), b.copy()
    a.flags.writeable = b.flags.writeable = False
    expected = op.matvec(a, b, simd="scalar")
    actual, audit = op.matvec(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, layout in ("c", "f"))
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0
    assert_no_workspace(audit)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("alias", ["matrix", "vector"])
@pytest.mark.parametrize("simd", ["scalar", "auto"])
def test_matvec_rejects_explicit_output_alias_before_writes(layout, alias, simd):
    a = positive_matrix(128, layout)
    b = np.linspace(-1, 1, 128)
    out = b if alias == "vector" else (a[0] if layout == "c" else a[:, 0])
    assert out.flags.c_contiguous and out.flags.writeable
    before_a, before_b = a.copy(), b.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIASES_INPUT"):
        op.matvec(a, b, out=out, simd=simd)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("view", ["row", "column", "reverse_row", "broadcast"])
def test_matvec_allows_readonly_matrix_and_vector_to_share_storage(layout, view):
    a = positive_matrix(128, layout)
    if view == "row":
        b = a[3]
    elif view == "column":
        b = a[:, 3]
    elif view == "reverse_row":
        b = a[3, ::-1]
    else:
        b = np.broadcast_to(a[3, :1], (128,))
    assert np.shares_memory(a, b)
    a.flags.writeable = b.flags.writeable = False
    before_a, before_b = a.copy(), b.copy()
    expected = op.matvec(a, b, simd="scalar")
    actual, audit = op.matvec(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, True)
    assert_no_workspace(audit)
    assert audit["algorithm_copy_bytes"] == 0
    assert not np.shares_memory(actual, a) and not np.shares_memory(actual, b)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("layout", ["reverse_rows", "reverse_columns", "column_slice", "broadcast_rows"])
def test_matvec_nonadmitted_matrix_layouts_keep_canonical_fallback(layout):
    a = np.arange(128 * 9, dtype=np.float64).reshape(128, 9) / 100
    if layout == "reverse_rows":
        a = a[::-1]
    elif layout == "reverse_columns":
        a = a[:, ::-1]
    elif layout == "column_slice":
        a = np.repeat(a, 2, axis=1)[:, ::2]
    else:
        a = np.broadcast_to(a[0], a.shape)
    b = np.linspace(-2, 2, 9)[::-1]
    a.flags.writeable = b.flags.writeable = False
    expected = op.matvec(a, b, simd="scalar")
    actual, audit = op.matvec(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, False)
    assert_no_workspace(audit)
    assert audit["algorithm_copy_bytes"] == 0


@pytest.mark.parametrize("layout", ["c", "f"])
def test_matvec_does_not_consume_padding_or_previous_output(layout):
    expected_matrix = positive_matrix(128)
    if layout == "c":
        storage = np.full((128, 131), np.nan)
        a = storage[:, :128]
    else:
        storage = np.full((131, 128), np.nan, order="F")
        a = storage[:128, :]
    a[:] = expected_matrix
    a.flags.writeable = False
    b = np.linspace(-1, 1, 128)
    out = np.full(128, np.nan)
    actual, audit = op.matvec(a, b, out=out, audit=True)
    assert_library_calls(audit, layout in ("c", "f"))
    assert_no_workspace(audit)
    assert audit["algorithm_copy_bytes"] == 0
    assert np.shares_memory(actual, out)
    assert_ieee_equal(actual, op.matvec(a, b, simd="scalar"))
    np.testing.assert_allclose(out, expected_matrix @ b, rtol=1e-12, atol=1e-12)
    padding = storage[:, 128:] if layout == "c" else storage[128:, :]
    assert np.all(np.isnan(padding))


@pytest.mark.parametrize("near_boundary", [False, True])
def test_cholesky_overwrites_caller_output(near_boundary):
    a = np.eye(128) if near_boundary else positive_matrix(128)
    if near_boundary:
        a[-1, -1] = 1e-24
    out = np.full((128, 128), np.nan)
    actual, audit = op.cholesky(a, out=out, audit=True)
    assert_library_calls(audit, False)
    assert_no_workspace(audit)
    assert np.shares_memory(actual, out)
    np.testing.assert_allclose(out @ out.T, a, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(out, np.tril(out))
    assert audit["algorithm_copy_bytes"] == 128 * 129 // 2 * 8


@pytest.mark.parametrize(
    "fault", ["nan", "nan_zero", "inf", "overflow", "overflow_cancel", "tiny", "cancellation"]
)
@pytest.mark.parametrize("large_f", [False, True])
def test_matvec_extremes_preserve_per_row_ieee_results(fault, large_f):
    a = np.zeros((4096, 32), order="F") if large_f else np.zeros((128, 128))
    b = np.ones(a.shape[1])
    a[:, 0] = 1
    if fault in ("nan", "nan_zero"):
        a[9, 5] = np.nan
        if fault == "nan_zero":
            b[5] = 0
    elif fault == "inf":
        a[9, 5] = np.inf
        b[5] = 0
    elif fault == "overflow":
        a[9, :3] = [1e308, 1e308, -1e308]
    elif fault == "overflow_cancel":
        a[9, :2] = [np.finfo(float).max, -np.finfo(float).max]
        b[:2] = 2
    elif fault == "tiny":
        a *= 1e-300
        b *= 1e-20
    else:
        a[9, :3] = [1e200, 1, -1e200]
    expected = op.matvec(a, b, simd="scalar")
    actual, audit = op.matvec(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, True)
    assert_no_workspace(audit)


@pytest.mark.parametrize("fault", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("stride", [-1, 0, 2])
@pytest.mark.parametrize("order", ["C", "F"])
def test_matvec_nonfinite_vector_is_consumed_in_original_order(fault, stride, order):
    a, _ = large_matvec_inputs(4097, 8)
    a = np.array(a, order=order, copy=True)
    if stride == 0:
        b = np.broadcast_to(np.array(fault), (8,))
    else:
        logical = np.ones(8)
        logical[3] = fault
        b = logical[::-1] if stride == -1 else np.repeat(logical, 2)[::2]
    a[::3] = 0.0
    a[1::3] = -0.0
    expected = op.matvec(a, b, simd="scalar")
    actual, audit = op.matvec(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, True)
    assert_no_workspace(audit)
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0


@pytest.mark.parametrize("order", ["C", "F"])
def test_matvec_keeps_separate_product_and_addition_at_block_tail(order):
    # The exact product is 1 - 2**-54, which separately rounds to 1. FMA
    # would leave -2**-54 after adding -1 and alter downstream comparisons.
    a = np.zeros((4097, 8), order=order)
    a[:, 0] = -1
    a[:, 1] = 1 + 2.0**-27
    b = np.ones(8)
    b[1] = 1 - 2.0**-27
    expected = op.matvec(a, b, simd="scalar")
    np.testing.assert_array_equal(expected, np.zeros(4097))
    actual, audit = op.matvec(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    assert_library_calls(audit, True)
    assert_no_workspace(audit)


def cancellation_matrix(layout):
    rows, columns = (4097, 32) if layout == "large_c" else (
        (4096, 32) if layout == "large_f" else (128, 128)
    )
    a = np.zeros((rows, columns))
    a[0, :3] = [1e16, 1, -1e16]
    a[1, :3] = [1e16, -1e16, 1]
    a[2, :3] = [-1e16, -1, 1e16]
    a[3, :3] = [-1e16, 1e16, -1]
    a[4, :3] = [1e-200, 1e-220, -1e-200]
    a[5, :3] = [1e-200, -1e-200, 1e-220]
    a[6, :3] = [np.nextafter(0.0, 1.0), 0, 0]
    a[7, :3] = [-np.nextafter(0.0, 1.0), 0, 0]
    a[8, :] = -0.0
    a[9:, 0] = np.arange(rows - 9) + 2
    # A large nonzero result can still hide lost unit contributions. The
    # previous near-zero-only guard missed this order-sensitive reduction.
    a[10, :] = 1
    a[10, 0] = 1e16
    a[10, -1] = -1e16 + 1e11
    return layout_array(a, "f" if layout == "large_f" else layout)


@pytest.mark.parametrize("layout", ["c", "f", "large_c", "large_f"])
def test_matvec_cancellation_keeps_zero_sign_and_tiny_results(layout):
    a = cancellation_matrix(layout)
    b = np.ones(a.shape[1])
    before = a.copy()
    a.flags.writeable = b.flags.writeable = False
    expected = op.matvec(a, b, simd="scalar")
    # The two orders intentionally have different canonical results. Ordinary
    # allclose would miss a 0/1 discrepancy beside the 1e16 input magnitudes.
    np.testing.assert_array_equal(expected[:4], [0, 1, 0, -1])
    assert expected[10] == 1e11
    actual, audit = op.matvec(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    assert_library_calls(audit, True)
    assert_no_workspace(audit)
    assert audit["algorithm_copy_bytes"] == 0
    np.testing.assert_array_equal(a, before)


@pytest.mark.parametrize("layout", ["c", "f", "large_c", "large_f"])
def test_matvec_subnormal_product_sums_keep_classification(layout):
    rng = np.random.default_rng(962)
    shape = (4097, 32) if layout == "large_c" else (
        (4096, 32) if layout == "large_f" else (128, 128)
    )
    a = rng.integers(-2, 3, shape).astype(float) * 1e-162
    a[0] = 0.0
    a[1] = -0.0
    a = layout_array(a, "f" if layout == "large_f" else layout)
    b = rng.integers(-2, 3, a.shape[1]).astype(float) * 1e-162
    expected = op.matvec(a, b, simd="scalar")
    actual, audit = op.matvec(a, b, audit=True)
    assert_ieee_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_array_equal(actual > 0, expected > 0)
    assert_library_calls(audit, True)
    assert_no_workspace(audit)


@pytest.mark.parametrize("layout", ["c", "f", "large_c", "large_f"])
@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_matvec_cancellation_boolean_downstream_survives_isolation(layout, lane):
    a = cancellation_matrix(layout)
    b = np.ones(a.shape[1])
    expected = op.matvec(a, b, simd="scalar")
    _, direct = op.matvec(a, b, audit=True)
    graph = GraphCompiler(
        {
            "a": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["N", "D"]},
            "b": {"kind": "vector", "axes": ["asset"], "shape": ["D"]},
            "bad": matrix(symbol="B"),
        }
    ).compile(
        [
            "matvec(a,b)", "greater_than(matvec(a,b),0)",
            "greater_than(matvec(a,b),100000000000)", "cholesky(bad)", "7",
        ],
        error_policy="isolate",
    )
    result = run(graph, {"a": a, "b": b, "bad": np.zeros((2, 2))}, lane)
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    for row in range(2):
        assert_ieee_equal(result.outputs[0].values[row], expected)
        np.testing.assert_array_equal(result.outputs[1].values[row], expected > 0)
        np.testing.assert_array_equal(result.outputs[2].values[row], expected > 1e11)
        assert result.outputs[1].values[row].dtype == np.bool_
    assert np.all(result.outputs[0].root_statuses == 0)
    assert np.all(result.outputs[1].root_statuses == 0)
    assert np.all(result.outputs[2].root_statuses == 0)
    assert np.all(result.outputs[3].root_statuses != 0)
    assert np.all(np.asarray(result.outputs[4].values) == 7)
    calls = sum(chunk["eigen_matvec_calls"] for chunk in result.audit["native_chunks"])
    # Input transport may materialize F arrays in C order. Both admitted
    # layouts now keep the same ordered library calculation inside workers.
    assert calls == 2 * direct["eigen_matvec_calls"]


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_worker_local_transpose_preserves_nonzero_library_audit(lane):
    large, b = large_matvec_inputs(4096, 64)
    a = large.T
    assert a.flags.c_contiguous
    expected = op.matvec(a.T, b, simd="scalar")
    _, direct = op.matvec(a.T, b, audit=True)
    assert_library_calls(direct, True)
    a.flags.writeable = b.flags.writeable = False
    graph = GraphCompiler(
        {
            "a": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["K", "D"]},
            "b": {"kind": "vector", "axes": ["asset"], "shape": ["K"]},
        }
    ).compile(["matvec(transpose(a),b)", "sum(matvec(transpose(a),b))"])
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(
            graph, {"a": a, "b": b}, np.zeros(2, np.int64), np.full(2, 64, np.int64)
        )
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    calls = sum(chunk["eigen_matvec_calls"] for chunk in result.audit["native_chunks"])
    assert calls == 2 * direct["eigen_matvec_calls"]
    for row in range(2):
        actual = result.outputs[0].values[row]
        assert_ieee_equal(actual, expected)
        assert not np.shares_memory(actual, a) and not actual.flags.writeable


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_nonfinite_rows_and_boolean_consumers_survive_worker_transport(lane):
    columns = 8
    a = np.zeros((columns, 4097))
    a[0] = 1
    a[1, 0] = np.nan
    a[1, 1] = np.inf
    a[1, 2] = -np.inf
    a[1, 4096] = np.nan
    b = np.ones(columns)
    expected = op.matvec(a.T, b, simd="scalar")
    _, direct = op.matvec(a.T, b, audit=True)
    assert_library_calls(direct, True)
    graph = GraphCompiler(
        {
            "a": {"kind": "matrix", "axes": ["asset", "asset"], "shape": ["K", "D"]},
            "b": {"kind": "vector", "axes": ["asset"], "shape": ["K"]},
            "bad": matrix(symbol="B"),
        }
    ).compile(
        ["matvec(transpose(a),b)", "greater_than(matvec(transpose(a),b),0)",
         "cholesky(bad)", "7"],
        error_policy="isolate",
    )
    # Typed isolate materialization records nonfinite numeric positions as
    # status 4, while comparisons retain their existing IEEE boolean values.
    # A worker-local column slice keeps stride two after process transport, so
    # this independent status oracle cannot enter either ordered Eigen path.
    padded = np.full((a.shape[1], columns * 2), np.nan)
    padded[:, ::2] = a.T
    reference_product = f"matvec(slice_axis(padded,1,0,{columns * 2},2),b)"
    reference_graph = GraphCompiler(
        {
            "padded": {
                "kind": "matrix", "axes": ["asset", "asset"],
                "shape": ["D", str(columns * 2)],
            },
            # slice_axis infers a literal extent, so its consumer must declare
            # the same literal rather than an unrelated symbolic dimension.
            "b": {"kind": "vector", "axes": ["asset"], "shape": [str(columns)]},
            "bad": matrix(symbol="B"),
        }
    ).compile(
        [reference_product, f"greater_than({reference_product},0)", "cholesky(bad)", "7"],
        error_policy="isolate",
    )
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(
            graph, {"a": a, "b": b, "bad": np.zeros((2, 2))},
            np.zeros(2, np.int64), np.full(2, columns, np.int64),
        )
        reference = scheduler.execute(
            reference_graph,
            {"padded": padded, "b": b, "bad": np.zeros((2, 2))},
            np.zeros(2, np.int64), np.full(2, columns, np.int64),
        )
    for row in range(2):
        assert_ieee_equal(result.outputs[0].values[row], expected)
        np.testing.assert_array_equal(result.outputs[1].values[row], expected > 0)
    for output, original in zip(result.outputs, reference.outputs, strict=True):
        np.testing.assert_array_equal(output.root_statuses, original.root_statuses)
        for row in range(2):
            np.testing.assert_array_equal(output.statuses[row], original.statuses[row])
    assert np.all(result.outputs[2].root_statuses != 0)
    assert np.all(result.outputs[3].root_statuses == 0)
    assert np.all(np.asarray(result.outputs[3].values) == 7)
    calls = sum(chunk["eigen_matvec_calls"] for chunk in result.audit["native_chunks"])
    assert calls == 2 * direct["eigen_matvec_calls"]
    assert sum(chunk["eigen_matvec_calls"] for chunk in reference.audit["native_chunks"]) == 0


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_library_call_audit_cse_transport_and_isolation(lane):
    a, b = positive_matrix(128), np.linspace(-1, 1, 128)
    lower, cholesky_audit = op.cholesky(a, audit=True)
    product, matvec_audit = op.matvec(a, b, audit=True)
    declarations = {
        "a": matrix(),
        "b": {"kind": "vector", "axes": ["asset"], "shape": ["N"]},
    }
    graph = GraphCompiler(declarations).compile(
        ["cholesky(a)", "trace(cholesky(a))", "matvec(a,b)", "sum(matvec(a,b))"]
    )
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    result = run(graph, {"a": a, "b": b}, lane)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    calls = sum(chunk["eigen_matvec_calls"] for chunk in result.audit["native_chunks"])
    assert cholesky_audit["eigen_matvec_calls"] == 0
    assert calls == 2 * matvec_audit["eigen_matvec_calls"]
    for row in range(2):
        np.testing.assert_allclose(result.outputs[0].values[row], lower, atol=1e-12)
        np.testing.assert_allclose(result.outputs[1].values[row], np.trace(lower), atol=1e-12)
        assert_ieee_equal(result.outputs[2].values[row], product)
    a[-1, -1] = -1
    isolated = GraphCompiler(declarations).compile(
        ["cholesky(a)", "matvec(a,b)", "7"], error_policy="isolate"
    )
    failed = run(isolated, {"a": a, "b": b}, lane)
    assert np.all(failed.outputs[0].root_statuses != 0)
    assert np.all(failed.outputs[1].root_statuses == 0)
    assert np.all(np.asarray(failed.outputs[2].values) == 7)
    np.testing.assert_array_equal(result.outputs[0].values[0], lower)


def test_prepared_snapshot_rereads_inputs_and_keeps_prior_results():
    a, b = positive_matrix(128), np.linspace(0, 1, 128)
    graph = GraphCompiler(
        {"a": matrix(), "b": {"kind": "vector", "axes": ["asset"], "shape": ["N"]}}
    ).compile(["cholesky(a)", "matvec(a,b)"])
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
        first = prepared.run_snapshot()
        saved = [out.values[0].copy() for out in first.outputs]
        a *= 4
        b *= 2
        second = prepared.run_snapshot()
        np.testing.assert_allclose(second.outputs[0].values[0], saved[0] * 2, atol=1e-12)
        np.testing.assert_allclose(second.outputs[1].values[0], saved[1] * 8, atol=1e-12)
        for output, snapshot in zip(first.outputs, saved, strict=True):
            np.testing.assert_array_equal(output.values[0], snapshot)
        plan = scheduler.plan(graph, {"a": a, "b": b}, bounds, bounds)
        with pytest.raises(MemoryError):
            scheduler.plan(
                graph, {"a": a, "b": b}, bounds, bounds,
                memory_budget_bytes=plan.estimated_total_memory_bytes - 1,
            )


@pytest.mark.parametrize("order", ["C", "F"])
def test_large_matvec_prepared_snapshot_rereads_inputs_and_retains_output(order):
    a, b = large_matvec_inputs(4097, 32)
    a = np.array(a, order=order, copy=True)
    _, direct = op.matvec(a, b, audit=True)
    assert_library_calls(direct, True)
    graph = GraphCompiler(rectangular_declarations()).compile("matvec(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        assert_ieee_equal(saved, op.matvec(a, b, simd="scalar"))
        first_calls = sum(c["eigen_matvec_calls"] for c in first.audit["native_chunks"])
        assert first_calls == direct["eigen_matvec_calls"]
        a *= 4
        b *= 2
        second = prepared.run_snapshot()
        np.testing.assert_allclose(second.outputs[0].values[0], saved * 8, rtol=1e-12, atol=1e-12)
        assert_ieee_equal(second.outputs[0].values[0], op.matvec(a, b, simd="scalar"))
        np.testing.assert_array_equal(first.outputs[0].values[0], saved)
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
        second_calls = sum(c["eigen_matvec_calls"] for c in second.audit["native_chunks"])
        assert second_calls == direct["eigen_matvec_calls"]
    for result in (first, second):
        actual = result.outputs[0].values[0]
        assert not actual.flags.writeable
        assert not np.shares_memory(actual, a) and not np.shares_memory(actual, b)
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_c_matvec_worker_results_survive_reexecution_and_scheduler_close(lane):
    a, b = large_matvec_inputs(4097, 32)
    a = np.ascontiguousarray(a)
    next_b = b * 2
    a.flags.writeable = b.flags.writeable = next_b.flags.writeable = False
    expected = op.matvec(a, b, simd="scalar")
    next_expected = op.matvec(a, next_b, simd="scalar")
    _, direct = op.matvec(a, b, audit=True)
    assert_library_calls(direct, True)
    graph = GraphCompiler(rectangular_declarations()).compile(
        ["matvec(a,b)", "sum(matvec(a,b))"]
    )
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        retained = tuple(first.outputs[0].values)
        second = scheduler.execute(graph, {"a": a, "b": next_b}, bounds, bounds)
        assert first.plan.lane == second.plan.lane == (
            "process" if lane.startswith("process") else lane
        )
        for result in (first, second):
            calls = sum(c["eigen_matvec_calls"] for c in result.audit["native_chunks"])
            assert calls == 2 * direct["eigen_matvec_calls"]
    # The scheduler and worker owners are gone. Retained arrays must still
    # own independent, immutable results, including the R1 tail at row 4096.
    for row, old in enumerate(retained):
        new = second.outputs[0].values[row]
        assert_ieee_equal(old, expected)
        assert_ieee_equal(new, next_expected)
        assert not old.flags.writeable and not new.flags.writeable
        assert not np.shares_memory(old, new)
        for actual in (old, new):
            for source in (a, b, next_b):
                assert not np.shares_memory(actual, source)
    assert not np.shares_memory(retained[0], retained[1])


def test_nested_iteration_preserves_library_calls():
    a, b = large_matvec_inputs()
    expected, direct = op.matvec(a, b, audit=True)
    assert_library_calls(direct, True)
    declarations = rectangular_declarations()
    declarations["seed"] = {"kind": "vector", "axes": ["asset"], "shape": ["N"]}
    graph = GraphCompiler(declarations).compile(
        "iterate(matvec(a,b)+iterate_x*0,seed,1e-12,2)"
    )
    result = run(graph, {"a": a, "b": b, "seed": np.zeros(len(a))})
    calls = sum(c["eigen_matvec_calls"] for c in result.audit["native_chunks"])
    assert calls == 4 * direct["eigen_matvec_calls"]
    np.testing.assert_allclose(result.outputs[0].values[0], expected, atol=1e-12)


@pytest.mark.skipif(
    platform.system() != "Darwin" or int(platform.mac_ver()[0].split(".")[0] or 0) < 15,
    reason="Apple thread-local BLAS API requires macOS 15",
)
def test_matvec_does_not_change_existing_blas_thread_policy():
    lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    lib.BLASGetThreading.restype = ctypes.c_uint
    lib.BLASSetThreading.argtypes = [ctypes.c_uint]
    lib.BLASSetThreading.restype = ctypes.c_int

    def check_thread():
        original = lib.BLASGetThreading()
        a = positive_matrix(128)
        large, b = large_matvec_inputs()
        try:
            for policy in (0, 1):
                assert lib.BLASSetThreading(policy) == 0
                _, audit = op.cholesky(a, audit=True)
                assert_library_calls(audit, False)
                assert lib.BLASGetThreading() == policy
                _, audit = op.matvec(large, b, audit=True)
                assert_library_calls(audit, True)
                assert lib.BLASGetThreading() == policy
                near = np.eye(128)
                near[-1, -1] = 1e-24
                op.cholesky(near)
                assert lib.BLASGetThreading() == policy
                near[-1, -1] = -1
                with pytest.raises(ValueError, match="DOMAIN_ERROR"):
                    op.cholesky(near)
                assert lib.BLASGetThreading() == policy
                near[0, 0] = np.nan
                with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
                    op.cholesky(near)
                assert lib.BLASGetThreading() == policy
        finally:
            assert lib.BLASSetThreading(original) == 0

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: check_thread(), range(3)))
