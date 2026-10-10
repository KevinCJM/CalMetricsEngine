"""C/F matmul and repeated rows: original tolerance, IEEE and result ownership."""

import gc
import os

import numpy as np
import pytest

from calmetrics_engine import (
    AdaptiveScheduler,
    GraphCompiler,
    PlannerConfig,
    SharedInputBundle,
)
from calmetrics_engine import operators as op

from .test_matrix_library_backend import PACKING_PLATFORM

RTOL, ATOL = 2e-12, 2e-14  # Existing test_operator_contracts.py MatMul tolerance.


def settings(lane):
    return PlannerConfig(
        thread_work_units=1e100 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )


def declarations(*, transposed_rhs=False):
    # Static asset axes avoid accidentally slicing a matrix by time intervals.
    return {
        "a": dict(kind="matrix", axes=["asset", "asset"], shape=["M", "K"]),
        ("bt" if transposed_rhs else "b"): dict(
            kind="matrix",
            axes=["asset", "asset"],
            shape=["N", "K"] if transposed_rhs else ["K", "N"],
        ),
    }


def inputs(shape=(129, 65, 33), layout="cf"):
    m, k, n = shape
    rng = np.random.default_rng(764391)
    a, b = rng.normal(size=(m, k)), rng.normal(size=(k, n))
    owners = []
    if layout in {"fc", "ff"}:
        a = np.asfortranarray(a)
    if layout in {"cf", "ff"}:
        b = np.asfortranarray(b)
    elif layout == "padded_f_b":
        owner = np.full((k + 3, n), np.nan, order="F")
        owner[:k] = b
        b = owner[:k]
        owners.append(owner)
    elif layout == "padded_c_b":
        owner = np.full((k, n + 3), np.nan)
        owner[:, :n] = b
        b = owner[:, :n]
        owners.append(owner)
    elif layout == "general":
        aa, bb = np.full((2 * m, 2 * k), np.nan), np.full((2 * k, 2 * n), np.nan)
        aa[::2, ::2], bb[::2, ::2] = a, b
        a, b = aa[::2, ::2], bb[::2, ::2]
        owners.extend([aa, bb])
    elif layout == "negative":
        a, b = a[::-1, ::-1], b[::-1, ::-1]
    elif layout == "zero_b":
        b = np.broadcast_to(b[:, :1], (k, n))
    return a, b, owners


def assert_ieee_classes_and_close(actual, expected):
    """Exact observable classes and zero signs, existing tolerance otherwise."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_array_equal(actual == 0, expected == 0)
    zero = expected == 0
    np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "shape,layout",
    [
        ((129, 65, 33), "cf"),
        ((129, 65, 33), "ff"),
        ((33, 35, 37), "cc"),
        ((36, 33, 40), "cc"),
        ((35, 37, 39), "cc"),
        ((33, 35, 37), "fc"),
        ((33, 35, 40), "fc"),
        ((40, 40, 40), "fc"),
        ((33, 35, 37), "padded_c_b"),
        ((31, 32, 32), "cc"),
        ((32, 31, 32), "cc"),
        ((32, 32, 31), "cc"),
        ((32, 32, 32), "cc"),
        ((33, 35, 37), "padded_f_b"),
        ((31, 32, 32), "cf"),
        ((32, 31, 32), "cf"),
        ((32, 32, 31), "cf"),
        ((32, 32, 32), "cf"),
        ((33, 33, 33), "cf"),
        ((9, 33, 17), "general"),
        ((9, 33, 17), "negative"),
        ((9, 33, 17), "zero_b"),
        ((1, 7, 1), "cf"),
        ((7, 0, 3), "cf"),
    ],
)
def test_matmul_layouts_keep_independent_reference_and_readonly_inputs(shape, layout):
    a, b, owners = inputs(shape, layout)
    before = [x.copy() for x in [a, b, *owners]]
    a.flags.writeable = b.flags.writeable = False
    expected = a @ b
    work = op.Workspace()
    actual, audit = op.matmul(a, b, workspace=work, audit=True)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    eligible = min(shape) >= 32 and (
        (b.strides[0] == 8 and b.strides[1] >= shape[1] * 8)
        or (
            b.strides[1] == 8
            and b.strides[0] == shape[2] * 8
            and shape[2] % 8 == 0
            and shape[0] == shape[1] == shape[2]
        )
    )
    assert_dispatch(audit, eligible)
    assert actual.shape == (shape[0], shape[2])
    assert actual.dtype == np.float64
    assert not np.shares_memory(actual, a) and not np.shares_memory(actual, b)
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0
    for array, saved in zip([a, b, *owners], before, strict=True):
        np.testing.assert_array_equal(array, saved)
    if shape[1] == 0:
        np.testing.assert_array_equal(actual, 0.0)
        assert not np.signbit(actual).any()


@pytest.mark.parametrize("right_order", ["C", "F"])
@pytest.mark.parametrize("fault", ["mixed_rows", "zero_times_inf"])
def test_matmul_nonfinite_and_tiny_terms_remain_observable(fault, right_order):
    rows, inner, columns = (40, 40, 40) if right_order == "C" else (129, 65, 33)
    a = np.zeros((rows, inner))
    b = np.ones((inner, columns), order=right_order)
    if fault == "zero_times_inf":
        b[7] = np.inf
        # Every row consumes 0*Inf. Skipping zero coefficients is invalid.
    else:
        tiny = np.nextafter(0.0, 1.0)
        a[0] = -0.0
        a[1, 0], a[2, 0] = tiny, -tiny
        a[3, :3] = [1e308, 1e308, -1e308]
        a[4, :2] = [np.finfo(float).max, -np.finfo(float).max]
        b[:2] = 2.0
        a[5, 5], b[5] = np.nan, 0.0
        a[6, 6], a[7, 6] = np.inf, -np.inf
    a.flags.writeable = b.flags.writeable = False
    before_a, before_b = a.copy(), b.copy()
    expected = op.matmul(a, b, simd="scalar")
    actual, audit = op.matmul(a, b, audit=True)
    assert_dispatch(audit, True)
    assert_ieee_classes_and_close(actual, expected)
    if fault == "mixed_rows":
        # Tail tolerance must not make underflow-to-zero look acceptable.
        np.testing.assert_array_equal(actual[1:3] > 0, expected[1:3] > 0)
        np.testing.assert_array_equal(actual[1:3] < 0, expected[1:3] < 0)
    else:
        assert np.isnan(actual).all()
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("right_order", ["C", "F"])
def test_matmul_shared_readonly_operands_and_workspace_reuse_keep_old_result(right_order):
    size = 128 if right_order == "C" else 129
    owner = np.random.default_rng(727).normal(size=(size, size))
    owner.flags.writeable = False
    a, b = (owner.T, owner) if right_order == "C" else (owner, owner.T)
    assert np.shares_memory(a, b)
    before = owner.copy()
    workspace = op.Workspace()
    first, audit = op.matmul(a, b, workspace=workspace, audit=True)
    assert_dispatch(audit, True, packed=right_order == "C")
    saved = first.copy()
    np.testing.assert_allclose(first, a @ b, rtol=RTOL, atol=ATOL)
    second = op.matmul(a * 2, b, workspace=workspace)
    np.testing.assert_allclose(second, saved * 2, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(first, saved)
    np.testing.assert_array_equal(owner, before)
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, owner)
    bad = before.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIASES_INPUT"):
        lhs, rhs = (bad.T, bad) if right_order == "C" else (bad, bad.T)
        op.matmul(lhs, rhs, out=bad)
    np.testing.assert_array_equal(bad, before)


@pytest.mark.parametrize("layout", ["cc", "cf"])
@pytest.mark.parametrize("repeated", [False, True])
def test_matmul_prepared_independent_snapshots_refresh_bound_values(layout, repeated):
    shape = (40, 40, 40) if layout == "cc" and not repeated else (129, 65, 33)
    a, b, _ = inputs(shape, layout)
    row_owner = a[:1]
    if repeated:
        a = np.broadcast_to(row_owner, a.shape)
    graph = GraphCompiler(declarations()).compile(["matmul(a,b)", "b"])
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
        expected = a @ b
        first = prepared.run_snapshot()
        saved = [out.values[0].copy() for out in first.outputs]
        np.testing.assert_allclose(saved[0], expected, rtol=RTOL, atol=ATOL)
        b *= 2.0  # Same pointer/shape/strides; a numeric cache would be stale.
        if repeated:
            row_owner += 0.25
        expected = a @ b
        second = prepared.run_snapshot()
        np.testing.assert_allclose(second.outputs[0].values[0], expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(second.outputs[1].values[0], b)
    for old, new, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
        np.testing.assert_array_equal(old.values[0], snapshot)
        assert not old.values[0].flags.writeable and not new.values[0].flags.writeable
        assert not np.shares_memory(old.values[0], new.values[0])
        assert not np.shares_memory(new.values[0], a)
        assert not np.shares_memory(new.values[0], b)


def boundary_inputs(pattern, *, square=False):
    rows, inner, columns = (40, 40, 40) if square else (129, 65, 33)
    a = np.zeros((rows, inner))
    b = np.zeros((inner, columns), order="F")
    if pattern == "fma":
        a[:, :2] = [-1.0, 1.0 + 2.0**-27]
        b[:2] = np.array([1.0, 1.0 - 2.0**-27])[:, None]
    else:
        a[:, :3] = [1.0, 2.0**-54, -1.0]
        b[:3] = 1.0
    # Transfer ordinary C storage, then transpose in the worker. Passing a
    # host F array alone does not prove the process kernel sees F storage.
    bt = np.array(b.T, order="C", copy=True)
    a.flags.writeable = bt.flags.writeable = False
    return a, bt


@pytest.mark.parametrize("pattern", ["fma", "split_accumulators"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("right_order", ["C", "F"])
def test_matmul_worker_local_rhs_keeps_comparison_and_error_boundary(pattern, lane, right_order):
    a, bt = boundary_inputs(pattern, square=right_order == "C")
    b = np.array(bt.T, order="C", copy=True) if right_order == "C" else bt
    b.flags.writeable = False
    expression = "matmul(a,b)" if right_order == "C" else "matmul(a,transpose(bt))"
    bound = {"a": a, "b" if right_order == "C" else "bt": b}
    graph = GraphCompiler(declarations(transposed_rhs=right_order == "F")).compile(
        [
            expression,
            f"greater_than({expression},0)",
            f"1/sum({expression})",
            f"greater_than(1/sum({expression}),0)",
            "7",
        ],
        error_policy="isolate",
    )
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    # These specific analytical fixtures produce exact zeros in the current
    # arithmetic. This is a discrete-boundary check, not a general bitwise rule.
    np.testing.assert_array_equal(op.matmul(a, bt.T, simd="scalar"), 0.0)
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, bound, bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    if lane.startswith("process"):
        assert result.plan.use_shared_memory == (lane == "process_shared")
    for row in range(2):
        matrix, comparison, broken, dependent, healthy = result.outputs
        np.testing.assert_array_equal(matrix.values[row], 0.0)
        np.testing.assert_array_equal(comparison.values[row], False)
        assert comparison.values[row].dtype == np.bool_
        assert matrix.root_statuses[row] == comparison.root_statuses[row] == 0
        assert broken.root_statuses[row] != 0
        assert dependent.root_statuses[row] == broken.root_statuses[row]
        assert healthy.root_statuses[row] == 0 and healthy.values[row] == 7
        for output in (matrix, comparison):
            assert not output.values[row].flags.writeable
            assert not np.shares_memory(output.values[row], a)
            assert not np.shares_memory(output.values[row], b)
    if lane == "process_shared":
        assert result.audit["output_copy_bytes"] == 0


def test_matmul_shared_bundle_release_keeps_independent_process_result():
    a, b, _ = inputs()
    bt = np.array(b.T, order="C", copy=True)
    expected = a @ b
    graph = GraphCompiler(declarations(transposed_rhs=True)).compile("matmul(a,transpose(bt))")
    bounds = np.zeros(2, np.int64)
    with SharedInputBundle.from_inputs({"a": a, "bt": bt}) as bundle:
        for array in bundle.arrays.values():
            assert not array.flags.writeable
        with AdaptiveScheduler(cpu_budget=2, config=settings("process_shared")) as scheduler:
            result = scheduler.execute(graph, bundle, bounds, bounds)
        assert result.plan.lane == "process" and result.plan.use_shared_memory
    gc.collect()
    for value in result.outputs[0].values:
        np.testing.assert_allclose(value, expected, rtol=RTOL, atol=ATOL)
        assert not value.flags.writeable
        assert not np.shares_memory(value, a) and not np.shares_memory(value, bt)
    assert result.audit["output_copy_bytes"] == 0


def assert_dispatch(audit, eligible, *, repeated=False, packed=False):
    enabled = os.environ.get("CME_EXPECT_EIGEN")
    library_names = {"eigen-array-matmul", "matmul-repeated-rows", "eigen-packed-matmul"}
    expected = (
        "eigen-packed-matmul"
        if packed
        else "matmul-repeated-rows"
        if repeated
        else "eigen-array-matmul"
    )
    if not eligible or not PACKING_PLATFORM or enabled == "0":
        assert audit["isa"] not in library_names
    elif enabled == "1":
        assert audit["isa"] == expected
    else:
        assert audit["isa"] in {"scalar", *op.available_simd(), expected}
    return audit["isa"] == expected


@pytest.mark.parametrize("layout", ["cc", "cf"])
@pytest.mark.parametrize("repeated", [False, True])
def test_matmul_explicit_isa_and_auto_dispatch(layout, repeated):
    shape = (40, 40, 40) if layout == "cc" and not repeated else (33, 35, 37)
    a, b, _ = inputs(shape, layout)
    if repeated:
        a = np.broadcast_to(a[:1], a.shape)
    expected = a @ b
    for isa in ["auto", *op.available_simd()]:
        actual, audit = op.matmul(a, b, simd=isa, audit=True)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
        dispatched = assert_dispatch(audit, isa == "auto", repeated=repeated)
        expected_copy = (a.shape[0] - 1) * b.shape[1] * 8 if repeated and dispatched else 0
        assert audit["algorithm_copy_bytes"] == expected_copy
    if PACKING_PLATFORM:
        with pytest.raises(ValueError, match="UNSUPPORTED_ISA"):
            op.matmul(a, b, simd="avx2")


@pytest.mark.parametrize("right_layout", ["cc", "cf"])
@pytest.mark.parametrize(
    "layout", ["negative_columns", "negative_rows", "sliced_columns", "broadcast_rows"]
)
def test_matmul_library_borrows_strided_left_operand(layout, right_layout):
    shape = (40, 40, 40) if right_layout == "cc" and layout != "broadcast_rows" else (33, 35, 37)
    a, b, _ = inputs(shape, right_layout)
    if layout == "negative_columns":
        a = a[:, ::-1]
    elif layout == "negative_rows":
        a = a[::-1]
    elif layout == "sliced_columns":
        owner = np.full((a.shape[0], a.shape[1] * 2), np.nan)
        owner[:, ::2] = a
        a = owner[:, ::2]
    else:
        a = np.broadcast_to(a[:1], a.shape)
    a.flags.writeable = b.flags.writeable = False
    before_a, before_b = a.copy(), b.copy()
    actual, audit = op.matmul(a, b, audit=True)
    dispatched = assert_dispatch(audit, True, repeated=layout == "broadcast_rows")
    np.testing.assert_allclose(actual, a @ b, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)
    expected_copy = (
        (a.shape[0] - 1) * b.shape[1] * 8 if layout == "broadcast_rows" and dispatched else 0
    )
    assert audit["input_copy_bytes"] == 0
    assert audit["algorithm_copy_bytes"] == expected_copy
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0


@pytest.mark.parametrize("layout", ["cc", "cf"])
def test_matmul_repeated_rows_copy_to_independent_storage_and_refresh_inputs(layout):
    _, b, _ = inputs((33, 35, 37), layout)
    owner = np.random.default_rng(8307).normal(size=(1, 35))
    row = owner[:, ::-1]
    a = np.broadcast_to(row, (33, 35))
    workspace = op.Workspace()
    first, audit = op.matmul(a, b, workspace=workspace, audit=True)
    dispatched = assert_dispatch(audit, True, repeated=True)
    expected = np.repeat(row @ b, a.shape[0], axis=0)
    np.testing.assert_allclose(first, expected, rtol=RTOL, atol=ATOL)
    assert first.flags.c_contiguous
    assert first.strides == (b.shape[1] * 8, 8)
    assert not np.shares_memory(first, owner) and not np.shares_memory(first, b)
    assert audit["input_copy_bytes"] == 0
    assert audit["algorithm_copy_bytes"] == ((a.shape[0] - 1) * b.shape[1] * 8 if dispatched else 0)
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0

    # Identical values must not be returned as a zero-stride view of one row.
    unchanged = first[0, 0]
    first[1, 0] += 1.0
    assert first[0, 0] == unchanged
    saved = first.copy()
    owner *= 2.0
    b += 0.125
    before_a, before_b = a.copy(), b.copy()
    output = np.full(first.shape, np.nan)
    second, audit = op.matmul(a, b, out=output, workspace=workspace, audit=True)
    dispatched = assert_dispatch(audit, True, repeated=True)
    assert second is output
    np.testing.assert_allclose(second, np.repeat(row @ b, a.shape[0], axis=0), rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(first, saved)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)
    assert not np.shares_memory(first, second)
    assert audit["algorithm_copy_bytes"] == ((a.shape[0] - 1) * b.shape[1] * 8 if dispatched else 0)
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0


@pytest.mark.parametrize("right_order", ["C", "F"])
def test_matmul_repeated_rows_preserve_zero_nonfinite_and_subnormal_columns(right_order):
    owner = np.zeros((1, 35))
    owner[0, :3] = [1.0, 1.0, -1.0]
    a = np.broadcast_to(owner, (33, 35))
    b = np.zeros((35, 37), order=right_order)
    b[:3, 0] = [1.0, 2.0**-54, 1.0]
    b[3, 1] = np.inf  # The zero coefficient must still consume 0*Inf.
    b[0, 2:6] = [np.inf, -np.inf, np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0)]
    b[:3, 6] = 1e308
    b[3, 7] = np.nan
    owner.flags.writeable = b.flags.writeable = False
    before_a, before_b = a.copy(), b.copy()
    expected = op.matmul(a, b, simd="scalar")
    actual, audit = op.matmul(a, b, audit=True)
    dispatched = assert_dispatch(audit, True, repeated=True)
    assert_ieee_classes_and_close(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], 0.0)
    assert not np.signbit(actual[:, 0]).any()
    assert np.isnan(actual[:, [1, 7]]).all()
    assert np.isposinf(actual[:, [2, 6]]).all()
    assert np.isneginf(actual[:, 3]).all()
    np.testing.assert_array_equal(actual[:, 4], np.nextafter(0.0, 1.0))
    np.testing.assert_array_equal(actual[:, 5], -np.nextafter(0.0, 1.0))
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)
    assert audit["input_copy_bytes"] == 0
    assert audit["algorithm_copy_bytes"] == ((a.shape[0] - 1) * b.shape[1] * 8 if dispatched else 0)
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0
