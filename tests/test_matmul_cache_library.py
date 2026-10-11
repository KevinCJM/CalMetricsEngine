"""Cache-blocked MatMul: existing math, late-panel boundaries and owned results.

Exact checks provide bounded fixture evidence and do not establish a general
bitwise API contract.
"""

import gc
import math
import os
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_matmul_library import (
    ATOL,
    RTOL,
    assert_ieee_classes_and_close,
    declarations,
    inputs,
    settings,
)
from .test_matmul_library import (
    assert_dispatch as assert_unblocked_dispatch,
)
from .test_matrix_library_backend import PACKING_PLATFORM

BLOCKED_ISA = "eigen-block-matmul"


def assert_blocked_dispatch(audit, eligible=True):
    enabled = os.environ.get("CME_EXPECT_EIGEN")
    if not eligible or not PACKING_PLATFORM or enabled == "0":
        assert audit["isa"] != BLOCKED_ISA
    elif enabled == "1":
        assert audit["isa"] == BLOCKED_ISA
    else:
        assert audit["isa"] in {"scalar", *op.available_simd(), BLOCKED_ISA}


def assert_no_numeric_scratch_or_copy(audit):
    assert audit["input_copy_bytes"] == audit["algorithm_copy_bytes"] == 0
    assert audit["workspace_bytes"] == audit["workspace_capacity_bytes"] == 0


def fsum_product_reference(a, b):
    # These fixed cancellation-heavy fixtures already exceeded the original
    # tolerance against NumPy GEMM on the previous engine. Independent fsum of
    # rounded binary64 products avoids using that differently rounded GEMM as
    # the oracle; the input values and original RTOL/ATOL remain unchanged.
    return np.asarray(
        [
            [
                math.fsum(float(x) * float(y) for x, y in zip(row, column, strict=True))
                for column in b.T
            ]
            for row in a
        ],
        dtype=np.float64,
    )


def assert_fixture_matches_scalar(actual, a, b):
    # Supplemental evidence that these particular fixtures keep the current
    # fold, not a new all-input/public bitwise MatMul contract.
    scalar = op.matmul(a, b, simd="scalar")
    assert_ieee_classes_and_close(actual, scalar)
    np.testing.assert_array_equal(actual.view(np.uint64), scalar.view(np.uint64))


@pytest.mark.parametrize("inner", [682, 683, 684])
@pytest.mark.parametrize("layout", ["cc", "padded_c_b"])
def test_cache_gate_keeps_reference_and_output_boundaries(inner, layout):
    a, b, owners = inputs((33, inner, 65), layout)
    saved = [array.copy() for array in [a, b, *owners]]
    a.flags.writeable = b.flags.writeable = False
    expected = fsum_product_reference(a, b)
    # A nonzero/NaN-filled out must not become the initial panel accumulator.
    storage = np.full(33 * 65 + 2, np.nan)
    storage[[0, -1]] = [1701.0, -1701.0]
    out = storage[1:-1].reshape(33, 65)
    actual, audit = op.matmul(a, b, out=out, audit=True)
    assert actual is out
    assert_blocked_dispatch(audit, inner >= 683)
    assert_no_numeric_scratch_or_copy(audit)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    assert_fixture_matches_scalar(actual, a, b)
    np.testing.assert_array_equal(storage[[0, -1]], [1701.0, -1701.0])
    for array, before in zip([a, b, *owners], saved, strict=True):
        np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize(
    "shape,left_layout",
    [
        ((32, 683, 33), "c"),
        ((33, 684, 63), "f"),
        ((64, 683, 64), "padded"),
        ((65, 684, 65), "negative_rows"),
        ((65, 683, 65), "negative_columns"),
        ((33, 684, 65), "sliced"),
        ((33, 683, 33), "broadcast_columns"),
    ],
)
def test_cache_blocks_borrow_strided_left_and_padded_right(shape, left_layout):
    a, b, owners = inputs(shape, "padded_c_b")
    if left_layout == "f":
        a = np.asfortranarray(a)
    elif left_layout == "padded":
        owner = np.full((shape[0], shape[1] + 3), np.nan)
        owner[:, : shape[1]] = a
        a = owner[:, : shape[1]]
        owners.append(owner)
    elif left_layout == "negative_rows":
        a = a[::-1]
    elif left_layout == "negative_columns":
        a = a[:, ::-1]
    elif left_layout == "sliced":
        owner = np.full((shape[0] * 2, shape[1] * 2), np.nan)
        owner[::2, ::2] = a
        a = owner[::2, ::2]
        owners.append(owner)
    elif left_layout == "broadcast_columns":
        a = np.broadcast_to(a[:, :1], a.shape)
    saved = [array.copy() for array in [a, b, *owners]]
    a.flags.writeable = b.flags.writeable = False
    actual, audit = op.matmul(a, b, audit=True)
    assert_blocked_dispatch(audit)
    assert_no_numeric_scratch_or_copy(audit)
    np.testing.assert_allclose(actual, fsum_product_reference(a, b), rtol=RTOL, atol=ATOL)
    assert_fixture_matches_scalar(actual, a, b)
    assert not np.shares_memory(actual, a) and not np.shares_memory(actual, b)
    for array, before in zip([a, b, *owners], saved, strict=True):
        np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize(
    "shape,layout",
    [
        ((31, 683, 33), "cc"),
        ((33, 683, 31), "cc"),
        ((33, 683, 33), "cf"),
        ((33, 683, 33), "general"),
        ((33, 683, 33), "negative"),
        ((33, 683, 33), "zero_b"),
    ],
)
def test_cache_dispatch_leaves_other_layouts_and_small_axes_supported(shape, layout):
    a, b, _ = inputs(shape, layout)
    actual, audit = op.matmul(a, b, audit=True)
    assert_blocked_dispatch(audit, False)
    np.testing.assert_allclose(actual, a @ b, rtol=RTOL, atol=ATOL)


def late_panel_inputs(pattern, shape=(65, 684, 65)):
    rows, inner, columns = shape
    a = np.zeros((rows, inner))
    b = np.zeros((inner, columns))
    if pattern == "fma":
        a[:, 639:641] = [-1.0, 1.0 + 2.0**-27]
        b[639:641] = np.array([1.0, 1.0 - 2.0**-27])[:, None]
    elif pattern == "split_accumulators":
        a[:, 639:642] = [1.0, 2.0**-54, -1.0]
        b[639:642] = 1.0
    elif pattern == "overflow":
        a[:, 639:642] = [1e308, 1e308, -1e308]
        b[639:642] = 2.0
    elif pattern == "zero_times_inf":
        b[640] = np.inf
    elif pattern == "nan_times_zero":
        a[:, 640] = np.nan
    else:
        assert pattern == "subnormal_and_zero"
        a[:] = -0.0
        a[:, 640] = 1.0
        tiny = np.nextafter(0.0, 1.0)
        b[640, ::3] = tiny
        b[640, 1::3] = -tiny
    return a, b


@pytest.mark.parametrize(
    "pattern",
    [
        "fma",
        "split_accumulators",
        "overflow",
        "zero_times_inf",
        "nan_times_zero",
        "subnormal_and_zero",
    ],
)
def test_cache_late_panel_order_and_ieee_include_row_and_column_edges(pattern):
    a, b = late_panel_inputs(pattern)
    a.flags.writeable = b.flags.writeable = False
    before_a, before_b = a.copy(), b.copy()
    expected = op.matmul(a, b, simd="scalar")
    actual, audit = op.matmul(a, b, audit=True)
    assert_blocked_dispatch(audit)
    assert_no_numeric_scratch_or_copy(audit)
    assert_ieee_classes_and_close(actual, expected)
    if pattern in {"fma", "split_accumulators"}:
        np.testing.assert_array_equal(actual, 0.0)
        assert not np.signbit(actual).any()
    elif pattern in {"overflow", "zero_times_inf", "nan_times_zero"}:
        assert np.isnan(actual).all()
    else:
        tiny = np.nextafter(0.0, 1.0)
        np.testing.assert_array_equal(actual[:, ::3], tiny)
        np.testing.assert_array_equal(actual[:, 1::3], -tiny)
        np.testing.assert_array_equal(actual[:, 2::3], 0.0)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


def test_cache_explicit_isa_preserves_existing_dispatch():
    a, b, _ = inputs((33, 683, 65), "padded_c_b")
    expected = a @ b
    for isa in ["auto", "scalar", *op.available_simd()]:
        actual, audit = op.matmul(a, b, simd=isa, audit=True)
        assert_blocked_dispatch(audit, isa == "auto")
        if isa == "scalar":
            assert audit["isa"] == "scalar"
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_cache_gate_keeps_repeated_rows_first_and_refreshes_current_values():
    _, b, _ = inputs((33, 683, 65), "padded_c_b")
    owner = np.random.default_rng(7041).normal(size=(1, 683))
    a = np.broadcast_to(owner, (33, 683))
    first, audit = op.matmul(a, b, audit=True)
    assert_blocked_dispatch(audit, False)
    used = assert_unblocked_dispatch(audit, True, repeated=True)
    assert audit["algorithm_copy_bytes"] == ((33 - 1) * 65 * 8 if used else 0)
    saved = first.copy()
    np.testing.assert_allclose(first, a @ b, rtol=RTOL, atol=ATOL)
    owner *= 2
    second = op.matmul(a, b)
    np.testing.assert_allclose(second, a @ b, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(first, saved)
    assert not np.shares_memory(first, second)
    assert first.strides == (65 * 8, 8)


def test_cache_readonly_aliased_inputs_and_workspace_do_not_own_result_storage():
    owner = np.random.default_rng(5261).normal(size=33 * 683)
    a, b = owner.reshape(33, 683), owner.reshape(683, 33)
    assert np.shares_memory(a, b)
    before = owner.copy()
    a.flags.writeable = b.flags.writeable = owner.flags.writeable = False
    workspace = op.Workspace()
    first, audit = op.matmul(a, b, workspace=workspace, audit=True)
    assert_blocked_dispatch(audit)
    assert_no_numeric_scratch_or_copy(audit)
    np.testing.assert_allclose(first, a @ b, rtol=RTOL, atol=ATOL)
    saved = first.copy()
    second = op.matmul(a * 2, b, workspace=workspace)
    np.testing.assert_allclose(second, saved * 2, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(first, saved)
    np.testing.assert_array_equal(owner, before)
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, owner)
    writable = before.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIASES_INPUT"):
        op.matmul(
            writable.reshape(33, 683),
            writable.reshape(683, 33),
            out=writable[: 33 * 33].reshape(33, 33),
        )
    np.testing.assert_array_equal(writable, before)
    del owner, a, b, workspace, second
    gc.collect()
    np.testing.assert_array_equal(first, saved)


@pytest.mark.parametrize("layout", ["cc", "padded_c_b"])
def test_cache_prepared_snapshots_refresh_bound_content_and_keep_old_results(layout):
    a, b, _ = inputs((33, 683, 65), layout)
    graph = GraphCompiler(declarations()).compile(["matmul(a,b)", "b"])
    bounds = np.zeros(1, np.int64)
    # Establish dispatch for these exact bound strides; runtime audit does not
    # expose a per-MatMul ISA label and must not be presented as such evidence.
    _, direct_audit = op.matmul(a, b, audit=True)
    assert_blocked_dispatch(direct_audit)
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
        first = prepared.run_snapshot()
        saved = [output.values[0].copy() for output in first.outputs]
        np.testing.assert_allclose(saved[0], fsum_product_reference(a, b), rtol=RTOL, atol=ATOL)
        assert_fixture_matches_scalar(saved[0], a, b)
        a[:, 640] += 0.25
        b *= 2.0
        second = prepared.run_snapshot()
        np.testing.assert_allclose(
            second.outputs[0].values[0], fsum_product_reference(a, b), rtol=RTOL, atol=ATOL
        )
        assert_fixture_matches_scalar(second.outputs[0].values[0], a, b)
        np.testing.assert_array_equal(second.outputs[1].values[0], b)
        assert first.audit["result_lifetime"] == second.audit["result_lifetime"] == "independent"
    del prepared
    gc.collect()
    for old, new, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
        np.testing.assert_array_equal(old.values[0], snapshot)
        assert not old.values[0].flags.writeable and not new.values[0].flags.writeable
        assert not np.shares_memory(old.values[0], new.values[0])
        assert not np.shares_memory(new.values[0], a)
        assert not np.shares_memory(new.values[0], b)


@pytest.mark.parametrize("pattern", ["fma", "split_accumulators"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_cache_pickled_dag_preserves_typed_boundaries_and_isolated_errors(pattern, lane):
    a, b = late_panel_inputs(pattern, (33, 683, 33))
    ids = np.array([2**53 + 1, -(2**63), 2**63 - 1], dtype=np.int64)
    decl = declarations()
    decl["ids"] = dict(kind="vector", dtype="int64", axes=["asset"], shape=[3])
    expression = "matmul(a,b)"
    original = GraphCompiler(decl).compile(
        [
            expression,
            f"greater_than({expression},0)",
            f"1/sum({expression})",
            f"greater_than(1/sum({expression}),0)",
            "ids",
            "7",
        ],
        error_policy="isolate",
    )
    graph = pickle.loads(pickle.dumps(original))
    assert graph.fingerprint == original.fingerprint
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    a.flags.writeable = b.flags.writeable = ids.flags.writeable = False
    expected, audit = op.matmul(a, b, audit=True)
    assert_blocked_dispatch(audit)
    np.testing.assert_array_equal(expected, 0.0)
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b, "ids": ids}, bounds, bounds)
        assert scheduler.peak_active_cpu_tokens <= 2
    assert result.output_kind == "typed"
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    if lane.startswith("process"):
        assert result.plan.use_shared_memory == (lane == "process_shared")
    for row in range(2):
        matrix, comparison, broken, dependent, exact_ids, healthy = result.outputs
        np.testing.assert_array_equal(matrix.values[row], 0.0)
        np.testing.assert_array_equal(comparison.values[row], False)
        assert matrix.values[row].dtype == np.float64
        assert comparison.values[row].dtype == np.bool_
        assert matrix.root_statuses[row] == comparison.root_statuses[row] == 0
        assert broken.root_statuses[row] != 0
        assert dependent.root_statuses[row] == broken.root_statuses[row]
        np.testing.assert_array_equal(exact_ids.values[row], ids)
        assert exact_ids.values[row].dtype == np.int64
        assert exact_ids.root_statuses[row] == healthy.root_statuses[row] == 0
        assert healthy.values[row] == 7
        for output in (matrix, comparison, exact_ids):
            assert not output.values[row].flags.writeable
            assert not np.shares_memory(output.values[row], a)
            assert not np.shares_memory(output.values[row], b)
            assert not np.shares_memory(output.values[row], ids)
    assert result.audit["python_operator_calls"] == 0
    if lane == "process_shared":
        assert result.audit["output_copy_bytes"] == 0


def test_cache_shared_mapping_result_views_survive_all_input_owners():
    a, b, _ = inputs((33, 683, 65), "cc")
    expected = a @ b
    graph = GraphCompiler(declarations()).compile(["matmul(a,b)", "b"])
    bounds = np.zeros(2, np.int64)
    with SharedInputBundle.from_inputs({"a": a, "b": b}) as bundle:
        for array in bundle.arrays.values():
            assert not array.flags.writeable
        del array
        with AdaptiveScheduler(cpu_budget=2, config=settings("process_shared")) as scheduler:
            result = scheduler.execute(graph, bundle, bounds, bounds)
        assert result.plan.lane == "process" and result.plan.use_shared_memory
        views = tuple(result.outputs[0].values)
        rhs_views = tuple(result.outputs[1].values)
    assert result.audit["output_copy_bytes"] == 0
    before_b = b.copy()
    del result, bundle, scheduler, a, b
    gc.collect()
    for actual, rhs in zip(views, rhs_views, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(rhs, before_b)
        assert not actual.flags.writeable and not rhs.flags.writeable
