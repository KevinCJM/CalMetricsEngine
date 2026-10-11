"""Bounded signed GEMM and canonical F roots through the complete native API."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import declarations, settings
from .test_nonnegative_matmul_accelerate import ACCELERATE


def matrices(layout, shape=(513, 257, 515)):
    rng = np.random.default_rng(221005)
    orders = "cf" if layout == "padded" else layout
    a = np.array(rng.uniform(-1, 1, (shape[0], shape[1])), order=orders[0].upper())
    b = np.array(rng.uniform(-1, 1, (shape[1], shape[2])), order=orders[1].upper())
    if layout == "padded":
        left = np.full((shape[0], shape[1] + 3), np.nan)
        right = np.full((shape[1] + 3, shape[2]), np.nan, order="F")
        left[:, : shape[1]], right[: shape[1]] = a, b
        a, b = left[:, : shape[1]], right[: shape[1]]
    a.flags.writeable = b.flags.writeable = False
    return a, b


def graph(order="F", transpose=False, isolate=False):
    expression = "transpose(matmul(a,b))" if transpose else "matmul(a,b)"
    return GraphCompiler(declarations()).compile(
        expression,
        output_order=order,
        error_policy="isolate" if isolate else "raise",
    )


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff", "padded"])
@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_signed_root_library_layout_and_retained_results(layout, order, transpose, lane):
    a, b = matrices(layout)
    saved_a, saved_b = a.copy(), b.copy()
    # Independent decimal cells plus the whole canonical product.
    reference = op.matmul(a, b)
    for i, j in [(0, 0), (17, 23), (a.shape[0] - 1, b.shape[1] - 1)]:
        with localcontext() as context:
            context.prec = 80
            independent = sum(
                (
                    Decimal.from_float(float(x)) * Decimal.from_float(float(y))
                    for x, y in zip(a[i], b[:, j], strict=False)
                ),
                Decimal(0),
            )
        assert abs(reference[i, j] - float(independent)) < 1e-10
    expected = reference.T if transpose else reference
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        compiled = graph(order, transpose)
        first = scheduler.execute(compiled, {"a": a, "b": b}, bounds, bounds)
        preserved = first.outputs[0].values[0].copy()
        second = scheduler.execute(compiled, {"a": a, "b": b}, bounds, bounds)
    for result in (first, second):
        for value in result.outputs[0].values:
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-10)
            assert not value.flags.writeable
            assert not np.shares_memory(value, a) and not np.shares_memory(value, b)
            assert (
                value.flags.f_contiguous
                if order == "F" or (transpose and order == "auto")
                else value.flags.c_contiguous
            )
        assert sum(c["accelerate_matmul_calls"] for c in result.audit["native_chunks"]) == (
            2 if ACCELERATE else 0
        )
        for chunk in result.audit["native_chunks"]:
            if not transpose or order != "C":
                assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
    assert not np.shares_memory(first.outputs[0].values[0], first.outputs[0].values[1])
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    np.testing.assert_array_equal(first.outputs[0].values[0], preserved)
    np.testing.assert_array_equal(a, saved_a)
    np.testing.assert_array_equal(b, saved_b)


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_cancellation_and_sparse_zero_cells_replay_exact_fold(layout, order):
    a, b = matrices(layout, (512, 256, 512))
    a.setflags(write=True)
    b.setflags(write=True)
    a[0] = b[:, 0] = 0.0
    a[1] = 0.0
    a[1, :3] = [1.0, 2.0**-53, -1.0]
    b[:3, 1] = 1.0
    a[2] = 0.0
    a[2, :3] = [1.0, 2.0**-50, -1.0]
    b[:3, 2] = 1.0
    a.flags.writeable = b.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            graph(order), {"a": a, "b": b}, np.zeros(1, np.int64), np.zeros(1, np.int64)
        )
    actual = result.outputs[0].values[0]
    expected = op.matmul(a, b)
    for i, j in [(0, 0), (1, 1), (2, 2)]:
        assert actual[i, j].view(np.uint64) == expected[i, j].view(np.uint64)
    np.testing.assert_array_equal(actual == 0, expected == 0)
    assert not np.signbit(actual[actual == 0]).any()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("layout", ["cc", "cf", "ff", "reverse", "broadcast"])
@pytest.mark.parametrize("shape", [(7, 33, 9), (3, 17, 5), (7, 0, 9), (0, 33, 9)])
def test_f_root_canonical_fallback_and_empty_geometry(layout, shape):
    a, b = matrices("ff" if layout in {"reverse", "broadcast"} else layout, shape)
    if layout == "reverse":
        a, b = a[:, ::-1], b[::-1]
    elif layout == "broadcast" and shape[0]:
        a = np.broadcast_to(a[:1], a.shape)
    expected = op.matmul(a, b, simd="scalar")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(), {"a": a, "b": b}, bounds, bounds)
    actual = result.outputs[0].values[0]
    np.testing.assert_array_equal(
        np.array(actual, order="C").view(np.uint64), expected.view(np.uint64)
    )
    assert actual.flags.f_contiguous
    chunk = result.audit["native_chunks"][0]
    assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0


@pytest.mark.parametrize("fault", ["bound", "negative_zero", "nan", "inf"])
def test_general_signed_rejection_keeps_canonical_f_root(fault):
    a, b = matrices("ff", (512, 256, 512))
    a.setflags(write=True)
    if fault == "bound":
        a *= 2.0**30
    else:
        a[11, 13] = {"negative_zero": -0.0, "nan": np.nan, "inf": np.inf}[fault]
    a.setflags(write=False)
    expected = op.matmul(a, b)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(), {"a": a, "b": b}, bounds, bounds)
    actual = np.array(result.outputs[0].values[0], order="C")
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    chunk = result.audit["native_chunks"][0]
    assert chunk["accelerate_matmul_calls"] == 0
    assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("scale", [512.0, 1024.0, 2048.0])
def test_scaled_signed_domain_respects_absolute_error_admission(layout, order, scale):
    a, b = matrices(layout, (512, 256, 512))
    a, b = (
        np.array(a * scale, order=layout[0].upper()),
        np.array(b * scale, order=layout[1].upper()),
    )
    a.flags.writeable = b.flags.writeable = False
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(order), {"a": a, "b": b}, bounds, bounds)
    actual = result.outputs[0].values[0]
    expected = op.matmul(a, b)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.000049)
    calls = result.audit["native_chunks"][0]["accelerate_matmul_calls"]
    if ACCELERATE:
        assert calls == 1 if scale <= 1024.0 else calls > 1
        # A higher scale now uses bounded pieces, not a looser error budget.
        with localcontext() as context:
            context.prec = 80
            for i, j in [(0, 0), (17, 23), (511, 511)]:
                exact = sum(
                    (
                        Decimal.from_float(float(x)) * Decimal.from_float(float(y))
                        for x, y in zip(a[i], b[:, j], strict=False)
                    ),
                    Decimal(0),
                )
                assert abs(Decimal.from_float(float(actual[i, j])) - exact) <= Decimal.from_float(
                    0.000049
                )
    else:
        assert calls == 0
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_rereads_mixed_values_then_rejects_changed_bound(order):
    a, b = matrices("ff", (512, 256, 512))
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph(order), {"a": a, "b": b}, bounds, bounds)
        first = prepared.run_snapshot()
        preserved = first.outputs[0].values[0].copy()
        a.setflags(write=True)
        a *= 2.0**30
        a.setflags(write=False)
        second = prepared.run_snapshot()
    assert first.audit["native_chunks"][0]["accelerate_matmul_calls"] == (1 if ACCELERATE else 0)
    assert second.audit["native_chunks"][0]["accelerate_matmul_calls"] == 0
    np.testing.assert_array_equal(second.outputs[0].values[0], op.matmul(a, b))
    np.testing.assert_array_equal(first.outputs[0].values[0], preserved)


@pytest.mark.parametrize("mode", ["isolate", "discrete", "trace", "shared"])
def test_general_signed_consumers_retain_canonical_arithmetic(mode):
    a, b = matrices("ff", (512, 256, 512))
    expr = {
        "isolate": "matmul(a,b)",
        "discrete": "matmul(a,b)>0",
        "trace": "trace(matmul(a,b))",
        "shared": ["matmul(a,b)", "trace(matmul(a,b))"],
    }[mode]
    types = declarations()
    if mode in {"trace", "shared"}:
        # The trace contract requires the same symbolic axes and dimensions.
        types["b"]["shape"] = ["K", "M"]
    compiled = GraphCompiler(types).compile(
        expr,
        result_format="typed",
        output_order="F",
        error_policy="isolate" if mode == "isolate" else "raise",
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(compiled, {"a": a, "b": b}, bounds, bounds)
    canonical = op.matmul(a, b)
    expected = (
        canonical > 0
        if mode == "discrete"
        else op.trace(canonical)
        if mode == "trace"
        else canonical
    )
    np.testing.assert_array_equal(result.outputs[0].values[0], expected)
    assert all(c["accelerate_matmul_calls"] == 0 for c in result.audit["native_chunks"])
