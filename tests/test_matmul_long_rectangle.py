"""Long-axis root geometry, complete bounds and independent C/F ownership."""

import hashlib
from decimal import Decimal, localcontext

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import assert_ieee_classes_and_close, declarations, settings
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks

SHAPES = [(8193, 256, 32), (32, 256, 8193)]


def payload(shape, layout):
    m, k, n = shape
    rng = np.random.default_rng(5218193)
    a = np.array(rng.uniform(-0.01, 0.01, (m, k)), order=layout[0].upper())
    b = np.array(rng.uniform(-0.01, 0.01, (k, n)), order=layout[1].upper())
    return a, b


def readonly(a, b):
    a.flags.writeable = b.flags.writeable = False


def execute(scheduler, graph, a, b, intervals=1):
    bounds = np.zeros(intervals, np.int64)
    return scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)


def calls(result):
    return sum(c["accelerate_matmul_calls"] for c in chunks(result))


@pytest.mark.skipif(not ACCELERATE, reason="enabled macOS arm64 Accelerate")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_long_axis_graph_transport_layout_budget_and_generations(shape, layout, order, lane):
    a, b = payload(shape, layout)
    readonly(a, b)

    def digest(x):
        return hashlib.sha256(x.tobytes()).hexdigest()

    hashes = digest(a), digest(b)
    expected = a @ b
    graph = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = execute(scheduler, graph, a, b, 2)
        old_hashes = [digest(x) for x in first.outputs[0].values]
        second = execute(scheduler, graph, a, b, 2)
        assert scheduler.peak_active_cpu_tokens <= 2
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    reserve = (1024 * 1024 + 512 * shape[1] + (max(shape[0], shape[2]) + 7) // 8) * 8
    for result in (first, second):
        assert calls(result) == 2
        observed = max(
            c["numeric_arena_bytes"]
            + c["mask_arena_bytes"]
            + c["operator_workspace_capacity_bytes"]
            + c["native_library_workspace_reserve_bytes"]
            for c in chunks(result)
        )
        assert result.plan.estimated_worker_scratch_bytes >= observed
        assert all(c["native_library_workspace_reserve_bytes"] == reserve for c in chunks(result))
        assert (
            result.audit["python_operator_calls"] == result.audit["request_time_compilation"] == 0
        )
        for x in result.outputs[0].values:
            np.testing.assert_allclose(x, expected, atol=0.000049, rtol=0)
            assert not x.flags.writeable
            assert x.flags.c_contiguous if order == "C" else x.flags.f_contiguous
            assert not np.shares_memory(x, a) and not np.shares_memory(x, b)
        assert not np.shares_memory(*result.outputs[0].values)
    assert old_hashes == [digest(x) for x in first.outputs[0].values]
    assert (digest(a), digest(b)) == hashes
    assert all(
        not np.shares_memory(x, y)
        for x in first.outputs[0].values
        for y in second.outputs[0].values
    )
    with localcontext() as context:
        context.prec = 80
        for i, j in ((0, 0), (shape[0] - 1, shape[2] - 1)):
            exact = sum(
                (
                    Decimal.from_float(float(a[i, h])) * Decimal.from_float(float(b[h, j]))
                    for h in range(shape[1])
                ),
                Decimal(0),
            )
            assert abs(
                Decimal.from_float(float(first.outputs[0].values[0][i, j])) - exact
            ) < Decimal(".000049")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize(
    "pattern",
    [
        "positive_zero",
        "consistent",
        "late_zero",
        "late_negative_zero",
        "late_nan",
        "late_inf",
        "late_tiny",
        "late_huge",
    ],
)
def test_full_long_axis_scan_and_ieee_rejection(shape, layout, pattern):
    a, b = payload(shape, layout)
    if pattern == "positive_zero":
        a[:] = np.abs(a)
        b[:] = np.abs(b)
        a[-1, -1] = 0.0
    elif pattern == "consistent":
        a[:] = np.abs(a)
        b[:] = np.abs(b)
        a[1::2] *= -1
        b[:, 1::2] *= -1
    else:
        target = a if shape[0] > 8192 else b
        target[-1, -1] = {
            "late_zero": 0.0,
            "late_negative_zero": -0.0,
            "late_nan": np.nan,
            "late_inf": np.inf,
            "late_tiny": 2.0**-129,
            "late_huge": 2.0**129,
        }[pattern]
    readonly(a, b)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph, a, b)
    expected = op.matmul(a, b, simd="scalar")
    assert_ieee_classes_and_close(result.outputs[0].values[0], expected)
    permitted = pattern in {"positive_zero", "consistent", "late_zero"}
    assert calls(result) == int(ACCELERATE and permitted)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_long_prepared_rereads_zero_then_restores_library_and_old_owner(shape, layout, order):
    a, b = payload(shape, layout)
    readonly(a, b)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        bounds = np.zeros(1, np.int64)
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
        old = prepared.run_snapshot()
        saved = old.outputs[0].values[0].copy()
        target = a if shape[0] > 8192 else b
        original = target[-1, -1]
        target.setflags(write=True)
        target[-1, -1] = -0.0
        target.setflags(write=False)
        rejected = prepared.run_snapshot()
        assert calls(rejected) == 0
        assert_ieee_classes_and_close(rejected.outputs[0].values[0], op.matmul(a, b, simd="scalar"))
        target.setflags(write=True)
        target[-1, -1] = original
        target.setflags(write=False)
        restored = prepared.run_snapshot()
        assert calls(restored) == int(ACCELERATE)
        np.testing.assert_array_equal(old.outputs[0].values[0], saved)
        assert not np.shares_memory(old.outputs[0].values[0], restored.outputs[0].values[0])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("mode", ["compare", "isolate", "multiple"])
def test_long_internal_discrete_isolated_shared_consumers_stay_canonical(shape, layout, mode):
    a, b = payload(shape, layout)
    readonly(a, b)
    base = "matmul(a,b)"
    expression = (
        base + ">0"
        if mode == "compare"
        else [base, "sum(" + base + ")"]
        if mode == "multiple"
        else base
    )
    graph = GraphCompiler(declarations()).compile(
        expression, error_policy="isolate" if mode == "isolate" else "raise"
    )
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph, a, b)
    assert calls(result) == 0
    canonical = op.matmul(a, b, simd="scalar")
    if mode == "compare":
        np.testing.assert_array_equal(result.outputs[0].values[0], canonical > 0)
    else:
        np.testing.assert_array_equal(
            result.outputs[0].values[0].view(np.uint64), canonical.view(np.uint64)
        )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "ff"])
def test_long_padding_nan_is_unconsumed(shape, layout):
    a, b = payload(shape, layout)
    views = []
    for value, order in ((a, layout[0]), (b, layout[1])):
        if order == "c":
            owner = np.full((value.shape[0], value.shape[1] + 3), np.nan)
            view = owner[:, : value.shape[1]]
        else:
            owner = np.full((value.shape[0] + 3, value.shape[1]), np.nan, order="F")
            view = owner[: value.shape[0]]
        view[:] = value
        view.flags.writeable = False
        views.append(view)
    a, b = views
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph, a, b)
    assert calls(result) == int(ACCELERATE)
    np.testing.assert_allclose(result.outputs[0].values[0], a @ b, rtol=0, atol=0.000049)


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_long_time_intervals_cover_mixed_admission_with_one_original_budget(layout, order):
    a, b = payload((16385, 256, 32), layout)
    readonly(a, b)
    types = dict(
        a=dict(kind="matrix", axes=["time", "factor"], shape=["T", "K"]),
        b=dict(kind="matrix", axes=["factor", "asset"], shape=["K", "N"]),
    )
    graph = GraphCompiler(types).compile("matmul(a,b)", output_order=order)
    starts = np.array([0, 17, 41], np.int64)
    ends = np.array([16385, 17 + 8193, 41 + 8191], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(
            graph, {"a": a, "b": b}, starts, ends, memory_budget_bytes=64 << 20
        )
        saved = [x.copy() for x in first.outputs[0].values]
        second = scheduler.execute(
            graph,
            {"a": a, "b": b},
            starts[::-1].copy(),
            ends[::-1].copy(),
            memory_budget_bytes=64 << 20,
        )
    for result, left, right in ((first, starts, ends), (second, starts[::-1], ends[::-1])):
        observed = max(
            c["numeric_arena_bytes"]
            + c["mask_arena_bytes"]
            + c["operator_workspace_capacity_bytes"]
            + c["native_library_workspace_reserve_bytes"]
            for c in chunks(result)
        )
        assert result.plan.estimated_worker_scratch_bytes >= observed
        for value, lo, hi in zip(result.outputs[0].values, left, right, strict=True):
            np.testing.assert_allclose(value, a[lo:hi] @ b, atol=0.000049, rtol=0)
    for value, original in zip(first.outputs[0].values, saved, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("shape", [(8193, 1024, 8), (8, 1024, 8193)])
@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_long_high_scale_partitioned_bounds_do_not_depend_on_external_axis(shape, layout, order):
    a, b = payload(shape, layout)
    a *= 102400
    b *= 102400
    readonly(a, b)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph, a, b)
    actual = result.outputs[0].values[0]
    assert calls(result) > 1 if ACCELERATE else calls(result) == 0
    with localcontext() as context:
        context.prec = 80
        for i, j in ((0, 0), (shape[0] - 1, shape[2] - 1), (shape[0] // 2, shape[2] // 2)):
            exact = sum(
                (
                    Decimal.from_float(float(a[i, h])) * Decimal.from_float(float(b[h, j]))
                    for h in range(shape[1])
                ),
                Decimal(0),
            )
            assert abs(Decimal.from_float(float(actual[i, j])) - exact) < Decimal(".000049")


@pytest.mark.parametrize("inner", [127, 128])
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_short_inner_dimension_keeps_original_minimum_work(inner, layout, order):
    a, b = payload((8192, inner, 64), layout)
    readonly(a, b)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph, a, b)
    assert calls(result) == int(ACCELERATE and inner == 128)
    np.testing.assert_allclose(result.outputs[0].values[0], a @ b, rtol=0, atol=0.000049)
