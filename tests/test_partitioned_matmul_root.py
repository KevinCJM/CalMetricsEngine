"""High-amplitude independent roots: real graph/worker path and exact fallback."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import declarations, settings
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks, no_library


def inputs(layout="cc", shape=(513, 1025, 515)):
    m, k, n = shape
    rng = np.random.default_rng(3501025)
    a = np.array(rng.uniform(-1024, 1024, (m, k)), order=layout[0].upper())
    b = np.array(rng.uniform(-1024, 1024, (k, n)), order=layout[1].upper())
    a.flags.writeable = b.flags.writeable = False
    return a, b


def decimal_check(actual, a, b, transposed=False):
    matrix = actual.T if transposed else actual
    with localcontext() as ctx:
        ctx.prec = 80
        for i, j in [
            (0, 0),
            (1, 3),
            (a.shape[0] - 1, b.shape[1] - 1),
            (a.shape[0] // 2, b.shape[1] // 2),
        ]:
            exact = sum(
                (
                    Decimal.from_float(float(a[i, k])) * Decimal.from_float(float(b[k, j]))
                    for k in range(a.shape[1])
                ),
                Decimal(0),
            )
            assert abs(Decimal.from_float(float(matrix[i, j])) - exact) <= Decimal.from_float(
                0.000049
            )


@pytest.mark.skipif(not ACCELERATE, reason="enabled macOS 15 arm64 library")
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_high_amplitude_graph_worker_owned_layout(layout, order, transposed, lane):
    a, b = inputs(layout)
    saved = (a.tobytes(), b.tobytes())
    expected = op.matmul(a, b)
    graph = GraphCompiler(declarations()).compile(
        "transpose(matmul(a,b))" if transposed else "matmul(a,b)", output_order=order
    )
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        snapshot = first.outputs[0].values[0].copy()
        again = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        assert scheduler.peak_active_cpu_tokens <= 2
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for value in first.outputs[0].values:
        np.testing.assert_allclose(
            value, expected.T if transposed else expected, rtol=0, atol=0.000049
        )
        decimal_check(value, a, b, transposed)
        assert not value.flags.writeable
        assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
        assert not any(np.shares_memory(value, x) for x in [a, b, *again.outputs[0].values])
    assert not np.shares_memory(*first.outputs[0].values)
    np.testing.assert_array_equal(first.outputs[0].values[0], snapshot)
    assert saved == (a.tobytes(), b.tobytes())
    assert sum(c["accelerate_matmul_calls"] for c in chunks(first)) > 2
    reserve = 8 * 1024 * 1024 + 512 * a.shape[1] * 8
    explicit = (320 * a.shape[1] + 72) * 8
    assert first.plan.estimated_worker_scratch_bytes >= reserve + explicit
    for c in chunks(first):
        assert c["native_library_workspace_reserve_bytes"] == reserve
        assert 0 < c["operator_workspace_capacity_bytes"] <= explicit
        assert c["algorithm_copy_bytes"] == 0


@pytest.mark.parametrize("mode", ["comparison", "shared", "isolate"])
def test_high_amplitude_internal_consumers_remain_canonical(mode):
    a, b = inputs("fc", (512, 513, 512))
    decl = declarations()
    expected = op.matmul(a, b)
    expression = "matmul(a,b)>0" if mode == "comparison" else "matmul(a,b)"
    if mode == "shared":
        decl["b"]["shape"] = ["K", "M"]
        expression = ["matmul(a,b)", "trace(matmul(a,b))"]
    graph = GraphCompiler(decl).compile(
        expression, error_policy="isolate" if mode == "isolate" else "raise"
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    no_library(result)
    actual = result.outputs[0].values[0]
    np.testing.assert_array_equal(
        actual if mode == "comparison" else actual.view(np.uint64),
        expected > 0 if mode == "comparison" else expected.view(np.uint64),
    )


@pytest.mark.parametrize("fault", [-0.0, np.nan, np.inf, 2.0**-129, 2.0**129, 1e9])
def test_high_amplitude_rejected_domain_exact_fallback(fault):
    a, b = inputs("cf", (512, 513, 512))
    a.setflags(write=True)
    a[-1, -1] = fault
    a.setflags(write=False)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    no_library(result)
    np.testing.assert_array_equal(
        result.outputs[0].values[0].view(np.uint64), op.matmul(a, b).view(np.uint64)
    )


@pytest.mark.skipif(not ACCELERATE, reason="enabled macOS 15 arm64 library")
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
def test_partitioned_padded_storage_ignores_unconsumed_nan(layout):
    original = inputs(layout, (513, 1025, 515))
    owners = []
    views = []
    for value, order in zip(original, layout, strict=False):
        shape = (
            (value.shape[0], value.shape[1] + 3)
            if order == "c"
            else (value.shape[0] + 3, value.shape[1])
        )
        owner = np.full(shape, np.nan, order=order.upper())
        view = owner[: value.shape[0], : value.shape[1]]
        view[:] = value
        view.flags.writeable = False
        owners.append(owner)
        views.append(view)
    a, b = views
    graph = GraphCompiler(declarations()).compile("matmul(a,b)", output_order="F")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert chunks(result)[0]["accelerate_matmul_calls"] > 1
    decimal_check(result.outputs[0].values[0], a, b)
    for owner, view in zip(owners, views, strict=False):
        assert np.isnan(owner).sum() == owner.size - view.size


def test_prepared_rereads_high_amplitude_input_and_preserves_previous_output():
    a, b = inputs("ff", (512, 513, 512))
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        a.setflags(write=True)
        a[-1, -1] = -0.0
        a.setflags(write=False)
        again = prepared.run_snapshot()
        no_library(again)
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
    np.testing.assert_array_equal(
        again.outputs[0].values[0].view(np.uint64), op.matmul(a, b).view(np.uint64)
    )
    assert not np.shares_memory(first.outputs[0].values[0], again.outputs[0].values[0])


@pytest.mark.skipif(not ACCELERATE, reason="enabled macOS 15 arm64 library")
def test_partitioned_scratch_and_vendor_workspace_both_admitted():
    a, b = inputs("cc", (512, 513, 512))
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    reserve = (1024 * 1024 + 512 * 513 + 320 * 513 + 72) * 8
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, {"a": a, "b": b}, bounds, bounds)
        assert plan.estimated_worker_scratch_bytes >= reserve
        with pytest.raises(
            (ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc|bad allocation"
        ):
            scheduler.execute(
                graph, {"a": a, "b": b}, bounds, bounds, memory_budget_bytes=reserve - 1
            )


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_high_amplitude_timeout_drains_then_scheduler_can_run(lane):
    a, b = inputs("ff", (512, 513, 512))
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        with pytest.raises(TimeoutError):
            scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds, timeout=0)
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds, timeout=20)
        assert scheduler.peak_active_cpu_tokens <= 2
    decimal_check(result.outputs[0].values[0], a, b)


@pytest.mark.skipif(not ACCELERATE, reason="enabled macOS 15 arm64 library")
def test_late_rejection_overwrites_entire_private_root():
    a = np.ones((512, 8192))
    b = np.zeros((8192, 512))
    a[:, :2048] = 1024.0
    b[:2048] = 1024.0
    b[2048:4096] = 2.0**-24
    a[:, 4096:6144] = 1024.0
    b[4096:6144] = -1024.0
    a.flags.writeable = b.flags.writeable = False
    graph = GraphCompiler(declarations()).compile("matmul(a,b)", output_order="F")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        snapshot = first.outputs[0].values[0].copy()
        again = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert chunks(first)[0]["accelerate_matmul_calls"] > 1
    np.testing.assert_array_equal(
        first.outputs[0].values[0].view(np.uint64), op.matmul(a, b).view(np.uint64)
    )
    np.testing.assert_array_equal(first.outputs[0].values[0], snapshot)
    assert not np.shares_memory(first.outputs[0].values[0], again.outputs[0].values[0])
    assert first.outputs[0].values[0].flags.f_contiguous
