"""Independent float-root BLAS admission, fallback and worker ownership."""

import os
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import (
    assert_ieee_classes_and_close,
    declarations,
    settings,
)

ACCELERATE = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
    and int(platform.mac_ver()[0].split(".")[0]) >= 15
    and os.environ.get("CME_EXPECT_ACCELERATE") != "0"
)
active = pytest.mark.skipif(not ACCELERATE, reason="enabled macOS 15 arm64 library")


def inputs(layout="cc", shape=(513, 257, 515), signed=False):
    m, k, n = shape
    rng = np.random.default_rng(903021)
    a = np.array(rng.random((m, k)), order=layout[0].upper())
    b = np.array(rng.random((k, n)), order=layout[1].upper())
    if signed:
        a[::2] *= -1
        b[:, ::2] *= -1
    a.flags.writeable = b.flags.writeable = False
    return a, b


def chunks(result):
    return result.audit["native_chunks"]


def no_library(result):
    assert all(c["accelerate_matmul_calls"] == c["matmul_zero_calls"] == 0
               for c in chunks(result))


@active
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_independent_root_library_and_worker_layout(layout, order, transposed, lane, signed):
    a, b = inputs(layout, signed=signed)
    before = a.copy(), b.copy()
    expected = op.matmul(a, b)
    expr = "transpose(matmul(a,b))" if transposed else "matmul(a,b)"
    if transposed:
        expected = expected.T
    graph = GraphCompiler(declarations()).compile(expr, output_order=order)
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        again = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    for value in result.outputs[0].values:
        assert_ieee_classes_and_close(value, expected)
        assert not value.flags.writeable
        assert not np.shares_memory(value, a) and not np.shares_memory(value, b)
        if order == "F" or (order == "auto" and transposed):
            assert value.flags.f_contiguous
        else:
            assert value.flags.c_contiguous
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[0].values[1])
    assert not np.shares_memory(result.outputs[0].values[0], again.outputs[0].values[0])
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])
    assert sum(c["accelerate_matmul_calls"] for c in chunks(result)) == 2
    reserve = 8 * 1024 * 1024 + 512 * a.shape[1] * 8
    assert result.plan.estimated_worker_scratch_bytes >= reserve
    for c in chunks(result):
        assert c["native_library_workspace_reserve_bytes"] == reserve
        assert c["algorithm_copy_bytes"] == c["operator_workspace_capacity_bytes"] == 0
        if transposed and order in {"auto", "F"}:
            assert c["root_copy_bytes"] == c["numeric_arena_bytes"] == 0


@active
@pytest.mark.parametrize("signed", [False, True])
def test_prepared_rereads_domain_and_releases_old_ordered_panels(signed):
    a, b = inputs("ff", (1024, 257, 1024), signed=signed)
    graph = GraphCompiler(declarations()).compile("transpose(matmul(a,b))")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, bounds, bounds)
        first = prepared.run_snapshot()
        preserved = first.outputs[0].values[0].copy()
        a.setflags(write=True)
        a[0, 0] = -0.0
        a.setflags(write=False)
        fallback = prepared.run_snapshot()
        no_library(fallback)
        np.testing.assert_array_equal(fallback.outputs[0].values[0], op.matmul(a, b).T)
        assert chunks(fallback)[0]["operator_workspace_capacity_bytes"] > 0
        a.setflags(write=True)
        a.fill(0.0)
        a.setflags(write=False)
        zero = prepared.run_snapshot()
        assert chunks(zero)[0]["matmul_zero_calls"] == 1
        assert chunks(zero)[0]["accelerate_matmul_calls"] == 0
        assert chunks(zero)[0]["operator_workspace_capacity_bytes"] == 0
        np.testing.assert_array_equal(zero.outputs[0].values[0].view(np.uint64), 0)
        np.testing.assert_array_equal(first.outputs[0].values[0], preserved)


@pytest.mark.parametrize("fault", [-1.0, -0.0, np.inf, np.nan, 2.0 ** -129, 2.0 ** 129])
@pytest.mark.parametrize("operand", [0, 1])
def test_domain_rejection_preserves_original_ieee_results(fault, operand):
    a, b = inputs("cf")
    changed = (a, b)[operand]
    changed.setflags(write=True)
    changed[2, 3] = fault
    changed.setflags(write=False)
    expected = op.matmul(a, b)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    if fault == -1.0 and ACCELERATE:
        assert chunks(result)[0]["accelerate_matmul_calls"] == 1
    else:
        no_library(result)
    assert_ieee_classes_and_close(result.outputs[0].values[0], expected)


@pytest.mark.parametrize("layout", ["reverse", "step", "broadcast"])
def test_unsupported_strides_preserve_original_results(layout):
    a, b = inputs()
    if layout == "reverse":
        a, b = a[:, ::-1], b[::-1]
    elif layout == "step":
        owner = np.empty((a.shape[0], 2 * a.shape[1]))
        owner[:, ::2] = a
        a = owner[:, ::2]
    else:
        a = np.broadcast_to(a[:1], a.shape)
    a.flags.writeable = False
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    no_library(result)
    np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64),
                                  op.matmul(a, b).view(np.uint64))


@active
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("layout", ["cc", "ff"])
def test_padded_storage_ignores_unconsumed_nan_padding(layout, signed):
    a, b = inputs(layout, signed=signed)
    owners = []
    for value, order in ((a, layout[0]), (b, layout[1])):
        if order == "c":
            owner = np.full((value.shape[0], value.shape[1] + 3), np.nan)
            view = owner[:, :value.shape[1]]
        else:
            owner = np.full((value.shape[0] + 3, value.shape[1]), np.nan, order="F")
            view = owner[:value.shape[0]]
        view[:] = value
        view.flags.writeable = False
        owners.append((owner, view))
    a, b = owners[0][1], owners[1][1]
    expected = op.matmul(a, b)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert chunks(result)[0]["accelerate_matmul_calls"] == 1
    assert_ieee_classes_and_close(result.outputs[0].values[0], expected)
    for owner, view in owners:
        assert np.isnan(owner).sum() == owner.size - view.size


@pytest.mark.parametrize("mode", ["comparison", "multiple", "isolate", "iterate"])
@pytest.mark.parametrize("signed", [False, True])
def test_discrete_shared_isolated_and_iterated_consumers_remain_ordered(mode, signed):
    a, b = inputs("cc", (512, 512, 512), signed=signed)
    decl = declarations()
    if mode == "comparison":
        expr = "matmul(a,b)>0"
        expected = op.matmul(a, b) > 0
    elif mode == "multiple":
        decl["b"]["shape"] = ["K", "M"]
        expr = ["matmul(a,b)", "trace(matmul(a,b))"]
        expected = op.matmul(a, b)
    elif mode == "iterate":
        decl["b"]["shape"] = ["K", "K"]
        expr = "iterate(matmul(iterate_x,b),a,0,2)"
        expected = op.matmul(op.matmul(a, b), b)
    else:
        expr = "matmul(a,b)"
        expected = op.matmul(a, b)
    graph = GraphCompiler(decl).compile(expr, error_policy="isolate" if mode == "isolate" else "raise")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    no_library(result)
    actual = result.outputs[0].values[0]
    np.testing.assert_array_equal(actual if mode == "comparison" else actual.view(np.uint64),
                                  expected if mode == "comparison" else expected.view(np.uint64))


@active
@pytest.mark.parametrize("value", [2.0 ** -128, 2.0 ** 128, -2.0 ** -128, -2.0 ** 128, 0.0])
def test_boundaries_and_zeros_follow_existing_matmul_tolerance(value):
    a, b = inputs("fc")
    a.setflags(write=True)
    b.setflags(write=True)
    a.fill(value)
    b.fill(value)
    a.setflags(write=False)
    b.setflags(write=False)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert_ieee_classes_and_close(result.outputs[0].values[0], op.matmul(a, b))
    key = "matmul_zero_calls" if value == 0 else "accelerate_matmul_calls"
    assert chunks(result)[0][key] == 1


@active
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
def test_signed_sparse_dots_preserve_positive_zero(layout):
    a, b = inputs(layout, signed=True)
    a.setflags(write=True)
    b.setflags(write=True)
    # Leading zeros, all-zero row/column and disjoint support at [2,2].
    a[:, 0] = b[0, :] = 0.0
    a[0, :] = b[:, 0] = 0.0
    a[2, 128:] = b[:128, 2] = 0.0
    a.setflags(write=False)
    b.setflags(write=False)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert chunks(result)[0]["accelerate_matmul_calls"] == 1
    actual = result.outputs[0].values[0]
    assert_ieee_classes_and_close(actual, op.matmul(a, b))
    assert actual[2, 2] == 0 and not np.signbit(actual[2, 2])


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("fault", ["mixed_sign", "negative_zero", "low", "high"])
def test_signed_rejection_preserves_canonical_output(operand, fault):
    a, b = inputs("fc", signed=True)
    changed = (a, b)[operand]
    changed.setflags(write=True)
    changed[2, 2] = {"mixed_sign": 1.0, "negative_zero": -0.0,
                     "low": -2.0 ** -129, "high": -2.0 ** 129}[fault]
    changed.setflags(write=False)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    if fault == "mixed_sign" and ACCELERATE:
        assert chunks(result)[0]["accelerate_matmul_calls"] == 1
        assert_ieee_classes_and_close(result.outputs[0].values[0], op.matmul(a, b))
    else:
        no_library(result)
        np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64),
                                      op.matmul(a, b).view(np.uint64))


@active
def test_signed_max_k_and_large_exponent_span():
    rng = np.random.default_rng(1945)
    a, b = inputs("cf", (512, 4096, 512), signed=True)
    for value in (a, b):
        value.setflags(write=True)
        value[:] = np.copysign(np.ldexp(1.0, rng.integers(-128, 129, value.shape)), value)
        value.setflags(write=False)
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    assert chunks(result)[0]["accelerate_matmul_calls"] == 1
    assert_ieee_classes_and_close(result.outputs[0].values[0], op.matmul(a, b))


@active
def test_planner_rejects_before_library_when_reserve_does_not_fit():
    a, b = inputs()
    graph = GraphCompiler(declarations()).compile("matmul(a,b)")
    bounds = np.zeros(1, np.int64)
    reserve = 8 * 1024 * 1024 + 512 * a.shape[1] * 8
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, {"a": a, "b": b}, bounds, bounds)
        assert plan.estimated_worker_scratch_bytes >= reserve
        with pytest.raises((ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc|bad allocation"):
            scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds, memory_budget_bytes=reserve - 1)
