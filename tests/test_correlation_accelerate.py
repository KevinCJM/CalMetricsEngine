"""Correlation-specific normalization bounds and symmetric independent roots."""
import ctypes
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_covariance_accelerate import ACTIVE, MATRIX, active, compare, counts, inputs
from .test_matmul_library import settings


def graph(order="C", expression="correlation(x)", isolate=False):
    return GraphCompiler({"x": MATRIX}).compile(
        expression, output_order=order, error_policy="isolate" if isolate else "raise",
    )


def execute(scheduler, compiled, x, rows=1):
    return scheduler.execute(compiled, {"x": x}, np.zeros(rows, np.int64),
                             np.full(rows, len(x), np.int64))


@pytest.mark.parametrize("layout", ["c", "f", "padded_c", "padded_f", "reverse_c", "reverse_f"])
@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_correlation_complete_result_transport_layout_and_budget(layout, order, lane):
    x = inputs(layout, n=129, d=257)
    before = x.view(np.uint64).copy()
    # Match the existing worker packing; local descending F stays ordered.
    transported = x
    if lane.startswith("process") and layout in {"padded_f", "reverse_f"}:
        transported = np.array(x, order="F" if layout == "reverse_f" else "C")
    expected = op.correlation(transported)
    compiled = pickle.loads(pickle.dumps(graph(order)))
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = execute(scheduler, compiled, x, 2)
        second = execute(scheduler, compiled, x, 2)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    if lane.startswith("process"):
        assert first.plan.use_shared_memory == (lane == "process_shared")
    assert counts(first, "accelerate_covariance_calls") == (2 if ACTIVE else 0)
    for actual in first.outputs[0].values:
        compare(actual, expected)
        np.testing.assert_allclose(actual, np.corrcoef(x.T), rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual.view(np.uint64), actual.T.view(np.uint64))
        assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
        assert not actual.flags.writeable and not np.shares_memory(actual, x)
    assert not np.shares_memory(*first.outputs[0].values)
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    np.testing.assert_array_equal(before, x.view(np.uint64))
    for chunk in first.audit["native_chunks"]:
        assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
        assert chunk["direct_output_bytes"] == chunk["rows"] * expected.nbytes
        if ACTIVE:
            reserve = (len(x) + 3) * x.shape[1] * 8 + 32 * 1024 * 1024
            assert chunk["native_library_workspace_reserve_bytes"] == reserve
            assert first.plan.estimated_worker_scratch_bytes >= reserve


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("pattern", ["small", "heterogeneous", "signed", "orthogonal", "scaled", "large_scale", "high_scale", "wide_scale", "source_edge", "source_reject", "nan", "inf", "reverse", "tiny"])
def test_normalized_error_not_covariance_absolute_tolerance(layout, pattern):
    x = inputs(layout, n=1024 if pattern == "orthogonal" else 513, d=256).copy(order=layout.upper())
    if pattern == "small":
        x *= 2.0**-200
    elif pattern in {"heterogeneous", "signed"}:
        x *= np.exp2(np.linspace(-380, 0, x.shape[1]))
        if pattern == "signed":
            x[:, ::2] *= -1
    elif pattern == "orthogonal":
        x[:, 0] = np.tile([1., 1., -1., -1.], len(x)//4)
        x[:, 1] = np.tile([1., -1., 1., -1.], len(x)//4)
        x[:, :2] *= 2.0**-200
    elif pattern in {"scaled", "large_scale", "high_scale"}:
        x *= {"scaled": 32., "large_scale": 1024., "high_scale": 16384.}[pattern]
    elif pattern == "wide_scale":
        x *= np.exp2(np.linspace(-360, 390, x.shape[1]))
        x[:, ::2] *= -1
    elif pattern in {"source_edge", "source_reject"}:
        x *= 2.0**396
        x[0, 0] = 2.0**400 if pattern == "source_edge" else np.nextafter(2.0**400, np.inf)
    elif pattern == "nan":
        x[3, -1] = np.nan
    elif pattern == "inf":
        x[3, -1] = np.inf
    elif pattern == "reverse":
        x = x[::-1]
    else:
        x *= 2.0**-450
    x.flags.writeable = False
    expected = op.correlation(x)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph("F"), x)
    actual = result.outputs[0].values[0]
    compare(actual, expected)
    if pattern not in {"nan", "inf"}:
        np.testing.assert_allclose(actual, np.corrcoef(x.T), rtol=1e-12, atol=1e-12)
    if ACTIVE and pattern in {"small", "heterogeneous", "signed", "orthogonal", "scaled", "large_scale", "high_scale", "wide_scale", "source_edge"}:
        assert counts(result, "accelerate_covariance_calls") == 1
    if ACTIVE and pattern == "reverse":
        assert counts(result, "accelerate_covariance_calls") == 1
    if pattern in {"source_reject", "tiny", "nan", "inf"} or (pattern == "reverse" and not ACTIVE):
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert counts(result, "covariance_zero_calls") == 0


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("layout", ["c", "f"])
def test_large_source_range_keeps_transport_and_covariance_contract(lane, layout):
    x = inputs(layout, n=129, d=256).copy(order=layout.upper()) * 2.0**200
    x.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = execute(scheduler, graph("F"), x, 2)
        covariance = execute(scheduler, graph("F", "covariance(x)"), x, 2)
    for actual in result.outputs[0].values:
        compare(actual, op.correlation(x))
        np.testing.assert_allclose(actual, np.corrcoef(x.T), rtol=1e-12, atol=1e-12)
    assert counts(result, "accelerate_covariance_calls") == (2 if ACTIVE else 0)
    assert counts(covariance, "accelerate_covariance_attempts") == 0
    for actual in covariance.outputs[0].values:
        np.testing.assert_array_equal(actual.view(np.uint64), op.covariance(x).view(np.uint64))


@pytest.mark.parametrize("layout", ["c", "f", "reverse_c", "reverse_f"])
@pytest.mark.parametrize("pattern", ["all_zero", "signed_zero", "constant_column", "large_constant_column"])
def test_invalid_variance_retains_error_and_prepared_recovery(layout, pattern):
    x = inputs(layout, n=65, d=256)
    if pattern == "large_constant_column":
        x = x * 2.0**200
        x.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph("F"), {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
        first = prepared.run_snapshot()
        saved, original = first.outputs[0].values[0].copy(), x.copy()
        x.setflags(write=True)
        if pattern in {"constant_column", "large_constant_column"}:
            x[:, -1] = 2.0**200 if pattern == "large_constant_column" else 2
        else:
            x.fill(-0. if pattern == "signed_zero" else 0.)
        x.setflags(write=False)
        with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
            op.correlation(x)
        with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
            prepared.run_snapshot()
        x.setflags(write=True)
        x[:] = original
        x.setflags(write=False)
        recovered = prepared.run_snapshot()
    compare(recovered.outputs[0].values[0], op.correlation(x))
    np.testing.assert_array_equal(first.outputs[0].values[0].view(np.uint64), saved.view(np.uint64))
    assert not np.shares_memory(first.outputs[0].values[0], recovered.outputs[0].values[0])


@pytest.mark.parametrize("mode", ["comparison", "multiple", "isolate", "iterate"])
def test_nonindependent_graphs_remain_canonical(mode):
    x = inputs(n=65, d=256)
    expected = op.correlation(x)
    expr = "correlation(x)>0" if mode == "comparison" else "correlation(x)"
    if mode == "multiple":
        expr = [expr, f"trace({expr})"]
    elif mode == "iterate":
        expr = f"iterate({expr},{expr},0,1)"
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph("F", expr, mode == "isolate"), x)
    assert counts(result, "accelerate_covariance_attempts") == 0
    actual = result.outputs[0].values[0]
    np.testing.assert_array_equal(actual if mode == "comparison" else actual.view(np.uint64),
                                  expected > 0 if mode == "comparison" else expected.view(np.uint64))


@active
@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("correlated", [False, True])
def test_long_normalization_bounds_replay_and_tail(layout, correlated):
    x = inputs(layout, n=8193, d=256)
    if correlated:
        x.setflags(write=True)
        x[:] = x*.1 + x[:, :1]
        x.setflags(write=False)
    expected = op.correlation(x)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph(), x)
    compare(result.outputs[0].values[0], expected)
    assert counts(result, "covariance_syrk_calls") == (len(x)+511)//512
    assert counts(result, "accelerate_covariance_calls") + counts(result, "covariance_fallbacks") == 1


@active
def test_correlation_budget_and_thread_policy_after_error():
    x = inputs(n=4096, d=256)
    starts, ends = np.array([0, 1], np.int64), np.array([4096, 1042], np.int64)
    compiled = graph("F")
    reserve = (2048+3)*256*8 + 32*1024*1024
    library = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    getter, setter = library.BLASGetThreading, library.BLASSetThreading
    getter.restype = ctypes.c_uint
    setter.argtypes, setter.restype = (ctypes.c_uint,), ctypes.c_int
    previous = getter()
    try:
        assert setter(0) == 0
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            plan = scheduler.plan(compiled, {"x": x}, starts, ends)
            assert plan.estimated_worker_scratch_bytes >= reserve
            with pytest.raises((ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc"):
                scheduler.execute(compiled, {"x": x}, starts, ends, memory_budget_bytes=reserve-1)
            result = scheduler.execute(compiled, {"x": x}, starts, ends)
            for first, last, actual in zip(starts, ends, result.outputs[0].values, strict=True):
                compare(actual, op.correlation(x[first:last]))
            x.setflags(write=True)
            x[:, -1] = 2
            x.setflags(write=False)
            with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
                execute(scheduler, compiled, x)
            assert getter() == 0
    finally:
        assert setter(previous) == 0


@active
@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("exponent", [-200, 200])
def test_power_of_two_scaling_keeps_normalized_certificate(layout, exponent):
    x = inputs(layout, n=32769, d=256).copy(order=layout.upper())
    x[:] = x*.1 + x[:, :1]
    x.flags.writeable = False
    scaled = np.array(x*2.0**exponent, order=layout.upper())
    scaled.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        original = execute(scheduler, graph(), x)
        small = execute(scheduler, graph(), scaled)
    # A covariance-only absolute bound would wrongly accept the tiny input.
    for result, data in ((original, x), (small, scaled)):
        assert counts(result, "covariance_fallbacks") == 1
        assert counts(result, "accelerate_covariance_calls") == 0
        np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64),
                                      op.correlation(data).view(np.uint64))
    assert counts(original, "covariance_replayed_cells") == counts(small, "covariance_replayed_cells")


@pytest.mark.parametrize("d", [0, 1, 17])
def test_small_symmetric_f_root_uses_same_budget_as_c(d):
    x = inputs("f", n=17, d=d)
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        c = scheduler.plan(graph("C"), {"x": x}, starts, ends)
        f = scheduler.plan(graph("F"), {"x": x}, starts, ends)
        assert c.estimated_total_memory_bytes == f.estimated_total_memory_bytes
        assert c.estimated_worker_scratch_bytes == f.estimated_worker_scratch_bytes
        with pytest.raises(MemoryError):
            scheduler.plan(graph("F"), {"x": x}, starts, ends,
                           memory_budget_bytes=f.estimated_total_memory_bytes-1)
        result = execute(scheduler, graph("F"), x)
    np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64),
                                  op.correlation(x).view(np.uint64))
    assert counts(result, "numeric_arena_bytes") == counts(result, "root_copy_bytes") == 0
