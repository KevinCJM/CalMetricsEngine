"""Ordered Eigen packing writes independent F roots, including rectangular tails."""
import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import declarations, settings
from .test_matrix_library_backend import PACKING_PLATFORM


def graph():
    return GraphCompiler(declarations()).compile("matmul(a,b)", result_format="typed", output_order="F")


def integers(shape, layout):
    m, k, n = shape
    rows = (np.arange(m) % 5 - 2).astype(np.float64)
    columns = (np.arange(n) % 7 - 3).astype(np.float64)
    signs = np.where(np.arange(k) % 2, -1., 1.)
    a = np.array(rows[:, None] * signs[None, :] * 1024., order=layout[0].upper())
    b = np.array(signs[:, None] * columns[None, :] * 1024., order=layout[1].upper())
    expected = rows[:, None] * columns[None, :] * k * 1024. ** 2
    a.flags.writeable = b.flags.writeable = False
    return a, b, expected


def execute_and_check(a, b, expected, lane="single", check_packing=True):
    snapshots = [v.copy() for v in (a, b)]
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(graph(), {"a": a, "b": b}, bounds, bounds)
        old = first.outputs[0].values[0].copy()
        second = scheduler.execute(graph(), {"a": a, "b": b}, bounds, bounds)
    for result in (first, second):
        for v in result.outputs[0].values:
            np.testing.assert_array_equal(v, expected)
            assert v.flags.f_contiguous and not v.flags.writeable
            assert not np.shares_memory(v, a) and not np.shares_memory(v, b)
        for chunk in result.audit["native_chunks"]:
            assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
            assert chunk["accelerate_matmul_calls"] == 0
        if check_packing and PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN") == "1":
            m, k = a.shape
            prefix, bulk = m // 4 * 4, b.shape[1] // 6 * 6
            copied = (prefix + ((prefix + 127) // 128) * bulk) * k * 8
            assert sum(c["algorithm_copy_bytes"] for c in result.audit["native_chunks"]) == 2 * copied
            assert max(c["operator_workspace_capacity_bytes"] for c in result.audit["native_chunks"]) == (320 * k + 72) * 8
    np.testing.assert_array_equal(first.outputs[0].values[0], old)
    for i, v in enumerate(first.outputs[0].values):
        assert not np.shares_memory(v, first.outputs[0].values[1 - i])
        assert not np.shares_memory(v, second.outputs[0].values[i])
    for v, saved in zip((a, b), snapshots, strict=True):
        np.testing.assert_array_equal(v, saved)
    return first


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("shape", [(32, 33, 128), (33, 33, 129), (35, 33, 131),
                                    (36, 32, 132), (63, 65, 133), (65, 127, 253)])
def test_rectangular_f_pack_all_axes_and_remainders(shape, layout):
    execute_and_check(*integers(shape, layout))


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
def test_large_f_pack_independent_integer_formula(layout):
    execute_and_check(*integers((1025, 257, 1027), layout))


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("case", ["cancellation", "fma", "overflow", "zero_inf", "signed_tiny"])
def test_f_pack_preserves_canonical_ieee_and_zero_classes(layout, case):
    a = np.zeros((35, 33), order=layout[0].upper())
    b = np.ones((33, 131), order=layout[1].upper())
    if case == "cancellation":
        a[:, :3] = [1e16, 1., -1e16]
    elif case == "fma":
        a[:, :2] = [-1., 1 + 2 ** -27]
        b[:2] = np.array([1., 1 - 2 ** -27])[:, None]
    elif case == "overflow":
        a[:, :3] = [1e308, 1e308, -1e308]
        b[:3] = 2.
    elif case == "zero_inf":
        b[7] = np.inf
    else:
        a.fill(-0.)
        a[0, 0] = a[-1, 0] = np.nextafter(0., 1.)
    # Independent NumPy elementwise left fold; no dot/GEMM/BLAS reference.
    expected = np.zeros((a.shape[0], b.shape[1]))
    with np.errstate(all="ignore"):
        for k in range(a.shape[1]):
            expected += a[:, k, None] * b[k, None, :]
    a.flags.writeable = b.flags.writeable = False
    result = execute_and_check(a, b, expected)
    actual = np.array(result.outputs[0].values[0], order="C")
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_f_pack_borrows_padded_positive_layout(layout, operand):
    a, b, expected = integers((35, 33, 131), layout)
    source = a if operand == "a" else b
    shape = source.shape
    expanded = (shape[0], shape[1] + 7) if layout[0] == "c" else (shape[0] + 7, shape[1])
    owner = np.full(expanded, np.nan, order=layout[0].upper())
    view = owner[:shape[0], :shape[1]]
    view[:] = source
    owner.flags.writeable = view.flags.writeable = False
    before = owner.copy()
    execute_and_check(view if operand == "a" else a, view if operand == "b" else b, expected)
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_f_pack_worker_transport_and_result_lifetime(lane):
    execute_and_check(*integers((35, 33, 131), "ff"), lane=lane)


@pytest.mark.parametrize("mode", ["negative", "broadcast"])
def test_unadmitted_f_input_keeps_canonical_fallback(mode):
    a, b, expected = integers((35, 33, 131), "cc")
    if mode == "negative":
        a, b = a[:, ::-1], b[::-1]
    else:
        a = np.broadcast_to(a[:1], a.shape)
        expected = np.broadcast_to(expected[:1], expected.shape)
    execute_and_check(a, b, expected, check_packing=False)


def test_f_pack_memory_budget_includes_existing_panels():
    a, b, expected = integers((35, 33, 131), "ff")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        planned = scheduler.plan(graph(), {"a": a, "b": b}, bounds, bounds)
        with pytest.raises((ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc"):
            scheduler.execute(graph(), {"a": a, "b": b}, bounds, bounds, memory_budget_bytes=1)
    assert planned.estimated_worker_scratch_bytes >= op.get("matmul").requirements(a, b)["scratch_doubles"] * 8
