"""Existing ordered Eigen tiles write F roots outside packing admission."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler
from calmetrics_engine import operators as op

from .test_matmul_library import settings
from .test_packed_f_root import execute_and_check, graph, integers


def fold(a, b):
    expected = np.zeros((a.shape[0], b.shape[1]))
    with np.errstate(all="ignore"):
        for k in range(a.shape[1]):
            expected += a[:, k, None] * b[k, None, :]
    return expected


def ordered_integers(shape, layout):
    # The closed-form product is exact in value, but its signed zeros are not
    # the result of the canonical +0, ascending-K accumulation.
    a, b, closed = integers(shape, layout)
    expected = fold(a, b)
    np.testing.assert_array_equal(expected, closed)
    return a, b, expected


def check(a, b, expected, lane="single"):
    result = execute_and_check(a, b, expected, lane=lane, check_packing=False)
    assert op.get("matmul").requirements(a, b)["scratch_doubles"] == 0
    for chunk in result.audit["native_chunks"]:
        assert chunk["algorithm_copy_bytes"] == 0
        assert chunk["operator_workspace_capacity_bytes"] == 0
    actual = np.array(result.outputs[0].values[0], order="C")
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(
        np.signbit(actual[~np.isnan(expected)]), np.signbit(expected[~np.isnan(expected)])
    )


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("shape", [(32, 683, 32), (33, 689, 35), (35, 1025, 37), (65, 1025, 67)])
def test_cache_panels_and_root_stride_exact(shape, layout):
    check(*ordered_integers(shape, layout))


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("case", ["cancellation", "fma", "overflow", "zero_inf", "signed_tiny"])
def test_cache_carries_ieee_classes_across_panels(layout, case):
    a = np.zeros((33, 689), order=layout[0].upper())
    b = np.ones((689, 35), order=layout[1].upper())
    if case == "cancellation":
        a[:, [0, 64, 688]] = [1e16, 1.0, -1e16]
    elif case == "fma":
        a[:, [0, 64]] = [-1.0, 1 + 2**-27]
        b[[0, 64]] = np.array([1.0, 1 - 2**-27])[:, None]
    elif case == "overflow":
        a[:, [0, 64, 688]] = [1e308, 1e308, -1e308]
        b[[0, 64, 688]] = 2.0
    elif case == "zero_inf":
        b[64] = np.inf
    else:
        a.fill(-0.0)
        a[0, 64] = a[-1, 64] = np.nextafter(0.0, 1.0)
    expected = fold(a, b)
    a.flags.writeable = b.flags.writeable = False
    check(a, b, expected)


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_cache_borrows_padded_operands(layout, operand):
    a, b, expected = ordered_integers((35, 689, 37), layout)
    source = a if operand == "a" else b
    order = layout[0 if operand == "a" else 1].upper()
    shape = source.shape
    expanded = (shape[0], shape[1] + 7) if order == "C" else (shape[0] + 7, shape[1])
    owner = np.full(expanded, np.nan, order=order)
    view = owner[: shape[0], : shape[1]]
    view[:] = source
    owner.flags.writeable = view.flags.writeable = False
    before = owner.copy()
    check(view if operand == "a" else a, view if operand == "b" else b, expected)
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("mode", ["negative_a", "negative_b", "broadcast"])
def test_cache_alias_geometry_keeps_canonical_fold(layout, mode):
    a, b, _ = ordered_integers((35, 689, 37), layout)
    if mode == "negative_a":
        a = a[::-1]
    elif mode == "negative_b":
        a, b = a[:, ::-1], b[::-1]
    else:
        a = np.broadcast_to(a[:1], a.shape)
    check(a, b, fold(a, b))


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_cache_worker_and_independent_generation(lane):
    check(*ordered_integers((35, 689, 37), "ff"), lane=lane)


def test_cache_still_enforces_root_result_budget():
    a, b, _ = ordered_integers((35, 689, 37), "cc")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        with pytest.raises(
            (ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc|bad allocation"
        ):
            scheduler.execute(graph(), {"a": a, "b": b}, bounds, bounds, memory_budget_bytes=1)
