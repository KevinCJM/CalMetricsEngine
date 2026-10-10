"""Shared products: independent terminal roots, ordered fallback and reuse."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import declarations, settings
from .test_matmul_scalar_chain import payload
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks

EXPRESSIONS = ["matmul(a,b)*0.5+0.25", "matmul(a,b)*0.25-0.125"]


def expected(a, b):
    value = op.matmul(a, b)
    return [op.add(op.multiply(value, 0.5), 0.25), op.subtract(op.multiply(value, 0.25), 0.125)]


def check(result, wants, a, b, order, library, arena=True):
    assert len(result.outputs) == len(wants)
    arrays = []
    for root, want in zip(result.outputs, wants, strict=True):
        assert np.all(root.root_statuses == 0)
        assert all(status is None for status in root.statuses)
        for got in root.values:
            assert not got.flags.writeable
            assert got.flags.c_contiguous if order == "C" else got.flags.f_contiguous
            assert not np.shares_memory(got, a) and not np.shares_memory(got, b)
            np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
            np.testing.assert_array_equal(np.isposinf(got), np.isposinf(want))
            np.testing.assert_array_equal(np.isneginf(got), np.isneginf(want))
            np.testing.assert_array_equal(got == 0, want == 0)
            np.testing.assert_array_equal(np.signbit(got[want == 0]), np.signbit(want[want == 0]))
            np.testing.assert_allclose(got, want, rtol=0, atol=0.000049)
            arrays.append(got)
    for i, left in enumerate(arrays):
        for right in arrays[i + 1 :]:
            assert not np.shares_memory(left, right)
    for c in chunks(result):
        assert c["accelerate_matmul_calls"] == (c["rows"] if library and ACCELERATE else 0)
        if arena:
            assert c["numeric_arena_bytes"] == c["root_copy_bytes"] == 0
        observed = (
            c["numeric_arena_bytes"]
            + c["mask_arena_bytes"]
            + c["operator_workspace_capacity_bytes"]
            + c["native_library_workspace_reserve_bytes"]
        )
        assert observed <= result.plan.estimated_worker_scratch_bytes
    assert result.audit["python_operator_calls"] == result.audit["python_fallback"] == 0


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_shared_product_all_lanes_and_retained_generations(layout, order, lane):
    a, b = payload(layout)
    before = a.copy(), b.copy()
    wants = expected(a, b)
    g = GraphCompiler(declarations()).compile(EXPRESSIONS, output_order=order)
    z = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as s:
        first = s.execute(g, {"a": a, "b": b}, z, z)
        saved = [r.values[0].copy() for r in first.outputs]
        second = s.execute(g, {"a": a, "b": b}, z, z)
        assert s.peak_active_cpu_tokens <= 2
    for r in (first, second):
        check(r, wants, a, b, order, True)
    for i in range(2):
        np.testing.assert_array_equal(first.outputs[i].values[0], saved[i])
        assert all(
            not np.shares_memory(first.outputs[i].values[0], r.values[0]) for r in second.outputs
        )
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("roots", [2, 3, 8])
@pytest.mark.parametrize("reverse", [False, True])
def test_root_count_and_user_order_do_not_determine_backing(order, roots, reverse):
    a, b = payload("ff")
    m = op.matmul(a, b)
    expr = [f"matmul(a,b)*{(i + 1) / 16}+{(i + 1) / 8}" for i in range(roots)]
    wants = [op.add(op.multiply(m, (i + 1) / 16), (i + 1) / 8) for i in range(roots)]
    if reverse:
        expr.reverse()
        wants.reverse()
    g = GraphCompiler(declarations()).compile(expr, output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(g, {"a": a, "b": b}, z, z)
    check(r, wants, a, b, order, True)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "fault",
    [
        "zero",
        "negative_zero",
        "nan",
        "infinity",
        "small",
        "large",
        "reverse",
        "step",
        "broadcast",
        "pitched",
    ],
)
def test_rejected_library_inputs_keep_original_ieee_and_inputs(fault, order):
    a, b = payload("ff")
    a.setflags(write=True)
    if fault in ("zero", "negative_zero", "nan", "infinity", "small", "large"):
        a[2, 3] = {
            "zero": 0.0,
            "negative_zero": -0.0,
            "nan": np.nan,
            "infinity": np.inf,
            "small": 2.0**-129,
            "large": 2.0**129,
        }[fault]
    elif fault == "reverse":
        a = a[:, ::-1]
        b = b[::-1]
    elif fault in ("step", "pitched"):
        n = a.shape[1]
        owner = np.full((a.shape[0], 2 * n if fault == "step" else n + 3), np.nan)
        if fault == "step":
            owner[:, ::2] = a
            a = owner[:, ::2]
        else:
            owner[:, :n] = a
            a = owner[:, :n]
    else:
        a = np.broadcast_to(a[:1], a.shape)
    a.flags.writeable = b.flags.writeable = False
    before = a.copy(), b.copy()
    g = GraphCompiler(declarations()).compile(EXPRESSIONS, output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(g, {"a": a, "b": b}, z, z)
    wants = expected(a, b)
    check(r, wants, a, b, order, fault == "pitched")
    if fault != "pitched":
        for got, want in zip(r.outputs, wants, strict=True):
            np.testing.assert_array_equal(
                np.array(got.values[0], order="C").view(np.uint64),
                np.array(want, order="C").view(np.uint64),
            )
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("scale", [0.0, 2.0, 1e-20, -0.7])
def test_current_parameter_qualification_and_prepared_ownership(order, scale):
    a, b = payload("cf")
    decl = declarations()
    decl.update(s="scalar", t="scalar")
    expr = ["matmul(a,b)*s+0.25", "matmul(a,b)*t-0.125"]
    g = GraphCompiler(decl).compile(expr, output_order=order)
    z = np.zeros(1, np.int64)
    params = np.array([0.5, 0.25])
    with AdaptiveScheduler(cpu_budget=1) as s:
        prepared = s.prepare_execution(g, {"a": a, "b": b}, z, z, parameters=params)
        first = prepared.run_snapshot()
        saved = [r.values[0].copy() for r in first.outputs]
        params[1] = scale
        second = prepared.run_snapshot()
        third = s.execute(g, {"a": a, "b": b}, z, z, parameters={"s": 0.5, "t": scale})
    m = op.matmul(a, b)
    wants = [op.add(op.multiply(m, 0.5), 0.25), op.subtract(op.multiply(m, scale), 0.125)]
    check(first, expected(a, b), a, b, order, True)
    for r in (second, third):
        check(r, wants, a, b, order, abs(scale) >= 2.0**-24 and abs(scale) <= 1)
    for i in range(2):
        np.testing.assert_array_equal(first.outputs[i].values[0], saved[i])


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "mode", ["raw_root", "comparison", "divide", "shared_tail", "isolate", "many"]
)
def test_observable_or_unproven_graphs_keep_canonical_path(order, mode):
    a, b = payload("ff")
    expr = list(EXPRESSIONS)
    if mode == "raw_root":
        expr.append("matmul(a,b)")
    elif mode == "comparison":
        expr.append("matmul(a,b)>0.0")
    elif mode == "divide":
        expr[1] = "matmul(a,b)/2"
    elif mode == "shared_tail":
        expr[1] = "matmul(a,b)*0.5-0.125"
    elif mode == "many":
        expr = [f"matmul(a,b)*{(i + 1) / 16}+{i / 8}" for i in range(9)]
    g = GraphCompiler(declarations()).compile(
        expr, output_order=order, error_policy="isolate" if mode == "isolate" else "raise"
    )
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(g, {"a": a, "b": b}, z, z)
    assert all(c["accelerate_matmul_calls"] == 0 for c in chunks(r))
    if mode != "many":
        np.testing.assert_array_equal(r.outputs[0].values[0], expected(a, b)[0])


@pytest.mark.parametrize("order", ["C", "F"])
def test_each_root_replays_ambiguous_cancellation_separately(order):
    a, b = payload("ff")
    a.setflags(write=True)
    b.setflags(write=True)
    a.fill(0.01)
    b.fill(0.01)
    b[::2] *= -1
    a[0, :3] = [1.0, 2.0**-54, -1.0]
    b[:3, 0] = [1.0, 1.0, 1.0]
    a.flags.writeable = b.flags.writeable = False
    decl = declarations()
    decl.update(s="scalar", t="scalar")
    g = GraphCompiler(decl).compile(["matmul(a,b)*s+0.0", "matmul(a,b)*t-0.0"], output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        r = scheduler.execute(g, {"a": a, "b": b}, z, z, parameters={"s": -0.7, "t": 0.5})
    m = op.matmul(a, b)
    check(
        r,
        [op.add(op.multiply(m, -0.7), 0.0), op.subtract(op.multiply(m, 0.5), 0.0)],
        a,
        b,
        order,
        True,
    )
