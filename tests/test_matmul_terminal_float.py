"""Bounded private mixed precision: full results, fallbacks and ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_matmul_library import declarations, settings
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks


def inputs(layout):
    rng = np.random.default_rng(2026100866)
    a = np.array(rng.uniform(-0.01, 0.01, (512, 256)), order=layout[0].upper())
    b = np.array(rng.uniform(-0.01, 0.01, (256, 512)), order=layout[1].upper())
    a.flags.writeable = b.flags.writeable = False
    return a, b


def execute(a, b, order, expressions, lane="single", parameters=None):
    decl = declarations()
    if parameters:
        decl.update({n: "scalar" for n in parameters})
    g = GraphCompiler(decl).compile(expressions, output_order=order)
    z = np.zeros(1 if lane == "single" else 2, np.int64)
    with AdaptiveScheduler(
        cpu_budget=1 if lane == "single" else 2, config=settings(lane)
    ) as scheduler:
        first = scheduler.execute(g, {"a": a, "b": b}, z, z, parameters=parameters)
        saved = [[v.copy() for v in r.values] for r in first.outputs]
        second = scheduler.execute(g, {"a": a, "b": b}, z, z, parameters=parameters)
    arrays = [v for r in first.outputs for v in r.values]
    for index, r in enumerate(first.outputs):
        got = r.values[0]
        np.testing.assert_array_equal(got, saved[index][0])
        for j, value in enumerate(r.values):
            np.testing.assert_array_equal(value, saved[index][j])
            assert not value.flags.writeable
            assert all(not np.shares_memory(value, v) for s in second.outputs for v in s.values)
        assert not got.flags.writeable and got.dtype == np.float64
        assert got.flags.c_contiguous if order == "C" else got.flags.f_contiguous
        assert not np.shares_memory(got, a) and not np.shares_memory(got, b)
        assert r.statuses[0] is None and r.root_statuses[0] == 0
        assert all(not np.shares_memory(got, s.values[0]) for s in second.outputs)
    for j, value in enumerate(arrays):
        assert all(not np.shares_memory(value, other) for other in arrays[j + 1 :])
    return first


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_private_conversion_all_layouts_and_workers(layout, order, lane):
    a, b = inputs(layout)
    before = a.copy(), b.copy()
    r = execute(a, b, order, ["matmul(a,b)*0.5+0.25", "matmul(a,b)*0.25-0.125"], lane)
    raw = a @ b
    for out, want in zip(r.outputs, [raw * 0.5 + 0.25, raw * 0.25 - 0.125], strict=True):
        for value in out.values:
            np.testing.assert_allclose(value, want, rtol=0, atol=0.000049)
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])
    for c in chunks(r):
        if ACCELERATE:
            assert c["algorithm_copy_bytes"] == c["rows"] * ((a.size + b.size) * 4 + raw.size * 8)
        assert c["numeric_arena_bytes"] == 0
        observed = (
            c["numeric_arena_bytes"]
            + c["mask_arena_bytes"]
            + c["operator_workspace_capacity_bytes"]
            + c["native_library_workspace_reserve_bytes"]
        )
        assert observed <= r.plan.estimated_worker_scratch_bytes


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("roots", [1, 2, 3, 8])
@pytest.mark.parametrize("reverse", [False, True])
def test_negative_scalars_reversed_subtract_and_root_order(order, roots, reverse):
    a, b = inputs("ff")
    raw = a @ b
    expr = []
    wants = []
    parameters = {}
    for i in range(roots):
        scale = -(i + 1) / 16
        shift = (i + 1) / 8
        parameters[f"s{i}"] = scale
        if i % 2:
            expr.append(f"{shift}-matmul(a,b)*s{i}")
            wants.append(shift - raw * scale)
        else:
            expr.append(f"matmul(a,b)*s{i}+{shift}")
            wants.append(raw * scale + shift)
    if reverse:
        expr.reverse()
        wants.reverse()
    r = execute(a, b, order, expr, parameters=parameters)
    for out, want in zip(r.outputs, wants, strict=True):
        np.testing.assert_allclose(out.values[0], want, rtol=0, atol=0.000049)
    if ACCELERATE:
        assert all(
            c["algorithm_copy_bytes"] == (a.size + b.size) * 4 + raw.size * 8 for c in chunks(r)
        )


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "case", ["below_grid", "large", "zero_interval", "no_pack_space", "padded"]
)
def test_rejected_precision_or_storage_preserves_double_route(order, case):
    a, b = inputs("ff")
    expressions = ["matmul(a,b)*0.5+0.25", "matmul(a,b)*0.25-0.125"]
    if case in ["below_grid", "large"]:
        a = a.copy(order="F")
        a[2, 3] = 2.0**-40 if case == "below_grid" else 2.0
    elif case == "zero_interval":
        expressions = ["matmul(a,b)*0.5+0.0", "matmul(a,b)*0.25-0.0"]
    elif case == "no_pack_space":
        a = a[:256, :]
    elif case == "padded":
        owner = np.full((a.shape[0], a.shape[1] + 3), np.nan)
        owner[:, : a.shape[1]] = a
        a = owner[:, : a.shape[1]]
    a.flags.writeable = b.flags.writeable = False
    r = execute(a, b, order, expressions)
    raw = a @ b
    wants = (
        [raw * 0.5, raw * 0.25]
        if case == "zero_interval"
        else [raw * 0.5 + 0.25, raw * 0.25 - 0.125]
    )
    for out, want in zip(r.outputs, wants, strict=True):
        np.testing.assert_allclose(out.values[0], want, rtol=0, atol=1e-11)
        np.testing.assert_array_equal(out.values[0] == 0, want == 0)
    if ACCELERATE:
        assert all(c["algorithm_copy_bytes"] == 0 for c in chunks(r))
