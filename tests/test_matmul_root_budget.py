"""Optional mature-library paths must preserve the old admissible budget."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig, SharedInputBundle

from .test_matmul_library import declarations
from .test_matmul_long_rectangle import SHAPES, calls, payload, readonly
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks

BASE = json.loads(Path(__file__).with_name("matmul_original_budget_cases.json").read_text())[
    "cases"
]
REASON = "memory_budget_uses_original_matmul_geometry"


def config(lane):
    return PlannerConfig(
        thread_work_units=1e100 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=1,
    )


def old(shape, order, lane):
    return BASE[f"{shape[0]},{shape[1]},{shape[2]}:{order}:{lane}"]


def bound(result):
    for c in chunks(result):
        observed = (
            c["numeric_arena_bytes"]
            + c["mask_arena_bytes"]
            + c["operator_workspace_capacity_bytes"]
            + c["native_library_workspace_reserve_bytes"]
        )
        assert observed <= result.plan.estimated_worker_scratch_bytes


def check(result, a, b, order, library, constrained):
    assert calls(result) == (len(result.outputs[0].values) if library and ACCELERATE else 0)
    assert (REASON in result.plan.reason_codes) == bool(constrained and ACCELERATE)
    assert result.audit["python_operator_calls"] == result.audit["python_fallback"] == 0
    bound(result)
    for value in result.outputs[0].values:
        assert not value.flags.writeable
        assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
        assert not np.shares_memory(value, a) and not np.shares_memory(value, b)
        np.testing.assert_allclose(value, a @ b, rtol=0, atol=0.000049)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_original_budget_restores_lane_and_worker_repeated_requests(shape, layout, order, lane):
    a, b = payload(shape, layout)
    readonly(a, b)
    before = a.tobytes(), b.tobytes()
    g = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    legacy = old(shape, order, lane)
    z = np.zeros(2, np.int64)
    results = []
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as s:
        for budget in (None, legacy["budget_bytes"], None, legacy["budget_bytes"]):
            r = s.execute(g, {"a": a, "b": b}, z, z, memory_budget_bytes=budget)
            results.append(r)
            check(r, a, b, order, library=budget is None, constrained=budget is not None)
            if budget is not None:
                assert r.plan.lane == legacy["lane"]
                workers = (
                    r.plan.process_count if lane.startswith("process") else r.plan.thread_count
                )
                assert workers == legacy["workers"]
                assert r.plan.estimated_total_memory_bytes <= budget
        assert s.peak_active_cpu_tokens <= 2
    assert before == (a.tobytes(), b.tobytes())
    for i, r in enumerate(results):
        for later in results[i + 1 :]:
            assert not any(
                np.shares_memory(x, y) for x in r.outputs[0].values for y in later.outputs[0].values
            )
    np.testing.assert_allclose(results[0].outputs[0].values[0], a @ b, rtol=0, atol=0.000049)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_budget_policy_is_per_plan_with_independent_snapshots(shape, layout, order):
    a, b = payload(shape, layout)
    readonly(a, b)
    z = np.zeros(2, np.int64)
    g = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=2, config=config("single")) as s:
        low_plan = s.plan(
            g,
            {"a": a, "b": b},
            z,
            z,
            memory_budget_bytes=old(shape, order, "single")["budget_bytes"],
        )
        low = s.prepare_execution(g, {"a": a, "b": b}, z, z, plan=low_plan)
        high = s.prepare_execution(g, {"a": a, "b": b}, z, z)
        first = low.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        second = high.run_snapshot()
        last = low.run_snapshot()
    check(first, a, b, order, False, True)
    check(second, a, b, order, True, False)
    check(last, a, b, order, False, True)
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
    assert not np.shares_memory(first.outputs[0].values[0], last.outputs[0].values[0])


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_same_logical_graph_concurrent_budgets_do_not_leak(layout, order):
    shape = SHAPES[0]
    a, b = payload(shape, layout)
    readonly(a, b)
    z = np.zeros(2, np.int64)
    g = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)

    async def run():
        with AdaptiveScheduler(cpu_budget=2, config=config("single")) as s:
            budgets = [old(shape, order, "single")["budget_bytes"], None] * 2
            jobs = [
                dict(graph=g, inputs={"a": a, "b": b}, starts=z, ends=z, memory_budget_bytes=budget)
                for budget in budgets
            ]
            results = await s.execute_many_async(jobs)
            assert s.peak_active_cpu_tokens <= 2
        for r, budget in zip(results, budgets, strict=True):
            check(r, a, b, order, budget is None, budget is not None)
        assert not np.shares_memory(
            results[0].outputs[0].values[0], results[2].outputs[0].values[0]
        )

    asyncio.run(run())


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_shared_output_budget_policy_and_old_owner(layout, order):
    shape = SHAPES[0]
    a, b = payload(shape, layout)
    readonly(a, b)
    z = np.zeros(2, np.int64)
    g = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=2, config=config("process_shared")) as s:
        first = s.execute(
            g,
            {"a": a, "b": b},
            z,
            z,
            memory_budget_bytes=old(shape, order, "process_shared")["budget_bytes"],
        )
        saved = first.outputs[0].values[0].copy()
        second = s.execute(g, {"a": a, "b": b}, z, z)
        assert first.plan.use_shared_memory and second.plan.use_shared_memory
    check(first, a, b, order, False, True)
    check(second, a, b, order, True, False)
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_pre_shared_inputs_keep_budget_policy_and_retained_outputs(layout, order):
    shape = SHAPES[0]
    a, b = payload(shape, layout)
    readonly(a, b)
    z = np.zeros(2, np.int64)
    g = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    budget = BASE[f"{shape[0]},{shape[1]},{shape[2]}:{order}:pre_shared"]["budget_bytes"]
    with SharedInputBundle.from_inputs({"a": a, "b": b}) as inputs:
        with AdaptiveScheduler(cpu_budget=2, config=config("process_shared")) as s:
            first = s.execute(g, inputs, z, z, memory_budget_bytes=budget)
            saved = first.outputs[0].values[0].copy()
            high = s.execute(g, inputs, z, z)
            last = s.execute(g, inputs, z, z, memory_budget_bytes=budget)
        for r in (first, high, last):
            assert r.plan.use_shared_memory
            assert r.audit["boundary_copy_bytes"] == z.nbytes * 2
    check(first, a, b, order, False, True)
    check(high, a, b, order, True, False)
    check(last, a, b, order, False, True)
    assert first.plan.estimated_total_memory_bytes <= budget
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
    assert not np.shares_memory(first.outputs[0].values[0], last.outputs[0].values[0])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_below_original_required_budget_still_fails_and_can_recover(shape, layout, order):
    a, b = payload(shape, layout)
    readonly(a, b)
    z = np.zeros(2, np.int64)
    g = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=2, config=config("single")) as s:
        low = s.execute(
            g,
            {"a": a, "b": b},
            z,
            z,
            memory_budget_bytes=old(shape, order, "single")["budget_bytes"],
        )
        with pytest.raises(MemoryError):
            s.execute(
                g,
                {"a": a, "b": b},
                z,
                z,
                memory_budget_bytes=low.plan.estimated_total_memory_bytes - 1,
            )
        recovered = s.execute(g, {"a": a, "b": b}, z, z)
    check(recovered, a, b, order, True, False)


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_original_geometry_keeps_library_and_its_original_budget(layout, order):
    shape = (512, 256, 512)
    a, b = payload(shape, layout)
    readonly(a, b)
    z = np.zeros(2, np.int64)
    g = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=2, config=config("single")) as s:
        r = s.execute(
            g,
            {"a": a, "b": b},
            z,
            z,
            memory_budget_bytes=old(shape, order, "single")["budget_bytes"],
        )
    check(r, a, b, order, True, False)


@pytest.mark.parametrize("layout", ["cc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_mixed_temporal_geometries_use_one_consistent_budget_policy(layout, order):
    a, b = payload((16385, 256, 32), layout)
    readonly(a, b)
    kinds = {
        "a": dict(kind="matrix", axes=["time", "factor"], shape=["T", "K"]),
        "b": dict(kind="matrix", axes=["factor", "asset"], shape=["K", "N"]),
    }
    g = GraphCompiler(kinds).compile("matmul(a,b)", output_order=order)
    starts = np.array([0, 17, 41], np.int64)
    ends = np.array([16385, 8210, 8232], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        low = s.execute(g, {"a": a, "b": b}, starts, ends, memory_budget_bytes=44 << 20)
        high = s.execute(g, {"a": a, "b": b}, starts, ends)
    assert calls(low) == 0
    assert calls(high) == (2 if ACCELERATE else 0)
    assert (REASON in low.plan.reason_codes) == ACCELERATE
    assert low.plan.estimated_total_memory_bytes <= 44 << 20
    bound(low)
    bound(high)
    for lo, hi, result in zip(starts, ends, low.outputs[0].values, strict=True):
        np.testing.assert_allclose(result, a[lo:hi] @ b, rtol=0, atol=0.000049)
