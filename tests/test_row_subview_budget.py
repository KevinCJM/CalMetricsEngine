"""Optional row proofs preserve original admission and independent Plans."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, PlannerConfig

from .test_row_subview import data, digest, exact, graph

REASON = "memory_budget_uses_original_row_geometry"


def settings(lane):
    return PlannerConfig(
        thread_work_units=1 if lane != "single" else 1e100,
        process_work_units=1 if lane.startswith("process") else 1e100,
        min_rows_per_worker=1,
        max_processes=2,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
    )


def original_budget(g, plan):
    per_worker = 65536 + len(g.nodes) * (np.dtype(np.uintp).itemsize + 5)
    workers = max(plan.thread_count, plan.process_count)
    return plan.estimated_total_memory_bytes - per_worker * workers


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_original_budget_preserves_lane_workers_and_two_generations(lane):
    x = data()
    g = graph(x)
    rows = 1 if lane == "single" else 2
    starts = np.zeros(rows, np.int64)
    ends = np.full(rows, len(x["raw"]), np.int64)
    before = digest(x)
    with AdaptiveScheduler(cpu_budget=1) as reference:
        control = reference.prepare_execution(g, x, starts, ends).run_snapshot()
    with AdaptiveScheduler(cpu_budget=rows, config=settings(lane)) as scheduler:
        high = scheduler.plan(g, x, starts, ends)
        budget = original_budget(g, high)
        retained = []
        for limit in (None, budget, None, budget):
            result = scheduler.execute(g, x, starts, ends, memory_budget_bytes=limit)
            exact(result, control)
            assert result.plan.lane == ("process" if lane.startswith("process") else lane)
            assert (result.plan.thread_count, result.plan.process_count) == (
                high.thread_count,
                high.process_count,
            )
            assert (REASON in result.plan.reason_codes) == (limit is not None)
            if limit is not None:
                assert result.plan.estimated_total_memory_bytes <= budget
            for old, hashes in retained:
                assert hashes == [
                    digest({str(i): value for i, value in enumerate(o.values)}) for o in old.outputs
                ]
                for a, b in zip(old.outputs, result.outputs, strict=True):
                    assert all(
                        not np.shares_memory(u, v) for u, v in zip(a.values, b.values, strict=True)
                    )
            retained.append(
                (
                    result,
                    [digest({str(i): v for i, v in enumerate(o.values)}) for o in result.outputs],
                )
            )
    for result, _ in retained:
        exact(result, control)
    assert before == digest(x)


def test_original_budget_minus_one_still_rejects_and_permission_recovers():
    x = data()
    g = graph(x)
    starts = np.array([0], np.int64)
    ends = np.array([len(x["raw"])], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        high = scheduler.plan(g, x, starts, ends)
        budget = original_budget(g, high)
        old = scheduler.execute(g, x, starts, ends)
        with pytest.raises(MemoryError):
            scheduler.execute(g, x, starts, ends, memory_budget_bytes=budget - 1)
        invalid = dict(x)
        positions = x["positions"].copy()
        positions[10] = -1
        positions.flags.writeable = False
        invalid["positions"] = positions
        with pytest.raises(Exception) as canonical:
            scheduler.prepare_execution(g, invalid, starts, ends).run_snapshot()
        with pytest.raises(type(canonical.value)) as low_error:
            scheduler.execute(g, invalid, starts, ends, memory_budget_bytes=budget)
        assert str(canonical.value) == str(low_error.value)
        low = scheduler.execute(g, x, starts, ends, memory_budget_bytes=budget)
        restored = scheduler.execute(g, x, starts, ends)
        exact(old, low)
        exact(old, restored)
        assert REASON in low.plan.reason_codes and REASON not in restored.plan.reason_codes
        low_bytes = max(c["numeric_arena_bytes"] for c in low.audit["native_chunks"])
        high_bytes = max(c["numeric_arena_bytes"] for c in restored.audit["native_chunks"])
        assert high_bytes < low_bytes - 2 * 1024**2
