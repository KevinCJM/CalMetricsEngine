"""Contents, IEEE values and transport for ordinary per-call row reuse."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig

from .test_row_subview import data, digest, exact, exercise, graph


def test_same_graph_same_index_owner_reproves_each_request():
    x = data()
    g = graph(x)
    n, nt = len(x["raw"]), len(x["raw"]) * 3 // 4
    starts, ends = np.array([0], np.int64), np.array([n], np.int64)
    original_raw = digest({"raw": x["raw"]})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        old = scheduler.execute(g, x, starts, ends)
        retained = [digest({str(i): v for i, v in enumerate(o.values)}) for o in old.outputs]
        for mode in ("shift", "swap", "recover"):
            positions = x["positions"]
            positions.flags.writeable = True
            positions[:] = np.arange(n, dtype=np.int64)
            if mode == "shift":
                positions[:nt] += 1
            elif mode == "swap":
                positions[4], positions[5] = positions[5], positions[4]
            positions.flags.writeable = False
            control = scheduler.prepare_execution(g, x, starts, ends).run_snapshot()
            result = scheduler.execute(g, x, starts, ends)
            exact(result, control)
            actual = max(c["numeric_arena_bytes"] for c in result.audit["native_chunks"])
            canonical = max(c["numeric_arena_bytes"] for c in control.audit["native_chunks"])
            assert actual == canonical if mode == "swap" else actual < canonical - 2 * 1024**2
            assert retained == [
                digest({str(i): v for i, v in enumerate(o.values)}) for o in old.outputs
            ]
        assert original_raw == digest({"raw": x["raw"]})


@pytest.mark.parametrize("stride", [2, -1])
def test_noncontiguous_index_storage_with_contiguous_values(stride):
    x = data()
    n = len(x["raw"])
    positions = np.zeros(n * 2, np.int64)[::2] if stride == 2 else np.zeros(n, np.int64)[::-1]
    positions[:] = np.arange(n, dtype=np.int64)
    positions.flags.writeable = False
    x["positions"] = positions
    exercise(x, graph(x), True)


@pytest.mark.parametrize("first,second", [("-", "/"), ("+", "*"), ("*", "-"), ("*", "/")])
def test_pure_math_reuse_preserves_ieee_values(first, second):
    x = data()
    raw = x["raw"]
    raw.flags.writeable = True
    raw[0] = np.array([-0.0, np.inf, -np.inf, np.nan])
    raw[10] = np.array([np.nan, -0.0, 0.0, -np.inf])
    raw.flags.writeable = False
    x["shift"] = np.array([0.0, 1.0, -1.0, 2.0])
    x["scale"] = np.array([2.0, 1.0, -3.0, 4.0])
    for value in x.values():
        value.flags.writeable = False
    n, nt = len(raw), len(raw) * 3 // 4
    types = {
        "raw": {"kind": "matrix", "axes": ["time", "factor"], "shape": [str(n), "4"]},
        "positions": {
            "kind": "series",
            "axes": ["time"],
            "shape": [str(n)],
            "dtype": "int64",
            "semantic_dimension": "index",
        },
        "shift": {"kind": "vector", "axes": ["factor"], "shape": ["4"]},
        "scale": {"kind": "vector", "axes": ["factor"], "shape": ["4"]},
    }
    bindings = {
        "idx": f"slice_axis(positions,0,0,{nt},1)",
        "train": "take_axis(raw,idx,0)",
        "full": f"(raw{first}broadcast_like(shift,raw)){second}broadcast_like(scale,raw)",
        "selected": f"(train{first}broadcast_like(shift,train)){second}broadcast_like(scale,train)",
    }
    g = GraphCompiler(types).compile(
        {"full": "full", "summary": "sum_axis(selected,0)"},
        root_bindings=[bindings, bindings],
        result_format="typed",
        output_order="C",
    )
    exercise(x, g, True)


@pytest.mark.parametrize("lane", ["process_inline", "process_shared"])
def test_worker_row_views_keep_inputs_and_two_generations_after_close(lane):
    x = data()
    g = graph(x)
    starts = np.zeros(2, np.int64)
    ends = np.full(2, len(x["raw"]), np.int64)
    before = digest(x)
    with AdaptiveScheduler(cpu_budget=1) as canonical:
        control = canonical.prepare_execution(g, x, starts, ends).run_snapshot()
    config = PlannerConfig(
        thread_work_units=1,
        process_work_units=1,
        min_rows_per_worker=1,
        max_processes=2,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
    )
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        one = scheduler.execute(g, x, starts, ends)
        assert one.plan.lane == "process"
        retained = [digest({str(i): v for i, v in enumerate(o.values)}) for o in one.outputs]
        two = scheduler.execute(g, x, starts, ends)
        exact(one, control)
        exact(two, control)
    exact(one, control)
    assert before == digest(x)
    assert retained == [digest({str(i): v for i, v in enumerate(o.values)}) for o in one.outputs]
    for old, new in zip(one.outputs, two.outputs, strict=True):
        for a, b in zip(old.values, new.values, strict=True):
            assert not a.flags.writeable and not np.shares_memory(a, b)
            assert all(not np.shares_memory(a, source) for source in x.values())
