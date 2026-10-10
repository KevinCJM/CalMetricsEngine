"""High-arity scope dependency values, transport and independent ownership."""
import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler, PlannerConfig


def definition(captures):
    variables = {"x": "series"}
    variables.update({f"p{i}": "scalar" for i in range(captures)})
    body = "sum(x)" + "".join(f"+p{i}" for i in range(captures))
    return GraphCompiler(variables), f"sum(block_apply({body},2))"


@pytest.mark.parametrize("captures", [0, 6, 7, 8, 29, 30])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("error_policy", ["raise", "isolate"])
def test_parent_prefix_capture_boundaries_and_lifetime(captures, lane, error_policy):
    compiler, expression = definition(captures)
    graph = compiler.compile([expression], error_policy=error_policy)
    scope = next(node for node in graph.nodes if node.kind == "apply_scope")
    prefix = scope.parents
    assert len(prefix) == captures + 2
    native = graph.native_nodes()
    restored = pickle.loads(pickle.dumps(graph))
    assert restored.native_nodes() == native
    assert restored.metadata() == graph.metadata()
    del graph
    gc.collect()
    assert scope.parents == prefix

    config = PlannerConfig(
        thread_work_units=1 if lane != "single" else 1e99,
        process_work_units=1 if lane.startswith("process") else 1e100,
        min_rows_per_worker=1, max_processes=2,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
    )
    x = np.arange(1., 7.)
    saved_input = x.copy()
    x.flags.writeable = False
    starts, ends = np.array([0, 2], np.int64), np.array([6, 6], np.int64)
    parameters = {f"p{i}": float(i + 1) for i in range(captures)}
    total = captures * (captures + 1) // 2
    expected = np.array([[21 + 3 * total], [18 + 2 * total]], np.float64)
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        first = scheduler.execute(restored, {"x": x}, starts, ends,
                                  parameters=parameters, timeout=20.)
        second = scheduler.execute(restored, {"x": x}, starts, ends,
                                   parameters=parameters, timeout=20.)
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        for result in (first, second):
            np.testing.assert_array_equal(result.values, expected)
            if error_policy == "isolate":
                np.testing.assert_array_equal(result.statuses, np.zeros((2, 1), np.int16))
            else:
                assert result.statuses is None
            assert result.audit["python_worker_callbacks"] == 0
    del compiler, restored, scope
    gc.collect()
    np.testing.assert_array_equal(first.values, expected)
    np.testing.assert_array_equal(x, saved_input)
    assert not np.shares_memory(first.values, second.values)
    assert not np.shares_memory(first.values, x)


def test_scope_parent_limit_remains_32():
    compiler, expression = definition(31)
    with pytest.raises(GraphCompileError, match="too many scope captures"):
        compiler.compile([expression])
