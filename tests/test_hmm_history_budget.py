"""A maximum-window plan must cover shorter rows across a physical cutoff."""
import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler


@pytest.mark.parametrize("k", [3, 64, 256])
@pytest.mark.parametrize("ends", [[4095], [4096], [4095, 4096], [4096, 4095]])
def test_hmm_history_plan_covers_shorter_window_at_cutoff(k, ends):
    inputs = {
        "x": np.zeros((4096, 1)), "w": np.full(k, 1.0 / k),
        "a": np.full((k, k), 1.0 / k), "m": np.zeros((k, 1)),
        "v": np.ones((k, 1)),
    }
    for array in inputs.values():
        array.flags.writeable = False
    types = {
        "x": {"kind": "matrix", "axes": ["time", "factor"], "shape": ["N", "D"]},
        "w": {"kind": "vector", "axes": ["component"], "shape": ["K"]},
        "a": {"kind": "matrix", "axes": ["component", "component"], "shape": ["K", "K"]},
        "m": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "v": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
    }
    graph = GraphCompiler(types).compile("hmm_initial(gaussian_hmm_em(x,w,a,m,v,1))")
    ends = np.asarray(ends, np.int64)
    starts = np.zeros(ends.size, np.int64)
    # 4095-row execution still requires all three independent histories.
    smaller_required = (3 * 4095 * k + 2 * k * k + 4 * k + 128) * 8
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        assert plan.estimated_worker_scratch_bytes >= smaller_required
        if k == 256:
            # Reject before numerical execution, including either row order.
            with pytest.raises((ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc|bad allocation"):
                scheduler.execute(graph, inputs, starts, ends, memory_budget_bytes=smaller_required - 1)
