"""Binary conditional counts equal the canonical graph including retained owners."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_row_batch_execution import TYPES, Y


@pytest.mark.parametrize("length", [0, 1, 17, 252, 2520])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("yes,no", [(1, 0), (0, 1), (1, -0.0), (-0.0, 1), (2, 0), (3, -1)])
def test_conditional_counts_keep_exact_outputs_geometry_and_ownership(length, layout, yes, no):
    rng = np.random.default_rng(8717)
    w = rng.uniform(0.1, 1, (193, 7))
    w /= w.sum(axis=1, keepdims=True)
    r = rng.normal(scale=0.01, size=(length, 7))
    if length > 17:
        r[17] = 0.0
    if layout == "F":
        w = np.array(w, order="F")
        r = np.array(r, order="F")
    elif layout == "negative":
        w = w[::-1, ::-1]
        r = r[::-1, ::-1]
    elif layout == "padded":
        owner = np.full((193, 14), np.nan)
        owner[:, ::2] = w
        w = owner[:, ::2]
        owner = np.full((length, 14), np.nan)
        owner[:, ::2] = r
        r = owner[:, ::2]
    data = dict(w=w, r=r, p=np.array([0.05]))
    saved = {name: x.copy() for name, x in data.items()}
    for x in data.values():
        x.flags.writeable = False
    expression = f"sum_axis(where(less_than({Y},0),{yes!r},{no!r}),1)"
    compiler = GraphCompiler(TYPES)
    graph = compiler.compile(expression, result_format="typed")
    canonical = compiler.compile([expression, Y], result_format="typed")
    starts = np.array([0], np.int64)
    ends = np.array([length], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, data, starts, ends)
        reference = scheduler.execute(canonical, data, starts, ends)
        second = scheduler.execute(graph, data, starts, ends)
    actual = first.outputs[0].values[0]
    np.testing.assert_array_equal(
        actual.view(np.uint64), reference.outputs[0].values[0].view(np.uint64)
    )
    np.testing.assert_array_equal(
        actual.view(np.uint64), second.outputs[0].values[0].view(np.uint64)
    )
    assert actual.shape == (193,) and not actual.flags.writeable
    assert not np.shares_memory(actual, second.outputs[0].values[0])
    assert all(not np.shares_memory(actual, x) for x in data.values())
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in first.audit["native_chunks"]
    )
    assert first.plan.metadata()["estimated_worker_scratch_bytes"] >= observed
    for name, x in data.items():
        np.testing.assert_array_equal(x.view(np.uint64), saved[name].view(np.uint64))
