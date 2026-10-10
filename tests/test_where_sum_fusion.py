"""Private selection/reduction retains logical nodes, arithmetic and owners."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_row_batch_execution import TYPES, Y


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("length", [0, 1, 17, 257])
def test_private_where_sum_retains_order_selection_geometry_and_owners(layout, length):
    rng = np.random.default_rng(719)
    w = rng.uniform(0.1, 1, (193, 7))
    w /= w.sum(axis=1, keepdims=True)
    r = rng.normal(scale=0.01, size=(length, 7))
    if length > 17:
        r[17] = -0.0
    w = np.array(w, order="F" if layout == "F" else "C")
    r = np.array(r, order="F" if layout == "F" else "C")
    if layout == "negative":
        w = w[::-1, ::-1]
        r = r[::-1, ::-1]
    if layout == "padded":
        a = np.full((193, 14), np.nan)
        a[:, ::2] = w
        w = a[:, ::2]
        b = np.full((length, 14), np.nan)
        b[:, ::2] = r
        r = b[:, ::2]
    data = dict(w=w, r=r, p=np.array([0.05]))
    snapshots = {k: a.copy() for k, a in data.items()}
    for a in data.values():
        a.flags.writeable = False
    predicate = f"less_than({Y},0)"
    roots = [
        f"sum_axis(where({predicate},{Y},0),1)",
        f"sum_axis(where({predicate},1,0),1)",
        f"sum_axis(where({predicate},0,add(1,{Y})),1)",
    ]
    g = GraphCompiler(TYPES).compile(roots, result_format="typed")
    control = GraphCompiler(TYPES).compile(roots + [Y], result_format="typed")
    starts = np.zeros(1, np.int64)
    ends = np.full(1, length, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        actual = s.execute(g, data, starts, ends)
        canonical = s.execute(control, data, starts, ends)
        current = s.execute(g, data, starts, ends)
    for a, b, again in zip(actual.outputs, canonical.outputs[:3], current.outputs, strict=True):
        np.testing.assert_array_equal(a.values[0].view(np.uint64), b.values[0].view(np.uint64))
        np.testing.assert_array_equal(a.values[0].view(np.uint64), again.values[0].view(np.uint64))
        assert a.values[0].shape == (193,) and not a.values[0].flags.writeable
        assert not np.shares_memory(a.values[0], again.values[0])
        assert all(not np.shares_memory(a.values[0], x) for x in data.values())
    chunks = actual.audit["native_chunks"]
    assert sum(c["fused_pointwise_calls"] for c in chunks) >= 3
    assert sum(c["algorithm_copy_bytes"] for c in chunks) >= 3 * 193 * 8
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in chunks
    )
    assert actual.plan.metadata()["estimated_worker_scratch_bytes"] >= observed
    for k in data:
        np.testing.assert_array_equal(data[k], snapshots[k])


def test_private_where_keeps_unconsumed_nonfinite_values_and_canonical_failure_order():
    w = np.zeros((193, 1))
    w[:, 0] = 0.25
    r = np.ones((257, 1))
    p = np.array([0.05])
    data = dict(w=w, r=r, p=p)
    # The non-selected branch may contain NaN without being selected or rejected.
    overflow = f"multiply(1e308,add(2,{Y}))"
    expression = f"sum_axis(where(greater_than({Y},0),{Y},subtract({overflow},{overflow})),1)"
    g = GraphCompiler(TYPES).compile(expression, result_format="typed")
    starts = np.zeros(1, np.int64)
    ends = np.full(1, 257, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        actual = s.execute(g, data, starts, ends)
    np.testing.assert_array_equal(actual.outputs[0].values[0], np.full(193, 257 * 0.25))
    for roots in [
        [
            f"sum_axis(where(greater_than({Y},0),1,0),1)",
            f"mean_axis(divide(1,subtract({Y},.25)),1)",
        ],
        [f"mean_axis(sqrt(subtract({Y},1)),1)", f"sum_axis(where(less_than({Y},1),{Y},0),1)"],
    ]:
        errors = []
        for expressions in [roots, roots + [Y]]:
            graph = GraphCompiler(TYPES).compile(expressions, result_format="typed")
            with AdaptiveScheduler(cpu_budget=1) as s, pytest.raises(ValueError) as caught:
                s.execute(graph, data, starts, ends)
            errors.append(str(caught.value))
        assert errors[0] == errors[1]
