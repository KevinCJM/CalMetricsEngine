"""Closed arithmetic/selection reductions preserve exact math and lifetime."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_row_batch_execution import TYPES, Y


def execute(roots, data, length):
    g = GraphCompiler(TYPES).compile(roots, result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as s:
        return s.execute(g, data, np.zeros(1, np.int64), np.full(1, length, np.int64))


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("length", [0, 1, 257])
def test_closed_square_reduction_geometry_selected_math_and_owners(layout, length):
    rng = np.random.default_rng(4242)
    w = rng.uniform(0.1, 1, (193, 7))
    w /= w.sum(axis=1, keepdims=True)
    r = rng.normal(scale=0.01, size=(length, 7))
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
    saved = {k: v.copy() for k, v in data.items()}
    for a in data.values():
        a.flags.writeable = False
    mask = f"less_than({Y},0)"
    delta = f"subtract({Y},.001)"
    expression = f"sum_axis(where({mask},multiply({delta},{delta}),.1),1)"
    actual = execute([expression], data, length)
    canonical = execute([expression, delta, Y], data, length)
    current = execute([expression], data, length)
    a = actual.outputs[0].values[0]
    b = canonical.outputs[0].values[0]
    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    np.testing.assert_array_equal(a.view(np.uint64), current.outputs[0].values[0].view(np.uint64))
    assert a.shape == (193,) and not a.flags.writeable
    assert not np.shares_memory(a, current.outputs[0].values[0])
    assert not any(np.shares_memory(a, v) for v in data.values())
    assert sum(c["fused_pointwise_calls"] for c in actual.audit["native_chunks"]) >= 1
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in actual.audit["native_chunks"]
    )
    assert actual.plan.metadata()["estimated_worker_scratch_bytes"] >= observed
    for k, v in data.items():
        np.testing.assert_array_equal(v, saved[k])


@pytest.mark.parametrize("center", [1e308, float("inf"), float("nan")])
def test_closed_square_reduction_unselected_nonfinite_values(center):
    data = dict(w=np.full((193, 1), 0.25), r=np.ones((257, 1)), p=np.array([0.05]))
    # Nonfinite values arise from canonical arithmetic, not external bad literals.
    other = (
        "multiply(1e308,2)"
        if np.isinf(center)
        else "subtract(multiply(1e308,2),multiply(1e308,2))"
        if np.isnan(center)
        else "1e308"
    )
    mask = f"less_than({Y},0)"
    delta = f"subtract({Y},{other})"
    expression = f"sum_axis(where({mask},multiply({delta},{delta}),.1),1)"
    actual = execute([expression], data, 257)
    canonical = execute([expression, delta, Y], data, 257)
    np.testing.assert_array_equal(
        actual.outputs[0].values[0].view(np.uint64), canonical.outputs[0].values[0].view(np.uint64)
    )


def test_closed_square_reduction_preserves_later_domain_failure_and_extra_consumer():
    data = dict(w=np.full((193, 1), 0.25), r=np.ones((257, 1)), p=np.array([0.05]))
    mask = f"less_than({Y},1)"
    delta = f"subtract({Y},.001)"
    total = f"sum_axis(where({mask},multiply({delta},{delta}),0),1)"
    failure = f"mean_axis(sqrt(subtract({Y},1)),1)"
    errors = []
    for roots in [[total, failure], [total, failure, Y]]:
        with pytest.raises(ValueError) as caught:
            execute(roots, data, 257)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]
    actual = execute([total, delta], data, 257)
    canonical = execute([total, delta, Y], data, 257)
    for a, b in zip(actual.outputs, canonical.outputs[:2], strict=True):
        np.testing.assert_array_equal(a.values[0], b.values[0])
