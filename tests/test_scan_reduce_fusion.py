"""Continuous private scan closure retains errors, IEEE values and public paths."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_row_batch_execution import TYPES, Y


def path(seed=1):
    nav = f"cumulative_product_axis(add(1,{Y}),1,{seed})"
    peak = f"cumulative_max_axis({nav},1,{seed})"
    return f"max_axis(subtract(1,divide({nav},{peak})),1)", nav


def execute(roots, data, length):
    graph = GraphCompiler(TYPES).compile(roots, result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        return scheduler.execute(graph, data, np.zeros(1, np.int64), np.full(1, length, np.int64))


def ieee_equal(a, b):
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    non_nan = ~np.isnan(b)
    np.testing.assert_array_equal(a[non_nan].view(np.uint64), b[non_nan].view(np.uint64))


@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("length", [0, 1, 257])
def test_scan_reduce_cache_retains_original_steps_outputs_and_ownership(layout, length):
    rng = np.random.default_rng(818)
    w = rng.uniform(0.1, 1.0, (193, 7))
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
    saved = {k: a.copy() for k, a in data.items()}
    for a in data.values():
        a.flags.writeable = False
    reduction, nav = path()
    try:
        canonical = execute([reduction, nav], data, length)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error)):
            execute([reduction], data, length)
        return
    actual = execute([reduction], data, length)
    again = execute([reduction], data, length)
    ieee_equal(actual.outputs[0].values[0], canonical.outputs[0].values[0])
    ieee_equal(actual.outputs[0].values[0], again.outputs[0].values[0])
    assert not np.shares_memory(actual.outputs[0].values[0], again.outputs[0].values[0])
    assert not actual.outputs[0].values[0].flags.writeable
    for k in data:
        np.testing.assert_array_equal(data[k], saved[k])
    chunks = actual.audit["native_chunks"]
    if length:
        assert sum(c["fused_pointwise_calls"] for c in chunks) > 0
        assert sum(c["algorithm_copy_bytes"] for c in chunks) >= 193 * 8
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in chunks
    )
    assert actual.plan.metadata()["estimated_worker_scratch_bytes"] >= observed


@pytest.mark.parametrize("seed", [0, 1, 2, -1])
def test_seed_zero_failure_and_separate_initial_states_stay_canonical(seed):
    data = dict(w=np.full((193, 1), 0.25), r=np.ones((257, 1)) * 0.1, p=np.array([0.05]))
    reduction, nav = path(seed)
    try:
        reference = execute([reduction, nav], data, 257)
    except ValueError as error:
        with pytest.raises(ValueError) as caught:
            execute([reduction], data, 257)
        assert str(caught.value) == str(error)
        return
    actual = execute([reduction], data, 257)
    ieee_equal(actual.outputs[0].values[0], reference.outputs[0].values[0])


@pytest.mark.parametrize("earlier", [False, True])
def test_scan_fault_replays_at_original_division_or_maximum(earlier):
    w = np.ones((193, 1))
    w[130:] = 0.0
    data = dict(w=w, r=np.full((257, 1), 1e308), p=np.array([0.05]))
    reduction, nav = path()
    independent = f"mean_axis(divide(1,{Y}),1)"
    roots = [independent, reduction] if earlier else [reduction, independent]
    errors = []
    for expressions in [roots, roots + [nav]]:
        with pytest.raises(ValueError) as caught:
            execute(expressions, data, 257)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]
    assert ("DIVIDE_BY_ZERO" if earlier else "NON_FINITE_RESULT") in errors[0]


def test_another_private_consumer_and_public_path_keep_materialization():
    rng = np.random.default_rng(881)
    data = dict(
        w=rng.normal(scale=0.01, size=(193, 1)), r=rng.normal(size=(257, 1)), p=np.array([0.05])
    )
    reduction, nav = path()
    extra = f"sum_axis({nav},1)"
    actual = execute([reduction, extra], data, 257)
    control = execute([reduction, extra, nav], data, 257)
    for a, b in zip(actual.outputs, control.outputs[:2], strict=True):
        ieee_equal(a.values[0], b.values[0])
    assert sum(c["fused_pointwise_calls"] for c in actual.audit["native_chunks"]) == 0
