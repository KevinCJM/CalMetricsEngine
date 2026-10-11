"""Projection quantiles: shared ranks, exceptional domains and ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_matmul_quantile_statistics import TYPES, Y, compare, data, run


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("t", [127, 128, 1023, 1024, 2520])
@pytest.mark.parametrize("probability", [0.0, 0.05, 0.5, 0.95, 1.0])
def test_common_probability_matches_full_projection(order, t, probability):
    values = data(65, t, 7, order)
    values["q"] = np.array([probability, probability])
    values["p"] = np.array([probability])
    types = dict(TYPES, p=dict(kind="vector", axes=["asset"], shape=["P"]))
    roots = [
        f"mean_axis({Y},1)",
        f"variance_axis({Y},1,1)",
        f"quantiles_axis({Y},q,1)",
        f"quantiles_axis({Y},p,1)",
    ]
    compiler = GraphCompiler(types)
    graph = compiler.compile(roots, result_format="typed", output_order=order)
    control = compiler.compile([*roots, Y], result_format="typed", output_order=order)
    saved = {key: value.copy() for key, value in values.items()}
    for value in values.values():
        value.flags.writeable = False
    first = run(graph, values)
    snapshots = [out.values[0].copy() for out in first.outputs]
    second = run(graph, values)
    reference = run(control, values)
    compare(first, reference, len(roots))
    projected = values["w"].astype(np.longdouble) @ values["r"].astype(np.longdouble).T
    ordered = np.sort(projected, axis=1)
    position = (t - 1) * probability
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    expected = ordered[:, lo] * (1 - (position - lo)) + ordered[:, hi] * (position - lo)
    np.testing.assert_allclose(first.outputs[2].values[0][:, 0], expected, atol=4e-5, rtol=1e-12)
    for before, after, snapshot in zip(first.outputs, second.outputs, snapshots, strict=True):
        a, b = before.values[0], after.values[0]
        np.testing.assert_array_equal(a, snapshot)
        assert not np.shares_memory(a, b) and not a.flags.writeable
        if a.ndim == 2:
            assert a.flags.c_contiguous if order == "C" else a.flags.f_contiguous
        assert all(not np.shares_memory(a, value) for value in values.values())
    for key, value in values.items():
        np.testing.assert_array_equal(value, saved[key])


@pytest.mark.parametrize(
    "case", ["zero", "negative_zero", "subnormal", "nan", "different", "empty", "reverse"]
)
def test_exceptional_or_different_probabilities_keep_logical_behavior(case):
    values = data(65, 1024, 7)
    values["q"] = np.array([0.05])
    if case == "zero":
        values["r"][:] = 0.0
    elif case == "negative_zero":
        values["r"][:] = -0.0
    elif case == "subnormal":
        values["r"][:] = np.nextafter(0.0, 1.0)
    elif case == "nan":
        values["q"][0] = np.nan
    elif case == "different":
        values["q"] = np.array([0.05, 0.95])
    elif case == "empty":
        values["q"] = np.array([])
    else:
        values["r"] = values["r"][::-1, ::-1]
        values["w"] = values["w"][:, ::-1]
    roots = [f"mean_axis({Y},1)", f"quantiles_axis({Y},q,1)"]
    compiler = GraphCompiler(TYPES)
    graph = compiler.compile(roots, result_format="typed")
    control = compiler.compile([*roots, Y], result_format="typed")
    try:
        expected = run(control, values)
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            run(graph, values)
        assert str(actual.value) == str(error)
    else:
        compare(run(graph, values), expected, len(roots))


def test_prepared_probability_change_rechecks_selection_and_retains_result():
    values = data(129, 1024, 7)
    values["q"] = np.array([0.05, 0.05])
    roots = [f"quantiles_axis({Y},q,1)", f"mean_axis({Y},1)"]
    compiler = GraphCompiler(TYPES)
    graph = compiler.compile(roots, result_format="typed")
    control = compiler.compile([*roots, Y], result_format="typed")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, values, bounds, bounds)
        first = prepared.run_snapshot()
        snapshot = first.outputs[0].values[0].copy()
        values["q"][1] = 0.95
        second = prepared.run_snapshot()
        reference = scheduler.execute(control, values, bounds, bounds)
        compare(second, reference, len(roots))
        np.testing.assert_array_equal(first.outputs[0].values[0], snapshot)
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
