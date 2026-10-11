"""Public F roots preserve independent IEEE arithmetic and result ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_guarded_lu_graph import EXPECTED_LIBRARY, count, run
from .test_solve_library import same_ieee
from .test_typed_results import config

TYPES = {key: dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]) for key in ["a", "r"]}


def inputs(n=1025, layout="f", special=False, uniform=True):
    rng = np.random.default_rng(20261004)
    a = np.array(rng.normal(size=(n, n)), order=layout.upper())
    r = np.array(rng.normal(size=(n, n)), order=layout.upper() if uniform else "F")
    if special:
        a.flat[:8] = [0.0, -0.0, np.nan, np.inf, -np.inf, 1e308, -1e308, 1.0]
        r.flat[:8] = [-0.0, 0.0, np.nan, -np.inf, np.inf, 1e308, 1e308, np.nan]
    return dict(a=a, r=r)


def compile_graph(expression, output_order="F"):
    # The independent reduction retains a generic multi-root execution graph.
    return GraphCompiler(TYPES).compile(
        [expression, "sum_axis(r,0)"], result_format="typed", output_order=output_order
    )


def expected(expression, values):
    a, r = values["a"], values["r"]
    return {
        "a+r": lambda: op.add(a, r),
        "a-r": lambda: op.subtract(a, r),
        "a*r": lambda: op.multiply(a, r),
        "0-a": lambda: op.subtract(0.0, a),
        "a-0": lambda: op.subtract(a, 0.0),
        "a*1": lambda: op.multiply(a, 1.0),
        "0+a": lambda: op.add(0.0, a),
    }[expression]()


@pytest.mark.parametrize("expression", ["a+r", "a-r", "a*r", "0-a", "a-0", "a*1", "0+a"])
@pytest.mark.parametrize("special", [False, True])
@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("output_order", ["C", "F"])
def test_public_dense_arithmetic_matches_original_ieee_and_retains_two_results(
    expression, special, layout, output_order
):
    values = inputs(layout=layout, special=special)
    saved = {k: v.copy() for k, v in values.items()}
    for v in values.values():
        v.flags.writeable = False
    graph = compile_graph(expression, output_order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = run(graph, values, scheduler)
        second = run(graph, values, scheduler)
    actual = first.outputs[0].values[0]
    same_ieee(actual, expected(expression, values))
    same_ieee(first.outputs[1].values[0], op.sum_axis(values["r"], 0))
    assert actual.flags.c_contiguous if output_order == "C" else actual.flags.f_contiguous
    assert first.outputs[0].order == output_order and not actual.flags.writeable
    assert not np.shares_memory(actual, second.outputs[0].values[0])
    if EXPECTED_LIBRARY:
        # Odd row tails are scalar in C-to-F row tiles.
        assert count(first, "vector_elements") >= actual.size - actual.shape[0]
    for k, v in values.items():
        same_ieee(v, saved[k])
        assert not np.shares_memory(v, actual)


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("n", [1023, 1025])
def test_mixed_layout_and_small_geometry_still_match_canonical(layout, n):
    values = inputs(n, layout, uniform=False)
    result = run(compile_graph("a+r"), values)
    same_ieee(result.outputs[0].values[0], op.add(values["a"], values["r"]))
    assert result.outputs[0].values[0].flags.f_contiguous


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("output_order", ["C", "F"])
def test_public_dense_roots_across_native_transport_keep_snapshots(lane, layout, output_order):
    values = inputs(layout=layout)
    saved = {k: v.copy() for k, v in values.items()}
    graph = compile_graph("a+r", output_order)
    for v in values.values():
        v.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(values) as bundle:
                first = run(graph, bundle, scheduler, 2)
                second = run(graph, bundle, scheduler, 2)
        else:
            first = run(graph, values, scheduler, 2)
            second = run(graph, values, scheduler, 2)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for row in range(2):
        same_ieee(first.outputs[0].values[row], op.add(values["a"], values["r"]))
        actual = first.outputs[0].values[row]
        assert actual.flags.c_contiguous if output_order == "C" else actual.flags.f_contiguous
        assert not np.shares_memory(first.outputs[0].values[row], second.outputs[0].values[row])
    for k, v in values.items():
        same_ieee(v, saved[k])
