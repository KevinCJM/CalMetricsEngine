"""Scalar fusion preserves existing storage-order and strided matrix contracts."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_solve_library import same_ieee

DECLARATIONS = {"x": dict(kind="matrix", shape=["N", "M"], axes=["asset", "asset"])}
STEPS = [
    ("x*0.5+0.25", "multiply", 0.5, False, "add", 0.25, False),
    ("0.5*x-0.25", "multiply", 0.5, True, "subtract", 0.25, False),
    ("0.25-x*0.5", "multiply", 0.5, False, "subtract", 0.25, True),
    ("(x+0.25)*0.5", "add", 0.25, False, "multiply", 0.5, False),
    ("(x-0.25)/0.5", "subtract", 0.25, False, "divide", 0.5, False),
    ("0.5*(0.25-x)", "subtract", 0.25, True, "multiply", 0.5, True),
]


def matrix(layout, special):
    # Odd dimensions cover both tile/packet tails, above the scalar-span cutoff.
    rows, columns = 257, 263
    order = layout[0].upper()
    shape = (
        (rows + 3, columns)
        if layout == "f_pitched"
        else ((rows, columns + 3) if layout == "c_pitched" else (rows, columns))
    )
    owner = np.array(np.random.default_rng(6310).normal(size=shape), order=order)
    x = owner[:rows, :columns]
    if layout.endswith("reversed"):
        x = x[::-1]
    if special:
        x[:2, :4] = [[0.0, -0.0, np.nan, np.inf], [-np.inf, 1e308, -1e308, 0.25]]
    owner.flags.writeable = x.flags.writeable = False
    assert x.size >= 65536
    if layout.endswith("pitched") or layout.endswith("reversed"):
        assert not x.flags.c_contiguous and not x.flags.f_contiguous
    return owner, x


@pytest.mark.parametrize("layout", ["c", "f", "c_pitched", "f_pitched", "c_reversed", "f_reversed"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("special", [False, True])
@pytest.mark.parametrize("steps", STEPS, ids=[step[0] for step in STEPS])
def test_two_root_scalar_pairs_preserve_layout_ieee_input_and_snapshots(
    layout, order, special, steps
):
    expression, first, p, p_left, second, q, q_left = steps
    owner, x = matrix(layout, special)
    before = owner.copy()
    middle = getattr(op, first)(p, x) if p_left else getattr(op, first)(x, p)
    expected = getattr(op, second)(q, middle) if q_left else getattr(op, second)(middle, q)
    # Keep the existing general multi-root executor; no new DSL output syntax.
    graph = GraphCompiler(DECLARATIONS).compile(
        [expression, "transpose(x)"], result_format="typed", output_order=order
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first_result = scheduler.execute(graph, {"x": x}, bounds, bounds)
        saved = [root.values[0].copy() for root in first_result.outputs]
        second_result = scheduler.execute(graph, {"x": x}, bounds, bounds)
        assert scheduler.peak_active_cpu_tokens <= 1
    for result in [first_result, second_result]:
        assert sum(chunk["fused_pointwise_calls"] for chunk in result.audit["native_chunks"]) == 1
        assert all(chunk["pointwise_tiles"] == 0 for chunk in result.audit["native_chunks"])
        assert result.audit["python_operator_calls"] == result.audit["python_fallback"] == 0
        for root, reference in zip(result.outputs, [expected, x.T], strict=False):
            actual = root.values[0]
            same_ieee(actual, reference)
            assert actual.flags.c_contiguous if order == "C" else actual.flags.f_contiguous
            assert not actual.flags.writeable and not np.shares_memory(actual, owner)
            assert root.root_statuses[0] == 0
        assert not np.shares_memory(result.outputs[0].values[0], result.outputs[1].values[0])
    for old, current, reference in zip(
        first_result.outputs, second_result.outputs, saved, strict=False
    ):
        same_ieee(old.values[0], reference)
        assert not np.shares_memory(old.values[0], current.values[0])
    same_ieee(owner, before)
