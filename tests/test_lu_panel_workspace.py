"""LU-to-residual lifetime reuse preserves immutable inputs and owned results."""

import numpy as np
import pytest

from .test_guarded_lu_graph import EXPECTED_LIBRARY, check, count, data, graph, run
from .test_solve_library import same_ieee


@pytest.mark.parametrize("n", [1537, 2049])
@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("mode", ["input", "private"])
@pytest.mark.parametrize("k", [1, 64])
def test_odd_512_panels_dead_factor_reuse_and_two_generations(n, layout, mode, k):
    values = data(layout, n, k)
    saved = {name: value.copy() for name, value in values.items()}
    for value in values.values():
        value.flags.writeable = False
    expression = "solve_multi(a,b)" if mode == "input" else "solve_multi(a+0,b)"
    g = graph(expression, output_order=layout.upper())
    first = run(g, values)
    snapshot = first.outputs[0].values[0].copy()
    second = run(g, values)
    check(first, values)
    check(second, values)
    assert count(first, "accelerate_lu_calls") == EXPECTED_LIBRARY
    if EXPECTED_LIBRARY:
        maximum = (n * k + 5 * n + (n * n if mode == "input" else 0)) * 8
        assert count(first, "operator_workspace_capacity_bytes") <= maximum
    same_ieee(first.outputs[0].values[0], snapshot)
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    for name, value in values.items():
        same_ieee(value, saved[name])
