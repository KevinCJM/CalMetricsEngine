"""Non-power-of-two tree tails through actual typed Graph and four transports."""

import hashlib
from decimal import Decimal, localcontext

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_matmul_library import declarations, settings
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks


@pytest.mark.skipif(not ACCELERATE, reason="enabled macOS 15 arm64 library")
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_balanced_tree_tail_graph_transport_and_owned_results(layout, order, lane):
    # K yields a non-power-of-two number of pieces: exercise carry and final merges.
    m, k, n = 513, 8065, 515
    rng = np.random.default_rng(5108065)
    a = np.array(rng.uniform(-1024, 1024, (m, k)), order=layout[0].upper())
    b = np.array(rng.uniform(-1024, 1024, (k, n)), order=layout[1].upper())
    a.flags.writeable = b.flags.writeable = False

    def digest(value):
        return hashlib.sha256(value.tobytes()).hexdigest()

    input_hashes = (digest(a), digest(b))
    expected = a @ b
    graph = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        old_hashes = [digest(v) for v in first.outputs[0].values]
        again = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        assert scheduler.peak_active_cpu_tokens <= 2
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    explicit = (320 * k + 72) * 8
    for result in (first, again):
        records = chunks(result)
        assert sum(c["accelerate_matmul_calls"] for c in records) > 32
        assert all(0 < c["operator_workspace_capacity_bytes"] <= explicit for c in records)
        assert result.audit["python_operator_calls"] == 0
        assert result.audit["request_time_compilation"] == 0
        assert len(result.outputs[0].values) == 2
        for value in result.outputs[0].values:
            np.testing.assert_allclose(value, expected, atol=0.000049, rtol=0)
            assert not value.flags.writeable
            assert value.flags.c_contiguous if order == "C" else value.flags.f_contiguous
            assert not np.shares_memory(value, a) and not np.shares_memory(value, b)
        assert not np.shares_memory(*result.outputs[0].values)
    old = first.outputs[0].values
    current = again.outputs[0].values
    assert all(not np.shares_memory(x, y) for x in old for y in current)
    assert old_hashes == [digest(v) for v in old]
    assert input_hashes == (digest(a), digest(b))
    with localcontext() as context:
        context.prec = 80
        for i, j in ((0, 0), (1, 3), (m - 1, n - 1), (m // 2, n // 2)):
            exact = sum(
                (
                    Decimal.from_float(float(a[i, h])) * Decimal.from_float(float(b[h, j]))
                    for h in range(k)
                ),
                Decimal(0),
            )
            for value in (*old, *current):
                assert abs(Decimal.from_float(float(value[i, j])) - exact) < Decimal(".000049")
