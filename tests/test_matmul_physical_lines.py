"""Large-K input guards and independently retained actual graph results."""

import hashlib
from decimal import Decimal, localcontext

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_matmul_library import declarations
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks


@pytest.mark.skipif(not ACCELERATE, reason="enabled macOS 15 arm64 library")
@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_large_matmul_physical_layout_owned_results(layout, order):
    rng = np.random.default_rng(3708192)
    a = np.array(rng.uniform(-1024, 1024, (2048, 4096)), order=layout[0].upper())
    b = np.array(rng.uniform(-1024, 1024, (4096, 2048)), order=layout[1].upper())

    def digest(value):
        return hashlib.sha256(value.tobytes()).hexdigest()

    saved_inputs = (digest(a), digest(b))
    a.flags.writeable = b.flags.writeable = False
    expected = a @ b
    graph = GraphCompiler(declarations()).compile("matmul(a,b)", output_order=order)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
        old = first.outputs[0].values[0]
        old_digest = digest(old)
        second = scheduler.execute(graph, {"a": a, "b": b}, bounds, bounds)
    current = second.outputs[0].values[0]
    for result, value in [(first, old), (second, current)]:
        np.testing.assert_allclose(value, expected, atol=0.000049, rtol=0)
        assert not value.flags.writeable
        assert value.flags.c_contiguous if order == "C" else value.flags.f_contiguous
        assert not np.shares_memory(value, a) and not np.shares_memory(value, b)
        assert sum(c["accelerate_matmul_calls"] for c in chunks(result)) > 2
        assert result.audit["python_operator_calls"] == 0
        assert result.audit["request_time_compilation"] == 0
        with localcontext() as ctx:
            ctx.prec = 80
            for i, j in [(0, 0), (1, 3), (2047, 2047), (256, 256)]:
                exact = sum(
                    (
                        Decimal.from_float(float(a[i, k])) * Decimal.from_float(float(b[k, j]))
                        for k in range(4096)
                    ),
                    Decimal(0),
                )
                assert abs(Decimal.from_float(float(value[i, j])) - exact) < Decimal(".000049")
    assert not np.shares_memory(old, current)
    assert digest(old) == old_digest
    assert saved_inputs == (digest(a), digest(b))
