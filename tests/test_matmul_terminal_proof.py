"""Same-execution proof reuse cannot cache contents or expose intermediates."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_matmul_scalar_chain import canonical, payload, verify


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("offset", [0.0, 0.25, -0.125])
def test_terminal_proof_defers_only_unobservable_matrix_cancellation(order, offset):
    a, b = payload("cf")
    a.setflags(write=True)
    b.setflags(write=True)
    a.fill(0.01)
    b.fill(0.01)
    b[::2] *= -1
    a[0, :3] = [1.0, 2.0**-54, -1.0]
    b[:3, 0] = [1.0, 1.0, 1.0]
    a.flags.writeable = b.flags.writeable = False
    declarations = {
        "a": dict(kind="matrix", shape=["M", "K"], axes=["asset", "asset"]),
        "b": dict(kind="matrix", shape=["K", "N"], axes=["asset", "asset"]),
        "scale": "scalar",
        "offset": "scalar",
    }
    graph = GraphCompiler(declarations).compile("matmul(a,b)*scale+offset", output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            graph, {"a": a, "b": b}, z, z, parameters={"scale": -0.7, "offset": offset}
        )
    expected = canonical(a, b, [("multiply", -0.7, False), ("add", offset, False)])
    verify(result, expected, a, b, order, True)


@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_input_contents_are_scanned_again_not_reused_by_identity(order):
    a, b = payload("ff")
    declarations = {
        "a": dict(kind="matrix", shape=["M", "K"], axes=["asset", "asset"]),
        "b": dict(kind="matrix", shape=["K", "N"], axes=["asset", "asset"]),
    }
    graph = GraphCompiler(declarations).compile("matmul(a,b)*0.5+0.25", output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a, "b": b}, z, z)
        old = prepared.run_snapshot()
        saved = old.outputs[0].values[0].copy()
        a.setflags(write=True)
        a[2, 3] = 1e50
        a.setflags(write=False)
        result = prepared.run_snapshot()
    expected = canonical(a, b, [("multiply", 0.5, False), ("add", 0.25, False)])
    verify(result, expected, a, b, order, False)
    np.testing.assert_array_equal(old.outputs[0].values[0], saved)
    assert not np.shares_memory(old.outputs[0].values[0], result.outputs[0].values[0])
