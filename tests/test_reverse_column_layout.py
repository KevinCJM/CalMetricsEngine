"""Reverse column views keep logical order across native kernels and transport."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op
from tests.test_column_library import enabled, identical
from tests.test_tensor_protocol import configuration


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["auto", "C", "F"])
@pytest.mark.parametrize("isolate", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_reverse_standardization_windows_and_ownership(lane, order, isolate, padded):
    rng = np.random.default_rng(518)
    storage = np.asfortranarray(rng.normal(size=(71 if padded else 65, 9)))
    x = (storage[3:68] if padded else storage)[::-1]
    x.flags.writeable = False
    saved = x.copy()
    mean = "mean_axis(x,0)"
    scale = "sqrt(variance_axis(x,0))"
    z = f"(x-broadcast_like({mean},x))/broadcast_like({scale},x)"
    graph = GraphCompiler(
        {"x": dict(kind="matrix", axes=["time", "asset"], shape=["T", "N"])}
    ).compile(
        {"z": z, "mean": mean, "scale": scale, "independent": "7"},
        output_order=order,
        error_policy="isolate" if isolate else "raise",
    )
    starts = np.array([0, 3], np.int64)
    ends = np.array([65, 60], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        prepared = (
            engine.prepare_execution(graph, {"x": x}, starts, ends) if lane == "single" else None
        )
        run = (
            prepared.run_snapshot
            if prepared
            else lambda: engine.execute(graph, {"x": x}, starts, ends)
        )
        first, second = run(), run()
    if order != "auto":
        assert first.outputs[0].order == order
    elif isolate:
        assert first.outputs[0].order == "C"
    elif enabled():
        assert first.outputs[0].order == "F"
    for row, (begin, end) in enumerate(zip(starts, ends, strict=True)):
        v = x[begin:end]
        m = op.call("mean_axis", v, 0, simd="scalar")
        s = np.sqrt(op.call("variance_axis", v, 0, simd="scalar"))
        expected = ((v - m) / s, m, s, np.asarray(7.0))
        for column, wanted in zip(first.outputs, expected, strict=True):
            value = column.values[row]
            identical(value, wanted)
            assert not value.flags.writeable and not np.shares_memory(value, x)
            assert not np.shares_memory(value, second.outputs[0].values[row])
            if isolate:
                assert column.statuses[row].flags.c_contiguous
                assert not column.statuses[row].any()
        z_value = first.outputs[0].values[row]
        assert (
            z_value.flags.f_contiguous
            if first.outputs[0].order == "F"
            else z_value.flags.c_contiguous
        )
    identical(x, saved)


@pytest.mark.parametrize("rows,columns", [(31, 8), (32, 8), (33, 9), (65, 17)])
def test_reverse_reduction_preserves_cancellation_and_signed_zero(rows, columns):
    x = np.empty((rows, columns), order="F")
    x[:] = np.resize([1e16, 1.0, -1e16, -0.0, 0.0, 1.0], rows)[:, None]
    x = x[::-1]
    for name in ("sum_axis", "mean_axis", "variance_axis"):
        identical(op.call(name, x, 0), op.call(name, x, 0, simd="scalar"))


@pytest.mark.parametrize("roots", [1, 63, 64, 65, 129])
@pytest.mark.parametrize("lane", ["single", "process_inline", "process_shared"])
def test_many_f_roots_keep_layout_bits_and_worker_policy(roots, lane):
    x = np.asfortranarray(np.arange(24.0).reshape(8, 3))[::-1]
    expressions = {f"value_{i}": "x" for i in range(roots)}
    graph = GraphCompiler(
        {"x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"])}
    ).compile(expressions, output_order="F")
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        result = engine.execute(graph, {"x": x}, bounds, bounds)
    for column in result.outputs:
        assert column.order == "F"
        for value in column.values:
            assert value.flags.f_contiguous
            identical(value, x)
