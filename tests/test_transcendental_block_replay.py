"""Outliers replay only their bounded block in independent unary graph roots."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler
from calmetrics_engine import operators as op

from .test_matmul_library import settings
from .test_nonnegative_matmul_accelerate import ACCELERATE
from .test_transcendental_accelerate import calls, graph
from .test_transcendental_cross_layout import compare, sample


@pytest.mark.parametrize("name,value", [("exp", 701.), ("exp", -701.), ("exp", -np.inf),
                                      ("log", np.nextafter(0., 1.)), ("log", np.inf),
                                      ("log", np.nan)])
@pytest.mark.parametrize("position", [0, 65535, 65536, 131072])
def test_root_block_boundaries_and_exact_outlier(name, value, position):
    x = np.linspace(-2., 2., 131073) if name == "exp" else np.linspace(.1, 10., 131073)
    x[position] = value
    x.flags.writeable = False
    expected = op.call(name, x, simd="scalar")
    bounds = np.array([0], np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(name, 1), {"x": x}, *bounds)
    compare(result.outputs[0].values[0], expected)
    assert result.outputs[0].values[0][position].view(np.uint64) == expected[position].view(np.uint64)
    chunk = result.audit["native_chunks"][0]
    replay = 1 if position == 131072 else 65536
    assert calls(result) == (2 if ACCELERATE else 0)
    assert chunk["vector_elements"] == (len(x) - replay if ACCELERATE else 0)
    assert chunk["operator_workspace_capacity_bytes"] == chunk["algorithm_copy_bytes"] == 0


@pytest.mark.parametrize("name,value", [("exp", 701.), ("log", np.nextafter(0., 1.))])
@pytest.mark.parametrize("source,target", [("C", "C"), ("F", "F"), ("C", "F"), ("F", "C")])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_outlier_layout_transport_and_independent_results(name, value, source, target, lane):
    x = sample(name, (1031, 129), source).copy(order=source)
    x[515, 66] = value
    x.flags.writeable = False
    before = x.copy(order="K")
    starts, ends = np.zeros(2, np.int64), np.full(2, len(x), np.int64)
    expected = op.call(name, x, simd="scalar")
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(graph(name, 2, target), {"x": x}, starts, ends)
        second = scheduler.execute(graph(name, 2, target), {"x": x}, starts, ends)
    assert (calls(first) > 0) == ACCELERATE
    for row, actual in enumerate(first.outputs[0].values):
        compare(actual, expected)
        assert actual[515, 66].view(np.uint64) == expected[515, 66].view(np.uint64)
        assert not actual.flags.writeable
        assert actual.flags.f_contiguous if target == "F" else actual.flags.c_contiguous
        assert not np.shares_memory(actual, second.outputs[0].values[row])
        assert not np.shares_memory(actual, x)
    np.testing.assert_array_equal(x.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize("name,bad", [("exp", 710.), ("exp", np.nan),
                                    ("log", 0.), ("log", -1.)])
def test_failed_prepared_snapshot_does_not_corrupt_retained_result(name, bad):
    x = np.full(131073, 1.)
    bounds = np.array([0], np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph(name, 1), {"x": x}, *bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        x[-1] = bad
        with pytest.raises(ValueError, match="DOMAIN_ERROR"):
            op.call(name, x, simd="scalar")
        with pytest.raises(ValueError, match="DOMAIN_ERROR"):
            prepared.run_snapshot()
        np.testing.assert_array_equal(first.outputs[0].values[0], saved)
        x[-1] = 701. if name == "exp" else np.nextafter(0., 1.)
        second = prepared.run_snapshot()
        compare(second.outputs[0].values[0], op.call(name, x, simd="scalar"))
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])


@pytest.mark.parametrize("name,value", [("exp", 701.), ("log", np.nextafter(0., 1.))])
def test_consumers_and_isolation_retain_canonical_order(name, value):
    x = np.full(131073, .5)
    x[-1] = value
    bounds = np.array([0], np.int64), np.array([len(x)], np.int64)
    expected = op.call(name, x, simd="scalar")
    for isolate, expression in ((True, f"{name}(x)"), (False, f"{name}(x)+0")):
        compiled = graph(name, 1, expression=expression, isolate=isolate)
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            result = scheduler.execute(compiled, {"x": x}, *bounds)
        assert calls(result) == 0
        np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("name,value", [("exp", 701.), ("log", np.nextafter(0., 1.))])
def test_all_outlier_blocks_report_scalar_execution(name, value):
    x = np.full(131073, value)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(name, 1), {"x": x}, np.array([0], np.int64),
                                   np.array([len(x)], np.int64))
    assert calls(result) == 0
    assert result.audit["native_chunks"][0]["vector_elements"] == 0
    np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64),
                                  op.call(name, x, simd="scalar").view(np.uint64))
