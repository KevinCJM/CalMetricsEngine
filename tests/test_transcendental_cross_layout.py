"""Bounded cross-layout library tiles preserve axes, ownership and fallback."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler
from calmetrics_engine import operators as op

from .test_matmul_library import settings
from .test_nonnegative_matmul_accelerate import ACCELERATE
from .test_transcendental_accelerate import calls, graph


def sample(name, shape, order, padded=False):
    x = np.random.default_rng(101001).uniform(.1, 10, shape)
    if name == "exp":
        x -= 5
    if padded:
        outer = list(shape)
        outer[0 if order == "F" else -1] += 3
        owner = np.full(outer, np.nan, order=order)
        x_slice = tuple(slice(0, n) for n in shape)
        owner[x_slice] = x
        x = owner[x_slice]
    else:
        x = np.array(x, order=order)
    x.flags.writeable = False
    return x


def compare(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)
    for predicate in (np.isnan, np.isfinite, np.signbit, lambda v: v == 0):
        np.testing.assert_array_equal(predicate(actual), predicate(expected))


@pytest.mark.parametrize("name", ["exp", "log"])
@pytest.mark.parametrize("shape", [(129, 513), (129, 33, 17)])
@pytest.mark.parametrize("source,target", [("F", "C"), ("C", "F")])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_tiles_in_actual_graph_and_transport(name, shape, source, target, lane):
    x = sample(name, shape, source)
    before = x.copy(order="K")
    compiled = pickle.loads(pickle.dumps(graph(name, len(shape), target)))
    starts, ends = np.zeros(2, np.int64), np.full(2, len(x), np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(compiled, {"x": x}, starts, ends)
        second = scheduler.execute(compiled, {"x": x}, starts, ends)
    expected_calls = (np.prod(shape[1:]) * ((shape[0] + 127) // 128) if source == "F"
                      else shape[0] * ((np.prod(shape[1:]) + 63) // 64))
    assert calls(first) == (2 * expected_calls if ACCELERATE else 0)
    assert first.outputs[0].order == target
    for i, value in enumerate(first.outputs[0].values):
        compare(value, op.call(name, x, simd="scalar"))
        assert value.flags.f_contiguous if target == "F" else value.flags.c_contiguous
        assert not value.flags.writeable
        assert not np.shares_memory(value, x)
        assert not np.shares_memory(value, second.outputs[0].values[i])
    assert not np.shares_memory(*first.outputs[0].values)
    np.testing.assert_array_equal(x.view(np.uint64), before.view(np.uint64))
    for chunk in first.audit["native_chunks"]:
        assert chunk["root_copy_bytes"] == chunk["input_copy_bytes"] == chunk["numeric_arena_bytes"] == 0
        if ACCELERATE:
            assert 0 < chunk["operator_workspace_capacity_bytes"] <= 65536
            assert chunk["algorithm_copy_bytes"] == 2 * x.nbytes * chunk["rows"]


@pytest.mark.parametrize("name", ["exp", "log"])
@pytest.mark.parametrize("shape", [(129, 513), (131, 2, 257), (1031, 1, 65), (1031, 65, 1)])
@pytest.mark.parametrize("source,target", [("F", "C"), ("C", "F")])
def test_padding_and_prepared_live_values(name, shape, source, target):
    # F first-axis padding and C row-pitch padding are both excluded from scans.
    # Rank-three C inner padding is deliberately ineligible.
    x = sample(name, shape, source, padded=True)
    starts, ends = np.array([1], np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph(name, len(shape), target), {"x": x}, starts, ends)
        first = prepared.run_snapshot()
        compare(first.outputs[0].values[0], op.call(name, x[1:], simd="scalar"))
        saved = first.outputs[0].values[0].copy()
        eligible = source == "F" or len(shape) == 2
        assert (calls(first) > 0) == (ACCELERATE and eligible)
        x.flags.writeable = True
        x[0] = np.nan  # Unconsumed window values must remain irrelevant.
        x[4] = .5
        x.flags.writeable = False
        second = prepared.run_snapshot()
        compare(second.outputs[0].values[0], op.call(name, x[1:], simd="scalar"))
        assert calls(second) == calls(first)
        np.testing.assert_array_equal(first.outputs[0].values[0], saved)
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])


@pytest.mark.parametrize("name", ["exp", "log"])
@pytest.mark.parametrize("source,target", [("F", "C"), ("C", "F")])
@pytest.mark.parametrize("shape,eligible", [((255, 257), False), ((256, 256), True),
                                          ((31, 2200), False), ((10000, 7), False)])
def test_tile_admission_boundaries(name, source, target, shape, eligible):
    x = sample(name, shape, source)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(name, 2, target), {"x": x},
                                   np.array([0], np.int64), np.array([len(x)], np.int64))
    assert (calls(result) > 0) == (ACCELERATE and eligible)
    compare(result.outputs[0].values[0], op.call(name, x, simd="scalar"))


@pytest.mark.parametrize("name,fault", [("exp", 701.), ("exp", np.nan),
                                      ("log", np.nextafter(0., 1.)), ("log", np.inf)])
@pytest.mark.parametrize("source,target", [("F", "C"), ("C", "F")])
def test_tiles_replay_late_outlier_and_preserve_canonical_math(name, fault, source, target):
    x = sample(name, (129, 513), source).copy(order=source)
    x[-1, -1] = fault
    x.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if np.isnan(fault):
            with pytest.raises(ValueError, match="DOMAIN_ERROR"):
                op.call(name, x, simd="scalar")
            with pytest.raises(ValueError, match="DOMAIN_ERROR"):
                scheduler.execute(graph(name, 2, target), {"x": x},
                                  np.array([0], np.int64), np.array([len(x)], np.int64))
            return
        result = scheduler.execute(graph(name, 2, target), {"x": x},
                                   np.array([0], np.int64), np.array([len(x)], np.int64))
    assert (calls(result) > 0) == ACCELERATE
    expected = op.call(name, x, simd="scalar")
    compare(result.outputs[0].values[0], expected)
    assert result.outputs[0].values[0][-1, -1].view(np.uint64) == expected[-1, -1].view(np.uint64)


@pytest.mark.parametrize("name", ["exp", "log"])
def test_padded_c_rows_use_no_tile(name):
    x = sample(name, (65, 1031), "C", padded=True)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(name, 2, "C"), {"x": x},
                                   np.array([0], np.int64), np.array([len(x)], np.int64))
    assert calls(result) == (65 if ACCELERATE else 0)
    compare(result.outputs[0].values[0], op.call(name, x, simd="scalar"))
    for chunk in result.audit["native_chunks"]:
        assert chunk["operator_workspace_capacity_bytes"] == chunk["algorithm_copy_bytes"] == 0
