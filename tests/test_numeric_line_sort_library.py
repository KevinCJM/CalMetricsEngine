"""Whole-operator/graph coverage of guarded native numeric-line sorting."""
import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler
from calmetrics_engine import operators as op

from .test_axis_path_primitives import graph_for, layout_view
from .test_compact_path_execution import (
    canonical_spec,
    check_ownership,
    chunk_sum,
    compare_selected,
    compile_compact,
    execute,
    fixture,
    process_config,
    saved_outputs,
)
from .test_typed_results import config


def bits_equal(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
                                  np.asarray(expected).view(np.uint64))


def library_enabled():
    _, audit = op.quantiles_axis(np.arange(2048, dtype=np.float64) + 1.,
                                np.array([.25, .5]), 0, audit=True)
    return audit["highway_vqsort_calls"] == 1


@pytest.mark.parametrize("layout", ["c", "f", "padded", "negative"])
@pytest.mark.parametrize("domain", ["finite", "nan", "negative_zero", "mixed_zero", "subnormal"])
def test_direct_large_lines_preserve_bits_and_guard(layout, domain):
    base = ((np.arange(3 * 2048).reshape(3, 2048) * 17) % 4093).astype(np.float64) + 1.
    if domain == "nan":
        base[:, -1] = np.array([0x7ff8000000000021], np.uint64).view(np.float64)[0]
    elif domain == "negative_zero":
        base[:, -1] = -0.
    elif domain == "mixed_zero":
        base[:, -2:] = [0., -0.]
    elif domain == "subnormal":
        base[:, -1] = np.nextafter(0., 1.)
    values = layout_view(base, layout)
    q = np.array([1., .125, .5, 0., .5])
    original = values.copy()
    expected = op.quantiles_axis(values, q, 1, simd="scalar")
    actual, audit = op.quantiles_axis(values, q, 1, audit=True)
    bits_equal(actual, expected)
    bits_equal(values, original)
    assert audit["highway_vqsort_calls"] == (3 if domain == "finite" and library_enabled() else 0)
    assert audit["algorithm_copy_bytes"] == values.size * 8
    assert audit["input_copy_bytes"] == 0
    assert not np.shares_memory(actual, values)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_large_axis_graph_library_audit_and_independent_results(lane, order):
    values = layout_view((np.arange(3 * 2048).reshape(3, 2048) % 1021 + 1.).astype(float), "f")
    q = np.array([1., .25, 0., .5])
    inputs = {"values": values, "q": q}
    graph = graph_for("quantiles_axis", 2, 1, order)
    bounds = np.zeros(4, np.int64)
    expected = op.quantiles_axis(values, q, 1, simd="scalar")
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(graph, inputs, bounds, bounds)
        second = scheduler.execute(graph, inputs, bounds, bounds)
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        for result in (first, second):
            assert chunk_sum(result, "highway_vqsort_calls") == (12 if library_enabled() else 0)
            for value in result.outputs[0].values:
                bits_equal(value, expected)
                assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
            np.testing.assert_array_equal(result.outputs[0].root_statuses, np.zeros(4, np.int16))
    for previous, current in zip(first.outputs[0].values, second.outputs[0].values, strict=True):
        bits_equal(previous, expected)
        assert not np.shares_memory(previous, current)
        assert not np.shares_memory(previous, values)


@pytest.mark.parametrize("lane", ["single", "process_inline", "process_shared", "hard_stop"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_compact_fan_and_post_use_one_native_sort_contract(lane, order):
    spec, inputs = fixture("period", steps=4, paths=2048, combinations=2, layout="F")
    graph = compile_compact(spec, order=order)
    generic = compile_compact(canonical_spec(spec), order=order)
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    parameters = {"n0": 17.3, "target": .1}
    with AdaptiveScheduler(cpu_budget=1) as reference:
        expected = execute(reference, generic, inputs, starts, ends, parameters)
    cfg = config("single") if lane == "single" else process_config(lane)
    with AdaptiveScheduler(cpu_budget=2, config=cfg) as scheduler:
        options = {"hard_stop": lane == "hard_stop", "timeout": 20.}
        first = execute(scheduler, graph, inputs, starts, ends, parameters, **options)
        saved = saved_outputs(first)
        second = execute(scheduler, graph, inputs, starts, ends, parameters, **options)
        for result in (first, second):
            compare_selected(result, expected)
            assert chunk_sum(result, "compact_path_groups") == 2
            # Two singleton-probability post nodes now use rank selection;
            # the twelve shared multi-probability fan sorts remain unchanged.
            assert chunk_sum(result, "highway_vqsort_calls") == (12 if library_enabled() else 0)
        check_ownership(first, second, saved, inputs)
