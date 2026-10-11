"""Eigen affine producer: canonical graph, guards and independent ownership.

Run with CALMETRICS_TEST_EXPECT_EIGEN=1 against an Eigen-enabled AOT wheel.
Portable/Eigen-off arithmetic is also covered by the native interval fixture.
NaNs are compared by classification; every non-NaN IEEE bit is compared exactly.
"""

import ctypes
import os
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import calmetrics_engine as package
from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_mc_projection_primitives import (
    chunk_sum,
    declarations,
    layout_view,
    projection_expressions,
    selected_inputs,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CALMETRICS_TEST_EXPECT_EIGEN") != "1",
    reason="explicit Eigen-on acceptance; set CALMETRICS_TEST_EXPECT_EIGEN=1",
)


@pytest.fixture(autouse=True)
def selected_candidate_install():
    selected = os.environ.get("CME_SCREEN_INSTALL")
    if selected:
        assert Path(package._native.__file__).resolve().is_relative_to(Path(selected).resolve())


def ieee_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint64), expected[valid].view(np.uint64))


def data(steps=5, paths=3, factors=4, assets=9, layout="c", writable=False):
    arrays = {
        "x": ((np.arange(steps * paths * factors, dtype=np.float64).reshape(steps, paths, factors)
               % 31) - 15) / 512,
        "beta": ((np.arange(assets * factors, dtype=np.float64).reshape(assets, factors)
                  % 13) - 6) / 8192,
        "initial": (np.arange(assets, dtype=np.float64) % 3 - 1) / 1024,
        "w0": (np.arange(assets, dtype=np.float64) % 7 - 2) / 16,
        "w1": (np.arange(assets, dtype=np.float64) % 5 - 1) / 8,
    }
    result = {name: layout_view(value, layout) for name, value in arrays.items()}
    if writable:
        for value in result.values():
            value.flags.writeable = True
    return result


def graphs(assets, paths, order="C", period_only=False, initial="initial"):
    expressions = projection_expressions(assets, paths)
    if initial != "initial":
        expressions = {name: expression.replace(",initial)", f",{initial})")
                       for name, expression in expressions.items()}
    if period_only:
        expressions = {name: expression for name, expression in expressions.items()
                       if name.startswith("period_")}
    compiler = GraphCompiler(declarations())
    private = compiler.compile(expressions, result_format="typed", output_order=order)
    public = dict(expressions)
    public["public_R"] = f"expm1(affine_contract(x,transpose(beta),{initial}))"
    canonical = compiler.compile(public, result_format="typed", output_order=order)
    return private, canonical, tuple(expressions)


def bounds(steps):
    return np.array([0], np.int64), np.array([steps], np.int64)


def compare(actual, expected, names, inputs):
    for name in names:
        got, want = actual.named_outputs[name], expected.named_outputs[name]
        assert got.axes == want.axes and got.dtype == want.dtype
        np.testing.assert_array_equal(got.root_statuses, want.root_statuses)
        np.testing.assert_array_equal(got.root_statuses, np.zeros(len(got.values), np.int16))
        for status, target_status in zip(got.statuses, want.statuses, strict=True):
            if target_status is None:
                assert status is None
            else:
                np.testing.assert_array_equal(status, target_status)
                assert not status.flags.writeable
        for value, target in zip(got.values, want.values, strict=True):
            ieee_bits(value, target)
            assert not value.flags.writeable
            assert all(not np.shares_memory(value, source) for source in inputs.values())
    assert actual.audit["result_lifetime"] == "independent"
    for field in ("python_fallback", "python_operator_calls", "request_time_compilation"):
        assert actual.audit[field] == 0
    # This observer root prevents the whole private closure, rather than merely
    # switching the producer to another optimized implementation.
    assert chunk_sum(expected, "pointwise_tiles") == 0
    for chunk in actual.audit["native_chunks"]:
        assert chunk["operator_workspace_capacity_bytes"] <= actual.plan.estimated_worker_scratch_bytes


def hits(result):
    return chunk_sum(result, "vector_elements")


@pytest.mark.parametrize("assets", [8, 9, 17])
@pytest.mark.parametrize("layout", ["c", "f", "strided", "negative"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_packet_assets_strides_tail_and_canonical_scalar_R(assets, layout, order):
    inputs = data(assets=assets, layout=layout)
    originals = {name: value.copy() for name, value in inputs.items()}
    graph, generic, names = graphs(assets, 3, order)
    starts, ends = bounds(5)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        expected = scheduler.execute(generic, selected_inputs(generic, inputs), starts, ends)
        actual = scheduler.execute(graph, selected_inputs(graph, inputs), starts, ends)
        later = scheduler.execute(graph, selected_inputs(graph, inputs), starts, ends)
    compare(actual, expected, names, inputs)
    compare(later, expected, names, inputs)
    assert hits(actual) == 5 * 3 * (assets - assets % 8)
    assert chunk_sum(actual, "numeric_arena_bytes") == 0
    coefficient_stride = inputs["beta"].strides[0] // 8
    for chunk in actual.audit["native_chunks"]:
        expected_copy = 4 * assets * 8 if coefficient_stride != 1 else 0
        assert chunk["algorithm_copy_bytes"] == expected_copy
    scalar_R = op.expm1(op.affine_contract(inputs["x"], inputs["beta"].T,
                                         inputs["initial"], simd="scalar"), simd="scalar")
    ieee_bits(expected.named_outputs["public_R"].values[0], scalar_R)
    for name in names:
        old = actual.named_outputs[name].values[0]
        new = later.named_outputs[name].values[0]
        assert not np.shares_memory(old, new)
        ieee_bits(old, expected.named_outputs[name].values[0])
        if old.ndim == 2:
            assert old.flags.f_contiguous if order == "F" else old.flags.c_contiguous
    for name, original in originals.items():
        ieee_bits(inputs[name], original)


@pytest.mark.parametrize("layout", ["c", "negative"])
def test_intervals_share_fixed_tail_pack_across_different_tile_capacities(layout):
    inputs = data(steps=259, paths=17, assets=17, layout=layout)
    graph, generic, names = graphs(17, 17, "F")
    starts = np.array([0, 1, 258, 0], np.int64)
    ends = np.array([1, 258, 259, 259], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        expected = scheduler.execute(generic, inputs, starts, ends)
        prepared = scheduler.prepare_execution(graph, inputs, starts, ends)
        actual = prepared.run_snapshot()
        snapshots = {name: [value.copy() for value in actual.named_outputs[name].values]
                     for name in names}
        later = prepared.run_snapshot()
    compare(actual, expected, names, inputs)
    compare(later, expected, names, inputs)
    assert hits(actual) == (1 + 257 + 1 + 259) * 17 * 16
    assert chunk_sum(actual, "pointwise_tiles") == 6
    for chunk in actual.audit["native_chunks"]:
        assert chunk["algorithm_copy_bytes"] == 4 * 17 * 8
    for name in names:
        for old, saved, new in zip(actual.named_outputs[name].values, snapshots[name],
                                   later.named_outputs[name].values, strict=True):
            ieee_bits(old, saved)
            assert not np.shares_memory(old, new)


def test_prepared_reads_current_coefficients_initial_and_preserves_previous_roots():
    inputs = data(assets=17, writable=True)
    graph, generic, names = graphs(17, 3)
    starts, ends = bounds(5)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, starts, ends)
        first = prepared.run_snapshot()
        snapshots = {name: first.named_outputs[name].values[0].copy() for name in names}
        inputs["beta"][:] *= 2.0
        inputs["initial"][:] += 0.125
        inputs["x"][:] += 1.0 / 16
        current = prepared.run_snapshot()
        expected = scheduler.execute(generic, inputs, starts, ends)
    compare(current, expected, names, inputs)
    assert hits(first) == hits(current) == 5 * 3 * 16
    assert chunk_sum(first, "algorithm_copy_bytes") == chunk_sum(current, "algorithm_copy_bytes") == 4 * 17 * 8
    assert not np.array_equal(first.named_outputs["period_0"].values[0], current.named_outputs["period_0"].values[0])
    for name in names:
        ieee_bits(first.named_outputs[name].values[0], snapshots[name])
        assert not np.shares_memory(first.named_outputs[name].values[0], current.named_outputs[name].values[0])


@pytest.mark.parametrize("steps,paths,factors,assets,packet", [
    (3, 4, 0, 9, False), (3, 4, 4, 0, False),
    (0, 4, 4, 9, False), (3, 0, 4, 9, False),
    (2, 3, 2048, 8, True), (2, 3, 2049, 8, False),
    (2, 3, 4, 7, False),
])
def test_empty_axes_and_exact_pack_capacity_guards(steps, paths, factors, assets, packet):
    inputs = data(steps, paths, factors, assets)
    empty = steps == 0 or paths == 0
    graph, generic, names = graphs(assets, paths, period_only=empty)
    starts, ends = bounds(steps)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        expected = scheduler.execute(generic, selected_inputs(generic, inputs), starts, ends)
        actual = scheduler.execute(graph, selected_inputs(graph, inputs), starts, ends)
    compare(actual, expected, names, inputs)
    assert bool(hits(actual)) is packet
    if not packet:
        assert chunk_sum(actual, "algorithm_copy_bytes") == 0
    else:
        assert hits(actual) == steps * paths * 8
        assert chunk_sum(actual, "algorithm_copy_bytes") == 128 * 1024
    for name in ("period_0", "period_1"):
        assert actual.named_outputs[name].values[0].shape == (paths, steps)
        if assets == 0:
            ieee_bits(actual.named_outputs[name].values[0], np.zeros((paths, steps)))


def test_initial_first_and_scalar_initial_keep_factor_tree():
    inputs = data(steps=2, paths=2, factors=2, assets=9, writable=True)
    inputs["x"][:] = np.array([1e16, -1e16])
    inputs["beta"][:] = 1.0
    inputs["initial"][:] = 1.0
    starts, ends = bounds(2)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        for initial in ("initial", "1.0"):
            graph, generic, names = graphs(9, 2, initial=initial)
            actual = scheduler.execute(graph, selected_inputs(graph, inputs), starts, ends)
            expected = scheduler.execute(generic, selected_inputs(generic, inputs), starts, ends)
            compare(actual, expected, names, inputs)
            assert hits(actual) == 2 * 2 * 8
            assert np.all(actual.named_outputs["period_0"].values[0] == 0.0)


@pytest.mark.parametrize("layout", ["c", "f", "negative"])
def test_ieee_extremes_zero_products_and_healthy_independent_assets(layout):
    values = data(steps=4, paths=2, factors=2, assets=17, writable=True)
    values["x"][:] = np.array([[[1e16, -1e16], [np.inf, 0.]],
                               [[-0., -0.], [1e-20, -1e-20]],
                               [[710., 0.], [-np.inf, 0.]],
                               [[np.nan, 0.], [-745., 0.]]])
    values["beta"][:] = 0.0
    values["beta"][0] = [1., 1.]
    values["beta"][8] = [1., 0.]
    values["beta"][16] = [1., -1.]
    values["initial"][:] = -0.0
    values["initial"][0] = 1.0
    values["w0"][1] = values["w1"][8] = 0.0
    inputs = {name: layout_view(value, layout) for name, value in values.items()}
    originals = {name: value.copy() for name, value in inputs.items()}
    graph, generic, names = graphs(17, 2, "F")
    starts, ends = bounds(4)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = scheduler.execute(graph, inputs, starts, ends)
        expected = scheduler.execute(generic, inputs, starts, ends)
    compare(actual, expected, names, inputs)
    assert hits(actual) == 4 * 2 * 16
    scalar_R = op.expm1(op.affine_contract(inputs["x"], inputs["beta"].T,
                                         inputs["initial"], simd="scalar"), simd="scalar")
    ieee_bits(expected.named_outputs["public_R"].values[0], scalar_R)
    assert np.isnan(scalar_R[0, 1, 1])  # 0*Inf is consumed, never skipped.
    for name, original in originals.items():
        ieee_bits(inputs[name], original)


# Exact current SDK values, read statically under its __arm64__ branch:
# /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/fenv.h:127,147-150.
# No inferred Linux/x86 constants are used for this macOS arm64 screening file.
DARWIN_ARM64_ROUNDS = {"nearest": 0x00000000, "up": 0x00400000,
                       "down": 0x00800000, "zero": 0x00C00000}


@contextmanager
def rounding(mode):
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        pytest.skip("screening round constants are pinned to the observed macOS arm64 SDK")
    library = ctypes.CDLL(None)
    library.fegetround.argtypes, library.fegetround.restype = [], ctypes.c_int
    library.fesetround.argtypes, library.fesetround.restype = [ctypes.c_int], ctypes.c_int
    original = library.fegetround()
    assert library.fesetround(mode) == 0
    try:
        assert library.fegetround() == mode
        yield
        assert library.fegetround() == mode
    finally:
        assert library.fesetround(original) == 0


@pytest.mark.parametrize("mode", tuple(DARWIN_ARM64_ROUNDS))
def test_directed_rounds_use_scalar_duty_and_preserve_canonical_bits(mode):
    inputs = data(steps=3, paths=3, assets=17)
    graph, generic, names = graphs(17, 3)
    starts, ends = bounds(3)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with rounding(DARWIN_ARM64_ROUNDS[mode]):
            expected = scheduler.execute(generic, inputs, starts, ends)
            actual = scheduler.execute(graph, inputs, starts, ends)
            compare(actual, expected, names, inputs)
            assert hits(actual) == (3 * 3 * 16 if mode == "nearest" else 0)
            assert chunk_sum(actual, "algorithm_copy_bytes") == (4 * 17 * 8 if mode == "nearest" else 0)


def test_structure_failure_after_pack_candidate_has_same_error_and_recovery():
    inputs = data(assets=9)
    graph, generic, names = graphs(9, 3)
    starts, ends = bounds(5)
    invalid = dict(inputs, initial=np.zeros(8))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        messages = []
        for selected in (graph, generic):
            with pytest.raises((ValueError, RuntimeError)) as failure:
                scheduler.execute(selected, invalid, starts, ends)
            messages.append(str(failure.value))
        assert messages[0] == messages[1]
        current = scheduler.execute(graph, inputs, starts, ends)
        expected = scheduler.execute(generic, inputs, starts, ends)
    compare(current, expected, names, inputs)
    assert hits(current) == 5 * 3 * 8
