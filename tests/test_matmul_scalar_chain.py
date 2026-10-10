"""Private terminal scalar chains: geometry, numerical guard and ownership."""

import ctypes

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import declarations, settings
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks


def payload(layout="cc", shape=(512, 256, 512)):
    m, k, n = shape
    rng = np.random.default_rng(6207)
    a = np.array(rng.uniform(-0.01, 0.01, (m, k)), order=layout[0].upper())
    b = np.array(rng.uniform(-0.01, 0.01, (k, n)), order=layout[1].upper())
    a.flags.writeable = b.flags.writeable = False
    return a, b


def canonical(a, b, steps):
    value = op.matmul(a, b)
    for name, scalar, left in steps:
        value = getattr(op, name)(scalar, value) if left else getattr(op, name)(value, scalar)
    return value


def verify(result, expected, a, b, order, library, arena=True):
    for value in result.outputs[0].values:
        assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
        assert not value.flags.writeable
        assert not np.shares_memory(value, a) and not np.shares_memory(value, b)
        np.testing.assert_array_equal(np.isnan(value), np.isnan(expected))
        np.testing.assert_array_equal(np.isposinf(value), np.isposinf(expected))
        np.testing.assert_array_equal(np.isneginf(value), np.isneginf(expected))
        np.testing.assert_array_equal(value == 0, expected == 0)
        np.testing.assert_array_equal(
            np.signbit(value[expected == 0]), np.signbit(expected[expected == 0])
        )
        np.testing.assert_allclose(value, expected, rtol=0, atol=0.000049)
    for chunk in chunks(result):
        assert chunk["accelerate_matmul_calls"] == (chunk["rows"] if library and ACCELERATE else 0)
        if arena:
            assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
        observed = (
            chunk["numeric_arena_bytes"]
            + chunk["mask_arena_bytes"]
            + chunk["operator_workspace_capacity_bytes"]
            + chunk["native_library_workspace_reserve_bytes"]
        )
        assert observed <= result.plan.estimated_worker_scratch_bytes
    assert result.audit["python_operator_calls"] == result.audit["python_fallback"] == 0


@pytest.mark.parametrize("layout", ["cc", "cf", "fc", "ff"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_terminal_chain_all_lanes_use_library_and_keep_old_outputs(layout, order, lane):
    a, b = payload(layout)
    before = a.copy(), b.copy()
    graph = GraphCompiler(declarations()).compile("matmul(a,b)*0.5+0.25", output_order=order)
    z = np.zeros(2, np.int64)
    expected = canonical(a, b, [("multiply", 0.5, False), ("add", 0.25, False)])
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(graph, {"a": a, "b": b}, z, z)
        saved = first.outputs[0].values[0].copy()
        second = scheduler.execute(graph, {"a": a, "b": b}, z, z)
        assert scheduler.peak_active_cpu_tokens <= 2
    for result in (first, second):
        verify(result, expected, a, b, order, True)
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    assert not np.shares_memory(first.outputs[0].values[0], first.outputs[0].values[1])
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])


STEPS = [
    ("0.7*matmul(a,b)-0.125", [("multiply", 0.7, True), ("subtract", 0.125, False)]),
    ("0.125-matmul(a,b)*(-0.7)", [("multiply", -0.7, False), ("subtract", 0.125, True)]),
    ("(matmul(a,b)+0.125)*0.7", [("add", 0.125, False), ("multiply", 0.7, False)]),
    (
        "((matmul(a,b)*0.7+0.125)*(-0.6)-0.0625)*0.9",
        [
            ("multiply", 0.7, False),
            ("add", 0.125, False),
            ("multiply", -0.6, False),
            ("subtract", 0.0625, False),
            ("multiply", 0.9, False),
        ],
    ),
    (
        "((((((((matmul(a,b)*0.7)*0.8)*0.9)*0.6)*0.5)*0.4)*0.3)*0.2)",
        [("multiply", x, False) for x in [0.7, 0.8, 0.9, 0.6, 0.5, 0.4, 0.3, 0.2]],
    ),
]


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("expression,steps", STEPS)
def test_general_scalars_and_long_pairs_preserve_original_arithmetic(expression, steps, order):
    a, b = payload("ff")
    g = GraphCompiler(declarations()).compile(expression, output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(g, {"a": a, "b": b}, z, z)
    literal_negation = "(-" in expression
    verify(r, canonical(a, b, steps), a, b, order, not literal_negation, arena=not literal_negation)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "expression,steps",
    [
        ("matmul(a,b)/2", [("divide", 2.0, False)]),
        ("matmul(a,b)*2", [("multiply", 2.0, False)]),
        ("matmul(a,b)*0", [("multiply", 0.0, False)]),
        ("matmul(a,b)*(-0.0)", [("multiply", -0.0, False)]),
        ("matmul(a,b)*0.00000000000000000001", [("multiply", 1e-20, False)]),
        ("matmul(a,b)+32", [("add", 32.0, False)]),
    ],
)
def test_unqualified_scalars_keep_bitwise_original_math(expression, steps, order):
    a, b = payload("ff")
    g = GraphCompiler(declarations()).compile(expression, output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(g, {"a": a, "b": b}, z, z)
    expected = canonical(a, b, steps)
    verify(r, expected, a, b, order, False, arena="/" not in expression and "(-" not in expression)
    np.testing.assert_array_equal(
        np.asarray(r.outputs[0].values[0], order="C").view(np.uint64),
        np.asarray(expected, order="C").view(np.uint64),
    )


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "fault",
    [
        "zero",
        "negative_zero",
        "nonfinite",
        "too_small",
        "too_large",
        "reverse",
        "step",
        "broadcast",
    ],
)
def test_input_rejection_retains_classes_and_exact_result(fault, order):
    a, b = payload("ff")
    a.setflags(write=True)
    if fault in {"zero", "negative_zero", "nonfinite", "too_small", "too_large"}:
        a[2, 3] = {
            "zero": 0.0,
            "negative_zero": -0.0,
            "nonfinite": np.inf,
            "too_small": 2.0**-129,
            "too_large": 2.0**129,
        }[fault]
    elif fault == "reverse":
        a = a[:, ::-1]
        b = b[::-1]
    elif fault == "step":
        owner = np.full((a.shape[0], a.shape[1] * 2), np.nan)
        owner[:, ::2] = a
        a = owner[:, ::2]
    else:
        a = np.broadcast_to(a[:1], a.shape)
    a.flags.writeable = b.flags.writeable = False
    before = a.copy(), b.copy()
    steps = [("multiply", 0.5, False), ("add", 0.25, False)]
    g = GraphCompiler(declarations()).compile("matmul(a,b)*0.5+0.25", output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(g, {"a": a, "b": b}, z, z)
    expected = canonical(a, b, steps)
    verify(r, expected, a, b, order, False)
    np.testing.assert_array_equal(
        np.asarray(r.outputs[0].values[0], order="C").view(np.uint64),
        np.asarray(expected, order="C").view(np.uint64),
    )
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])


@pytest.mark.parametrize("order", ["C", "F"])
def test_final_cancellation_replays_original_zero_sign(order):
    a, b = payload("ff")
    a.setflags(write=True)
    b.setflags(write=True)
    # Every cell has cancellation; no zero inputs force a shortcut rejection.
    a.fill(0.01)
    b.fill(0.01)
    b[::2] *= -1
    a[0, :3] = [1.0, 2.0**-54, -1.0]
    b[:3, 0] = [1.0, 1.0, 1.0]
    a.flags.writeable = b.flags.writeable = False
    expected = canonical(a, b, [("multiply", -0.7, False), ("add", 0.0, False)])
    decl = declarations()
    decl["scale"] = "scalar"
    g = GraphCompiler(decl).compile("matmul(a,b)*scale+0.0", output_order=order)
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(g, {"a": a, "b": b}, z, z, parameters={"scale": -0.7})
    verify(r, expected, a, b, order, True)
    assert np.any(expected == 0)


@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_parameters_are_requalified_and_results_are_independent(order):
    a, b = payload("cf")
    decl = declarations()
    decl["scale"] = "scalar"
    g = GraphCompiler(decl).compile("matmul(a,b)*scale+0.25", output_order=order)
    z = np.zeros(1, np.int64)
    parameters = np.array([0.7])
    with AdaptiveScheduler(cpu_budget=1) as s:
        prepared = s.prepare_execution(g, {"a": a, "b": b}, z, z, parameters=parameters)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        parameters[0] = -0.7
        second = prepared.run_snapshot()
        fallback = s.execute(g, {"a": a, "b": b}, z, z, parameters={"scale": 2.0})
    for r, scale in ((first, 0.7), (second, -0.7)):
        verify(
            r,
            canonical(a, b, [("multiply", scale, False), ("add", 0.25, False)]),
            a,
            b,
            order,
            True,
        )
    verify(
        fallback,
        canonical(a, b, [("multiply", 2.0, False), ("add", 0.25, False)]),
        a,
        b,
        order,
        False,
    )
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])


@pytest.mark.parametrize("mode", ["comparison", "multiple", "isolate"])
def test_nonterminal_or_observed_consumers_retain_ordered_matmul(mode):
    a, b = payload("cf")
    expr = "matmul(a,b)*0.5+0.25"
    if mode == "comparison":
        expr = f"({expr})>0.25"
    elif mode == "multiple":
        expr = [expr, "matmul(a,b)"]
    g = GraphCompiler(declarations()).compile(
        expr, error_policy="isolate" if mode == "isolate" else "raise"
    )
    z = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(g, {"a": a, "b": b}, z, z)
    expected = canonical(a, b, [("multiply", 0.5, False), ("add", 0.25, False)])
    if mode == "comparison":
        expected = expected > 0.25
    np.testing.assert_array_equal(r.outputs[0].values[0], expected)
    assert all(c["accelerate_matmul_calls"] == 0 for c in chunks(r))


@pytest.mark.skipif(not ACCELERATE, reason="qualified macOS arm64 rounding environment")
def test_non_nearest_rounding_declines_chain_library_and_restores_mode():
    # No matrix arithmetic is reused across rounding modes. The process's
    # original mode is restored even if an assertion fails.
    runtime = ctypes.CDLL(None)
    previous = runtime.fegetround()
    a, b = payload("ff")
    g = GraphCompiler(declarations()).compile("matmul(a,b)*0.5+0.25")
    z = np.zeros(1, np.int64)
    try:
        assert runtime.fesetround(0x400000) == 0  # FE_DOWNWARD on arm64 macOS.
        expected = canonical(a, b, [("multiply", 0.5, False), ("add", 0.25, False)])
        with AdaptiveScheduler(cpu_budget=1) as s:
            r = s.execute(g, {"a": a, "b": b}, z, z)
        assert all(c["accelerate_matmul_calls"] == 0 for c in chunks(r))
        np.testing.assert_array_equal(r.outputs[0].values[0], expected)
    finally:
        assert runtime.fesetround(previous) == 0
