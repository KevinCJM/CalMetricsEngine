"""Independent covariance roots, canonical replay, budget and worker evidence."""

import ctypes
import os
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import settings

MATRIX = dict(kind="matrix", axes=["time", "asset"], shape=["T", "D"])
ACTIVE = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and int(platform.mac_ver()[0].split(".")[0] or 0) >= 15
    and os.environ.get("CME_EXPECT_ACCELERATE") != "0"
    and os.environ.get("CME_EXPECT_EIGEN") != "0"
)
active = pytest.mark.skipif(not ACTIVE, reason="enabled macOS15 arm64 libraries")


def inputs(layout="c", n=513, d=257):
    x = np.random.default_rng(91021).normal(size=(n, d))
    if layout in {"f", "reverse_f"}:
        x = np.array(x, order="F")
    elif layout == "padded_c":
        owner = np.full((n, d + 3), np.nan)
        owner[:, :d] = x
        x = owner[:, :d]
    elif layout == "padded_f":
        owner = np.full((n + 3, d), np.nan, order="F")
        owner[:n] = x
        x = owner[:n]
    if layout in {"reverse_c", "reverse_f"}:
        x = x[::-1]
    x.flags.writeable = False
    return x


def compare(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_array_equal(actual > 0, expected > 0)
    np.testing.assert_array_equal(actual == 0, expected == 0)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite]), np.signbit(expected[finite]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.diag(actual).view(np.uint64), np.diag(expected).view(np.uint64))


def counts(result, key):
    return sum(chunk[key] for chunk in result.audit["native_chunks"])


@active
@pytest.mark.parametrize("layout", ["c", "f", "padded_c", "padded_f", "reverse_c", "reverse_f"])
@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_library_root_transport_and_owned_results(layout, order, lane):
    x = inputs(layout)
    before = x.view(np.uint64).copy()
    expected = op.covariance(x)
    # Existing IPC packs padded F to C and descending F to positive F.
    # Keep exact diagonal parity to the actual canonical worker input.
    transported = expected
    if lane.startswith("process") and layout in {"padded_f", "reverse_f"}:
        transported = op.covariance(np.array(x, order="F" if layout == "reverse_f" else "C"))
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)", output_order=order)
    starts, ends = np.zeros(2, np.int64), np.full(2, len(x), np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"x": x}, starts, ends)
        again = scheduler.execute(graph, {"x": x}, starts, ends)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    assert counts(result, "accelerate_covariance_calls") == 2
    assert counts(result, "accelerate_covariance_attempts") == 2
    assert counts(result, "covariance_syrk_calls") == 2
    assert counts(result, "covariance_fallbacks") == 0
    assert counts(result, "covariance_panel_materialization_bytes") == 2 * x.size * 8
    assert counts(result, "input_copy_bytes") == 0
    assert counts(result, "algorithm_copy_bytes") == 0
    for actual in result.outputs[0].values:
        compare(actual, transported)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert not actual.flags.writeable and not np.shares_memory(actual, x)
        assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[0].values[1])
    assert not np.shares_memory(result.outputs[0].values[0], again.outputs[0].values[0])
    np.testing.assert_array_equal(x.view(np.uint64), before)
    reserve = (len(x) + 3) * x.shape[1] * 8 + 32 * 1024 * 1024
    assert result.plan.estimated_worker_scratch_bytes >= reserve
    for chunk in result.audit["native_chunks"]:
        assert chunk["native_library_workspace_reserve_bytes"] == reserve
        assert chunk["operator_workspace_capacity_bytes"] >= (len(x) + 3) * x.shape[1] * 8


@active
@pytest.mark.parametrize("layout", ["c", "f", "reverse_f"])
@pytest.mark.parametrize("pattern", ["constant", "sparse", "dense", "scaled", "tiny", "late_tiny"])
def test_replay_preserves_classes_and_accounts_for_rejected_work(layout, pattern):
    n = 1024 if pattern == "sparse" else 1041
    x = inputs(layout, n=n, d=256)
    x.setflags(write=True)
    if pattern == "constant":
        x[:, -1] = 7
    elif pattern == "sparse":
        x[:, 0] = np.tile([1.0, 1.0, -1.0, -1.0], n // 4)
        x[:, 1] = np.tile([1.0, -1.0, 1.0, -1.0], n // 4)
    elif pattern == "dense":
        x.fill(0)
    elif pattern == "scaled":
        x *= 1e3
    elif pattern == "tiny":
        x *= 2.0**-450
    else:
        x[:, -1] = 0
        x[-1, -1] = 2.0**-450
    x.setflags(write=False)
    expected = op.covariance(x)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.zeros(1, np.int64), np.array([n], np.int64))
    actual = result.outputs[0].values[0]
    compare(actual, expected)
    if pattern == "dense":
        assert counts(result, "covariance_zero_calls") == 1
        assert counts(result, "accelerate_covariance_attempts") == 0
        assert counts(result, "covariance_panel_materialization_bytes") == 0
        assert counts(result, "covariance_fallbacks") == 0
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"] == 0
    elif pattern in {"constant", "sparse"}:
        assert counts(result, "accelerate_covariance_calls") == 1
        assert counts(result, "covariance_replayed_cells") > 0
    else:
        assert counts(result, "covariance_fallbacks") == 1
        assert counts(result, "accelerate_covariance_calls") == 0
        assert counts(result, "covariance_panel_materialization_bytes") == x.size * 8
        assert counts(result, "covariance_replayed_cells") >= x.shape[1] * (x.shape[1] + 1) // 2
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"] == 0


@pytest.mark.parametrize("layout", ["c", "f", "reverse_f"])
@pytest.mark.parametrize("fault", [np.nan, np.inf, 1e8])
def test_unadmitted_payload_uses_canonical_without_attempt(layout, fault):
    x = inputs(layout)
    x.setflags(write=True)
    x[7, -1] = fault
    x.setflags(write=False)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
    assert counts(result, "accelerate_covariance_attempts") == 0
    expected = op.covariance(x)
    np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("mode", ["compare", "multiple", "isolate", "iterate"])
def test_discrete_shared_isolated_and_scoped_graphs_remain_canonical(mode):
    x = inputs(n=129, d=256)
    expected = op.covariance(x)
    expression = "covariance(x)>0" if mode == "compare" else "covariance(x)"
    if mode == "multiple":
        expression = ["covariance(x)", "trace(covariance(x))"]
    elif mode == "iterate":
        expression = "iterate(covariance(x),covariance(x),0,1)"
    graph = GraphCompiler({"x": MATRIX}).compile(expression, error_policy="isolate" if mode == "isolate" else "raise")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
    assert counts(result, "accelerate_covariance_attempts") == 0
    actual = result.outputs[0].values[0]
    np.testing.assert_array_equal(actual if mode == "compare" else actual.view(np.uint64),
                                  expected > 0 if mode == "compare" else expected.view(np.uint64))


@active
@pytest.mark.parametrize("layout,d", [("c", 257), ("f", 257), ("f", 256), ("padded_f", 256), ("reverse_c", 257), ("reverse_f", 257), ("reverse_f", 256)])
def test_prepared_rereads_payload_releases_rejected_panel_and_preserves_snapshot(layout, d):
    x = inputs(layout, d=d)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
        first = prepared.run_snapshot()
        before = first.outputs[0].values[0].copy()
        x.setflags(write=True)
        x *= 1e3
        x.setflags(write=False)
        second = prepared.run_snapshot()
        assert counts(second, "covariance_fallbacks") == 1
        assert second.audit["native_chunks"][0]["operator_workspace_capacity_bytes"] == 0
        compare(second.outputs[0].values[0], op.covariance(x))
        x.setflags(write=True)
        x *= 1e-3
        x.setflags(write=False)
        third = prepared.run_snapshot()
        assert counts(third, "accelerate_covariance_calls") == 1
        compare(third.outputs[0].values[0], op.covariance(x))
    np.testing.assert_array_equal(first.outputs[0].values[0], before)
    assert not np.shares_memory(first.outputs[0].values[0], third.outputs[0].values[0])


@active
def test_interval_reserve_covers_nonmonotonic_medium_panels_and_rejects_low_budget():
    x = inputs(n=4096, d=256)
    starts, ends = np.array([0, 1], np.int64), np.array([4096, 1042], np.int64)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    reserve = (2048 + 3) * x.shape[1] * 8 + 32 * 1024 * 1024
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, {"x": x}, starts, ends)
        assert plan.estimated_worker_scratch_bytes >= reserve
        with pytest.raises((ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc"):
            scheduler.execute(graph, {"x": x}, starts, ends, memory_budget_bytes=reserve - 1)
        result = scheduler.execute(graph, {"x": x}, starts, ends)
    for a, b, actual in zip(starts, ends, result.outputs[0].values, strict=True):
        compare(actual, op.covariance(x[a:b]))
    assert counts(result, "accelerate_covariance_calls") == 2


@active
@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("n", [4097, 8193, 32769])
def test_long_sequences_preserve_block_bounds_and_tail(layout, n):
    x = inputs(layout, n=n, d=256)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.zeros(1, np.int64), np.array([n], np.int64))
    compare(result.outputs[0].values[0], op.covariance(x))
    assert counts(result, "accelerate_covariance_calls") == 1
    assert counts(result, "covariance_syrk_calls") == (n + 511) // 512
    assert counts(result, "covariance_panel_materialization_bytes") == x.size * 8


@active
@pytest.mark.parametrize("layout", ["c", "f"])
def test_long_correlated_replay_keeps_all_attempt_costs(layout):
    x = inputs(layout, n=32769, d=256)
    x.setflags(write=True)
    x[:] = x * 0.1 + x[:, :1]
    x.setflags(write=False)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
    expected = op.covariance(x)
    np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64), expected.view(np.uint64))
    assert counts(result, "covariance_fallbacks") == 1
    assert counts(result, "covariance_syrk_calls") == (len(x) + 511) // 512
    assert counts(result, "covariance_replayed_cells") == x.shape[1] * (x.shape[1] + 1) // 2 + (x.shape[1] if layout == "f" else 0)
    assert counts(result, "covariance_panel_materialization_bytes") == x.size * 8


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("case", ["reverse", "step", "broadcast", "short", "narrow"])
def test_geometry_admission_preserves_original_values(layout, case):
    x = inputs(layout, n=31 if case == "short" else 65,
               d=255 if case == "narrow" else 512 if case == "step" else 256)
    if case == "reverse":
        x = x[::-1]
    elif case == "step":
        x = x[::2, ::2]
    elif case == "broadcast":
        x = np.broadcast_to(x[:1], x.shape)
    expected = op.covariance(x)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
    admitted = ACTIVE and case == "reverse"
    assert counts(result, "accelerate_covariance_attempts") == int(admitted)
    compare(result.outputs[0].values[0], expected)
    if not admitted:
        np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64), expected.view(np.uint64))


@active
@pytest.mark.parametrize("layout", ["c", "f", "reverse_c", "reverse_f"])
@pytest.mark.parametrize("policy", [0, 1])  # SDK thread_api.h: MULTI=0, SINGLE=1.
@pytest.mark.parametrize("replay", [False, True])
def test_calling_thread_policy_is_restored_after_acceptance_and_replay(layout, policy, replay):
    library = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    getter, setter = library.BLASGetThreading, library.BLASSetThreading
    getter.restype = ctypes.c_uint
    setter.argtypes, setter.restype = (ctypes.c_uint,), ctypes.c_int
    previous = getter()
    x = inputs(layout, d=256)
    if replay:
        x.setflags(write=True)
        x *= 1e3
        x.setflags(write=False)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    try:
        assert setter(policy) == 0 and getter() == policy
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            result = scheduler.execute(graph, {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
        assert getter() == policy
        assert counts(result, "accelerate_covariance_attempts") == 1
        assert counts(result, "covariance_fallbacks") == int(replay)
        assert counts(result, "accelerate_covariance_calls") == int(not replay)
    finally:
        assert setter(previous) == 0 and getter() == previous


@active
@pytest.mark.parametrize(
    "n,d,layout,order,lane",
    [
        (32, 256, "f", "C", "single"),
        (33, 256, "padded_f", "F", "single"),
        (34, 256, "f", "C", "thread"),
        (35, 256, "padded_f", "F", "thread"),
        (35, 256, "f", "C", "process_inline"),
        (34, 256, "f", "F", "process_shared"),
        (33, 256, "padded_f", "C", "process_inline"),
        (35, 256, "padded_f", "F", "process_shared"),
        (35, 2050, "f", "F", "single"),
    ],
)
def test_complete_centered_replay_keeps_packet_tails_transport_and_owned_results(n, d, layout, order, lane):
    x = inputs(layout, n=n, d=d)
    x.setflags(write=True)
    x *= 32.0
    x[:, 0] = np.where(np.arange(n) % 2, -64.0, 64.0)
    x.setflags(write=False)
    before = x.view(np.uint64).copy()
    # Dense F survives IPC; padded F is packed into logical C before execution.
    transported = np.array(x, order="C") if lane.startswith("process") and layout == "padded_f" else x
    expected = op.covariance(transported)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)", output_order=order)
    starts, ends = np.zeros(2, np.int64), np.full(2, n, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"x": x}, starts, ends)
        again = scheduler.execute(graph, {"x": x}, starts, ends)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    if lane.startswith("process"):
        assert result.plan.use_shared_memory == (lane == "process_shared")
    assert counts(result, "accelerate_covariance_attempts") == 2
    assert counts(result, "accelerate_covariance_calls") == 0
    assert counts(result, "covariance_syrk_calls") == 0
    assert counts(result, "covariance_fallbacks") == 2
    assert counts(result, "eigen_covariance_calls") == 2
    assert counts(result, "covariance_replayed_cells") == d * (d + 1)
    assert counts(result, "covariance_panel_materialization_bytes") == 2 * x.size * 8
    assert counts(result, "algorithm_copy_bytes") == 0
    panel_rows = n if d <= 2048 else 512
    reserve = (panel_rows + 3) * d * 8 + 32 * 1024 * 1024
    assert result.plan.estimated_worker_scratch_bytes >= reserve
    for chunk in result.audit["native_chunks"]:
        assert chunk["native_library_workspace_reserve_bytes"] == reserve
        assert chunk["operator_workspace_capacity_bytes"] == 0
    for old, new in zip(result.outputs[0].values, again.outputs[0].values, strict=True):
        np.testing.assert_array_equal(old.view(np.uint64), expected.view(np.uint64))
        np.testing.assert_array_equal(new.view(np.uint64), expected.view(np.uint64))
        assert old.flags.f_contiguous if order == "F" else old.flags.c_contiguous
        assert not old.flags.writeable and not new.flags.writeable
        assert not np.shares_memory(old, x) and not np.shares_memory(old, new)
    assert not np.shares_memory(*result.outputs[0].values)
    np.testing.assert_array_equal(x.view(np.uint64), before)
    if lane == "process_shared":
        assert result.audit["output_copy_bytes"] == 0


@active
@pytest.mark.parametrize("tiny_first", [False, True])
@pytest.mark.parametrize("rounding", [0, 0x00400000, 0x00800000, 0x00C00000])
def test_centered_replay_rejection_order_preserves_fenv_and_rounding(tiny_first, rounding):
    # Darwin arm64 fenv.h: two uint64 fields, FE_ALL_EXCEPT=0x9f, DIVBYZERO=2.
    libc = ctypes.CDLL(None)
    libc.fegetenv.argtypes = libc.fesetenv.argtypes = (ctypes.c_void_p,)
    libc.fesetround.argtypes = libc.feclearexcept.argtypes = libc.feraiseexcept.argtypes = (ctypes.c_int,)
    libc.fetestexcept.argtypes = (ctypes.c_int,)
    saved = (ctypes.c_uint64 * 2)()
    assert libc.fegetenv(ctypes.byref(saved)) == 0
    x = inputs("f", n=35, d=256)
    x.setflags(write=True)
    x.fill(0.0)
    signs = np.where(np.arange(len(x)) % 2, -1.0, 1.0)
    x[:, 0 if tiny_first else 255] = signs * 2.0**-600
    x[:, 1 if tiny_first else 0] = signs * 64.0
    x.setflags(write=False)
    before = x.view(np.uint64).copy()
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    try:
        with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
            assert libc.fesetround(rounding) == 0 and libc.fegetround() == rounding
            assert libc.feclearexcept(0x9F) == 0 and libc.feraiseexcept(2) == 0
            expected = op.covariance(x)
            expected_flags = libc.fetestexcept(0x9F)
            assert libc.fegetround() == rounding
            assert libc.feclearexcept(0x9F) == 0 and libc.feraiseexcept(2) == 0
            result = scheduler.execute(graph, {"x": x}, starts, ends)
            actual_flags = libc.fetestexcept(0x9F)
            assert libc.fegetround() == rounding
            assert actual_flags == expected_flags and actual_flags & 2
            actual = result.outputs[0].values[0]
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            attempted = int(rounding == 0)
            assert counts(result, "accelerate_covariance_attempts") == attempted
            assert counts(result, "covariance_fallbacks") == attempted
            assert counts(result, "covariance_syrk_calls") == 0
            assert counts(result, "covariance_panel_materialization_bytes") == attempted * x.size * 8
            assert result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"] == 0
        np.testing.assert_array_equal(x.view(np.uint64), before)
    finally:
        assert libc.fesetenv(ctypes.byref(saved)) == 0


@active
@pytest.mark.parametrize("layout", ["c", "f", "reverse_f"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("signed", [False, True])
def test_exact_zero_specialization_and_signed_zero_canonical_transport(layout, order, lane, signed):
    x = inputs(layout, n=65, d=256)
    x.setflags(write=True)
    x.fill(0)
    if signed:
        x[::2, ::2] = -0.0
    x.setflags(write=False)
    before = x.view(np.uint64).copy()
    expected = op.covariance(x)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)", output_order=order)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.zeros(2, np.int64), np.full(2, len(x), np.int64))
    assert counts(result, "covariance_zero_calls") == (0 if signed else 2)
    assert counts(result, "accelerate_covariance_attempts") == 0
    assert counts(result, "accelerate_covariance_calls") == 0
    assert counts(result, "covariance_panel_materialization_bytes") == 0
    assert counts(result, "covariance_fallbacks") == 0
    for actual in result.outputs[0].values:
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert not actual.flags.writeable and not np.shares_memory(actual, x)
        assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
    np.testing.assert_array_equal(x.view(np.uint64), before)
    assert not np.shares_memory(*result.outputs[0].values)


@active
@pytest.mark.parametrize("layout", ["c", "f", "reverse_f"])
def test_prepared_zero_rereads_sign_and_releases_earlier_panel(layout):
    x = inputs(layout)
    graph = GraphCompiler({"x": MATRIX}).compile("covariance(x)")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": x}, np.zeros(1, np.int64), np.array([len(x)], np.int64))
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        assert counts(first, "accelerate_covariance_calls") == 1
        for signed in (False, True):
            x.setflags(write=True)
            x.fill(0)
            if signed:
                x[0, 0] = -0.0
            x.setflags(write=False)
            result = prepared.run_snapshot()
            assert counts(result, "covariance_zero_calls") == int(not signed)
            assert counts(result, "accelerate_covariance_attempts") == 0
            assert result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"] == 0
            np.testing.assert_array_equal(result.outputs[0].values[0].view(np.uint64), op.covariance(x).view(np.uint64))
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("expression", ["covariance(x)", "correlation(x)", ["covariance(x)", "correlation(x)"]])
@pytest.mark.parametrize("pattern", ["normal", "tiny", "scaled", "nan", "constant", "internal"])
@pytest.mark.parametrize("layout", ["c", "f"])
def test_descending_padded_rows_keep_intervals_errors_and_snapshots(order, expression, pattern, layout):
    n, d = 133, 257
    if layout == "c":
        owner = np.full((2*n, d+3), np.nan)
        owner[::2, :d] = inputs(n=n, d=d)
        x = owner[-2::-2, :d]
    else:
        owner = np.full((n+3, d), np.nan, order="F")
        owner[:n] = inputs(n=n, d=d)
        x = owner[n-1::-1]
    if pattern == "tiny":
        x *= 2.0**-200
    elif pattern == "scaled":
        x *= 32.
    elif pattern == "nan":
        x[5, -1] = np.nan
    elif pattern == "constant":
        x[:, -1] = 2
    expressions = [expression] if isinstance(expression, str) else expression
    if pattern == "internal":
        expressions = [e+">0" for e in expressions]
    x.flags.writeable = False
    owner.flags.writeable = False
    before = owner.view(np.uint64).copy()
    compiled = GraphCompiler({"x": MATRIX}).compile(expressions, output_order=order,
        error_policy="isolate" if pattern == "constant" else "raise")
    starts, ends = np.array([0, 3], np.int64), np.array([n, n-5], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(compiled, {"x": x}, starts, ends)
        with pytest.raises(MemoryError):
            scheduler.plan(compiled, {"x": x}, starts, ends,
                           memory_budget_bytes=plan.estimated_total_memory_bytes-1)
        prepared = scheduler.prepare_execution(compiled, {"x": x}, starts, ends)
        result, again = prepared.run_snapshot(), prepared.run_snapshot()
    for root, expr in enumerate(expressions):
        fn = op.correlation if "correlation" in expr else op.covariance
        if pattern == "constant" and fn is op.correlation:
            assert np.all(result.outputs[root].root_statuses != 0)
            continue
        assert np.all(result.outputs[root].root_statuses == 0)
        for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
            expected = fn(x[start:end])
            actual = result.outputs[root].values[row]
            if pattern == "internal":
                np.testing.assert_array_equal(actual, expected > 0)
            else:
                compare(actual, expected)
            assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
            assert not actual.flags.writeable and not np.shares_memory(actual, x)
            assert not np.shares_memory(actual, again.outputs[root].values[row])
    if pattern in {"internal", "nan", "constant"}:
        assert counts(result, "accelerate_covariance_attempts") == 0
    elif ACTIVE:
        assert counts(result, "accelerate_covariance_attempts") == 2
    np.testing.assert_array_equal(owner.view(np.uint64), before)
