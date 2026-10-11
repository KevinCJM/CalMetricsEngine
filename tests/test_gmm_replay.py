"""Bounded GMM replay: geometry envelope, transport and immutable generations."""

import hashlib
import os
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_typed_results import config

CUTOFF = 16 * 1024 * 1024


def enabled():
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and int(platform.mac_ver()[0].split(".")[0]) >= 15
        and os.environ.get("CME_EXPECT_EIGEN", "1") != "0"
    )


def fixture(n=262145, d=8, k=64, layout="C"):
    rng = np.random.default_rng(5631)
    x = rng.normal(size=(n, d))
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1, ::-1].copy()[::-1, ::-1]
    elif layout == "sliced":
        backing = np.empty((n * 2, d * 2))
        backing[::2, ::2] = x
        x = backing[::2, ::2]
    data = {
        "x": x,
        "w": np.full(k, 1 / k),
        "m": rng.normal(scale=0.2, size=(k, d)),
        "v": rng.uniform(0.7, 1.3, size=(k, d)),
    }
    for value in data.values():
        value.flags.writeable = False
    return data


def graph(data):
    types = {
        "x": {"kind": "matrix", "axes": ["time", "factor"], "shape": ["N", "D"]},
        "w": {"kind": "vector", "axes": ["component"], "shape": ["K"]},
        "m": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "v": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
    }
    fit = "gaussian_mixture_em(x,w,m,v,1,0)"
    return GraphCompiler(types).compile(
        [
            f"{projection}({fit})"
            for projection in (
                "mixture_weights",
                "mixture_means",
                "mixture_variances",
                "em_iterations",
                "em_log_likelihood",
                "em_converged",
            )
        ],
        result_format="typed",
    )


def exact(a, b):
    for first, second in zip(a.outputs, b.outputs, strict=True):
        np.testing.assert_array_equal(first.root_statuses, second.root_statuses)
        for x, y in zip(first.values, second.values, strict=True):
            np.testing.assert_array_equal(x.ravel().view(np.uint8), y.ravel().view(np.uint8))


@pytest.mark.parametrize("layout", ["C", "F", "reverse", "sliced"])
def test_replay_exact_scalar_and_copy_accounting(layout):
    n, d, k = 262145, 8, 64
    data = fixture(n, d, k, layout)
    before = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in data.items()}
    args = [data[key] for key in ("x", "w", "m", "v")]
    scalar = op.gaussian_mixture_em(*args, 1, 0.0, simd="scalar")
    first, audit = op.gaussian_mixture_em(*args, 1, 0.0, audit=True)
    np.testing.assert_array_equal(first.ravel().view(np.uint8), scalar.ravel().view(np.uint8))
    if enabled():
        scratch = k * (128 + 2 * d + 3) + 128 * d
        assert audit["workspace_bytes"] == scratch * 8
        # Init model + saved old means + committed sums + optional input packing.
        copies = k * (1 + 2 * d) + 3 * k * d + (0 if layout == "F" else 2 * n * d)
        assert audit["algorithm_copy_bytes"] == copies * 8
        assert audit["input_copy_bytes"] == 0
    assert before == {
        key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in data.items()
    }
    assert all(not value.flags.writeable for value in data.values())


@pytest.mark.parametrize(
    "n,d,k,selected",
    [
        (262143, 8, 64, False),
        (262144, 8, 64, True),
        (262145, 8, 64, True),
        (1118482, 8, 15, False),
        (262144, 7, 64, False),
        (262144, 33, 64, False),
    ],
)
def test_measured_shape_dispatch(n, d, k, selected):
    # Constant input exercises geometry without adding an enormous fixture allocation.
    x = np.broadcast_to(np.zeros((1, d)), (n, d))
    w, m, v = np.full(k, 1 / k), np.zeros((k, d)), np.ones((k, d))
    for value in (x, w, m, v):
        value.flags.writeable = False
    result, audit = op.gaussian_mixture_em(x, w, m, v, 1, 0.0, audit=True)
    expected = k * (128 + 2 * d + 3) + 128 * d if selected and enabled() else n * k + 3 * k
    assert audit["workspace_bytes"] == expected * 8
    assert op.em_iterations(result) == 1
    assert np.isfinite(result).all()


@pytest.mark.parametrize("ends", [[262143], [262144], [262143, 262144], [262144, 262143]])
def test_plan_covers_original_history_below_replay_cutoff(ends):
    data = fixture(262144)
    g = graph(data)
    ends = np.asarray(ends, np.int64)
    starts = np.zeros(ends.size, np.int64)
    largest_original = (262143 * 64 + 3 * 64) * 8
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(g, data, starts, ends)
        assert plan.estimated_worker_scratch_bytes >= largest_original
        # This bound applies also when only a maximum-sized interval was requested.
        with pytest.raises(
            (ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc|bad allocation"
        ):
            scheduler.execute(g, data, starts, ends, memory_budget_bytes=largest_original - 1)
        prepared = scheduler.prepare_execution(g, data, starts, ends).run_snapshot()
        actual = scheduler.execute(g, data, starts, ends)
        exact(prepared, actual)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_replay_transport_and_retained_generation(lane):
    data = fixture()
    g = graph(data)
    ends = np.asarray([262145, 262145], np.int64)
    starts = np.zeros(2, np.int64)
    # Prepared execution deliberately supports a single lane only. Compare
    # forced transport against its independent single-lane snapshot.
    with AdaptiveScheduler(cpu_budget=1) as control:
        prepared = control.prepare_execution(g, data, starts, ends).run_snapshot()
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(g, data, starts, ends)
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        frozen = [[x.copy() for x in output.values] for output in first.outputs]
        second = scheduler.execute(g, data, starts, ends)
        exact(prepared, first)
        exact(first, second)
        assert scheduler.peak_active_cpu_tokens <= 2
        assert first.audit["python_operator_calls"] == first.audit["python_fallback"] == 0
        for output, saved, newer in zip(first.outputs, frozen, second.outputs, strict=True):
            for old, copy, new in zip(output.values, saved, newer.values, strict=True):
                np.testing.assert_array_equal(
                    old.ravel().view(np.uint8), copy.ravel().view(np.uint8)
                )
                assert not np.shares_memory(old, new)


@pytest.mark.parametrize("fault", ["nonfinite_input", "zero_weights", "negative_variance"])
def test_replay_failure_isolation_and_independent_root(fault):
    data = fixture()
    types = {
        "x": {"kind": "matrix", "axes": ["time", "factor"], "shape": ["N", "D"]},
        "w": {"kind": "vector", "axes": ["component"], "shape": ["K"]},
        "m": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "v": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
    }
    fit = "gaussian_mixture_em(x,w,m,v,1,0)"
    g = GraphCompiler(types).compile(
        [f"mixture_means({fit})", f"em_iterations({fit})", "sum_axis(w,0)"],
        error_policy="isolate",
        result_format="typed",
    )
    bad = dict(data)
    key = "x" if fault == "nonfinite_input" else "w" if fault == "zero_weights" else "v"
    replacement = data[key].copy()
    replacement.reshape(-1)[-1] = (
        np.nan if fault == "nonfinite_input" else -1.0 if fault == "negative_variance" else 0.0
    )
    if fault == "zero_weights":
        replacement.fill(0.0)
    replacement.flags.writeable = False
    bad[key] = replacement
    starts, ends = np.zeros(1, np.int64), np.array([262145], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        retained = scheduler.execute(g, data, starts, ends)
        expected = scheduler.prepare_execution(g, bad, starts, ends).run_snapshot()
        failed = scheduler.execute(g, bad, starts, ends)
        exact(expected, failed)
        assert failed.outputs[0].root_statuses[0] != 0
        assert failed.outputs[1].root_statuses[0] != 0
        assert failed.outputs[2].root_statuses[0] == 0
        assert failed.outputs[2].values[0] == np.sum(bad["w"])
        recovered = scheduler.execute(g, data, starts, ends)
        exact(retained, recovered)


def test_plan_covers_storage_crossing_for_many_components_without_execution():
    # Here S/K exceeds the N*K cutoff length. Virtual readonly views exercise
    # geometry without allocating the hundreds of MB required for numerical EM.
    n, d, k = 196, 32, 131072
    data = {
        "x": np.broadcast_to(np.array(0.0), (n, d)),
        "w": np.broadcast_to(np.array(1 / k), (k,)),
        "m": np.broadcast_to(np.array(0.0), (k, d)),
        "v": np.broadcast_to(np.array(1.0), (k, d)),
    }
    storage = k * (128 + 2 * d + 3) + 128 * d
    need = op.get("gaussian_mixture_em").requirements(
        data["x"], data["w"], data["m"], data["v"], 1, 0.0
    )
    assert need["scratch_doubles"] == (storage if enabled() else n * k + 3 * k)
    last_original = max((CUTOFF - 1) // k, storage // k)
    assert last_original == n - 1
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph(data), data, np.zeros(1, np.int64), np.array([n], np.int64))
        assert plan.estimated_worker_scratch_bytes >= (last_original * k + 3 * k) * 8
