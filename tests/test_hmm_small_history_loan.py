"""Measured HMM history ownership below the former two-million-cell cutoff."""
import hashlib

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig


def exact(actual, expected):
    assert len(actual.outputs) == len(expected.outputs)
    for a, b in zip(actual.outputs, expected.outputs, strict=True):
        np.testing.assert_array_equal(a.root_statuses, b.root_statuses)
        for x, y in zip(a.values, b.values, strict=True):
            assert x.shape == y.shape and x.dtype == y.dtype
            np.testing.assert_array_equal(x.reshape(-1).view(np.uint8),
                                          y.reshape(-1).view(np.uint8))


def fixture(n=180000, order="C", policy="raise"):
    rng = np.random.default_rng(817)
    data = dict(x=rng.normal(size=(n, 4)), weights=np.full(3, 1 / 3),
                transition=np.full((3, 3), 1 / 3), means=rng.normal(size=(3, 4)),
                variances=np.ones((3, 4)), positions=np.arange(n, dtype=np.int64))
    axes = dict(x=["time", "factor"], weights=["component"],
                transition=["component", "component"], means=["component", "factor"],
                variances=["component", "factor"], positions=["time"])
    types = {}
    for name, value in data.items():
        if order == "F" and value.ndim == 2:
            value = np.asfortranarray(value)
        elif order == "reverse" and value.ndim == 2:
            value = value[::-1, ::-1].copy()[::-1, ::-1]
        value.flags.writeable = False
        data[name] = value
        types[name] = dict(kind="matrix" if value.ndim == 2 else "series" if name == "positions" else "vector",
                           axes=axes[name], shape=[str(size) for size in value.shape], dtype=str(value.dtype),
                           semantic_dimension="index" if name == "positions" else "dimensionless")
    fit = "gaussian_hmm_em(x,weights,transition,means,variances,2,0)"
    density = "gaussian_log_density(x,hmm_means(trained),hmm_variances(trained))"
    posterior = f"hmm_filter({density},hmm_initial(trained),hmm_transition(trained))"
    roots = dict(means="hmm_means(trained)", variances="hmm_variances(trained)",
                 initial="hmm_initial(trained)", transition="hmm_transition(trained)",
                 iterations="em_iterations(trained)", likelihood="em_log_likelihood(trained)",
                 probabilities=f"scatter_rows({posterior},positions,{n},0)")
    g = GraphCompiler(types).compile(roots, root_bindings=[{"trained": fit}] * len(roots),
                                     result_format="typed", error_policy=policy)
    return {name: data[name] for name in g.input_names}, g


def fingerprints(data):
    return {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in data.items()}


def exercise(n=180000, order="C", policy="raise", lane="single", loan=True):
    data, g = fixture(n, order, policy)
    before = fingerprints(data)
    rows = 1 if lane == "single" else 2
    starts, ends = np.zeros(rows, np.int64), np.full(rows, n, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as reference:
        canonical = reference.prepare_execution(g, data, starts, ends).run_snapshot()
    config = PlannerConfig(thread_work_units=1e100 if lane == "single" else 1,
                           process_work_units=1 if lane.startswith("process") else 1e100,
                           shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
                           min_rows_per_worker=1, max_processes=2)
    with AdaptiveScheduler(cpu_budget=rows, config=config) as scheduler:
        first = scheduler.execute(g, data, starts, ends)
        saved = [fingerprints({str(i): value for i, value in enumerate(out.values)}) for out in first.outputs]
        second = scheduler.execute(g, data, starts, ends)
        exact(first, canonical)
        exact(second, canonical)
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        actual = max(c["operator_workspace_capacity_bytes"] for c in first.audit["native_chunks"])
        control = max(c["operator_workspace_capacity_bytes"] for c in canonical.audit["native_chunks"])
        if loan:
            assert actual < control - 1024 * 1024
        else:
            assert actual == control
        for old, new in zip(first.outputs, second.outputs, strict=True):
            for a, b in zip(old.values, new.values, strict=True):
                assert not a.flags.writeable and not np.shares_memory(a, b)
                assert all(not np.shares_memory(a, item) for item in data.values())
        assert saved == [fingerprints({str(i): value for i, value in enumerate(out.values)}) for out in first.outputs]
    assert before == fingerprints(data)
    exact(first, canonical)


@pytest.mark.parametrize("order", ["C", "F", "reverse"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_small_hmm_history_layout_and_independent_results(order, policy):
    exercise(order=order, policy=policy, loan=policy == "raise")


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
def test_small_hmm_history_process_and_thread_transport(lane):
    exercise(lane=lane)


@pytest.mark.parametrize("n,loan", [(43690, False), (43691, True)])
def test_small_hmm_history_reservation_threshold(n, loan):
    exercise(n=n, loan=loan)


@pytest.mark.parametrize("fault", [np.nan, np.inf, -1.0])
def test_small_hmm_history_error_recovery_and_retained_results(fault):
    data, g = fixture()
    starts, ends = np.array([0], np.int64), np.array([len(data["x"])], np.int64)
    bad = dict(data, variances=np.full((3, 4), fault))
    bad["variances"].flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        old = scheduler.execute(g, data, starts, ends)
        errors = []
        for prepared in (False, True):
            with pytest.raises(Exception) as failure:
                if prepared:
                    scheduler.prepare_execution(g, bad, starts, ends).run_snapshot()
                else:
                    scheduler.execute(g, bad, starts, ends)
            errors.append((type(failure.value), str(failure.value)))
        assert errors[0] == errors[1]
        exact(old, scheduler.execute(g, data, starts, ends))
