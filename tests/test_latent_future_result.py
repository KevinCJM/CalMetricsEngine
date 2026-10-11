"""Future-result scratch ownership and eligibility; canonical fit is unchanged."""
import hashlib

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig


def inputs(n=350000, order="C"):
    rng = np.random.default_rng(714)
    x = rng.normal(size=(n, 4))
    if order == "F":
        x = np.asfortranarray(x)
    elif order == "reverse":
        x = x[::-1, ::-1].copy()[::-1, ::-1]
    data = dict(x=x, weights=np.full(3, 1 / 3), means=rng.normal(size=(3, 4)),
                variances=np.ones((3, 4)), positions=np.arange(n, dtype=np.int64))
    for value in data.values():
        value.flags.writeable = False
    return data


def graph(data, mode="future", policy="raise"):
    n = len(data["x"])
    types = {
        "x": {"kind": "matrix", "shape": [str(n), "4"], "axes": ["time", "factor"]},
        "weights": {"kind": "vector", "shape": ["3"], "axes": ["component"]},
        "means": {"kind": "matrix", "shape": ["3", "4"], "axes": ["component", "factor"]},
        "variances": {"kind": "matrix", "shape": ["3", "4"], "axes": ["component", "factor"]},
        "positions": {"kind": "series", "dtype": "int64", "semantic_dimension": "index",
                      "shape": [str(n)], "axes": ["time"]},
    }
    fit = "gaussian_mixture_em(x,weights,means,variances,2,0)"
    bindings = {"trained": fit}
    roots = {}
    if mode == "past":
        roots["earlier"] = f"scatter_rows(x,positions,{n},0)"
    roots.update(means="mixture_means(trained)", variances="mixture_variances(trained)",
                 weights="mixture_weights(trained)", iterations="em_iterations(trained)",
                 converged="em_converged(trained)", likelihood="em_log_likelihood(trained)")
    if mode == "two_fits":
        bindings["other"] = "gaussian_mixture_em(x,weights,means+0.2,variances,2,0)"
        roots["other_means"] = "mixture_means(other)"
        roots["other_iterations"] = "em_iterations(other)"
    density = "gaussian_log_density(x,mixture_means(trained),mixture_variances(trained))"
    posterior = f"softmax_axis({density}+broadcast_like(log(maximum(mixture_weights(trained),1e-12)),{density}),1)"
    if mode in ("future", "two_fits"):
        roots["probabilities"] = f"scatter_rows({posterior},positions,{n},0)"
    elif mode == "small_future":
        roots["small"] = f"scatter_rows(mean_axis(x,1),positions,{n},0)"
    return GraphCompiler(types).compile(roots, root_bindings=[bindings] * len(roots),
                                       result_format="typed", error_policy=policy)


def selected(data, g):
    return {key: data[key] for key in g.input_names}


def hashes(data):
    return {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in data.items()}


def exact(a, b):
    assert len(a.outputs) == len(b.outputs)
    for x, y in zip(a.outputs, b.outputs, strict=True):
        np.testing.assert_array_equal(x.root_statuses, y.root_statuses)
        for u, v in zip(x.values, y.values, strict=True):
            assert u.shape == v.shape and u.dtype == v.dtype
            np.testing.assert_array_equal(u.reshape(-1).view(np.uint8),
                                          v.reshape(-1).view(np.uint8))


def exercise(data, g, lane="single", loan=True):
    config = PlannerConfig(thread_work_units=1e100 if lane == "single" else 1,
                           process_work_units=1 if lane.startswith("process") else 1e100,
                           shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
                           min_rows_per_worker=1, max_processes=2)
    rows = 1 if lane == "single" else 2
    starts, ends = np.zeros(rows, np.int64), np.full(rows, len(data["x"]), np.int64)
    before = hashes(data)
    x = selected(data, g)
    # Prepared execution has a single-lane contract. Its independent snapshot
    # is the unchanged canonical reference for the forced transport lanes.
    with AdaptiveScheduler(cpu_budget=1) as canonical_scheduler:
        canonical = canonical_scheduler.prepare_execution(g, x, starts, ends).run_snapshot()
    with AdaptiveScheduler(cpu_budget=rows, config=config) as scheduler:
        first = scheduler.execute(g, x, starts, ends)
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        retained = [hashes({str(i): v for i, v in enumerate(o.values)}) for o in first.outputs]
        second = scheduler.execute(g, x, starts, ends)
        exact(first, canonical)
        exact(second, canonical)
        for old, new in zip(first.outputs, second.outputs, strict=True):
            for a, b in zip(old.values, new.values, strict=True):
                assert not np.shares_memory(a, b)
                assert not a.flags.writeable
                assert all(not np.shares_memory(a, value) for value in data.values())
        assert retained == [hashes({str(i): v for i, v in enumerate(o.values)}) for o in first.outputs]
        current = [c["operator_workspace_capacity_bytes"] for c in first.audit["native_chunks"]]
        control = max(c["operator_workspace_capacity_bytes"] for c in canonical.audit["native_chunks"])
        if loan:
            assert max(current) < control - 8 * 1024 * 1024 + 128 * 1024
        else:
            assert max(current) == control
    exact(first, canonical)
    assert before == hashes(data)
    assert all(not value.flags.writeable for value in data.values())


@pytest.mark.parametrize("order", ["C", "F", "reverse"])
def test_future_result_exact_fit_and_independent_old_outputs(order):
    data = inputs(order=order)
    exercise(data, graph(data))


@pytest.mark.parametrize("n,loan", [(349525, False), (349526, True)])
def test_future_result_nk_threshold(n, loan):
    data = inputs(n)
    exercise(data, graph(data), loan=loan)


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
def test_future_result_two_rows_transport_and_lifetime(lane):
    data = inputs()
    exercise(data, graph(data), lane=lane)


@pytest.mark.parametrize("mode", ["past", "small_future"])
def test_written_or_insufficient_result_is_not_borrowed(mode):
    data = inputs()
    exercise(data, graph(data, mode), loan=False)


def test_two_distinct_fits_reuse_only_unpublished_future_storage():
    data = inputs()
    exercise(data, graph(data, "two_fits"))


def test_isolation_keeps_canonical_workspace():
    data = inputs()
    exercise(data, graph(data, policy="isolate"), loan=False)


def test_zero_variance_preserves_existing_regularization():
    data = inputs()
    data["variances"] = np.zeros((3, 4))
    data["variances"].flags.writeable = False
    exercise(data, graph(data))


@pytest.mark.parametrize("fault", [np.nan, np.inf, -1.0])
def test_future_result_error_parity_recovery_and_retained_results(fault):
    data = inputs()
    g = graph(data)
    starts, ends = np.array([0], np.int64), np.array([len(data["x"])], np.int64)
    good = selected(data, g)
    bad = dict(good, variances=np.full((3, 4), fault))
    bad["variances"].flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        old = scheduler.execute(g, good, starts, ends)
        errors = []
        for prepared in (False, True):
            with pytest.raises(Exception) as error:
                if prepared:
                    scheduler.prepare_execution(g, bad, starts, ends).run_snapshot()
                else:
                    scheduler.execute(g, bad, starts, ends)
            errors.append((type(error.value), str(error.value)))
        assert errors[0] == errors[1]
        fresh = scheduler.execute(g, good, starts, ends)
        exact(old, fresh)
