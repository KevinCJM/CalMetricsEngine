"""Large fit scratch loan: exact canonical parity, transport and lifetime."""

import hashlib

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from . import latent_chain as bench
from .test_typed_results import config


@pytest.fixture(scope="module")
def fixture():
    data = bench.inputs(240000, 4, 12)
    expected = bench.chain_reference(
        data["raw"], data["training_positions"], data["classify_positions"], 12, 1, 17, True
    )
    return data, expected


def selected(data, g):
    return {key: data[key] for key in g.input_names}


def exact(a, b):
    for x, y in zip(a.outputs, b.outputs, strict=True):
        np.testing.assert_array_equal(x.root_statuses, y.root_statuses)
        for u, v in zip(x.values, y.values, strict=True):
            assert u.dtype == v.dtype and u.shape == v.shape
            np.testing.assert_array_equal(
                u.reshape(-1).view(np.uint8), v.reshape(-1).view(np.uint8)
            )


@pytest.mark.parametrize("order", ["C", "F", "reverse"])
def test_large_fit_exact_canonical_and_owned_results(fixture, order):
    data, expected = fixture
    data = dict(data)
    raw = data["raw"]
    if order == "F":
        raw = np.asfortranarray(raw)
    elif order == "reverse":
        raw = raw[::-1, ::-1].copy()[::-1, ::-1]
    raw.flags.writeable = False
    data["raw"] = raw
    before = hashlib.sha256(raw.tobytes()).hexdigest()
    g, names = bench.graph_for(data, 12, 1, 17, "hmm")
    starts = np.array([0], np.int64)
    ends = np.array([240000], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        x = selected(data, g)
        canonical = s.prepare_execution(g, x, starts, ends).run_snapshot()
        first = s.execute(g, x, starts, ends)
        bench.compare(first, expected, names)
        exact(first, canonical)
        assert (
            first.audit["native_chunks"][0]["operator_workspace_capacity_bytes"]
            < canonical.audit["native_chunks"][0]["operator_workspace_capacity_bytes"]
        )
        second = s.execute(g, x, starts, ends)
        exact(second, canonical)
        exact(first, canonical)
        for a, b in zip(first.outputs, second.outputs, strict=True):
            assert not np.shares_memory(a.values[0], b.values[0])
    assert before == hashlib.sha256(raw.tobytes()).hexdigest()
    assert not raw.flags.writeable


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
def test_large_fit_two_rows_transport_and_lifetime(fixture, lane):
    data, expected = fixture
    g, names = bench.graph_for(data, 12, 1, 17, "hmm")
    starts = np.zeros(2, np.int64)
    ends = np.full(2, 240000, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as s:
        a = s.execute(g, selected(data, g), starts, ends)
        b = s.execute(g, selected(data, g), starts, ends)
        exact(a, b)
        for i, (name, item, target) in enumerate(zip(names, a.outputs, expected, strict=True)):
            np.testing.assert_array_equal(item.root_statuses, [0, 0])
            target = np.asarray(target)
            for row, value in enumerate(item.values):
                if target.dtype.kind in "biu":
                    np.testing.assert_array_equal(value, target, err_msg=name)
                else:
                    np.testing.assert_allclose(
                        value, target, rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=name
                    )
                assert not np.shares_memory(value, b.outputs[i].values[row])
        assert all(
            c["operator_workspace_capacity_bytes"] < 30000000 for c in a.audit["native_chunks"]
        )


def test_large_fit_error_parity_and_recovery(fixture):
    data, expected = fixture
    g, names = bench.graph_for(data, 12, 1, 17, "hmm")
    starts = np.array([0], np.int64)
    ends = np.array([240000], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        good = selected(data, g)
        retained = s.execute(g, good, starts, ends)
        for fault in (np.nan, -1.0, 0.0):
            bad = dict(good)
            bad["uniform"] = np.full(12, fault)
            bad["uniform"].flags.writeable = False
            errors = []
            for prepared in (False, True):
                with pytest.raises(Exception) as e:
                    if prepared:
                        s.prepare_execution(g, bad, starts, ends).run_snapshot()
                    else:
                        s.execute(g, bad, starts, ends)
                errors.append((type(e.value), str(e.value)))
            assert errors[0] == errors[1]
            recovered = s.execute(g, good, starts, ends)
            bench.compare(recovered, expected, names)
            exact(retained, recovered)


@pytest.mark.parametrize("model", ["gmm", "hmm"])
@pytest.mark.parametrize("order", ["C", "F", "reverse"])
def test_wide_fit_library_paths_and_all_projections(model, order):
    n, d, k = 300000, 32, 8
    rng = np.random.default_rng(537)
    data = {
        "x": rng.normal(size=(n, d)),
        "initial": np.full(k, 1 / k),
        "transition": np.full((k, k), 0.15 / (k - 1)),
        "means": rng.normal(scale=0.3, size=(k, d)),
        "variances": rng.uniform(0.5, 1.5, size=(k, d)),
        "positions": np.arange(n, dtype=np.int64),
    }
    np.fill_diagonal(data["transition"], 0.85)
    axes = {
        "x": ["time", "factor"],
        "initial": ["component"],
        "transition": ["component", "component"],
        "means": ["component", "factor"],
        "variances": ["component", "factor"],
        "positions": ["time"],
    }
    definitions = {}
    for name, value in data.items():
        if order == "F" and value.ndim == 2:
            value = np.asfortranarray(value)
        elif order == "reverse":
            reverse = (slice(None, None, -1),) * value.ndim
            value = value[reverse].copy()[reverse]
        value.flags.writeable = False
        data[name] = value
        definitions[name] = {
            "kind": "matrix" if value.ndim == 2 else "series" if name == "positions" else "vector",
            "axes": axes[name],
            "shape": [str(size) for size in value.shape],
            "dtype": str(value.dtype),
            "semantic_dimension": "index" if name == "positions" else "dimensionless",
        }
    prefix = "hmm" if model == "hmm" else "mixture"
    density = f"gaussian_log_density(x,{prefix}_means(trained),{prefix}_variances(trained))"
    if model == "hmm":
        fit = "gaussian_hmm_em(x,initial,transition,means,variances,1,0)"
        posterior = f"hmm_filter({density},hmm_initial(trained),hmm_transition(trained))"
    else:
        fit = "gaussian_mixture_em(x,initial,means,variances,1,0)"
        posterior = f"softmax_axis({density}+broadcast_like(log(maximum(mixture_weights(trained),1e-12)),{density}),1)"
    roots = {
        "means": f"{prefix}_means(trained)",
        "variances": f"{prefix}_variances(trained)",
        "iterations": "em_iterations(trained)",
        "likelihood": "em_log_likelihood(trained)",
        "converged": "em_converged(trained)",
        "probabilities": f"scatter_rows({posterior},positions,{n},0)",
    }
    if model == "hmm":
        roots.update(initial="hmm_initial(trained)", transition="hmm_transition(trained)")
    else:
        roots.update(weights="mixture_weights(trained)")
    g = GraphCompiler(definitions).compile(
        roots, root_bindings=[{"trained": fit}] * len(roots), result_format="typed"
    )
    inputs = selected(data, g)
    before = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in inputs.items()}
    starts, ends = np.array([0], np.int64), np.array([n], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        canonical = scheduler.prepare_execution(g, inputs, starts, ends).run_snapshot()
        first = scheduler.execute(g, inputs, starts, ends)
        second = scheduler.execute(g, inputs, starts, ends)
        exact(canonical, first)
        exact(first, second)
        for old, new in zip(first.outputs, second.outputs, strict=True):
            assert not np.shares_memory(old.values[0], new.values[0])
        actual = first.audit["native_chunks"][0]
        control = canonical.audit["native_chunks"][0]
        assert (
            actual["operator_workspace_capacity_bytes"]
            < control["operator_workspace_capacity_bytes"]
        )
        if model == 'hmm' and actual['eigen_density_calls']:
            # One EM density plus inference. The free third history removes
            # exactly the bounded replay calls, without changing either model.
            assert control['eigen_density_calls']-actual['eigen_density_calls'] == (n-1+127)//128
        else:
            assert actual['eigen_density_calls'] == control['eigen_density_calls']
    assert before == {
        name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in inputs.items()
    }
    assert all(not value.flags.writeable for value in inputs.values())
