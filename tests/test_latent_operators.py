"""Independent invariants, frozen-platform parity and full native DAG lifecycle."""

import gc
import itertools

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, excel
from calmetrics_engine import operators as op

from . import latent_reference as ref
from .test_typed_results import config


def fixture(n=37, d=3, k=3, seed=91):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) + (np.arange(n) % k)[:, None] * 2.0
    means = x[np.arange(k)].copy()
    weights = np.full(k, 1.0 / k)
    variances = np.tile(np.maximum(op.variance_axis(x, 0), 1e-4), (k, 1))
    transition = np.full((k, k), 0.1 / max(k - 1, 1))
    np.fill_diagonal(transition, 0.9 if k > 1 else 1.0)
    transition /= transition.sum(axis=1, keepdims=True)
    return x, weights, transition, means, variances


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 5), (1, 3), (0, 3), (2, 0, 3)])
@pytest.mark.parametrize("layout", ["c", "f", "reversed"])
def test_axis_math(shape, layout):
    x = np.random.default_rng(8).normal(size=shape)
    if layout == "f":
        x = np.asfortranarray(x)
    if layout == "reversed":
        x = x[::-1]
    for axis in range(x.ndim):
        if not shape[axis] and np.prod(shape[:axis] + shape[axis + 1 :]):
            for name in ("variance_axis", "max_axis", "argmax_axis", "logsumexp_axis"):
                with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
                    getattr(op, name)(x, axis)
            continue
        np.testing.assert_allclose(
            op.variance_axis(x, axis), np.var(x, axis=axis), atol=1e-15
        )
        if not x.size:
            continue
        np.testing.assert_array_equal(op.argmax_axis(x, axis), np.argmax(x, axis=axis))
        np.testing.assert_array_equal(op.max_axis(x, axis), np.max(x, axis=axis))
        m = np.max(x, axis=axis, keepdims=True)
        expected = np.exp(x - m) / np.exp(x - m).sum(axis=axis, keepdims=True)
        np.testing.assert_allclose(op.softmax_axis(x, axis), expected, atol=1e-15)
        np.testing.assert_allclose(
            op.logsumexp_axis(x, axis),
            np.squeeze(m, axis) + np.log(np.exp(x - m).sum(axis=axis)),
            atol=1e-15,
        )


@pytest.mark.parametrize("n,d,k", [(1, 1, 1), (17, 2, 2), (37, 3, 3), (91, 5, 4)])
@pytest.mark.parametrize("iterations", [1, 7, 40])
def test_frozen_platform_em_and_inference(n, d, k, iterations):
    x, weights, transition, means, variances = fixture(n, d, k)
    expected = ref.gmm_fit_kernel(x, k, iterations, means)
    model = op.gaussian_mixture_em(x, weights, means, variances, iterations)
    for actual, target in zip(
        (
            op.mixture_weights(model),
            op.mixture_means(model),
            op.mixture_variances(model),
        ),
        expected[:3],
        strict=True,
    ):
        np.testing.assert_allclose(actual, target, rtol=3e-10, atol=3e-10)
    assert op.em_iterations(model) == expected[3]
    log_density = op.gaussian_log_density(
        x, op.mixture_means(model), op.mixture_variances(model)
    )
    posterior = op.softmax_axis(
        log_density + np.log(np.maximum(op.mixture_weights(model), 1e-12)), 1
    )
    np.testing.assert_allclose(
        posterior, ref.gmm_posterior_kernel(x, *expected[:3]), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_array_equal(
        op.argmax_axis(posterior, 1), ref.posterior_assignment_kernel(posterior)[0]
    )
    warm = ref.gmm_fit_kernel(x, k, min(iterations, 30), means)
    initial = np.maximum(warm[0], 1e-8)
    initial /= initial.sum()
    hmm = op.gaussian_hmm_em(x, initial, transition, warm[1], warm[2], iterations)
    expected_hmm = ref.hmm_fit_kernel(x, k, iterations, means)
    parameters = (
        op.hmm_initial(hmm),
        op.hmm_transition(hmm),
        op.hmm_means(hmm),
        op.hmm_variances(hmm),
    )
    for actual, target in zip(parameters, expected_hmm[:4], strict=True):
        np.testing.assert_allclose(actual, target, rtol=1e-8, atol=1e-8)
    assert op.em_iterations(hmm) == expected_hmm[4]
    em = op.gaussian_log_density(x, parameters[2], parameters[3])
    smooth = op.hmm_smooth(em, *parameters[:2])
    np.testing.assert_allclose(
        smooth, ref.hmm_smoothed_posterior_kernel(x, *parameters), rtol=1e-9, atol=1e-9
    )
    cut = n // 2
    training = op.hmm_filter(em[:cut], *parameters[:2])
    continued_initial = training[-1] if cut else parameters[0]
    filtered = op.hmm_filter(em[cut:], continued_initial, parameters[1], 1)
    np.testing.assert_allclose(
        filtered,
        ref.hmm_filtered_posterior_kernel(x[:cut], x[cut:], *parameters),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(smooth.sum(axis=1), 1, atol=1e-12)
    # Projections retain their own owner, and are readonly views.
    assert np.shares_memory(hmm, parameters[2]) and not parameters[2].flags.writeable
    del hmm
    gc.collect()
    assert np.isfinite(parameters[2]).all()


def test_hmm_independent_exhaustive_paths_and_causality():
    emissions = np.log(np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5], [0.1, 0.9]]))
    initial, transition = np.array([0.6, 0.4]), np.array([[0.8, 0.2], [0.3, 0.7]])
    expected = np.zeros_like(emissions)
    for path in itertools.product(range(2), repeat=4):
        mass = initial[path[0]] * np.exp(emissions[0, path[0]])
        for i in range(1, 4):
            mass *= transition[path[i - 1], path[i]] * np.exp(emissions[i, path[i]])
        for i, state in enumerate(path):
            expected[i, state] += mass
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        op.hmm_smooth(emissions, initial, transition), expected, atol=1e-14
    )
    whole = op.hmm_filter(emissions, initial, transition)
    np.testing.assert_array_equal(
        whole[:2], op.hmm_filter(emissions[:2], initial, transition)
    )


@pytest.mark.parametrize(
    "n,k,seed",
    [(1, 1, 0), (9, 9, 1), (37, 5, -3), (100, 17, 2**53 - 1), (100, 17, -(2**53) + 1)],
)
def test_initializers_and_scatter(n, k, seed):
    x = np.arange(n * 2, dtype=float).reshape(n, 2)
    order = np.argsort(x[:, 0]).astype(np.int64)
    means = op.partition_mean(x, order, k)
    expected = np.array(
        [x[order[n * c // k : n * (c + 1) // k]].mean(axis=0) for c in range(k)]
    )
    np.testing.assert_array_equal(means, expected)
    sampled = op.sample_unique_indices(x, k, seed)
    assert sampled.dtype == np.int64 and len(set(sampled)) == k
    # Frozen source random strategy = 1.
    target = ref.initialize_gaussian_means_kernel(x, k, 1, seed, np.empty((0, 0)))
    np.testing.assert_array_equal(x[sampled], target)
    selected = x[sampled]
    scattered = op.scatter_rows(selected, sampled, n)
    np.testing.assert_array_equal(scattered[sampled], selected)
    assert np.isnan(scattered[np.setdiff1d(np.arange(n), sampled)]).all()


def test_transition_missing_does_not_bridge():
    codes = np.array([0, 1, -1, 0, 0, 1, 1], np.int64)
    np.testing.assert_array_equal(op.transition_counts(codes, 2), [[1, 2], [0, 1]])
    np.testing.assert_array_equal(
        op.one_hot(codes, 2), [[1, 0], [0, 1], [0, 0], [1, 0], [1, 0], [0, 1], [0, 1]]
    )
    np.testing.assert_array_equal(
        op.argmax_axis(np.ones((3, 2)), 1), np.zeros(3, np.int64)
    )


def graph_fixture():
    types = {
        "x": {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]},
        "m": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "v": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "w": {"kind": "vector", "axes": ["component"], "shape": ["K"]},
    }
    fit = "gaussian_mixture_em(x,w,m,v,7)"
    graph = GraphCompiler(types).compile(
        {
            "means": f"mixture_means({fit})",
            "weights": f"mixture_weights({fit})",
            "iterations": f"em_iterations({fit})",
            "converged": f"em_converged({fit})",
            "posterior": f"softmax_axis(gaussian_log_density(x,mixture_means({fit}),mixture_variances({fit}))+broadcast_like(log(maximum(mixture_weights({fit}),1e-12)),gaussian_log_density(x,mixture_means({fit}),mixture_variances({fit}))),1)",
        },
        result_format="typed",
        error_policy="isolate",
    )
    x, w, _, m, v = fixture()
    return graph, {"x": x, "w": w, "m": m, "v": v}


@pytest.mark.parametrize(
    "lane", ["single", "thread", "process_inline", "process_shared"]
)
def test_model_dag_multioutput_ownership(lane):
    graph, inputs = graph_fixture()
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(
            graph, inputs, np.zeros(2, np.int64), np.zeros(2, np.int64)
        )
        assert result.plan.lane == ("process" if lane.startswith("process") else lane)
        assert result.audit["python_operator_calls"] == 0
        if lane == "process_shared":
            assert result.audit["output_copy_bytes"] == 0
        for output in result.outputs:
            np.testing.assert_array_equal(output.root_statuses, [0, 0])
        means, weights, iterations, converged, posterior = result.outputs
        for row in range(2):
            assert means.values[row].shape == (3, 3)
            assert iterations.values[row].dtype == np.int64
            assert converged.values[row].dtype == np.bool_
            np.testing.assert_allclose(posterior.values[row].sum(axis=1), 1)
        saved = means.values[0].copy()
        inputs["x"][:] = 99
    np.testing.assert_array_equal(means.values[0], saved)
    assert not np.shares_memory(means.values[0], means.values[1])


@pytest.mark.parametrize(
    "name,args,code",
    [
        (
            "gaussian_log_density",
            [np.ones((3, 2)), np.ones((2, 2)), np.ones((2, 3))],
            "SHAPE",
        ),
        (
            "gaussian_mixture_em",
            [np.ones((3, 2)), np.ones(3), np.ones((2, 2)), np.ones((2, 2))],
            "SHAPE",
        ),
        (
            "gaussian_hmm_em",
            [
                np.ones((3, 2)),
                np.ones(2),
                np.ones((2, 3)),
                np.ones((2, 2)),
                np.ones((2, 2)),
            ],
            "SHAPE",
        ),
        (
            "gaussian_log_density",
            [np.ones((3, 2)), np.ones((2, 2)), -np.ones((2, 2))],
            "DOMAIN",
        ),
        ("hmm_filter", [np.ones((3, 2)), np.zeros(2), np.eye(2)], "DOMAIN"),
        ("softmax_axis", [np.array([[1.0, np.nan]]), 1], "NON_FINITE"),
        ("variance_axis", [np.ones((3, 2)), 0, 3], "INSUFFICIENT"),
        ("transition_counts", [np.array([-2, 0], np.int64), 2], "INDEX"),
        ("one_hot", [np.array([0, 2], np.int64), 2], "INDEX"),
        (
            "partition_mean",
            [np.ones((2, 2)), np.array([0, 0], np.int64), 1],
            "DUPLICATE",
        ),
        ("scatter_rows", [np.ones((2, 2)), np.array([0, 0], np.int64), 3], "DUPLICATE"),
        ("sample_unique_indices", [np.ones((2, 2)), 3], "SHAPE"),
    ],
)
def test_invalid_inputs(name, args, code):
    with pytest.raises((ValueError, RuntimeError), match=code):
        getattr(op, name)(*args)


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("layout", ["c", "f", "negative"])
@pytest.mark.parametrize("tolerance", [0.0, 1e10])
def test_explicit_initialization_all_strides_and_stop(hmm, layout, tolerance):
    x, w, a, m, v = fixture(23, 2, 2)
    if layout == "f":
        x, m, v, a = map(np.asfortranarray, (x, m, v, a))
    elif layout == "negative":
        x = x[::-1, ::-1]
        m = m[:, ::-1]
        v = v[:, ::-1]
        w = w[::-1]
        a = a[::-1, ::-1]
    args = (x, w, a, m, v, 7, tolerance) if hmm else (x, w, m, v, 7, tolerance)
    reference = (ref.hmm_em_reference if hmm else ref.gmm_em_reference)(*args)
    model = (op.gaussian_hmm_em if hmm else op.gaussian_mixture_em)(*args)
    projections = (
        (op.hmm_initial, op.hmm_transition, op.hmm_means, op.hmm_variances)
        if hmm
        else (op.mixture_weights, op.mixture_means, op.mixture_variances)
    )
    for project, target in zip(projections, reference[:-2], strict=True):
        np.testing.assert_allclose(project(model), target, rtol=1e-10, atol=1e-10)
    assert op.em_iterations(model) == reference[-2]
    assert bool(op.em_converged(model)) == (tolerance > 0)
    np.testing.assert_allclose(op.em_log_likelihood(model), reference[-1], rtol=1e-12)


def test_nominal_models_and_literal_layout_rejected_at_compile_time():
    types = {
        "x": {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]},
        "axis": "scalar",
    }
    for expression, code in [
        ("mixture_means(x)", "TYPE_MISMATCH"),
        ("variance_axis(x,axis)", "CONSTANT_LAYOUT_REQUIRED"),
    ]:
        with pytest.raises(ValueError, match=code):
            GraphCompiler(types).compile(expression)


@pytest.mark.parametrize(
    "lane", ["single", "thread", "process_inline", "process_shared"]
)
def test_failure_isolation_and_independent_index_validation(lane):
    types = {
        "x": {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]},
        "i": {"kind": "vector", "dtype": "int64", "axes": ["scenario"], "shape": ["N"]},
    }
    graph = GraphCompiler(types).compile(
        ["scatter_rows(log(x),i,5)", "sum_axis(x,0)"], error_policy="isolate"
    )
    x = -np.ones((3, 2))
    good = np.array([0, 2, 4], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(
            graph, {"x": x, "i": good}, np.zeros(2, np.int64), np.zeros(2, np.int64)
        )
        assert np.all(result.outputs[0].root_statuses != 0)
        assert np.all(result.outputs[1].root_statuses == 0)
        for bad in [np.array([0, 2, 5], np.int64), np.array([0, 2, 2], np.int64)]:
            with pytest.raises(
                (ValueError, RuntimeError), match="INDEX_OUT_OF_BOUNDS|DUPLICATE_INDEX"
            ):
                scheduler.execute(
                    graph,
                    {"x": x, "i": bad},
                    np.zeros(2, np.int64),
                    np.zeros(2, np.int64),
                )


def test_excel_symbolic_limit_and_every_recipe():
    from .excel_cases import latent_cases

    for _name, make in latent_cases():
        plan = make()
        meta = plan.metadata()
        assert meta["cells"] > 0
        assert meta["cells"] <= meta["upper_bound"]["cells"]
    x, w, a, m, v = fixture(1000, 5, 4)
    with pytest.raises(ValueError, match="OVER_BUDGET"):
        excel.plan_operator("gaussian_hmm_em", [x, w, a, m, v, 10000])


def test_markov_is_composition_not_an_extra_training_kernel():
    x, w, _, m, v = fixture(53, 3, 2)
    types = {
        "x": {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]},
        "m": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["2", "D"]},
        "v": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["2", "D"]},
        "w": {"kind": "vector", "axes": ["component"], "shape": ["2"]},
    }
    model = "gaussian_mixture_em(x,w,m,v,12)"
    density = (
        f"gaussian_log_density(x,mixture_means({model}),mixture_variances({model}))"
    )
    posterior = f"softmax_axis({density}+broadcast_like(log(maximum(mixture_weights({model}),1e-12)),{density}),1)"
    codes = f"argmax_axis({posterior},1)"
    counts = f"transition_counts({codes},2)+0.001"
    transition = f"({counts})/transpose(broadcast_like(sum_axis({counts},1),{counts}))"
    initial = f"sum_axis(slice_axis(one_hot({codes},2),0,0,1,1),0)+0.001"
    initial = f"({initial})/sum_axis({initial},0)"
    graph = GraphCompiler(types).compile(
        [
            initial,
            transition,
            f"mixture_means({model})",
            f"mixture_variances({model})",
            f"em_iterations({model})",
        ]
    )
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            graph,
            {"x": x, "w": w, "m": m, "v": v},
            np.zeros(1, np.int64),
            np.zeros(1, np.int64),
        )
    expected = ref.markov_fit_kernel(x, 2, 12, m)
    for actual, target in zip(result.outputs, expected, strict=True):
        np.testing.assert_allclose(actual.values[0], target, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("name", ["softmax_axis", "logsumexp_axis"])
def test_excel_refuses_overflowing_shift_before_ready(name):
    with pytest.raises(ValueError, match="UNREPRESENTABLE_VALUE"):
        excel.plan_operator(name, [np.array([[-1e308, 1e308]]), 1])


def test_axis_permuted_negative_inner_strides():
    x = np.arange(120.0).reshape(4, 5, 6).transpose(2, 0, 1)[::-2, :, ::-1]
    for axis in range(3):
        np.testing.assert_array_equal(op.max_axis(x, axis), np.max(x, axis=axis))
        np.testing.assert_array_equal(op.argmax_axis(x, axis), np.argmax(x, axis=axis))
        np.testing.assert_allclose(
            op.variance_axis(x, axis, 1), np.var(x, axis=axis, ddof=1)
        )
        shifted = x - np.max(x, axis=axis, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=axis, keepdims=True)
        np.testing.assert_allclose(op.softmax_axis(x, axis), expected)


@pytest.mark.parametrize("layout", ["c", "f", "negative"])
def test_scatter_tensor_and_strided_codes(layout):
    x = np.arange(60.0).reshape(5, 3, 4)
    if layout == "f":
        x = np.asfortranarray(x)
    elif layout == "negative":
        x = x[::-1, :, ::-1]
    indices = np.arange(0, 10, 2, dtype=np.int64)[::-1]
    result = op.scatter_rows(x, indices, 10)
    np.testing.assert_array_equal(result[indices], x)
    assert np.isnan(result[1::2]).all()
    codes = np.array([0, -1, 2, 2, 1, 0], np.int64)[::-1]
    expected = np.zeros((3, 3))
    for first, second in zip(codes[:-1], codes[1:], strict=True):
        if first >= 0 and second >= 0:
            expected[first, second] += 1
    np.testing.assert_array_equal(op.transition_counts(codes, 3), expected)
    np.testing.assert_array_equal(op.one_hot(codes, 3), codes[:, None] == np.arange(3))


def test_sampling_empty_template_and_excel_integer_boundary():
    empty = np.empty((0, 2))
    assert op.sample_unique_indices(empty, 0).shape == (0,)
    plan = excel.plan_operator("sample_unique_indices", [empty, 0])
    assert plan.metadata()["cells"] <= plan.metadata()["upper_bound"]["cells"]
    with pytest.raises(ValueError, match="UNREPRESENTABLE_VALUE.*integer scalar"):
        excel.plan_operator("sample_unique_indices", [np.zeros((2, 2)), 1, 2**53 - 1])
    # The explicit float64 seed contract supports exactly represented integers.
    plan = excel.plan_operator(
        "sample_unique_indices", [np.zeros((2, 2)), 1, float(2**53 - 1)]
    )
    assert plan.metadata()["cells"] <= plan.metadata()["upper_bound"]["cells"]


def test_prepared_model_refits_changed_input_without_overwriting_snapshot():
    graph, inputs = graph_fixture()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(
            graph, inputs, np.zeros(1, np.int64), np.zeros(1, np.int64)
        )
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        inputs["x"] += 7
        second = prepared.run_snapshot()
        assert not np.allclose(second.outputs[0].values[0], saved)
        np.testing.assert_array_equal(first.outputs[0].values[0], saved)
        assert not np.shares_memory(
            first.outputs[0].values[0], second.outputs[0].values[0]
        )


@pytest.mark.parametrize("hmm", [False, True])
def test_planner_covers_latent_nk_workspace_when_k_exceeds_features(hmm):
    n, d, k = 5000, 1, 30
    x, w, a, m, v = fixture(n, d, k)
    types = {
        "x": {"kind": "matrix", "axes": ["scenario", "factor"], "shape": ["N", "D"]},
        "w": {"kind": "vector", "axes": ["component"], "shape": ["K"]},
        "m": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
        "v": {"kind": "matrix", "axes": ["component", "factor"], "shape": ["K", "D"]},
    }
    inputs = {"x": x, "w": w, "m": m, "v": v}
    expression = "mixture_weights(gaussian_mixture_em(x,w,m,v,1))"
    expected = (n * k + 3 * k) * 8
    if hmm:
        types["a"] = {
            "kind": "matrix",
            "axes": ["component", "component"],
            "shape": ["K", "K"],
        }
        inputs["a"] = a
        expression = "hmm_initial(gaussian_hmm_em(x,w,a,m,v,1))"
        expected = (2 * n * k + 2 * k * k + 4 * k + 512 + 128 * k) * 8
    graph = GraphCompiler(types).compile(expression)
    starts = ends = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        assert plan.estimated_worker_scratch_bytes >= expected
        with pytest.raises(
            (ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc|bad allocation"
        ):
            scheduler.execute(
                graph, inputs, starts, ends, memory_budget_bytes=expected - 1
            )
