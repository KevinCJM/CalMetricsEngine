"""Controlled complete-chain test reference; never a production numerical path."""

import numpy as np
from numba import njit

from calmetrics_engine import GraphCompiler

from . import latent_reference as ref


@njit(cache=True, nogil=True)
def chain_reference(raw, training_positions, classify_positions, k, iterations, seed, hmm):
    # Service takes/copies both row subsets, standardizes the complete original
    # matrix, and retains only the OOS posterior in realtime mode.
    training_raw = raw[training_positions]
    training, mean, scale = ref.standardize_fit_kernel(training_raw)
    standardized = ref.apply_standardization_kernel(raw, mean, scale)
    classify = standardized[classify_positions]
    initialized = ref.initialize_gaussian_means_kernel(training, k, 1, seed, np.empty((0, 0)))
    gmm_iterations = min(iterations, 30) if hmm else iterations
    warm_weights, warm_means, warm_variances, warm_completed = ref.gmm_fit_kernel(
        training, k, gmm_iterations, initialized
    )
    if hmm:
        # Reuse the identical explicit pretraining instead of calling hmm_fit
        # which would perform that same fit again. Initial/transition formulas
        # and ascending sums match the frozen service kernel.
        initial = np.empty(k)
        initial_total = 0.0
        for c in range(k):
            initial[c] = max(warm_weights[c], 1e-8)
            initial_total += initial[c]
        initial /= initial_total
        transition = np.empty((k, k))
        for s in range(k):
            row_total = 0.0
            for t in range(k):
                value = 0.9 if s == t and k > 1 else (1.0 if k == 1 else 0.1 / (k - 1))
                transition[s, t] = value
                row_total += value
            for t in range(k):
                transition[s, t] /= row_total
        initial, transition, means, variances, completed, likelihood = ref.hmm_em_reference(
            training, initial, transition, warm_means, warm_variances, iterations, 1e-5
        )
        posterior = ref.hmm_filtered_posterior_kernel(
            training, classify, initial, transition, means, variances
        )
        model_weights = initial
        model_transition = transition
    else:
        model_weights, means, variances = warm_weights, warm_means, warm_variances
        model_transition = np.empty((0, 0))
        completed = warm_completed
        posterior = ref.gmm_posterior_kernel(classify, model_weights, means, variances)
    assignments, _ = ref.posterior_assignment_kernel(posterior)
    order = np.argsort(means[:, 0]).astype(np.int64)
    mapping = np.empty(k, np.int64)
    for rank in range(k):
        mapping[order[rank]] = k - 1 - rank
    states = np.full(raw.shape[0], -1, np.int64)
    confidence = np.full(raw.shape[0], np.nan)
    probabilities = np.zeros((raw.shape[0], k))
    for i in range(assignments.size):
        target = classify_positions[i]
        states[target] = mapping[assignments[i]]
        winner = 0.0
        for c in range(k):
            probabilities[target, mapping[c]] += posterior[i, c]
            if posterior[i, c] > winner:
                winner = posterior[i, c]
        confidence[target] = winner
    score = raw[:, 0].copy()
    return (
        states,
        score,
        confidence,
        probabilities,
        initialized,
        order,
        np.int64(completed),
        np.int64(warm_completed),
        standardized,
        model_weights,
        means,
        variances,
        model_transition,
    )


def inputs(n=20000, d=4, k=3, training_count=None, seed=17):
    count = n * 3 // 4 if training_count is None else training_count
    if not 1 <= k <= 12 or not count >= max(k * 5, k) or count >= n or d < 1:
        raise ValueError("require 1<=k<=12, d>=1, k*5<=training_count<n")
    # Input creation is outside both warmed computations, with a reproducible
    # generated fixture. Both backends consume the exact same readonly buffers.
    rng = np.random.default_rng(7391)
    raw = rng.normal(size=(n, d))
    raw += ((np.arange(n) // 50) % k)[:, None] * 1.7
    positions = np.arange(n, dtype=np.int64)
    training_positions = positions[:count]
    classify_positions = positions[count:]
    valid = np.zeros(n, np.bool_)
    valid[classify_positions] = True
    transition_template = np.full((k, k), 1.0 if k == 1 else 0.1 / (k - 1))
    if k > 1:
        np.fill_diagonal(transition_template, 0.9)
    data = {
        "raw": raw,
        "positions": positions,
        "training_positions": training_positions,
        "classify_positions": classify_positions,
        "valid": valid,
        "uniform": np.full(k, 1.0 / k),
        "reverse_order": np.arange(k - 1, -1, -1, dtype=np.int64),
        "transition_template": transition_template,
    }
    for array in data.values():
        array.flags.writeable = False
    return data


def graph_for(data, k, iterations, seed, model):
    n, d = data["raw"].shape
    nt = data["training_positions"].size

    def vector(axis, size, dtype="float64", semantic="dimensionless"):
        return {
            "kind": "series" if axis == "time" else "vector",
            "axes": [axis],
            "shape": [str(size)],
            "dtype": dtype,
            "semantic_dimension": semantic,
        }

    types = {
        "raw": {"kind": "matrix", "axes": ["time", "factor"], "shape": [str(n), str(d)]},
        "positions": vector("time", n, "int64", "index"),
        "valid": vector("time", n, "bool", "mask"),
        "uniform": vector("component", k),
        "reverse_order": vector("component", k, "int64", "index"),
        "transition_template": {
            "kind": "matrix",
            "axes": ["component", "component"],
            "shape": [str(k), str(k)],
        },
    }
    bindings = {}

    def bind(name, expression):
        bindings["chain_" + name] = expression
        return "chain_" + name

    training_positions = bind("train_indices", f"slice_axis(positions,0,0,{nt},1)")
    classify_positions = bind("classify_indices", f"slice_axis(positions,0,{nt},{n},1)")
    training_raw = bind("training_raw", f"take_axis(raw,{training_positions},0)")
    mean = bind("mean", f"mean_axis({training_raw},0)")
    raw_scale = bind("raw_scale", f"sqrt(variance_axis({training_raw},0,0))")
    scale = bind("scale", f"where({raw_scale}<1e-10,1,{raw_scale})")
    training = bind(
        "training",
        f"({training_raw}-broadcast_like({mean},{training_raw}))/broadcast_like({scale},{training_raw})",
    )
    standardized = bind(
        "standardized", f"(raw-broadcast_like({mean},raw))/broadcast_like({scale},raw)"
    )
    classify = bind("classify", f"take_axis({standardized},{classify_positions},0)")
    sampled = bind(
        "sampled", f"take_axis({training},sample_unique_indices({training},{k},{seed}),0)"
    )
    # Each partition contains exactly one selected row; partition_mean makes
    # the component axis explicit without relabelling a time-axis input.
    identity = f"slice_axis(positions,0,0,{k},1)"
    initialized = bind("initialized", f"partition_mean({sampled},{identity},{k})")
    variances = bind(
        "variances", f"broadcast_like(maximum(variance_axis({training},0,0),1e-4),{initialized})"
    )
    gi = min(iterations, 30) if model == "hmm" else iterations
    gmm = bind(
        "gmm", f"gaussian_mixture_em({training},uniform,{initialized},{variances},{gi},1e-6)"
    )
    if model == "hmm":
        initial = bind("initial_raw", f"maximum(mixture_weights({gmm}),1e-8)")
        initial = bind("initial", f"({initial})/sum_axis({initial},0)")
        transition = bind(
            "transition",
            "transition_template/transpose(broadcast_like(sum_axis(transition_template,1),transition_template))",
        )
        fitted = bind(
            "fit",
            f"gaussian_hmm_em({training},{initial},{transition},mixture_means({gmm}),mixture_variances({gmm}),{iterations},1e-5)",
        )
        means = bind("means", f"hmm_means({fitted})")
        variance = bind("variance", f"hmm_variances({fitted})")
        weights = bind("weights", f"hmm_initial({fitted})")
        matrix = bind("matrix", f"hmm_transition({fitted})")
        train_filter = bind(
            "train_filter",
            f"hmm_filter(gaussian_log_density({training},{means},{variance}),{weights},{matrix})",
        )
        continued = bind("continued", f"sum_axis(slice_axis({train_filter},0,{nt-1},{nt},1),0)")
        posterior = bind(
            "posterior",
            f"hmm_filter(gaussian_log_density({classify},{means},{variance}),{continued},{matrix},1)",
        )
    else:
        fitted = gmm
        means = bind("means", f"mixture_means({fitted})")
        variance = bind("variance", f"mixture_variances({fitted})")
        weights = bind("weights", f"mixture_weights({fitted})")
        matrix = "slice_axis(slice_axis(transition_template,0,0,0,1),1,0,0,1)"
        density = bind("density", f"gaussian_log_density({classify},{means},{variance})")
        posterior = bind(
            "posterior",
            f"softmax_axis({density}+broadcast_like(log(maximum({weights},1e-12)),{density}),1)",
        )
    order = bind("order", f"argsort(max_axis(slice_axis({means},1,0,1,1),1))")
    reordered = bind("reordered", f"take_axis({posterior},gather({order},reverse_order),1)")
    probabilities = bind("probabilities", f"scatter_rows({reordered},{classify_positions},{n},0)")
    # Decide the winner before reordering: service ties retain the first
    # original component, which need not be the first mapped state.
    winner_hot = bind("winner_hot", f"one_hot(argmax_axis({posterior},1),{k})")
    reordered_hot = bind(
        "reordered_hot", f"take_axis({winner_hot},gather({order},reverse_order),1)"
    )
    full_hot = bind("full_hot", f"scatter_rows({reordered_hot},{classify_positions},{n},0)")
    code = bind("code", f"argmax_axis({full_hot},1)")
    # argmax is an index, not a nominal state. Explicit scalar decisions retain
    # that distinction and map each supported code through canonical state_select.
    state = "-1"
    for s in range(k):
        state = f"state_select({code}=={s},{s},{state},valid)"
    roots = {
        "state": state,
        "score": "max_axis(slice_axis(raw,1,0,1,1),1)",
        "confidence": f"scatter_rows(max_axis({posterior},1),{classify_positions},{n})",
        "probabilities": probabilities,
        "initial_means": initialized,
        "component_order": order,
        "iterations": f"em_iterations({fitted})",
        "pretrain_iterations": f"em_iterations({gmm})",
        "standardized": standardized,
        "model_weights": weights,
        "model_means": means,
        "model_variances": variance,
        "model_transition": matrix,
    }
    return GraphCompiler(types).compile(
        roots, root_bindings=[bindings] * len(roots), result_format="typed"
    ), list(roots)


def compare(result, expected, names):
    details = {}
    for output, target, name in zip(result.outputs, expected, names, strict=True):
        np.testing.assert_array_equal(output.root_statuses, [0])
        actual = output.values[0]
        target = np.asarray(target)
        if actual.shape != target.shape or actual.dtype != target.dtype:
            raise AssertionError(
                f"{name}: shape/dtype {actual.shape}/{actual.dtype} != {target.shape}/{target.dtype}"
            )
        if target.dtype.kind in "biu":
            np.testing.assert_array_equal(actual, target, err_msg=name)
        else:
            np.testing.assert_allclose(
                actual, target, rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=name
            )
        finite = (
            np.isfinite(target) if target.dtype.kind == "f" else np.ones(target.shape, np.bool_)
        )
        delta = np.abs(actual[finite] - target[finite]) if target.dtype.kind == "f" else np.zeros(1)
        details[name] = {
            "shape": list(target.shape),
            "dtype": str(target.dtype),
            "max_absolute_error": float(delta.max(initial=0)),
        }
    return details
