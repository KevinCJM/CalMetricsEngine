"""Frozen FIRP latent-model NJIT reference; test/benchmark only.

Source: backend/historical_regimes/numba_kernels.py
Commit: 9b99048d188017816358f581a86b27df1c388161
SHA256: 6be805b1284912f39c79196489f81e03715c195526ecf4c868579077cbc186e3
No production imports or runtime fallback. Function bodies are unchanged;
explicit decorators become lazy NJIT to isolate service startup dependencies.
"""

from __future__ import annotations

import numpy as np

from .reference_backend import njit


@njit(cache=True, nogil=True)
def standardize_fit_kernel(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = matrix.shape
    mean = np.empty(columns, dtype=np.float64)
    scale = np.empty(columns, dtype=np.float64)
    standardized = np.empty_like(matrix)
    for column in range(columns):
        total = 0.0
        for row in range(rows):
            total += matrix[row, column]
        column_mean = total / rows
        mean[column] = column_mean
        squared = 0.0
        for row in range(rows):
            difference = matrix[row, column] - column_mean
            squared += difference * difference
        column_scale = np.sqrt(squared / rows)
        if column_scale < 1e-10:
            column_scale = 1.0
        scale[column] = column_scale
        for row in range(rows):
            standardized[row, column] = (
                matrix[row, column] - column_mean
            ) / column_scale
    return standardized, mean, scale


@njit(cache=True, nogil=True)
def apply_standardization_kernel(
    matrix: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(matrix)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                output[row, column] = (value - mean[column]) / scale[column]
            else:
                output[row, column] = np.nan
    return output


@njit(cache=True, nogil=True)
def _logsumexp_vector(values: np.ndarray) -> float:
    maximum = -np.inf
    for index in range(values.size):
        if values[index] > maximum:
            maximum = values[index]
    if not np.isfinite(maximum):
        return maximum
    total = 0.0
    for index in range(values.size):
        total += np.exp(values[index] - maximum)
    return maximum + np.log(max(total, 1e-300))


@njit(cache=True, nogil=True)
def _gaussian_log_density_impl(
    matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    rows = matrix.shape[0]
    components = means.shape[0]
    dimensions = matrix.shape[1]
    output = np.empty((rows, components), dtype=np.float64)
    constant = dimensions * np.log(2.0 * np.pi)
    for row in range(rows):
        for component in range(components):
            log_variance = 0.0
            quadratic = 0.0
            for dimension in range(dimensions):
                variance = max(variances[component, dimension], 1e-12)
                difference = matrix[row, dimension] - means[component, dimension]
                log_variance += np.log(variance)
                quadratic += difference * difference / variance
            output[row, component] = -0.5 * (constant + log_variance + quadratic)
    return output


@njit(cache=True, nogil=True)
def initialize_gaussian_means_kernel(
    matrix: np.ndarray,
    components: int,
    strategy_code: int,
    random_seed: int,
    explicit_means: np.ndarray,
) -> np.ndarray:
    """Create deterministic model starting means for quantile/random/explicit modes."""

    rows, dimensions = matrix.shape
    if rows < components or components < 1:
        raise ValueError("insufficient rows for component initialization")
    means = np.empty((components, dimensions), dtype=np.float64)
    if strategy_code == 2:
        if (
            explicit_means.shape[0] != components
            or explicit_means.shape[1] != dimensions
        ):
            raise ValueError("explicit means shape mismatch")
        for component in range(components):
            for dimension in range(dimensions):
                value = explicit_means[component, dimension]
                if not np.isfinite(value):
                    raise ValueError("explicit means must be finite")
                means[component, dimension] = value
        return means
    if strategy_code == 1:
        # Park-Miller LCG gives a local, reproducible stream without touching
        # Numba's process-global RNG state. Duplicate draws are advanced to an
        # unused row so every component has a genuinely distinct starting row.
        state = random_seed % 2147483646
        if state < 0:
            state = -state
        state += 1
        selected = np.full(components, -1, dtype=np.int64)
        for component in range(components):
            state = (state * 48271) % 2147483647
            candidate = state % rows
            duplicate = True
            while duplicate:
                duplicate = False
                for prior in range(component):
                    if selected[prior] == candidate:
                        candidate = (candidate + 1) % rows
                        duplicate = True
                        break
            selected[component] = candidate
            for dimension in range(dimensions):
                means[component, dimension] = matrix[candidate, dimension]
        return means
    if strategy_code != 0:
        raise ValueError("unsupported initialization strategy")
    order = np.argsort(matrix[:, 0])
    for component in range(components):
        start = rows * component // components
        end = rows * (component + 1) // components
        count = max(end - start, 1)
        for dimension in range(dimensions):
            total = 0.0
            for offset in range(start, end):
                total += matrix[order[offset], dimension]
            means[component, dimension] = total / count
    return means


@njit(cache=True, nogil=True)
def gmm_fit_kernel(
    matrix: np.ndarray,
    components: int,
    iterations: int,
    initialized_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows, dimensions = matrix.shape
    if (
        initialized_means.shape[0] != components
        or initialized_means.shape[1] != dimensions
    ):
        raise ValueError("initialized means shape mismatch")
    means = initialized_means.copy()
    global_variance = np.empty(dimensions, dtype=np.float64)
    for dimension in range(dimensions):
        total = 0.0
        for row in range(rows):
            total += matrix[row, dimension]
        mean = total / rows
        squared = 0.0
        for row in range(rows):
            difference = matrix[row, dimension] - mean
            squared += difference * difference
        global_variance[dimension] = max(squared / rows, 1e-4)
    variances = np.empty((components, dimensions), dtype=np.float64)
    for component in range(components):
        for dimension in range(dimensions):
            variances[component, dimension] = global_variance[dimension]
    weights = np.full(components, 1.0 / components, dtype=np.float64)
    prior_likelihood = -np.inf
    completed = 0
    for iteration in range(1, iterations + 1):
        completed = iteration
        log_joint = _gaussian_log_density_impl(matrix, means, variances)
        responsibilities = np.empty_like(log_joint)
        likelihood = 0.0
        for row in range(rows):
            for component in range(components):
                log_joint[row, component] += np.log(max(weights[component], 1e-12))
            normalizer = _logsumexp_vector(log_joint[row])
            likelihood += normalizer
            for component in range(components):
                responsibilities[row, component] = np.exp(
                    log_joint[row, component] - normalizer
                )
        effective = np.zeros(components, dtype=np.float64)
        for component in range(components):
            for row in range(rows):
                effective[component] += responsibilities[row, component]
            effective[component] = max(effective[component], 1e-8)
            weights[component] = effective[component] / rows
            for dimension in range(dimensions):
                total = 0.0
                for row in range(rows):
                    total += responsibilities[row, component] * matrix[row, dimension]
                means[component, dimension] = total / effective[component]
        for component in range(components):
            for dimension in range(dimensions):
                total = 0.0
                for row in range(rows):
                    difference = matrix[row, dimension] - means[component, dimension]
                    total += responsibilities[row, component] * difference * difference
                variances[component, dimension] = max(
                    total / effective[component],
                    1e-5,
                )
        if abs(likelihood - prior_likelihood) < 1e-6:
            break
        prior_likelihood = likelihood
    return weights, means, variances, completed


@njit(cache=True, nogil=True)
def gmm_posterior_kernel(
    matrix: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    log_joint = _gaussian_log_density_impl(matrix, means, variances)
    posterior = np.empty_like(log_joint)
    for row in range(matrix.shape[0]):
        for component in range(weights.size):
            log_joint[row, component] += np.log(max(weights[component], 1e-12))
        normalizer = _logsumexp_vector(log_joint[row])
        for component in range(weights.size):
            posterior[row, component] = np.exp(log_joint[row, component] - normalizer)
    return posterior


@njit(cache=True, nogil=True)
def markov_fit_kernel(
    matrix: np.ndarray,
    components: int,
    iterations: int,
    initialized_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit a hard-state Gaussian Markov-switching model.

    Unlike ``hmm_fit_kernel`` this does not run Baum-Welch.  Gaussian regimes
    are estimated first, then the transition matrix is estimated from the
    locked training assignments with Laplace smoothing.  This gives the
    Markov node distinct, auditable estimation semantics.
    """

    weights, means, variances, completed = gmm_fit_kernel(
        matrix,
        components,
        iterations,
        initialized_means,
    )
    posterior = gmm_posterior_kernel(matrix, weights, means, variances)
    assignments = np.empty(matrix.shape[0], dtype=np.int64)
    for row in range(matrix.shape[0]):
        winner = 0
        for component in range(1, components):
            if posterior[row, component] > posterior[row, winner]:
                winner = component
        assignments[row] = winner

    smoothing = 1e-3
    initial = np.full(components, smoothing, dtype=np.float64)
    initial[assignments[0]] += 1.0
    initial /= np.sum(initial)
    transition = np.full((components, components), smoothing, dtype=np.float64)
    for row in range(1, assignments.size):
        transition[assignments[row - 1], assignments[row]] += 1.0
    for source in range(components):
        total = np.sum(transition[source])
        transition[source] /= total
    return initial, transition, means, variances, completed


@njit(cache=True, nogil=True)
def hmm_fit_kernel(
    matrix: np.ndarray,
    components: int,
    iterations: int,
    initialized_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    weights, means, variances, _ = gmm_fit_kernel(
        matrix,
        components,
        min(iterations, 30),
        initialized_means,
    )
    initial = np.empty(components, dtype=np.float64)
    initial_total = 0.0
    for component in range(components):
        initial[component] = max(weights[component], 1e-8)
        initial_total += initial[component]
    initial /= initial_total
    transition = np.empty((components, components), dtype=np.float64)
    for row in range(components):
        row_total = 0.0
        for column in range(components):
            value = (
                0.9
                if row == column and components > 1
                else (1.0 if components == 1 else 0.1 / (components - 1))
            )
            transition[row, column] = value
            row_total += value
        for column in range(components):
            transition[row, column] /= row_total
    prior_likelihood = -np.inf
    completed = 0
    rows = matrix.shape[0]
    dimensions = matrix.shape[1]
    for iteration in range(1, iterations + 1):
        completed = iteration
        emissions = _gaussian_log_density_impl(matrix, means, variances)
        log_transition = np.empty_like(transition)
        for row in range(components):
            for column in range(components):
                log_transition[row, column] = np.log(
                    max(transition[row, column], 1e-12)
                )
        alpha = np.empty_like(emissions)
        for component in range(components):
            alpha[0, component] = (
                np.log(max(initial[component], 1e-12)) + emissions[0, component]
            )
        workspace = np.empty(components, dtype=np.float64)
        for index in range(1, rows):
            for target in range(components):
                for source in range(components):
                    workspace[source] = (
                        alpha[index - 1, source] + log_transition[source, target]
                    )
                alpha[index, target] = emissions[index, target] + _logsumexp_vector(
                    workspace
                )
        beta = np.zeros_like(emissions)
        for index in range(rows - 2, -1, -1):
            for source in range(components):
                for target in range(components):
                    workspace[target] = (
                        log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                    )
                beta[index, source] = _logsumexp_vector(workspace)
        likelihood = _logsumexp_vector(alpha[rows - 1])
        gamma = np.empty_like(emissions)
        for index in range(rows):
            for component in range(components):
                workspace[component] = alpha[index, component] + beta[index, component]
            normalizer = _logsumexp_vector(workspace)
            for component in range(components):
                gamma[index, component] = np.exp(
                    alpha[index, component] + beta[index, component] - normalizer
                )
        xi_sum = np.zeros_like(transition)
        for index in range(rows - 1):
            xi_normalizer = -np.inf
            for source in range(components):
                for target in range(components):
                    value = (
                        alpha[index, source]
                        + log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                    )
                    if xi_normalizer == -np.inf:
                        xi_normalizer = value
                    elif value > xi_normalizer:
                        xi_normalizer = value + np.log1p(np.exp(xi_normalizer - value))
                    else:
                        xi_normalizer = xi_normalizer + np.log1p(
                            np.exp(value - xi_normalizer)
                        )
            for source in range(components):
                for target in range(components):
                    value = (
                        alpha[index, source]
                        + log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                        - xi_normalizer
                    )
                    xi_sum[source, target] += np.exp(value)
        initial_total = 0.0
        for component in range(components):
            initial[component] = max(gamma[0, component], 1e-8)
            initial_total += initial[component]
        initial /= initial_total
        for source in range(components):
            row_total = 0.0
            for target in range(components):
                transition[source, target] = max(
                    xi_sum[source, target],
                    1e-8,
                )
                row_total += transition[source, target]
            for target in range(components):
                transition[source, target] /= row_total
        effective = np.zeros(components, dtype=np.float64)
        for component in range(components):
            for index in range(rows):
                effective[component] += gamma[index, component]
            effective[component] = max(effective[component], 1e-8)
            for dimension in range(dimensions):
                total = 0.0
                for index in range(rows):
                    total += gamma[index, component] * matrix[index, dimension]
                means[component, dimension] = total / effective[component]
        for component in range(components):
            for dimension in range(dimensions):
                total = 0.0
                for index in range(rows):
                    difference = matrix[index, dimension] - means[component, dimension]
                    total += gamma[index, component] * difference * difference
                variances[component, dimension] = max(
                    total / effective[component],
                    1e-5,
                )
        if abs(likelihood - prior_likelihood) < 1e-5:
            break
        prior_likelihood = likelihood
    return initial, transition, means, variances, completed


@njit(cache=True, nogil=True)
def hmm_smoothed_posterior_kernel(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    emissions = _gaussian_log_density_impl(observations, means, variances)
    rows = observations.shape[0]
    components = initial.size
    log_transition = np.empty_like(transition)
    for source in range(components):
        for target in range(components):
            log_transition[source, target] = np.log(
                max(transition[source, target], 1e-12)
            )
    alpha = np.empty_like(emissions)
    beta = np.zeros_like(emissions)
    workspace = np.empty(components, dtype=np.float64)
    for component in range(components):
        alpha[0, component] = (
            np.log(max(initial[component], 1e-12)) + emissions[0, component]
        )
    for index in range(1, rows):
        for target in range(components):
            for source in range(components):
                workspace[source] = (
                    alpha[index - 1, source] + log_transition[source, target]
                )
            alpha[index, target] = emissions[index, target] + _logsumexp_vector(
                workspace
            )
    for index in range(rows - 2, -1, -1):
        for source in range(components):
            for target in range(components):
                workspace[target] = (
                    log_transition[source, target]
                    + emissions[index + 1, target]
                    + beta[index + 1, target]
                )
            beta[index, source] = _logsumexp_vector(workspace)
    posterior = np.empty_like(emissions)
    for index in range(rows):
        for component in range(components):
            workspace[component] = alpha[index, component] + beta[index, component]
        normalizer = _logsumexp_vector(workspace)
        for component in range(components):
            posterior[index, component] = np.exp(
                alpha[index, component] + beta[index, component] - normalizer
            )
    return posterior


@njit(cache=True, nogil=True)
def hmm_filtered_posterior_kernel(
    training: np.ndarray,
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    components = initial.size
    previous = initial.copy()
    train_emissions = _gaussian_log_density_impl(training, means, variances)
    for index in range(training.shape[0]):
        current = np.empty(components, dtype=np.float64)
        maximum = np.max(train_emissions[index])
        if index == 0:
            for component in range(components):
                current[component] = previous[component] * np.exp(
                    train_emissions[index, component] - maximum
                )
        else:
            for target in range(components):
                predicted = 0.0
                for source in range(components):
                    predicted += previous[source] * transition[source, target]
                current[target] = predicted * np.exp(
                    train_emissions[index, target] - maximum
                )
        total = np.sum(current)
        previous = current / max(total, 1e-300)
    emissions = _gaussian_log_density_impl(observations, means, variances)
    posterior = np.empty((observations.shape[0], components), dtype=np.float64)
    for index in range(observations.shape[0]):
        maximum = np.max(emissions[index])
        current = np.empty(components, dtype=np.float64)
        for target in range(components):
            predicted = 0.0
            for source in range(components):
                predicted += previous[source] * transition[source, target]
            current[target] = predicted * np.exp(emissions[index, target] - maximum)
        total = np.sum(current)
        previous = current / max(total, 1e-300)
        posterior[index] = previous
    return posterior


@njit(cache=True, nogil=True)
def posterior_assignment_kernel(
    posterior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, components = posterior.shape
    assignments = np.empty(rows, dtype=np.int64)
    confidence = np.empty(rows, dtype=np.float64)
    for row in range(rows):
        winner = 0
        for component in range(1, components):
            if posterior[row, component] > posterior[row, winner]:
                winner = component
        assignments[row] = winner
        confidence[row] = posterior[row, winner]
    return assignments, confidence


# Controlled benchmark adapter: same E/M body, explicit starting parameters.
@njit(cache=True, nogil=True)
def gmm_em_reference(
    matrix, weights_input, means_input, variances_input, iterations, tolerance
):
    weights = weights_input.copy()
    means = means_input.copy()
    variances = variances_input.copy()
    rows, dimensions = matrix.shape
    components = means.shape[0]
    prior_likelihood = -np.inf
    completed = 0
    for iteration in range(1, iterations + 1):
        completed = iteration
        log_joint = _gaussian_log_density_impl(matrix, means, variances)
        responsibilities = np.empty_like(log_joint)
        likelihood = 0.0
        for row in range(rows):
            for component in range(components):
                log_joint[row, component] += np.log(max(weights[component], 1e-12))
            normalizer = _logsumexp_vector(log_joint[row])
            likelihood += normalizer
            for component in range(components):
                responsibilities[row, component] = np.exp(
                    log_joint[row, component] - normalizer
                )
        effective = np.zeros(components, dtype=np.float64)
        for component in range(components):
            for row in range(rows):
                effective[component] += responsibilities[row, component]
            effective[component] = max(effective[component], 1e-8)
            weights[component] = effective[component] / rows
            for dimension in range(dimensions):
                total = 0.0
                for row in range(rows):
                    total += responsibilities[row, component] * matrix[row, dimension]
                means[component, dimension] = total / effective[component]
        for component in range(components):
            for dimension in range(dimensions):
                total = 0.0
                for row in range(rows):
                    difference = matrix[row, dimension] - means[component, dimension]
                    total += responsibilities[row, component] * difference * difference
                variances[component, dimension] = max(
                    total / effective[component],
                    1e-5,
                )
        if abs(likelihood - prior_likelihood) < tolerance:
            break
        prior_likelihood = likelihood
    return weights, means, variances, completed, likelihood


# Controlled benchmark adapter: same E/M body, explicit starting parameters.
@njit(cache=True, nogil=True)
def hmm_em_reference(
    matrix,
    initial_input,
    transition_input,
    means_input,
    variances_input,
    iterations,
    tolerance,
):
    initial = initial_input.copy()
    transition = transition_input.copy()
    means = means_input.copy()
    variances = variances_input.copy()
    rows, dimensions = matrix.shape
    components = means.shape[0]
    prior_likelihood = -np.inf
    completed = 0
    rows = matrix.shape[0]
    dimensions = matrix.shape[1]
    for iteration in range(1, iterations + 1):
        completed = iteration
        emissions = _gaussian_log_density_impl(matrix, means, variances)
        log_transition = np.empty_like(transition)
        for row in range(components):
            for column in range(components):
                log_transition[row, column] = np.log(
                    max(transition[row, column], 1e-12)
                )
        alpha = np.empty_like(emissions)
        for component in range(components):
            alpha[0, component] = (
                np.log(max(initial[component], 1e-12)) + emissions[0, component]
            )
        workspace = np.empty(components, dtype=np.float64)
        for index in range(1, rows):
            for target in range(components):
                for source in range(components):
                    workspace[source] = (
                        alpha[index - 1, source] + log_transition[source, target]
                    )
                alpha[index, target] = emissions[index, target] + _logsumexp_vector(
                    workspace
                )
        beta = np.zeros_like(emissions)
        for index in range(rows - 2, -1, -1):
            for source in range(components):
                for target in range(components):
                    workspace[target] = (
                        log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                    )
                beta[index, source] = _logsumexp_vector(workspace)
        likelihood = _logsumexp_vector(alpha[rows - 1])
        gamma = np.empty_like(emissions)
        for index in range(rows):
            for component in range(components):
                workspace[component] = alpha[index, component] + beta[index, component]
            normalizer = _logsumexp_vector(workspace)
            for component in range(components):
                gamma[index, component] = np.exp(
                    alpha[index, component] + beta[index, component] - normalizer
                )
        xi_sum = np.zeros_like(transition)
        for index in range(rows - 1):
            xi_normalizer = -np.inf
            for source in range(components):
                for target in range(components):
                    value = (
                        alpha[index, source]
                        + log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                    )
                    if xi_normalizer == -np.inf:
                        xi_normalizer = value
                    elif value > xi_normalizer:
                        xi_normalizer = value + np.log1p(np.exp(xi_normalizer - value))
                    else:
                        xi_normalizer = xi_normalizer + np.log1p(
                            np.exp(value - xi_normalizer)
                        )
            for source in range(components):
                for target in range(components):
                    value = (
                        alpha[index, source]
                        + log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                        - xi_normalizer
                    )
                    xi_sum[source, target] += np.exp(value)
        initial_total = 0.0
        for component in range(components):
            initial[component] = max(gamma[0, component], 1e-8)
            initial_total += initial[component]
        initial /= initial_total
        for source in range(components):
            row_total = 0.0
            for target in range(components):
                transition[source, target] = max(
                    xi_sum[source, target],
                    1e-8,
                )
                row_total += transition[source, target]
            for target in range(components):
                transition[source, target] /= row_total
        effective = np.zeros(components, dtype=np.float64)
        for component in range(components):
            for index in range(rows):
                effective[component] += gamma[index, component]
            effective[component] = max(effective[component], 1e-8)
            for dimension in range(dimensions):
                total = 0.0
                for index in range(rows):
                    total += gamma[index, component] * matrix[index, dimension]
                means[component, dimension] = total / effective[component]
        for component in range(components):
            for dimension in range(dimensions):
                total = 0.0
                for index in range(rows):
                    difference = matrix[index, dimension] - means[component, dimension]
                    total += gamma[index, component] * difference * difference
                variances[component, dimension] = max(
                    total / effective[component],
                    1e-5,
                )
        if abs(likelihood - prior_likelihood) < tolerance:
            break
        prior_likelihood = likelihood
    return initial, transition, means, variances, completed, likelihood


# Same smoothing body with emissions supplied explicitly on both backends.
@njit(cache=True, nogil=True)
def smooth_emission_reference(emissions, initial, transition):
    rows = emissions.shape[0]
    components = initial.size
    log_transition = np.empty_like(transition)
    for source in range(components):
        for target in range(components):
            log_transition[source, target] = np.log(
                max(transition[source, target], 1e-12)
            )
    alpha = np.empty_like(emissions)
    beta = np.zeros_like(emissions)
    workspace = np.empty(components, dtype=np.float64)
    for component in range(components):
        alpha[0, component] = (
            np.log(max(initial[component], 1e-12)) + emissions[0, component]
        )
    for index in range(1, rows):
        for target in range(components):
            for source in range(components):
                workspace[source] = (
                    alpha[index - 1, source] + log_transition[source, target]
                )
            alpha[index, target] = emissions[index, target] + _logsumexp_vector(
                workspace
            )
    for index in range(rows - 2, -1, -1):
        for source in range(components):
            for target in range(components):
                workspace[target] = (
                    log_transition[source, target]
                    + emissions[index + 1, target]
                    + beta[index + 1, target]
                )
            beta[index, source] = _logsumexp_vector(workspace)
    posterior = np.empty_like(emissions)
    for index in range(rows):
        for component in range(components):
            workspace[component] = alpha[index, component] + beta[index, component]
        normalizer = _logsumexp_vector(workspace)
        for component in range(components):
            posterior[index, component] = np.exp(
                alpha[index, component] + beta[index, component] - normalizer
            )
    return posterior
