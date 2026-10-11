"""Frozen source-shaped MC projection mathematics; controlled NJIT evidence only.

No FundInvestmentResearchPlatform import or production fallback is permitted.
The frozen_one body preserves its intercept-first, coefficient-first ascending
factor fold, weight-first asset fold, step/path contribution order and checks.
Only healthy, already-transformed tapes are compared with the selected DAG
outputs; the full ordered first-fault prefix protocol is not migrated here.
Decorators are lazy: importing this file does not compile numerical signatures.
"""

import numpy as np
from numba import njit

FROZEN_SOURCE_DATE = "2026-10-03"
FROZEN_SOURCE_SHA256 = "bb53056653d697c698dab4ac5e716d96dbaec9071157d755c3fbef32e9831b79"
FROZEN_SOURCE_SYMBOL = "backend/scenario_stress/numba_kernels.py::monte_carlo_projection_kernel"


@njit(cache=True, nogil=True, fastmath=False)
def frozen_one(factor_draws, beta, intercepts, effective_weights, available_assets):
    step_count, path_count, factor_count = factor_draws.shape
    asset_count = beta.shape[0]
    period_returns = np.empty((path_count, step_count), dtype=np.float64)
    mean_contributions = np.zeros(asset_count, dtype=np.float64)
    minimum_return = np.inf
    for step_index in range(step_count):
        for path_index in range(path_count):
            portfolio_return = 0.0
            for asset_index in range(asset_count):
                if available_assets[asset_index] == 0:
                    continue
                log_return = intercepts[asset_index]
                for factor_index in range(factor_count):
                    log_return += beta[asset_index, factor_index] * factor_draws[step_index, path_index, factor_index]
                simple_return = np.expm1(log_return)
                contribution = effective_weights[asset_index] * simple_return
                portfolio_return += contribution
                mean_contributions[asset_index] += contribution
            if portfolio_return < minimum_return:
                minimum_return = portfolio_return
            if not np.isfinite(portfolio_return):
                return period_returns, mean_contributions, 1, step_index, path_index, minimum_return
            if portfolio_return <= -1.0:
                return period_returns, mean_contributions, 2, step_index, path_index, minimum_return
            period_returns[path_index, step_index] = portfolio_return
    for asset_index in range(asset_count):
        if available_assets[asset_index] == 0:
            mean_contributions[asset_index] = np.nan
        else:
            mean_contributions[asset_index] /= path_count
    return period_returns, mean_contributions, 0, -1, -1, minimum_return


@njit(cache=True, nogil=True, fastmath=False)
def service_chain(x, beta, initial, weights):
    # The platform calls the original projection once for each combination.
    # Returning lists preserves independent complete period/contribution arrays
    # without constructing and copying another C*P*T packed output tensor.
    periods, contributions = [], []
    available = np.ones(beta.shape[0], np.int64)
    for combination in range(weights.shape[0]):
        period, contribution, status, step, path, minimum = frozen_one(
            x, beta, initial, weights[combination], available
        )
        if status:
            raise ValueError("REFERENCE_OUTSIDE_HEALTHY_PROJECTION_SCOPE")
        periods.append(period)
        contributions.append(contribution)
    return periods, contributions


@njit(cache=True, nogil=True, fastmath=False)
def shared_chain(x, beta, initial, weights):
    # Additional optimized NJIT comparator: no full N*A L/R tensor. Sharing is
    # legal because all combinations use identical tape/beta/intercepts.
    steps, paths, factors = x.shape
    combinations, assets = weights.shape
    periods, contributions = [], []
    for combination in range(combinations):
        periods.append(np.empty((paths, steps)))
        contributions.append(np.zeros(assets))
    simple = np.empty(assets)
    for step in range(steps):
        for path in range(paths):
            for asset in range(assets):
                value = initial[asset]
                for factor in range(factors):
                    value += beta[asset, factor] * x[step, path, factor]
                simple[asset] = np.expm1(value)
            for combination in range(combinations):
                total = 0.0
                for asset in range(assets):
                    value = weights[combination, asset] * simple[asset]
                    total += value
                    contributions[combination][asset] += value
                if not np.isfinite(total) or total <= -1.0:
                    raise ValueError("REFERENCE_OUTSIDE_HEALTHY_PROJECTION_SCOPE")
                periods[combination][path, step] = total
    for combination in range(combinations):
        for asset in range(assets):
            contributions[combination][asset] /= paths
    return periods, contributions
