# Frozen mathematical bodies; test-only, lazy compilation.
import numpy as np
from numba import njit

@njit(cache=True, nogil=True, fastmath=False)
def returns_to_nav_kernel(period_returns: np.ndarray, initial_nav: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, float]:
    path_count, step_count = period_returns.shape
    nav_paths = np.empty((path_count, step_count + 1), dtype=np.float64)
    terminal_returns = np.empty(path_count, dtype=np.float64)
    max_drawdowns = np.zeros(path_count, dtype=np.float64)
    minimum_return = np.inf
    for path_index in range(path_count):
        nav = initial_nav
        peak = initial_nav
        nav_paths[path_index, 0] = initial_nav
        for step_index in range(step_count):
            period_return = period_returns[path_index, step_index]
            if period_return < minimum_return:
                minimum_return = period_return
            if not np.isfinite(period_return):
                return (nav_paths, terminal_returns, max_drawdowns, 1, step_index, path_index, minimum_return)
            if period_return <= -1.0:
                return (nav_paths, terminal_returns, max_drawdowns, 2, step_index, path_index, minimum_return)
            nav *= 1.0 + period_return
            if not np.isfinite(nav) or nav <= 0.0:
                return (nav_paths, terminal_returns, max_drawdowns, 3, step_index, path_index, minimum_return)
            nav_paths[path_index, step_index + 1] = nav
            if nav > peak:
                peak = nav
            drawdown = 1.0 - nav / peak
            if drawdown > max_drawdowns[path_index]:
                max_drawdowns[path_index] = drawdown
        terminal_returns[path_index] = nav / initial_nav - 1.0
    return (nav_paths, terminal_returns, max_drawdowns, 0, -1, -1, minimum_return)

@njit(cache=True, nogil=True, fastmath=False)
def quantile_sorted_kernel(sorted_values: np.ndarray, probability: float) -> float:
    count = sorted_values.shape[0]
    if count == 1:
        return sorted_values[0]
    position = probability * (count - 1)
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction

@njit(cache=True, nogil=True, fastmath=False)
def fan_statistics_kernel(nav_paths: np.ndarray, terminal_returns: np.ndarray, max_drawdowns: np.ndarray, target_return: float) -> tuple[np.ndarray, np.ndarray]:
    path_count, nav_step_count = nav_paths.shape
    probabilities = np.asarray((0.05, 0.25, 0.5, 0.75, 0.95), dtype=np.float64)
    fan = np.empty((5, nav_step_count), dtype=np.float64)
    for step_index in range(nav_step_count):
        sorted_nav = np.sort(nav_paths[:, step_index])
        for quantile_index in range(5):
            fan[quantile_index, step_index] = quantile_sorted_kernel(sorted_nav, probabilities[quantile_index])
    sorted_returns = np.sort(terminal_returns)
    sorted_drawdowns = np.sort(max_drawdowns)
    sorted_terminal_nav = np.sort(nav_paths[:, nav_step_count - 1])
    metrics = np.empty(14, dtype=np.float64)
    for quantile_index in range(5):
        metrics[quantile_index] = quantile_sorted_kernel(sorted_terminal_nav, probabilities[quantile_index])
    metrics[5] = quantile_sorted_kernel(sorted_returns, 0.05)
    metrics[6] = quantile_sorted_kernel(sorted_returns, 0.5)
    metrics[7] = quantile_sorted_kernel(sorted_returns, 0.95)
    boundary = metrics[5]
    tail_total = 0.0
    tail_count = 0
    loss_count = 0
    target_count = 0
    drawdown_total = 0.0
    for path_index in range(path_count):
        terminal_return = terminal_returns[path_index]
        if terminal_return <= boundary:
            tail_total += terminal_return
            tail_count += 1
        if terminal_return < 0.0:
            loss_count += 1
        if terminal_return >= target_return:
            target_count += 1
        drawdown_total += max_drawdowns[path_index]
    metrics[8] = max(0.0, -boundary)
    metrics[9] = max(0.0, -(tail_total / tail_count)) if tail_count > 0 else metrics[8]
    metrics[10] = loss_count / path_count
    metrics[11] = target_count / path_count
    metrics[12] = drawdown_total / path_count
    metrics[13] = quantile_sorted_kernel(sorted_drawdowns, 0.95)
    return (fan, metrics)
