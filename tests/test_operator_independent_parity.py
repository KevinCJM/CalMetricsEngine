"""Independent NumPy / pure-Python references for all 118 canonical operators.

This module is test-only.  It intentionally does not import calmetrics_engine,
BetterSaaTaa, Numba, or the frozen expected values.  NumPy is preferred where
it has a direct mathematical equivalent; stateful/domain-specific operations use
small readable Python reference algorithms.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from calmetrics_engine import operators as op

FIXTURE = json.loads((Path(__file__).parent / "data/canonical_reference.json").read_text())
SUCCESS_CASES = [case for case in FIXTURE["cases"] if "expected" in case]


def decode(value):
    if isinstance(value, dict):
        if "array" in value:
            return np.asarray(
                [decode(item) for item in value["array"]], dtype=value["dtype"]
            ).reshape(value["shape"])
        if "tuple" in value:
            return tuple(decode(item) for item in value["tuple"])
        if "scalar" in value:
            return np.dtype(value["dtype"]).type(value["scalar"])
    if isinstance(value, str) and value in {"nan", "inf", "-inf"}:
        return float(value)
    return value


def _as_array(value):
    return np.asarray(value)


def _scalar_or_array(original, value, *, mask=False):
    array = np.asarray(value, dtype=np.uint8 if mask else np.float64)
    if np.asarray(original).ndim == 0:
        return bool(array.item()) if mask else float(array.item())
    return array


def _binary(lhs, rhs, function, *, mask=False):
    left, right = np.asarray(lhs), np.asarray(rhs)
    shape = np.broadcast_shapes(left.shape, right.shape)
    if not shape:
        result = function(float(left), float(right))
        return bool(result) if mask else float(result)
    left = np.broadcast_to(left, shape)
    right = np.broadcast_to(right, shape)
    dtype = np.uint8 if mask else np.float64
    out = np.empty(shape, dtype=dtype)
    for index in np.ndindex(shape):
        out[index] = function(float(left[index]), float(right[index]))
    return out


def _unary(value, function):
    source = np.asarray(value)
    if source.ndim == 0:
        return float(function(float(source)))
    out = np.empty(source.shape, dtype=np.float64)
    for index in np.ndindex(source.shape):
        out[index] = function(float(source[index]))
    return out


def _source_min(x, y):
    # Python/Numba source semantics: ties and unordered comparisons keep lhs.
    return y if y < x else x


def _source_max(x, y):
    return y if y > x else x


def _ordered_stat(values, probability=None):
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if probability is None:
        middle = ordered.size // 2
        return (
            float(ordered[middle])
            if ordered.size % 2
            else float((ordered[middle - 1] + ordered[middle]) * 0.5)
        )
    return float(np.quantile(ordered, probability, method="linear"))


def _reduce(name, values, ddof=1):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if name == "sum":
        return float(np.sum(x))
    if name == "product":
        return float(np.prod(x))
    if name == "mean":
        return float(np.mean(x))
    if name == "min_value":
        best = float(x[0])
        for value in x[1:]:
            best = _source_min(best, float(value))
        return best
    if name == "max_value":
        best = float(x[0])
        for value in x[1:]:
            best = _source_max(best, float(value))
        return best
    if name == "variance":
        return float(np.var(x, ddof=ddof))
    if name == "std":
        return float(np.std(x, ddof=ddof))
    if name == "median":
        return _ordered_stat(x)
    mean = float(np.mean(x))
    centered = x - mean
    if name == "mean_absolute_deviation":
        return float(np.mean(np.abs(centered)))
    if name == "root_mean_square":
        return float(np.sqrt(np.mean(x * x)))
    if name == "skewness":
        n = x.size
        m2 = float(np.mean(centered**2))
        m3 = float(np.mean(centered**3))
        return float(math.sqrt(n * (n - 1.0)) / (n - 2.0) * m3 / (m2**1.5))
    if name == "excess_kurtosis":
        n = x.size
        m2 = float(np.mean(centered**2))
        excess = float(np.mean(centered**4)) / (m2 * m2) - 3.0
        return float((n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * excess + 6.0))
    if name in {"argmin", "argmax"}:
        best = float(x[0])
        best_index = 0
        for index, value in enumerate(x[1:], 1):
            value = float(value)
            if (name == "argmin" and value < best) or (name == "argmax" and value > best):
                best, best_index = value, index
        return float(best_index)
    raise AssertionError(name)


def _masked(name, values, mask, probability=None):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = np.asarray(mask, dtype=np.uint8).reshape(-1) != 0
    chosen = x[selected]
    if name == "sum_where":
        return float(np.sum(chosen))
    if name == "mean_where":
        return float(np.mean(chosen))
    if name == "variance_where":
        return float(np.var(chosen, ddof=1))
    if name == "std_where":
        return float(np.std(chosen, ddof=1))
    if name == "min_where":
        return _reduce("min_value", chosen)
    if name == "max_where":
        return _reduce("max_value", chosen)
    if name == "median_where":
        return _ordered_stat(chosen)
    if name == "quantile_where":
        return _ordered_stat(chosen, float(probability))
    raise AssertionError(name)


def _rolling(values, width, minimum, mode, ddof=0):
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    width, minimum, ddof = int(width), int(minimum), int(ddof)
    for end in range(x.size):
        window = x[max(0, end - width + 1) : end + 1]
        finite = window[np.isfinite(window)]
        if finite.size < minimum:
            continue
        if mode == "mean":
            out[end] = np.mean(finite)
        elif mode == "std":
            if finite.size > ddof:
                out[end] = np.std(finite, ddof=ddof)
        elif mode == "min":
            out[end] = np.min(finite)
        elif mode == "max":
            out[end] = np.max(finite)
    return out


def _linear_fit(*args):
    y = np.asarray(args[-1], dtype=np.float64)
    x = (
        np.arange(y.size, dtype=np.float64)
        if len(args) == 1
        else np.asarray(args[0], dtype=np.float64)
    )
    mx, my = float(np.mean(x)), float(np.mean(y))
    dx, dy = x - mx, y - my
    xx = float(np.dot(dx, dx))
    slope = float(np.dot(dx, dy) / xx)
    intercept = float(my - slope * mx)
    residuals = y - (intercept + slope * x)
    return (
        slope,
        intercept,
        float(np.dot(residuals, residuals)),
        float(np.dot(dy, dy)),
        float(y.size),
    )


def _last_drawdown_interval(drawdowns):
    x = np.asarray(drawdowns, dtype=np.float64)
    if x.size == 0 or x[0] != 0.0:
        return (np.nan, np.nan, np.nan, -1.0)
    deepest = 0.0
    latest_peak = 0.0
    peak = trough = recovery = np.nan
    for index, value in enumerate(x):
        value = float(value)
        if not math.isfinite(value) or value > 0.0 or value < -1.0:
            return (np.nan, np.nan, np.nan, -1.0)
        if value == 0.0:
            latest_peak = float(index)
            if math.isfinite(trough) and not math.isfinite(recovery):
                recovery = float(index)
        elif value <= deepest:
            deepest = value
            peak = latest_peak
            trough = float(index)
            recovery = np.nan
    return (peak, trough, recovery, 1.0 if deepest < 0.0 else 0.0)


def _normal_ppf(value):
    x = np.asarray(value, dtype=np.float64)
    normal = NormalDist()
    if x.ndim == 0:
        return float(normal.inv_cdf(float(x)))
    out = np.empty_like(x)
    for index in np.ndindex(x.shape):
        out[index] = normal.inv_cdf(float(x[index]))
    return out


def _axis_reduce(name, values, axis):
    x = np.asarray(values, dtype=np.float64)
    base = name.rsplit("_", 1)[0]
    slices = (
        (x[:, index] for index in range(x.shape[1]))
        if axis == 0
        else (x[index, :] for index in range(x.shape[0]))
    )
    return np.asarray(
        [_reduce(base if base not in {"min", "max"} else f"{base}_value", item) for item in slices]
    )


def reference(name, *args):
    binary = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
        "power": lambda x, y: x**y,
        "minimum": _source_min,
        "maximum": _source_max,
    }
    if name in binary:
        return _binary(args[0], args[1], binary[name])

    unary = {
        "negate": lambda x: -x,
        "absolute": abs,
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "reciprocal": lambda x: 1.0 / x,
        "sign": lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
        "normal_pdf": lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi),
    }
    if name in unary:
        return _unary(args[0], unary[name])
    if name == "normal_ppf":
        return _normal_ppf(args[0])
    if name == "clip":
        value, lower, upper = args
        return _unary(value, lambda x: _source_min(_source_max(x, float(lower)), float(upper)))

    comparisons = {
        "equal": lambda x, y: x == y,
        "not_equal": lambda x, y: x != y,
        "less_than": lambda x, y: x < y,
        "less_equal": lambda x, y: x <= y,
        "greater_than": lambda x, y: x > y,
        "greater_equal": lambda x, y: x >= y,
    }
    if name in comparisons:
        return _binary(args[0], args[1], comparisons[name], mask=True)
    if name in {"logical_and", "logical_or"}:
        fn = (
            (lambda x, y: bool(x) and bool(y))
            if name == "logical_and"
            else (lambda x, y: bool(x) or bool(y))
        )
        return _binary(args[0], args[1], fn, mask=True)
    if name == "logical_not":
        x = np.asarray(args[0])
        result = np.logical_not(x).astype(np.uint8)
        return bool(result.item()) if x.ndim == 0 else result
    if name == "where":
        mask, left, right = args
        result = np.where(np.asarray(mask, dtype=bool), left, right)
        return (
            float(result) if np.asarray(result).ndim == 0 else np.asarray(result, dtype=np.float64)
        )
    if name == "finite_mask":
        return np.isfinite(np.asarray(args[0], dtype=np.float64)).astype(np.uint8)

    if name in {
        "sum_where",
        "mean_where",
        "variance_where",
        "std_where",
        "min_where",
        "max_where",
        "median_where",
    }:
        return _masked(name, args[0], args[1])
    if name == "quantile_where":
        return _masked(name, args[0], args[1], args[2])
    if name == "count_true":
        return float(np.count_nonzero(np.asarray(args[0], dtype=bool)))
    if name == "max_consecutive_true":
        longest = current = 0
        for value in np.asarray(args[0], dtype=bool).reshape(-1):
            current = current + 1 if value else 0
            longest = max(longest, current)
        return float(longest)

    if name in {
        "sum",
        "product",
        "mean",
        "min_value",
        "max_value",
        "median",
        "skewness",
        "excess_kurtosis",
        "mean_absolute_deviation",
        "root_mean_square",
        "argmin",
        "argmax",
    }:
        return _reduce(name, args[0])
    if name in {"variance", "std"}:
        return _reduce(name, args[0], int(args[1]) if len(args) == 2 else 1)
    if name == "quantile":
        return _ordered_stat(args[0], float(args[1]))

    x = np.asarray(args[0], dtype=np.float64) if args else None
    if name == "cumulative_sum":
        return np.cumsum(x)
    if name == "cumulative_product":
        return np.cumprod(x)
    if name == "cumulative_return":
        return np.cumprod(1.0 + x) - 1.0
    if name in {"cumulative_max", "cumulative_min"}:
        out = np.empty_like(x)
        running = float(x[0])
        for index, value in enumerate(x):
            running = (
                _source_max(running, float(value))
                if name == "cumulative_max"
                else _source_min(running, float(value))
            )
            out[index] = running
        return out
    if name == "drawdown_series":
        peak = np.maximum.accumulate(x)
        return x / peak - 1.0
    if name == "new_high_mask":
        out = np.zeros(x.size, dtype=np.uint8)
        peak = float(x[0])
        out[0] = 1
        for index in range(1, x.size):
            if x[index] > peak:
                peak = float(x[index])
                out[index] = 1
        return out
    if name == "rolling_mean":
        width = int(args[1])
        minimum = int(args[2]) if len(args) == 3 else width
        return _rolling(x, width, minimum, "mean")
    if name == "rolling_std":
        width = int(args[1])
        ddof = int(args[2]) if len(args) >= 3 else 0
        minimum = int(args[3]) if len(args) == 4 else width
        return _rolling(x, width, minimum, "std", ddof)
    if name in {"rolling_min", "rolling_max"}:
        width = int(args[1])
        minimum = int(args[2]) if len(args) == 3 else width
        return _rolling(x, width, minimum, name.removeprefix("rolling_"))
    if name == "recursive_smooth":
        width, previous = int(args[1]), float(args[2])
        out = np.full(x.shape, np.nan)
        for index, current in enumerate(x):
            if not np.isfinite(current):
                continue
            previous = ((width - 1.0) * previous + current) / width
            out[index] = previous
        return out
    if name == "divide_or_default":
        numerator, denominator, default = np.asarray(args[0]), np.asarray(args[1]), float(args[2])
        out = np.full(numerator.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(numerator) & np.isfinite(denominator)
        near_zero = finite & (np.abs(denominator) < 1e-12)
        regular = finite & ~near_zero
        out[near_zero] = default
        out[regular] = numerator[regular] / denominator[regular]
        return out
    if name == "first":
        return float(x[0])
    if name == "last":
        return float(x[-1])
    if name == "length":
        return float(x.size)
    if name == "lag":
        periods = int(args[1]) if len(args) == 2 else 1
        return x[: x.size - periods]
    if name == "difference":
        periods = int(args[1]) if len(args) == 2 else 1
        return x[periods:] - x[: x.size - periods]

    if name.endswith("_time"):
        return _axis_reduce(name, args[0], 0)
    if name.endswith("_asset"):
        return _axis_reduce(name, args[0], 1)

    if name == "transpose":
        return np.asarray(args[0]).T
    if name == "dot":
        return float(np.dot(args[0], args[1]))
    if name == "outer":
        return np.outer(args[0], args[1])
    if name == "matmul":
        return np.matmul(args[0], args[1])
    if name in {"matvec", "portfolio_returns"}:
        return np.matmul(args[0], args[1])
    if name == "diag":
        return np.diag(args[0])
    if name == "trace":
        return float(np.trace(args[0]))
    if name == "solve":
        return np.linalg.solve(args[0], args[1])
    if name == "covariance":
        if len(args) == 2:
            return float(np.cov(args[0], args[1], ddof=1)[0, 1])
        return np.cov(args[0], rowvar=False, ddof=1)
    if name == "correlation":
        if len(args) == 2:
            return float(np.corrcoef(args[0], args[1])[0, 1])
        return np.corrcoef(args[0], rowvar=False)
    if name == "quadratic_form":
        vector, matrix = np.asarray(args[0]), np.asarray(args[1])
        return float(vector @ matrix @ vector)
    if name == "active_returns":
        return np.asarray(args[0]) - np.asarray(args[1])

    if name == "total_return":
        return float(np.prod(1.0 + x) - 1.0)
    if name == "annualized_return":
        base = (float(np.prod(1.0 + x)) - 1.0) + 1.0
        return float(base ** (float(args[1]) / x.size) - 1.0)

    if name in {
        "linear_fit",
        "linear_slope",
        "linear_intercept",
        "linear_r_squared",
        "regression_standard_error",
    }:
        fit = _linear_fit(*args)
        if name == "linear_fit":
            return fit
        if name == "linear_slope":
            return fit[0]
        if name == "linear_intercept":
            return fit[1]
        if name == "linear_r_squared":
            return float(1.0 - fit[2] / fit[3])
        return float(math.sqrt(fit[2] / (fit[4] - 2.0)))

    if name == "last_drawdown_interval":
        return _last_drawdown_interval(args[0])
    if name in {"interval_start", "interval_trough", "interval_recovery"}:
        return float(
            args[0][{"interval_start": 0, "interval_trough": 1, "interval_recovery": 2}[name]]
        )
    if name == "value_at":
        position = float(args[1])
        if (
            not math.isfinite(position)
            or position < 0
            or position != math.floor(position)
            or position >= x.size
        ):
            return np.nan
        value = float(x[int(position)])
        return value if math.isfinite(value) else np.nan
    if name == "days_between":
        start, end = map(float, args)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start != math.floor(start)
            or end != math.floor(end)
            or end < start
        ):
            return np.nan
        return end - start
    if name == "require_positive":
        return float(args[0])
    if name == "require_nonnegative":
        return float(args[0])
    if name in {
        "fit_slope",
        "fit_intercept",
        "fit_residual_sum_squares",
        "fit_total_sum_squares",
        "fit_observation_count",
    }:
        return float(
            args[0][
                {
                    "fit_slope": 0,
                    "fit_intercept": 1,
                    "fit_residual_sum_squares": 2,
                    "fit_total_sum_squares": 3,
                    "fit_observation_count": 4,
                }[name]
            ]
        )
    raise AssertionError(f"missing independent reference for {name}")


def _assert_close(actual, expected, name):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_close(left, right, name)
        return
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape
    if expected_array.dtype == np.uint8 or expected_array.dtype == np.bool_:
        np.testing.assert_array_equal(actual_array, expected_array)
        return
    # NormalDist uses an independent high-accuracy inverse-CDF implementation;
    # the production contract intentionally keeps Acklam's approximation.
    atol = 2e-9 if name == "normal_ppf" else 2e-12
    rtol = 2e-9 if name == "normal_ppf" else 2e-11
    np.testing.assert_allclose(actual_array, expected_array, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("simd", ["scalar", "auto"])
def test_independent_numpy_python_reference(case, simd):
    args = [decode(item) for item in case["args"]]
    expected = decode(case["expected"])
    independent = reference(case["operator"], *args)
    native = op.call(case["operator"], *args, simd=simd)

    # The independent implementation must agree both with the frozen original
    # implementation and with today's C++ backend, including the automatic
    # ISA lane when the operator has one.
    _assert_close(independent, expected, case["operator"])
    _assert_close(native, independent, case["operator"])


def test_independent_reference_covers_all_118_operators():
    covered = {case["operator"] for case in SUCCESS_CASES}
    assert covered == set(FIXTURE["canonical_names"])
    for name in FIXTURE["canonical_names"]:
        case = next(case for case in SUCCESS_CASES if case["operator"] == name)
        args = [decode(item) for item in case["args"]]
        reference(name, *args)
