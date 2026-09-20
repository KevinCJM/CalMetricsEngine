"""Capture oracle results by executing the explicitly supplied BetterSaaTaa NJIT source.

Developer-only tool: NumPy and Numba must exist in the selected interpreter.
Neither is imported from CalMetricsEngine. CI uses the frozen JSON, not this tool.
The source tree is read-only; bytecode and Numba caches are redirected outside it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "typed_numba_kernels.py",
    "typed_numba_plan.py",
    "operator_lowering.py",
    "typed_operators.py",
    "primitive_access.py",
    "regression_state.py",
    "drawdown_interval.py",
)


def encode(value):
    if isinstance(value, np.ndarray):
        return {
            "array": [encode(x.item()) for x in value.flat],
            "dtype": value.dtype.name,
            "shape": list(value.shape),
        }
    if isinstance(value, tuple):
        return {"tuple": [encode(x) for x in value]}
    if isinstance(value, (np.uint8, np.bool_)):
        return {"scalar": value.item(), "dtype": value.dtype.name}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
    return value


def source_names(directory: Path) -> tuple[str, ...]:
    tree = ast.parse((directory / "typed_numba_kernels.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CANONICAL_OPERATOR_IDS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("CANONICAL_OPERATOR_IDS not found")


def load_oracle(source: Path):
    sys.dont_write_bytecode = True
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / ".build-reference-cache")
    sys.path.insert(0, str(source / "backend"))
    kernels = importlib.import_module("cal_indicators.typed_numba_kernels")
    access = importlib.import_module("cal_indicators.primitive_access")
    fits = importlib.import_module("cal_indicators.regression_state")
    intervals = importlib.import_module("cal_indicators.drawdown_interval")

    def call(name, *args):
        rank = args[0].ndim if isinstance(args[0], np.ndarray) else 0
        if name in kernels.BASIC_OPCODES or name in kernels.COMPARISON_OPCODES:
            comparison = name in kernels.COMPARISON_OPCODES
            opcode = (kernels.COMPARISON_OPCODES if comparison else kernels.BASIC_OPCODES)[name]
            right_rank = args[1].ndim if isinstance(args[1], np.ndarray) else 0
            output_rank = max(rank, right_rank)
            suffix = "scalar" if not output_rank else f"{output_rank}d"
            if output_rank and not right_rank:
                suffix += "_right_scalar"
            elif output_rank and not rank:
                suffix += "_left_scalar"
            fn = getattr(kernels, f"{'comparison' if comparison else 'binary'}_{suffix}")
            return fn(opcode, *args)
        if name in kernels.UNARY_OPCODES:
            fn = getattr(kernels, f"unary_{'scalar' if not rank else str(rank) + 'd'}")
            return fn(kernels.UNARY_OPCODES[name], *args)
        if name == "clip":
            return getattr(kernels, f"clip_{'scalar' if not rank else str(rank) + 'd'}")(*args)
        if name in {"logical_and", "logical_or", "logical_not"}:
            suffix = "scalar" if not rank else f"{rank}d"
            logical_args = tuple(np.uint8(a) if not isinstance(a, np.ndarray) else a for a in args)
            if name == "logical_not":
                return getattr(kernels, f"logical_not_{suffix}")(*logical_args)
            return getattr(kernels, f"logical_{suffix}")(
                1 if name == "logical_and" else 2, *logical_args
            )
        if name == "where":
            numeric_rank = max(a.ndim if isinstance(a, np.ndarray) else 0 for a in args[1:])
            suffix = "scalar" if not max(rank, numeric_rank) else f"{max(rank, numeric_rank)}d"
            if suffix != "scalar":
                if not isinstance(args[2], np.ndarray):
                    suffix += "_false_scalar"
                elif not isinstance(args[1], np.ndarray):
                    suffix += "_true_scalar"
            return getattr(kernels, f"where_{suffix}")(*args)
        if name in kernels.REDUCTION_OPCODES:
            degrees = args[1] if len(args) == 2 else 1
            if isinstance(degrees, float) and not degrees.is_integer():
                return getattr(kernels, f"reduce_{rank}d_parameter")(
                    args[0], degrees, kernels.REDUCTION_OPCODES[name]
                )
            return getattr(kernels, f"reduce_{rank}d")(
                kernels.REDUCTION_OPCODES[name], args[0], int(degrees)
            )
        if name == "quantile":
            return getattr(kernels, f"quantile_{rank}d")(*args)
        if name in kernels.MASK_REDUCTION_OPCODES:
            return getattr(kernels, f"masked_reduce_{rank}d")(
                kernels.MASK_REDUCTION_OPCODES[name], *args
            )
        if name == "quantile_where":
            return getattr(kernels, f"masked_quantile_{rank}d")(*args)
        if name == "count_true":
            return getattr(kernels, f"count_true_{rank}d")(*args)
        if name == "max_consecutive_true":
            return kernels.max_consecutive_true_1d(*args)
        if name in kernels.SCAN_OPCODES:
            return kernels.scan_1d(kernels.SCAN_OPCODES[name], *args)
        if name.endswith("_time") or name.endswith("_asset"):
            suffix = name.rsplit("_", 1)[0]
            fn = (
                kernels.axis_reduce_time_fixed
                if name.endswith("_time")
                else kernels.axis_reduce_asset
            )
            return fn(kernels.AXIS_REDUCTION_OPCODES[suffix], *args)
        if name in {"lag", "difference"}:
            return getattr(kernels, f"{name}_1d_parameter")(
                args[0], float(args[1] if len(args) == 2 else 1)
            )
        if name in {"rolling_mean", "rolling_min", "rolling_max"}:
            minimum = args[2] if len(args) > 2 else args[1]
            return getattr(kernels, name + "_1d")(args[0], float(args[1]), float(minimum))
        if name == "rolling_std":
            degrees = args[2] if len(args) > 2 else 0
            minimum = args[3] if len(args) > 3 else args[1]
            return kernels.rolling_std_1d(args[0], float(args[1]), float(degrees), float(minimum))
        if name in {
            "drawdown_series",
            "new_high_mask",
            "first",
            "last",
            "length",
            "recursive_smooth",
            "divide_or_default",
        }:
            return getattr(kernels, name + "_1d")(*args)
        if name in {"dot", "outer"}:
            return getattr(kernels, name + "_1d")(*args)
        if name in {"transpose", "matmul", "matvec", "trace", "solve"}:
            return getattr(kernels, name + "_2d")(*args)
        if name == "diag":
            return getattr(kernels, f"diag_{rank}d")(*args)
        if name in {"covariance", "correlation"}:
            return getattr(kernels, name + ("_2d" if len(args) == 1 else "_1d"))(*args)
        if name in access.ACCESS_KERNELS:
            return access.ACCESS_KERNELS[name](*args)
        if name in intervals.INTERVAL_KERNELS:
            return intervals.INTERVAL_KERNELS[name](*args)
        if name in fits.FIT_PROJECTION_KERNELS:
            return fits.FIT_PROJECTION_KERNELS[name](*args)
        if name == "linear_fit" or name in {
            "linear_slope",
            "linear_intercept",
            "linear_r_squared",
            "regression_standard_error",
        }:
            state = (
                fits.linear_fit_time_kernel(args[0])
                if len(args) == 1
                else fits.linear_fit_pair_kernel(*args)
            )
            if name == "linear_fit":
                return state
            if name in {"linear_slope", "linear_intercept"}:
                return fits.FIT_PROJECTION_KERNELS[name.replace("linear_", "fit_")](state)
            if name == "linear_r_squared":
                return call(
                    "subtract", 1.0, call("divide", state[2], call("require_positive", state[3]))
                )
            return call("sqrt", call("divide", state[2], call("require_positive", state[4] - 2.0)))
        # Compiler-owned compositions from operator_lowering.py, evaluated using
        # the source's actual primitive dispatchers rather than a new oracle formula.
        if name == "active_returns":
            return call("subtract", *args)
        if name == "portfolio_returns":
            return call("matvec", *args)
        if name == "quadratic_form":
            return call("dot", call("matvec", call("transpose", args[1]), args[0]), args[0])
        if name == "cumulative_return":
            return call("subtract", call("cumulative_product", call("add", args[0], 1.0)), 1.0)
        if name in {"total_return", "annualized_return"}:
            total = call("subtract", call("product", call("add", args[0], 1.0)), 1.0)
            if name == "total_return":
                return total
            base = call("require_nonnegative", call("add", total, 1.0))
            exponent = call("divide", call("require_positive", args[1]), call("length", args[0]))
            return call("subtract", call("power", base, exponent), 1.0)
        raise KeyError(name)

    return call, kernels


def samples(names, kernels):
    rng = np.random.default_rng(13017)
    x = np.array([0.1, -0.02, 0.05, 0.0, -0.06, 0.03, 0.11, -0.025], dtype=np.float64)
    positive = x + 2.0
    matrix = rng.normal(size=(7, 3))
    mask = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8)
    matrix_mask = (matrix > 0).astype(np.uint8)
    levels = np.array([1.0, 1.2, 1.1, 1.05, 1.2, 1.3, 1.0, 1.1])
    draws = np.array([0.0, -0.25, 0.0, -0.1, -0.25, -0.1, 0.0, -0.2])
    finite_gap = x.copy()
    finite_gap[[1, 4]] = np.nan
    square = np.array([[4.0, 2.0, -1.0], [1.0, 5.0, 3.0], [2.0, -1.0, 6.0]])
    weights = np.array([0.2, 0.3, 0.5])
    fit = (0.2, 1.0, 0.25, 3.0, 8.0)
    interval = (2.0, 4.0, 6.0, 1.0)
    cases = []

    def add(name, label, *args):
        cases.append((name, label, args))

    for name in names:
        if name in kernels.BASIC_OPCODES or name in kernels.COMPARISON_OPCODES:
            add(name, "scalar", 2.0, 3.0)
            add(name, "vector", positive, positive[::-1].copy())
            add(name, "right_scalar", positive, 2.0)
            add(name, "left_scalar", 2.0, positive)
            a = np.abs(matrix) + 1
            add(name, "matrix", a, a + 0.5)
            add(name, "matrix_scalar", a, 2.0)
            add(name, "scalar_matrix", 2.0, a)
        elif name in kernels.UNARY_OPCODES:
            values = np.linspace(0.1, 0.9, 8) if name == "normal_ppf" else positive
            add(name, "scalar", 0.3)
            add(name, "vector", values)
            add(name, "matrix", values.reshape(4, 2))
        elif name == "clip":
            for label, a in (("scalar", 2.0), ("vector", x), ("matrix", matrix)):
                add(name, label, a, -0.02, 0.05)
            add(name, "bad_bounds", x, 1.0, 0.0)
        elif name in {"logical_and", "logical_or", "logical_not"}:
            for label, a in (("scalar", np.uint8(1)), ("vector", mask), ("matrix", matrix_mask)):
                add(name, label, a, *(() if name == "logical_not" else (a,)))
        elif name == "where":
            add(name, "scalar", np.uint8(1), 1.0, 2.0)
            add(name, "vector", mask, x, positive)
            add(name, "true_scalar", mask, 0.0, positive)
            add(name, "false_scalar", mask, x, 1.0)
            add(name, "matrix", matrix_mask, matrix, matrix + 1)
        elif name in kernels.MASK_REDUCTION_OPCODES or name == "quantile_where":
            extra = (0.37,) if name == "quantile_where" else ()
            add(name, "vector", x, mask, *extra)
            add(name, "excluded_nan", finite_gap, mask, *extra)
            add(name, "matrix", matrix, matrix_mask, *extra)
            add(name, "empty_selection", x, np.zeros_like(mask), *extra)
        elif name in {"count_true", "max_consecutive_true"}:
            add(name, "vector", mask)
            add(name, "empty", np.empty(0, dtype=np.uint8))
            if name == "count_true":
                add(name, "matrix", matrix_mask)
        elif name in kernels.REDUCTION_OPCODES:
            add(name, "vector", x)
            add(name, "matrix", matrix)
            add(name, "empty", np.empty(0))
            add(name, "ties", np.array([2.0, 1.0, 2.0, 1.0]))
            if name in {"variance", "std"}:
                add(name, "ddof_zero", x, 0.0)
                add(name, "bad_ddof", x, 8.0)
                add(name, "fractional_ddof", x, 0.5)
                add(name, "large_offset", 1e12 + np.arange(12) / 16.0)
        elif name == "quantile":
            add(name, "vector", x, 0.37)
            add(name, "matrix", matrix, 0.5)
            add(name, "probability_zero", x, 0.0)
            add(name, "probability_one", x, 1.0)
            add(name, "empty", np.empty(0), 0.5)
        elif name in kernels.SCAN_OPCODES:
            add(name, "vector", x)
            add(name, "empty", np.empty(0))
        elif name.endswith("_time") or name.endswith("_asset"):
            add(name, "matrix", matrix)
        elif name in {"lag", "difference"}:
            add(name, "default", x)
            add(name, "periods_three", x, 3.0)
            add(name, "periods_zero", x, 0.0)
            add(name, "too_long", x, float(len(x)))
            add(name, "fractional", x, 1.5)
        elif name in {"rolling_mean", "rolling_min", "rolling_max", "rolling_std"}:
            add(name, "default", x, 3.0)
            add(name, "empty", np.empty(0), 3.0)
            add(name, "long_window", x, 100.0)
            extra = (1.0, 2.0) if name == "rolling_std" else (2.0,)
            add(name, "gaps", finite_gap, 3.0, *extra)
            add(name, "nonfinite", np.array([1.0, np.inf, 2.0, -np.inf, 3.0, np.nan]), 3.0, *extra)
            add(name, "bad_window", x, 0.0)
            add(name, "fractional_window", x, 1.5)
        elif name == "recursive_smooth":
            add(name, "dense", x, 3.0, 0.5)
            add(name, "gaps", finite_gap, 3.0, 0.5)
            add(name, "empty", np.empty(0), 3.0, 0.5)
            add(name, "invalid_initial", x, 3.0, np.nan)
        elif name == "divide_or_default":
            add(name, "dense", x, positive, 0.25)
            add(
                name,
                "threshold",
                x,
                np.array([0, 1e-13, 1e-12, -1e-13, -1e-12, np.nan, np.inf, 1.0]),
                0.25,
            )
        elif name in {"first", "last", "length"}:
            add(name, "vector", x)
            add(name, "empty", np.empty(0))
        elif name in {"drawdown_series", "new_high_mask"}:
            add(name, "levels", levels)
            add(name, "plateau", np.ones(8))
            add(name, "bad_levels", x)
            add(name, "empty", np.empty(0))
        elif name == "transpose":
            add(name, "matrix", matrix)
        elif name == "diag":
            add(name, "vector", x)
            add(name, "matrix", matrix)
        elif name == "trace":
            add(name, "matrix", matrix)
        elif name in {"dot", "outer"}:
            add(name, "vectors", x, positive)
        elif name == "matmul":
            add(name, "matrix", matrix, square)
        elif name == "matvec":
            add(name, "matrix_vector", matrix, weights)
        elif name == "solve":
            add(name, "pivoted", square, weights)
            add(name, "singular", np.ones((3, 3)), weights)
        elif name in {"covariance", "correlation"}:
            add(name, "pair", x, positive[::-1].copy())
            add(name, "matrix", matrix)
            add(name, "small_pair", np.array([1.0]), np.array([2.0]))
            if name == "correlation":
                add(name, "constant_pair", np.ones(8), x)
        elif name in {
            "linear_fit",
            "linear_slope",
            "linear_intercept",
            "linear_r_squared",
            "regression_standard_error",
        }:
            add(name, "implicit_axis", x)
            add(name, "pair", positive, positive * 0.5 + x[::-1])
            add(name, "nonfinite", finite_gap)
            add(name, "small", np.array([1.0]))
        elif name in {"total_return", "cumulative_return"}:
            add(name, "vector", x)
            add(name, "empty", np.empty(0))
        elif name == "annualized_return":
            add(name, "annual", x, 252.0)
            add(name, "bad_growth", np.array([-2.0, 0.1]), 252.0)
            add(name, "bad_frequency", x, 0.0)
        elif name == "active_returns":
            add(name, "vector", x, positive)
        elif name == "portfolio_returns":
            add(name, "matrix_vector", matrix, weights)
        elif name == "quadratic_form":
            add(name, "nonsymmetric", weights, square)
        elif name == "last_drawdown_interval":
            add(name, "latest_tie", draws)
            add(name, "unrecovered", np.array([0.0, -0.2, -0.1]))
            add(name, "no_event", np.zeros(5))
            add(name, "invalid", np.array([-0.1, -0.2]))
            add(name, "empty", np.empty(0))
        elif name in {"interval_start", "interval_trough", "interval_recovery"}:
            add(name, "record", interval)
            add(name, "missing", (np.nan, np.nan, np.nan, 0.0))
        elif name.startswith("fit_"):
            add(name, "record", fit)
        elif name == "value_at":
            for pos in (0.0, 7.0, -1.0, 8.0, 0.5, np.nan):
                add(name, f"position_{pos}", x, pos)
        elif name == "days_between":
            add(name, "days", 100.0, 120.0)
            add(name, "reverse", 120.0, 100.0)
            add(name, "missing", np.nan, 120.0)
            add(name, "fraction", 100.5, 120.0)
        elif name in {"require_positive", "require_nonnegative"}:
            for value in (0.0, 1.0, -1.0, np.nan, np.inf):
                add(name, f"value_{value}", value)
        elif name == "finite_mask":
            add(name, "dense", x)
            add(name, "nonfinite", np.array([0.0, np.nan, np.inf, -np.inf, 1.0]))
        else:
            raise RuntimeError(f"Missing sample for {name}")

    # IEEE and order contracts are tested directly against the original kernels.
    ieee = np.array([np.nan, -0.0, 0.0, -np.inf, np.inf, 2.0, np.nan, -1.0])
    for name in (
        "add",
        "subtract",
        "multiply",
        "minimum",
        "maximum",
        "equal",
        "not_equal",
        "less_than",
        "less_equal",
        "greater_than",
        "greater_equal",
    ):
        add(name, "ieee_vector", ieee, ieee[::-1].copy())
    for name in (
        "negate",
        "absolute",
        "sign",
        "finite_mask",
        "sum",
        "mean",
        "std",
        "min_value",
        "max_value",
        "median",
        "cumulative_sum",
        "cumulative_max",
        "cumulative_min",
    ):
        add(name, "ieee_vector", ieee)
    for name, args in (
        ("divide", (x, 0.0)),
        ("power", (-1.0, 0.5)),
        ("sqrt", (-1.0,)),
        ("log", (0.0,)),
        ("exp", (1000.0,)),
        ("reciprocal", (0.0,)),
        ("normal_ppf", (1.0,)),
    ):
        add(name, "domain", *args)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/data/canonical_reference.json")
    args = parser.parse_args()
    source = args.source_root.resolve()
    directory = source / "backend/cal_indicators"
    names = source_names(directory)
    if len(names) != 118 or len(set(names)) != 118:
        raise RuntimeError("Source canonical set drifted; audit before updating this fixture")
    call, kernels = load_oracle(source)
    cases = []
    for name, label, values in samples(names, kernels):
        case = {"id": f"{name}:{label}", "operator": name, "args": [encode(a) for a in values]}
        try:
            case["expected"] = encode(call(name, *values))
        except ValueError as exc:
            case["error"] = str(exc)
        cases.append(case)
    if set(names) != {case["operator"] for case in cases if "expected" in case}:
        raise RuntimeError("Every canonical operator must have a successful oracle case")
    import numba

    typed_operators = importlib.import_module("cal_indicators.typed_operators")
    argument_names = typed_operators._OPERATOR_ARGUMENT_NAMES
    payload = {
        "source_repository": "BetterSaaTaa",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES
        },
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "method": "actual source NJIT dispatchers; composite names evaluated through source primitive lowering",
        "canonical_names": names,
        "parameter_names": {
            name: {
                str(arity): list(parameters)
                for (operator, arity), parameters in argument_names.items()
                if operator == name
            }
            for name in names
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        f"Captured {len(cases)} cases for {len(names)} operators; {sum('error' in c for c in cases)} expected failures"
    )


if __name__ == "__main__":
    main()
