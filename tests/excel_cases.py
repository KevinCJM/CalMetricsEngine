"""Deterministic conformance inputs; production code never imports this file."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calmetrics_engine import GraphCompiler, excel


def decode(v):
    if isinstance(v, dict):
        if "array" in v:
            return np.asarray([decode(x) for x in v["array"]], dtype=v["dtype"]).reshape(v["shape"])
        if "tuple" in v:
            return tuple(decode(x) for x in v["tuple"])
        if "scalar" in v:
            return np.dtype(v["dtype"]).type(v["scalar"])
    return float(v) if isinstance(v, str) and v in {"nan", "inf", "-inf"} else v


def cases(all_cases=False):
    seen = set()
    frozen = json.loads((Path(__file__).parent / "data/canonical_reference.json").read_text())
    for case in frozen["cases"]:
        name = case["operator"]
        if "expected" not in case or (not all_cases and name in seen):
            continue
        args = [decode(x) for x in case["args"]]
        if any(np.isinf(v).any() for v in args if isinstance(v, np.ndarray | float)):
            continue  # Explicitly refused numerical domain, tested separately.
        seen.add(name)
        yield case["id"], lambda name=name, args=args: excel.plan_operator(name, args)
    for name in ("normal_pdf", "normal_cdf"):
        yield name + ":large_finite", lambda name=name: excel.plan_operator(
            name, [np.array([-1e308, -1e200, -40., 40., 1e200, 1e308])])
    for label, body in [("root", "solve_x"), ("non_convergence", "solve_x+1")]:
        graph = GraphCompiler({"x": "series"}).compile(
            f"bisect({body},-1e308,1e308,1e-10,2)", result_format="typed", error_policy="isolate")
        yield "bisect:wide:" + label, lambda graph=graph: excel.plan(graph, {"x": np.array([1.])})
    # Formula results have a higher finite ceiling than directly entered literals.
    for index, value in enumerate([1e308, -1e308, 1.797693134862315e308]):
        yield f"formula_ceiling:mean:{index}", lambda value=value: excel.plan_operator("mean", [np.array([value])])
        graph = GraphCompiler({"x": "series"}).compile(repr(float(value)), result_format="typed")
        yield f"formula_ceiling:constant:{index}", lambda graph=graph: excel.plan(graph, {"x": np.array([1.])})
    x = np.array([1., 3., 2., 4., np.nan, 2., 3.])
    mask = np.array([1, 0, 1, 1, 1, 1, 1], np.uint8)
    reset = np.array([0, 0, 0, 0, 1, 0, 0], np.uint8)
    codes = np.array([-1, 0, 1, 1, -1, 0, 0], np.int64)
    state = np.array([[0, 0, 0], [1, 1, 0], [1, 2, 0]], np.int64)
    events = np.array([1, -1, 1], np.int64)
    segments = np.array([[0, 1], [1, 2], [-1, -1]], np.int64)
    extra = {
        "normal_cdf": [np.array([-8., -1., 0., 1., 8.])],
        "aligned_shift": [x, 2],
        "recursive_filter": [x, .2, 1., mask, 2, 1],
        "argsort": [x], "gather": [x, np.array([3, 0, 4], np.int64)],
        "distinct_count": [codes, mask], "floor": [np.array([-1.3, 0., 2.8])],
        "cos": [np.array([-2., 0., 2.])],
        "recursive_filter_adaptive": [x, np.full(7, .3), mask, reset, 0., 1, 2, 1],
        "linear_filter2": [x, .5, .1, .2, .1, reset, 2, 1],
        "scalar_kalman": [x, .1, .3],
        "state_estimate": [np.array([[1., 1.], [2., .5]])],
        "state_variance": [np.array([[1., 1.], [2., .5]])],
        "state_hysteresis": [x, 3., 2., 1., 2.],
        "state_confirm": [codes, 2, 2],
        "state_continuous": [codes, np.zeros(7, np.int64), np.ones(7), 2, 3],
        "continuous_state_values": [state], "continuous_state_evidence": [state],
        "continuous_state_pending": [state],
        "drawdown_cycle_state": [np.array([100., 80., 85., 100.]), np.array([0., -.2, -.15, 0.]), np.ones(4, np.uint8), .12, .05, .02],
        "local_extrema": [np.array([1., 3., 3., 1., 2., 1., np.nan, 2., 4., 1., 3.]), 1, 1, 0, 0],
        "ps_filter": [np.array([100., 80., 100.]), events, 2, 2, .25],
        "between_events": [events], "segment_starts": [segments], "segment_ends": [segments],
        "phase_direction": [events, segments],
        "drawdown_cycle_reference": [np.array([1, 0, -1], np.int64), np.array([-.2, .25, np.nan]), segments, .12],
        "state_select": [mask, 0., codes, mask],
    }
    for name, args in extra.items():
        yield name + ":extended", lambda name=name, args=args: excel.plan_operator(name, args)
    for name, args in [("power", [np.nan, 0.]), ("power", [np.nan, 2.]),
                       ("divide", [np.nan, 0.]), ("exp", [np.nan]),
                       ("clip", [np.array([1., 3., np.nan]), np.nan, 2.]),
                       ("min_where", [np.array([np.nan]), np.ones(1, np.uint8)]),
                       ("solve", [np.ones((2, 2)), np.ones(2)])]:
        yield name + ":domain", lambda name=name, args=args: excel.plan_operator(name, args)
    x = np.array([1., 2., 4., 3., 5., 8.])
    for name, expression in {
        "rolling_scope": "rolling_apply(mean(x)+std(x,0),3)",
        "block_scope": "block_apply(sum(x),2)",
        "filter_scope": "filter_apply(mean(x),x>2)",
        "filter_with_short_unselected_candidates": "filter_apply(mean(lag(x,2)),x>2)",
        "group_scope": "group_apply(mean(x),key)",
        "bisect_scope": "bisect(solve_x*solve_x-mean(x),0,10,1e-8,40)",
        "iteration_scope": "iterate(iterate_x*0.5,8,1e-6,30)",
        "segment_scope": "segment_apply(mean(x),between_events(local_extrema(x,1,1,0,0)))",
    }.items():
        variables = {"x": "series"}
        data = {"x": x}
        if name == "group_scope":
            variables["key"] = {"kind": "series", "dtype": "int64"}
            data["key"] = np.array([0, 1, 0, 1, 0, 1], np.int64)
        outputs = {"value": expression}
        if name == "iteration_scope":
            outputs.update({field: f"iteration_{field}({expression})" for field in ["status", "count", "residual"]})
        graph = GraphCompiler(variables).compile(outputs, result_format="typed", error_policy="isolate")
        yield name, lambda graph=graph, data=data: excel.plan(graph, data)
    # Failures are checked per position and must not poison independent roots.
    for name, expression in {
        "rolling_failure": "rolling_apply(1/std(x,0),2)>0",
        "group_failure": "group_apply(1/std(x,0),key)>0",
        "block_failure": "block_apply(1/std(x,0),2)>0",
        "scalar_failure": "sqrt(-1)+1/0",
        "scalar_failure_reversed": "1/0+sqrt(-1)",
        "mapped_healthy": "last(group_apply(1/std(x,0),key))",
        "missing_lookup": "value_at(x,99)",
        "iteration_limit": "iterate(iterate_x+1,0,1e-8,3)",
    }.items():
        variables = {"x": "series", "key": {"kind": "series", "dtype": "int64"}}
        data = {"x": np.array([1., 1., 2., 4.]), "key": np.array([0, 0, 1, 1], np.int64)}
        graph = GraphCompiler(variables).compile({"checked": expression, "healthy": "sum(x)"},
                                                result_format="typed", error_policy="isolate")
        yield name, lambda graph=graph, data=data: excel.plan(graph, data)
    for seed in range(4):
        random = np.random.default_rng(seed)
        prices = random.uniform(.5, 2., 7)
        if seed & 1:
            prices[3] = np.nan
        extrema = np.array([0, 1, -1, 1, -1, 1, 0], np.int64)
        for name, args in {
            "local_extrema": [prices, 1, 1, 0, 0],
            "ps_filter": [prices, extrema, 2, 3, .2],
            "scalar_kalman": [prices, .1, .2],
            "recursive_filter_adaptive": [prices, np.full(7, .25), np.ones(7, np.uint8),
                                          np.array([0, 0, 0, 1, 0, 0, 0], np.uint8), 0., 2, 1, 0],
        }.items():
            yield f"{name}:seed{seed}", lambda name=name, args=args: excel.plan_operator(name, args)
    graph = GraphCompiler({"x": {"kind": "matrix"},
                           "w": {"kind": "vector", "shape": ["N"]}}).compile({
        "scenario": "matvec(x,w)", "covariance": "covariance(x)",
        "mean_asset": "mean_time(x)", "positive": "matvec(x,w)>0",
        "fit_slope": "fit_slope(linear_fit(matvec(x,w)))",
        "fit_intercept": "fit_intercept(linear_fit(matvec(x,w)))",
    }, result_format="typed", error_policy="isolate")
    data = {"x": np.arange(1., 13.).reshape(4, 3)[:, ::-1], "w": np.array([.2, .3, .5])}
    yield "matrix_attribution_and_fit", lambda: excel.plan(graph, data)
    tensor = np.arange(1., 25.).reshape(2, 3, 4)
    yield "tensor_scalar_broadcast", lambda: excel.plan_operator("multiply", [tensor, .3])
