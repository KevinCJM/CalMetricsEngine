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
    yield from axis_path_cases()
    yield from projection_primitive_cases()
    yield from foundation_cases()
    yield from latent_cases()
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
    for name in ("skewness", "excess_kurtosis"):
        yield name + ":large_safe", lambda name=name: excel.plan_operator(name, [np.array([1e50, 0., -1e50, 1.])])
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


def axis_path_cases():
    """Axis recurrences and one shared NaN-last ranking for all probabilities.

    These are deterministic target-engine inputs, not recalculation evidence.
    """
    samples = {
        "product_axis": ("cumulative_product_axis", [
            np.array([[1.25, .5], [.8, 2.], [1.5, .75]]), 0, 3.]),
        "strict_max_axis": ("cumulative_max_axis", [
            np.array([[1., 1., np.nan, 2.], [-2., np.nan, -1., -1.]])[:, ::-1],
            1, -1.]),
        "multi_quantiles_axis": ("quantiles_axis", [
            np.array([[[3., 8.], [1., np.nan], [2., 4.]],
                      [[-1., 1.], [-1., 3.], [2., 2.]]]),
            np.array([1., .5, 0., .25, .5]), 1]),
    }
    for label, (name, args) in samples.items():
        yield "axis_path:" + label, lambda name=name, args=args: excel.plan_operator(name, args)


def projection_primitive_cases():
    """Initial-first contraction and stable expm1, including shared graph use."""
    samples = {
        "affine_rank2": ("affine_contract", [
            np.array([[1., 2.], [-1., .5]]),
            np.array([[.2, -.3, .1], [.4, .1, -.2]]), np.array([.01, -.02, .03])]),
        "affine_rank3": ("affine_contract", [
            np.arange(12., dtype=np.float64).reshape(2, 3, 2)[:, ::-1],
            np.array([[.2, -.3], [.4, .1]]), .01]),
        "affine_initial_order": ("affine_contract", [
            np.array([[1e16, -1e16]]), np.ones((2, 1)), 1.]),
        "affine_empty_factor": ("affine_contract", [
            np.empty((2, 0)), np.empty((0, 3)), np.array([.1, .2, .3])]),
        "affine_empty_output": ("affine_contract", [
            np.zeros((2, 3)), np.empty((3, 0)), np.empty(0)]),
        "affine_missing": ("affine_contract", [
            np.array([[np.nan, 1.], [2., 3.]]), np.ones((2, 1)), .1]),
        "expm1_domain": ("expm1", [np.array([
            -1000., -40., -.13, -.125, -1e-16, -0., 0., 1e-16, .125, .13, 1., 700., np.nan])]),
        "expm1_scalar_tiny": ("expm1", [1e-16]),
        "expm1_empty": ("expm1", [np.empty((0, 2))]),
    }
    for label, (name, args) in samples.items():
        yield "projection:" + label, lambda name=name, args=args: excel.plan_operator(name, args)
    graph = GraphCompiler({
        "x": {"kind": "tensor", "axes": ["time", "path", "factor"], "shape": ["T", "P", "F"]},
        "k": {"kind": "matrix", "axes": ["factor", "asset"], "shape": ["F", "A"]},
        "i": {"kind": "vector", "axes": ["asset"], "shape": ["A"]},
        "w": {"kind": "vector", "axes": ["asset"], "shape": ["A"]},
    }).compile({
        "period": "sum_axis(multiply(broadcast_like(w,expm1(affine_contract(x,k,i))),expm1(affine_contract(x,k,i))),2)",
        "asset": "sum(multiply(first(slice_axis(w,0,0,1,1)),slice_axis(expm1(affine_contract(x,k,i)),2,0,1,1)))/3",
    }, result_format="typed")
    data = {"x": np.arange(12., dtype=np.float64).reshape(2, 3, 2) * .01,
            "k": np.array([[.2, -.3], [.4, .1]]), "i": np.array([.01, -.02]), "w": np.array([.3, .7])}
    yield "projection:shared_graph", lambda: excel.plan(graph, data)


def foundation_cases():
    """All matrix/array recipes plus domain and composed-graph cases."""
    a = np.array([[4., 1., 2.], [1., 3., 0.], [2., 0., 5.]])
    b = np.arange(6., dtype=np.float64).reshape(3, 2)
    bundle = np.array([[1., 1., 0.], [2., 0., 1.]])
    samples = {
        "cholesky": [a], "solve_multi": [a, b],
        "triangular_solve": [np.tril(a), b], "symmetric_eigh": [a],
        "eigenvalues": [bundle], "eigenvectors": [bundle],
        "take_axis": [b, np.array([2, 0], np.int64), 0],
        "slice_axis": [b, 0, 0, 3, 2], "concatenate": [b, b, 1],
        "stack": [b, b, 1], "broadcast_like": [np.arange(2.), b],
        "sum_axis": [b, 0], "mean_axis": [b, 1], "swap_axes": [b, 0, 1],
    }
    for name, args in samples.items():
        yield "foundation:" + name, lambda name=name,args=args: excel.plan_operator(name,args)
    for label, name, args in [
        ("singular", "solve_multi", [np.ones((2, 2)), np.ones((2, 2))]),
        ("indefinite", "cholesky", [np.array([[1., 2.], [2., 1.]])]),
        ("asymmetric", "symmetric_eigh", [np.array([[1., 2.], [1., 3.]])]),
        ("no_convergence", "symmetric_eigh", [a, 1e-15, 1]),
        ("upper_unit", "triangular_solve", [np.array([[np.nan, 2.], [np.nan, np.nan]]), np.array([3., 4.]), 0, 1]),
        ("mean_cancellation", "mean_axis", [np.array([[1e16, 1., -1e16]]), 1]),
        ("mean_nan", "mean_axis", [np.array([[np.nan, 1.], [2., 3.]]), 1]),
        ("empty_mean", "mean_axis", [np.empty((2, 0)), 1]),
        ("empty_sum", "sum_axis", [np.empty((2, 0)), 1]),
        ("empty_eigh", "symmetric_eigh", [np.empty((0, 0))]),
        ("repeat_eigh", "symmetric_eigh", [np.eye(3)]),
        ("mask_view", "slice_axis", [np.array([[True, False], [False, True]]), 1, 0, 2, 1]),
        ("int_view", "swap_axes", [np.arange(6, dtype=np.int64).reshape(2, 3), 0, 1]),
    ]:
        yield "foundation:" + label, lambda name=name,args=args: excel.plan_operator(name,args)
    g = GraphCompiler({"a": {"kind":"matrix", "axes":["asset","asset"],"shape":["N","N"]}}).compile({
        "values":"eigenvalues(symmetric_eigh(a))", "vectors":"eigenvectors(symmetric_eigh(a))",
        "sum":"sum_axis(slice_axis(a,1,0,3,2),0)"}, result_format="typed", error_policy="isolate")
    yield "foundation:composed", lambda: excel.plan(g,{"a":a})


def latent_cases():
    """Every latent opcode, iteration stop/budget, edge cases and composed graph."""
    from calmetrics_engine import operators as op
    x=np.array([[-1.,-.7],[-.8,-1.],[.9,1.],[1.2,.8]])
    w=np.array([.5,.5])
    m=np.array([[-1.,-1.],[1.,1.]])
    v=np.ones((2,2))
    transition=np.array([[.9,.1],[.1,.9]])
    gm=op.gaussian_mixture_em(x,w,m,v,4)
    hm=op.gaussian_hmm_em(x,w,transition,m,v,4)
    emission=op.gaussian_log_density(x,m,v)
    args={
        "variance_axis":[x,0],"max_axis":[x,1],"argmax_axis":[x,1],
        "logsumexp_axis":[x,1],"softmax_axis":[x,1],
        "gaussian_log_density":[x,m,v],"gaussian_mixture_em":[x,w,m,v,4],
        "gaussian_hmm_em":[x,w,transition,m,v,4],
        "mixture_weights":[gm],"mixture_means":[gm],"mixture_variances":[gm],
        "em_iterations":[gm],"em_log_likelihood":[hm],"em_converged":[hm],
        "hmm_initial":[hm],"hmm_transition":[hm],"hmm_means":[hm],"hmm_variances":[hm],
        "hmm_filter":[emission,w,transition,1],"hmm_smooth":[emission,w,transition],
        "transition_counts":[np.array([0,0,-1,1,0],np.int64),2],
        "one_hot":[np.array([0,0,-1,1,0],np.int64),2],
        "partition_mean":[x,np.array([1,0,2,3],np.int64),2],
        "sample_unique_indices":[x,4,-3],
        "scatter_rows":[x,np.array([0,2,3,5],np.int64),6],
    }
    for name,values in args.items():
        yield "latent:"+name, lambda name=name,values=values: excel.plan_operator(name,values)
    for seed in [2**53-1,-2**53+1,2147483646*4194304-1,-(2147483646*4194304-1)]:
        # Seeds use the float64 parameter contract; generic Excel int64 literals
        # retain their existing 15-digit input restriction.
        yield f"latent:sample_unique_indices:seed:{seed}", lambda seed=seed: excel.plan_operator("sample_unique_indices",[np.zeros((17,2)),17,float(seed)])
    for name,values in [
        ("gaussian_mixture_em",[np.array([[-1e154],[0.],[1e154]]),np.ones(1),np.zeros((1,1)),np.full((1,1),1e308),1]),
        ("gaussian_hmm_em",[np.array([[-1e154],[0.],[1e154]]),np.ones(1),np.ones((1,1)),np.zeros((1,1)),np.full((1,1),1e308),1]),
    ]:
        yield "latent:"+name+":variance_overflow",lambda name=name,values=values: excel.plan_operator(name,values)
    for iterations,tolerance in [(1,0.),(5,1e10),(5,0.)]:
        for name,values in [("gaussian_mixture_em",[x,w,m,v]),("gaussian_hmm_em",[x,w,transition,m,v])]:
            yield f"latent:{name}:stop:{iterations}:{tolerance}", lambda name=name,values=values,iterations=iterations,tolerance=tolerance: excel.plan_operator(name,values+[iterations,tolerance])
    types={"x":{"kind":"matrix","axes":["scenario","factor"],"shape":["N","D"]},
           "m":{"kind":"matrix","axes":["component","factor"],"shape":["K","D"]},
           "v":{"kind":"matrix","axes":["component","factor"],"shape":["K","D"]},
           "w":{"kind":"vector","axes":["component"],"shape":["K"]}}
    fit="gaussian_mixture_em(x,w,m,v,4)"
    graph=GraphCompiler(types).compile([f"mixture_means({fit})",f"em_iterations({fit})",f"em_converged({fit})",f"argmax_axis(gaussian_log_density(x,mixture_means({fit}),mixture_variances({fit})),1)"],result_format="typed",error_policy="isolate")
    yield "latent:dag",lambda: excel.plan(graph,{"x":x,"m":m,"v":v,"w":w})
