"""Editable complete fixed-tape mathematical consumer for opcodes 188..190.

The small scalar-loop oracle is frozen here, without an upstream import or file
read. It follows the readable projection/NAV/fan arithmetic trees. This tests
the successful numeric chain, not old-seed generation, coverage/limits, JSON,
or the upstream stage-ordered first-fault/partial-array protocol. No Python
helper below participates in native runtime planning or numerical execution.
"""

import gc
import math
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

FROZEN_SOURCE_SHA256 = "bb53056653d697c698dab4ac5e716d96dbaec9071157d755c3fbef32e9831b79"
METRIC_FIELDS = (
    "terminal_nav_p05", "terminal_nav_p25", "terminal_nav_p50",
    "terminal_nav_p75", "terminal_nav_p95", "terminal_return_p05",
    "terminal_return_p50", "terminal_return_p95", "var_95", "es_95",
    "loss_probability", "target_hit_probability", "average_max_drawdown",
    "p95_max_drawdown",
)


def probability_inputs():
    values = {
        "q5": np.array([.05, .25, .5, .75, .95], np.float64),
        "qreturn": np.array([.05, .5, .95], np.float64),
        "qdd": np.array([.95], np.float64),
    }
    for array in values.values():
        array.flags.writeable = False
    return values


def declaration(kind, axes=None, shape=None, semantic="dimensionless"):
    out = dict(kind=kind, dtype="float64", semantic_dimension=semantic)
    if axes is not None:
        out.update(axes=axes, shape=shape)
    return out


class FormulaBindings:
    """Test/business template aliases; the C++ compiler builds the actual DAG."""

    def __init__(self):
        self.formulas = {}
        self.dependencies = {}

    def add(self, name, expression, *dependencies):
        self.formulas[name] = expression
        self.dependencies[name] = dependencies
        return name

    def reachable(self, name):
        # Dependencies are declared with the template, not parsed from AST text.
        found = {}

        def visit(alias):
            if alias in found:
                return
            for parent in self.dependencies[alias]:
                visit(parent)
            found[alias] = self.formulas[alias]

        visit(name)
        return found


def tail_bindings(formulas, returns="terminal_return", boundary="q05"):
    """Existing scalar where is eager: sentinel and safe denominator are needed."""
    f = formulas.add
    f("tail_mask", f"less_equal({returns},{boundary})", returns, boundary)
    f("tail_head", f"slice_axis({returns},0,0,1,1)", returns)
    f("tail_zero", "where(finite_mask(tail_head),0,0)", "tail_head")
    f("tail_true", "equal(tail_zero,tail_zero)", "tail_zero")
    f("tail_values", f"concatenate({returns},tail_zero,0)", returns, "tail_zero")
    f("tail_selected", "concatenate(tail_mask,tail_true,0)", "tail_mask", "tail_true")
    f("tail_total", "sum_where(tail_values,tail_selected)", "tail_values", "tail_selected")
    f("tail_count", "count_true(tail_mask)", "tail_mask")
    f("has_tail", "greater_than(tail_count,0)", "tail_count")
    f("safe_tail_count", "where(has_tail,tail_count,1)", "has_tail", "tail_count")
    f("var", f"maximum(0,negate({boundary}))", boundary)
    f("es", "where(has_tail,maximum(0,negate(divide(tail_total,safe_tail_count))),var)",
      "has_tail", "tail_total", "safe_tail_count", "var")


def compact_spec(steps, paths, assets=0, combinations=1, *, source="tape",
                 initial_semantic="dimensionless"):
    """Real restricted-AST roots and only their reachable expression bindings.

    q vectors are declared inputs, never unsupported list literals. T and S
    are literal geometry of this template; rebuild it when these sizes change.
    P>=1 and T>=1 are this successful consumer's domain, not new engine limits.
    """
    if steps < 1 or paths < 1 or assets < 0 or combinations < 1:
        raise ValueError("compact consumer needs T>=1, P>=1, A>=0 and C>=1")
    if source not in {"period", "tape"}:
        raise ValueError("source must be period or tape")
    variables = {
        "q5": declaration("vector", ["component"], ["5"]),
        "qreturn": declaration("vector", ["component"], ["3"]),
        "qdd": declaration("vector", ["component"], ["1"]),
        "n0": declaration("scalar", semantic=initial_semantic),
        "target": declaration("scalar"),
    }
    if source == "tape":
        variables.update({
            "x": declaration("tensor", ["time", "path", "factor"], ["T", "P", "F"]),
            "beta": declaration("matrix", ["asset", "factor"], ["A", "F"]),
            "initial": declaration("vector", ["asset"], ["A"]),
        })
    expressions, bindings, chains = {}, [], []
    sample_count = min(paths, 12)
    for combination in range(combinations):
        f = FormulaBindings()
        add = f.add
        if source == "tape":
            weights = f"w{combination}"
            variables[weights] = declaration("vector", ["asset"], ["A"])
            add("simple", "expm1(affine_contract(x,transpose(beta),initial))")
            add("weighted", f"multiply(broadcast_like({weights},simple),simple)", "simple")
            add("period", "sum_axis(weighted,2)", "weighted")
        else:
            period_input = f"period{combination}"
            variables[period_input] = declaration("matrix", ["time", "path"], ["T", "P"])
            add("period", period_input)
        add("nav", "cumulative_product_axis(add(1,period),0,n0)", "period")
        add("peak", "cumulative_max_axis(nav,0,n0)", "nav")
        add("drawdown", "subtract(1,divide(nav,peak))", "nav", "peak")
        add("max_dd", "max_axis(drawdown,0)", "drawdown")
        add("first_time", "slice_axis(nav,0,0,1,1)", "nav")
        add("seed", "where(finite_mask(first_time),n0,n0)", "first_time")
        add("full_nav", "concatenate(seed,nav,0)", "seed", "nav")
        add("fan", "swap_axes(quantiles_axis(full_nav,q5,1),0,1)", "full_nav")
        add("samples", f"swap_axes(slice_axis(full_nav,1,0,{sample_count},1),0,1)", "full_nav")
        add("terminal_nav", f"sum_axis(slice_axis(nav,0,{steps - 1},{steps},1),0)", "nav")
        add("terminal_return", "subtract(divide(terminal_nav,n0),1)", "terminal_nav")
        add("return_quantiles", "quantiles_axis(terminal_return,qreturn,0)", "terminal_return")
        add("dd_quantiles", "quantiles_axis(max_dd,qdd,0)", "max_dd")
        add("terminal_fan", f"sum_axis(slice_axis(fan,1,{steps},{steps + 1},1),1)", "fan")
        add("q05", "first(slice_axis(return_quantiles,0,0,1,1))", "return_quantiles")
        tail_bindings(f)
        metric_aliases = []
        for index in range(5):
            name = f"metric{index}"
            add(name, f"first(slice_axis(terminal_fan,0,{index},{index + 1},1))", "terminal_fan")
            metric_aliases.append(name)
        for index in range(3):
            name = f"metric{index + 5}"
            add(name, f"first(slice_axis(return_quantiles,0,{index},{index + 1},1))", "return_quantiles")
            metric_aliases.append(name)
        metric_aliases.extend(("var", "es"))
        add("loss", "divide(count_true(less_than(terminal_return,0)),length(terminal_return))", "terminal_return")
        add("hit", "divide(count_true(greater_equal(terminal_return,target)),length(terminal_return))", "terminal_return")
        add("average_dd", "divide(sum(max_dd),length(terminal_return))", "max_dd", "terminal_return")
        add("dd95", "first(dd_quantiles)", "dd_quantiles")
        metric_aliases.extend(("loss", "hit", "average_dd", "dd95"))

        def root(name, alias, formulas=f, output_expressions=expressions, output_bindings=bindings):
            output_expressions[name] = alias
            output_bindings.append(formulas.reachable(alias))

        root(f"fan_{combination}", "fan")
        root(f"samples_{combination}", "samples")
        for field, alias in zip(METRIC_FIELDS, metric_aliases, strict=True):
            root(f"stat_{combination}_{field}", alias)
        if source == "tape":
            for asset in range(assets):
                alias = f"contribution{asset}"
                weight = f"first(slice_axis(w{combination},0,{asset},{asset + 1},1))"
                values = f"slice_axis(simple,2,{asset},{asset + 1},1)"
                add(alias, f"divide(sum(multiply({weight},{values})),{paths})", "simple")
                root(f"contribution_{combination}_{asset}", alias)
        chains.append(f.formulas.copy())
    return dict(variables=variables, expressions=expressions, root_bindings=bindings,
                chains=chains, steps=steps, paths=paths, assets=assets,
                combinations=combinations, source=source, sample_count=sample_count)


def compile_compact(spec, *, order="C", error_policy="raise"):
    return GraphCompiler(spec["variables"]).compile(
        spec["expressions"], root_bindings=spec["root_bindings"],
        result_format="typed", output_order=order, error_policy=error_policy)


def reference_projection(x, beta, initial, weights):
    """The source's initial-first coefficient-first ascending F, then T/P/A."""
    steps, paths, factors = x.shape
    period = np.empty((paths, steps), np.float64)
    contribution = np.zeros(beta.shape[0], np.float64)
    with np.errstate(all="ignore"):
        for step in range(steps):
            for path in range(paths):
                portfolio_return = np.float64(0.)
                for asset in range(beta.shape[0]):
                    log_return = np.float64(initial[asset])
                    for factor in range(factors):
                        product = beta[asset, factor] * x[step, path, factor]
                        log_return = np.float64(log_return + product)
                    try:
                        simple = np.float64(math.expm1(float(log_return)))
                    except OverflowError:
                        simple = np.float64(np.inf)
                    value = weights[asset] * simple
                    portfolio_return = np.float64(portfolio_return + value)
                    contribution[asset] += value
                if not np.isfinite(portfolio_return) or portfolio_return <= -1:
                    raise ValueError("reference fixture is outside the successful projection domain")
                period[path, step] = portfolio_return
        contribution /= paths
    return period, contribution


def reference_nav(period, initial_nav):
    paths, steps = period.shape
    nav = np.empty((paths, steps + 1), np.float64)
    terminal, drawdowns = np.empty(paths), np.zeros(paths)
    with np.errstate(all="ignore"):
        for path in range(paths):
            current = peak = np.float64(initial_nav)
            nav[path, 0] = current
            for step in range(steps):
                value = period[path, step]
                if not np.isfinite(value) or value <= -1:
                    raise ValueError("reference fixture is outside the successful NAV input domain")
                current = np.float64(current * np.float64(1 + value))
                if not np.isfinite(current) or current <= 0:
                    raise ValueError("reference fixture is outside the successful NAV state domain")
                nav[path, step + 1] = current
                if current > peak:
                    peak = current
                drawdown = np.float64(1 - current / peak)
                if drawdown > drawdowns[path]:
                    drawdowns[path] = drawdown
            terminal[path] = np.float64(current / np.float64(initial_nav) - 1)
    return nav, terminal, drawdowns


def reference_quantile(sorted_values, probability):
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    position = np.float64(probability) * (count - 1)
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = np.float64(position - lower)
    return np.float64(sorted_values[lower] * np.float64(1 - fraction)
                      + sorted_values[upper] * fraction)


def reference_fan(nav, terminal, drawdowns, target):
    paths, count = nav.shape
    probabilities = (.05, .25, .5, .75, .95)
    fan, metrics = np.empty((5, count)), np.empty(14)
    with np.errstate(all="ignore"):
        for step in range(count):
            ordered = np.sort(nav[:, step])
            for index, q in enumerate(probabilities):
                fan[index, step] = reference_quantile(ordered, q)
        ordered_returns, ordered_dd = np.sort(terminal), np.sort(drawdowns)
        ordered_nav = np.sort(nav[:, -1])
        for index, q in enumerate(probabilities):
            metrics[index] = reference_quantile(ordered_nav, q)
        for index, q in enumerate((.05, .5, .95)):
            metrics[index + 5] = reference_quantile(ordered_returns, q)
        boundary = metrics[5]
        total, dd_total = np.float64(0.), np.float64(0.)
        tail_count = loss_count = hit_count = 0
        for path in range(paths):
            value = terminal[path]
            if value <= boundary:
                total = np.float64(total + value)
                tail_count += 1
            if value < 0:
                loss_count += 1
            if value >= target:
                hit_count += 1
            dd_total = np.float64(dd_total + drawdowns[path])
        metrics[8] = max(0., -boundary)
        metrics[9] = max(0., -(total / tail_count)) if tail_count else metrics[8]
        metrics[10], metrics[11] = loss_count / paths, hit_count / paths
        metrics[12] = dd_total / paths
        metrics[13] = reference_quantile(ordered_dd, .95)
    return fan, metrics


def expected_named(spec, inputs, *, n0=1., target=.1):
    expected = {}
    for combination in range(spec["combinations"]):
        if spec["source"] == "tape":
            period, contributions = reference_projection(
                inputs["x"], inputs["beta"], inputs["initial"], inputs[f"w{combination}"])
        else:
            period = inputs[f"period{combination}"].T
            contributions = ()
        nav, terminal, drawdowns = reference_nav(period, n0)
        fan, metrics = reference_fan(nav, terminal, drawdowns, target)
        expected[f"fan_{combination}"] = fan
        expected[f"samples_{combination}"] = nav[:spec["sample_count"]].copy()
        for index, field in enumerate(METRIC_FIELDS):
            expected[f"stat_{combination}_{field}"] = np.asarray(metrics[index])
        for asset, contribution in enumerate(contributions):
            expected[f"contribution_{combination}_{asset}"] = np.asarray(contribution)
    assert tuple(expected) == tuple(spec["expressions"])
    return expected


def ieee_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))


def compare_named(result, expected, *, exact=False, row=0):
    assert tuple(result.named_outputs) == tuple(expected)
    for name, target in expected.items():
        out = result.named_outputs[name]
        actual = out.values[row]
        assert actual.shape == target.shape and actual.dtype == np.float64
        assert not actual.flags.writeable
        assert out.root_statuses[row] == 0 and out.shape_known[row]
        if exact or name.endswith(("loss_probability", "target_hit_probability")):
            ieee_bits(actual, target)
        else:
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(target))
            np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(target))
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(target))
            np.testing.assert_allclose(actual, target, rtol=2e-14, atol=2e-16, equal_nan=True)


def layout_view(value, layout):
    value = np.asarray(value, dtype=np.float64)
    if layout == "F":
        value = np.array(value, order="F")
    elif layout == "strided":
        owner = np.full(tuple(2 * n for n in value.shape), -731.)
        index = (slice(None, None, 2),) * value.ndim
        owner[index] = value
        value = owner[index]
    elif layout == "negative":
        index = (slice(None, None, -1),) * value.ndim
        value = value[index].copy()[index]
    else:
        value = value.copy()
    value.flags.writeable = False
    return value


def small_tape(steps=3, paths=7, factors=2, assets=3, combinations=2, layout="C"):
    x = (np.arange(steps * paths * factors, dtype=np.float64).reshape(steps, paths, factors) % 17 - 8) / 256
    beta = (np.arange(assets * factors, dtype=np.float64).reshape(assets, factors) % 7 - 3) / 32
    initial = np.arange(assets, dtype=np.float64) / 2048
    values = dict(x=x, beta=beta, initial=initial)
    for combination in range(combinations):
        values[f"w{combination}"] = ((np.arange(assets, dtype=np.float64) + 1)
                                     * (-1 if combination % 2 else 1) / max(assets, 1))
    values = {name: layout_view(value, layout) for name, value in values.items()}
    values.update(probability_inputs())
    return values


def run_twice(spec, inputs, *, n0=1., target=.1, mode="ordinary", order="C", exact=False):
    graph = compile_compact(spec, order=order)
    expected = expected_named(spec, inputs, n0=n0, target=target)
    originals = {name: value.copy() for name, value in inputs.items()}
    starts, ends = np.array([0], np.int64), np.array([spec["steps"]], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        parameters = dict(n0=float(n0), target=float(target))
        if mode == "prepared":
            prepared = scheduler.prepare_execution(graph, inputs, starts, ends, parameters=parameters)
            call = prepared.run_snapshot
        else:
            def call(native_graph=graph, native_inputs=inputs, begin=starts, stop=ends,
                     native_parameters=parameters):
                return scheduler.execute(native_graph, native_inputs, begin, stop,
                                         parameters=native_parameters)
        first = call()
        saved = {name: out.values[0].copy() for name, out in first.named_outputs.items()}
        second = call()
        for result in (first, second):
            compare_named(result, expected, exact=exact)
            assert result.audit["result_lifetime"] == "independent"
            assert result.audit["python_fallback"] == result.audit["python_operator_calls"] == 0
            assert result.audit["request_time_compilation"] == 0
            for out in result.outputs:
                value = out.values[0]
                assert all(not np.shares_memory(value, source) for source in inputs.values())
                if value.ndim == 2:
                    assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
        for name, snapshot in saved.items():
            old, new = first.named_outputs[name].values[0], second.named_outputs[name].values[0]
            ieee_bits(old, snapshot)
            assert not np.shares_memory(old, new)
    for name, original in originals.items():
        ieee_bits(inputs[name], original)
    del second, call, graph
    gc.collect()
    for name, snapshot in saved.items():
        ieee_bits(first.named_outputs[name].values[0], snapshot)
    return first, expected


@pytest.mark.parametrize("paths,n0", [(1, .1), (7, .1), (11, 17.3), (12, .1), (13, 17.3), (21, 1.)])
@pytest.mark.parametrize("layout,mode,order", [("C", "ordinary", "C"), ("F", "prepared", "F"),
                                               ("negative", "ordinary", "F"), ("strided", "prepared", "C")])
def test_period_nav_fan_all_statistics_nonunit_initial_and_sample_count(paths, n0, layout, mode, order):
    period = (np.arange(4 * paths, dtype=np.float64).reshape(4, paths) % 9 - 4) / 64
    inputs = dict(period0=layout_view(period, layout), **probability_inputs())
    spec = compact_spec(4, paths, source="period")
    result, expected = run_twice(spec, inputs, n0=n0, mode=mode, order=order, exact=True)
    assert result.named_outputs["samples_0"].values[0].shape == (min(paths, 12), 5)
    # The initial fan is quantile interpolation of equal N0s, not a filled N0
    # literal; nonunit values may retain a rounding difference from that fill.
    ieee_bits(result.named_outputs["fan_0"].values[0][:, 0], expected["fan_0"][:, 0])


@pytest.mark.parametrize("factors,assets", [(2, 3), (0, 3), (2, 0), (0, 0)])
@pytest.mark.parametrize("layout,mode,order", [("C", "ordinary", "C"), ("F", "prepared", "F"),
                                               ("negative", "ordinary", "F"), ("strided", "prepared", "C")])
def test_fixed_tape_shared_complete_compact_roots_and_empty_model_axes(factors, assets, layout, mode, order):
    inputs = small_tape(factors=factors, assets=assets, layout=layout)
    spec = compact_spec(3, 7, assets, 2)
    graph = compile_compact(spec)
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    run_twice(spec, inputs, n0=13.2, mode=mode, order=order)


@pytest.mark.parametrize("mode,order", [("ordinary", "C"), ("prepared", "F")])
def test_terminal_infinity_is_preserved_at_integer_quantile_rank(mode, order):
    # NAV stays finite/positive; only terminal NAV/N0 overflows. Source imposes
    # no additional terminal finite gate, and integer-rank quantile returns Inf.
    period = np.full((2, 21), 1e155)
    inputs = dict(period0=layout_view(period, "C"), **probability_inputs())
    spec = compact_spec(2, 21, source="period")
    result, _ = run_twice(spec, inputs, n0=np.finfo(np.float64).tiny,
                          mode=mode, order=order, exact=True)
    for field in ("terminal_return_p05", "terminal_return_p50", "terminal_return_p95"):
        assert np.isposinf(result.named_outputs[f"stat_0_{field}"].values[0])
    assert result.named_outputs["stat_0_es_95"].values[0] == 0.


def test_zero_return_ties_include_all_paths_and_use_source_probability_thresholds():
    inputs = dict(period0=np.zeros((3, 21)), **probability_inputs())
    spec = compact_spec(3, 21, source="period")
    result, _ = run_twice(spec, inputs, n0=.1, target=0., exact=True)
    assert result.named_outputs["stat_0_loss_probability"].values[0] == 0.
    assert result.named_outputs["stat_0_target_hit_probability"].values[0] == 1.
    assert result.named_outputs["stat_0_es_95"].values[0] == 0.


def test_heterogeneous_statistic_units_are_separate_named_roots():
    inputs = dict(period0=np.zeros((2, 7)), **probability_inputs())
    spec = compact_spec(2, 7, source="period", initial_semantic="currency_amount")
    graph = compile_compact(spec)
    root_types = dict(zip(spec["expressions"], graph.metadata()["root_types"], strict=True))
    assert root_types["stat_0_terminal_nav_p05"]["semantic_dimension"] == "currency_amount"
    assert root_types["stat_0_terminal_return_p05"]["semantic_dimension"] == "dimensionless"
    run_twice(spec, inputs, n0=17.3, exact=True)


@pytest.mark.parametrize("values,boundary", [([0., 0., 0.], -1.),
                                           ([-.1, -.1, 0., .2], -.1),
                                           ([-1e16, -1., 1e16, -1.], 1e16),
                                           ([-0., 0., 0.], 0.)])
def test_es_empty_tail_sentinel_eager_safe_denominator_and_raw_path_fold(values, boundary):
    f = FormulaBindings()
    f.add("terminal_return", "r")
    f.add("q05", "boundary")
    tail_bindings(f)
    graph = GraphCompiler({"r": declaration("vector", ["path"], ["P"]),
                           "boundary": declaration("scalar")}).compile(
        {"es": "es", "count": "tail_count"},
        root_bindings=[f.reachable("es"), f.reachable("tail_count")], result_format="typed")
    total, count = np.float64(0.), 0
    for value in np.asarray(values, np.float64):
        if value <= boundary:
            total = np.float64(total + value)
            count += 1
    expected = max(0., -total / count) if count else max(0., -boundary)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"r": np.asarray(values, np.float64)},
                                   np.array([0], np.int64), np.array([1], np.int64),
                                   parameters={"boundary": float(boundary)})
    ieee_bits(result.named_outputs["es"].values[0], np.asarray(expected))
    assert result.named_outputs["count"].values[0] == count
    assert all(out.root_statuses[0] == 0 for out in result.outputs)


def test_full_tape_contribution_keeps_step_path_fold_without_subtotals():
    # Large opposite asset weights cancel each period, keeping the numeric NAV
    # chain valid. The upstream definition/weight-limit gate is outside scope.
    inputs = dict(x=np.array([[[math.log1p(1.)], [1e-16]], [[-40.], [1e-16]]]),
                  beta=np.ones((2, 1)), initial=np.zeros(2), w0=np.array([1e16, -1e16]),
                  **probability_inputs())
    spec = compact_spec(2, 2, 2, 1)
    result, _ = run_twice(spec, inputs, exact=True)
    ieee_bits(result.named_outputs["contribution_0_0"].values[0], np.asarray(.5))
    ieee_bits(result.named_outputs["contribution_0_1"].values[0], np.asarray(-.5))


def test_isolated_bad_return_probability_does_not_break_independent_compact_roots():
    inputs = small_tape(combinations=1)
    inputs["qreturn"] = np.array([1.5, .5, .95])
    spec = compact_spec(3, 7, 3, 1)
    graph = compile_compact(spec, error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, np.array([0], np.int64),
                                   np.array([3], np.int64), parameters={"n0": 1., "target": .1})
    failed = {f"stat_0_{field}" for field in METRIC_FIELDS[5:10]}
    expected = expected_named(spec, inputs)
    for name, out in result.named_outputs.items():
        if name in failed:
            assert out.root_statuses[0] != 0
        else:
            assert out.root_statuses[0] == 0
            np.testing.assert_allclose(out.values[0], expected[name], rtol=2e-14, atol=2e-16)


def test_compact_graph_pickle_preserves_bound_formulas_and_multi_shape_protocol():
    inputs = small_tape(combinations=1)
    spec = compact_spec(3, 7, 3, 1)
    graph = compile_compact(spec, order="F")
    restored = pickle.loads(pickle.dumps(graph))
    assert graph.fingerprint == restored.fingerprint
    starts, ends = np.array([0], np.int64), np.array([3], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(restored, inputs, starts, ends,
                                   parameters={"n0": 1., "target": .1})
    compare_named(result, expected_named(spec, inputs))


@pytest.mark.parametrize("steps,paths", [(0, 3), (3, 0)])
def test_compact_template_rejects_empty_axis_outside_its_source_success_domain(steps, paths):
    # Generic empty-axis operator contracts belong to primitive tests. This
    # consumer cannot silently invent samples/N0 shape or a zero-path quantile.
    with pytest.raises(ValueError, match="T>=1, P>=1"):
        compact_spec(steps, paths, source="period")
