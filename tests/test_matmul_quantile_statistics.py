"""Private matrix quantiles and row statistics: shared bounded physical path."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle

from .test_matmul_statistics import LIBRARY, count, fixture
from .test_typed_results import config

TYPES = {
    key: dict(kind="matrix", axes=["asset", "asset"], shape=shape)
    for key, shape in dict(w=["N", "K"], r=["T", "K"]).items()
}
TYPES["q"] = dict(kind="vector", axes=["asset"], shape=["Q"])
Y = "matmul(w,transpose(r))"
ROOTS = [f"mean_axis({Y},1)*252", f"sqrt(variance_axis({Y},1,1)*252)", f"-quantiles_axis({Y},q,1)"]


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("kind", ["quantile", "mean", "sum", "variance"])
@pytest.mark.parametrize("chain", ["overflow", "subnormal"])
@pytest.mark.parametrize("scalar_first", [False, True])
def test_intermediate_scale_preserves_finiteness_and_owned_results(
    layout, kind, chain, scalar_first
):
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1.0, (65, 7))
    w /= w.sum(axis=1, keepdims=True)
    values = dict(
        w=np.array(w, order=layout),
        r=np.array(rng.uniform(1.75, 1.85, (63, 7)), order=layout),
        q=np.array([0.5]),
    )
    bases = dict(
        quantile=f"quantiles_axis({Y},q,1)",
        mean=f"mean_axis({Y},1)",
        sum=f"sum_axis({Y},1)",
        variance=f"variance_axis({Y},1,1)",
    )
    expression = bases[kind]
    factors = (
        ["9.982128125599782e307", "1e-300", "1e-8"]
        if chain == "overflow"
        else ["1e-300", "1e-20", "1e300", "1e20"]
    )
    for factor in factors:
        expression = f"({factor}*({expression}))" if scalar_first else f"(({expression})*{factor})"
    for source in values.values():
        source.flags.writeable = False
    g = graph([expression], output_order=layout)
    control = graph([expression, Y], output_order=layout)
    first = run(g, values)
    saved = [value.copy() for value in first.outputs[0].values]
    second = run(g, values)
    reference = run(control, values)
    compare(first, reference, 1, 0.0)
    assert count(first, "accelerate_matmul_calls") == 0
    assert count(first, "covariance_syrk_calls") == 0
    assert count(first, "numeric_arena_bytes") < 65 * 63 * 8
    for old, new, snapshot in zip(
        first.outputs[0].values, second.outputs[0].values, saved, strict=True
    ):
        np.testing.assert_array_equal(old, snapshot)
        assert not np.shares_memory(old, new)
        if old.ndim == 2:
            assert old.flags.c_contiguous if layout == "C" else old.flags.f_contiguous
        for source in values.values():
            assert not np.shares_memory(old, source)


@pytest.mark.parametrize("layout", ["C", "F"])
def test_one_extreme_tail_preserves_other_public_quantile_branches(layout):
    values = data(layout=layout)
    qt = f"quantiles_axis({Y},q,1)"
    roots = [qt, f"(({qt}*9.982128125599782e307)*1e-300)*1e-8", ROOTS[0], ROOTS[1]]
    first = run(graph(roots, output_order=layout), values)
    reference = run(graph([*roots, Y], output_order=layout), values)
    compare(first, reference, 4, 0.0)
    assert count(first, "accelerate_matmul_calls") == 0
    assert count(first, "numeric_arena_bytes") < 65 * 63 * 8


def data(n=65, t=63, k=7, layout="C"):
    return dict(fixture(n, t, k, layout), q=np.array([1.0, 0.05, 0.5, 0.0, 0.5]))


def graph(roots=ROOTS, **kw):
    return GraphCompiler(TYPES).compile(roots, result_format="typed", **kw)


def run(g, values, lane="single", **kw):
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        return scheduler.execute(g, values, bounds, bounds, **kw)


def compare(a, b, roots=3, atol=4e-5):
    for actual, want in zip(a.outputs, b.outputs[:roots], strict=True):
        np.testing.assert_array_equal(actual.root_statuses, want.root_statuses)
        for value, expected in zip(actual.values, want.values, strict=True):
            np.testing.assert_allclose(value, expected, rtol=1e-12, atol=atol, equal_nan=True)
            assert not value.flags.writeable and value.shape == expected.shape
            np.testing.assert_array_equal(np.isfinite(value), np.isfinite(expected))


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("n,t,k", [(32, 32, 1), (65, 63, 7), (129, 1025, 30), (65, 63, 0)])
def test_shared_rows_match_independent_extended_precision(layout, n, t, k):
    values = data(n, t, k, layout)
    saved = {key: value.copy() for key, value in values.items()}
    for value in values.values():
        value.flags.writeable = False
    actual = run(graph(), values)
    reference = run(graph([*ROOTS, Y]), values)
    compare(actual, reference)
    y = values["w"].astype(np.longdouble) @ values["r"].astype(np.longdouble).T
    mean = y.mean(axis=1)
    std = np.sqrt(((y - mean[:, None]) ** 2).sum(axis=1) / (t - 1) * 252)
    np.testing.assert_allclose(actual.outputs[0].values[0], mean * 252, atol=4e-5, rtol=1e-12)
    np.testing.assert_allclose(actual.outputs[1].values[0], std, atol=4e-5, rtol=1e-12)
    assert count(actual, "numeric_arena_bytes") < n * t * 8
    assert count(actual, "input_copy_bytes") == 0
    if k and LIBRARY:
        assert count(actual, "accelerate_matmul_calls") > 0
    else:
        assert count(actual, "accelerate_matmul_calls") == 0
    for key, value in values.items():
        np.testing.assert_array_equal(value, saved[key])
    for output in actual.outputs:
        for value in output.values:
            for source in values.values():
                assert not np.shares_memory(value, source)


@pytest.mark.parametrize(
    "view", ["reverse_w", "reverse_r", "padded_w", "padded_r", "zero_w", "reverse_q"]
)
def test_views_stay_bounded_and_readonly(view):
    values = data()
    key = view.split("_")[-1]
    a = values[key]
    if view.startswith("reverse"):
        values[key] = a[::-1] if a.ndim == 1 else a[::-1, ::-1]
    elif view.startswith("padded"):
        owner = np.full((a.shape[0] * 2, a.shape[1] * 2), np.nan)
        owner[::2, ::2] = a
        values[key] = owner[::2, ::2]
    else:
        values[key] = np.broadcast_to(a[:1], a.shape)
    actual = run(graph(), values)
    compare(actual, run(graph([*ROOTS, Y]), values))
    assert count(actual, "numeric_arena_bytes") < 65 * 63 * 8


@pytest.mark.parametrize(
    "fault", ["nan", "inf", "overflow", "constant", "zero", "cancel", "large_offset"]
)
def test_exceptional_values_and_canonical_replay(fault):
    values = data()
    roots = ROOTS
    if fault == "nan":
        values["r"][3, 0] = np.nan
    elif fault == "inf":
        values["w"][3, 0] = np.inf
    elif fault == "overflow":
        values["w"][:] = 1e200
        values["r"][:] = 1e200
    elif fault == "constant":
        values["r"][:] = np.linspace(0.01, 0.1, 7)
    elif fault == "zero":
        values["r"][:] = 0.0
    elif fault == "cancel":
        values["w"][:, 1] *= -1
        values["r"][:, 1] = values["r"][:, 0]
    elif fault == "large_offset":
        values["r"] += 1e10
    try:
        reference = run(graph([*roots, Y]), values)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            run(graph(roots), values)
    else:
        actual = run(graph(roots), values)
        compare(actual, reference)
        assert count(actual, "numeric_arena_bytes") < 65 * 63 * 8


@pytest.mark.parametrize(
    "roots",
    [
        [ROOTS[0], "divide(sum_axis(w,1),0)", ROOTS[2]],
        [ROOTS[2], "divide(sum_axis(w,1),0)", ROOTS[1]],
        [ROOTS[1], ROOTS[2]],
    ],
)
def test_invalid_probabilities_preserve_original_error_priority(roots):
    values = data()
    values["q"][0] = np.nan
    values["r"][3, 0] = np.nan
    with pytest.raises(ValueError) as want:
        run(graph([*roots, Y]), values)
    with pytest.raises(ValueError) as actual:
        run(graph(roots), values)
    assert str(actual.value) == str(want.value)


@pytest.mark.parametrize(
    "tail", ["greater_than(qt,0)", "argsort(sum_axis(qt,1))", "qt+1", "sqrt(qt)"]
)
def test_discrete_and_unproven_tails_do_not_lower(tail):
    roots = [tail.replace("qt", f"quantiles_axis({Y},q,1)")]
    values = data()
    try:
        reference = run(graph([*roots, Y]), values)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            run(graph(roots), values)
    else:
        actual = run(graph(roots), values)
        compare(actual, reference, 1, 0)
        assert count(actual, "accelerate_matmul_calls") == 0
        assert count(actual, "numeric_arena_bytes") >= 65 * 63 * 8


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_transports_and_results_survive_repeated_execution(lane, layout):
    values = data(layout=layout)
    g = pickle.loads(pickle.dumps(graph(output_order=layout)))
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(values) as bundle:
                first = scheduler.execute(g, bundle, bounds, bounds)
                saved = [[value.copy() for value in out.values] for out in first.outputs]
                second = scheduler.execute(g, bundle, bounds, bounds)
        else:
            first = scheduler.execute(g, values, bounds, bounds)
            saved = [[value.copy() for value in out.values] for out in first.outputs]
            second = scheduler.execute(g, values, bounds, bounds)
    compare(first, second)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for old, new, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
        for a, b, s in zip(old.values, new.values, snapshot, strict=True):
            assert not np.shares_memory(a, b)
            np.testing.assert_array_equal(a, s)
            if a.ndim == 2:
                assert a.flags.c_contiguous if layout == "C" else a.flags.f_contiguous


@pytest.mark.parametrize("q", [[], [0.0], [1.0], [np.nan], [np.inf], [-0.1], [1.1]])
def test_empty_and_bad_probability_vectors(q):
    values = data()
    values["q"] = np.array(q)
    roots = [ROOTS[2]]
    try:
        reference = run(graph([*roots, Y]), values)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            run(graph(roots), values)
    else:
        compare(run(graph(roots), values), reference, 1)


def test_multiple_probability_consumers_and_private_producers():
    values = data()
    values["p"] = np.array([0.2, 0.8])
    types = dict(TYPES, p=dict(kind="vector", axes=["asset"], shape=["P"]))
    _other = "matmul(w,transpose(r))*2"
    roots = [ROOTS[0], f"quantiles_axis({Y},q,1)", f"quantiles_axis({Y},p,1)", ROOTS[1]]
    compiler = GraphCompiler(types)
    actual = run(compiler.compile(roots, result_format="typed"), values)
    reference = run(compiler.compile([*roots, Y], result_format="typed"), values)
    compare(actual, reference, 4)
    assert count(actual, "numeric_arena_bytes") < 65 * 63 * 8


def test_prepared_input_changes_and_geometry_shrink():
    values = data(1024, 1024, 7)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        g = graph()
        plan = scheduler.plan(g, values, bounds, bounds)
        assert plan.estimated_worker_scratch_bytes < 2 * 1024 * 1024
        admitted = scheduler.execute(
            g, values, bounds, bounds, memory_budget_bytes=plan.estimated_total_memory_bytes
        )
        assert plan.estimated_worker_scratch_bytes >= count(
            admitted, "numeric_arena_bytes"
        ) + count(admitted, "operator_workspace_capacity_bytes")
        prepared = scheduler.prepare_execution(g, values, bounds, bounds)
        first = prepared.run_snapshot()
        saved = [out.values[0].copy() for out in first.outputs]
        values["r"] *= 2
        values["q"][1] = 0.25
        second = prepared.run_snapshot()
        reference = scheduler.execute(graph([*ROOTS, Y]), values, bounds, bounds)
        compare(second, reference)
        for out, s in zip(first.outputs, saved, strict=True):
            np.testing.assert_array_equal(out.values[0], s)
        small = scheduler.execute(g, data(32, 32, 1), bounds, bounds)
        assert count(small, "numeric_arena_bytes") < count(first, "numeric_arena_bytes")


def test_isolated_failure_retains_independent_values_and_canonical_errors():
    values = data()
    values["r"][5, 0] = np.nan
    roots = [*ROOTS, "sum_axis(w,1)"]
    actual = run(graph(roots, error_policy="isolate"), values)
    reference = run(graph([*roots, Y], error_policy="isolate"), values)
    compare(actual, reference, 4)
    assert actual.outputs[3].root_statuses[0] == 0 and count(actual, "accelerate_matmul_calls") == 0


@pytest.mark.parametrize("length", [0, 1, 2, 31, 32, 63])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_variable_time_interval_geometry_and_errors(length, layout):
    values = data(layout=layout)
    types = dict(TYPES, r=dict(kind="matrix", axes=["time", "asset"], shape=["T", "K"]))
    roots = [f"-quantiles_axis({Y},q,1)"]
    compiler = GraphCompiler(types)
    g = compiler.compile(roots, result_format="typed")
    full = compiler.compile([*roots, Y], result_format="typed")
    starts = np.array([0], np.int64)
    ends = np.array([length], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        try:
            reference = scheduler.execute(full, values, starts, ends)
        except ValueError as error:
            with pytest.raises(ValueError, match=str(error).split(":")[0]):
                scheduler.execute(g, values, starts, ends)
        else:
            compare(scheduler.execute(g, values, starts, ends), reference, 1)


@pytest.mark.parametrize("length", [0, 1, 2, 31, 32, 63])
def test_empty_probabilities_consume_no_empty_time_line(length):
    values = data()
    values["q"] = np.empty(0)
    types = dict(TYPES, r=dict(kind="matrix", axes=["time", "asset"], shape=["T", "K"]))
    roots = [f"quantiles_axis({Y},q,1)"]
    compiler = GraphCompiler(types)
    starts = np.array([0], np.int64)
    ends = np.array([length], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = scheduler.execute(
            compiler.compile(roots, result_format="typed"), values, starts, ends
        )
        expected = scheduler.execute(
            compiler.compile([*roots, Y], result_format="typed"), values, starts, ends
        )
        compare(actual, expected, 1)
        assert actual.outputs[0].values[0].shape == (65, 0)


@pytest.mark.parametrize("d", [0, 1, 3, 63, -1, 0.5])
def test_dynamic_ddof_and_quantiles_preserve_original_error(d):
    values = data()
    roots = [ROOTS[2], f"variance_axis({Y},1,d)"]
    compiler = GraphCompiler(dict(TYPES, d="scalar"))
    try:
        expected = run(
            compiler.compile([*roots, Y], result_format="typed"), values, parameters={"d": d}
        )
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            run(compiler.compile(roots, result_format="typed"), values, parameters={"d": d})
    else:
        actual = run(compiler.compile(roots, result_format="typed"), values, parameters={"d": d})
        compare(actual, expected, 2, 0)
        assert count(actual, "accelerate_matmul_calls") == 0


def test_large_probability_cache_does_not_replace_smaller_trajectory():
    values = data(1024, 32, 1)
    values["q"] = np.linspace(0.0, 1.0, 128)
    roots = [ROOTS[2]]
    actual = run(graph(roots), values)
    expected = run(graph([*roots, Y]), values)
    compare(actual, expected, 1, 0)
    assert count(actual, "accelerate_matmul_calls") == 0
    assert count(actual, "numeric_arena_bytes") >= 1024 * 32 * 8


@pytest.mark.parametrize("layout", ["C", "F"])
def test_wide_direct_quantile_root_has_no_output_sized_cache(layout):
    values = data(4096, 257, 7, layout)
    values["q"] = np.linspace(0.0, 1.0, 128)
    roots = [ROOTS[0], ROOTS[1], f"quantiles_axis({Y},q,1)"]
    for value in values.values():
        value.flags.writeable = False
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        g = graph(roots, output_order=layout)
        first = scheduler.execute(g, values, bounds, bounds)
        saved = [out.values[0].copy() for out in first.outputs]
        second = scheduler.execute(g, values, bounds, bounds)
        expected = scheduler.execute(
            graph([*roots, Y], output_order=layout), values, bounds, bounds
        )
    compare(first, expected)
    compare(second, expected)
    result = first.outputs[2].values[0]
    # A duplicate N*Q cache alone would exceed this bound; the required public
    # result is independent of the bounded trajectory/statistics scratch.
    assert count(first, "numeric_arena_bytes") < result.nbytes // 4
    assert result.flags.c_contiguous if layout == "C" else result.flags.f_contiguous
    y = values["w"][:5].astype(np.longdouble) @ values["r"].astype(np.longdouble).T
    np.testing.assert_allclose(
        result[:5], np.quantile(y, values["q"], axis=1).T, atol=4e-5, rtol=1e-12
    )
    for old, new, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
        np.testing.assert_array_equal(old.values[0], snapshot)
        assert not np.shares_memory(old.values[0], new.values[0])
        for source in values.values():
            assert not np.shares_memory(old.values[0], source)


@pytest.mark.parametrize("layout", ["C", "F"])
def test_direct_root_and_private_quantile_tail_share_sort_without_cache_overlap(layout):
    values = data(4096, 257, 7, layout)
    values["q"] = np.linspace(0.0, 1.0, 128)
    values["p"] = np.array([0.1, 0.9])
    types = dict(TYPES, p=dict(kind="vector", axes=["asset"], shape=["P"]))
    roots = [f"quantiles_axis({Y},q,1)", f"-quantiles_axis({Y},p,1)", ROOTS[0], ROOTS[1]]
    compiler = GraphCompiler(types)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        g = compiler.compile(roots, result_format="typed", output_order=layout)
        actual = scheduler.execute(g, values, bounds, bounds)
        expected = scheduler.execute(
            compiler.compile([*roots, Y], result_format="typed", output_order=layout),
            values,
            bounds,
            bounds,
        )
    compare(actual, expected, 4)
    assert count(actual, "numeric_arena_bytes") < actual.outputs[0].values[0].nbytes // 4
    assert not np.shares_memory(actual.outputs[0].values[0], actual.outputs[1].values[0])


@pytest.mark.parametrize("layout", ["C", "F"])
def test_direct_root_can_feed_tails_and_duplicate_public_roots(layout):
    values = data(layout=layout)
    qt = f"quantiles_axis({Y},q,1)"
    roots = [qt, f"-{qt}", f"{qt}*2", qt]
    actual = run(graph(roots, output_order=layout), values)
    expected = run(graph([*roots, Y], output_order=layout), values)
    compare(actual, expected, 4)
    for i, out in enumerate(actual.outputs):
        for other in actual.outputs[i + 1 :]:
            assert not np.shares_memory(out.values[0], other.values[0])


@pytest.mark.parametrize("layout", ["C", "F"])
def test_direct_root_wider_than_trajectory_keeps_bounded_storage(layout):
    values = data(4096, 32, 1, layout)
    values["q"] = np.linspace(0.0, 1.0, 128)
    qt = f"quantiles_axis({Y},q,1)"
    actual = run(graph([qt], output_order=layout), values)
    expected = run(graph([qt, Y], output_order=layout), values)
    compare(actual, expected, 1)
    assert count(actual, "numeric_arena_bytes") < actual.outputs[0].values[0].nbytes // 4


@pytest.mark.parametrize("layout", ["C", "F"])
def test_prepared_direct_roots_read_current_probabilities_and_keep_old_results(layout):
    values = data(129, 1025, 7, layout)
    qt = f"quantiles_axis({Y},q,1)"
    roots = [ROOTS[0], qt]
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        g = graph(roots, output_order=layout)
        prepared = scheduler.prepare_execution(g, values, bounds, bounds)
        first = prepared.run_snapshot()
        saved = [out.values[0].copy() for out in first.outputs]
        values["r"] *= 1.3
        values["q"][1] = 0.3
        second = prepared.run_snapshot()
        expected = scheduler.execute(
            graph([*roots, Y], output_order=layout), values, bounds, bounds
        )
        compare(second, expected, 2)
    for old, new, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
        np.testing.assert_array_equal(old.values[0], snapshot)
        assert not np.shares_memory(old.values[0], new.values[0])


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_direct_root_transports_preserve_layout_and_independent_generations(lane, layout):
    values = data(layout=layout)
    values["q"] = np.linspace(0.0, 1.0, 17)
    roots = [f"quantiles_axis({Y},q,1)", ROOTS[0]]
    bounds = np.zeros(2, np.int64)
    g = pickle.loads(pickle.dumps(graph(roots, output_order=layout)))
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(values) as bundle:
                first = scheduler.execute(g, bundle, bounds, bounds)
                saved = [[value.copy() for value in out.values] for out in first.outputs]
                second = scheduler.execute(g, bundle, bounds, bounds)
        else:
            first = scheduler.execute(g, values, bounds, bounds)
            saved = [[value.copy() for value in out.values] for out in first.outputs]
            second = scheduler.execute(g, values, bounds, bounds)
    compare(first, second, 2)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for old, new, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
        for a, b, copy in zip(old.values, new.values, snapshot, strict=True):
            np.testing.assert_array_equal(a, copy)
            assert not np.shares_memory(a, b)
            if a.ndim == 2:
                assert a.flags.c_contiguous if layout == "C" else a.flags.f_contiguous


@pytest.mark.parametrize("qt_first", [True, False])
def test_invalid_direct_root_probability_preserves_independent_error_priority(qt_first):
    values = data()
    values["q"][0] = np.nan
    qt = f"quantiles_axis({Y},q,1)"
    bad = "divide(sum_axis(w,1),0)"
    roots = [qt, bad] if qt_first else [ROOTS[0], bad, qt]
    with pytest.raises(ValueError) as expected:
        run(graph([*roots, Y]), values)
    with pytest.raises(ValueError) as actual:
        run(graph(roots), values)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("case", ["empty_q", "empty_time", "empty_rows"])
def test_mixed_intervals_share_upper_bound_but_use_actual_geometry(case):
    values = data()
    roots = [ROOTS[2]]
    types = dict(TYPES)
    if case == "empty_rows":
        types["w"] = dict(kind="matrix", axes=["time", "asset"], shape=["N", "K"])
        end = 65
    else:
        types["r"] = dict(kind="matrix", axes=["time", "asset"], shape=["T", "K"])
        end = 63
    if case == "empty_q":
        values["q"] = np.empty(0)
    compiler = GraphCompiler(types)
    g = compiler.compile(roots, result_format="typed")
    control = compiler.compile([*roots, Y], result_format="typed")
    starts = np.array([0, 0], np.int64)
    ends = np.array([0, end], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        try:
            expected = scheduler.execute(control, values, starts, ends)
        except ValueError as error:
            with pytest.raises(ValueError, match=str(error).split(":")[0]):
                scheduler.execute(g, values, starts, ends)
        else:
            compare(scheduler.execute(g, values, starts, ends), expected, 1)


@pytest.mark.parametrize("layout", ["C", "F"])
def test_long_history_has_bounded_working_set_and_certified_sqrt(layout):
    values = data(32, 200000, 1, layout)
    values["r"] += 0.002
    bounds = np.zeros(1, np.int64)
    for value in values.values():
        value.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        g = graph()
        plan = scheduler.plan(g, values, bounds, bounds)
        first = scheduler.execute(
            g, values, bounds, bounds, memory_budget_bytes=plan.estimated_total_memory_bytes
        )
        saved = [out.values[0].copy() for out in first.outputs]
        second = scheduler.execute(g, values, bounds, bounds)
    y = values["w"][:3].astype(np.longdouble) @ values["r"].astype(np.longdouble).T
    mean = y.mean(axis=1)
    variance = ((y - mean[:, None]) ** 2).sum(axis=1) / (y.shape[1] - 1)
    np.testing.assert_allclose(first.outputs[0].values[0][:3], mean * 252, rtol=1e-12, atol=4e-5)
    np.testing.assert_allclose(
        first.outputs[1].values[0][:3], np.sqrt(variance * 252), rtol=1e-12, atol=4e-5
    )
    np.testing.assert_allclose(
        first.outputs[2].values[0][:3],
        -np.quantile(y, np.asarray(values["q"], dtype=np.longdouble), axis=1).T,
        rtol=1e-12,
        atol=4e-5,
    )
    # Full Y would be 51.2 MB. The longest single trajectory is the only
    # time-dependent working-set lower bound, not 32 complete trajectories.
    assert count(first, "numeric_arena_bytes") < 2_000_000
    assert count(first, "operator_workspace_capacity_bytes") < 2_000_000
    # Admission also reserves a canonical sorting line and vendor workspace;
    # those estimates are distinct from the actually allocated private cache.
    assert plan.estimated_worker_scratch_bytes < 5_000_000
    if LIBRARY:
        assert count(first, "accelerate_matmul_calls") == 32
        assert count(first, "summary_source_scans") == 0
    else:
        assert count(first, "accelerate_matmul_calls") == 0
    for old, new, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
        np.testing.assert_array_equal(old.values[0], snapshot)
        assert not np.shares_memory(old.values[0], new.values[0])


def test_time_upper_bound_covers_larger_short_interval_batch():
    values = data(65, 100000, 3)
    types = dict(TYPES)
    types["r"] = dict(kind="matrix", axes=["time", "asset"], shape=["T", "K"])
    roots = [*ROOTS]
    compiler = GraphCompiler(types)
    starts = np.array([0, 0], np.int64)
    ends = np.array([2520, 100000], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        g = compiler.compile(roots, result_format="typed")
        plan = scheduler.plan(g, values, starts, ends)
        actual = scheduler.execute(
            g, values, starts, ends, memory_budget_bytes=plan.estimated_total_memory_bytes
        )
        expected = scheduler.execute(
            compiler.compile([*roots, Y], result_format="typed"), values, starts, ends
        )
    compare(actual, expected)
    assert count(actual, "numeric_arena_bytes") < 3_000_000
    assert count(actual, "operator_workspace_capacity_bytes") < 3_000_000
    assert plan.estimated_worker_scratch_bytes < 4_000_000
