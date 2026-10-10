"""Private row-statistics lowering: geometry, domain, ownership and native transport."""

import os
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_typed_results import config

TYPES = {
    key: dict(kind="matrix", axes=["asset", "asset"], shape=shape)
    for key, shape in dict(w=["N", "K"], r=["T", "K"]).items()
}
Y = "matmul(w,transpose(r))"
ROOTS = [f"mean_axis({Y},1)*252", f"sqrt(variance_axis({Y},1,1)*252)"]
LIBRARY = int(os.environ.get("CALMETRICS_TEST_EXPECT_ACCELERATE", "1"))


def fixture(n=65, t=63, k=7, layout="C"):
    rng = np.random.default_rng(728)
    w = rng.uniform(0.2, 1, (n, k))
    if k:
        w /= w.sum(axis=1, keepdims=True)
    r = rng.normal(scale=0.01, size=(t, k))
    values = {name: np.array(a, order=layout) for name, a in dict(w=w, r=r).items()}
    return values


def graph(roots=ROOTS, **kw):
    return GraphCompiler(TYPES).compile(roots, result_format="typed", **kw)


def run(g, data, lane="single", rows=1, **kw):
    bounds = np.zeros(rows, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        return scheduler.execute(g, data, bounds, bounds, **kw)


def count(result, name):
    return sum(c[name] for c in result.audit["native_chunks"])


def canonical(data, roots=ROOTS):
    return run(graph([*roots, Y]), data)


def check(result, data, row=0):
    # Independent long-double formula, rather than merely two similar kernels.
    y = data["w"].astype(np.longdouble) @ data["r"].astype(np.longdouble).T
    mean = y.mean(axis=1)
    expected = [
        mean * 252,
        np.sqrt(((y - mean[:, None]) ** 2).sum(axis=1) / (y.shape[1] - 1) * 252),
    ]
    arrays = []
    for out, want in zip(result.outputs, expected, strict=True):
        value = out.values[row]
        np.testing.assert_allclose(value, want, rtol=1e-12, atol=4e-5)
        assert out.root_statuses[row] == 0 and out.shape_known[row]
        assert value.shape == (data["w"].shape[0],) and value.dtype == np.float64
        assert not value.flags.writeable
        for source in data.values():
            assert not np.shares_memory(source, value)
        arrays.append(value)
    assert not np.shares_memory(*arrays)
    assert np.all(arrays[1] >= 0)
    return arrays


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize(
    "n,t,k", [(32, 32, 1), (65, 63, 7), (1025, 513, 30), (33, 4097, 8), (65, 63, 0)]
)
def test_matrix_statistics_finite_boundaries(layout, n, t, k):
    data = fixture(n, t, k, layout)
    saved = {name: a.copy() for name, a in data.items()}
    for a in data.values():
        a.flags.writeable = False
    result = run(graph(), data)
    check(result, data)
    for name, value in data.items():
        np.testing.assert_array_equal(value, saved[name])
    assert count(result, "numeric_arena_bytes") < n * t * 8
    assert count(result, "input_copy_bytes") == 0
    if k and LIBRARY:
        assert count(result, "covariance_syrk_calls") > 0
    else:
        assert count(result, "covariance_syrk_calls") == 0


@pytest.mark.parametrize("view", ["reverse_w", "reverse_r", "padded_w", "padded_r", "zero_w"])
def test_views_bounded_and_input_immutable(view):
    data = fixture()
    key = "w" if view.endswith("w") else "r"
    if view.startswith("reverse"):
        data[key] = data[key][::-1, ::-1]
    elif view.startswith("padded"):
        a = data[key]
        base = np.full((a.shape[0] * 2, a.shape[1] * 2), np.nan)
        base[::2, ::2] = a
        data[key] = base[::2, ::2]
    else:
        data[key] = np.broadcast_to(data[key][:1], data[key].shape)
    saved = {name: a.copy() for name, a in data.items()}
    for a in data.values():
        a.flags.writeable = False
    result = run(graph(), data)
    check(result, data)
    assert count(result, "numeric_arena_bytes") < data["w"].shape[0] * data["r"].shape[0] * 8
    for name, value in data.items():
        np.testing.assert_array_equal(value, saved[name])


@pytest.mark.parametrize(
    "case", ["constant", "cancel", "large_offset", "zero", "large_amplification"]
)
def test_guarded_roundoff_and_replay(case):
    data = fixture()
    roots = ROOTS
    if case == "constant":
        data["r"][:] = np.linspace(0.01, 0.1, 7)
    elif case == "cancel":
        data["w"][:, ::2] *= -1
        data["r"][:, 1] = data["r"][:, 0]
    elif case == "large_offset":
        data["r"] += 1e10
    elif case == "zero":
        data["r"][:] = 0
    else:
        roots = [f"mean_axis({Y},1)*1e100", f"variance_axis({Y},1,1)*1e100"]
    actual = run(graph(roots), data)
    reference = canonical(data, roots)
    for a, b in zip(actual.outputs, reference.outputs[:2], strict=True):
        np.testing.assert_allclose(a.values[0], b.values[0], rtol=1e-12, atol=4e-5)
    if case in ("large_offset", "large_amplification"):
        assert count(actual, "covariance_syrk_calls") == 0
    assert count(actual, "numeric_arena_bytes") < 65 * 63 * 8


@pytest.mark.parametrize("fault", ["nan", "inf", "overflow", "bad_ddof"])
@pytest.mark.parametrize("roots", [ROOTS, ROOTS[::-1], [f"mean_axis({Y},1)"]])
def test_negative_domain_matches_original_logical_error(fault, roots):
    data = fixture()
    if fault == "nan":
        data["r"][10, 0] = np.nan
    elif fault == "inf":
        data["w"][3, 1] = np.inf
    elif fault == "overflow":
        data["w"][:] = 1e200
        data["r"][:] = 1e200
    else:
        roots = [r.replace(",1,1)", ",1,63)") for r in roots]
    try:
        expected = canonical(data, roots)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            run(graph(roots), data)
    else:
        actual = run(graph(roots), data)
        for a, b in zip(actual.outputs, expected.outputs[:-1], strict=True):
            np.testing.assert_allclose(
                a.values[0], b.values[0], rtol=1e-12, atol=4e-5, equal_nan=True
            )
            np.testing.assert_array_equal(np.isnan(a.values[0]), np.isnan(b.values[0]))
            np.testing.assert_array_equal(np.isposinf(a.values[0]), np.isposinf(b.values[0]))
            np.testing.assert_array_equal(np.isneginf(a.values[0]), np.isneginf(b.values[0]))
            np.testing.assert_array_equal(a.root_statuses, b.root_statuses)


@pytest.mark.parametrize("tail", ["greater_than(mean,0)", "argsort(mean)", "mean+1", "mean*mean"])
def test_unproven_consumers_keep_original_matrix(tail):
    data = fixture()
    expression = tail.replace("mean", f"mean_axis({Y},1)")
    actual = run(graph([expression]), data)
    reference = canonical(data, [expression])
    np.testing.assert_array_equal(actual.outputs[0].values[0], reference.outputs[0].values[0])
    assert count(actual, "covariance_syrk_calls") == 0
    assert count(actual, "numeric_arena_bytes") >= 65 * 63 * 8


@pytest.mark.parametrize("n,t,k", [(31, 63, 7), (65, 31, 7), (65, 63, 129), (0, 63, 7), (65, 1, 7)])
def test_geometry_fallback(n, t, k):
    data = fixture(n, t, k)
    try:
        reference = canonical(data)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            run(graph(), data)
    else:
        actual = run(graph(), data)
        for a, b in zip(actual.outputs, reference.outputs[:2], strict=True):
            np.testing.assert_array_equal(a.values[0], b.values[0])
        assert count(actual, "covariance_syrk_calls") == 0


def test_several_statistics_and_private_matrices_do_not_alias():
    data = fixture()
    types = dict(TYPES, b=dict(kind="matrix", axes=["asset", "asset"], shape=["K", "T"]))
    other = "matmul(w,b)"
    data["b"] = data["r"].T.copy()
    roots = [
        f"sum_axis({Y},1)",
        f"mean_axis({Y},1)",
        f"variance_axis({Y},1,0)",
        f"variance_axis({Y},1,3)",
        f"mean_axis({other},1)",
    ]
    actual = run(GraphCompiler(types).compile(roots, result_format="typed"), data)
    full = op.matmul(data["w"], op.transpose(data["r"]))
    expected = [
        op.sum_axis(full, 1),
        op.mean_axis(full, 1),
        op.variance_axis(full, 1, 0),
        op.variance_axis(full, 1, 3),
        op.mean_axis(full, 1),
    ]
    for out, want in zip(actual.outputs, expected, strict=True):
        np.testing.assert_allclose(out.values[0], want, rtol=1e-12, atol=4e-5)
    assert count(actual, "numeric_arena_bytes") < 2 * 65 * 63 * 8


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_transport_intervals_and_retained_owned_generations(lane, layout):
    data = fixture(layout=layout)
    for a in data.values():
        a.flags.writeable = False
    g = pickle.loads(pickle.dumps(graph()))
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(data) as bundle:
                first = scheduler.execute(g, bundle, bounds, bounds)
                second = scheduler.execute(g, bundle, bounds, bounds)
        else:
            first = scheduler.execute(g, data, bounds, bounds)
            second = scheduler.execute(g, data, bounds, bounds)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for row in range(2):
        old = check(first, data, row)
        new = check(second, data, row)
        for a in old:
            for b in new:
                assert not np.shares_memory(a, b)


def test_prepared_live_inputs_and_snapshot_ownership():
    data = fixture()
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph(), data, bounds, bounds)
        first = prepared.run_snapshot()
        saved = [a.copy() for a in check(first, data)]
        data["r"] *= 2
        check(prepared.run_audit(), data)
        second = prepared.run_snapshot()
        check(second, data)
    for out, snapshot in zip(first.outputs, saved, strict=True):
        np.testing.assert_array_equal(out.values[0], snapshot)


def test_isolated_bad_branch_preserves_independent_values():
    data = fixture()
    data["r"][5, 0] = np.nan
    roots = [*ROOTS, "sum_axis(w,1)"]
    actual = run(graph(roots, error_policy="isolate"), data)
    expected = run(graph([*roots, Y], error_policy="isolate"), data)
    for out, want in zip(actual.outputs, expected.outputs[:3], strict=True):
        np.testing.assert_array_equal(out.values[0], want.values[0])
        np.testing.assert_array_equal(out.root_statuses, want.root_statuses)
    assert actual.outputs[2].root_statuses[0] == 0
    assert count(actual, "covariance_syrk_calls") == 0


def test_plan_budget_and_geometry_shrink():
    big = fixture(2048, 2048, 12)
    small = fixture(32, 32, 1)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        plan = scheduler.plan(graph(), big, bounds, bounds)
        assert plan.estimated_worker_scratch_bytes < 2 * 1024 * 1024
        result = scheduler.execute(
            graph(), big, bounds, bounds, memory_budget_bytes=plan.estimated_total_memory_bytes
        )
        assert plan.estimated_worker_scratch_bytes >= count(result, "numeric_arena_bytes") + count(
            result, "operator_workspace_capacity_bytes"
        )
        reduced = scheduler.execute(graph(), small, bounds, bounds)
        assert count(reduced, "operator_workspace_capacity_bytes") < count(
            result, "operator_workspace_capacity_bytes"
        )
        assert count(reduced, "operator_workspace_capacity_bytes") < 32768


@pytest.mark.parametrize("d", [0, 1, 3, 63, -1, 0.5])
def test_dynamic_ddof_keeps_logical_validation_and_bounded_replay(d):
    data = fixture()
    roots = [f"mean_axis({Y},1)", f"variance_axis({Y},1,d)"]
    compiler = GraphCompiler(dict(TYPES, d="scalar"))
    actual_graph = compiler.compile(roots, result_format="typed")
    control_graph = compiler.compile([*roots, Y], result_format="typed")
    try:
        expected = run(control_graph, data, parameters={"d": d})
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            run(actual_graph, data, parameters={"d": d})
    else:
        actual = run(actual_graph, data, parameters={"d": d})
        for a, b in zip(actual.outputs, expected.outputs[:2], strict=True):
            np.testing.assert_array_equal(a.values[0], b.values[0])
        assert count(actual, "covariance_syrk_calls") == 0
        assert count(actual, "numeric_arena_bytes") < 65 * 63 * 8


@pytest.mark.parametrize(
    "roots",
    [
        [f"mean_axis({Y},1)", "divide(sum_axis(w,1),0)", f"variance_axis({Y},1,1)"],
        [f"variance_axis({Y},1,1)", "divide(sum_axis(w,1),0)", f"mean_axis({Y},1)"],
    ],
)
def test_independent_error_priority_is_not_moved_to_producer(roots):
    data = fixture()
    data["r"][5, 0] = np.nan
    with pytest.raises(ValueError) as reference:
        canonical(data, roots)
    with pytest.raises(ValueError, match=str(reference.value).split(":")[0]):
        run(graph(roots), data)


def test_public_trajectory_remains_available():
    data = fixture()
    result = run(graph([*ROOTS, Y]), data)
    check_values = op.matmul(data["w"], op.transpose(data["r"]))
    np.testing.assert_array_equal(result.outputs[2].values[0], check_values)
    assert result.plan.estimated_output_bytes >= 65 * 63 * 8
    assert count(result, "covariance_syrk_calls") == 0


def test_observed_excel_reference_does_not_use_geometry_placeholder():
    from calmetrics_engine import excel

    data = fixture(32, 32, 2)
    plan = excel.plan(graph(), data, max_formula_characters=512 * 1024 * 1024)
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"
    reference = plan.reference()
    actual = run(graph(), data)
    for out, want in zip(actual.outputs, reference, strict=True):
        assert not want["error"]
        np.testing.assert_allclose(out.values[0], want["values"], rtol=1e-12, atol=4e-5)
    assert (
        count(actual, "covariance_syrk_calls") > 0
        if LIBRARY
        else count(actual, "covariance_syrk_calls") == 0
    )
