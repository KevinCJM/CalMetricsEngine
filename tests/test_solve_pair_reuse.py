"""Two terminal solve_multi roots share ordered A elimination, not a cached LU.

The scalar direct kernel controls canonical arithmetic. The independent matvec
fixture identifies the shared Eigen gate; actual audit calls distinguish the
guarded Accelerate path, which can remain enabled with Eigen disabled.
"""

import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_solve_library import eigen_backend_available as _probe_eigen_backend
from .test_solve_library import residual, same_ieee, sample
from .test_typed_results import config, execute

EXPRESSIONS = ["solve_multi(a,b1)", "solve_multi(a,b2)"]
DECLARATIONS = {
    "a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]),
    "b1": dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K1"]),
    "b2": dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K2"]),
}


@pytest.fixture(scope="module")
def eigen_backend_available():
    return _probe_eigen_backend.__wrapped__()


def pair_graph(*, expressions=None, declarations=None, **kwargs):
    return GraphCompiler(DECLARATIONS if declarations is None else declarations).compile(
        EXPRESSIONS if expressions is None else expressions, **kwargs
    )


def inputs(n=129, layout="c"):
    a, b1 = sample("solve_multi", n, columns=2, layout=layout)
    _, b2 = sample("solve_multi", n, columns=5, layout=layout)
    return dict(a=a, b1=b1, b2=b2)


def run(graph, data, *, lane="single", rows=1):
    return execute(graph, data, [0] * rows, [0] * rows, lane)


def count(result, name):
    return sum(chunk[name] for chunk in result.audit["native_chunks"])


def expected_copies(data, shared):
    return ((1 if shared else 2) * data["a"].size + data["b1"].size + data["b2"].size) * 8


def scalar_pair(data):
    return [op.solve_multi(data["a"], data[key], simd="scalar") for key in ("b1", "b2")]


def check_values(result, expected, data, *, row=0, order="C", allow_roundoff=False):
    values = []
    for output, want in zip(result.outputs[:2], expected, strict=True):
        actual = output.values[row]
        if allow_roundoff:
            assert actual.shape == want.shape and actual.dtype == want.dtype
            np.testing.assert_allclose(actual, want, rtol=1e-11, atol=4.9e-5)
        else:
            same_ieee(actual, want)
        assert output.root_statuses[row] == 0 and output.shape_known[row]
        assert output.order == order
        assert actual.flags.c_contiguous if order == "C" else actual.flags.f_contiguous
        assert not actual.flags.writeable
        for source in data.values():
            assert not np.shares_memory(actual, source)
        values.append(actual)
    assert not np.shares_memory(*values)


@pytest.mark.parametrize("n,layout", [(127, "c"), (128, "c"), (129, "f"),
                                       (257, "negative"), (1025, "c")])
def test_terminal_pair_layout_boundary_and_a_copy_accounting(n, layout, eigen_backend_available):
    data = inputs(n, layout)
    saved = {key: value.copy() for key, value in data.items()}
    for value in data.values():
        value.flags.writeable = False
    expected = scalar_pair(data)
    graph = pair_graph()
    # Discovery must accept the static b2 input between the two solve nodes.
    solve_ids = [node.node_id for node in graph.nodes if node.operator_id == "solve_multi"]
    result = run(graph, data)
    library_calls = count(result, "accelerate_lu_calls")
    check_values(result, expected, data, allow_roundoff=bool(library_calls))
    for key, value in data.items():
        same_ieee(value, saved[key])
    for output, rhs in zip(result.outputs, (data["b1"], data["b2"]), strict=True):
        residual(data["a"], output.values[0], rhs)
    assert len(solve_ids) == 2
    assert solve_ids[1] > solve_ids[0] + 1
    if library_calls:
        assert not eigen_backend_available and (n, layout) == (1025, "c")
        assert library_calls == count(result, "accelerate_lu_attempts") == 2
        assert count(result, "accelerate_lu_fallbacks") == 0
        # Each independent root copies/transposes A and packs/returns its RHS.
        copies = (4 * data["a"].size + 2 * (data["b1"].size + data["b2"].size)) * 8
    else:
        copies = expected_copies(data, eigen_backend_available and n >= 128)
    assert count(result, "algorithm_copy_bytes") == copies
    assert count(result, "input_copy_bytes") == 0


@pytest.mark.parametrize("case", ["cross_panel_pivot", "equal_pivots", "signed_zero"])
def test_shared_pivots_and_zero_bits_keep_each_rhs_order(case, eigen_backend_available):
    n = 129
    data = dict(a=np.eye(n) * 2, b1=np.ones((n, 2)), b2=np.ones((n, 5)))
    if case == "cross_panel_pivot":
        indices = [32, 33, 65]
        data["a"][np.ix_(indices, indices)] = [[4., 1., 2.], [1., 1., 7.], [2., 5., 3.]]
        data["b1"][indices] = [[1., 4.], [2., 5.], [3., 6.]]
        data["b2"][indices] = np.arange(15.).reshape(3, 5)
    elif case == "equal_pivots":
        data["a"][0, 0], data["a"][1, 0], data["a"][65, 0] = 1., -1., 1.
        data["b1"][1] = [-3., 2.]
        data["b2"][65] = [1., -2., 3., -4., 5.]
    else:
        data["a"][data["a"] == 0.] = -0.
        data["b1"][:] = -0.
        data["b2"][:, ::2] = -0.
        data["b2"][:, 1::2] = 0.
    result = run(pair_graph(), data)
    check_values(result, scalar_pair(data), data)
    assert count(result, "algorithm_copy_bytes") == expected_copies(data, eigen_backend_available)


@pytest.mark.parametrize("empty", ["first", "second", "both"])
def test_empty_rhs_columns_still_share_a_and_validate_it(empty, eigen_backend_available):
    data = inputs()
    if empty in ("first", "both"):
        data["b1"] = np.empty((129, 0))
    if empty in ("second", "both"):
        data["b2"] = np.empty((129, 0))
    result = run(pair_graph(), data)
    check_values(result, scalar_pair(data), data)
    assert count(result, "algorithm_copy_bytes") == expected_copies(data, eigen_backend_available)
    data["a"][-1, :] = 0.0
    with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
        run(pair_graph(), data)


@pytest.mark.parametrize("sign", [1., -1.])
@pytest.mark.parametrize("position", ["below", "equal", "above"])
def test_pivot_threshold_keeps_strict_inequality(sign, position, eigen_backend_available):
    data = dict(a=np.eye(128), b1=np.ones((128, 2)), b2=np.ones((128, 3)))
    magnitude = {"below": np.nextafter(1e-14, 0.), "equal": 1e-14,
                 "above": np.nextafter(1e-14, np.inf)}[position]
    data["a"][-1, -1] = sign * magnitude
    if position != "above":
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            run(pair_graph(), data)
    else:
        result = run(pair_graph(), data)
        check_values(result, scalar_pair(data), data)
        assert count(result, "algorithm_copy_bytes") == expected_copies(data, eigen_backend_available)


@pytest.mark.parametrize("fault", ["singular_then_nan", "nan_then_singular", "matrix_inf",
                                   "elimination_overflow", "first_backsolve_overflow",
                                   "second_backsolve_overflow", "zero_times_intermediate_inf"])
def test_failure_order_matches_two_canonical_calls_and_preserves_inputs(fault):
    n = 129
    data = dict(a=np.eye(n) * 2, b1=np.ones((n, 2)), b2=np.ones((n, 3)))
    a, b1, b2 = data["a"], data["b1"], data["b2"]
    maximum = np.finfo(np.float64).max
    if fault == "singular_then_nan":
        a[-1, -1], b2[-1, 0] = 0., np.nan
    elif fault == "nan_then_singular":
        b1[-1, 0], a[-1, -1] = np.nan, 0.
    elif fault == "matrix_inf":
        a[-1, -1] = np.inf
    elif fault == "elimination_overflow":
        a[-2:, -2:] = [[maximum, maximum], [-maximum, maximum]]
    elif fault in ("first_backsolve_overflow", "second_backsolve_overflow"):
        a[-1, -1] = 2e-14
        (b1 if fault == "first_backsolve_overflow" else b2)[-1, 1] = 1e308
    else:
        # Do not skip factor==0 after an earlier elimination creates Inf.
        a[0, 0], a[1, 0] = 1., -1.
        a[0, -1] = a[1, -1] = maximum
    saved = {key: value.copy() for key, value in data.items()}
    for value in data.values():
        value.flags.writeable = False
    with pytest.raises(ValueError) as reference:
        scalar_pair(data)
    if fault == "singular_then_nan":
        assert str(reference.value) == "SINGULAR_MATRIX"
    elif fault == "nan_then_singular":
        assert str(reference.value) == "NON_FINITE_RESULT"
    with pytest.raises(ValueError) as actual:
        run(pair_graph(), data)
    assert str(actual.value) == str(reference.value)
    for key, value in data.items():
        same_ieee(value, saved[key])


@pytest.mark.parametrize("mode", ["isolate", "different_a", "f_outputs", "third_root",
                                  "consumer", "temporary_rhs", "temporary_a", "scope"])
def test_unqualified_graphs_keep_two_coefficient_copies(mode):
    data = inputs(128)
    declarations = dict(DECLARATIONS)
    expressions = list(EXPRESSIONS)
    kwargs = {}
    if mode == "isolate":
        kwargs["error_policy"] = "isolate"
    elif mode == "different_a":
        declarations["other_a"] = DECLARATIONS["a"]
        data["other_a"] = data["a"]  # Same address/value is not the same DAG node.
        expressions[1] = "solve_multi(other_a,b2)"
    elif mode == "f_outputs":
        kwargs["output_order"] = "F"
    elif mode == "third_root":
        expressions.append("7")
    elif mode == "consumer":
        expressions[1] += "*1"
    elif mode == "temporary_rhs":
        expressions[1] = "solve_multi(a,b2+0)"
    elif mode == "temporary_a":
        expressions = ["solve_multi(a+0,b1)", "solve_multi(a+0,b2)"]
    else:
        data["a"] = np.eye(128)
        expressions = [f"iterate({expression},b{i},0,1)"
                       for i, expression in enumerate(expressions, 1)]
    graph = pair_graph(expressions=expressions, declarations=declarations, **kwargs)
    result = run(graph, data)
    check_values(result, scalar_pair(data), data, order="F" if mode == "f_outputs" else "C")
    assert count(result, "algorithm_copy_bytes") == expected_copies(data, False)
    if mode == "third_root":
        assert result.outputs[2].values[0] == 7


@pytest.mark.parametrize("bad_rhs", ["b1", "b2"])
def test_isolation_keeps_healthy_root_and_position_statuses(bad_rhs):
    data = inputs(128)
    data[bad_rhs][-1, 0] = np.nan
    result = run(pair_graph(error_policy="isolate"), data)
    # Distinct source identity forces the same original two-node execution.
    declarations = {**DECLARATIONS, "other_a": DECLARATIONS["a"]}
    reference = run(pair_graph(expressions=[EXPRESSIONS[0], "solve_multi(other_a,b2)"],
                               declarations=declarations, error_policy="isolate"),
                    {**data, "other_a": data["a"]})
    for index, (got, want) in enumerate(zip(result.outputs, reference.outputs, strict=True)):
        np.testing.assert_array_equal(got.root_statuses, want.root_statuses)
        np.testing.assert_array_equal(got.shape_known, want.shape_known)
        same_ieee(got.values[0], want.values[0])
        np.testing.assert_array_equal(got.statuses[0], want.statuses[0])
        if ("b1", "b2")[index] != bad_rhs:
            assert got.root_statuses[0] == 0
            same_ieee(got.values[0], op.solve_multi(data["a"], data[("b1", "b2")[index]], simd="scalar"))
        else:
            assert got.root_statuses[0] != 0


def test_prepared_reads_live_coefficients_and_rhs_and_keeps_old_snapshots(eigen_backend_available):
    data = inputs(129, "f")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(pair_graph(), data, bounds, bounds)
        first = prepared.run_snapshot()
        saved = [output.values[0].copy() for output in first.outputs]
        data["a"][np.diag_indices(129)] += 3.
        data["b1"] *= 2.
        data["b2"] -= 1.
        expected = scalar_pair(data)
        current = prepared.run_audit()
        check_values(current, expected, data)
        assert count(current, "algorithm_copy_bytes") == expected_copies(data, eigen_backend_available)
        second = prepared.run_snapshot()
        check_values(second, expected, data)
        valid_a = data["a"].copy()
        data["a"][-1, :] = 0.
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            prepared.run_audit()
        data["a"][:] = valid_a
        recovered = prepared.run_snapshot()
        check_values(recovered, expected, data)
        check_values(first, saved, data)
        assert first.audit["result_lifetime"] == second.audit["result_lifetime"] == "independent"
    for index in range(2):
        same_ieee(first.outputs[index].values[0], saved[index])
        same_ieee(second.outputs[index].values[0], expected[index])
        assert not np.shares_memory(first.outputs[index].values[0], second.outputs[index].values[0])
        assert not np.shares_memory(second.outputs[index].values[0], recovered.outputs[index].values[0])


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_pair_transport_owns_each_root_interval_and_request(lane, eigen_backend_available):
    data = inputs(129, "f")
    saved = {key: value.copy() for key, value in data.items()}
    for value in data.values():
        value.flags.writeable = False
    expected = scalar_pair(data)
    graph = pickle.loads(pickle.dumps(pair_graph()))
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        if lane == "process_shared":
            with SharedInputBundle.from_inputs(data) as bundle:
                result = scheduler.execute(graph, bundle, bounds, bounds)
                again = scheduler.execute(graph, bundle, bounds, bounds)
        else:
            result = scheduler.execute(graph, data, bounds, bounds)
            again = scheduler.execute(graph, data, bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    assert count(result, "algorithm_copy_bytes") == 2 * expected_copies(data, eigen_backend_available)
    if lane == "process_shared":
        assert result.plan.use_shared_memory
        assert result.audit["output_copy_bytes"] == 0
    retained = []
    for row in range(2):
        check_values(result, expected, data, row=row)
        check_values(again, expected, data, row=row)
        for index in range(2):
            value = result.outputs[index].values[row]
            assert not np.shares_memory(value, result.outputs[index].values[1 - row])
            assert not np.shares_memory(value, again.outputs[index].values[row])
            retained.append((value, expected[index]))
    for key, value in data.items():
        same_ieee(value, saved[key])
    del result, again, scheduler
    gc.collect()
    for value, want in retained:
        same_ieee(value, want)
        assert not value.flags.writeable
