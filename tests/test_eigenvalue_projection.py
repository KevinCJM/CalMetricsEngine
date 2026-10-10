"""Bounded Jacobi projection: exact values, graph liveness and worker ownership."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import settings
from .test_matrix_array_foundation import matrix


def symmetric(n, layout="C"):
    x = np.random.default_rng(903033 + n).normal(size=(n, n))
    a = np.array(x + x.T, order="C" if layout == "reverse" else layout)
    if layout == "reverse":
        a = a[::-1, ::-1]
    a.flags.writeable = False
    return a


def exact(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def execute(expression, a, lane="single", isolate=False):
    graph = GraphCompiler({"a": matrix()}).compile(
        expression, error_policy="isolate" if isolate else "raise"
    )
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        return scheduler.execute(graph, {"a": a}, bounds, bounds)


@pytest.mark.parametrize("n", [0, 1, 8, 32, 127, 128, 129, 256, 257])
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_exact_projection_in_every_lane(n, layout, lane):
    a = symmetric(n, layout)
    before = a.copy()
    expected = op.eigenvalues(op.symmetric_eigh(a))
    result = execute("eigenvalues(symmetric_eigh(a))", a, lane)
    expected_lane = "single" if n == 0 else "process" if lane.startswith("process") else lane
    assert result.plan.lane == expected_lane
    for value in result.outputs[0].values:
        exact(value, expected)
        assert not value.flags.writeable
        assert not np.shares_memory(value, a)
    assert not np.shares_memory(*result.outputs[0].values)
    exact(a, before)
    for chunk in result.audit["native_chunks"]:
        assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
        assert chunk["direct_output_bytes"] == chunk["rows"] * n * 8


@pytest.mark.parametrize("layout", ["padded_C", "padded_F", "reverse"])
@pytest.mark.parametrize("expression", [
    "eigenvalues(symmetric_eigh(a))",
    "eigenvalues(symmetric_eigh(a+0))",
    "eigenvalues(symmetric_eigh(a+0))>0",
])
def test_dead_arena_and_unconsumed_padding(layout, expression):
    a = symmetric(128)
    if layout == "reverse":
        a = a[::-1, ::-1]
    else:
        owner = np.full((131, 131), np.nan, order=layout[-1])
        owner[:128, :128] = a
        a = owner[:128, :128]
    a.flags.writeable = False
    before = a.copy()
    expected = op.eigenvalues(op.symmetric_eigh(a + 0))
    actual = execute(expression, a).outputs[0].values[0]
    if expression.endswith(">0"):
        np.testing.assert_array_equal(actual, expected > 0)
    else:
        exact(actual, expected)
    exact(a, before)


@pytest.mark.parametrize("fixture", ["descending", "ties", "zeros", "negative_zero"])
@pytest.mark.parametrize("n", [128, 256])
def test_stable_full_record_permutation(fixture, n):
    diagonal = np.arange(n, 0, -1, dtype=float)
    if fixture == "ties":
        diagonal %= 4
    elif "zero" in fixture:
        diagonal.fill(-0.0 if fixture == "negative_zero" else 0.0)
    a = np.diag(diagonal)
    record = op.symmetric_eigh(a)
    # Original stable insertion order, including equal eigenvalues and zero signs.
    order = np.argsort(diagonal, kind="stable")
    expected = np.concatenate([diagonal[order, None], np.eye(n)[order]], axis=1)
    # The established zero-scale kernel yields positive zero values.
    if "zero" in fixture:
        expected[:, 0] = 0.0
    exact(record, expected)
    exact(execute("eigenvalues(symmetric_eigh(a))", a).outputs[0].values[0], record[:, 0])


@pytest.mark.parametrize("parameters", ["1e-15,128", "1e-3,32", "1e-12,32"])
@pytest.mark.parametrize("scale", [2.0**-500, 1.0, 2.0**500, 1e-308])
def test_explicit_parameters_and_scale(parameters, scale):
    a = symmetric(32) * scale
    tolerance, sweeps = map(float, parameters.split(","))
    expected = op.eigenvalues(op.symmetric_eigh(a, tolerance, sweeps))
    actual = execute(f"eigenvalues(symmetric_eigh(a,{parameters}))", a)
    exact(actual.outputs[0].values[0], expected)


@pytest.mark.parametrize("fault,code", [
    ("nan", "NON_FINITE_RESULT"), ("infinity", "NON_FINITE_RESULT"),
    ("asymmetric", "DOMAIN_ERROR"), ("sweeps", "NON_CONVERGENCE"),
    ("tolerance", "INVALID_PARAMETER"),
])
def test_projection_preserves_errors_and_independent_isolated_roots(fault, code):
    a = symmetric(32).copy()
    parameters = "1e-12,32"
    if fault == "nan":
        a[0, 0] = np.nan
    elif fault == "infinity":
        a[0, 0] = np.inf
    elif fault == "asymmetric":
        a[0, 1] += 1.0
    elif fault == "sweeps":
        parameters = "1e-15,1"
    elif fault == "tolerance":
        parameters = "0,32"
    expr = f"eigenvalues(symmetric_eigh(a,{parameters}))"
    with pytest.raises((ValueError, RuntimeError), match=code):
        execute(expr, a)
    result = execute([expr, "7"], a, isolate=True)
    assert np.isnan(result.outputs[0].values[0]).all()
    assert result.outputs[0].statuses[0].all()
    assert result.outputs[1].values[0] == 7


@pytest.mark.parametrize("shared", ["record", "vectors", "isolate"])
def test_shared_records_vectors_and_isolate_retain_full_computation(shared):
    a = symmetric(128)
    expressions = ["eigenvalues(symmetric_eigh(a))"]
    if shared == "record":
        expressions.append("symmetric_eigh(a)")
        with pytest.raises(GraphCompileError, match="PUBLIC_ROOT_TYPE"):
            execute(expressions, a)
        return
    elif shared == "vectors":
        expressions.append("eigenvectors(symmetric_eigh(a))")
    result = execute(expressions, a, isolate=shared == "isolate")
    record = op.symmetric_eigh(a)
    exact(result.outputs[0].values[0], record[:, 0])
    if shared != "isolate":
        expected = record if shared == "record" else op.eigenvectors(record)
        exact(result.outputs[1].values[0], expected)
    # A full record rooted directly has no arena, but vector sharing/isolation does.
    if shared != "record":
        assert result.audit["native_chunks"][0]["numeric_arena_bytes"] >= record.nbytes


def test_prepared_serialized_snapshots_reread_inputs_and_own_results():
    a = symmetric(128).copy()
    graph = GraphCompiler({"a": matrix()}).compile("eigenvalues(symmetric_eigh(a))")
    graph = pickle.loads(pickle.dumps(graph))
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a}, bounds, bounds)
        first = prepared.run_snapshot()
        preserved = first.outputs[0].values[0].copy()
        a += np.eye(128)
        second = prepared.run_snapshot()
        exact(second.outputs[0].values[0], op.eigenvalues(op.symmetric_eigh(a)))
        exact(first.outputs[0].values[0], preserved)
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
        plan = scheduler.plan(graph, {"a": a}, bounds, bounds)
        assert plan.estimated_output_bytes == 128 * 8
        with pytest.raises(MemoryError):
            scheduler.plan(graph, {"a": a}, bounds, bounds,
                           memory_budget_bytes=plan.estimated_total_memory_bytes - 1)


def test_covariance_projection_keeps_source_arithmetic_and_shape():
    x = np.random.default_rng(903034).normal(size=(300, 128))
    graph = GraphCompiler({"x": dict(kind="matrix", axes=["time", "asset"],
                                   shape=["T", "N"])}).compile(
        "eigenvalues(symmetric_eigh(covariance(x)))"
    )
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.array([0], np.int64),
                                   np.array([len(x)], np.int64))
    exact(result.outputs[0].values[0], op.eigenvalues(op.symmetric_eigh(op.covariance(x))))


@pytest.mark.parametrize("nested", [False, True])
def test_inherited_isolation_admits_materialized_child_record(nested):
    n = 128
    a = symmetric(n)
    decl = {"a": matrix(),
            "seed": dict(kind="vector", axes=["component"], shape=["N"])}
    projections = [f"eigenvalues(symmetric_eigh(a+{i}))" for i in range(4)]
    step = projections[-1]
    for projection in reversed(projections[:-1]):
        step = f"({projection}+{step})"
    step += "+iterate_x*0"
    expression = f"iterate({step},seed,1e-12,2)"
    if nested:
        expression = f"iterate({expression}+iterate_x*0,seed,1e-12,2)"
    bounds = np.zeros(1, np.int64)
    data = {"a": a, "seed": np.zeros(n)}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        strict_graph = GraphCompiler(decl).compile(expression)
        isolated_graph = GraphCompiler(decl).compile(expression, error_policy="isolate")
        strict = scheduler.plan(strict_graph, data, bounds, bounds)
        isolated = scheduler.plan(isolated_graph, data, bounds, bounds)
        # Convergence/fault validation may execute the unfused child even when
        # the outer policy is strict. Four live borrowed records coexist here.
        minimum = (4 * n * (n + 1) + n * n) * 8
        assert min(strict.estimated_worker_scratch_bytes,
                   isolated.estimated_worker_scratch_bytes) >= minimum
        result = scheduler.execute(isolated_graph, data, bounds, bounds)
        values = [op.eigenvalues(op.symmetric_eigh(a + i)) for i in range(4)]
        expected = values[0] + (values[1] + (values[2] + values[3]))
        exact(result.outputs[0].values[0], expected)
        assert not result.outputs[0].statuses[0].any()
        with pytest.raises(MemoryError):
            scheduler.plan(isolated_graph, data, bounds, bounds,
                           memory_budget_bytes=isolated.estimated_total_memory_bytes - 1)
