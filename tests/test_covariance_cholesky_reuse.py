"""Consumed covariance storage: exact math, failure isolation and result owners."""
import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, excel
from calmetrics_engine import operators as op

from .test_typed_results import execute

DECL = {"x": dict(kind="matrix", axes=["time", "asset"], shape=["T", "D"])}
EXPR = "cholesky(covariance(x))"


def bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def data(n, d, order="c"):
    x = np.random.default_rng(3281).normal(size=(n, d))
    if order == "f":
        x = np.asfortranarray(x)
    elif order == "reverse":
        x = x[::-1, ::-1]
    elif order == "pad":
        owner = np.full((n, 2*d+1), np.nan)
        owner[:, 1::2] = x
        x = owner[:, 1::2]
    return x


@pytest.mark.parametrize("n,d", [(3, 0), (8, 1), (37, 8), (80, 31), (1041, 1024)])
@pytest.mark.parametrize("order", ["c", "f", "pad", "reverse"])
@pytest.mark.parametrize("output_order", ["C", "F"])
def test_consumed_covariance_matches_direct_operators_and_removes_one_arena(n, d, order, output_order):
    x = data(n, d, order)
    saved = x.copy()
    x.flags.writeable = False
    expected = op.cholesky(op.covariance(x))
    graph = GraphCompiler(DECL).compile(EXPR, output_order=output_order)
    result = execute(graph, {"x": x})
    actual = result.outputs[0].values[0]
    bits(actual, expected)
    bits(x, saved)
    assert not actual.flags.writeable and not np.shares_memory(actual, x)
    assert actual.flags.c_contiguous if output_order == "C" else actual.flags.f_contiguous
    chunk = result.audit["native_chunks"][0]
    assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
    assert chunk["direct_output_bytes"] == expected.nbytes
    assert chunk["fused_pointwise_calls"] == 0


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("output_order", ["C", "F"])
def test_consumed_covariance_transport_has_independent_roots_and_intervals(lane, output_order):
    x = data(90, 8, "f")
    graph = GraphCompiler(DECL).compile(EXPR, output_order=output_order)
    result = execute(graph, {"x": x}, [0, 10], [60, 90], lane)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    for value, a, b in zip(result.outputs[0].values, [0, 10], [60, 90], strict=True):
        bits(value, op.cholesky(op.covariance(x[a:b])))
        assert not value.flags.writeable
        assert value.flags.c_contiguous if output_order == "C" else value.flags.f_contiguous
        assert not np.shares_memory(value, x)
    assert not np.shares_memory(*result.outputs[0].values)
    for chunk in result.audit["native_chunks"]:
        assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
        assert chunk["direct_output_bytes"] == chunk["rows"] * result.outputs[0].values[0].nbytes


@pytest.mark.parametrize("mode", ["public_covariance", "shared_covariance", "temporary_source", "isolate"])
@pytest.mark.parametrize("output_order", ["C", "F"])
def test_unproven_private_storage_keeps_original_materialization(mode, output_order):
    x = data(60, 8)
    expressions = [EXPR]
    if mode == "public_covariance":
        expressions.append("covariance(x)")
    elif mode == "shared_covariance":
        expressions.append("trace(covariance(x))")
    elif mode == "temporary_source":
        expressions = ["cholesky(covariance(x*2))"]
    graph = GraphCompiler(DECL).compile(
        expressions, output_order=output_order,
        error_policy="isolate" if mode == "isolate" else "raise",
    )
    result = execute(graph, {"x": x})
    bits(result.outputs[0].values[0], op.cholesky(op.covariance(x*2 if mode == "temporary_source" else x)))
    if mode == "public_covariance":
        bits(result.outputs[1].values[0], op.covariance(x))
        assert not np.shares_memory(result.outputs[0].values[0], result.outputs[1].values[0])
    else:
        assert result.audit["native_chunks"][0]["numeric_arena_bytes"] > 0
    if output_order == "F":
        actual = result.outputs[0].values[0]
        assert actual.flags.f_contiguous and not actual.flags.writeable
        assert result.audit["native_chunks"][0]["numeric_arena_bytes"] >= actual.nbytes


@pytest.mark.parametrize("fault", ["constant", "nan", "inf", "overflow", "panel_singular"])
@pytest.mark.parametrize("output_order", ["C", "F"])
def test_failures_match_unfused_chain_and_prepared_can_run_again(fault, output_order):
    n, d = (1041, 1024) if fault == "panel_singular" else (60, 8)
    x = data(n, d)
    graph = GraphCompiler(DECL).compile(EXPR, output_order=output_order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": x}, np.array([0], np.int64), np.array([n], np.int64))
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        original = x.copy()
        if fault == "constant":
            x[:] = 1.
        elif fault == "nan":
            x[3, 2] = np.nan
        elif fault == "inf":
            x[3, 2] = np.inf
        elif fault == "overflow":
            x *= 1e200
        else:
            x[:, -1] = 0.
        with pytest.raises((ValueError, RuntimeError)) as expected:
            op.cholesky(op.covariance(x))
        with pytest.raises((ValueError, RuntimeError)) as actual:
            prepared.run_audit()
        assert str(actual.value) == str(expected.value)
        bits(first.outputs[0].values[0], saved)
        x[:] = original
        second = prepared.run_snapshot()
        bits(second.outputs[0].values[0], saved)
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
        value = second.outputs[0].values[0]
        assert not value.flags.writeable
        assert value.flags.c_contiguous if output_order == "C" else value.flags.f_contiguous


@pytest.mark.parametrize("d", [0, 1, 33, 1024])
def test_f_pair_planner_retains_panel_budget_without_a_second_matrix(d):
    x = data(d + 17, d)
    bounds = np.array([0], np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        c_graph = GraphCompiler(DECL).compile(EXPR, output_order="C")
        f_graph = GraphCompiler(DECL).compile(EXPR, output_order="F")
        c_plan = scheduler.plan(c_graph, {"x": x}, *bounds)
        f_plan = scheduler.plan(f_graph, {"x": x}, *bounds)
        assert f_plan.estimated_worker_scratch_bytes == c_plan.estimated_worker_scratch_bytes
        assert f_plan.estimated_total_memory_bytes == c_plan.estimated_total_memory_bytes
        with pytest.raises(MemoryError):
            scheduler.plan(f_graph, {"x": x}, *bounds,
                           memory_budget_bytes=f_plan.estimated_total_memory_bytes - 1)


@pytest.mark.parametrize("layout", ["c", "f"])
def test_f_pair_preserves_bits_and_accounts_for_inplace_layout_writes(layout):
    x = data(80, 33, layout)
    x.flags.writeable = False
    c_result = execute(GraphCompiler(DECL).compile(EXPR, output_order="C"), {"x": x})
    f_result = execute(GraphCompiler(DECL).compile(EXPR, output_order="F"), {"x": x})
    actual = f_result.outputs[0].values[0]
    bits(actual, c_result.outputs[0].values[0])
    bits(np.triu(actual, 1), np.zeros_like(actual))
    c_audit = c_result.audit["native_chunks"][0]
    f_audit = f_result.audit["native_chunks"][0]
    assert f_audit["algorithm_copy_bytes"] - c_audit["algorithm_copy_bytes"] == (33 * 33 - 33) * 8
    for key in ("input_copy_bytes", "operator_workspace_capacity_bytes", "accelerate_covchol_calls"):
        assert f_audit[key] == c_audit[key]
    assert f_audit["numeric_arena_bytes"] == f_audit["root_copy_bytes"] == 0


@pytest.mark.parametrize("mode", ["multiple", "iterate"])
def test_f_pair_excluded_graphs_keep_their_c_arena(mode):
    x = data(60, 8)
    expression = [EXPR, "7"] if mode == "multiple" else f"iterate({EXPR},{EXPR},0,1)"
    graph = GraphCompiler(DECL).compile(expression, output_order="F")
    result = execute(graph, {"x": x})
    expected = op.cholesky(op.covariance(x))
    bits(result.outputs[0].values[0], expected)
    assert result.outputs[0].values[0].flags.f_contiguous
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] >= expected.nbytes
    assert result.audit["native_chunks"][0]["root_copy_bytes"] >= expected.nbytes


def test_f_pair_excel_observer_keeps_canonical_values():
    x = data(10, 3, "f")
    graph = GraphCompiler(DECL).compile(EXPR, output_order="F")
    expected = op.cholesky(op.covariance(x))
    reference = excel.plan(graph, {"x": x}).reference()[0]
    assert not reference["error"]
    bits(np.asarray(reference["values"]).reshape(expected.shape), expected)


def test_failed_isolate_root_retains_healthy_covariance_and_scalar():
    x = np.ones((40, 8))
    graph = GraphCompiler(DECL).compile([EXPR, "covariance(x)", "7"], error_policy="isolate")
    result = execute(graph, {"x": x})
    assert result.outputs[0].root_statuses[0] != 0
    assert result.outputs[1].root_statuses[0] == result.outputs[2].root_statuses[0] == 0
    bits(result.outputs[1].values[0], op.covariance(x))
    assert result.outputs[2].values[0] == 7


def test_public_direct_out_still_rejects_input_alias():
    a = np.eye(8)
    with pytest.raises((ValueError, RuntimeError), match="OUTPUT_ALIASES_INPUT"):
        op.cholesky(a, out=a)
