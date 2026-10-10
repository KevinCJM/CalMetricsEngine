"""Independent F Cholesky roots retain canonical arithmetic and ownership."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, excel
from calmetrics_engine import operators as op

from .test_cholesky_panel_library import DECL, assert_bits, spd
from .test_matmul_library import settings


def compiled(expression="cholesky(a)", *, order="F", isolate=False):
    return GraphCompiler(DECL).compile(
        expression, output_order=order,
        error_policy="isolate" if isolate else "raise",
    )


def bounds(rows=1):
    return np.zeros(rows, np.int64), np.zeros(rows, np.int64)


def assert_direct(result, expected):
    assert result.outputs[0].order == "F"
    for value in result.outputs[0].values:
        assert value.flags.f_contiguous and not value.flags.writeable
        assert_bits(value, expected)
        assert_bits(np.triu(value, 1), np.zeros_like(value))
    for chunk in result.audit["native_chunks"]:
        assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
        assert chunk["direct_output_bytes"] == chunk["rows"] * expected.nbytes


@pytest.mark.parametrize("layout", ["c", "f", "negative", "padded", "signed_zero"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_f_root_bits_transport_input_and_result_lifetime(layout, lane):
    a = spd(17)
    if layout == "f":
        a = np.asfortranarray(a)
    elif layout == "negative":
        a = a[::-1, ::-1]
    elif layout == "padded":
        owner = np.full((19, 23), np.nan)
        owner[:17, :17] = a
        a = owner[:17, :17]
    elif layout == "signed_zero":
        a = np.eye(17)
        a[a == 0] = -0.
    a.flags.writeable = False
    before = a.copy()
    expected = op.cholesky(a, simd="scalar")
    graph = pickle.loads(pickle.dumps(compiled()))
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, {"a": a}, *bounds(2))
        again = scheduler.execute(graph, {"a": a}, *bounds(2))
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    assert_direct(result, expected)
    assert_direct(again, expected)
    values = result.outputs[0].values
    assert not np.shares_memory(values[0], values[1])
    assert not np.shares_memory(values[0], again.outputs[0].values[0])
    assert not np.shares_memory(values[0], a)
    assert_bits(a, before)


@pytest.mark.parametrize("n", [0, 1, 33, 1025])
@pytest.mark.parametrize("layout", ["c", "f"])
def test_f_root_matches_c_root_without_changing_panel_or_copy_audit(n, layout):
    a = np.array(spd(n), order=layout.upper())
    a.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        c_result = scheduler.execute(compiled(order="C"), {"a": a}, *bounds())
        f_result = scheduler.execute(compiled(), {"a": a}, *bounds())
    assert_direct(f_result, c_result.outputs[0].values[0])
    for c_chunk, f_chunk in zip(c_result.audit["native_chunks"], f_result.audit["native_chunks"], strict=True):
        for field in ("algorithm_copy_bytes", "operator_workspace_capacity_bytes", "input_copy_bytes"):
            assert f_chunk[field] == c_chunk[field]


def test_prepared_snapshots_read_live_inputs_and_survive_a_failed_run():
    a = spd(33)
    a.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(compiled(), {"a": a}, *bounds())
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        a.flags.writeable = True
        a[0, 0] = -1.
        a.flags.writeable = False
        with pytest.raises(ValueError, match="DOMAIN_ERROR"):
            prepared.run_snapshot()
        a.flags.writeable = True
        a[:] = spd(33) * 4
        a.flags.writeable = False
        second = prepared.run_snapshot()
    assert_bits(first.outputs[0].values[0], saved)
    assert_direct(second, op.cholesky(a, simd="scalar"))
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])


@pytest.mark.parametrize("n", [0, 17, 1024])
def test_planner_f_root_budget_matches_c_and_keeps_the_panel(n):
    a = np.eye(n)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        c_plan = scheduler.plan(compiled(order="C"), {"a": a}, *bounds())
        f_plan = scheduler.plan(compiled(), {"a": a}, *bounds())
        assert f_plan.estimated_worker_scratch_bytes == c_plan.estimated_worker_scratch_bytes
        assert f_plan.estimated_total_memory_bytes == c_plan.estimated_total_memory_bytes
        with pytest.raises(MemoryError):
            scheduler.plan(compiled(), {"a": a}, *bounds(),
                           memory_budget_bytes=f_plan.estimated_total_memory_bytes - 1)


@pytest.mark.parametrize("mode", ["multiple", "consumer", "isolate", "iterate"])
def test_unqualified_f_graphs_keep_the_c_arena_and_materialization(mode):
    a = spd(17)
    expression = "cholesky(a)"
    if mode == "multiple":
        expression = [expression, "7"]
    elif mode == "consumer":
        expression = [expression, f"trace({expression})"]
    elif mode == "iterate":
        expression = "iterate(cholesky(a),cholesky(a),0,1)"
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(compiled(expression, isolate=mode == "isolate"),
                                   {"a": a}, *bounds())
    assert_bits(result.outputs[0].values[0], op.cholesky(a))
    assert result.outputs[0].values[0].flags.f_contiguous
    assert sum(c["numeric_arena_bytes"] for c in result.audit["native_chunks"]) >= a.nbytes


def test_covariance_cholesky_pair_keeps_math_in_its_independent_f_root():
    a = np.random.default_rng(81617).normal(size=(33, 17))
    decl = {"a": dict(kind="matrix", axes=["time", "asset"], shape=["T", "N"])}
    graph = GraphCompiler(decl).compile("cholesky(covariance(a))", output_order="F")
    expected = op.cholesky(op.covariance(a))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"a": a},
                                   np.array([0], np.int64), np.array([len(a)], np.int64))
    assert_direct(result, expected)


@pytest.mark.parametrize("case,error", [("nonfinite", "NON_FINITE_RESULT"),
                                      ("asymmetry", "DOMAIN_ERROR"),
                                      ("domain_before_overflow", "DOMAIN_ERROR")])
def test_f_root_preserves_canonical_failure_priority(case, error):
    n = 1025 if case == "domain_before_overflow" else 17
    a = np.eye(n)
    if case == "nonfinite":
        a[0, 0] = -1.
        a[-2, -1] = np.nan
    elif case == "asymmetry":
        a[0, -1] = np.nextafter(0., 1.)
    else:
        a[16, 16] = -1.
        a[0, 0] = np.nextafter(0., 1.)
        a[0, 17] = a[17, 0] = 1e308
    a.flags.writeable = False
    for order in ("C", "F"):
        with AdaptiveScheduler(cpu_budget=1) as scheduler, pytest.raises(ValueError, match=error):
            scheduler.execute(compiled(order=order), {"a": a}, *bounds())


def test_direct_out_contract_and_excel_observer_remain_canonical():
    a = spd(5)
    out = np.full(a.shape, -777., order="F")
    with pytest.raises((ValueError, TypeError)):
        op.cholesky(a, out=out)
    assert np.all(out == -777.)
    expected = op.cholesky(a, simd="scalar")
    # Excel rebuilds typed root schemas with default C order before calling
    # execute_observed. Python cannot request an F observer descriptor there.
    reference = excel.plan(compiled(), {"a": a}).reference()[0]
    assert not reference["error"]
    assert_bits(np.asarray(reference["values"]).reshape(a.shape), expected)
