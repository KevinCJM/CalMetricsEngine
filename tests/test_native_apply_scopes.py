"""Generic native scopes: selection, grouping, recursion bounds and transport."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler, PlannerConfig


def run(graph, values, *, starts=None, ends=None, lane="single", shared=True):
    config = PlannerConfig(
        thread_work_units=1 if lane != "single" else 1e99,
        process_work_units=1 if lane == "process" else 1e100,
        min_rows_per_worker=1,
        shared_memory_threshold_bytes=1 if shared else 1 << 30,
        max_processes=2,
    )
    n = len(next(iter(values.values())))
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        return scheduler.execute(
            graph, values, np.asarray(starts or [0], np.int64),
            np.asarray(ends or [n], np.int64), timeout=20,
        )


def test_complete_blocks_drop_tail_and_reset_each_interval():
    graph = GraphCompiler({"x": "series"}).compile(["sum(block_apply(sum(x),3))"])
    values = np.arange(1.0, 15.0)[::-1].copy()
    values.flags.writeable = False
    result = run(graph, {"x": values}, starts=[0, 7], ends=[7, 14])
    np.testing.assert_allclose(result.values[:, 0], [values[:6].sum(), values[7:13].sum()])


def test_filter_compacts_in_original_order_and_explicit_empty_result():
    graph = GraphCompiler({"x": "series"}).compile([
        "filter_apply(sum(difference(x)),finite_mask(x))",
        "filter_apply(sum(x),x>100,0)",
        "filter_apply(mean(x),x>100)",
    ])
    x = np.array([3.0, np.nan, 8.0, 2.0])
    result = run(graph, {"x": x})
    np.testing.assert_allclose(result.values, [[-1.0, 0.0, np.nan]], equal_nan=True)


def test_scope_captures_derived_arrays_and_shares_repeated_block_nodes():
    graph = GraphCompiler({"x": "series"}).compile(
        ["filter_apply(mean(b),finite_mask(b))+mean(b)"],
        root_bindings=[{"b": "block_apply(sum(x),2)"}],
    )
    result = run(graph, {"x": np.arange(1.0, 8.0)})
    np.testing.assert_allclose(result.values, [[14.0]])
    scopes = [node for node in graph.native_nodes() if node["kind"] == "apply_scope"]
    assert len(scopes) == 2


def test_scope_child_error_respects_strict_and_isolated_root_contract():
    compiler = GraphCompiler({"x": "series"})
    expressions = ["mean(block_apply(1/std(x,0),2))", "mean(x)"]
    with pytest.raises(ValueError, match="DIVIDE_BY_ZERO"):
        run(compiler.compile(expressions), {"x": np.ones(4)})
    result = run(compiler.compile(expressions, error_policy="isolate"), {"x": np.ones(4)})
    assert result.statuses[0, 0] != 0
    assert result.statuses[0, 1] == 0
    np.testing.assert_allclose(result.values, [[np.nan, 1.0]], equal_nan=True)


@pytest.mark.parametrize("lane,shared", [("single", True), ("thread", True), ("process", True), ("process", False)])
def test_group_exact_int64_keys_broadcast_and_process_roundtrip(lane, shared):
    graph = GraphCompiler({"x": "series", "key": {"kind": "series", "dtype": "int64"}}).compile([
        "group_apply(mean(x),key)",
        "group_apply(bisect(solve_x*solve_x-mean(x),0,10,1e-12,100),key)",
    ])
    x = np.array([1., 9., 3., 7., 4., 16., 12., 8.])
    key = np.array([2**60, 2**60 + 1, 2**60, 2**60 + 1] * 2, np.int64)
    x.flags.writeable = key.flags.writeable = False
    result = run(graph, {"x": x, "key": key}, starts=[0, 4], ends=[4, 8], lane=lane, shared=shared)
    expected = np.array([2., 8., 2., 8., 8., 12., 8., 12.])
    np.testing.assert_allclose(result.values, np.column_stack([expected, np.sqrt(expected)]), atol=1e-12)
    assert result.audit["lane"] == lane
    assert result.audit["python_worker_callbacks"] == 0


def test_bisect_failure_modes_and_endpoint_root():
    compiler = GraphCompiler({"x": "series"})
    graph = compiler.compile([
        "bisect(solve_x-2,2,3,1e-12,20)",
        "bisect(solve_x*solve_x+1,0,2,1e-12,100)",
        "bisect(solve_x*solve_x-2,0,2,1e-12,1)",
        "mean(x)",
        "bisect(solve_x*solve_x-2,0,2,1,1)",
    ], error_policy="isolate")
    result = run(graph, {"x": np.array([1., 3.])})
    np.testing.assert_allclose(result.values, [[2., np.nan, np.nan, 2., 1.5]], equal_nan=True)
    np.testing.assert_array_equal(result.statuses, [[0, 4, 4, 0, 0]])


def test_invalid_scope_shapes_types_and_nesting_fail_closed():
    compiler = GraphCompiler({"x": "series"})
    for expression in ["group_apply(mean(x),x)", "filter_apply(mean(x),x)", "block_apply(x,2)"]:
        with pytest.raises(GraphCompileError):
            compiler.compile([expression])
    expression = "mean(x)"
    for _ in range(9):
        expression = f"filter_apply({expression},finite_mask(x))"
    with pytest.raises(GraphCompileError, match="SCOPE_NESTING_LIMIT"):
        compiler.compile([expression])


def test_constant_scope_body_uses_selector_or_interval_extent():
    compiler = GraphCompiler({"x": "series", "key": {"kind": "series", "dtype": "int64"}})
    x = np.array([2., 3., 5., 7., 11.])
    inputs = {"x": x, "key": np.array([1, 1, 2, 2, 2], np.int64)}
    scalar = run(compiler.compile(["filter_apply(9,finite_mask(x))", "sum(block_apply(2,2))"]), inputs)
    np.testing.assert_allclose(scalar.values, [[9., 4.]])
    grouped = run(compiler.compile(["group_apply(3,key)"]), inputs)
    np.testing.assert_array_equal(grouped.values[:, 0], np.full(5, 3.))


def test_group_isolation_retains_successful_groups():
    compiler = GraphCompiler({"x": "series", "key": {"kind": "series", "dtype": "int64"}})
    formula = "group_apply(bisect(solve_x*solve_x-mean(x),0,10,1e-12,100),key)"
    inputs = {"x": np.array([4., -1., 4., -1.]), "key": np.array([1, 2, 1, 2], np.int64)}
    with pytest.raises(ValueError, match="ROOT_NOT_BRACKETED"):
        run(compiler.compile([formula]), inputs)
    result = run(compiler.compile([formula], error_policy="isolate"), inputs)
    np.testing.assert_allclose(result.values[:, 0], [2., np.nan, 2., np.nan], atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(result.statuses[:, 0], [0, 4, 0, 4])


def test_invalid_filter_mask_and_scope_budget_are_structural_errors():
    graph = GraphCompiler({"x": "series", "m": {"kind": "series", "dtype": "bool"}}).compile(
        ["filter_apply(sum(x),m)"], error_policy="isolate")
    with pytest.raises(ValueError, match="INVALID_MASK"):
        run(graph, {"x": np.array([1., 2.]), "m": np.array([0, 2], np.uint8)})
    expression = "filter_apply(sum(x),finite_mask(x))"
    compiler = GraphCompiler({"x": "series"})
    graph = compiler.compile([expression], error_policy="isolate", scope_work_budget=1)
    with pytest.raises(ValueError, match="SCOPE_COMPUTE_BUDGET_EXCEEDED"):
        run(graph, {"x": np.ones(4)})
    expanded = compiler.compile([expression], scope_work_budget=100)
    assert graph.fingerprint != expanded.fingerprint
    np.testing.assert_allclose(run(expanded, {"x": np.ones(4)}).values, [[4.]])
    nested = compiler.compile(
        ["rolling_apply(sum(x)+filter_apply(sum(x),finite_mask(x)),2)"],
        error_policy="isolate", scope_work_budget=1,
    )
    with pytest.raises(ValueError, match="SCOPE_COMPUTE_BUDGET_EXCEEDED"):
        run(nested, {"x": np.ones(4)})


def test_explicit_filter_missing_does_not_poison_unselected_where_branch():
    graph = GraphCompiler({"x": "series"}).compile([
        "where(mean(x)>0,mean(x),filter_apply(sum(x),x>100))",
        "filter_apply(sum(x),x>100)",
        "where(mean(x)>0,mean(x),filter_apply(1/std(x,0),finite_mask(x)))",
    ], error_policy="isolate")
    result = run(graph, {"x": np.ones(4)})
    np.testing.assert_allclose(result.values, [[1., np.nan, np.nan]], equal_nan=True)
    assert result.statuses[0, 0] == 0
    assert result.statuses[0, 1] == 4
    assert result.statuses[0, 2] != 0  # A real divide error is still propagated.


def test_bisect_local_variable_shadows_an_outer_expression_binding():
    graph = GraphCompiler({"x": "series"}).compile(
        ["bisect(solve_x*solve_x-2,0,2,1e-12,100)"],
        root_bindings=[{"solve_x": "100"}],
    )
    np.testing.assert_allclose(run(graph, {"x": np.ones(2)}).values, [[np.sqrt(2)]], atol=1e-12)


def test_group_static_asset_vector_preserves_axis_kind_and_body_semantics():
    # A symbolic length named T does not turn an asset axis into a time axis.
    vector = {"kind": "vector", "axes": ["asset"], "shape": ["T"]}
    compiler = GraphCompiler({
        "x": {**vector, "dtype": "float64", "semantic_dimension": "return_decimal"},
        "key": {**vector, "dtype": "int64"},
    })
    expression = "group_apply(mean(x),key)"
    assert compiler.compile([expression]).metadata()["output_kind"] == "typed"
    graph = compiler.compile([f"sum({expression})"])
    scope = next(node for node in graph.nodes if node.kind == "apply_scope")
    inferred = scope.inferred_type
    assert inferred["kind"] == "vector"
    assert inferred["axes"] == ["asset"]
    assert inferred["shape"] == ["T"]
    assert inferred["dtype"] == "float64"
    assert inferred["semantic_dimension"] == "return_decimal"
    result = run(graph, {"x": np.array([1., 4., 3., 8.]), "key": np.array([1, 2, 1, 2], np.int64)})
    np.testing.assert_allclose(result.values, [[16.]])
