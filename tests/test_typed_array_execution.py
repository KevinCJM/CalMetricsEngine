"""Typed numerical/index input boundaries across the actual native lanes."""
import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import (
    AdaptiveScheduler,
    GraphCompileError,
    GraphCompiler,
    PlannerConfig,
    SharedInputBundle,
)
from calmetrics_engine import (
    operators as op,
)


def declaration(kind, dtype="float64", semantic="dimensionless", **extra):
    return dict(kind=kind, dtype=dtype, semantic_dimension=semantic, **extra)


def lane_config(lane):
    return PlannerConfig(
        thread_work_units=1e100 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("strided", [False, True, "negative"])
def test_matrix_and_static_vector_inputs_preserve_interval_axes(lane, strided):
    source = np.arange(1.0, 193.0).reshape(32, 6)
    matrix = source[::2, ::2] if strided else source[:16, :3].copy()
    if strided == "negative":
        matrix = matrix[::-1, ::-1]
    weights = np.array([0.2, 0.3, 0.5])
    matrix.setflags(write=False)
    weights.setflags(write=False)
    graph = GraphCompiler({"x": declaration("matrix"), "w": declaration("vector")}).compile(
        ["sum(matvec(x,w))", "mean(sum_asset(x))", "sum(mean_time(x))"]
    )
    starts, ends = np.arange(4, dtype=np.int64) * 4, np.arange(1, 5, dtype=np.int64) * 4
    expected = [[(matrix[a:z] @ weights).sum(), matrix[a:z].sum(axis=1).mean(),
                 matrix[a:z].mean(axis=0).sum()] for a, z in zip(starts, ends, strict=True)]
    with AdaptiveScheduler(cpu_budget=2, config=lane_config(lane)) as scheduler:
        result = scheduler.execute(graph, {"x": matrix, "w": weights}, starts, ends)
    np.testing.assert_allclose(result.values, expected)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    assert result.plan.estimated_input_bytes == matrix.nbytes + weights.nbytes
    assert all(chunk["input_copy_bytes"] == 0 for chunk in result.audit["native_chunks"])
    if lane.startswith("process"):
        assert result.audit["boundary_copy_bytes"] > 0
    else:
        assert result.audit["boundary_copy_bytes"] == 0
    np.testing.assert_array_equal(source, np.arange(1.0, 193.0).reshape(32, 6))


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_exact_int64_category_and_mask_transport(lane):
    ids = np.array([2**53 + 1, 2**53 + 2, 2**53 + 1, -(2**63),
                    2**63 - 1, 2**63 - 1, 7, 8], dtype=np.int64)
    mask = np.array([True, True, True, False, True, False, True, True])
    graph = GraphCompiler({"ids": declaration("series", "int64", "category"),
                           "valid": declaration("series", "bool", "mask")}).compile(
        ["distinct_count(ids)", "distinct_count(ids,valid)"]
    )
    starts, ends = np.array([0, 4], np.int64), np.array([4, 8], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=lane_config(lane)) as scheduler:
        result = scheduler.execute(graph, {"ids": ids, "valid": mask}, starts, ends)
    np.testing.assert_array_equal(result.values, [[3, 2], [3, 3]])
    assert result.plan.estimated_input_bytes == ids.nbytes + mask.nbytes
    assert result.audit["input_dtypes"] == {"ids": "int64", "valid": "bool"}
    assert result.audit["python_fallback"] == 0


def test_stable_integer_sort_gather_and_direct_output_dtype():
    ids = np.array([2**63 - 1, 2**53 + 2, 2**53 + 1, -(2**63), 2**53 + 1], dtype=np.int64)
    indices, audit = op.argsort(ids[::-1], audit=True)
    np.testing.assert_array_equal(indices, np.argsort(ids[::-1], kind="stable"))
    assert indices.dtype == np.int64
    assert op.get("argsort").requirements(ids)["kind"] == "int64"
    assert audit["input_copy_bytes"] == 0
    ordered = op.gather(ids[::-1], indices)
    np.testing.assert_array_equal(ordered, np.sort(ids, kind="stable"))
    assert ordered.dtype == np.int64
    assert op.distinct_count(ids) == 4
    for bad in [np.array([-1], np.int64), np.array([len(ids)], np.int64)]:
        with pytest.raises(ValueError, match="INDEX_OUT_OF_BOUNDS"):
            op.gather(ids, bad)
    with pytest.raises(TypeError):
        op.sum(ids)


def test_integer_intermediates_use_native_arena_and_float_projection():
    graph = GraphCompiler({"x": "series"}).compile("sum(gather(x,argsort(x)))")
    x = np.array([7.0, 2.0, 4.0, 1.0])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, np.array([0], np.int64), np.array([4], np.int64))
    assert result.values[0, 0] == 14
    assert graph.metadata()["integer_slots"] >= 1
    # Arena audit reports retained thread-local capacity from previous calls;
    # the plan estimates this request's required working storage, not cached RSS.
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] >= x.nbytes


def test_shared_typed_bundle_results_outlive_all_input_owners():
    graph = GraphCompiler({"x": declaration("matrix"), "ids": declaration("series", "int64", "category")}).compile(
        ["mean(sum_asset(x))", "distinct_count(ids)"]
    )
    x = np.arange(48.0).reshape(16, 3)
    ids = np.tile(np.array([2**53 + 1, 2**53 + 2], np.int64), 8)
    starts, ends = np.array([0, 8], np.int64), np.array([8, 16], np.int64)
    bundle = SharedInputBundle.from_inputs({"x": x, "ids": ids})
    with AdaptiveScheduler(cpu_budget=2, config=lane_config("process_shared")) as scheduler:
        result = scheduler.execute(graph, bundle, starts, ends)
        for array in bundle.arrays.values():
            assert not array.flags.writeable
        bundle.release()
    gc.collect()
    np.testing.assert_allclose(result.values, [[x[:8].sum(axis=1).mean(), 2], [x[8:].sum(axis=1).mean(), 2]])
    assert result.audit["boundary_copy_bytes"] == starts.nbytes + ends.nbytes


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_prepared_matrix_shape_and_stride_changes_fail_closed():
    graph = GraphCompiler({"x": declaration("matrix")}).compile("mean(sum_asset(x))")
    x = np.arange(12.0).reshape(4, 3)
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": x}, starts, ends)
        retained = prepared.run_snapshot()
        expected = retained.values.copy()
        x[:] += 10
        prepared.run()
        np.testing.assert_array_equal(retained.values, expected)
        # Keep the same owner/pointer: reshape() would create a different view
        # and no longer test mutation of the already-bound descriptor.
        x.shape = (6, 2)
        with pytest.raises(ValueError, match="PREPARED_INPUT_CHANGED"):
            prepared.run()


def test_declared_dtype_and_symbolic_dimensions_are_checked():
    graph = GraphCompiler({"x": declaration("matrix"), "w": declaration("vector")}).compile("sum(matvec(x,w))")
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match="symbolic dimensions"):
            scheduler.execute(graph, {"x": np.ones((4, 3)), "w": np.ones(2)}, starts, ends)
        with pytest.raises(TypeError, match="exact native dtype"):
            scheduler.execute(graph, {"x": np.ones((4, 3), np.float32), "w": np.ones(3)}, starts, ends)
    with pytest.raises(GraphCompileError, match="TYPE_MISMATCH"):
        GraphCompiler({"x": declaration("matrix", "int64", "category")}).compile("mean(sum_asset(x))")


def test_matrix_arena_budget_includes_quadratic_output():
    graph = GraphCompiler({"x": declaration("matrix")}).compile("trace(covariance(x))")
    x = np.arange(30.0).reshape(3, 10)
    starts, ends = np.array([0], np.int64), np.array([3], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x}, starts, ends)
        assert result.plan.estimated_worker_scratch_bytes >= 10 * 10 * 8
        with pytest.raises(MemoryError):
            scheduler.execute(graph, {"x": x}, starts, ends, memory_budget_bytes=x.nbytes)
    assert result.values[0, 0] == pytest.approx(np.trace(np.cov(x, rowvar=False)))


@pytest.mark.parametrize("shape, expected", [((0, 3), 0.0), ((4, 0), 4.0)])
def test_empty_matrix_axes_keep_explicit_time_length(shape, expected):
    graph = GraphCompiler({"x": declaration("matrix"), "w": declaration("vector")}).compile("length(matvec(x,w))")
    x, w = np.empty(shape), np.ones(shape[1])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": x, "w": w}, np.array([0], np.int64), np.array([shape[0]], np.int64))
        with pytest.raises(ValueError, match="intervals"):
            scheduler.execute(graph, {"x": x, "w": w}, np.array([0], np.int64), np.array([shape[0] + 1], np.int64))
    assert result.values[0, 0] == expected


def test_reused_plan_rejects_same_element_count_changed_matrix_geometry():
    graph = GraphCompiler({"x": declaration("matrix")}).compile("mean(sum_asset(x))")
    starts, ends = np.array([0], np.int64), np.array([3], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, {"x": np.ones((4, 3))}, starts, ends)
        with pytest.raises(ValueError, match="stale execution plan"):
            scheduler.execute(graph, {"x": np.ones((6, 2))}, starts, ends, plan=plan)


def test_typed_graph_pickle_preserves_contract_and_bound_roots():
    graph = GraphCompiler({"x": declaration("matrix"),
                           "ids": declaration("series", "int64", "category")}).compile(
        ["mean(sum_asset(x))+adjustment", "distinct_count(ids)"],
        root_bindings=[{"adjustment": "2"}, {}], error_policy="isolate",
        source_contracts=["typed-matrix-v1", "typed-ids-v1"], minimum_observations=2,
        scope_work_budget=123456,
    )
    restored = pickle.loads(pickle.dumps(graph))
    assert restored.fingerprint == graph.fingerprint
    assert restored.metadata()["variable_types"] == graph.metadata()["variable_types"]
    assert restored.metadata()["source_contracts"] == ["typed-matrix-v1", "typed-ids-v1"]
    assert restored.metadata()["minimum_observations"] == 2
    assert restored.metadata()["scope_work_budget"] == 123456
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(restored, {"x": np.arange(6.0).reshape(3, 2),
                                   "ids": np.array([2**53 + 1, 2**53 + 2, 2**53 + 1], np.int64)},
                                   np.array([0], np.int64), np.array([3], np.int64))
    np.testing.assert_allclose(result.values, [[7, 2]])
    np.testing.assert_array_equal(result.statuses, [[0, 0]])


def test_planner_accounts_static_vector_work_independently_of_interval_length():
    graph = GraphCompiler({"v": declaration("vector")}).compile("sum(v)")
    starts, ends = np.array([0], np.int64), np.array([1], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        small = scheduler.plan(graph, {"v": np.ones(3)}, starts, ends)
        large = scheduler.plan(graph, {"v": np.ones(100_000)}, starts, ends)
    assert large.estimated_work_units > small.estimated_work_units * 10_000
    assert large.estimated_typed_array_work_units > 100_000
    assert "typed_array_shape_work_accounted" in large.reason_codes


@pytest.mark.parametrize("operation", ["covariance", "solve", "matmul"])
def test_planner_matrix_operation_work_reflects_dimensions(operation):
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    if operation == "covariance":
        graph = GraphCompiler({"x": declaration("matrix")}).compile("trace(covariance(x))")

        def inputs(n):
            return {"x": np.ones((4, n))}
    elif operation == "solve":
        graph = GraphCompiler({"a": declaration("matrix", axes=["asset", "asset"], shape=["N", "N"]),
                               "b": declaration("vector")}).compile("sum(solve(a,b))")

        def inputs(n):
            return {"a": np.eye(n), "b": np.ones(n)}
    else:
        graph = GraphCompiler({"x": declaration("matrix"),
                               "a": declaration("matrix", axes=["asset", "asset"], shape=["N", "N"])}).compile("sum(matmul(x,a))")

        def inputs(n):
            return {"x": np.ones((4, n)), "a": np.ones((n, n))}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        small = scheduler.plan(graph, inputs(4), starts, ends)
        large = scheduler.plan(graph, inputs(8), starts, ends)
    assert large.estimated_work_units > small.estimated_work_units * 3
    assert large.estimated_typed_array_work_units > small.estimated_typed_array_work_units


def test_scope_matrix_workspace_uses_captured_shape_for_admission():
    graph = GraphCompiler({"v": declaration("vector")}).compile(
        # Scope captures use a local sequence axis; gather explicitly yields an
        # asset vector before constructing the dense matrix.
        "filter_apply(sum(outer(gather(v,argsort(v)),gather(v,argsort(v)))),finite_mask(v))"
    )
    n = 128
    values = np.ones(n)
    starts, ends = np.array([0], np.int64), np.array([1], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, {"v": values}, starts, ends)
        assert plan.estimated_worker_scratch_bytes >= n * n * 8
        small = scheduler.plan(graph, {"v": np.ones(n // 2)}, starts, ends)
        assert plan.estimated_work_units > small.estimated_work_units * 3
        assert plan.estimated_typed_array_work_units >= n * n
        with pytest.raises(MemoryError):
            scheduler.execute(graph, {"v": values}, starts, ends, memory_budget_bytes=n * n * 8 - 1)
        result = scheduler.execute(graph, {"v": values}, starts, ends)
    assert result.values[0, 0] == n * n
