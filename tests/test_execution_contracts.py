"""Platform-compatible isolation, native bindings and output ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig


def configs(lane):
    if lane == "single":
        return PlannerConfig(thread_work_units=1e99, process_work_units=1e100)
    if lane == "thread":
        return PlannerConfig(
            thread_work_units=1, process_work_units=1e100, min_rows_per_worker=1
        )
    return PlannerConfig(
        thread_work_units=1,
        process_work_units=1,
        min_rows_per_worker=1,
        shared_memory_threshold_bytes=1,
    )


@pytest.mark.parametrize("lane", ["single", "thread", "process"])
def test_bad_root_and_descendants_do_not_poison_other_roots_or_rows(lane):
    graph = GraphCompiler({"x": "series"}).compile(
        ["1/std(x,1)", "mean(x)", "sqrt(-1)", "(1/std(x,1))+2"], error_policy="isolate"
    )
    x = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    with AdaptiveScheduler(cpu_budget=2, config=configs(lane)) as s:
        r = s.execute(
            graph, {"x": x}, np.array([0, 3], np.int64), np.array([3, 6], np.int64)
        )
        assert r.audit["lane"] == lane
        np.testing.assert_array_equal(r.statuses, [[2, 0, 3, 2], [0, 0, 3, 0]])
        np.testing.assert_allclose(
            r.values, [[np.nan, 1, np.nan, np.nan], [1, 3, np.nan, 3]], equal_nan=True
        )
        assert r.statuses.dtype == np.int16
        assert r.audit["python_worker_callbacks"] == 0
        assert r.audit["execution_backend"] == "cpp_aot"


def test_strict_graph_and_standalone_operators_keep_raise_contract():
    from calmetrics_engine import operators

    g = GraphCompiler({"x": "series"}).compile(["1/std(x,1)"])
    with AdaptiveScheduler(cpu_budget=1) as s:
        with pytest.raises(ValueError, match="DIVIDE_BY_ZERO"):
            s.execute(
                g, {"x": np.ones(3)}, np.array([0], np.int64), np.array([3], np.int64)
            )
    with pytest.raises(ValueError, match="DIVIDE_BY_ZERO"):
        operators.divide(1.0, 0.0)


def test_bindings_and_version_namespaces_are_native_and_fingerprinted():
    compiler = GraphCompiler({"nav": "series", "m0": "scalar", "m1": "scalar"})
    expressions = ["mean(returns)*factor"] * 2
    binding = {"returns": "divide(difference(nav,1),lag(nav,1))"}
    g = compiler.compile(
        expressions,
        root_bindings=[{**binding, "factor": "m0"}, {**binding, "factor": "m1"}],
        source_contracts=["dsl2.4:registry2.4.1"] * 2,
        error_policy="isolate",
    )
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(
            g,
            {"nav": np.array([100.0, 110.0, 121.0, 50.0, 55.0])},
            np.array([0, 3], np.int64),
            np.array([3, 5], np.int64),
            parameters={"m0": 2.0, "m1": 3.0},
        )
        np.testing.assert_allclose(r.values, [[0.2, 0.3], [0.2, 0.3]])
    first = compiler.compile(["mean(nav)"], source_contracts=["2.0"])
    second = compiler.compile(["mean(nav)"], source_contracts=["2.4"])
    assert first.fingerprint != second.fingerprint
    shared = compiler.compile(
        ["mean(nav)", "mean(nav)"], source_contracts=["2.4", "2.4"]
    )
    separate = compiler.compile(
        ["mean(nav)", "mean(nav)"], source_contracts=["2.0", "2.4"]
    )
    assert separate.metadata()["node_count"] > shared.metadata()["node_count"]
    with pytest.raises(ValueError, match="cyclic"):
        compiler.compile(["a"], root_bindings=[{"a": "b", "b": "a"}])


def test_prepared_snapshot_and_statuses_survive_reuse_and_close():
    g = GraphCompiler({"x": "series"}).compile(
        ["1/std(x,1)", "mean(x)"], error_policy="isolate"
    )
    x = np.ones(3)
    with AdaptiveScheduler(cpu_budget=1) as s:
        p = s.prepare_execution(
            g, {"x": x}, np.array([0], np.int64), np.array([3], np.int64)
        )
        retained = p.run_snapshot()
        before = retained.statuses
        borrowed = p.run_audit()
        x[:] = [1.0, 2.0, 3.0]
        fresh = p.run_audit()
        assert np.shares_memory(borrowed.values, fresh.values)
        assert not np.shares_memory(retained.values, fresh.values)
        assert retained.audit["result_lifetime"] == "independent"
        assert borrowed.audit["result_lifetime"] == "borrowed_until_next_run"
        assert retained.values.flags.writeable is False
    np.testing.assert_allclose(retained.values, [[np.nan, 1]], equal_nan=True)
    np.testing.assert_array_equal(retained.statuses, [[2, 0]])
    np.testing.assert_array_equal(before, [[2, 0]])


def test_minimum_samples_nonfinite_and_invalid_geometry():
    g = GraphCompiler({"x": "series"}).compile(
        ["mean(x)"], error_policy="isolate", minimum_observations=2
    )
    with AdaptiveScheduler(cpu_budget=1) as s:
        r = s.execute(
            g,
            {"x": np.full(3, np.nan)},
            np.array([0, 0, 0], np.int64),
            np.array([0, 1, 3], np.int64),
        )
        np.testing.assert_array_equal(r.statuses, [[1], [1], [4]])
        with pytest.raises(ValueError):
            s.execute(
                g, {"x": np.ones(3)}, np.array([0], np.int64), np.array([4], np.int64)
            )


def test_operator_sample_errors_are_isolated_with_platform_unavailable_status():
    graph = GraphCompiler({"x": "series"}).compile(
        ["std(x,1)", "mean(x)", "mean_where(x,greater_than(x,100))"],
        error_policy="isolate",
    )
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            graph,
            {"x": np.array([1.0])},
            np.array([0], np.int64),
            np.array([1], np.int64),
        )
    np.testing.assert_array_equal(result.statuses, [[4, 0, 4]])
    np.testing.assert_allclose(result.values, [[np.nan, 1.0, np.nan]], equal_nan=True)


def test_dag_branch_statuses_are_mapped_back_to_root_order():
    graph = GraphCompiler({"x": "series"}).compile(
        ["1/std(x)", "median(x)", "mean(x)", "linear_slope(x)"], error_policy="isolate"
    )
    config = PlannerConfig(
        thread_work_units=1,
        dag_branch_work_units=1,
        process_work_units=1e100,
        min_rows_per_worker=1,
    )
    x = np.ones(10000)
    with AdaptiveScheduler(cpu_budget=4, config=config) as scheduler:
        result = scheduler.execute(
            graph, {"x": x}, np.array([0], np.int64), np.array([len(x)], np.int64)
        )
    assert result.audit["parallel_dimension"] == "dag_branch"
    np.testing.assert_array_equal(result.statuses, [[2, 0, 0, 0]])
    np.testing.assert_allclose(result.values, [[np.nan, 1.0, 1.0, 0.0]], equal_nan=True)


def test_series_statuses_offsets_and_readonly_strided_input():
    x = np.array([1.0, 999.0, -1.0, 999.0, 4.0, 999.0])[::2]
    x.flags.writeable = False
    graph = GraphCompiler({"x": "series"}).compile(
        ["sqrt(x)", "x+1"], error_policy="isolate"
    )
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match="C-contiguous"):
            scheduler.execute(
                graph, {"x": x}, np.array([0, 1], np.int64), np.array([1, 3], np.int64)
            )
        x = np.ascontiguousarray(
            x
        )  # Explicit boundary conversion, never a hidden runtime copy.
        x.flags.writeable = False
        result = scheduler.execute(
            graph, {"x": x}, np.array([0, 1], np.int64), np.array([1, 3], np.int64)
        )
    np.testing.assert_array_equal(result.offsets, [0, 1, 3])
    # A domain error affects this root/interval; other roots remain available.
    np.testing.assert_array_equal(result.statuses, [[0, 0], [4, 0], [4, 0]])
    np.testing.assert_allclose(
        result.values, [[1.0, 2.0], [np.nan, 0.0], [np.nan, 5.0]], equal_nan=True
    )


def test_interval_tail_binding_uses_exact_return_order_and_borrow_liveness():
    graph = GraphCompiler({"nav": "series"}).compile(
        ["mean(r)", "std(r)", "mean(lag(nav))"],
        error_policy="isolate",
        root_bindings=[{"r": "interval_tail(nav)/lag(nav)-1"}] * 3,
    )
    nav = np.array([1.23, 1.33, 1.17])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            graph, {"nav": nav}, np.array([0], np.int64), np.array([3], np.int64)
        )
    returns = nav[1:] / nav[:-1] - 1
    assert result.values[0, 0] == np.mean(returns)
    np.testing.assert_allclose(
        result.values[0], [np.mean(returns), np.std(returns, ddof=1), np.mean(nav[:-1])]
    )
    assert graph.metadata()["error_policy"] == "isolate"
    assert (
        graph.fingerprint
        != GraphCompiler({"nav": "series"})
        .compile(["mean(nav)"], minimum_observations=2)
        .fingerprint
    )
