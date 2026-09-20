import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler, PlannerConfig


def typed_series(semantic="dimensionless", *, price_basis=None):
    value = {
        "kind": "series",
        "dtype": "float64",
        "axes": ["time"],
        "shape": ["T"],
        "semantic_dimension": semantic,
    }
    if price_basis is not None:
        value["price_basis"] = price_basis
    return value


def execute(graph, inputs, starts, ends, *, config=None, cpu=1, parameters=None):
    with AdaptiveScheduler(cpu_budget=cpu, config=config) as scheduler:
        return scheduler.execute(
            graph,
            inputs,
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
            parameters=parameters,
            timeout=20,
        )


def test_typed_cpp_ir_metadata_and_aliases_are_canonicalized():
    spec = {"returns": typed_series("return_decimal")}
    alias = GraphCompiler(spec).compile(["mean(sub(returns,0))"])
    canonical = GraphCompiler(spec).compile(["mean(subtract(returns,0))"])

    assert alias.fingerprint == canonical.fingerprint
    metadata = alias.metadata()
    assert metadata["typed_ir_version"] == "cpp-typed-ir-1"
    assert metadata["variable_types"]["returns"]["axes"] == ["time"]
    assert metadata["variable_types"]["returns"]["semantic_dimension"] == "return_decimal"
    assert metadata["root_types"][0]["kind"] == "scalar"


def test_typed_cpp_ir_rejects_semantic_price_and_mask_mismatches():
    with pytest.raises(GraphCompileError, match="PRICE_BASIS_MISMATCH"):
        GraphCompiler(
            {
                "a": typed_series("adjusted_nav", price_basis="hfq"),
                "b": typed_series("adjusted_nav", price_basis="qfq"),
            }
        ).compile(["add(a,b)"])

    with pytest.raises(GraphCompileError, match="SEMANTIC_DIMENSION_MISMATCH"):
        GraphCompiler(
            {
                "returns": typed_series("return_decimal"),
                "days": typed_series("calendar_days"),
            }
        ).compile(["add(returns,days)"])

    with pytest.raises(GraphCompileError, match="TYPE_MISMATCH"):
        GraphCompiler({"x": typed_series()}).compile(["add(finite_mask(x),x)"])


def test_series_root_returns_contiguous_values_and_interval_offsets():
    graph = GraphCompiler({"x": "series"}).compile(["rolling_mean(x,3)"])
    values = np.arange(1.0, 13.0)
    result = execute(graph, {"x": values}, [0, 6], [6, 12])

    assert result.output_kind == "series"
    np.testing.assert_array_equal(result.offsets, np.array([0, 6, 12], dtype=np.int64))
    assert result.values.shape == (12, 1)
    expected = np.array(
        [np.nan, np.nan, 2, 3, 4, 5, np.nan, np.nan, 8, 9, 10, 11],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.values[:, 0], expected, equal_nan=True)


def test_rolling_window_std_uses_reduction_default_ddof_one():
    graph = GraphCompiler({"x": typed_series()}).compile(
        ["std(rolling_window(x,3))", "rolling_std(x,3)"]
    )
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = execute(graph, {"x": x}, [0], [4])
    np.testing.assert_allclose(
        result.values[:, 0],
        np.array([np.nan, np.nan, 1.0, 1.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.values[:, 1],
        np.array([np.nan, np.nan, np.sqrt(2.0 / 3.0), np.sqrt(2.0 / 3.0)]),
        equal_nan=True,
    )


def test_rolling_window_is_compiler_fused_without_materializing_window_matrix():
    graph = GraphCompiler({"x": typed_series("return_decimal")}).compile(
        [
            "mean(rolling_window(x,3,2))",
            "variance(rolling_window(x,3,2),1)",
        ]
    )
    metadata = graph.metadata()
    assert metadata["output_kind"] == "series"
    assert metadata["rolling_scope_count"] == 0

    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = execute(graph, {"x": x}, [0], [5])
    np.testing.assert_allclose(
        result.values[:, 0],
        np.array([np.nan, 1.5, 1.5, 3.0, 4.5]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.values[:, 1],
        np.array([np.nan, 0.5, 0.5, 2.0, 0.5]),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_rolling_apply_resets_path_state_for_every_window():
    graph = GraphCompiler(
        {"nav": typed_series("adjusted_nav", price_basis="hfq")}
    ).compile(["rolling_apply(min_value(drawdown_series(nav)),3)"])
    nav = np.array([100.0, 90.0, 95.0, 120.0, 108.0, 114.0])
    result = execute(graph, {"nav": nav}, [0], [6])

    assert graph.metadata()["rolling_scope_count"] == 1
    np.testing.assert_allclose(
        result.values[:, 0],
        np.array([np.nan, np.nan, -0.1, 0.0, -0.1, -0.1]),
        equal_nan=True,
    )


def test_rolling_apply_min_periods_counts_joint_finite_rows_without_compression():
    graph = GraphCompiler({"x": typed_series()}).compile(
        ["rolling_apply(mean_where(x,finite_mask(x)),3,2)"]
    )
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = execute(graph, {"x": x}, [0], [5])
    np.testing.assert_allclose(
        result.values[:, 0],
        np.array([np.nan, 1.5, 1.5, 3.0, 4.5]),
        equal_nan=True,
    )


def test_rolling_apply_uses_preceding_level_observation_with_return_windows():
    spec = {
        "returns": typed_series("return_decimal"),
        "nav": {
            "kind": "series",
            "dtype": "float64",
            "axes": ["time"],
            "shape": ["L"],
            "semantic_dimension": "adjusted_nav",
            "price_basis": "hfq",
        },
    }
    graph = GraphCompiler(spec).compile(
        [
            "rolling_apply("
            "add(mean(returns),multiply(mean(divide(difference(nav,1),lag(nav,1))),0)),"
            "3)"
        ]
    )
    returns = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1])
    nav = np.array([100.0, 110.0, 121.0, 133.1, 146.41, 161.051])
    result = execute(graph, {"returns": returns, "nav": nav}, [0], [6])
    np.testing.assert_allclose(
        result.values[:, 0],
        np.array([np.nan, np.nan, np.nan, 0.1, 0.1, 0.1]),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_rolling_apply_observation_count_is_internal_context():
    spec = {
        "returns": typed_series("return_decimal"),
        "observation_count": {
            "kind": "scalar",
            "dtype": "float64",
            "semantic_dimension": "count",
        },
    }
    graph = GraphCompiler(spec).compile(
        ["rolling_apply(divide(mean(returns),observation_count),3)"]
    )
    assert graph.parameter_names == ()
    returns = np.full(5, 0.12, dtype=np.float64)
    result = execute(graph, {"returns": returns}, [0], [5])
    np.testing.assert_allclose(
        result.values[:, 0],
        np.array([np.nan, np.nan, 0.04, 0.04, 0.04]),
        equal_nan=True,
    )


def test_rolling_apply_short_form_injects_bettersaataa_system_context_when_declared():
    spec = {
        "x": typed_series(),
        "observation_dates": typed_series("date"),
        "annual_risk_free_rate_decimal": {
            "kind": "scalar",
            "dtype": "float64",
            "semantic_dimension": "rate_decimal",
        },
    }
    graph = GraphCompiler(spec).compile(["rolling_apply(mean(x),3)"])
    assert graph.input_names == ("x", "observation_dates")
    assert graph.parameter_names == ("annual_risk_free_rate_decimal",)
    x = np.arange(1.0, 6.0)
    dates = np.arange(10.0, 15.0)
    result = execute(
        graph,
        {"x": x, "observation_dates": dates},
        [0],
        [5],
        parameters={"annual_risk_free_rate_decimal": 0.03},
    )
    np.testing.assert_allclose(
        result.values[:, 0],
        np.array([np.nan, np.nan, 2.0, 3.0, 4.0]),
        equal_nan=True,
    )


def test_rolling_apply_date_context_is_derived_inside_cpp_scope():
    spec = {
        "returns": typed_series("return_decimal"),
        "dates": typed_series("date"),
        "annual": {
            "kind": "scalar",
            "dtype": "float64",
            "semantic_dimension": "rate_decimal",
        },
        "window_elapsed_days": {
            "kind": "scalar",
            "dtype": "float64",
            "semantic_dimension": "calendar_days",
        },
    }
    graph = GraphCompiler(spec).compile(
        [
            "rolling_apply("
            "divide(mean(returns),add(window_elapsed_days,1)),"
            "3,dates,annual)"
        ]
    )
    assert graph.parameter_names == ("annual",)

    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    dates = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    result = execute(
        graph,
        {"returns": returns, "dates": dates},
        [0],
        [5],
        parameters={"annual": 0.03},
    )
    np.testing.assert_allclose(
        result.values[:, 0],
        np.array([np.nan, np.nan, 0.005, 0.005, 0.005]),
        equal_nan=True,
    )


def test_rolling_apply_rejects_nested_or_history_required_scopes():
    with pytest.raises(GraphCompileError, match="ROLLING_AGGREGATION_REQUIRED"):
        GraphCompiler({"x": "series"}).compile(["rolling_apply(last(x),3)"])

    with pytest.raises(GraphCompileError, match="ROLLING_NESTED_SCOPE_UNSUPPORTED"):
        GraphCompiler({"x": "series"}).compile(
            ["rolling_apply(mean(rolling_window(x,3)),5)"]
        )

    with pytest.raises(GraphCompileError, match="ROLLING_INTERVAL_POLICY_REQUIRED"):
        GraphCompiler({"x": "series"}).compile(
            ["rolling_apply(last(recursive_smooth(x,3,0)),5)"]
        )


def test_series_execution_matches_single_thread_and_process_lanes():
    graph = GraphCompiler({"x": "series"}).compile(["rolling_apply(mean(x),3)"])
    values = np.arange(1.0, 33.0)
    starts = [0, 8, 16, 24]
    ends = [8, 16, 24, 32]

    single = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)
    thread = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1e100,
        min_rows_per_worker=1,
    )
    process = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1.0,
        shared_memory_threshold_bytes=1,
        min_rows_per_worker=1,
        max_processes=4,
    )

    expected = execute(
        graph, {"x": values}, starts, ends, config=single, cpu=4
    )
    threaded = execute(
        graph, {"x": values}, starts, ends, config=thread, cpu=4
    )
    processed = execute(
        graph, {"x": values}, starts, ends, config=process, cpu=4
    )

    assert threaded.plan.lane == "thread"
    assert processed.plan.lane == "process"
    assert processed.plan.use_shared_memory
    np.testing.assert_array_equal(threaded.offsets, expected.offsets)
    np.testing.assert_array_equal(processed.offsets, expected.offsets)
    np.testing.assert_allclose(threaded.values, expected.values, equal_nan=True)
    np.testing.assert_allclose(processed.values, expected.values, equal_nan=True)


def test_nonshared_process_series_lane_matches_single_lane():
    graph = GraphCompiler({"x": "series"}).compile(["rolling_apply(mean(x),3)"])
    values = np.arange(1.0, 17.0)
    starts = [0, 8]
    ends = [8, 16]
    single = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)
    process = PlannerConfig(
        thread_work_units=1.0,
        process_work_units=1.0,
        shared_memory_threshold_bytes=1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )
    expected = execute(graph, {"x": values}, starts, ends, config=single, cpu=2)
    actual = execute(graph, {"x": values}, starts, ends, config=process, cpu=2)
    assert actual.plan.lane == "process"
    assert not actual.plan.use_shared_memory
    np.testing.assert_array_equal(actual.offsets, expected.offsets)
    np.testing.assert_allclose(actual.values, expected.values, equal_nan=True)


def test_low_level_program_execute_validates_series_output_shape():
    graph = GraphCompiler({"x": "series"}).compile(["rolling_mean(x,3)"])
    values = np.arange(1.0, 9.0)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([8], dtype=np.int64)
    output = np.empty((8, 1), dtype=np.float64)
    graph._program.execute((values,), starts, ends, output)
    np.testing.assert_allclose(
        output[:, 0],
        np.array([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="batch shape mismatch"):
        graph._program.execute(
            (values,), starts, ends, np.empty((1, 1), dtype=np.float64)
        )


def test_prepared_series_execution_reuses_output_and_exposes_offsets():
    graph = GraphCompiler({"x": "series"}).compile(["rolling_apply(mean(x),3)"])
    values = np.arange(1.0, 9.0)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([8], dtype=np.int64)
    config = PlannerConfig(thread_work_units=1e99, process_work_units=1e100)
    with AdaptiveScheduler(cpu_budget=1, config=config) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": values}, starts, ends)
        first = prepared.run()
        snapshot = first.copy()
        second = prepared.run()
    assert first is second
    np.testing.assert_array_equal(prepared.offsets, np.array([0, 8], dtype=np.int64))
    np.testing.assert_allclose(second, snapshot, equal_nan=True)


def test_mixed_scalar_and_series_roots_fail_closed():
    with pytest.raises(GraphCompileError, match="MIXED_ROOT_TYPES"):
        GraphCompiler({"x": "series"}).compile(["mean(x)", "rolling_mean(x,3)"])
