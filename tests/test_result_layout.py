"""Physical result order preserves logical coordinates, exact dtype and ownership."""

import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler, ModelPayload
from tests.test_dense_layout_transport import field_type, source
from tests.test_model_payload import META
from tests.test_tensor_protocol import configuration


@pytest.mark.parametrize("shape", [(5, 3), (4, 3, 2), (0, 3), (1, 3)])
@pytest.mark.parametrize("dtype", ["float64", "int64", "bool"])
@pytest.mark.parametrize("input_order", ["C", "F"])
@pytest.mark.parametrize("output_order", ["C", "F", "auto"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_identity_result_order_and_retained_ownership(
    shape, dtype, input_order, output_order, lane
):
    x = source(shape, dtype, input_order)
    before = x.copy()
    graph = GraphCompiler({"x": field_type(dtype, shape)}).compile(
        {"first": "x", "again": "x"}, output_order=output_order, error_policy="isolate"
    )
    graph = pickle.loads(pickle.dumps(graph))
    starts = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        prepared = (
            engine.prepare_execution(graph, {"x": x}, starts, starts) if lane == "single" else None
        )
        run = (
            prepared.run_snapshot
            if prepared
            else lambda: engine.execute(graph, {"x": x}, starts, starts)
        )
        first = run()
        second = run()
    expected_order = output_order if output_order != "auto" else "C"  # isolated status traversal
    for column in first.outputs:
        assert column.order == expected_order
        for value, status in zip(column.values, column.statuses, strict=True):
            np.testing.assert_array_equal(value, before)
            assert value.dtype == x.dtype and value.shape == shape
            assert value.flags.f_contiguous if expected_order == "F" else value.flags.c_contiguous
            assert not value.flags.writeable and not np.shares_memory(value, x)
            assert not np.shares_memory(value, second.outputs[0].values[0])
            assert status.flags.c_contiguous and not status.any()
    retained = first.outputs[0].values[0]
    del first, second, prepared
    gc.collect()
    np.testing.assert_array_equal(retained, before)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("shape", [(65, 9), (17, 13, 19)])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
@pytest.mark.parametrize("isolate", [False, True])
def test_pointwise_mixed_outputs_physical_traversal(order, shape, layout, isolate):
    x = np.array(
        np.random.default_rng(52).normal(size=shape),
        order="F" if layout in ("F", "padded") else "C",
    )
    if layout == "negative":
        x = x[::-1]
    if layout == "padded":
        x = x[1:]
    expr = "((x*2+1)*3-4)*2+1"
    graph = GraphCompiler({"x": field_type("float64", x.shape)}).compile(
        {"value": expr, "mask": f"{expr}>0", "again": expr, "identity": "x"},
        output_order=order,
        error_policy="isolate" if isolate else "raise",
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=configuration("thread")) as engine:
        first = engine.execute(graph, {"x": x}, bounds, bounds)
    expected = ((x * 2 + 1) * 3 - 4) * 2 + 1
    for col, wanted in zip(first.outputs, (expected, expected > 0, expected, x), strict=True):
        np.testing.assert_array_equal(col.values[0], wanted)
        if order != "auto":
            assert col.order == order
        if order == "F":
            assert col.values[0].flags.f_contiguous
        if isolate:
            assert not col.statuses[0].any()
    assert first.audit["result_protocol"] == (
        "typed-results-1" if order == "C" else "typed-results-2"
    )


@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("lane", ["single", "process_inline", "process_shared"])
def test_mixed_column_chain_f_direct_and_c_only_kernel_conversion(order, lane):
    from calmetrics_engine import operators as op

    x = np.asfortranarray(np.random.default_rng(4).normal(size=(65, 9)))
    types = {"x": dict(kind="matrix", axes=["time", "asset"], shape=["T", "N"])}
    expr = "(x-broadcast_like(mean_axis(x,0),x))/2"
    graph = GraphCompiler(types).compile(
        {"standard": expr, "cov": "covariance(x)"}, output_order=order
    )
    b = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        result = engine.execute(graph, {"x": x}, b, np.full(2, len(x), np.int64))
    for col, expected in zip(
        result.outputs, ((x - op.mean_axis(x, 0)) / 2, op.covariance(x)), strict=True
    ):
        np.testing.assert_allclose(col.values[0], expected, rtol=1e-14, atol=1e-14)
        if order == "F":
            assert col.order == "F" and col.values[0].flags.f_contiguous
    if order == "F":
        assert sum(c["root_copy_bytes"] for c in result.audit["native_chunks"]) >= 9 * 9 * 8 * 2


@pytest.mark.parametrize("order", ["C", "F", "auto"])
def test_iteration_result_conversion_is_independent_of_state_layout(order):
    x = np.full((7, 3), 8.0, order="F")
    expr = "iterate(iterate_x*0.5,x,0,4)"
    graph = GraphCompiler({"x": field_type("float64", x.shape)}).compile(
        {"value": expr, "count": f"iteration_count({expr})"}, output_order=order
    )
    b = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x}, b, b)
    np.testing.assert_array_equal(result.outputs[0].values[0], x / 16)
    assert result.outputs[1].values[0] == 4
    if order == "F":
        assert result.outputs[0].values[0].flags.f_contiguous


def test_output_policy_pickle_legacy_and_native_identity():
    compiler = GraphCompiler({"x": field_type("float64", (5, 3))})
    graphs = [compiler.compile({"value": "x*2"}, output_order=o) for o in ("C", "F", "auto")]
    assert len({g.fingerprint for g in graphs}) == 3
    for graph in graphs:
        assert pickle.loads(pickle.dumps(graph)).fingerprint == graph.fingerprint
    state = list(graphs[2].__getstate__())
    state[0] = 4
    legacy = type(graphs[0]).__new__(type(graphs[0]))
    legacy.__setstate__(tuple(state[:10]))
    assert legacy.fingerprint == graphs[0].fingerprint
    for order in ("A", "K", "f", ""):
        with pytest.raises(GraphCompileError, match="output_order"):
            compiler.compile("x", output_order=order)


def test_f_result_to_model_preserves_physical_bytes_and_identity():
    x = np.array(np.arange(15.0).reshape(5, 3), order="F")
    types = {"x": field_type("float64", x.shape)}
    graph = GraphCompiler(types).compile({"x": "x"}, output_order="F")
    b = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x}, b, b)
    # Both constructors use the same logical schema and physical field bytes.
    model = result.to_model_payload(META)
    direct = ModelPayload.from_fields(types, {"x": x}, META)
    assert model.fields["x"].flags.f_contiguous
    assert model.to_bytes() == direct.to_bytes()


def test_excel_formula_and_reference_coordinates_do_not_depend_on_storage_order(tmp_path):
    import json

    from calmetrics_engine import excel

    x = np.array(np.arange(15.0).reshape(5, 3), order="F")
    compiler = GraphCompiler({"x": field_type("float64", x.shape)})
    plans = []
    for order in ("C", "F", "auto"):
        graph = compiler.compile({"value": "(x-2)*3", "mask": "x>7"}, output_order=order)
        plan = excel.plan(graph, {"x": x})
        assert plan.reference()[0]["values"] == ((x - 2) * 3).ravel().tolist()
        path = tmp_path / (order + ".jsonl")
        plan.write_cells(str(path))
        # Mathematical formulas, layout and inputs stay equal; export identity differs.
        plans.append(
            [c for c in map(json.loads, path.read_text().splitlines()) if c[:3] != ["Readme", 3, 1]]
        )
    assert plans[0] == plans[1] == plans[2]


@pytest.mark.parametrize("order", ["C", "F", "auto"])
def test_nonfinite_values_keep_existing_logical_status_coordinates(order):
    x = np.array(np.arange(12.0).reshape(4, 3), order="F")
    x[2, 1] = np.nan
    compiler = GraphCompiler({"x": field_type("float64", x.shape)})
    graph = compiler.compile(
        {"value": "x+1", "mask": "x+1>2", "independent": "7"},
        output_order=order,
        error_policy="isolate",
    )
    b = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x}, b, b)
    for index, column in enumerate(result.outputs[:2]):
        expected = np.zeros(x.shape, np.int16)
        if index == 0:
            expected[2, 1] = 4
        # NaN comparison is valid False under the existing mathematical contract;
        # a nonfinite input is not a caught operator failure propagated downstream.
        np.testing.assert_array_equal(column.statuses[0], expected)
        assert column.statuses[0].flags.c_contiguous
    assert result.outputs[2].values[0] == 7


@pytest.mark.parametrize("order", ["F", "auto"])
def test_f_root_remains_stride_aware_for_matrix_consumers(order):
    from calmetrics_engine import operators as op

    x = np.asfortranarray(np.random.default_rng(52).normal(size=(65, 9)))
    z = "(x-broadcast_like(mean_axis(x,0),x))/2"
    graph = GraphCompiler({"x": dict(kind="matrix")}).compile(
        {
            "z": z,
            "sum": f"sum_axis({z},0)",
            "cov": f"covariance({z})",
            "transposed": f"transpose({z})",
        },
        output_order=order,
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x}, bounds, np.array([len(x)], np.int64))
    expected = (x - op.mean_axis(x, 0)) / 2
    for col, value in zip(
        result.outputs,
        (expected, op.sum_axis(expected, 0), op.covariance(expected), expected.T),
        strict=True,
    ):
        np.testing.assert_allclose(col.values[0], value, rtol=1e-13, atol=1e-13)
    if order == "F":
        assert result.outputs[0].order == "F"


def test_explicit_f_fallback_workspace_is_budgeted():
    x = np.asfortranarray(np.random.default_rng(9).normal(size=(65, 17)))
    compiler = GraphCompiler({"x": dict(kind="matrix")})
    b, e = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        # Isolated graphs still require the original C arena and F copy.
        cgraph = compiler.compile("correlation(x)", output_order="C", error_policy="isolate")
        fgraph = compiler.compile("correlation(x)", output_order="F", error_policy="isolate")
        cp = engine.plan(cgraph, {"x": x}, b, e)
        fp = engine.plan(fgraph, {"x": x}, b, e)
        assert fp.estimated_worker_scratch_bytes >= cp.estimated_worker_scratch_bytes + 17 * 17 * 8
        with pytest.raises(MemoryError):
            engine.plan(
                fgraph, {"x": x}, b, e, memory_budget_bytes=fp.estimated_total_memory_bytes - 1
            )


@pytest.mark.parametrize("lane", ["single", "process_inline", "process_shared"])
def test_explicit_f_caught_failure_propagates_to_mask(lane):
    x = np.ones((5, 3), order="F")
    x[1, 2] = -1
    graph = GraphCompiler({"x": field_type("float64", x.shape)}).compile(
        {"value": "sqrt(x)", "mask": "sqrt(x)>0", "good": "x*2"},
        error_policy="isolate",
        output_order="F",
    )
    b = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        result = engine.execute(graph, {"x": x}, b, b)
    for col in result.outputs[:2]:
        assert col.order == "F"
        assert np.all(col.statuses[0] == 4)
    np.testing.assert_array_equal(result.outputs[2].values[0], x * 2)
    assert not result.outputs[2].statuses[0].any()


@pytest.mark.parametrize("shape", [(9, 4), (9, 4, 3)])
@pytest.mark.parametrize("dtype", ["float64", "int64"])
@pytest.mark.parametrize("reverse_columns", [False, True])
def test_partial_first_axis_runs_materialize_dense_f(shape, dtype, reverse_columns):
    x = source(shape, dtype, "F")
    if reverse_columns:
        x = x[:, ::-1]
    declaration = field_type(dtype, shape)
    declaration["axes"][0] = "time"
    declaration["shape"][0] = "T"
    graph = GraphCompiler({"x": declaration}).compile({"value": "x"}, output_order="F")
    starts, ends = np.array([2], np.int64), np.array([7], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x}, starts, ends)
    actual = result.outputs[0].values[0]
    np.testing.assert_array_equal(actual, x[2:7])
    assert actual.flags.f_contiguous and not np.shares_memory(actual, x)
    assert sum(c["root_copy_bytes"] for c in result.audit["native_chunks"]) == actual.nbytes
