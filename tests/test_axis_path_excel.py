"""Axis formula generation/admission tests, not target Excel recalculation.

The shared target-engine fixture chain contains these operators. A READY plan,
its frozen native references and an XLSX cache are never treated as Excel PASS.
Native Q0 does not consume values; Excel snapshots still obey the existing
whole-input representability policy, including values with no requested q.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, excel, operators
from tests.excel_cases import axis_path_cases

AXIS_CASES = list(axis_path_cases())
NAMES = ("cumulative_product_axis", "cumulative_max_axis", "quantiles_axis")


def arguments(name, value=None, axis=1):
    if value is None:
        value = np.array([[1., .5, 2., 1.25], [2., 1., .5, 1.5]])
    if name == "quantiles_axis":
        return [value, np.array([1., .5, 0., .25, .5]), axis]
    return [value, axis, 3.]


def assert_budget(metadata):
    assert metadata["status"] == "READY"
    assert metadata["verification"] == "NOT_RECALCULATED"
    upper = metadata["upper_bound"]
    assert upper["method"] == "symbolic-shape-upper-bound-v1"
    for field in ("cells", "formula_characters", "snapshot_bytes"):
        assert metadata[field] <= upper[field]
    assert metadata["estimated_work"] <= upper["work_units"]
    assert upper["cells"] <= 2_000_000


def written_cells(plan, path):
    plan.write_cells(str(path))
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("name,make_plan", AXIS_CASES,
                         ids=lambda item: item if isinstance(item, str) else None)
def test_axis_target_fixtures_have_complete_transparent_symbolic_bounds(name, make_plan, tmp_path):
    plan = make_plan()
    metadata = plan.metadata()
    assert_budget(metadata)
    cells = written_cells(plan, tmp_path / "steps.jsonl")
    formulas = [cell for cell in cells if cell[3] == "formula"]
    assert len(cells) == metadata["cells"], name
    assert len(formulas) <= metadata["upper_bound"]["formula_cells"], name
    assert sum(len(cell[4]) for cell in formulas) == metadata["formula_characters"]
    steps = [cell[4] for cell in formulas if cell[0].startswith("Steps")]
    assert steps and all("Results" not in formula for formula in steps)
    assert all("PERCENTILE(" not in formula and "SORT(" not in formula and
               "SUMPRODUCT(" not in formula and "CUMPROD(" not in formula
               for formula in steps)
    if "strict_max" in name:
        assert any(">" in formula and "IF(" in formula for formula in steps)
        assert all("MAX(" not in formula for formula in steps)
    references = plan.reference()
    assert len(references) == len(metadata["outputs"])
    assert all(not reference["error"] for reference in references)
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"


@pytest.mark.parametrize("name", NAMES)
def test_axis_recipe_budget_accepts_symbolic_upper_bound_and_rejects_one_less(name):
    args = arguments(name)
    initial = excel.plan_operator(name, args).metadata()
    upper = initial["upper_bound"]["cells"]
    accepted = excel.plan_operator(name, args, max_cells=upper).metadata()
    assert_budget(accepted)
    assert accepted["cells"] == initial["cells"]
    assert accepted["upper_bound"]["cells"] == upper
    with pytest.raises(excel.ExcelExportError, match="OVER_BUDGET.*symbolic.*bound"):
        excel.plan_operator(name, args, max_cells=upper - 1)
    # Ordinary native calculation is independent of all Excel budgets.
    native = getattr(operators, name)(*args)
    reference = excel.plan_operator(name, args).reference()[0]
    assert list(np.shape(native)) == reference["shape"]
    np.testing.assert_array_equal(np.asarray(native).ravel(), reference["values"])


@pytest.mark.parametrize("name", NAMES)
def test_axis_large_request_is_rejected_before_formula_expansion(name):
    args = ([np.ones((1200, 2)), np.array([.25, .5, .75]), 0]
            if name == "quantiles_axis" else [np.ones(400_000), 0, 1.])
    # q ranking is quadratic per line; recurrences include complete inputs,
    # updates, independent outputs and their verification columns.
    with pytest.raises(excel.ExcelExportError, match="OVER_BUDGET.*symbolic.*bound"):
        excel.plan_operator(name, args)


def layout_view(values, layout):
    if layout == "f":
        return np.asfortranarray(values)
    if layout == "negative":
        slices = (slice(None, None, -1),) * values.ndim
        return values[slices].copy()[slices]
    if layout == "padded":
        storage = np.empty(tuple(2 * size for size in values.shape))
        view = storage[(slice(None, None, 2),) * values.ndim]
        view[...] = values
        return view
    return np.array(values, order="C", copy=True)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("layout", ["c", "f", "negative", "padded"])
def test_axis_excel_snapshot_preserves_logical_coordinates_for_all_tensor_axes(name, axis, layout):
    base = (np.arange(1., 25.).reshape(2, 3, 4) % 7 + 1) / 8
    values = layout_view(base, layout)
    before = values.copy()
    plan = excel.plan_operator(name, arguments(name, values, axis))
    assert_budget(plan.metadata())
    reference = plan.reference()[0]
    expected = getattr(operators, name)(*arguments(name, base, axis))
    assert list(expected.shape) == reference["shape"]
    np.testing.assert_array_equal(reference["values"], expected.ravel())
    np.testing.assert_array_equal(values, before)
    values[...] = .25
    np.testing.assert_array_equal(plan.reference()[0]["values"], expected.ravel())
    changed = excel.plan_operator(name, arguments(name, values, axis))
    assert changed.identity != plan.identity
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"


def test_axis_prefix_initial_is_reset_for_every_line_and_nan_max_is_strict():
    product = excel.plan_operator("cumulative_product_axis", [
        np.array([[2., 3.], [4., .5]]), 1, 5.]).reference()[0]
    np.testing.assert_array_equal(product["values"], [10., 30., 20., 10.])
    maximum = excel.plan_operator("cumulative_max_axis", [
        np.array([[2., np.nan, 1., 2.], [-1., np.nan, 0., -2.]]), 1, 1.]).reference()[0]
    np.testing.assert_array_equal(maximum["values"], [2., 2., 2., 2., 1., 1., 1., 1.])
    missing_seed = excel.plan_operator("cumulative_max_axis", [
        np.array([[1., 2.], [3., np.nan]]), 0, np.nan])
    assert_budget(missing_seed.metadata())
    assert np.isnan(missing_seed.reference()[0]["values"]).all()


@pytest.mark.parametrize("name", ["cumulative_product_axis", "cumulative_max_axis"])
def test_axis_rank_one_prefix_retains_missing_value_policy(name):
    plan = excel.plan_operator(name, [np.array([2., .5, np.nan, 3.]), 0, 4.])
    assert_budget(plan.metadata())
    expected = [8., 4., np.nan, np.nan] if name == "cumulative_product_axis" else [4., 4., 4., 4.]
    np.testing.assert_array_equal(plan.reference()[0]["values"], expected)


def test_axis_quantiles_do_not_filter_nan_and_keep_duplicate_probability_order():
    plan = excel.plan_operator("quantiles_axis", [
        np.array([[3., 1., 2.], [np.nan, 4., 2.], [np.nan, np.nan, np.nan]]),
        np.array([1., .5, 0., .25, .5]), 1])
    assert_budget(plan.metadata())
    expected = np.array([[3., 2., 1., 1.5, 2.],
                         [np.nan, 4., 2., 3., 4.],
                         [np.nan, np.nan, np.nan, np.nan, np.nan]])
    reference = plan.reference()[0]
    assert reference["shape"] == [3, 5]
    np.testing.assert_array_equal(reference["values"], expected.ravel())
    singleton = excel.plan_operator("quantiles_axis", [
        np.array([[np.nan], [7.]]), np.array([0., .13, 1.]), 1]).reference()[0]
    np.testing.assert_array_equal(singleton["values"], [np.nan, np.nan, np.nan, 7., 7., 7.])


def test_axis_quantiles_share_rank_work_across_probabilities_without_freezing_content():
    values = np.arange(96., dtype=np.float64).reshape(32, 3)
    one = excel.plan_operator("quantiles_axis", [values, np.array([.5]), 0]).metadata()
    many = excel.plan_operator("quantiles_axis", [values, np.array([.1, .5, .9]), 0]).metadata()
    reverse = excel.plan_operator("quantiles_axis", [values[::-1], np.array([.1, .5, .9]), 0]).metadata()
    # Additional probabilities add selection/interpolation/output cells, not
    # another full quadratic rank network for each probability.
    assert 0 < many["upper_bound"]["formula_cells"] - one["upper_bound"]["formula_cells"] < 32 * 32 * 3
    assert reverse["upper_bound"] == many["upper_bound"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape,axis", [((0,), 0), ((0, 3), 0), ((2, 0), 1), ((0, 3, 2), 1)])
def test_axis_empty_outputs_have_a_finite_symbolic_plan(name, shape, axis):
    values = np.empty(shape)
    args = ([values, np.empty(0), axis] if name == "quantiles_axis" else [values, axis, 2.])
    plan = excel.plan_operator(name, args)
    assert_budget(plan.metadata())
    reference = plan.reference()[0]
    expected_shape = list(shape)
    if name == "quantiles_axis":
        expected_shape[axis] = 0
    assert reference["shape"] == expected_shape
    assert not reference["values"] and not reference["error"]


@pytest.mark.parametrize("probability", [-.1, 1.1, np.nan])
def test_axis_empty_outer_does_not_skip_invalid_probability_configuration(probability):
    args = [np.empty((0, 3)), np.array([probability]), 1]
    with pytest.raises((ValueError, RuntimeError), match="INVALID_PARAMETER"):
        operators.quantiles_axis(*args)
    plan = excel.plan_operator("quantiles_axis", args)
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"
    assert plan.reference()[0]["error"] == "INVALID_PARAMETER"


def test_axis_empty_outer_accepts_valid_nonempty_probabilities():
    plan = excel.plan_operator("quantiles_axis", [np.empty((0, 3)), np.array([0., .5, 1.]), 1])
    assert_budget(plan.metadata())
    reference = plan.reference()[0]
    assert reference["shape"] == [0, 3]
    assert not reference["values"] and not reference["error"]


def test_axis_zero_probabilities_skip_native_payload_but_keep_excel_snapshot_domain_policy():
    values = np.array([np.inf])
    probabilities = np.empty(0)
    assert operators.quantiles_axis(values, probabilities, 0).shape == (0,)
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
        excel.plan_operator("quantiles_axis", [values, probabilities, 0])


def test_axis_nonempty_quantile_output_requires_samples():
    plan = excel.plan_operator("quantiles_axis", [np.empty((2, 0)), np.array([.5]), 1])
    assert_budget(plan.metadata())
    assert plan.reference()[0]["error"] == "INSUFFICIENT_SAMPLE"
    graph = GraphCompiler(graph_types()).compile("quantiles_axis(x,q,1)", result_format="typed")
    with pytest.raises((ValueError, RuntimeError), match="INSUFFICIENT_SAMPLE"):
        excel.plan(graph, {"x": np.empty((2, 0)), "q": np.array([.5])})


def test_axis_isolated_empty_quantile_failure_does_not_corrupt_next_observed_axis_operation():
    # The first root creates x/axis/initial parents before q fails. The final
    # max then immediately consumes those existing parents, with no intervening
    # constant/input observer callback to reset an argument collector.
    graph = GraphCompiler(graph_types()).compile({
        "seed_parents": "cumulative_product_axis(x,1,2)",
        "missing_distribution": "quantiles_axis(x,q,1)",
        "healthy_next": "cumulative_max_axis(x,1,2)",
    }, result_format="typed", error_policy="isolate")
    plan = excel.plan(graph, {"x": np.empty((2, 0)), "q": np.array([.5])})
    assert_budget(plan.metadata())
    references = plan.reference()
    for index in (0, 2):
        assert not references[index]["error"]
        assert references[index]["shape"] == [2, 0]
        assert not references[index]["values"]
    failed = references[1]
    assert failed["error"] or (failed["statuses"] and all(status != 0 for status in failed["statuses"]))
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"


def test_axis_isolated_failure_does_not_corrupt_next_same_quantile_operation():
    types = graph_types()
    types["y"] = dict(kind="matrix", axes=["time", "scenario"], shape=["T", "D"])
    graph = GraphCompiler(types).compile({
        # y, axis0 and literal1 are all established first; q is established
        # before the failing operation. The final quantile creates no parent.
        "seed_parents": "cumulative_product_axis(y,0,1)",
        "missing_distribution": "quantiles_axis(x,q,1)",
        "healthy_next": "quantiles_axis(y,q,0)",
    }, result_format="typed", error_policy="isolate")
    plan = excel.plan(graph, {"x": np.empty((2, 0)),
                              "y": np.array([[1., 2., 3.], [3., 4., 5.]]),
                              "q": np.array([.5])})
    assert_budget(plan.metadata())
    references = plan.reference()
    np.testing.assert_array_equal(references[0]["values"], [1., 2., 3., 3., 8., 15.])
    failed = references[1]
    assert failed["error"] or (failed["statuses"] and all(status != 0 for status in failed["statuses"]))
    assert references[2]["shape"] == [1, 3]
    assert not references[2]["error"]
    np.testing.assert_array_equal(references[2]["values"], [2., 3., 4.])
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("axis", [-1, .5, 3])
def test_axis_excel_rejects_invalid_configuration_without_expanding(name, axis):
    with pytest.raises((ValueError, RuntimeError), match="INVALID_PARAMETER|UNKNOWN_BOUND"):
        excel.plan_operator(name, arguments(name, axis=axis))


def graph_types():
    return {"x": dict(kind="matrix", axes=["time", "scenario"], shape=["T", "P"]),
            "q": dict(kind="vector", axes=["component"], shape=["Q"])}


@pytest.mark.parametrize("name,empty_probabilities", [
    ("cumulative_product_axis", False),
    ("cumulative_max_axis", False),
    ("quantiles_axis", False),
    ("quantiles_axis", True),
])
def test_axis_healthy_all_empty_typed_graph_has_reference_and_visible_root_state(
        name, empty_probabilities, tmp_path):
    # All roots contain zero bytes. Before the reference-storage fix, passing
    # vector.data() to the graph executor failed with GRAPH_NULL_BATCH.
    types = {"x": graph_types()["x"]}
    values = np.ones((2, 3)) if empty_probabilities else np.empty((0, 3))
    inputs = {"x": values}
    expected_shape = list(values.shape)
    if name == "quantiles_axis":
        types["q"] = graph_types()["q"]
        inputs["q"] = np.empty(0) if empty_probabilities else np.array([.5])
        expected_shape[1] = len(inputs["q"])
        expression = "quantiles_axis(x,q,1)"
    else:
        expression = f"{name}(x,1,2)"
    graph = GraphCompiler(types).compile({"empty": expression}, result_format="typed")
    plan = excel.plan(graph, inputs)
    metadata = plan.metadata()
    assert_budget(metadata)
    assert metadata["outputs"][0]["shape"] == expected_shape
    reference = plan.reference()[0]
    assert reference["shape"] == expected_shape
    assert not reference["error"] and not reference["values"] and not reference["statuses"]
    cells = written_cells(plan, tmp_path / "healthy-empty.jsonl")
    state = [cell[4] for cell in cells if cell[:4] == ["Outputs", 0, 4, "formula"]]
    assert len(state) == 1 and "EMPTY" in state[0] and "LAYOUT_STALE" in state[0]
    assert not any(cell[0].startswith("Results") for cell in cells)
    assert len(cells) == metadata["cells"]


@pytest.mark.parametrize("probability", [-.1, 1.1, np.nan])
def test_axis_isolated_invalid_probability_empty_root_keeps_shape_failure_and_budget(
        probability, tmp_path):
    types = {"x": graph_types()["x"],
             "bad_q": graph_types()["q"], "good_q": graph_types()["q"]}
    graph = GraphCompiler(types).compile({
        "failed_empty": "quantiles_axis(x,bad_q,1)",
        "healthy_quantile": "quantiles_axis(x,good_q,1)",
        "healthy_product": "cumulative_product_axis(x,1,2)",
        "healthy_max": "cumulative_max_axis(x,1,2)",
    }, result_format="typed", error_policy="isolate")
    inputs = {"x": np.empty((0, 3)), "bad_q": np.array([probability]),
              "good_q": np.array([.5])}
    plan = excel.plan(graph, inputs)
    metadata = plan.metadata()
    assert_budget(metadata)
    shapes = [[0, 1], [0, 1], [0, 3], [0, 3]]
    assert [output["shape"] for output in metadata["outputs"]] == shapes
    references = plan.reference()
    assert [reference["shape"] for reference in references] == shapes
    assert references[0]["error"] == "STATUS:4"
    assert all(not reference["error"] for reference in references[1:])
    assert all(not reference["values"] and not reference["statuses"]
               for reference in references)
    cells = written_cells(plan, tmp_path / "failed-empty.jsonl")
    states = [cell for cell in cells if cell[0] == "Outputs" and
              cell[2:4] == [4, "formula"]]
    assert [cell[1] for cell in states] == list(range(4))
    assert "ERROR:STATUS:4" in states[0][4]
    assert all("EMPTY" in cell[4] and "LAYOUT_STALE" in cell[4] for cell in states)
    assert not any(cell[0].startswith("Results") for cell in cells)
    assert len(cells) == metadata["cells"]
    assert sum(cell[3] == "formula" for cell in cells) <= metadata["upper_bound"]["formula_cells"]
    # Root-state cells are admitted independently of numerical execution and
    # remain charged even when every result has zero elements.
    upper = metadata["upper_bound"]["cells"]
    accepted = excel.plan(graph, inputs, max_cells=upper).metadata()
    assert accepted["upper_bound"]["cells"] == upper
    assert accepted["cells"] == metadata["cells"]
    with pytest.raises(excel.ExcelExportError, match="OVER_BUDGET.*symbolic.*bound"):
        excel.plan(graph, inputs, max_cells=upper - 1)
    healthy_inputs = {**inputs, "bad_q": np.array([.5])}
    healthy = excel.plan(graph, healthy_inputs)
    assert healthy.metadata()["upper_bound"] == metadata["upper_bound"]
    assert all(not reference["error"] for reference in healthy.reference())
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"


def test_axis_shared_graph_roots_keep_shapes_and_cse_budget(tmp_path):
    source = "cumulative_product_axis(x,0,2)"
    types = graph_types()
    graph = GraphCompiler(types).compile({
        "paths": source, "paths_again": source,
        "peak": f"cumulative_max_axis({source},0,2)",
        "fan": f"swap_axes(quantiles_axis({source},q,1),0,1)",
    }, result_format="typed")
    inputs = {"x": np.array([[1., .5, 2.], [2., 1., .5]]),
              "q": np.array([0., .5, 1.])}
    plan = excel.plan(graph, inputs)
    metadata = plan.metadata()
    assert_budget(metadata)
    assert [output["name"] for output in metadata["outputs"]] == ["paths", "paths_again", "peak", "fan"]
    kinds = [node["kind"] for node in metadata["upper_bound"]["largest_nodes"]]
    assert kinds.count("cumulative_product_axis") == 1
    assert [output["shape"] for output in metadata["outputs"]] == [[2, 3], [2, 3], [2, 3], [3, 2]]
    reference = plan.reference()
    np.testing.assert_array_equal(reference[0]["values"], reference[1]["values"])
    cells = written_cells(plan, tmp_path / "graph.jsonl")
    assert len(cells) == metadata["cells"]
    assert all("Results" not in cell[4] for cell in cells
               if cell[3] == "formula" and cell[0].startswith("Steps"))


@pytest.mark.parametrize("entry", ["operator", "graph"])
@pytest.mark.parametrize("case", ["overflow", "underflow"])
def test_axis_product_excel_domain_sees_every_prefix_while_native_keeps_ieee(entry, case):
    values = (np.array([1e200, 1e200, np.nan]) if case == "overflow" else
              np.array([1e-200, 1e-120, np.nan]))
    args = [values, 0, 1.]
    before = values.copy()
    if entry == "operator":
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
            excel.plan_operator("cumulative_product_axis", args)
        result = operators.cumulative_product_axis(*args)
    else:
        graph = GraphCompiler({"x": dict(kind="vector", axes=["scenario"], shape=["N"])}).compile(
            "finite_mask(cumulative_product_axis(x,0,1))", result_format="typed")
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
            excel.plan(graph, {"x": values})
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            result = scheduler.execute(graph, {"x": values}, np.array([0], np.int64), np.array([0], np.int64))
        assert result.audit["python_fallback"] == result.audit["python_operator_calls"] == 0
        assert result.audit["request_time_compilation"] == 0
        np.testing.assert_array_equal(result.outputs[0].values[0], [True, case == "underflow", False])
        result = operators.cumulative_product_axis(*args)
    assert np.isnan(result[-1])
    if case == "overflow":
        assert np.isposinf(result[1])
    else:
        assert 0 < result[1] < np.finfo(np.float64).tiny
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("entry", ["operator", "graph"])
@pytest.mark.parametrize("other", [1., np.nan])
def test_axis_quantile_export_checks_small_weighted_term_hidden_by_sum_or_nan(entry, other):
    tiny = np.finfo(np.float64).tiny
    values = np.array([tiny, other])
    probability = np.array([.5])
    # lo/2 is subnormal, but lo/2+1/2 is ordinary .5. Checking only the result
    # would hide a spreadsheet-unrepresentable real mathematical step.
    if entry == "operator":
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE.*quantiles_axis intermediate"):
            excel.plan_operator("quantiles_axis", [values, probability, 0])
        np.testing.assert_array_equal(operators.quantiles_axis(values, probability, 0),
                                      [np.nan if np.isnan(other) else .5])
    else:
        graph = GraphCompiler({
            "x": dict(kind="vector", axes=["scenario"], shape=["N"]),
            "q": dict(kind="vector", axes=["component"], shape=["Q"]),
        }).compile("quantiles_axis(x,q,0)>0", result_format="typed")
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE.*quantiles_axis intermediate"):
            excel.plan(graph, {"x": values, "q": probability})
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            result = scheduler.execute(graph, {"x": values, "q": probability},
                                       np.array([0], np.int64), np.array([0], np.int64))
        np.testing.assert_array_equal(result.outputs[0].values[0], [not np.isnan(other)])


def test_axis_quantile_integer_rank_never_requires_weighted_intermediate_domain():
    tiny = np.finfo(np.float64).tiny
    plan = excel.plan_operator("quantiles_axis", [np.array([tiny, 1., 2.]), np.array([0., .5, 1.]), 0])
    assert_budget(plan.metadata())
    np.testing.assert_array_equal(plan.reference()[0]["values"], [tiny, 1., 2.])


@pytest.mark.parametrize("name", NAMES)
def test_axis_inf_is_an_export_domain_rejection_and_does_not_limit_native(name):
    args = arguments(name, np.array([[np.inf]]))
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
        excel.plan_operator(name, args)
    assert np.isposinf(getattr(operators, name)(*args)).all()


@pytest.mark.parametrize("name", NAMES)
def test_axis_written_workbook_keeps_not_recalculated_formula_caches(name, tmp_path):
    pytest.importorskip("xlsxwriter")
    openpyxl = pytest.importorskip("openpyxl")
    plan = excel.plan_operator(name, arguments(name))
    output = tmp_path / (name + ".xlsx")
    report = excel.export(plan, output)
    assert report["verification"] == "NOT_RECALCULATED"
    formulas = openpyxl.load_workbook(output, data_only=False)
    caches = openpyxl.load_workbook(output, data_only=True)
    count = 0
    try:
        for sheet in formulas:
            for row in sheet:
                for cell in row:
                    if cell.data_type == "f":
                        count += 1
                        assert caches[sheet.title][cell.coordinate].value == "NOT_RECALCULATED"
        assert count > 0
    finally:
        formulas.close()
        caches.close()
    assert not list(tmp_path.glob(".calmetrics-excel-*"))
