"""Projection recipe admission/ownership tests; never real Excel parity.

READY, generated formulas and frozen C++ references are separate from target
Excel recalculation. The optional workbook checks require NOT_RECALCULATED
caches, and do not interpret those caches as computed spreadsheet values.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, excel, operators
from tests.excel_cases import projection_primitive_cases

PROJECTION_CASES = list(projection_primitive_cases())


def small_operator_case(name):
    if name == "affine_contract":
        return [np.array([[.25, -.5], [.75, .125]]),
                np.array([[.5, -.25, .125], [.25, .5, -.5]]),
                np.array([.01, -.02, .03])]
    return [np.array([-40., -.125, -1e-20, -0., 0., 1e-20, .125, .13])]


def affine_types():
    return {
        "x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "F"]),
        "k": dict(kind="matrix", axes=["factor", "asset"], shape=["F", "A"]),
        "i": dict(kind="vector", axes=["asset"], shape=["A"]),
    }


def assert_upper_bound(metadata):
    assert metadata["status"] == "READY"
    assert metadata["verification"] == "NOT_RECALCULATED"
    upper = metadata["upper_bound"]
    assert upper["method"] == "symbolic-shape-upper-bound-v1"
    for field in ("cells", "formula_characters", "snapshot_bytes"):
        assert metadata[field] <= upper[field]
    assert metadata["estimated_work"] <= upper["work_units"]
    assert upper["cells"] <= 2_000_000


@pytest.mark.parametrize("name,make_plan", PROJECTION_CASES, ids=lambda value: value if isinstance(value, str) else None)
def test_projection_symbolic_bounds_cover_written_steps_and_all_outputs(name, make_plan, tmp_path):
    plan = make_plan()
    metadata = plan.metadata()
    assert_upper_bound(metadata)
    stream = tmp_path / "cells.jsonl"
    plan.write_cells(str(stream))
    cells = [json.loads(line) for line in stream.read_text().splitlines()]
    formulas = [cell for cell in cells if cell[3] == "formula"]
    assert len(cells) == metadata["cells"], name
    assert len(formulas) <= metadata["upper_bound"]["formula_cells"], name
    assert sum(len(cell[4]) for cell in formulas) == metadata["formula_characters"], name
    assert all("Results" not in cell[4] for cell in formulas if cell[0].startswith("Steps")), name
    assert all("MMULT(" not in cell[4] for cell in formulas if "affine" in name), name
    assert len(plan.reference()) == len(metadata["outputs"]), name
    # Calling the explicit native reference still does not recalculate Excel.
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"


@pytest.mark.parametrize("name", ["affine_contract", "expm1"])
def test_projection_symbolic_cell_budget_accepts_upper_bound_and_rejects_one_less(name):
    arguments = small_operator_case(name)
    initial = excel.plan_operator(name, arguments).metadata()
    upper = initial["upper_bound"]["cells"]
    accepted = excel.plan_operator(name, arguments, max_cells=upper).metadata()
    assert accepted["upper_bound"]["cells"] == upper
    assert accepted["cells"] == initial["cells"]
    with pytest.raises(excel.ExcelExportError, match="OVER_BUDGET.*symbolic.*bound"):
        excel.plan_operator(name, arguments, max_cells=upper - 1)
    # The independent native entry has no Excel budget or admission failure.
    actual = getattr(operators, name)(*arguments)
    expected = excel.plan_operator(name, arguments).reference()[0]
    np.testing.assert_array_equal(np.shape(actual), expected["shape"])
    np.testing.assert_allclose(np.asarray(actual).ravel(), expected["values"], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["affine_contract", "expm1"])
def test_projection_large_recipe_rejects_at_symbolic_stage(name):
    # Small affine ingress expands to one million outputs and three million
    # recipe steps; expm1 needs thirteen helpers per element. Neither request
    # may reach concrete formula expansion/native domain replay.
    arguments = ([np.zeros((1000, 1)), np.zeros((1, 1000)), np.zeros(1000)]
                 if name == "affine_contract" else [np.zeros(200_000)])
    with pytest.raises(excel.ExcelExportError, match="OVER_BUDGET.*symbolic.*bound"):
        excel.plan_operator(name, arguments)


def test_affine_symbolic_budget_grows_with_each_ordered_factor_step():
    plans = [excel.plan_operator("affine_contract", [np.ones((2, factors)), np.full((factors, 3), .01), .2]).metadata()
             for factors in (1, 5)]
    # Six outputs each add a real product and running accumulator per factor.
    assert plans[1]["upper_bound"]["formula_cells"] - plans[0]["upper_bound"]["formula_cells"] >= 2 * 6 * 4
    assert plans[1]["upper_bound"]["work_units"] > plans[0]["upper_bound"]["work_units"]


def test_projection_shared_recipe_is_budgeted_once_but_keeps_every_root():
    affine = "affine_contract(x,k,i)"
    simple = f"expm1({affine})"
    graph = GraphCompiler(affine_types()).compile({"left": simple, "right": simple, "linear": affine}, result_format="typed")
    x, k, i = small_operator_case("affine_contract")
    plan = excel.plan(graph, dict(x=x, k=k, i=i))
    metadata = plan.metadata()
    assert_upper_bound(metadata)
    assert [output["name"] for output in metadata["outputs"]] == ["left", "right", "linear"]
    kinds = [node["kind"] for node in metadata["upper_bound"]["largest_nodes"]]
    assert kinds.count("affine_contract") == kinds.count("expm1") == 1
    references = plan.reference()
    assert len(references) == 3
    np.testing.assert_array_equal(references[0]["values"], references[1]["values"])
    assert all(output["shape"] == [2, 3] for output in metadata["outputs"])


def unsafe_affine_arguments(case):
    if case == "product_overflow_hidden_by_nan":
        return [np.array([[1e200, np.nan]]), np.array([[1e200], [1.]]), np.array([0.])], np.nan
    if case == "product_subnormal_hidden_by_initial":
        return [np.array([[1e-200]]), np.array([[1e-120]]), np.array([1.])], 1.
    if case == "partial_overflow_hidden_by_nan":
        return [np.array([[1., 1., np.nan]]), np.array([[1e308], [1e308], [1.]]), np.array([0.])], np.nan
    tiny = np.finfo(np.float64).tiny
    return [np.ones((1, 2)), np.array([[-1.25 * tiny], [1.]]), np.array([1.5 * tiny])], 1.


@pytest.mark.parametrize("case", ["product_overflow_hidden_by_nan", "product_subnormal_hidden_by_initial",
                                  "partial_overflow_hidden_by_nan", "partial_subnormal_hidden_by_later_term"])
@pytest.mark.parametrize("entry", ["operator", "graph"])
def test_affine_excel_domain_checks_internal_products_and_partial_sums_without_limiting_cpp(case, entry):
    arguments, expected = unsafe_affine_arguments(case)
    originals = [value.copy() for value in arguments]
    if entry == "operator":
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE.*affine_contract intermediate"):
            excel.plan_operator("affine_contract", arguments)
        actual = operators.affine_contract(*arguments)
        assert actual.shape == (1, 1)
        if np.isnan(expected):
            assert np.isnan(actual[0, 0])
        else:
            assert actual[0, 0] == expected
    else:
        # The final bool is representable, so the export observer must check
        # the affine recipe's real dependency/internal arithmetic, not roots.
        graph = GraphCompiler(affine_types()).compile("finite_mask(affine_contract(x,k,i))", result_format="typed")
        inputs = dict(zip(("x", "k", "i"), arguments, strict=True))
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE.*affine_contract intermediate"):
            excel.plan(graph, inputs)
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            result = scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([0], np.int64))
        np.testing.assert_array_equal(result.outputs[0].values[0], [[np.isfinite(expected)]])
        np.testing.assert_array_equal(result.outputs[0].root_statuses, [0])
        assert result.audit["python_fallback"] == result.audit["python_operator_calls"] == 0
        assert result.audit["request_time_compilation"] == 0
    for value, before in zip(arguments, originals, strict=True):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("value,expected", [(710., np.inf), (np.inf, np.inf), (-np.inf, -1.), (1e-320, 1e-320)])
@pytest.mark.parametrize("entry", ["operator", "graph"])
def test_expm1_excel_rejects_unsupported_domain_while_cpp_keeps_ieee_contract(value, expected, entry):
    if entry == "operator":
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
            excel.plan_operator("expm1", [value])
        assert operators.expm1(value) == expected
    else:
        graph = GraphCompiler({"x": dict(kind="vector", axes=["asset"], shape=["N"])}).compile(
            "greater_than(expm1(x),0)", result_format="typed")
        inputs = {"x": np.array([value])}
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
            excel.plan(graph, inputs)
        with AdaptiveScheduler(cpu_budget=1) as scheduler:
            result = scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([0], np.int64))
        np.testing.assert_array_equal(result.outputs[0].values[0], [expected > 0])
        np.testing.assert_array_equal(result.outputs[0].root_statuses, [0])
        assert result.audit["python_fallback"] == result.audit["python_operator_calls"] == 0


def test_expm1_excel_supported_tiny_negative_and_missing_domains_remain_ready():
    values = np.array([-1000., -40., -.125, -1e-20, -0., 0., 1e-20, .125, .13, 700., np.nan])
    plan = excel.plan_operator("expm1", [values])
    assert_upper_bound(plan.metadata())
    reference = plan.reference()[0]
    assert not reference["error"]
    np.testing.assert_array_equal(reference["shape"], [len(values)])
    np.testing.assert_array_equal(reference["values"][:2], [-1., -1.])
    assert reference["values"][3] != 0. and reference["values"][6] != 0.
    assert np.isnan(reference["values"][-1])
    # These are frozen C++ values only; no Excel result is asserted here.
    assert plan.metadata()["verification"] == "NOT_RECALCULATED"


def test_affine_plain_missing_input_is_not_misreported_as_internal_overflow():
    arguments = [np.array([[np.nan, 1.], [2., 3.]]), np.ones((2, 1)), np.array([.1])]
    plan = excel.plan_operator("affine_contract", arguments)
    assert_upper_bound(plan.metadata())
    reference = plan.reference()[0]
    assert not reference["error"] and np.isnan(reference["values"][0])
    assert reference["values"][1] == 5.1


def test_projection_excel_snapshot_keeps_initial_coefficients_and_input_identity():
    arguments = small_operator_case("affine_contract")
    graph = GraphCompiler(affine_types()).compile("expm1(affine_contract(x,k,i))", result_format="typed")
    inputs = dict(zip(("x", "k", "i"), arguments, strict=True))
    plan = excel.plan(graph, inputs)
    original_identity = plan.identity
    original = np.asarray(plan.reference()[0]["values"])
    inputs["x"][:] += .25
    inputs["k"][:] -= .125
    inputs["i"][:] += .04
    np.testing.assert_array_equal(plan.reference()[0]["values"], original)
    assert plan.identity == original_identity
    changed = excel.plan(graph, inputs)
    assert changed.identity != original_identity
    assert not np.array_equal(changed.reference()[0]["values"], original)
    assert plan.metadata()["verification"] == changed.metadata()["verification"] == "NOT_RECALCULATED"


@pytest.mark.parametrize("name", ["affine_contract", "expm1"])
def test_projection_written_workbook_never_caches_cpp_as_recalculated_excel(name, tmp_path):
    pytest.importorskip("xlsxwriter")
    openpyxl = pytest.importorskip("openpyxl")
    plan = excel.plan_operator(name, small_operator_case(name))
    output = tmp_path / (name + ".xlsx")
    report = excel.export(plan, output)
    assert report["verification"] == "NOT_RECALCULATED"
    formula_book = openpyxl.load_workbook(output, data_only=False)
    cache_book = openpyxl.load_workbook(output, data_only=True)
    count = 0
    try:
        for sheet in formula_book:
            for row in sheet:
                for cell in row:
                    if cell.data_type == "f":
                        count += 1
                        assert cache_book[sheet.title][cell.coordinate].value == "NOT_RECALCULATED"
        assert count > 0
    finally:
        formula_book.close()
        cache_book.close()
    assert not list(tmp_path.glob(".calmetrics-excel-*"))
