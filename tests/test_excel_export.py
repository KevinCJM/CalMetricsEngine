"""Native formula generation, ownership, budget and optional real-Excel parity."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from calmetrics_engine import GraphCompiler, excel, operators
from tests.excel_cases import cases as conformance_cases


def test_registry_has_an_explicit_formula_recipe_for_every_operator():
    assert {name for name, _ in excel.coverage()} == {s["id"] for s in operators.catalog()}
    assert {state for _, state in excel.coverage()} == {"implemented_unverified"}


@pytest.mark.parametrize("operator", ["min_where", "max_where"])
def test_masked_extrema_do_not_emit_excel_unparseable_double_max(operator, tmp_path):
    Tokenizer = pytest.importorskip("openpyxl.formula").Tokenizer

    plan = excel.plan_operator(operator, [np.array([3., 1., 2.]), np.array([True, False, True])])
    stream = tmp_path / "cells.jsonl"
    plan.write_cells(str(stream))
    formulas = [c[4] for c in map(json.loads, stream.read_text().splitlines()) if c[3] == "formula"]
    for formula in formulas:
        for token in Tokenizer("=" + formula).items:
            if token.type == "OPERAND" and token.subtype == "NUMBER":
                assert abs(float(token.value)) <= 9.99999999999999e307
    assert any("2^(1023)" in formula for formula in formulas)


def test_graph_cse_snapshot_and_typed_outputs(tmp_path):
    x = np.arange(1., 8.)
    graph = GraphCompiler({"x": "series"}).compile(
        {"avg": "mean(x)", "twice": "mean(x)*2", "path": "cumulative_sum(x)", "mask": "x>3"},
        result_format="typed",
    )
    plan = excel.plan(graph, {"x": x})
    x[:] = -99
    reference = plan.reference()
    assert reference[0]["values"] == [4.]
    assert reference[1]["values"] == [8.]
    assert reference[3]["values"] == [0, 0, 0, 1, 1, 1, 1]
    path = tmp_path / "cells.jsonl"
    plan.write_cells(str(path))
    cells = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(cells) == plan.metadata()["cells"]
    assert int(next(c[4] for c in cells if c[:3] == ["Readme", 9, 1])) == len(cells)
    assert sum(c[0] == "Nodes" and c[3] == "text" and c[4] == "mean" for c in cells) == 1
    assert all("Results" not in c[4] for c in cells if c[0].startswith("Steps") and c[3] == "formula")


def test_budget_and_independent_cpp_execution():
    x = np.arange(1., 8.)
    p = excel.plan_operator("mean", [x])
    exact = p.metadata()["cells"]
    assert excel.plan_operator("mean", [x], max_cells=exact).metadata()["cells"] == exact
    with pytest.raises(excel.ExcelExportError, match="OVER_BUDGET"):
        excel.plan_operator("mean", [x], max_cells=exact - 1)
    assert operators.mean(x) == 4.
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
        excel.plan_operator("argsort", [np.array([2**60], np.int64)])


def test_strided_input_export_and_atomic_writer(tmp_path):
    pytest.importorskip("xlsxwriter")
    openpyxl = pytest.importorskip("openpyxl")
    x = np.arange(1., 7.).reshape(2, 3)[:, ::-1]
    p = excel.plan_operator("transpose", [x])
    destination = tmp_path / "计算步骤"
    destination.mkdir()
    output = destination / "计算结果.xlsx"
    report = excel.export(p, output)
    assert report["verification"] == "NOT_RECALCULATED"
    formulas = openpyxl.load_workbook(output, data_only=False)
    cached = openpyxl.load_workbook(output, data_only=True)
    assert formulas["Results0"]["B1"].data_type == "f"
    assert cached["Results0"]["B1"].value == "NOT_RECALCULATED"
    assert cached["Results0"]["C1"].value == 3
    assert not list(destination.glob(".calmetrics-excel-*"))


def decode(value):
    if isinstance(value, dict):
        if "array" in value:
            return np.asarray([decode(x) for x in value["array"]], dtype=value["dtype"]).reshape(value["shape"])
        if "tuple" in value:
            return tuple(decode(x) for x in value["tuple"])
        if "scalar" in value:
            return np.dtype(value["dtype"]).type(value["scalar"])
    if isinstance(value, str) and value in {"nan", "inf", "-inf"}:
        return float(value)
    return value


FIXTURE = json.loads((Path(__file__).parent / "data/canonical_reference.json").read_text())
SUCCESS = [c for c in FIXTURE["cases"] if "expected" in c]


@pytest.mark.parametrize("case", SUCCESS, ids=lambda c: c["id"])
def test_frozen_operator_cases_generate_real_formulas(case, tmp_path):
    args = [decode(x) for x in case["args"]]
    if any(np.isinf(v).any() for v in args if isinstance(v, np.ndarray | float)):
        with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
            excel.plan_operator(case["operator"], args)
        return
    p = excel.plan_operator(case["operator"], args)
    path = tmp_path / "cells.jsonl"
    p.write_cells(str(path))
    assert p.metadata()["cells"] == len(path.read_text().splitlines())
    assert not p.reference()[0]["error"]


@pytest.mark.parametrize("data, error", [
    ({"x": np.ones(3, np.float32)}, "dtype"),
    ({"x": np.ones((3, 1))}, "rank"),
    ({"x": [1., 2.]}, "ndarray"),
])
def test_graph_export_preserves_declared_types(data, error):
    graph = GraphCompiler({"x": "series"}).compile("mean(x)")
    with pytest.raises((ValueError, TypeError), match=error):
        excel.plan(graph, data)


def test_graph_symbolic_dimensions_and_parameter_snapshot():
    graph = GraphCompiler({"x": "series", "y": "series", "scale": "scalar"}).compile("mean(x+y)*scale")
    with pytest.raises(ValueError, match="dimension"):
        excel.plan(graph, {"x": np.ones(3), "y": np.ones(4)}, {"scale": 2.})
    params = {"scale": 2.}
    plan = excel.plan(graph, {"x": np.ones(3), "y": np.ones(3)}, params)
    params["scale"] = 9.
    assert plan.reference()[0]["values"] == [4.]


def test_early_input_and_work_budgets():
    with pytest.raises(excel.ExcelExportError, match="before allocation"):
        excel.plan_operator("mean", [np.ones(500_001)])
    with pytest.raises(excel.ExcelExportError, match="work bound"):
        excel.plan_operator("ps_filter", [np.ones(100), np.zeros(100, np.int64), 2, 2, .2])
    with pytest.raises(excel.ExcelExportError, match="integer scalar"):
        excel.plan_operator("add", [2**60, 1])
    with pytest.raises(ValueError, match="INVALID_MASK"):
        excel.plan_operator("logical_not", [np.array(2, np.uint8)])


@pytest.mark.parametrize("operator,args", [
    ("multiply", [1e200, 1e200]),
    ("multiply", [-1e200, 1e200]),
    ("multiply", [1e-200, 1e-120]),
    ("multiply", [np.array([1., 1e200]), np.array([2., 1e200])]),
    ("sum", [np.array([1e308, 1e308])]),
    ("dot", [np.array([1e200]), np.array([1e200])]),
])
def test_unrepresentable_arithmetic_rejected_before_ready(operator, args):
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE.*arithmetic"):
        excel.plan_operator(operator, args)


@pytest.mark.parametrize("expression", [
    "x*x", "x*x>0", "mean(x*x)",
    "rolling_apply(count_true(x*x>0),2)",
    "block_apply(count_true(x*x>0),2)",
    "filter_apply(count_true(x*x>0),x>0)",
    "group_apply(count_true(x*x>0),argsort(x))",
    "iterate(iterate_x*iterate_x,mean(x),1e-10,3)",
])
def test_numeric_domain_checks_graph_intermediates_and_scopes(expression):
    graph = GraphCompiler({"x": "series"}).compile(expression, result_format="typed", error_policy="isolate")
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE.*arithmetic"):
        excel.plan(graph, {"x": np.array([1e200, 1e200])})
    # A throwing observer cannot leak into another plan on the same thread.
    safe = excel.plan(GraphCompiler({"x": "series"}).compile("mean(x)"), {"x": np.array([1., 2.])})
    assert safe.reference()[0]["values"] == [1.5]


@pytest.mark.parametrize("expression", [
    "bisect(solve_x,-2.23e-308,2.24e-308,1e-300,2)",
    "iteration_residual(iterate(2.24e-308,2.23e-308,1e-300,2))",
])
def test_numeric_domain_checks_scope_values_without_operator_nodes(expression):
    graph = GraphCompiler({"x": "series"}).compile(expression, result_format="typed")
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE.*arithmetic"):
        excel.plan(graph, {"x": np.array([1., 2.])})


@pytest.mark.parametrize("expression", ["1e-320", "rolling_apply(1e-320,2)"])
def test_subnormal_constants_cannot_be_exported(expression):
    from calmetrics_engine.graph import GraphCompileError

    # libc++ rejects the numeric literal while parsing; other standard libraries
    # may accept it. Neither path may produce an exportable subnormal constant.
    try:
        graph = GraphCompiler({"x": "series"}).compile(expression, result_format="typed")
    except GraphCompileError as error:
        assert "numeric literal" in str(error)
        return
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
        excel.plan(graph, {"x": np.array([1., 2.])})


@pytest.mark.parametrize("value", [1e308, -1e308, 1.797693134862315e308])
def test_formula_numeric_domain_exceeds_direct_literal_ceiling(value, tmp_path):
    # Excel formulas can calculate well above 1e308. Only literal tokens obey the
    # lower 9.99999999999999e307 ceiling; binary reconstruction stays below it.
    plan = excel.plan_operator("mean", [np.array([value])])
    assert plan.reference()[0]["values"] == [value]
    stream = tmp_path / "large.jsonl"
    plan.write_cells(str(stream))
    input_cells = [c for c in map(json.loads, stream.read_text().splitlines())
                   if c[0] == "Inputs0" and c[2] in (1, 2)]
    assert all(c[3] == "formula" and "2^(1023)" in c[4] for c in input_cells)


@pytest.mark.parametrize("value", [np.finfo(float).max, -np.finfo(float).max])
def test_frozen_reference_serialization_cannot_overflow(value):
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
        excel.plan_operator("mean", [np.array([value])])
    graph = GraphCompiler({"x": "series"}).compile(repr(float(value)), result_format="typed")
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
        excel.plan(graph, {"x": np.array([1.])})


def test_numeric_domain_preserves_nan_extrema_sentinels_and_finite_limits():
    for name, expected in [("min_where", float("inf")), ("max_where", -float("inf"))]:
        plan = excel.plan_operator(name, [np.array([np.nan]), np.array([True])])
        assert plan.reference()[0]["values"] == [expected]
    assert excel.plan_operator("multiply", [1e150, 1e150]).reference()[0]["values"] == pytest.approx([1e300])
    assert excel.plan_operator("multiply", [1e-150, 1e-150]).reference()[0]["values"] == pytest.approx([1e-300], abs=0)


def test_export_rejection_does_not_change_ordinary_graph_execution():
    from calmetrics_engine import AdaptiveScheduler

    graph = GraphCompiler({"x": "series"}).compile("count_true(x*x>0)")
    inputs = {"x": np.array([1e200, 1e200])}
    with pytest.raises(excel.ExcelExportError, match="UNREPRESENTABLE_VALUE"):
        excel.plan(graph, inputs)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([2], np.int64))
    assert result.values[0, 0] == 2


def test_cancel_and_writer_failure_preserve_destination(tmp_path, monkeypatch):
    xlsxwriter = pytest.importorskip("xlsxwriter")
    path = tmp_path / "unchanged.xlsx"
    path.write_bytes(b"existing caller artifact")
    plan = excel.plan_operator("mean", [np.ones(3)])
    plan.cancel()
    with pytest.raises(excel.ExcelExportError, match="CANCELLED"):
        excel.export(plan, path)
    plan = excel.plan_operator("mean", [np.ones(3)])
    def failed_write(*args, **kwargs):
        raise OSError("test writer failure")
    monkeypatch.setattr(xlsxwriter.worksheet.Worksheet, "write_formula", failed_write)
    with pytest.raises(OSError, match="test writer failure"):
        excel.export(plan, path)
    assert path.read_bytes() == b"existing caller artifact"
    assert not list(tmp_path.glob(".calmetrics-excel-*"))


def test_normal_compute_does_not_load_optional_excel_writer():
    subprocess.run([sys.executable, "-c", "import sys; import calmetrics_engine as c; "
                    "assert 'calmetrics_engine.excel' not in sys.modules; "
                    "assert 'xlsxwriter' not in sys.modules; "
                    "assert c.operators.add(1.,2.) == 3."], check=True)


def test_export_identity_binds_names_and_resource_configuration():
    compiler = GraphCompiler({"x": "series"})
    first = compiler.compile({"average": "mean(x)"})
    renamed = compiler.compile({"renamed": "mean(x)"})
    inputs = {"x": np.array([1., 2., 3.])}
    plan = excel.plan(first, inputs)
    assert plan.identity == excel.plan(first, inputs).identity
    assert plan.identity != excel.plan(renamed, inputs).identity
    assert plan.identity != excel.plan(first, inputs, max_formula_characters=1_000_000).identity
    assert plan.identity != excel.plan(first, inputs, max_snapshot_bytes=1024).identity
    assert plan.identity != excel.plan(first, inputs, timeout_seconds=60).identity


def test_unmet_minimum_observations_is_explicitly_rejected():
    graph = GraphCompiler({"x": "series"}).compile("mean(x)", minimum_observations=4)
    with pytest.raises(excel.ExcelExportError, match="minimum observations"):
        excel.plan(graph, {"x": np.array([1., 2., 3.])})


# Symbolic bounds are a safety contract independent of the concrete emitter.
# Include every operator plus scopes, errors, records, matrices and tensors.
@pytest.mark.parametrize("name,make_plan", list(conformance_cases(True)), ids=lambda v: v if isinstance(v, str) else None)
def test_symbolic_bounds_cover_entire_conformance_catalog(name, make_plan):
    metadata = make_plan().metadata()
    upper = metadata["upper_bound"]
    assert upper["method"] == "symbolic-shape-upper-bound-v1", name
    for field in ("cells", "formula_characters", "snapshot_bytes"):
        assert metadata[field] <= upper[field], (name, field)
    assert metadata["estimated_work"] <= upper["work_units"], name
    assert upper["cells"] <= 2_000_000
    assert upper["formula_characters"] <= 128 * 1024 * 1024


@pytest.mark.parametrize("expression", [
    "rolling_apply(mean(x)+std(x,0),3,1)",
    "filter_apply(mean(lag(x,2)),x>2)",
    "mean(block_apply(filter_apply(mean(x),x>1),3))",
    "filter_apply(mean(block_apply(sum(x),2)),x>1)",
    "iterate(iterate_x*0.5+mean(x),8,1e-6,5)",
    "bisect(solve_x*solve_x-mean(x),0,10,1e-6,6)",
])
@pytest.mark.parametrize("length", [1, 3, 8, 33])
def test_symbolic_nested_scope_shapes(expression, length):
    graph = GraphCompiler({"x": "series"}).compile(expression, result_format="typed", error_policy="isolate")
    metadata = excel.plan(graph, {"x": np.arange(1., length + 1)}).metadata()
    assert metadata["cells"] <= metadata["upper_bound"]["cells"]


def test_symbolic_budget_counts_shared_dag_once_and_keeps_all_outputs():
    compiler = GraphCompiler({"x": "series"})
    data = {"x": np.arange(1., 8.)}
    one = excel.plan(compiler.compile("mean(x)"), data).metadata()
    two = excel.plan(compiler.compile({"a": "mean(x)", "b": "mean(x)"}), data).metadata()
    # One extra definition, four output metadata cells, five result cells.
    assert two["upper_bound"]["cells"] - one["upper_bound"]["cells"] == 10
    assert len(two["outputs"]) == 2
    assert sum(n["kind"] == "mean" for n in two["upper_bound"]["largest_nodes"]) == 1


def test_symbolic_iteration_budget_uses_limit_not_observed_convergence():
    compiler = GraphCompiler({"x": "series"})
    data = {"x": np.array([1., 2., 3.])}
    # This graph converges immediately, but both limits must be fully reserved.
    plans = [excel.plan(compiler.compile(f"iterate(iterate_x+mean(x)*0,0,1,{limit})",
                                          result_format="typed"), data).metadata() for limit in (2, 20)]
    assert plans[1]["upper_bound"]["cells"] > 5 * plans[0]["upper_bound"]["cells"]


def test_symbolic_budget_rejects_before_formula_expansion():
    # The native geometry-only analysis runs before formula layout, not after a
    # million-cell dry run. The error distinguishes this admission stage.
    graph = GraphCompiler({"x": {"kind": "vector", "shape": ["N"]}}).compile("outer(x,x)", result_format="typed")
    with pytest.raises(excel.ExcelExportError, match="symbolic.*bound"):
        excel.plan(graph, {"x": np.ones(2000)})
    with pytest.raises(excel.ExcelExportError, match="symbolic.*bound"):
        excel.plan(GraphCompiler({"x": "series"}).compile("mean(x)"),
                   {"x": np.ones(10)}, max_formula_characters=100)


def test_symbolic_scope_limit_boundary():
    graph = GraphCompiler({"x": "series"}).compile("rolling_apply(mean(x),3,1)", result_format="typed")
    data = {"x": np.arange(1., 8.)}
    upper = excel.plan(graph, data).metadata()["upper_bound"]["cells"]
    assert excel.plan(graph, data, max_cells=upper).metadata()["upper_bound"]["cells"] == upper
    with pytest.raises(excel.ExcelExportError, match="symbolic.*bound"):
        excel.plan(graph, data, max_cells=upper - 1)
