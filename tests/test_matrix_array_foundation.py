"""Independent numerical invariants and native lifecycle tests for opcodes 147..160."""

import gc

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler
from calmetrics_engine import operators as op

from .excel_cases import foundation_cases
from .test_typed_results import config


def matrix(axis="asset", symbol="N", dtype="float64"):
    return {"kind": "matrix", "dtype": dtype, "axes": [axis, axis], "shape": [symbol, symbol]}


def run(graph, inputs, lane="single"):
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        return scheduler.execute(graph, inputs, np.zeros(2, np.int64), np.zeros(2, np.int64))


@pytest.mark.parametrize("n", [0, 1, 2, 5, 17])
@pytest.mark.parametrize("stride", ["c", "f", "reverse"])
def test_decompositions_reconstruct(n, stride):
    rng = np.random.default_rng(850 + n)
    x = rng.normal(size=(n, n))
    a = x @ x.T + np.eye(n)
    if stride == "f":
        a = np.asfortranarray(a)
    if stride == "reverse":
        a = a[::-1, ::-1]
    a.flags.writeable = False
    lower = op.cholesky(a)
    np.testing.assert_allclose(lower @ lower.T, a, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(lower, np.tril(lower))
    pair = op.symmetric_eigh(a)
    w, v = op.eigenvalues(pair), op.eigenvectors(pair)
    assert pair.shape == (n, n + 1)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(a), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(v.T @ v, np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose((v * w) @ v.T, a, rtol=1e-10, atol=1e-10)
    if n:
        assert np.shares_memory(pair, w) and np.shares_memory(pair, v)
        assert np.all(v[np.argmax(abs(v), axis=0), np.arange(n)] >= 0)
    del pair
    gc.collect()
    assert w.shape == (n,)
    assert not w.flags.writeable and not v.flags.writeable


@pytest.mark.parametrize("n,k", [(0, 0), (0, 3), (1, 1), (3, 0), (3, 1), (4, 7), (17, 5)])
@pytest.mark.parametrize("lower,unit", [(False, False), (True, False), (False, True), (True, True)])
def test_solvers_residual_and_ignored_triangle(n, k, lower, unit):
    rng = np.random.default_rng(n + k)
    a = rng.normal(size=(n, n)) + np.eye(n) * 10
    b = rng.normal(size=(n, k))
    np.testing.assert_allclose(a @ op.solve_multi(a, b), b, rtol=1e-12, atol=1e-12)
    triangular = np.tril(a) if lower else np.triu(a)
    if unit:
        np.fill_diagonal(triangular, 1)
    stored = triangular.copy()
    stored[np.triu_indices(n, 1) if lower else np.tril_indices(n, -1)] = np.nan
    if unit:
        np.fill_diagonal(stored, np.nan)
    np.testing.assert_allclose(
        triangular @ op.triangular_solve(stored, b, int(lower), int(unit)),
        b,
        rtol=1e-12,
        atol=1e-12,
    )
    if k:
        np.testing.assert_allclose(
            triangular @ op.triangular_solve(stored, b[:, 0], int(lower), int(unit)),
            b[:, 0],
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
@pytest.mark.parametrize("shape", [(0,), (5,), (3, 4), (2, 3, 4), (2, 0, 3)])
@pytest.mark.parametrize("reverse", [False, True])
def test_array_operations_preserve_dtype_geometry_and_values(dtype, shape, reverse):
    x = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    if dtype == np.int64:
        x += 2**53 + 1
    if reverse:
        x = x[::-1]
    x.flags.writeable = False
    for axis in range(len(shape)):
        indices = (
            np.array([shape[axis] - 1, 0, 0], np.int64) if shape[axis] else np.array([], np.int64)
        )
        np.testing.assert_array_equal(
            op.take_axis(x, indices, axis), np.take(x, indices, axis=axis)
        )
        sl = [slice(None)] * len(shape)
        sl[axis] = slice(0, shape[axis], 2)
        y = op.slice_axis(x, axis, 0, shape[axis], 2)
        np.testing.assert_array_equal(y, x[tuple(sl)])
        assert not y.flags.writeable
        if y.size:
            assert np.shares_memory(y, x)
        np.testing.assert_array_equal(op.concatenate(x, x, axis), np.concatenate([x, x], axis=axis))
        for other in range(len(shape)):
            np.testing.assert_array_equal(op.swap_axes(x, axis, other), np.swapaxes(x, axis, other))
        if dtype == np.float64:
            np.testing.assert_allclose(op.sum_axis(x, axis), x.sum(axis=axis))
            if shape[axis] or np.prod(shape[:axis] + shape[axis + 1 :]) == 0:
                np.testing.assert_allclose(op.mean_axis(x, axis), x.mean(axis=axis))
            else:
                with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
                    op.mean_axis(x, axis)
    if len(shape) < 3:
        for axis in range(len(shape) + 1):
            np.testing.assert_array_equal(op.stack(x, x, axis), np.stack([x, x], axis=axis))


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
def test_broadcast_is_explicit_readonly_shared_view(dtype):
    x = np.array([1, 2, 3], dtype=dtype)
    target = np.empty((2, 4, 3), dtype=dtype)
    if dtype == np.bool_:
        target.fill(False)
    view = op.broadcast_like(x, target)
    np.testing.assert_array_equal(view, np.broadcast_to(x, target.shape))
    assert np.shares_memory(view, x) and not view.flags.writeable
    assert view.strides[:2] == (0, 0)
    del x
    gc.collect()
    assert view.shape == (2, 4, 3)


@pytest.mark.parametrize(
    "name,args,code",
    [
        ("cholesky", [np.ones((2, 3))], "SHAPE"),
        ("cholesky", [np.array([[1.0, 2.0], [1.0, 3.0]])], "DOMAIN"),
        ("cholesky", [np.zeros((2, 2))], "DOMAIN"),
        ("cholesky", [np.array([[np.nan]])], "NON_FINITE"),
        ("solve_multi", [np.ones((2, 2)), np.ones((2, 3))], "SINGULAR"),
        ("solve_multi", [np.eye(2), np.ones(2)], "RANK"),
        ("solve_multi", [np.eye(2), np.ones((3, 2))], "SHAPE"),
        ("triangular_solve", [np.zeros((2, 2)), np.ones(2)], "SINGULAR"),
        ("symmetric_eigh", [np.eye(2), 0], "INVALID_PARAMETER"),
        ("symmetric_eigh", [np.eye(2), 1e-12, 0], "INVALID_PARAMETER"),
        ("eigenvectors", [np.ones((2, 2))], "SHAPE"),
        ("take_axis", [np.ones((2, 3)), np.array([-1], np.int64), 0], "INDEX"),
        ("take_axis", [np.ones((2, 3)), np.array([2], np.int64), 0], "INDEX"),
        ("take_axis", [np.ones((2, 3)), np.array([1.0]), 0], "DTYPE"),
        ("slice_axis", [np.ones((2, 3)), 0, 0, 3, 1], "INDEX"),
        ("slice_axis", [np.ones((2, 3)), 0, 0, 2, 0], "INVALID_PARAMETER"),
        ("swap_axes", [np.ones((2, 3)), 0, 2], "INVALID_PARAMETER"),
        ("concatenate", [np.ones((2, 3)), np.ones((2, 4)), 0], "SHAPE"),
        ("broadcast_like", [np.ones(2), np.ones((2, 3))], "SHAPE"),
    ],
)
def test_invalid_inputs_fail_before_writes(name, args, code):
    with pytest.raises((ValueError, TypeError), match=code):
        getattr(op, name)(*args)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_graph_shared_decomposition_views_and_transport(lane):
    a = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]])
    g = GraphCompiler({"a": matrix()}).compile(
        {
            "factor": "cholesky(a)",
            "values": "eigenvalues(symmetric_eigh(a))",
            "vectors": "eigenvectors(symmetric_eigh(a))",
            "slice": "slice_axis(a,1,0,3,2)",
            "axis": "sum_axis(a,1)",
            "stack": "stack(a,a,0)",
        },
        error_policy="isolate",
    )
    assert g.metadata()["cse_eliminated_nodes"] > 0
    r = run(g, {"a": a}, lane)
    expected = [
        op.cholesky(a),
        op.eigenvalues(op.symmetric_eigh(a)),
        op.eigenvectors(op.symmetric_eigh(a)),
        a[:, ::2],
        a.sum(1),
        np.stack([a, a]),
    ]
    for output, value in zip(r.outputs, expected, strict=True):
        for actual in output.values:
            np.testing.assert_allclose(actual, value, rtol=1e-12, atol=1e-12)
            assert not np.shares_memory(actual, a) and not actual.flags.writeable
        np.testing.assert_array_equal(output.root_statuses, [0, 0])
    a[:] = 99
    np.testing.assert_allclose(r.outputs[0].values[0], expected[0])


@pytest.mark.parametrize("dtype", ["float64", "bool", "int64"])
@pytest.mark.parametrize("lane", ["single", "process_inline", "process_shared"])
def test_graph_array_dtype_and_nested_views(dtype, lane):
    g = GraphCompiler(
        {"a": matrix(dtype=dtype), "ix": {"kind": "vector", "dtype": "int64", "shape": ["K"]}}
    ).compile(
        [
            "take_axis(swap_axes(a,0,1),ix,1)",
            "concatenate(slice_axis(a,0,0,2,1),slice_axis(a,0,1,3,1),0)",
        ],
        error_policy="isolate",
    )
    a = np.arange(9).reshape(3, 3).astype(dtype)
    if dtype == "int64":
        a += 2**53 + 1
    r = run(g, {"a": a, "ix": np.array([2, 0], np.int64)}, lane)
    for output, value in zip(
        r.outputs, [a.T[:, [2, 0]], np.concatenate([a[:2], a[1:]])], strict=True
    ):
        for actual in output.values:
            assert actual.dtype == a.dtype
            np.testing.assert_array_equal(actual, value)


def test_compile_rejects_dynamic_layout_and_fake_decomposition():
    compiler = GraphCompiler({"a": matrix(), "axis": "scalar"})
    for expression, code in [
        ("sum_axis(a,axis)", "CONSTANT_LAYOUT"),
        ("eigenvectors(a)", "TYPE_MISMATCH"),
        ("slice_axis(a,0,0,2,0)", "INVALID_PARAMETER"),
    ]:
        with pytest.raises(GraphCompileError, match=code):
            compiler.compile(expression)


def test_eigen_failure_isolates_and_symmetry_is_not_repaired():
    a = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 5.0]])
    with pytest.raises(ValueError, match="NON_CONVERGENCE"):
        op.symmetric_eigh(a, 1e-15, 1)
    g = GraphCompiler({"a": matrix()}).compile(
        ["eigenvalues(symmetric_eigh(a,1e-15,1))", "sum_axis(a,0)"], error_policy="isolate"
    )
    r = run(g, {"a": a})
    assert all(r.outputs[0].root_statuses != 0)
    np.testing.assert_array_equal(r.outputs[1].values[0], a.sum(0))


def test_prepared_snapshot_does_not_alias_next_run():
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    g = GraphCompiler({"a": matrix()}).compile(
        ["eigenvalues(symmetric_eigh(a))", "slice_axis(cholesky(a),1,0,2,1)"]
    )
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        p = scheduler.prepare_execution(
            g, {"a": a}, np.array([0], np.int64), np.array([0], np.int64)
        )
        first = p.run_snapshot()
        saved = [v.values[0].copy() for v in first.outputs]
        p.run()
        for output, value in zip(first.outputs, saved, strict=True):
            np.testing.assert_array_equal(output.values[0], value)


@pytest.mark.parametrize(
    "name,args",
    [
        ("slice_axis", [0, 0, 2, 1]),
        ("swap_axes", [0, 0]),
    ],
)
def test_mask_views_reject_invalid_bytes(name, args):
    with pytest.raises(ValueError, match="INVALID_MASK"):
        getattr(op, name)(np.array([0, 2], np.uint8), *args)


@pytest.mark.parametrize(
    "name,make_plan",
    foundation_cases(),
)
def test_excel_recipe_symbolic_bound_and_stream(name, make_plan, tmp_path):
    import json

    plan = make_plan()
    report = plan.metadata()
    assert report["cells"] <= report["upper_bound"]["cells"]
    assert report["formula_characters"] <= report["upper_bound"]["formula_characters"]
    path = tmp_path / "cells.jsonl"
    plan.write_cells(str(path))
    cells = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(cells) == report["cells"]
    assert all(
        "Results" not in cell[4]
        for cell in cells
        if cell[0].startswith("Steps") and cell[3] == "formula"
    )


def test_eigen_excel_budget_uses_maximum_not_actual_sweeps():
    from calmetrics_engine import excel

    plans = [
        excel.plan_operator("symmetric_eigh", [np.eye(3), 1e-12, limit]).metadata()
        for limit in [2, 16]
    ]
    assert plans[1]["upper_bound"]["cells"] > plans[0]["upper_bound"]["cells"]
    assert plans[1]["cells"] > plans[0]["cells"]
    with pytest.raises(excel.ExcelExportError, match="symbolic.*bound"):
        excel.plan_operator("symmetric_eigh", [np.eye(100)])


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_scaled_and_repeated_eigenvalues_keep_residuals(scale):
    a = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]) * scale
    pair = op.symmetric_eigh(a)
    w, v = op.eigenvalues(pair), op.eigenvectors(pair)
    np.testing.assert_allclose(w / scale, [1.0, 1.0, 3.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(((v * w) @ v.T) / scale, a / scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v.T @ v, np.eye(3), atol=1e-12)


def test_near_singular_threshold_and_small_spd_cholesky():
    a = np.diag([1.0, 1e-14])
    np.testing.assert_allclose(op.cholesky(a), np.diag([1.0, 1e-7]), rtol=1e-14)
    for name in ("solve_multi", "triangular_solve"):
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            getattr(op, name)(a, np.eye(2))
        a[1, 1] = np.nextafter(1e-14, np.inf)
        np.testing.assert_allclose(a @ getattr(op, name)(a, np.eye(2)), np.eye(2), rtol=1e-13)
        a[1, 1] = 1e-14


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
def test_zero_strides_and_explicit_output_do_not_alias(dtype):
    x = np.broadcast_to(np.array([0, 1, 1], dtype=dtype), (3, 3))
    y = op.swap_axes(x, 0, 1)
    destination = np.empty((3, 3), dtype=np.uint8 if dtype == np.bool_ else dtype)
    op.swap_axes(x, 0, 1, out=destination)
    np.testing.assert_array_equal(destination, x.T)
    assert np.shares_memory(y, x) and not np.shares_memory(destination, x)


def test_error_does_not_skip_independent_selected_index_validation():
    g = GraphCompiler(
        {"a": matrix(), "ix": {"kind": "vector", "dtype": "int64", "shape": ["K"]}}
    ).compile("take_axis(cholesky(a),ix,0)", error_policy="isolate")
    with pytest.raises(ValueError, match="INDEX_OUT_OF_BOUNDS"):
        run(g, {"a": np.zeros((2, 2)), "ix": np.array([2], np.int64)})


def test_serialized_graph_and_memory_budget():
    import pickle

    g = GraphCompiler({"a": matrix()}).compile(
        ["eigenvalues(symmetric_eigh(a))", "eigenvectors(symmetric_eigh(a))"]
    )
    g = pickle.loads(pickle.dumps(g))
    data = {"a": np.eye(16)}
    bounds = np.array([0], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(g, data, bounds, bounds)
        assert plan.estimated_output_bytes == (16 + 256) * 8
        with pytest.raises(MemoryError):
            scheduler.plan(
                g, data, bounds, bounds, memory_budget_bytes=plan.estimated_total_memory_bytes - 1
            )
    np.testing.assert_array_equal(run(g, data).outputs[0].values[0], np.ones(16))


@pytest.mark.parametrize("operator", ["cholesky", "symmetric_eigh"])
def test_nonfinite_matrix_has_same_error_precedence_as_excel(operator):
    a = np.array([[1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, np.nan]])
    with pytest.raises(ValueError, match="NON_FINITE_RESULT"):
        getattr(op, operator)(a)


@pytest.mark.parametrize("dtype", ["float64", "int64", "bool"])
def test_stack_rank_zero_exact_values(dtype):
    x = np.asarray(2**60 + 1 if dtype == "int64" else 1, dtype=dtype)
    np.testing.assert_array_equal(op.stack(x, x, 0), np.stack([x, x]))
    graph = GraphCompiler({"x": {"kind": "value", "dtype": dtype}}).compile("stack(x,x,0)")
    result = run(graph, {"x": x})
    assert result.outputs[0].values[0].dtype == x.dtype
    np.testing.assert_array_equal(result.outputs[0].values[0], np.stack([x, x]))


def test_empty_selected_axis_does_not_iterate_large_outer_geometry():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import numpy as np
from calmetrics_engine import operators as op
x = np.empty((2**35, 0))
y = op.take_axis(x, np.empty(0, np.int64), 1)
assert y.shape == x.shape and y.size == 0
""",
        ],
        check=True,
        timeout=5,
    )
