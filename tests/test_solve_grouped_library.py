"""Large ordered LU grouping: actual native scalar control and owned graphs."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig
from calmetrics_engine import operators as op


def same_bits(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint64), expected[valid].view(np.uint64))


def sample(n=1025, layout="c", pivot=False):
    rng = np.random.default_rng(621 + n)
    a = rng.normal(size=(n, n))
    if not pivot:
        a[np.diag_indices(n)] += np.abs(a).sum(axis=1) + 1
    b = rng.normal(size=(n, 3))
    if layout == "f":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif layout == "reverse":
        a, b = a[::-1, ::-1], b[::-1]
    elif layout == "sliced":
        aa, bb = np.zeros((2 * n, 2 * n)), np.zeros((2 * n, 6))
        aa[::2, ::2], bb[::2, ::2] = a, b
        a, b = aa[::2, ::2], bb[::2, ::2]
    elif layout == "broadcast":
        b = np.broadcast_to(b[:1, :1], b.shape)
    return a, b


@pytest.mark.parametrize("n", [1023, 1024, 1025])
@pytest.mark.parametrize("layout", ["c", "f", "reverse", "sliced", "broadcast"])
def test_grouped_boundary_signed_strides_readonly_and_workspace(n, layout):
    a, b = sample(n, layout)
    original_a, original_b = a.copy(), b.copy()
    a.flags.writeable = b.flags.writeable = False
    # The canonical scalar path explicitly bypasses the private Eigen helper.
    expected = op.solve_multi(a, b, simd="scalar")
    work = op.Workspace()
    out = np.full(b.size + 8, -913.25)
    destination = out[4:-4].reshape(b.shape)
    actual, audit = op.solve_multi(a, b, out=destination, workspace=work, audit=True)
    same_bits(actual, expected)
    np.testing.assert_array_equal(out[:4], -913.25)
    np.testing.assert_array_equal(out[-4:], -913.25)
    same_bits(a, original_a)
    same_bits(b, original_b)
    assert audit["workspace_bytes"] == n * n * 8
    assert audit["algorithm_copy_bytes"] == (n * n + b.size) * 8
    assert np.shares_memory(actual, destination)
    second = op.solve_multi(a, b, workspace=work)
    same_bits(second, expected)
    assert not np.shares_memory(second, actual)


@pytest.mark.parametrize("layout", ["c", "f", "reverse"])
def test_dense_pivot_permutations_keep_exact_arithmetic(layout):
    a, b = sample(1025, layout, pivot=True)
    same_bits(op.solve_multi(a, b), op.solve_multi(a, b, simd="scalar"))


@pytest.mark.parametrize("fault", ["singular", "threshold", "above", "nan_rhs", "inf_matrix", "overflow"])
def test_delayed_panel_errors_keep_canonical_output_prefix(fault):
    a, b = np.eye(1025), np.ones((1025, 3))
    if fault == "singular":
        a[-1, -1] = 0
    elif fault == "threshold":
        a[-1, -1] = 1e-14
    elif fault == "above":
        a[-1, -1] = np.nextafter(1e-14, np.inf)
    elif fault == "nan_rhs":
        b[-1, 1] = np.nan
    elif fault == "inf_matrix":
        a[-1, -1] = np.inf
    else:
        a[-1, -1], b[-1, 1] = 2e-14, 1e308
    values, errors = [], []
    for simd in ("scalar", "auto"):
        out = np.full(b.shape, -913.25)
        try:
            op.solve_multi(a, b, out=out, simd=simd)
            errors.append(None)
        except ValueError as error:
            errors.append(str(error))
        values.append(out)
    assert errors[0] == errors[1]
    same_bits(values[0], values[1])


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_owned_multiroot_results_and_failed_root_after_scheduler_close(lane):
    a, b = sample(1024)
    bad = np.eye(1024)
    bad[-1, -1] = 0
    declarations = {
        name: dict(kind="matrix", axes=axes, shape=shape)
        for name, axes, shape in (
            ("a", ["asset", "asset"], ["N", "N"]),
            ("bad", ["asset", "asset"], ["N", "N"]),
            ("b", ["asset", "factor"], ["N", "K"]),
        )
    }
    expression = "solve_multi(a,b)"
    graph = GraphCompiler(declarations).compile(
        [expression, expression + ">0", "solve_multi(bad,b)", "b"], error_policy="isolate"
    )
    config = PlannerConfig(
        thread_work_units=1e100 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1, max_processes=2,
    )
    expected = op.solve_multi(a, b, simd="scalar")
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        result = scheduler.execute(graph, dict(a=a, b=b, bad=bad), bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    for row in range(2):
        same_bits(result.outputs[0].values[row], expected)
        np.testing.assert_array_equal(result.outputs[1].values[row], expected > 0)
        assert result.outputs[2].root_statuses[row] != 0
        same_bits(result.outputs[3].values[row], b)
        assert not result.outputs[0].values[row].flags.writeable
        assert not np.shares_memory(result.outputs[0].values[row], b)
    if lane == "process_shared":
        assert result.audit["output_copy_bytes"] == 0


def test_prepared_snapshot_reads_changed_coefficients_and_keeps_previous_result():
    a, b = sample(1024)
    graph = GraphCompiler({
        "a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]),
        "b": dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K"]),
    }).compile("solve_multi(a,b)", result_format="typed")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, dict(a=a, b=b), bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        a[np.diag_indices(1024)] += 1
        b *= 2
        second = prepared.run_audit()
        expected = op.solve_multi(a, b, simd="scalar")
        if sum(chunk["accelerate_lu_calls"] for chunk in second.audit["native_chunks"]):
            np.testing.assert_allclose(second.outputs[0].values[0], expected, rtol=1e-11, atol=4.9e-5)
            np.testing.assert_allclose(a @ second.outputs[0].values[0], b, rtol=1e-11, atol=1e-11)
        else:
            same_bits(second.outputs[0].values[0], expected)
    same_bits(first.outputs[0].values[0], saved)
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
