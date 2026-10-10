"""Paired terminal roots require both error bounds and independent storage."""
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_covariance_accelerate import ACTIVE, MATRIX, compare, counts, inputs
from .test_matmul_library import settings


@pytest.mark.parametrize("layout", ["c", "reverse_f"])
@pytest.mark.parametrize("d", [257, 289, 511])
@pytest.mark.parametrize("pattern", ["scaled", "heterogeneous", "orthogonal"])
def test_group_bound_heads_tails_and_replay_markers(layout, d, pattern):
    x = inputs(layout, n=1025, d=d)
    x.setflags(write=True)
    x *= 4
    if pattern == "heterogeneous":
        x[:, 0] *= 1.5
        x[:, 1:] *= .25
    elif pattern == "orthogonal":
        x[:-1, 0] = np.tile([4., 4., -4., -4.], 256)
        x[:-1, 1] = np.tile([4., -4., 4., -4.], 256)
        x[-1, :2] = 0
        x[:, -1] = -x[:, 0]
    x.setflags(write=False)
    before = x.view(np.uint64).copy()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph("F"), x)
    for output, fn in zip(result.outputs, (op.covariance, op.correlation), strict=True):
        reference = fn(x)
        actual = output.values[0]
        compare(actual, reference)
        np.testing.assert_array_equal(actual.view(np.uint64), actual.T.view(np.uint64))
        assert not actual.flags.writeable and not np.shares_memory(actual, x)
        if pattern == "orthogonal":
            np.testing.assert_array_equal(actual[0, 1].view(np.uint64), reference[0, 1].view(np.uint64))
    if ACTIVE and pattern == "scaled":
        assert counts(result, "accelerate_covariance_calls") == 1
        assert counts(result, "covariance_fallbacks") == 0
        assert counts(result, "covariance_syrk_calls") == 9
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[1].values[0])
    np.testing.assert_array_equal(x.view(np.uint64), before)


def graph(order="C", expressions=None, isolate=False):
    return GraphCompiler({"x": MATRIX, "y": MATRIX}).compile(
        expressions or ["covariance(x)", "correlation(x)"], output_order=order,
        error_policy="isolate" if isolate else "raise",
    )


def execute(scheduler, compiled, x, rows=1):
    return scheduler.execute(compiled, {"x": x, "y": x}, np.zeros(rows, np.int64),
                             np.full(rows, len(x), np.int64))


@pytest.mark.parametrize("layout", ["c", "f", "padded_c", "padded_f", "reverse_c", "reverse_f"])
@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_joint_bounds_transport_layout_and_independent_outputs(layout, order, lane):
    x = inputs(layout, n=129, d=256)
    before = x.view(np.uint64).copy()
    # Match the existing worker packing; local descending F stays ordered.
    transported = x
    if lane.startswith("process") and layout in {"padded_f", "reverse_f"}:
        transported = np.array(x, order="F" if layout == "reverse_f" else "C")
    expected = [op.covariance(transported), op.correlation(transported)]
    compiled = pickle.loads(pickle.dumps(graph(order)))
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = execute(scheduler, compiled, x, 2)
        second = execute(scheduler, compiled, x, 2)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    if lane.startswith("process"):
        assert first.plan.use_shared_memory == (lane == "process_shared")
    assert counts(first, "accelerate_covariance_calls") == (2 if ACTIVE else 0)
    assert counts(first, "algorithm_copy_bytes") == 2 * expected[0].nbytes
    assert counts(first, "numeric_arena_bytes") == counts(first, "root_copy_bytes") == 0
    all_values = []
    for root, reference in enumerate(expected):
        for actual in first.outputs[root].values:
            compare(actual, reference)
            np.testing.assert_allclose(actual, (op.covariance if root == 0 else op.correlation)(x), rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(actual.view(np.uint64), actual.T.view(np.uint64))
            assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
            assert not actual.flags.writeable and not np.shares_memory(actual, x)
            all_values.append(actual)
        assert not np.shares_memory(first.outputs[root].values[0], second.outputs[root].values[0])
    for i, a in enumerate(all_values):
        assert all(not np.shares_memory(a, b) for b in all_values[i+1:])
    np.testing.assert_array_equal(x.view(np.uint64), before)


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("pattern", ["tiny", "wide", "orthogonal", "dense_replay", "scaled", "source_reject", "nan", "reverse"])
def test_both_output_certificates_and_canonical_controls(layout, pattern):
    n = 32769 if pattern == "dense_replay" else 1024
    x = inputs(layout, n=n, d=256).copy(order=layout.upper())
    if pattern in {"tiny", "dense_replay"}:
        if pattern == "dense_replay":
            x[:] = x * .1 + x[:, :1]
        x *= 2.0**-200
    elif pattern == "wide":
        x *= np.exp2(np.linspace(-360, 0, x.shape[1]))
        x[:, ::2] *= -1
    elif pattern == "orthogonal":
        x[:, 0] = np.tile([1., 1., -1., -1.], n//4)
        x[:, 1] = np.tile([1., -1., 1., -1.], n//4)
    elif pattern in {"scaled", "source_reject"}:
        x *= 32. if pattern == "scaled" else 2.0**20
    elif pattern == "nan":
        x[3, -1] = np.nan
    else:
        x = x[::-1]
    x.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph("F"), x)
    for output, fn in zip(result.outputs, (op.covariance, op.correlation), strict=True):
        reference = fn(x)
        compare(output.values[0], reference)
        if pattern in {"dense_replay", "scaled", "source_reject", "nan"} or (pattern == "reverse" and not ACTIVE):
            np.testing.assert_array_equal(output.values[0].view(np.uint64), reference.view(np.uint64))
    if ACTIVE:
        assert counts(result, "accelerate_covariance_calls") == int(pattern in {"tiny", "wide", "orthogonal"} or pattern == "reverse")
        assert counts(result, "covariance_fallbacks") == int(pattern in {"dense_replay", "scaled"})
    assert counts(result, "numeric_arena_bytes") == counts(result, "root_copy_bytes") == 0


@pytest.mark.parametrize("case", ["reverse", "different", "extra", "cov_consumer", "corr_consumer", "duplicate", "isolate", "scope"])
def test_exclusions_do_not_approximate_internal_or_other_outputs(case):
    expressions = {
        "reverse": ["correlation(x)", "covariance(x)"],
        "different": ["covariance(x)", "correlation(y)"],
        "extra": ["covariance(x)", "correlation(x)", "7"],
        "cov_consumer": ["covariance(x)", "correlation(x)", "covariance(x)>0"],
        "corr_consumer": ["covariance(x)", "correlation(x)", "correlation(x)>0"],
        "duplicate": ["covariance(x)", "covariance(x)"],
        "isolate": ["covariance(x)", "correlation(x)"],
        "scope": ["covariance(x)", "iterate(correlation(x),correlation(x),0,1)"],
    }[case]
    x = inputs(n=129, d=256)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph("F", expressions, case == "isolate"), x)
    assert counts(result, "accelerate_covariance_attempts") == 0
    cov, corr = op.covariance(x), op.correlation(x)
    for expr, output in zip(expressions, result.outputs, strict=True):
        reference = corr if "correlation(" in expr else cov if "covariance(" in expr else 7.
        if ">0" in expr:
            np.testing.assert_array_equal(output.values[0], reference > 0)
        else:
            np.testing.assert_array_equal(np.asarray(output.values[0]).view(np.uint64), np.asarray(reference).view(np.uint64))


@pytest.mark.parametrize("layout", ["c", "f", "reverse_f"])
def test_prepared_error_recovery_retains_prior_pair_and_budget(layout):
    x = inputs(layout, n=129, d=256)
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        c = scheduler.plan(graph("C"), {"x": x, "y": x}, starts, ends)
        f = scheduler.plan(graph("F"), {"x": x, "y": x}, starts, ends)
        assert c.estimated_total_memory_bytes == f.estimated_total_memory_bytes
        with pytest.raises(MemoryError):
            scheduler.plan(graph("F"), {"x": x, "y": x}, starts, ends,
                           memory_budget_bytes=f.estimated_total_memory_bytes-1)
        prepared = scheduler.prepare_execution(graph("F"), {"x": x, "y": x}, starts, ends)
        first = prepared.run_snapshot()
        saved = [output.values[0].copy() for output in first.outputs]
        original = x.copy()
        x.setflags(write=True)
        x[:, -1] = 2
        x.setflags(write=False)
        with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
            prepared.run_snapshot()
        isolated = execute(scheduler, graph("F", isolate=True), x)
        assert isolated.outputs[0].root_statuses[0] == 0
        assert isolated.outputs[1].root_statuses[0] != 0
        x.setflags(write=True)
        x[:] = original
        x.setflags(write=False)
        recovered = prepared.run_snapshot()
    for i, fn in enumerate((op.covariance, op.correlation)):
        compare(recovered.outputs[i].values[0], fn(x))
        np.testing.assert_array_equal(first.outputs[i].values[0].view(np.uint64), saved[i].view(np.uint64))
        assert not np.shares_memory(first.outputs[i].values[0], recovered.outputs[i].values[0])


@pytest.mark.parametrize("layout", ["c", "f", "reverse_f"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n", [1024, 1025, 2048, 2049])
def test_paired_panel_boundaries_preserve_both_results(layout, order, n):
    x = inputs(layout, n=n, d=256)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph(order), x)
    for output, fn in zip(result.outputs, (op.covariance, op.correlation), strict=True):
        compare(output.values[0], fn(x))
    if ACTIVE:
        block = min(n, 1024) if layout != "f" and n <= 2048 else n if n <= 2048 else 512
        assert counts(result, "accelerate_covariance_calls") == 1
        assert counts(result, "covariance_syrk_calls") == (n+block-1)//block
        assert counts(result, "covariance_panel_materialization_bytes") == x.nbytes
        assert counts(result, "native_library_workspace_reserve_bytes") == (min(n, 2048)+3)*256*8+32*1024*1024


@pytest.mark.parametrize("layout", ["c", "reverse_c", "reverse_f", "padded_f"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_short_panel_retry_preserves_joint_roots_and_worker_layout(layout, order, lane):
    x = inputs(layout, n=1031, d=259)
    x.setflags(write=True)
    x *= 4
    x.setflags(write=False)
    before = x.view(np.uint64).copy()
    transported = x
    if lane.startswith("process") and layout in {"padded_f", "reverse_f"}:
        transported = np.array(x, order="F" if layout == "reverse_f" else "C")
    expected = (op.covariance(transported), op.correlation(transported))
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = execute(scheduler, graph(order), x, 2)
        retained = [output.values[0].copy() for output in result.outputs]
        again = execute(scheduler, graph(order), x, 2)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    retry = ACTIVE and transported.strides[0] != 8
    if ACTIVE:
        assert counts(result, "accelerate_covariance_calls") == (2 if retry else 0)
        assert counts(result, "covariance_fallbacks") == (0 if retry else 2)
        assert counts(result, "covariance_syrk_calls") == (18 if retry else 0)
        rows = 1024 + 1031 if retry else 1031
        assert counts(result, "covariance_panel_materialization_bytes") == 2 * rows * 259 * 8
    for index, reference in enumerate(expected):
        for actual in result.outputs[index].values:
            compare(actual, reference)
            assert not actual.flags.writeable and not np.shares_memory(actual, x)
            assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
        np.testing.assert_array_equal(result.outputs[index].values[0].view(np.uint64), retained[index].view(np.uint64))
        assert not np.shares_memory(result.outputs[index].values[0], again.outputs[index].values[0])
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[1].values[0])
    np.testing.assert_array_equal(x.view(np.uint64), before)


@pytest.mark.parametrize("amplitude", [8., np.nextafter(8., np.inf), 32., np.nextafter(32., np.inf), 64.])
def test_short_panel_amplitude_boundaries_do_not_change_values(amplitude):
    x = np.tile(np.array([amplitude, -amplitude])[:, None], (256, 256))
    x.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph(), x)
    for output, fn in zip(result.outputs, (op.covariance, op.correlation), strict=True):
        compare(output.values[0], fn(x))
    if ACTIVE:
        assert counts(result, "accelerate_covariance_calls") == int(amplitude <= 32.)
        assert counts(result, "covariance_syrk_calls") == (1 if amplitude <= 8. else 4 if amplitude <= 32. else 0)
        assert counts(result, "covariance_fallbacks") == int(amplitude > 32.)


@pytest.mark.parametrize("layout", ["c", "reverse_f"])
def test_short_panel_late_rejection_replays_the_complete_canonical_result(layout):
    x = inputs(layout, n=4096, d=256)
    x.setflags(write=True)
    x *= 4
    x[-1, 0] = 64
    x.setflags(write=False)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(scheduler, graph(), x)
    for output, fn in zip(result.outputs, (op.covariance, op.correlation), strict=True):
        np.testing.assert_array_equal(output.values[0].view(np.uint64), fn(x).view(np.uint64))
    if ACTIVE:
        assert counts(result, "covariance_fallbacks") == 1
        assert counts(result, "covariance_syrk_calls") == 31
        assert counts(result, "covariance_panel_materialization_bytes") == (512 + 4096) * 256 * 8


@pytest.mark.parametrize("layout", ["c", "reverse_f"])
@pytest.mark.parametrize("paired", [False, True])
def test_prepared_short_panel_transition_rereads_data_and_keeps_prior_owners(layout, paired):
    x = inputs(layout, n=1031, d=259)
    original = x.copy()
    expressions = ["covariance(x)", "correlation(x)"] if paired else ["covariance(x)"]
    compiled = graph("F", expressions)
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(compiled, {"x": x, "y": x}, starts, ends)
        with pytest.raises(MemoryError):
            scheduler.plan(compiled, {"x": x, "y": x}, starts, ends,
                           memory_budget_bytes=plan.estimated_total_memory_bytes-1)
        prepared = scheduler.prepare_execution(compiled, {"x": x, "y": x}, starts, ends)
        snapshots = []
        for scale in (1., 4., 64., 1.):
            x.setflags(write=True)
            x[:] = original * scale
            x.setflags(write=False)
            result = prepared.run_snapshot()
            if ACTIVE:
                assert counts(result, "covariance_fallbacks") == int(scale == 64.)
                assert counts(result, "covariance_syrk_calls") == (9 if scale == 4. else 0 if scale == 64. else 2 if paired else 1)
            for output, fn in zip(result.outputs, (op.covariance, op.correlation)[:len(expressions)], strict=True):
                compare(output.values[0], fn(x))
            snapshots.append((result, [out.values[0].copy() for out in result.outputs]))
    for result, saved in snapshots:
        for output, reference in zip(result.outputs, saved, strict=True):
            np.testing.assert_array_equal(output.values[0].view(np.uint64), reference.view(np.uint64))
            assert not output.values[0].flags.writeable
    assert not np.shares_memory(snapshots[0][0].outputs[0].values[0], snapshots[-1][0].outputs[0].values[0])
