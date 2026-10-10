"""Wide private replay: original math/order, complete error and ownership contracts."""

import hashlib

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler
from calmetrics_engine import operators as op

from . import latent_chain as chain
from .test_gaussian_filter_stream import fixture, iteration_graph, run
from .test_latent_workspace_loan import exact, selected


@pytest.mark.parametrize("n,d,k", [(16387, 8, 8), (10927, 32, 12), (8193, 32, 16), (26215, 5, 5)])
@pytest.mark.parametrize("order", ["c", "f", "reverse", "padded", "zero"])
@pytest.mark.parametrize("continuation", [0, 1])
def test_wide_replay_matches_all_rows_and_removes_private_matrices(n, d, k, order, continuation):
    data = fixture(n, d, k, order)
    before = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in data.items()}
    density = op.gaussian_log_density(data["x"], data["means"], data["variances"])
    history = op.hmm_filter(density, data["initial"], data["transition"], continuation)
    result, plan = run(data, continuation=continuation)
    original, _ = run(data, continuation=continuation, control=True)
    for value in result.outputs[0].values:
        assert value.tobytes() == history[-1].tobytes() == original.outputs[0].values[0].tobytes()
        assert not value.flags.writeable
    assert sum(c["gaussian_filter_streams"] for c in result.audit["native_chunks"]) == 2
    assert all(c["numeric_arena_bytes"] < density.nbytes for c in result.audit["native_chunks"])
    assert "gaussian_filter_stream_replay_work_accounted" in plan.reason_codes
    assert before == {
        key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in data.items()
    }


@pytest.mark.parametrize(
    "lane,prepared",
    [("single", True), ("thread", False), ("process_inline", False), ("process_shared", False)],
)
@pytest.mark.parametrize("order", ["c", "f"])
def test_wide_workers_and_prepared_outputs(lane, prepared, order):
    data = fixture(16387, 8, 8, order)
    result, _ = run(data, lane=lane, prepared=prepared)
    original, _ = run(data, control=True)
    assert sum(c["gaussian_filter_streams"] for c in result.audit["native_chunks"]) == 2
    for value, target in zip(result.outputs[0].values, original.outputs[0].values, strict=True):
        assert value.tobytes() == target.tobytes()
        assert not np.shares_memory(value, target)


@pytest.mark.parametrize(
    "kind",
    [
        "late_nan",
        "late_overflow_bad_initial",
        "late_overflow_bad_transition",
        "negative_variance",
        "bad_transition_shape",
    ],
)
def test_wide_errors_preserve_producer_priority(kind):
    data = {key: value.copy() for key, value in fixture(16387, 8, 8).items()}
    if kind == "late_nan":
        data["x"][-1, -1] = np.nan
    elif kind.startswith("late_overflow"):
        data["x"][-1, -1] = np.finfo(np.float64).max
        if kind.endswith("initial"):
            data["initial"][0] = -1
        else:
            data["transition"][0, 0] = -1
    elif kind == "negative_variance":
        data["variances"][-1, -1] = -1
    else:
        data["transition"] = np.ones((8, 7))
    errors = []
    for control in [False, True]:
        with pytest.raises(Exception) as error:
            run(data, control=control)
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]
    if kind.startswith("late_"):
        assert "NON_FINITE_RESULT" in errors[0][1]


@pytest.mark.parametrize(
    "n,d,k,eligible",
    [
        (16383, 8, 8, False),
        (16384, 8, 8, True),
        (8192, 32, 16, True),
        (8192, 33, 16, False),
        (8192, 32, 17, False),
    ],
)
def test_wide_admission_boundaries(n, d, k, eligible):
    data = fixture(n, d, k)
    result, plan = run(data)
    original, _ = run(data, control=True)
    assert result.outputs[0].values[0].tobytes() == original.outputs[0].values[0].tobytes()
    assert (
        sum(c["gaussian_filter_streams"] for c in result.audit["native_chunks"]) > 0
    ) == eligible
    assert ("gaussian_filter_stream_replay_work_accounted" in plan.reason_codes) == eligible


def test_wide_intervening_writer_declines_replay():
    data = fixture(16387, 8, 8)
    data["initial_rows"] = np.tile(data["initial"] / len(data["x"]), (len(data["x"]), 1))
    data["initial_rows"].flags.writeable = False
    result, _ = run(data, interleave=True)
    original, _ = run(data, interleave=True, control=True)
    assert result.outputs[0].values[0].tobytes() == original.outputs[0].values[0].tobytes()
    assert sum(c["gaussian_filter_streams"] for c in result.audit["native_chunks"]) == 0


def test_overlapping_columns_keep_original_density_library_admission():
    data = fixture(16387, 8, 8)
    backing = np.random.default_rng(8901).normal(size=16387 + 7 * 129)
    data["x"] = np.lib.stride_tricks.as_strided(
        backing, shape=(16387, 8), strides=(8, 129 * 8), writeable=False
    )
    density = op.gaussian_log_density(data["x"], data["means"], data["variances"])
    expected = op.hmm_filter(density, data["initial"], data["transition"])[-1]
    result, _ = run(data)
    assert sum(c["gaussian_filter_streams"] for c in result.audit["native_chunks"]) == 2
    assert sum(c["eigen_density_calls"] for c in result.audit["native_chunks"]) == 0
    assert result.outputs[0].values[0].tobytes() == expected.tobytes()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("isolate", [False, True])
def test_wide_child_work_budget_and_result_parity(nested, isolate):
    data = fixture(16387, 8, 8)
    starts = np.zeros(1, np.int64)
    ends = np.array([len(data["x"])], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        g = iteration_graph(data, nested=nested, isolate=isolate)
        control = iteration_graph(data, nested=nested, isolate=isolate, control=True)
        plan = scheduler.plan(g, data, starts, ends)
        original = scheduler.plan(control, data, starts, ends)
        result = scheduler.execute(g, data, starts, ends)
        reference = scheduler.execute(control, data, starts, ends)
    assert plan.estimated_work_units > original.estimated_work_units
    assert sum(c["gaussian_filter_streams"] for c in result.audit["native_chunks"]) > 0
    for root, target in zip(result.outputs, reference.outputs, strict=True):
        for value, expected in zip(root.values, target.values, strict=True):
            assert value.tobytes() == expected.tobytes()
            assert not np.shares_memory(value, expected)


@pytest.mark.parametrize("d,k,streaming", [(4, 12, False), (8, 8, True)])
def test_complete_hmm_keeps_free_beta_history_after_arena_elision(d, k, streaming):
    data = chain.inputs(240000, d, k)
    graph, _ = chain.graph_for(data, k, 3, 17, "hmm")
    starts = np.zeros(1, np.int64)
    ends = np.full(1, 240000, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        inputs = selected(data, graph)
        canonical = scheduler.prepare_execution(graph, inputs, starts, ends).run_snapshot()
        first = scheduler.execute(graph, inputs, starts, ends)
        second = scheduler.execute(graph, inputs, starts, ends)
    exact(first, canonical)
    exact(second, canonical)
    exact(first, second)
    beta_copies = 3 * len(data["training_positions"]) * k * 8
    for result in [first, second]:
        chunk = result.audit["native_chunks"][0]
        assert chunk["algorithm_copy_bytes"] < beta_copies
        assert (chunk["gaussian_filter_streams"] > 0) == streaming
        assert (
            chunk["operator_workspace_capacity_bytes"]
            < canonical.audit["native_chunks"][0]["operator_workspace_capacity_bytes"]
        )
