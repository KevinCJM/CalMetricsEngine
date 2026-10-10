"""Wider Eigen F lanes preserve direct, fit and complete-graph contracts."""

import os
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler
from calmetrics_engine import operators as op

from . import latent_chain as chain
from .test_latent_library_backend import arguments, execute, feature_stride_view
from .test_latent_operators import fixture
from .test_latent_workspace_loan import exact, selected


def bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def eigen_platform():
    eligible = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and int(platform.mac_ver()[0].split(".")[0]) >= 15
    )
    flag = os.environ.get("CME_EXPECT_EIGEN")
    return eligible and flag != "0"


@pytest.mark.parametrize(
    "n,d,k",
    [
        (7, 32, 12),
        (31, 32, 12),
        (32, 7, 12),
        (32, 8, 9),
        (33, 32, 12),
        (257, 32, 16),
        (257, 32, 17),
        (257, 8, 8),
    ],
)
@pytest.mark.parametrize("layout", ["f", "f_padded", "c", "reverse_rows"])
def test_wide_column_dispatch_boundaries_and_exact_density(n, d, k, layout):
    rng = np.random.default_rng(9301)
    raw = rng.normal(size=(n, d))
    x = np.array(raw, order="F")
    if layout == "f_padded":
        backing = np.empty((n + 11, d), order="F")
        backing[:n] = raw
        x = backing[:n]
    elif layout == "c":
        x = np.array(raw, order="C")
    elif layout == "reverse_rows":
        x = x[::-1]
    means = rng.normal(size=(k, d))
    variances = rng.uniform(0.1, 3.0, size=(k, d))
    arrays = (x, means, variances)
    saved = [a.tobytes() for a in arrays]
    for a in arrays:
        a.flags.writeable = False
    scalar, control = op.call("gaussian_log_density", *arrays, simd="scalar", audit=True)
    actual, audit = op.call("gaussian_log_density", *arrays, audit=True)
    bits(actual, scalar)
    eligible = n >= 32 and d >= 8 and k <= 16 and x.strides[0] == 8 and x.strides[1] >= n * 8
    if os.environ.get("CME_EXPECT_EIGEN") is not None:
        assert (audit["eigen_density_calls"] > 0) == (eligible and eigen_platform())
    else:
        assert not audit["eigen_density_calls"] or eligible
    assert control["eigen_density_calls"] == 0
    assert audit["input_copy_bytes"] == control["input_copy_bytes"] == 0
    assert audit["workspace_bytes"] == control["workspace_bytes"]
    assert saved == [a.tobytes() for a in arrays]
    assert all(not np.shares_memory(actual, a) for a in arrays)


@pytest.mark.parametrize("k", [9, 12, 16])
@pytest.mark.parametrize("kind", ["first_row", "later_packet", "last_row", "earlier_component"])
def test_wide_density_overflow_preserves_complete_error_and_output_prefix(k, kind):
    x = np.zeros((33, 32), order="F")
    means = np.zeros((k, 32))
    variances = np.ones((k, 32))
    row = {"first_row": 0, "later_packet": 9, "last_row": 32, "earlier_component": 32}[kind]
    x[row, 0] = np.finfo(np.float64).max
    if kind == "earlier_component":
        # A later row's overflow must not hide the first row's component error.
        means[1, 1] = np.finfo(np.float64).max
    for a in (x, means, variances):
        a.flags.writeable = False
    outputs = []
    errors = []
    error_types = []
    for simd in ("scalar", "auto"):
        out = np.full((33, k), 1234.5)
        with pytest.raises((ValueError, RuntimeError)) as error:
            op.call("gaussian_log_density", x, means, variances, out=out, simd=simd)
        errors.append(str(error.value))
        error_types.append(type(error.value))
        outputs.append(out)
    assert errors[0] == errors[1] and "NON_FINITE_RESULT" in errors[0]
    assert error_types[0] == error_types[1]
    bits(*outputs)


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("d,k", [(9, 9), (32, 12), (32, 16), (32, 17)])
def test_wide_f_fit_diagnostics_workspace_inputs_and_old_results(hmm, d, k):
    x, w, a, m, v = fixture(65, d, k, seed=9302)
    x = np.asfortranarray(x)
    arrays = (x, w, a, m, v)
    saved = [z.tobytes() for z in arrays]
    for z in arrays:
        z.flags.writeable = False
    expected, scalar = execute(
        hmm, arguments(hmm, feature_stride_view(x), w, a, m, v, 4), audit=True
    )
    workspace = op.Workspace()
    actual, audit = execute(hmm, arguments(hmm, *arrays, 4), workspace=workspace, audit=True)
    bits(actual, expected)
    assert audit["workspace_bytes"] == scalar["workspace_bytes"]
    assert audit["algorithm_copy_bytes"] == scalar["algorithm_copy_bytes"]
    assert audit["input_copy_bytes"] == scalar["input_copy_bytes"] == 0
    old = actual.copy()
    changed = np.array(x + 0.01, order="F")
    changed.flags.writeable = False
    later = execute(hmm, arguments(hmm, changed, w, a, m, v, 4), workspace=workspace)
    bits(actual, old)
    assert not np.shares_memory(actual, later)
    assert all(not np.shares_memory(actual, z) for z in arrays)
    assert saved == [z.tobytes() for z in arrays]


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("k", [9, 12, 16])
def test_wide_f_stopping_neighbors_remain_exact(hmm, k):
    x, w, a, m, v = fixture(65, 32, k, seed=9303)
    x = np.asfortranarray(x)
    scalar = feature_stride_view(x)
    first = execute(hmm, arguments(hmm, scalar, w, a, m, v, 1))
    second = execute(hmm, arguments(hmm, scalar, w, a, m, v, 2))
    difference = abs(op.em_log_likelihood(second) - op.em_log_likelihood(first))
    assert difference > 0
    for tolerance in (np.nextafter(difference, 0.0), difference, np.nextafter(difference, np.inf)):
        expected = execute(hmm, arguments(hmm, scalar, w, a, m, v, 3, tolerance))
        actual = execute(hmm, arguments(hmm, x, w, a, m, v, 3, tolerance))
        bits(actual, expected)


@pytest.mark.parametrize("model", ["gmm", "hmm"])
def test_wide_f_complete_thirteen_outputs_keep_snapshot_and_old_ownership(model):
    n, k = 16387, 12
    data = chain.inputs(n, 32, k)
    data["raw"] = np.array(data["raw"], order="F", copy=True)
    data["raw"].flags.writeable = False
    graph, _ = chain.graph_for(data, k, 3, 17, model)
    data = selected(data, graph)
    saved = {key: value.tobytes() for key, value in data.items()}
    starts = np.zeros(1, np.int64)
    ends = np.full(1, n, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        canonical = scheduler.prepare_execution(graph, data, starts, ends).run_snapshot()
        first = scheduler.execute(graph, data, starts, ends)
        frozen = [root.values[0].tobytes() for root in first.outputs]
        later = scheduler.execute(graph, data, starts, ends)
    assert len(first.outputs) == 13
    if os.environ.get("CME_EXPECT_EIGEN") is not None:
        calls = sum(chunk["eigen_density_calls"] for chunk in first.audit["native_chunks"])
        assert (calls > 0) == eigen_platform()
    exact(first, canonical)
    exact(later, canonical)
    assert frozen == [root.values[0].tobytes() for root in first.outputs]
    assert saved == {key: value.tobytes() for key, value in data.items()}
    assert all(
        not np.shares_memory(a.values[0], b.values[0])
        for a, b in zip(first.outputs, later.outputs, strict=True)
    )
