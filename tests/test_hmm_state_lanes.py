"""Fixed-iteration state batches reuse xi without changing stopped fits."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler
from calmetrics_engine import operators as op

from . import latent_reference as ref
from .test_hmm_transition_vforce import check_outputs, compare_models, projections
from .test_hmm_xi_tiles import bits, fit, rounding_mode, sample
from .test_latent_operators import fixture
from .test_nonnegative_matmul_accelerate import ACCELERATE
from .test_typed_results import config

CASES = [
    (4095, 1, 8, "c"),
    (4096, 1, 7, "c"),
    (4096, 1, 8, "c"),
    (4097, 1, 9, "negative"),
    (4225, 4, 12, "padded"),
    (4096, 8, 16, "f"),
    (4097, 4, 32, "c"),
    (4096, 1, 64, "c"),
    (4096, 1, 128, "c"),
    (4096, 1, 129, "c"),
]


@pytest.mark.parametrize("n,d,k,layout", CASES)
def test_states_geometry_order_and_owned_results(n, d, k, layout):
    values, owners = sample(n, d, k, layout)
    before = [v.copy() for v in values + tuple(owners)]
    actual, audit = fit(values, 1, audit=True)
    expected, scalar = fit(values, 1, simd="scalar", audit=True)
    compare_models(actual, expected)
    assert op.em_iterations(actual) == 1 and not op.em_converged(actual)
    old_calls = (3 * k * k - 2) * ((n - 2) // 128 + 1) if ACCELERATE and n >= 4096 else 0
    new_calls = 2 * (n - 1) if ACCELERATE and n >= 4096 and 8 <= k <= 128 else 0
    assert audit["accelerate_vforce_calls"] == old_calls + new_calls
    assert scalar["accelerate_vforce_calls"] == 0
    assert audit["workspace_bytes"] == scalar["workspace_bytes"]
    assert audit["algorithm_copy_bytes"] == scalar["algorithm_copy_bytes"]
    assert audit["input_copy_bytes"] == 0
    for old, saved in zip(values + tuple(owners), before, strict=True):
        bits(old, saved)
        assert not old.flags.writeable and not np.shares_memory(old, actual)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(op.hmm_initial(actual).sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(op.hmm_transition(actual).sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("k,layout", [(8, "c"), (12, "negative"), (16, "f"), (32, "padded")])
def test_fixed_iterations_reference_reuse_canaries_and_labels(k, layout):
    values, _ = sample(4097, 1, k, layout)
    work = op.Workspace()
    work.reserve(doubles=2 * 4097 * k + 2 * k * k + 4 * k + 512 + 128 * k + 19)
    capacity = work.capacity_bytes
    shape = (k, k + 6)
    storage = np.full(np.prod(shape) + 2, -771.0)
    output = storage[1:-1].reshape(shape)
    actual = fit(values, 2, workspace=work, out=output)
    expected = fit(values, 2, simd="scalar")
    compare_models(actual, expected)
    assert actual is output and storage[0] == storage[-1] == -771.0
    reference = ref.hmm_em_reference(*values, 2, 0.0)
    for project, target in zip(
        (op.hmm_initial, op.hmm_transition, op.hmm_means, op.hmm_variances),
        reference[:4],
        strict=True,
    ):
        np.testing.assert_allclose(project(actual), target, rtol=0, atol=0.000049)
    assert op.em_iterations(actual) == reference[4]
    np.testing.assert_allclose(op.em_log_likelihood(actual), reference[5], rtol=0, atol=0.000049)
    retained = fit(values, 2, workspace=work)
    saved = retained.tobytes()
    op.median(np.linspace(-1e6, 1e6, work.capacity_bytes // 8 + 1024), workspace=work)
    second = fit(values, 2, workspace=work)
    assert retained.tobytes() == saved
    bits(second, retained)
    assert not np.shares_memory(retained, second)
    assert work.capacity_bytes >= capacity
    classes = []
    for model in (actual, expected):
        emission = op.gaussian_log_density(values[0], op.hmm_means(model), op.hmm_variances(model))
        posterior = op.hmm_smooth(emission, op.hmm_initial(model), op.hmm_transition(model))
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=0, atol=1e-9)
        classes.append(op.argmax_axis(posterior, 1))
    bits(classes[0], classes[1])
    bits(op.argsort(op.hmm_means(actual)[:, 0]), op.argsort(op.hmm_means(expected)[:, 0]))


@pytest.mark.parametrize("mode", ["nearest", "upward", "downward", "towardzero"])
def test_rounding_modes_keep_scalar_sweeps_unless_nearest(mode):
    values, _ = sample(4097, 1, 8)
    with rounding_mode(mode) as (get, expected_mode):
        actual, audit = fit(values, 2, audit=True)
        expected = fit(values, 2, simd="scalar")
        assert get() == expected_mode
    compare_models(actual, expected, exact=mode != "nearest")
    assert bool(audit["accelerate_vforce_calls"]) == (ACCELERATE and mode == "nearest")


@pytest.mark.parametrize("kind", ["underflow", "zero_probabilities", "constant"])
def test_extreme_probabilities_and_symmetric_decisions(kind):
    values = list(fixture(4097, 1, 8))
    if kind == "underflow":
        values[3][1:] = 1000.0
        values[4][:] = 1.0
    elif kind == "zero_probabilities":
        values[1][:] = 0.0
        values[1][0] = 1.0
        values[2][:] = np.eye(8)
    else:
        values[0][:] = values[3][:] = 2.0
        values[4][:] = 0.0
    for value in values:
        value.flags.writeable = False
    actual = fit(values, 2)
    scalar = fit(values, 2, simd="scalar")
    compare_models(actual, scalar)
    decisions = []
    for model in (actual, scalar):
        emission = op.gaussian_log_density(values[0], op.hmm_means(model), op.hmm_variances(model))
        posterior = op.hmm_smooth(emission, op.hmm_initial(model), op.hmm_transition(model))
        decisions.append(op.argmax_axis(posterior, 1))
    bits(decisions[0], decisions[1])


@pytest.mark.parametrize(
    "kind,code",
    [
        ("nan", "NON_FINITE_RESULT"),
        ("variance", "DOMAIN_ERROR"),
        ("weights", "DOMAIN_ERROR"),
        ("transition", "DOMAIN_ERROR"),
        ("density", "NON_FINITE_RESULT"),
        ("forward", "NON_FINITE_RESULT"),
        ("shape", "SHAPE_MISMATCH"),
    ],
)
def test_failure_prefix_and_healthy_recovery(kind, code):
    values = list(fixture(4097, 1, 8))
    if kind == "nan":
        values[0][-128, 0] = np.nan
    elif kind == "variance":
        values[4][0, 0] = -1.0
    elif kind == "weights":
        values[1][:] = 0.0
    elif kind == "transition":
        values[2][0, :] = 0.0
    elif kind == "density":
        values[0][-128, 0] = np.finfo(float).max
    elif kind == "forward":
        values[0][:] = 2e153
        values[3][:] = 0.0
        values[4][:] = 1.0
    else:
        values[2] = np.ones((8, 9))
    for value in values:
        value.flags.writeable = False
    errors = []
    prefix = []
    for simd in ("scalar", "auto"):
        workspace = op.Workspace()
        raw = np.full(8 * 14 + 2, 771.0)
        with pytest.raises((ValueError, RuntimeError), match=code) as error:
            fit(values, 2, simd=simd, out=raw[1:-1].reshape(8, 14), workspace=workspace)
        errors.append((type(error.value), str(error.value)))
        prefix.append(raw)
        healthy, _ = sample(4096, 1, 8)
        compare_models(
            fit(healthy, 2, simd=simd, workspace=workspace), fit(healthy, 2, simd="scalar")
        )
    assert errors[0] == errors[1]
    bits(prefix[0], prefix[1])
    assert prefix[0][0] == prefix[0][-1] == 771.0


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("isolate", [False, True])
def test_transport_and_prepared_fixed_model(lane, isolate):
    values, _ = sample(4097, 1, 8)
    graph, inputs, targets = projections(values, isolate)
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(graph, inputs, bounds, bounds)
        saved = [[v.tobytes() for v in o.values] for o in first.outputs]
        second = scheduler.execute(graph, inputs, bounds, bounds)
        for result in (first, second):
            check_outputs(result, targets)
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        assert first.audit["use_shared_memory"] == (lane == "process_shared")
        for old, new in zip(first.outputs, second.outputs, strict=True):
            for a, b in zip(old.values, new.values, strict=True):
                bits(a, b)
                assert not np.shares_memory(a, b)
        assert [[v.tobytes() for v in o.values] for o in first.outputs] == saved
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, bounds, bounds)
        one = prepared.run_snapshot()
        snapshot = [[v.tobytes() for v in o.values] for o in one.outputs]
        two = prepared.run_snapshot()
        check_outputs(two, targets)
        assert [[v.tobytes() for v in o.values] for o in one.outputs] == snapshot
