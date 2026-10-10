"""HMM transition vForce route: math, planned scratch, errors and transport.

The scalar native path and frozen NJIT reference are independent of the new
batched transition helper. Float tails use the declared four-decimal budget;
stopping, classification, failure prefixes and retained result bytes stay exact.
"""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from . import latent_reference as ref
from .test_hmm_xi_tiles import bits, fit, rounding_mode, sample
from .test_latent_operators import fixture
from .test_nonnegative_matmul_accelerate import ACCELERATE
from .test_typed_results import config


def compare_models(actual, expected, *, exact=False):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.000049)
    bits(actual[:, -3], expected[:, -3])
    bits(actual[:, -1], expected[:, -1])
    if exact:
        bits(actual, expected)


CASES = [
    (4095, 1, 3, "c"),
    (4096, 1, 3, "c"),
    (4097, 1, 3, "negative"),
    (4224, 4, 3, "f"),
    (4225, 4, 12, "padded"),
    (8192, 1, 3, "c"),
    (8193, 4, 12, "f"),
    (4097, 1, 1, "c"),
    (4097, 1, 2, "negative"),
]


@pytest.mark.parametrize("n,d,k,layout", CASES)
def test_threshold_layout_scratch_canaries_and_reuse(n, d, k, layout):
    values, owners = sample(n, d, k, layout)
    originals = [a.copy() for a in values + tuple(owners)]
    expected, scalar = fit(values, 3, simd="scalar", audit=True)
    scratch = (2 if n >= 4096 and k >= 3 else 3) * n * k + 2 * k * k + 4 * k
    if n >= 4096 and k >= 3:
        scratch += 512 + 128 * k
    elif n >= 128 and k >= 3:
        scratch += min(n - 1, 128)
    work = op.Workspace()
    work.reserve(doubles=scratch + 19)
    capacity = work.capacity_bytes
    storage = np.full(expected.size + 2, -771.0)
    out = storage[1:-1].reshape(expected.shape)
    actual, audit = fit(values, 3, out=out, workspace=work, audit=True)
    admitted = ACCELERATE and n >= 4096 and k >= 3
    compare_models(actual, expected, exact=not admitted and d == 1)
    assert actual is out and storage[0] == storage[-1] == -771.0
    assert bool(audit["accelerate_vforce_calls"]) == admitted
    assert scalar["accelerate_vforce_calls"] == 0
    assert audit["workspace_bytes"] == scalar["workspace_bytes"] == scratch * 8
    assert work.capacity_bytes == capacity
    assert audit["input_copy_bytes"] == 0
    initial_copies = (k * (1 + 2 * d) + k * k) * 8
    replay_copies = 3 * n * k * 8 if n >= 4096 and k >= 3 else 0
    assert (
        audit["algorithm_copy_bytes"]
        == scalar["algorithm_copy_bytes"]
        == initial_copies + replay_copies
    )
    first = fit(values, 3, workspace=work)
    retained = first.tobytes()
    op.median(np.linspace(-1e6, 1e6, scratch + 1024), workspace=work)
    second = fit(values, 3, workspace=work)
    assert first.tobytes() == retained
    bits(second, first)
    assert not np.shares_memory(first, second)
    for a, original in zip(values + tuple(owners), originals, strict=True):
        bits(a, original)
        assert not a.flags.writeable and not np.shares_memory(first, a)


@pytest.mark.parametrize("n,d,k", [(4096, 1, 3), (4097, 4, 3)])
def test_independent_reference_and_posterior_decisions(n, d, k):
    values, _ = sample(n, d, k)
    model = fit(values, 3)
    expected = ref.hmm_em_reference(*values, 3, 0.0)
    for project, target in zip(
        (op.hmm_initial, op.hmm_transition, op.hmm_means, op.hmm_variances),
        expected[:-2],
        strict=True,
    ):
        np.testing.assert_allclose(project(model), target, rtol=0, atol=0.000049)
    assert op.em_iterations(model) == expected[-2] and not op.em_converged(model)
    np.testing.assert_allclose(op.em_log_likelihood(model), expected[-1], rtol=0, atol=0.000049)
    scalar = fit(values, 3, simd="scalar")
    posteriors = []
    for current in (model, scalar):
        em = op.gaussian_log_density(values[0], op.hmm_means(current), op.hmm_variances(current))
        p = op.hmm_smooth(em, op.hmm_initial(current), op.hmm_transition(current))
        # Full log-alpha/beta values grow with N. The unchanged scalar smoother
        # also has ~2e-12 normalization tails at this size; use the existing
        # inference contract, while checking every classification exactly.
        np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-10)
        posteriors.append(p)
    np.testing.assert_array_equal(
        op.argmax_axis(posteriors[0], 1), op.argmax_axis(posteriors[1], 1)
    )


@pytest.mark.parametrize("mode", ["nearest", "downward", "upward", "towardzero"])
@pytest.mark.parametrize("n,k", [(4096, 3), (4097, 12)])
def test_rounding_dispatch_and_environment_preservation(mode, n, k):
    values, _ = sample(n, 1, k)
    with rounding_mode(mode) as (get_rounding, expected_mode):
        expected = fit(values, 2, simd="scalar")
        actual, audit = fit(values, 2, audit=True)
        assert get_rounding() == expected_mode
    admitted = ACCELERATE and mode == "nearest"
    compare_models(actual, expected, exact=not admitted)
    assert bool(audit["accelerate_vforce_calls"]) == admitted


@pytest.mark.parametrize("n,k", [(4096, 3), (4097, 12), (8192, 3), (8193, 12)])
def test_strict_stopping_neighbors(n, k):
    values, _ = sample(n, 1, k)
    previous = fit(values, 1, simd="scalar")
    current = fit(values, 2, simd="scalar")
    delta = abs(op.em_log_likelihood(current) - op.em_log_likelihood(previous))
    assert delta > 0 and np.isfinite(delta)
    for tolerance in [np.nextafter(delta, 0.0), delta, np.nextafter(delta, np.inf)]:
        expected = fit(values, 3, float(tolerance), simd="scalar")
        actual = fit(values, 3, float(tolerance))
        compare_models(actual, expected)
        assert op.em_iterations(actual) == (2 if tolerance > delta else 3)


@pytest.mark.parametrize("factor,allowed", [(0.8, True), (1.2, False)])
def test_finite_bound_admission_and_canonical_rejection(factor, allowed):
    n, k = 4096, 3
    values = list(fixture(n, 1, k))
    scale = np.sqrt((np.finfo(float).max / 32) * 2 / (n - 1)) * factor
    values[0][:] = scale
    values[3][:] = 0.0
    values[4][:] = 1.0
    for value in values:
        value.flags.writeable = False
    expected = fit(values, 1, simd="scalar")
    actual, audit = fit(values, 1, audit=True)
    admitted = ACCELERATE and allowed
    compare_models(actual, expected, exact=not admitted)
    assert bool(audit["accelerate_vforce_calls"]) == admitted


@pytest.mark.parametrize("kind", ["underflow", "zero_probabilities", "constant"])
def test_underflow_probability_and_variance_floors(kind):
    values = list(fixture(4097, 1, 3))
    if kind == "underflow":
        values[3][1:] = 1000.0
        values[4][:] = 1.0
    elif kind == "zero_probabilities":
        values[1][:] = [1.0, 0.0, 0.0]
        values[2][:] = np.eye(3)
    else:
        values[0][:] = values[3][:] = 2.0
        values[4][:] = 0.0
    for value in values:
        value.flags.writeable = False
    model, audit = fit(values, 3, audit=True)
    compare_models(model, fit(values, 3, simd="scalar"))
    assert bool(audit["accelerate_vforce_calls"]) == ACCELERATE
    np.testing.assert_allclose(op.hmm_initial(model).sum(), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(op.hmm_transition(model).sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.all(op.hmm_variances(model) >= 1e-5)


@pytest.mark.parametrize(
    "kind,code",
    [
        ("nan", "NON_FINITE_RESULT"),
        ("negative_variance", "DOMAIN_ERROR"),
        ("zero_weights", "DOMAIN_ERROR"),
        ("zero_transition", "DOMAIN_ERROR"),
        ("density_overflow", "NON_FINITE_RESULT"),
        ("forward_overflow", "NON_FINITE_RESULT"),
        ("shape", "SHAPE_MISMATCH"),
    ],
)
def test_error_prefix_and_workspace_recovery(kind, code):
    values = list(fixture(4097, 1, 3))
    if kind == "nan":
        values[0][-128, 0] = np.nan
    elif kind == "negative_variance":
        values[4][0, 0] = -1.0
    elif kind == "zero_weights":
        values[1][:] = 0.0
    elif kind == "zero_transition":
        values[2][0] = 0.0
    elif kind == "density_overflow":
        values[0][-128, 0] = np.finfo(float).max
    elif kind == "forward_overflow":
        values[0][:] = 2e153
        values[3][:] = 0.0
        values[4][:] = 1.0
    else:
        values[2] = np.ones((3, 4))
    for value in values:
        value.flags.writeable = False
    originals = [value.copy() for value in values]
    outputs, errors = [], []
    healthy, _ = sample(4096, 1, 3)
    for simd in ["scalar", "auto"]:
        work = op.Workspace()
        storage = np.full(29, 771.0)
        with pytest.raises((ValueError, RuntimeError), match=code) as error:
            fit(values, 2, simd=simd, out=storage[1:-1].reshape(3, 9), workspace=work)
        errors.append((type(error.value), str(error.value)))
        outputs.append(storage)
        compare_models(fit(healthy, 2, simd=simd, workspace=work), fit(healthy, 2, simd="scalar"))
    assert errors[0] == errors[1]
    bits(outputs[0], outputs[1])
    assert outputs[0][0] == outputs[0][-1] == 771.0
    for value, original in zip(values, originals, strict=True):
        bits(value, original)


def projections(values, isolate=False):
    inputs = dict(zip(["x", "w", "a", "m", "v"], values, strict=True))
    axes = dict(
        x=["scenario", "factor"],
        w=["component"],
        a=["component", "component"],
        m=["component", "factor"],
        v=["component", "factor"],
    )
    definitions = {
        n: dict(
            kind="matrix" if v.ndim == 2 else "vector",
            shape=[str(size) for size in v.shape],
            axes=axes[n],
        )
        for n, v in inputs.items()
    }
    roots = dict(
        initial="hmm_initial(trained)",
        transition="hmm_transition(trained)",
        means="hmm_means(trained)",
        variances="hmm_variances(trained)",
        iterations="em_iterations(trained)",
        likelihood="em_log_likelihood(trained)",
        converged="em_converged(trained)",
        independent="sum(w)",
    )
    g = GraphCompiler(definitions).compile(
        roots,
        root_bindings=[{"trained": "gaussian_hmm_em(x,w,a,m,v,2,0)"}] * len(roots),
        result_format="typed",
        error_policy="isolate" if isolate else "raise",
    )
    scalar = fit(values, 2, simd="scalar")
    targets = [
        p(scalar)
        for p in [
            op.hmm_initial,
            op.hmm_transition,
            op.hmm_means,
            op.hmm_variances,
            op.em_iterations,
            op.em_log_likelihood,
            op.em_converged,
        ]
    ] + [np.asarray(sum(float(x) for x in values[1]))]
    return g, inputs, targets


def check_outputs(result, targets):
    for out, target in zip(result.outputs, targets, strict=True):
        target = np.asarray(target)
        np.testing.assert_array_equal(out.root_statuses, np.zeros(len(out.values), np.int16))
        for value in out.values:
            assert value.dtype == target.dtype and value.shape == target.shape
            assert not value.flags.writeable
            if target.dtype.kind in "ib":
                bits(value, target)
            else:
                np.testing.assert_allclose(value, target, rtol=0, atol=0.000049)


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("isolate", [False, True])
def test_typed_native_transport_retained_generations_and_failure_recovery(lane, isolate):
    values, _ = sample(4097, 1, 3)
    g, inputs, targets = projections(values, isolate)
    originals = [v.copy() for v in values]
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(g, inputs, bounds, bounds)
        saved = [[v.tobytes() for v in out.values] for out in first.outputs]
        second = scheduler.execute(g, inputs, bounds, bounds)
        for result in [first, second]:
            check_outputs(result, targets)
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        assert first.audit["use_shared_memory"] == (lane == "process_shared")
        for old, new in zip(first.outputs, second.outputs, strict=True):
            for a, b in zip(old.values, new.values, strict=True):
                bits(a, b)
                assert not np.shares_memory(a, b)
        assert (
            bool(sum(c["accelerate_vforce_calls"] for c in first.audit["native_chunks"]))
            == ACCELERATE
        )
        bad = dict(inputs)
        bad["w"] = np.zeros(3)
        bad["w"].flags.writeable = False
        if isolate:
            failed = scheduler.execute(g, bad, bounds, bounds)
            for out in failed.outputs[:-1]:
                assert np.all(out.root_statuses != 0)
            np.testing.assert_array_equal(failed.outputs[-1].root_statuses, [0, 0])
            for value in failed.outputs[-1].values:
                assert value == 0.0
        else:
            with pytest.raises((ValueError, RuntimeError), match="DOMAIN"):
                scheduler.execute(g, bad, bounds, bounds)
        check_outputs(scheduler.execute(g, inputs, bounds, bounds), targets)
        for out, snapshot in zip(first.outputs, saved, strict=True):
            assert [v.tobytes() for v in out.values] == snapshot
    for value, original in zip(values, originals, strict=True):
        bits(value, original)


@pytest.mark.parametrize("prepared", [False, True])
def test_graph_budget_and_snapshot_ownership(prepared):
    values, _ = sample(4097, 1, 12)
    g, inputs, targets = projections(values)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(g, inputs, bounds, bounds)
        with pytest.raises(
            (ValueError, RuntimeError, MemoryError), match="memory|MEMORY|bad_alloc"
        ):
            scheduler.execute(g, inputs, bounds, bounds, memory_budget_bytes=1000)
        session = scheduler.prepare_execution(g, inputs, bounds, bounds) if prepared else None
        call = (
            session.run_snapshot
            if session
            else lambda: scheduler.execute(g, inputs, bounds, bounds)
        )
        first, second = call(), call()
        check_outputs(first, targets)
        check_outputs(second, targets)
        assert (
            plan.estimated_worker_scratch_bytes
            >= first.audit["native_chunks"][0]["operator_workspace_capacity_bytes"]
        )
        for old, new in zip(first.outputs, second.outputs, strict=True):
            bits(old.values[0], new.values[0])
            assert not np.shares_memory(old.values[0], new.values[0])
