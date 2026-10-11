"""Large fixed fits: one history, factored BLAS rows, errors and owned results."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_hmm_transition_vforce import check_outputs, compare_models
from .test_hmm_xi_tiles import bits, fit, rounding_mode, sample
from .test_latent_operators import fixture
from .test_nonnegative_matmul_accelerate import ACCELERATE
from .test_typed_results import config

MIN_ELEMENTS = 8 * 1024 * 1024


def compact_doubles(n, k):
    return n * k + 4 * k * k + 4 * k + 512 + ((n - 1) // 128 + 1) * k + 265 * k


def full_doubles(n, k):
    return 2 * n * k + 2 * k * k + 4 * k + 512 + 128 * k


@pytest.mark.parametrize(
    "n,k,layout",
    [(524287, 16, "c"), (524288, 16, "f"), (262145, 32, "negative"), (1048576, 8, "c")],
)
def test_cutoff_replay_canaries_and_two_owned_generations(n, k, layout):
    values, owners = sample(n, 1, k, layout)
    before = [a.tobytes() for a in values + tuple(owners)]
    expected = fit(values, 1, simd="scalar")
    work = op.Workspace()
    storage = np.full(k * (k + 6) + 2, -771.0)
    out = storage[1:-1].reshape(k, k + 6)
    actual, audit = fit(values, 1, workspace=work, out=out, audit=True)
    compare_models(actual, expected)
    compact = k >= 16 and n * k >= MIN_ELEMENTS
    scratch = compact_doubles(n, k) if compact else full_doubles(n, k)
    assert audit["workspace_bytes"] == scratch * 8
    assert bool(audit["accelerate_matmul_calls"]) == (compact and ACCELERATE)
    assert audit["input_copy_bytes"] == 0
    assert actual is out and storage[0] == storage[-1] == -771.0
    first = fit(values, 1, workspace=work)
    saved = first.tobytes()
    second = fit(values, 1, workspace=work)
    assert first.tobytes() == saved
    bits(first, second)
    assert not np.shares_memory(first, second)
    for a, old in zip(values + tuple(owners), before, strict=True):
        assert a.tobytes() == old and not a.flags.writeable
        assert not np.shares_memory(first, a)


def test_positive_tolerance_preserves_full_reservation_and_strict_stop():
    values, _ = sample(524288, 1, 16)
    previous = fit(values, 1, tolerance=1e-5, simd="scalar")
    current = fit(values, 2, tolerance=1e-5, simd="scalar")
    delta = abs(op.em_log_likelihood(current) - op.em_log_likelihood(previous))
    assert np.isfinite(delta) and delta > 0
    for tolerance in (np.nextafter(delta, 0), delta, np.nextafter(delta, np.inf)):
        actual, audit = fit(values, 3, tolerance=tolerance, audit=True)
        scalar = fit(values, 3, tolerance=tolerance, simd="scalar")
        compare_models(actual, scalar)
        assert op.em_iterations(actual) == (2 if tolerance > delta else 3)
        assert audit["workspace_bytes"] == full_doubles(524288, 16) * 8
        assert audit["accelerate_matmul_calls"] == 0


@pytest.mark.parametrize("mode", ["upward", "downward", "towardzero"])
def test_nonnearest_compact_retains_ordered_math_and_environment(mode):
    values, _ = sample(524288, 1, 16)
    with rounding_mode(mode) as (get, selected):
        actual, audit = fit(values, 1, audit=True)
        scalar = fit(values, 1, simd="scalar")
        assert get() == selected
    compare_models(actual, scalar, exact=True)
    assert audit["accelerate_matmul_calls"] == audit["accelerate_vforce_calls"] == 0


@pytest.mark.parametrize(
    "kind", ["symmetric", "zero_probabilities", "range_reject", "unnormalized"]
)
def test_floors_bounds_and_exact_classification(kind):
    n, k = 524288, 16
    values = list(fixture(n, 1, k))
    if kind == "symmetric":
        values[0][:] = values[3][:] = 2.0
        values[4][:] = 1.0
        values[1][:] = 1.0 / k
        values[2][:] = np.eye(k) * 0.9 + (1.0 - np.eye(k)) * (0.1 / (k - 1))
    elif kind == "zero_probabilities":
        values[1][:] = 0.0
        values[1][0] = 1.0
        values[2][:] = np.eye(k)
    elif kind == "range_reject":
        values[3][-1] = 2e4
        values[4][:] = 1.0
    else:
        values[2] *= 2.0
    for a in values:
        a.flags.writeable = False
    actual, audit = fit(values, 1, audit=True)
    scalar = fit(values, 1, simd="scalar")
    compare_models(actual, scalar)
    if kind in ("unnormalized", "range_reject"):
        assert audit["accelerate_matmul_calls"] == 0
    if kind != "range_reject":
        labels = []
        for model in (actual, scalar):
            emission = op.gaussian_log_density(
                values[0], op.hmm_means(model), op.hmm_variances(model)
            )
            posterior = op.hmm_smooth(emission, op.hmm_initial(model), op.hmm_transition(model))
            labels.append(op.argmax_axis(posterior, 1))
        bits(labels[0], labels[1])


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
def test_error_prefix_canaries_and_healthy_recovery(kind, code):
    values = list(fixture(524288, 1, 16))
    if kind == "nan":
        values[0][-128, 0] = np.nan
    elif kind == "variance":
        values[4][0, 0] = -1.0
    elif kind == "weights":
        values[1][:] = 0.0
    elif kind == "transition":
        values[2][0] = 0.0
    elif kind == "density":
        values[0][-128, 0] = np.finfo(float).max
    elif kind == "forward":
        values[0][:] = 2e153
        values[3][:] = 0.0
        values[4][:] = 1.0
    else:
        values[2] = np.ones((16, 17))
    for a in values:
        a.flags.writeable = False
    prefixes, errors = [], []
    for simd in ("scalar", "auto"):
        work = op.Workspace()
        raw = np.full(16 * 22 + 2, 771.0)
        with pytest.raises((ValueError, RuntimeError), match=code) as error:
            fit(values, 1, simd=simd, workspace=work, out=raw[1:-1].reshape(16, 22))
        prefixes.append(raw)
        errors.append((type(error.value), str(error.value)))
        healthy, _ = sample(257, 1, 3)
        compare_models(fit(healthy, 1, simd=simd, workspace=work), fit(healthy, 1, simd="scalar"))
    assert errors[0] == errors[1]
    bits(prefixes[0], prefixes[1])
    assert prefixes[0][0] == prefixes[0][-1] == 771.0


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_graph_projections_transport_and_runtime_tolerance_bound(lane):
    values, _ = sample(524288, 1, 16)
    inputs = dict(zip(("x", "w", "a", "m", "v"), values, strict=True))
    axes = dict(
        x=["time", "factor"],
        w=["component"],
        a=["component", "component"],
        m=["component", "factor"],
        v=["component", "factor"],
    )
    declarations = {
        name: dict(
            kind="matrix" if value.ndim == 2 else "vector",
            shape=[str(z) for z in value.shape],
            axes=axes[name],
        )
        for name, value in inputs.items()
    }
    declarations["tol"] = "scalar"
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
    graph = GraphCompiler(declarations).compile(
        roots,
        root_bindings=[{"trained": "gaussian_hmm_em(x,w,a,m,v,1,tol)"}] * len(roots),
        result_format="typed",
    )
    scalar = fit(values, 1, simd="scalar")
    targets = [
        p(scalar)
        for p in (
            op.hmm_initial,
            op.hmm_transition,
            op.hmm_means,
            op.hmm_variances,
            op.em_iterations,
            op.em_log_likelihood,
            op.em_converged,
        )
    ]
    targets.append(np.asarray(sum(float(z) for z in values[1])))
    starts, ends = np.array([0, 0], np.int64), np.array([524288, 524288], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(graph, inputs, starts, ends, parameters={"tol": 0.0})
        saved = [a.values[0].tobytes() for a in first.outputs]
        check_outputs(first, targets)
        second = scheduler.execute(graph, inputs, starts, ends, parameters={"tol": 1e-5})
        check_outputs(second, targets)
    assert [a.values[0].tobytes() for a in first.outputs] == saved
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    assert second.audit["native_chunks"]
    for a, b in zip(first.outputs, second.outputs, strict=True):
        assert not a.values[0].flags.writeable and not np.shares_memory(a.values[0], b.values[0])


def test_planner_shorter_intervals_and_prepared_snapshot_ownership():
    values, _ = sample(524288, 1, 16)
    inputs = dict(zip(("x", "w", "a", "m", "v"), values, strict=True))
    declarations = {
        "x": {"kind": "matrix", "shape": ["N", "D"], "axes": ["time", "factor"]},
        "w": {"kind": "vector", "shape": ["K"], "axes": ["component"]},
        "a": {"kind": "matrix", "shape": ["K", "K"], "axes": ["component", "component"]},
        "m": {"kind": "matrix", "shape": ["K", "D"], "axes": ["component", "factor"]},
        "v": {"kind": "matrix", "shape": ["K", "D"], "axes": ["component", "factor"]},
        "tol": "scalar",
    }
    graph = GraphCompiler(declarations).compile("hmm_initial(gaussian_hmm_em(x,w,a,m,v,1,tol))")
    ends = np.array([524288, 524287, 4095, 129], np.int64)
    starts = np.zeros(ends.size, np.int64)
    targets = []
    for stop in ends:
        subset = (values[0][:stop], *values[1:])
        targets.append(op.hmm_initial(fit(subset, 1, simd="scalar")))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        session = scheduler.prepare_execution(graph, inputs, starts, ends, parameters={"tol": 0.0})
        first = session.run_snapshot()
        saved = [a.tobytes() for a in first.outputs[0].values]
        second = session.run_snapshot()
        ordinary = scheduler.execute(graph, inputs, starts, ends, parameters={"tol": 0.0})
        for result in (first, second, ordinary):
            for actual, expected in zip(result.outputs[0].values, targets, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0.000049)
                assert not actual.flags.writeable
        assert plan.estimated_worker_scratch_bytes >= max(
            c["operator_workspace_capacity_bytes"] for c in ordinary.audit["native_chunks"]
        )
        assert [a.tobytes() for a in first.outputs[0].values] == saved
        for a, b in zip(first.outputs[0].values, second.outputs[0].values, strict=True):
            assert not np.shares_memory(a, b)
