"""Independent component SIMD keeps ordered densities, EM decisions and ownership."""

import gc
import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from . import latent_reference as ref
from .test_latent_operators import fixture
from .test_matmul_library import settings
from .test_matrix_library_backend import PACKING_PLATFORM


def bits(actual, expected):
    assert np.asarray(actual).dtype == np.asarray(expected).dtype
    assert np.asarray(actual).shape == np.asarray(expected).shape
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def calls(audit, eligible):
    count = audit["eigen_density_calls"]
    if not eligible or not PACKING_PLATFORM or os.environ.get("CME_EXPECT_EIGEN") == "0":
        assert count == 0
    elif os.environ.get("CME_EXPECT_EIGEN") == "1":
        assert count > 0
    return count


def sample(shape=(4096, 32, 8), layout="c", model="c"):
    x, w, a, m, v = fixture(*shape)
    if layout == "f":
        x = np.asfortranarray(x)
    elif layout == "padded":
        owner = np.full((len(x), x.shape[1] + 3), np.nan)
        owner[:, :x.shape[1]] = x
        x = owner[:, :x.shape[1]]
    elif layout == "reverse":
        x = x[::-1]
    elif layout == "broadcast":
        x = np.broadcast_to(x[:1], x.shape)
    elif layout == "feature_stride":
        x = np.repeat(x, 2, axis=1)[:, ::2]
    elif layout == "negative_features":
        x = x[:, ::-1]
    if model == "f":
        m, v = np.asfortranarray(m), np.asfortranarray(v)
    elif model == "reverse_rows":
        m, v = m[::-1], v[::-1]
    elif model == "reverse_features":
        m, v = m[:, ::-1], v[:, ::-1]
    elif model == "padded":
        m, v = np.repeat(m, 2, axis=1)[:, ::2], np.repeat(v, 2, axis=1)[:, ::2]
    elif model == "broadcast":
        m, v = np.broadcast_to(m[:1], m.shape), np.broadcast_to(v[:1], v.shape)
    for value in (x, w, a, m, v):
        value.flags.writeable = False
    return x, w, a, m, v


@pytest.mark.parametrize("shape", [(4095, 32, 8), (4096, 31, 8), (4096, 32, 3),
                                    (4096, 32, 4), (4097, 33, 8), (4096, 32, 12),
                                    (4096, 32, 16)])
@pytest.mark.parametrize("layout", ["c", "padded", "reverse", "broadcast", "feature_stride", "f"])
def test_density_admission_bits_workspace_canaries_and_retained_results(shape, layout):
    x, _, _, m, v = sample(shape, layout)
    before = [value.copy() for value in (x, m, v)]
    expected, scalar = op.gaussian_log_density(x, m, v, simd="scalar", audit=True)
    storage = np.full(expected.size + 2, -889.)
    out = storage[1:-1].reshape(expected.shape)
    work = op.Workspace()
    actual, audit = op.gaussian_log_density(x, m, v, out=out, workspace=work, audit=True)
    assert actual is out
    bits(actual, expected)
    n, d, k = shape
    eligible = ((n >= 4096 and d >= 32 and k % 4 == 0 and layout not in {"f", "feature_stride"})
                or (layout == "f" and n >= 32 and d >= 8 and k <= 16))
    assert calls(audit, eligible) in {0, 1}
    assert scalar["eigen_density_calls"] == 0
    assert audit["workspace_bytes"] == scalar["workspace_bytes"]
    assert audit["algorithm_copy_bytes"] == scalar["algorithm_copy_bytes"]
    assert audit["input_copy_bytes"] == 0
    assert storage[0] == storage[-1] == -889.
    second = op.gaussian_log_density(x, m, v, workspace=work)
    assert not np.shares_memory(second, actual)
    bits(actual, expected)
    for value, original in zip((x, m, v), before, strict=True):
        bits(value, original)


@pytest.mark.parametrize("k", [4, 8, 12, 16])
@pytest.mark.parametrize("model", ["f", "reverse_rows", "reverse_features", "padded", "broadcast"])
def test_strided_models_keep_bits_and_independent_density_target(k, model):
    x, _, _, m, v = sample((4096, 32, k), model=model)
    expected = op.gaussian_log_density(x, m, v, simd="scalar")
    actual, audit = op.gaussian_log_density(x, m, v, audit=True)
    bits(actual, expected)
    calls(audit, True)
    np.testing.assert_allclose(actual, ref._gaussian_log_density_impl(x, m, v), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("row,component", [(0, 0), (0, 3), (63, 4), (4095, 7)])
def test_arithmetic_failure_preserves_entire_output_prefix(row, component):
    x = np.zeros((4096, 32))
    x[row, 0] = 1e154
    m = np.zeros((8, 32))
    v = np.full((8, 32), 1e308)
    v[component, 0] = 1e-12
    outputs = []
    for isa in ("scalar", "auto"):
        out = np.full((4096, 8), 776.)
        with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
            op.gaussian_log_density(x, m, v, out=out, simd=isa)
        outputs.append(out)
    bits(*outputs)
    assert np.all(outputs[0].ravel()[row * 8 + component + 1:] == 776.)


@pytest.mark.parametrize("kind", ["zero_variance", "constant", "subnormal", "large", "nan", "inf"])
def test_floors_special_values_and_preflight(kind):
    x, _, _, m, v = sample()
    x, m, v = x.copy(), m.copy(), v.copy()
    if kind == "zero_variance":
        v[:] = 0.
    elif kind == "constant":
        x[:] = m[:] = 2.
        v[:] = 0.
    elif kind == "subnormal":
        x *= np.nextafter(0., 1.)
        m[:] = 0.
    elif kind == "large":
        x *= 1e140
        m *= 1e140
        v *= 1e280
    else:
        x[-1, -1] = float(kind)
    if kind in {"nan", "inf"}:
        for isa in ("scalar", "auto"):
            out = np.full((4096, 8), 777.)
            with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
                op.gaussian_log_density(x, m, v, out=out, simd=isa)
            assert np.all(out == 777.)
    else:
        bits(op.gaussian_log_density(x, m, v), op.gaussian_log_density(x, m, v, simd="scalar"))


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("k", [4, 8, 12])
def test_em_full_records_and_stopping_neighbors(hmm, k):
    x, w, a, m, v = sample((4096, 32, k))
    args = (x, w, a, m, v) if hmm else (x, w, m, v)
    name = "gaussian_hmm_em" if hmm else "gaussian_mixture_em"
    first = op.call(name, *args, 1, 0., simd="scalar")
    second = op.call(name, *args, 2, 0., simd="scalar")
    difference = abs(second[0, -2] - first[0, -2])
    for tolerance in (0., np.nextafter(difference, 0.), difference, np.nextafter(difference, np.inf)):
        expected = op.call(name, *args, 4, float(tolerance), simd="scalar")
        actual, audit = op.call(name, *args, 4, float(tolerance), audit=True)
        if hmm and audit['accelerate_vforce_calls']:
            # Density/M-step fields and diagnostics remain bitwise ordered.
            # The separately audited xi vForce calls may differ from scalar
            # libm in transition probability tails; retain a tight ULP-scale
            # bound, finite/sign classes and exact stopping/inference decisions.
            bits(actual[:, 0], expected[:, 0])
            bits(actual[:, 1 + k:], expected[:, 1 + k:])
            transition = actual[:, 1:1 + k]
            reference = expected[:, 1:1 + k]
            np.testing.assert_allclose(transition, reference, rtol=0,
                                       atol=16 * np.finfo(np.float64).eps)
            np.testing.assert_array_equal(np.isfinite(transition), np.isfinite(reference))
            np.testing.assert_array_equal(np.signbit(transition), np.signbit(reference))
            labels = []
            for fitted in (actual, expected):
                em = op.gaussian_log_density(x, op.hmm_means(fitted), op.hmm_variances(fitted))
                posterior = op.hmm_smooth(em, op.hmm_initial(fitted), op.hmm_transition(fitted))
                labels.append(op.argmax_axis(posterior, 1))
            bits(labels[0], labels[1])
        else:
            bits(actual, expected)
        count = calls(audit, True)
        assert count in {0, op.em_iterations(actual) * (33 if hmm else 1)}
        assert op.em_iterations(actual) == op.em_iterations(expected)
        assert op.em_converged(actual) == op.em_converged(expected)


TYPES = {
    "x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"]),
    "w": dict(kind="vector", axes=["component"], shape=["K"]),
    "a": dict(kind="matrix", axes=["component", "component"], shape=["K", "K"]),
    "m": dict(kind="matrix", axes=["component", "factor"], shape=["K", "D"]),
    "v": dict(kind="matrix", axes=["component", "factor"], shape=["K", "D"]),
}


def graph_inputs(hmm):
    keys = ("x", "w", "a", "m", "v") if hmm else ("x", "w", "m", "v")
    arrays = sample()
    inputs = {key: value for key, value in zip(("x", "w", "a", "m", "v"), arrays, strict=True) if key in keys}
    name = "gaussian_hmm_em" if hmm else "gaussian_mixture_em"
    expression = f"{name}({','.join(keys)},3,0)"
    fields = (("hmm_initial", "hmm_transition", "hmm_means", "hmm_variances") if hmm
              else ("mixture_weights", "mixture_means", "mixture_variances"))
    fields += ("em_iterations", "em_log_likelihood", "em_converged")
    graph = GraphCompiler({key: TYPES[key] for key in keys}).compile(
        [f"{field}({expression})" for field in fields], result_format="typed")
    model = op.call(name, *(inputs[key] for key in keys), 3, 0., simd="scalar")
    return graph, inputs, [getattr(op, field)(model) for field in fields]


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_graph_cse_audit_transport_and_mixed_result_ownership(hmm, lane):
    graph, inputs, expected = graph_inputs(hmm)
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, inputs, bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    count = sum(chunk["eigen_density_calls"] for chunk in result.audit["native_chunks"])
    if PACKING_PLATFORM and os.environ.get("CME_EXPECT_EIGEN") == "1":
        assert count == 6 * (33 if hmm else 1)  # Full E plus 32 density replay blocks per HMM E step.
    elif not PACKING_PLATFORM or os.environ.get("CME_EXPECT_EIGEN") == "0":
        assert count == 0
    for output, want in zip(result.outputs, expected, strict=True):
        for got in output.values:
            bits(got, want)
        assert np.all(output.root_statuses == 0)
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[0].values[1])


@pytest.mark.parametrize("hmm", [False, True])
def test_prepared_inputs_are_reread_and_shared_owner_can_close(hmm):
    graph, inputs, expected = graph_inputs(hmm)
    inputs["x"] = inputs["x"].copy()
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, bounds, bounds)
        first = prepared.run_snapshot()
        frozen = first.outputs[2].values[0].copy()
        inputs["x"] += 1.
        second = prepared.run_snapshot()
        bits(first.outputs[2].values[0], frozen)
        assert not np.array_equal(second.outputs[2].values[0], frozen)
        assert not np.shares_memory(first.outputs[2].values[0], second.outputs[2].values[0])
        plan = scheduler.plan(graph, inputs, bounds, bounds)
        with pytest.raises(MemoryError):
            scheduler.plan(graph, inputs, bounds, bounds, memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
    graph, inputs, expected = graph_inputs(hmm)
    with SharedInputBundle.from_inputs(inputs) as shared:
        with AdaptiveScheduler(cpu_budget=1, config=settings("process_shared")) as scheduler:
            result = scheduler.execute(graph, shared.arrays, bounds, bounds, hard_stop=True)
    del shared
    gc.collect()
    for output, want in zip(result.outputs, expected, strict=True):
        bits(output.values[0], want)


def test_subscope_audit_and_derived_density_inputs():
    x, _, _, m, v = sample()
    direct, audit = op.gaussian_log_density(x, m, v, audit=True)
    seed = np.zeros_like(direct)
    declarations = {key: TYPES[key] for key in ("x", "m", "v")}
    declarations["seed"] = dict(kind="matrix", axes=["scenario", "component"], shape=["N", "K"])
    graph = GraphCompiler(declarations).compile(
        "iterate(gaussian_log_density(x+0,m,v)+iterate_x*0,seed,1e-12,2)", result_format="typed")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, dict(x=x, m=m, v=v, seed=seed), bounds, bounds)
    count = sum(chunk["eigen_density_calls"] for chunk in result.audit["native_chunks"])
    assert count == 2 * audit["eigen_density_calls"]
    bits(result.outputs[0].values[0], direct + 0)
