"""Column-native Eigen lanes preserve EM state and density failure prefixes."""

import gc
import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_latent_operators import fixture
from .test_matmul_library import settings
from .test_matrix_library_backend import PACKING_PLATFORM


def arguments(n, d, k, hmm, layout="f"):
    x, w, a, m, v = fixture(n, d, k)
    if layout == "f":
        x = np.asfortranarray(x)
    elif layout == "padded":
        holder = np.full((n + 7, d), np.nan, order="F")
        holder[:n] = x
        x = holder[:n]
    elif layout == "reverse":
        x = np.asfortranarray(x)[::-1]
    data = (x, w, a, m, v) if hmm else (x, w, m, v)
    for value in data:
        value.flags.writeable = False
    return data


def bits(a, b):
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


def check_isa(audit, eligible, name):
    if not eligible or not PACKING_PLATFORM or os.environ.get("CME_EXPECT_EIGEN") == "0":
        assert audit["isa"] != name
    elif os.environ.get("CME_EXPECT_EIGEN") == "1":
        assert audit["isa"] == name


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize(
    "shape", [(31, 16, 2), (32, 16, 3), (33, 32, 4), (65, 64, 8), (129, 128, 3), (65, 16, 9)]
)
@pytest.mark.parametrize("layout", ["c", "f", "padded", "reverse"])
def test_em_column_bits_workspace_and_retained_result(hmm, shape, layout):
    name = "gaussian_hmm_em" if hmm else "gaussian_mixture_em"
    args = arguments(*shape, hmm, layout)
    before = [x.copy() for x in args]
    expected, scalar = op.call(name, *args, 5, 0.0, simd="scalar", audit=True)
    work = op.Workspace()
    actual, audit = op.call(name, *args, 5, 0.0, workspace=work, audit=True)
    bits(actual, expected)
    check_isa(audit, shape[0] >= 32 and layout in {"f", "padded"}, "eigen-array-column-em")
    assert audit["workspace_bytes"] == scalar["workspace_bytes"]
    assert audit["algorithm_copy_bytes"] == scalar["algorithm_copy_bytes"]
    assert audit["input_copy_bytes"] == 0
    second = op.call(name, *args, 3, 0.0, workspace=work)
    assert not np.shares_memory(actual, second)
    bits(actual, expected)
    for a, b in zip(args, before, strict=False):
        bits(a, b)


@pytest.mark.parametrize(
    "n,k", [(31, 3), (32, 3), (33, 1), (39, 8), (40, 9), (40, 16), (40, 17), (65, 4)]
)
@pytest.mark.parametrize("layout", ["f", "padded"])
def test_density_row_tails_component_limit_strided_parameters(n, k, layout):
    x, w, m, v = arguments(n, 17, k, False, layout)
    m, v = np.asfortranarray(m), np.asfortranarray(v)
    m.flags.writeable = v.flags.writeable = False
    expected, scalar = op.gaussian_log_density(x, m, v, simd="scalar", audit=True)
    storage = np.full(n * k + 2, -882.0)
    out = storage[1:-1].reshape(n, k)
    actual, audit = op.gaussian_log_density(x, m, v, out=out, audit=True)
    assert actual is out
    bits(actual, expected)
    assert storage[0] == storage[-1] == -882.0
    check_isa(audit, n >= 32 and k <= 16, "eigen-array-column-density")
    assert audit["workspace_bytes"] == scalar["workspace_bytes"]
    assert audit["algorithm_copy_bytes"] == scalar["algorithm_copy_bytes"]
    assert audit["input_copy_bytes"] == 0


@pytest.mark.parametrize("row,component", [(0, 0), (7, 2), (8, 1), (33, 2)])
def test_density_error_preserves_entire_explicit_output_prefix(row, component):
    x = np.zeros((40, 8), order="F")
    x[row, 0] = 1e154
    m = np.zeros((3, 8))
    v = np.full((3, 8), 1e308)
    v[component, 0] = 1e-12
    x.flags.writeable = m.flags.writeable = v.flags.writeable = False
    outputs = []
    for isa in ("scalar", "auto"):
        out = np.full((40, 3), 772.0)
        with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
            op.gaussian_log_density(x, m, v, out=out, simd=isa)
        outputs.append(out)
    bits(*outputs)
    assert np.all(outputs[0].ravel()[row * 3 + component + 1 :] == 772.0)


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("kind", ["zero_weight", "constant", "large", "nan", "inf"])
def test_em_columns_keep_floors_nonfinite_and_zero_posteriors(hmm, kind):
    x, w, a, m, v = fixture(65, 16, 3)
    if kind == "zero_weight":
        w[:] = [1, 0, 0]
        m[1:] = 1000.0
    elif kind == "constant":
        x[:] = m[:] = 2.0
        v[:] = 0.0
    elif kind == "large":
        x *= 1e140
        m *= 1e140
        v *= 1e280
    else:
        x[5, 1] = float(kind)
    x = np.asfortranarray(x)
    args = (x, w, a, m, v) if hmm else (x, w, m, v)
    name = "gaussian_hmm_em" if hmm else "gaussian_mixture_em"
    if kind in {"nan", "inf"}:
        for isa in ("scalar", "auto"):
            with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
                op.call(name, *args, 5, 0.0, simd=isa)
    else:
        bits(op.call(name, *args, 5, 0.0), op.call(name, *args, 5, 0.0, simd="scalar"))


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("d", [16, 33, 64])
def test_em_stopping_neighbors_remain_bitwise_identical(hmm, d):
    args = arguments(65, d, 3, hmm)
    name = "gaussian_hmm_em" if hmm else "gaussian_mixture_em"
    records = [op.call(name, *args, i, 0.0, simd="scalar") for i in (1, 2, 3)]
    for previous, current in zip(records, records[1:], strict=False):
        difference = abs(current[0, -2] - previous[0, -2])
        for tolerance in (
            np.nextafter(difference, 0.0),
            difference,
            np.nextafter(difference, np.inf),
        ):
            expected = op.call(name, *args, 5, float(tolerance), simd="scalar")
            actual = op.call(name, *args, 5, float(tolerance))
            bits(actual, expected)
            assert op.em_iterations(actual) == op.em_iterations(expected)
            assert op.em_converged(actual) == op.em_converged(expected)


def graph_and_inputs(hmm):
    keys = ("x", "w", "a", "m", "v") if hmm else ("x", "w", "m", "v")
    types = {
        "x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"]),
        "w": dict(kind="vector", axes=["component"], shape=["K"]),
        "a": dict(kind="matrix", axes=["component", "component"], shape=["K", "K"]),
        "m": dict(kind="matrix", axes=["component", "factor"], shape=["K", "D"]),
        "v": dict(kind="matrix", axes=["component", "factor"], shape=["K", "D"]),
    }
    args = arguments(65, 16, 3, hmm)
    name = "gaussian_hmm_em" if hmm else "gaussian_mixture_em"
    fields = (
        ("hmm_initial", "hmm_transition", "hmm_means", "hmm_variances")
        if hmm
        else ("mixture_weights", "mixture_means", "mixture_variances")
    )
    fields += ("em_iterations", "em_log_likelihood", "em_converged")
    expression = f"{name}({','.join(keys)},5,0)"
    graph = GraphCompiler({k: types[k] for k in keys}).compile(
        [f"{field}({expression})" for field in fields], result_format="typed"
    )
    model = op.call(name, *args, 5, 0.0, simd="scalar")
    expected = [getattr(op, field)(model) for field in fields]
    return graph, dict(zip(keys, args, strict=False)), expected


@pytest.mark.parametrize("hmm", [False, True])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_em_column_graph_transport_and_result_ownership(hmm, lane):
    graph, inputs, expected = graph_and_inputs(hmm)
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        result = scheduler.execute(graph, inputs, bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    for output, want in zip(result.outputs, expected, strict=False):
        for got in output.values:
            assert np.asarray(got).dtype == np.asarray(want).dtype
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
        assert np.all(output.root_statuses == 0)
    assert not np.shares_memory(result.outputs[0].values[0], result.outputs[0].values[1])


@pytest.mark.parametrize("hmm", [False, True])
def test_em_column_prepared_snapshots_and_shared_owner_close(hmm):
    graph, inputs, expected = graph_and_inputs(hmm)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=settings("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, bounds, bounds)
        first, second = prepared.run_snapshot(), prepared.run_snapshot()
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    with SharedInputBundle.from_inputs(inputs) as shared:
        with AdaptiveScheduler(cpu_budget=1, config=settings("process_shared")) as scheduler:
            result = scheduler.execute(graph, shared.arrays, bounds, bounds, hard_stop=True)
    del shared
    gc.collect()
    for field, want in zip(result.outputs, expected, strict=False):
        assert np.asarray(field.values[0]).tobytes() == np.asarray(want).tobytes()
