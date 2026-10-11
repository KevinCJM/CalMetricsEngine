"""Ordered affine/expm1 primitives and an explicit fixed-tape projection DAG.

This is a mathematical subchain; seed generation, NAV/fan and the platform's
ordered first-fault prefix protocol are deliberately outside these tests.
"""

import gc
import math
from contextlib import nullcontext

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompileError, GraphCompiler, PlannerConfig
from calmetrics_engine import operators as op


def ordered_affine(x, coefficients, initial):
    output = np.empty(x.shape[:-1] + (coefficients.shape[1],))
    with np.errstate(all="ignore"):
        for outer in np.ndindex(x.shape[:-1]):
            for asset in range(coefficients.shape[1]):
                value = np.float64(initial if np.ndim(initial) == 0 else initial[asset])
                for factor in range(x.shape[-1]):
                    product = coefficients[factor, asset] * x[outer + (factor,)]
                    value = np.float64(value + product)
                output[outer + (asset,)] = value
    return output


def stable_expm1(x):
    output = np.empty(np.shape(x))
    for coordinate in np.ndindex(output.shape):
        try:
            output[coordinate] = math.expm1(float(np.asarray(x)[coordinate]))
        except OverflowError:
            output[coordinate] = np.inf
    return output


def ieee_equal(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))


def layout_view(value, layout):
    if layout == "f":
        value = np.array(value, order="F")
    elif layout == "strided":
        owner = np.full(tuple(2 * size for size in value.shape), -991.0)
        slices = (slice(None, None, 2),) * value.ndim
        owner[slices] = value
        value = owner[slices]
    elif layout == "negative":
        reverse = (slice(None, None, -1),) * value.ndim
        value = value[reverse].copy()[reverse]
    value.flags.writeable = False
    return value


@pytest.mark.parametrize("shape,assets", [
    ((5, 4), 3), ((2, 3, 4), 3), ((0, 4), 3), ((2, 0, 4), 3),
    ((3, 0), 3), ((2, 3, 0), 3), ((2, 3, 4), 0),
])
@pytest.mark.parametrize("layout", ["c", "f", "strided", "negative"])
@pytest.mark.parametrize("scalar_initial", [False, True])
def test_affine_direct_strides_empty_dimensions_and_ownership(shape, assets, layout, scalar_initial):
    x = layout_view((np.arange(np.prod(shape)).reshape(shape) - 7.0) / 8, layout)
    coefficients = layout_view((np.arange(shape[-1] * assets).reshape(shape[-1], assets) - 3.0) / 16, layout)
    initial = -0.0 if scalar_initial else layout_view(np.arange(assets, dtype=np.float64) / 8, layout)
    before = x.copy(), coefficients.copy(), np.asarray(initial).copy()
    expected = ordered_affine(x, coefficients, initial)
    work = op.Workspace()
    first, audit = op.affine_contract(x, coefficients, initial, workspace=work, audit=True)
    saved = first.copy()
    storage = np.full(expected.size + 2, 882.0)
    out = storage[1:-1].reshape(expected.shape)
    second = op.affine_contract(x, coefficients, initial, out=out, workspace=work, simd="scalar")
    assert second is out
    ieee_equal(first, expected)
    ieee_equal(second, expected)
    ieee_equal(first, saved)
    assert storage[0] == storage[-1] == 882.0
    assert audit["input_copy_bytes"] == 0
    for value, original in zip((x, coefficients, np.asarray(initial)), before, strict=True):
        ieee_equal(value, original)
        assert not np.shares_memory(first, value)
    assert not np.shares_memory(first, second)


def test_affine_initial_first_order_and_ieee_products_are_observable():
    x = np.array([[1e16, -1e16], [np.inf, 2.0], [-0.0, -0.0]])
    coefficients = np.array([[1.0, 0.0], [1.0, 0.0]])
    initial = np.array([1.0, -0.0])
    expected = ordered_affine(x, coefficients, initial)
    actual = op.affine_contract(x, coefficients, initial)
    ieee_equal(actual, expected)
    assert actual[0, 0] == 0.0  # matmul followed by +initial would produce 1.
    assert np.isnan(actual[1, 1])  # 0*Inf must be evaluated, never omitted.
    empty = op.affine_contract(np.empty((3, 0)), np.empty((0, 2)), -0.0)
    assert np.signbit(empty).all()


@pytest.mark.parametrize("shape", [(), (12,), (3, 4), (2, 2, 3), (0,), (2, 0, 3)])
@pytest.mark.parametrize("layout", ["c", "f", "strided", "negative"])
def test_expm1_direct_ieee_precision_and_independent_results(shape, layout):
    values = np.array([-0.0, 0.0, -1e-20, 1e-20, -745., 709., 710., np.nan,
                       np.inf, -np.inf, -1., 1.])
    x = layout_view(np.resize(values, np.prod(shape, dtype=np.int64)).reshape(shape), layout) if shape else np.array(-0.0)
    x.flags.writeable = False
    expected = stable_expm1(x)
    actual, audit = op.expm1(x, audit=True)
    ieee_equal(actual, expected)
    before = np.asarray(actual).copy()
    op.expm1(np.zeros(shape))
    ieee_equal(actual, before)
    assert audit["input_copy_bytes"] == 0
    assert not np.shares_memory(np.asarray(actual), x)
    if shape:
        storage = np.full(expected.size + 2, 981.0)
        out = storage[1:-1].reshape(shape)
        assert op.expm1(x, out=out, simd="scalar") is out
        ieee_equal(out, expected)
        assert storage[0] == storage[-1] == 981.0


def test_existing_exp_domain_contract_is_preserved():
    assert op.expm1(1e-20) != op.exp(1e-20) - 1.0
    assert np.signbit(op.expm1(-0.0))
    assert np.isposinf(op.expm1(710.0))
    for value in (710.0, np.inf, np.nan):
        with pytest.raises((ValueError, RuntimeError), match="DOMAIN_ERROR"):
            op.exp(value)


@pytest.mark.parametrize("x,coefficients,initial,code", [
    (np.ones(4), np.ones((4, 2)), 0., "RANK"),
    (np.ones((2, 4)), np.ones(4), 0., "RANK"),
    (np.ones((2, 4)), np.ones((3, 2)), 0., "SHAPE"),
    (np.ones((2, 4)), np.ones((4, 2)), np.ones(3), "SHAPE"),
    (np.ones((2, 4)), np.ones((4, 2)), np.ones((1, 2)), "RANK"),
    (np.ones((2, 4), dtype=np.float32), np.ones((4, 2)), 0., "float64|dtype|DTYPE"),
])
def test_invalid_affine_geometry_fails_before_output_write(x, coefficients, initial, code):
    out = np.full((2, 2), 731.0)
    with pytest.raises((ValueError, TypeError, RuntimeError), match=code):
        op.affine_contract(x, coefficients, initial, out=out)
    np.testing.assert_array_equal(out, np.full((2, 2), 731.0))


def declarations(rank=3):
    outer = ["time", "path"] if rank == 3 else ["time"]
    symbols = ["T", "P"] if rank == 3 else ["T"]
    return {
        "x": dict(kind="tensor" if rank == 3 else "matrix", axes=outer + ["factor"], shape=symbols + ["F"]),
        "beta": dict(kind="matrix", axes=["asset", "factor"], shape=["A", "F"]),
        "initial": dict(kind="vector", axes=["asset"], shape=["A"]),
        "w0": dict(kind="vector", axes=["asset"], shape=["A"]),
        "w1": dict(kind="vector", axes=["asset"], shape=["A"]),
    }


def projection_expressions(assets, paths):
    simple = "expm1(affine_contract(x,transpose(beta),initial))"
    expressions = {}
    for combination in range(2):
        weighted = f"multiply(broadcast_like(w{combination},{simple}),{simple})"
        expressions[f"period_{combination}"] = f"swap_axes(sum_axis({weighted},2),0,1)"
        for asset in range(assets):
            weight = f"first(slice_axis(w{combination},0,{asset},{asset + 1},1))"
            value = f"slice_axis({simple},2,{asset},{asset + 1},1)"
            expressions[f"contribution_{combination}_{asset}"] = f"sum(multiply({weight},{value}))/{paths}"
    return expressions


def projection_expected(x, beta, initial, weights):
    simple = stable_expm1(ordered_affine(x, beta.T, initial))
    results = []
    for weight in weights:
        period = np.zeros((x.shape[1], x.shape[0]))
        contributions = np.zeros(beta.shape[0])
        with np.errstate(all="ignore"):
            for step in range(x.shape[0]):
                for path in range(x.shape[1]):
                    for asset in range(beta.shape[0]):
                        value = weight[asset] * simple[step, path, asset]
                        period[path, step] += value
                        contributions[asset] += value
            contributions /= x.shape[1]
        results.append(period)
        results.extend(np.asarray(value) for value in contributions)
    return results


@pytest.mark.parametrize("layout", ["c", "f", "strided", "negative"])
@pytest.mark.parametrize("mode", ["ordinary", "prepared"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_shared_multiroot_projection_ast_typed_graph_and_result_lifetime(layout, mode, order):
    x = layout_view((np.arange(24.).reshape(2, 3, 4) - 11.) / 128, layout)
    beta = layout_view((np.arange(12.).reshape(3, 4) - 6.) / 16, layout)
    initial = layout_view(np.array([0., 1e-20, -1e-20]), layout)
    weights = [layout_view(np.array([0.2, -0.1, 0.9]), layout), layout_view(np.array([1., 0., -0.5]), layout)]
    inputs = dict(x=x, beta=beta, initial=initial, w0=weights[0], w1=weights[1])
    expressions = projection_expressions(3, 3)
    graph = GraphCompiler(declarations()).compile(expressions, output_order=order, result_format="typed")
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    expected = projection_expected(x, beta, initial, weights)
    bounds, ends = np.array([0], np.int64), np.array([2], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if mode == "prepared":
            prepared = scheduler.prepare_execution(graph, inputs, bounds, ends)
            call = prepared.run_snapshot
        else:
            def call(data=inputs):
                return scheduler.execute(graph, data, bounds, ends)
        first = call()
        saved = [output.values[0].copy() for output in first.outputs]
        later = call()
        for output, target, snapshot, new in zip(first.outputs, expected, saved, later.outputs, strict=True):
            actual = output.values[0]
            # std::expm1 and NumPy/libm may differ in a last bit; sum order and
            # zero/non-finite/ownership contracts are checked separately below.
            np.testing.assert_allclose(actual, target, rtol=2e-14, atol=2e-16, equal_nan=True)
            ieee_equal(actual, snapshot)
            assert not actual.flags.writeable
            assert not np.shares_memory(actual, new.values[0])
            assert all(not np.shares_memory(actual, value) for value in inputs.values())
            np.testing.assert_array_equal(output.root_statuses, [0])
            if actual.ndim == 2:
                assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous
        assert first.audit["result_lifetime"] == "independent"
        assert first.audit["python_fallback"] == first.audit["python_operator_calls"] == 0
        assert first.audit["request_time_compilation"] == 0
    del call, inputs, x, beta, initial, weights
    gc.collect()
    for output, target in zip(first.outputs, saved, strict=True):
        ieee_equal(output.values[0], target)


def test_graph_contribution_uses_step_path_order_without_subtotal_and_zero_weight_inf():
    # Weighting these returns yields [large, small, -large, small]. Sequential
    # step/path differs from per-step subtotals, so this catches regrouping.
    x = np.array([[[math.log1p(1.)], [1e-16]], [[-40.], [1e-16]]])
    beta, initial = np.ones((1, 1)), np.zeros(1)
    inputs = dict(x=x, beta=beta, initial=initial, w0=np.full(1, 1e16), w1=np.zeros(1))
    graph = GraphCompiler(declarations()).compile(projection_expressions(1, 2), result_format="typed")
    bounds, ends = np.array([0], np.int64), np.array([2], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, bounds, ends)
        expected = projection_expected(x, beta, initial, [inputs["w0"], inputs["w1"]])
        for output, target in zip(result.outputs, expected, strict=True):
            ieee_equal(output.values[0], target)
        simple = stable_expm1(ordered_affine(x, beta.T, initial))
        grouped = 0.0
        for step in range(x.shape[0]):
            subtotal = 0.0
            for path in range(x.shape[1]):
                subtotal += inputs["w0"][0] * simple[step, path, 0]
            grouped += subtotal
        assert result.outputs[1].values[0] != grouped / x.shape[1]
        inputs["x"] = np.full((2, 2, 1), 710.)
        failed_numeric = scheduler.execute(graph, inputs, bounds, ends)
    # IEEE propagation is a value contract of these primitives, not the full
    # MC service's ordered non-finite prefix/error protocol.
    assert np.isposinf(failed_numeric.outputs[0].values[0]).all()
    assert np.isnan(failed_numeric.outputs[2].values[0]).all()
    assert np.isnan(failed_numeric.outputs[3].values[0])


@pytest.mark.parametrize("rank", [2, 3])
def test_affine_graph_axes_inference_and_independent_shape_rejection(rank):
    declarations_good = {key: value for key, value in declarations(rank).items() if key in ("x", "beta", "initial")}
    graph = GraphCompiler(declarations_good).compile("affine_contract(x,transpose(beta),initial)", result_format="typed")
    shape = (2, 3, 4) if rank == 3 else (2, 4)
    x, beta, initial = np.ones(shape), np.ones((3, 4)), np.zeros(3)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, dict(x=x, beta=beta, initial=initial), np.array([0], np.int64), np.array([2], np.int64))
        assert result.outputs[0].axes == (["time", "path", "asset"] if rank == 3 else ["time", "asset"])
        with pytest.raises((ValueError, RuntimeError), match="symbolic dimensions|SHAPE"):
            scheduler.execute(graph, dict(x=x, beta=np.ones((3, 5)), initial=initial), np.array([0], np.int64), np.array([2], np.int64))
    wrong_axes = declarations(rank)
    wrong_axes["beta"]["axes"] = ["asset", "other_factor"]
    with pytest.raises(GraphCompileError, match="AXIS|TYPE"):
        GraphCompiler(wrong_axes).compile("affine_contract(x,transpose(beta),initial)")
    wrong_initial = declarations(rank)
    wrong_initial["initial"]["axes"] = ["other_asset"]
    with pytest.raises(GraphCompileError, match="AXIS|TYPE"):
        GraphCompiler(wrong_initial).compile("affine_contract(x,transpose(beta),initial)")


def test_affine_and_expm1_out_cannot_alias_inputs():
    x, coefficients = np.ones((3, 2)), np.eye(2)
    before = x.copy()
    with pytest.raises((ValueError, RuntimeError), match="ALIAS|overlap"):
        op.affine_contract(x, coefficients, 0., out=x)
    with pytest.raises((ValueError, RuntimeError), match="ALIAS|overlap"):
        op.expm1(x, out=x)
    ieee_equal(x, before)


@pytest.mark.parametrize("selection", ["period_0", "contribution_0_1", "intermediates", "isolate"])
def test_projection_closed_consumers_and_fallback_roots_preserve_values(selection):
    x = np.arange(24.).reshape(2, 3, 4) / 128
    beta, initial = np.ones((3, 4)), np.zeros(3)
    inputs = dict(x=x, beta=beta, initial=initial, w0=np.array([.2, .3, .5]), w1=np.ones(3))
    expressions = projection_expressions(3, 3)
    all_expected = dict(zip(expressions, projection_expected(x, beta, initial, [inputs["w0"], inputs["w1"]]), strict=True))
    if selection == "intermediates":
        expressions["linear"] = "affine_contract(x,transpose(beta),initial)"
        expressions["simple"] = "expm1(affine_contract(x,transpose(beta),initial))"
        all_expected["linear"] = ordered_affine(x, beta.T, initial)
        all_expected["simple"] = stable_expm1(all_expected["linear"])
    elif selection != "isolate":
        expressions = {selection: expressions[selection]}
    graph = GraphCompiler(declarations()).compile(expressions, error_policy="isolate" if selection == "isolate" else "raise", result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        used = {name: inputs[name] for name in graph.metadata()["input_names"]}
        first = scheduler.execute(graph, used, np.array([0], np.int64), np.array([2], np.int64))
        for name, output in zip(expressions, first.outputs, strict=True):
            np.testing.assert_allclose(output.values[0], all_expected[name], rtol=2e-14, atol=2e-16)


def test_prepared_projection_reads_current_content_and_rejects_changed_geometry():
    x = np.zeros((2, 3, 2))
    initial = np.zeros(2)
    inputs = dict(x=x, beta=np.eye(2), initial=initial, w0=np.array([.2, .8]), w1=np.ones(2))
    graph = GraphCompiler(declarations()).compile(projection_expressions(2, 3), result_format="typed")
    bounds, ends = np.array([0], np.int64), np.array([2], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, inputs, bounds, ends)
        first = prepared.run_snapshot()
        x[:] = .1
        initial[:] = .2
        current = prepared.run_snapshot()
        assert np.all(first.outputs[0].values[0] == 0.)
        np.testing.assert_allclose(current.outputs[0].values[0], math.expm1(.3), rtol=2e-14)
        address, size = initial.ctypes.data, initial.nbytes
        expected_warning = (
            pytest.warns(DeprecationWarning, match=r"^Setting the shape on a NumPy array has been deprecated")
            if np.lib.NumpyVersion(np.__version__) >= "2.5.0.dev0" else nullcontext()
        )
        with expected_warning:
            initial.shape = (1, 2)
        assert initial.ctypes.data == address and initial.nbytes == size
        with pytest.raises(ValueError, match="PREPARED_INPUT_CHANGED"):
            prepared.run_snapshot()


def test_projection_isolated_invalid_denominator_keeps_healthy_roots():
    inputs = dict(x=np.zeros((2, 3, 2)), beta=np.eye(2), initial=np.zeros(2),
                  w0=np.ones(2), w1=np.ones(2))
    expressions = projection_expressions(2, 3)
    expressions["bad_contribution"] = expressions["contribution_0_0"].rsplit("/", 1)[0] + "/denominator"
    types = declarations()
    types["denominator"] = dict(kind="scalar")
    graph = GraphCompiler(types).compile(expressions, error_policy="isolate", result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([2], np.int64), parameters={"denominator": 0.0})
    for output in result.outputs[:-1]:
        assert not np.any(output.root_statuses)
        assert np.all(output.values[0] == 0.)
    assert result.outputs[-1].root_statuses[0] != 0


@pytest.mark.parametrize("name", ["affine_contract", "expm1"])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("mode", ["ordinary", "prepared"])
@pytest.mark.parametrize("layout", ["c", "f"])
def test_independent_new_operator_f_root_writer_preserves_ownership(name, rank, mode, layout):
    shape = (3, 4) if rank == 2 else (3, 2, 4)
    x = layout_view((np.arange(np.prod(shape)).reshape(shape) - 9.) / 128, layout)
    if name == "affine_contract":
        beta = layout_view((np.arange(12.).reshape(3, 4) - 4.) / 8, layout)
        initial = layout_view(np.array([0., 1e-20, -0.0]), layout)
        inputs = dict(x=x, beta=beta, initial=initial)
        types = {key: value for key, value in declarations(rank).items() if key in inputs}
        expression = "affine_contract(x,transpose(beta),initial)"
        expected = ordered_affine(x, beta.T, initial)
    else:
        inputs = dict(x=x)
        types = {"x": declarations(rank)["x"]}
        expression, expected = "expm1(x)", op.expm1(x, simd="scalar")
        # Python and C++ may link different libm implementations on Windows.
        # Check the independent math reference with the existing projection
        # tolerance; the F writer must still match the canonical operator bits.
        np.testing.assert_allclose(expected, stable_expm1(x), rtol=2e-14, atol=2e-16)
    graph = GraphCompiler(types).compile({"value": expression}, result_format="typed", output_order="F")
    originals = {key: value.copy() for key, value in inputs.items()}
    starts, ends = np.array([0], np.int64), np.array([3], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if mode == "prepared":
            prepared = scheduler.prepare_execution(graph, inputs, starts, ends)
            call = prepared.run_snapshot
        else:
            def call(data=inputs):
                return scheduler.execute(graph, data, starts, ends)
        first = call()
        actual = first.outputs[0].values[0]
        saved = actual.copy()
        later = call()
        assert actual.flags.f_contiguous and not actual.flags.writeable
        assert first.outputs[0].order == "F"
        ieee_equal(actual, expected)
        ieee_equal(actual, saved)
        assert not np.shares_memory(actual, later.outputs[0].values[0])
        assert all(not np.shares_memory(actual, value) for value in inputs.values())
        for key, value in inputs.items():
            ieee_equal(value, originals[key])
        assert first.audit["result_lifetime"] == "independent"
    del call, inputs, x
    gc.collect()
    ieee_equal(actual, saved)


@pytest.mark.parametrize("name", ["affine_contract", "expm1"])
def test_direct_new_operator_f_out_remains_outside_c_only_public_contract(name):
    x = np.ones((3, 2))
    out = np.full((3, 2), 651., order="F")
    arguments = (x, np.eye(2), 0.) if name == "affine_contract" else (x,)
    with pytest.raises((ValueError, TypeError), match="contiguous|C"):
        getattr(op, name)(*arguments, out=out)
    np.testing.assert_array_equal(out, np.full((3, 2), 651.))


def projection_fixture(steps=16, paths=5, factors=2, assets=3, layout="c"):
    """One source per dependency; views preserve logical coordinates."""
    x = (np.arange(steps * paths * factors, dtype=np.float64).reshape(steps, paths, factors) % 31 - 15) / 256
    beta = (np.arange(assets * factors, dtype=np.float64).reshape(assets, factors) - 2) / 16
    values = dict(x=x, beta=beta, initial=np.arange(assets, dtype=np.float64) / 128,
                  w0=(np.arange(assets, dtype=np.float64) + 1) / 8,
                  w1=(np.arange(assets, dtype=np.float64) - 1) / 4)
    return {name: layout_view(value, layout) for name, value in values.items()}


def selected_inputs(graph, inputs):
    return {name: inputs[name] for name in graph.metadata()["input_names"]}


def chunk_sum(result, field):
    return sum(chunk[field] for chunk in result.audit["native_chunks"])


def assert_same_outputs(actual, reference):
    assert len(actual.outputs) == len(reference.outputs)
    for got, want in zip(actual.outputs, reference.outputs, strict=True):
        assert got.axes == want.axes and got.dtype == want.dtype
        np.testing.assert_array_equal(got.root_statuses, want.root_statuses)
        for value, expected in zip(got.values, want.values, strict=True):
            ieee_equal(value, expected)


@pytest.mark.parametrize("layout", ["c", "f", "strided", "negative"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("mode", ["ordinary", "prepared"])
def test_private_projection_tiles_share_all_period_and_contribution_roots(layout, order, mode):
    # More than two tile capacities: the contribution left fold must span tile
    # boundaries without a subtotal, and no full [T,P,A] arena is permitted.
    inputs = projection_fixture(steps=513, paths=17, layout=layout)
    expressions = projection_expressions(3, 17)
    graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", output_order=order)
    reference_graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", output_order=order, error_policy="isolate")
    starts, ends = np.array([0], np.int64), np.array([513], np.int64)
    originals = {name: value.copy() for name, value in inputs.items()}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        reference = scheduler.execute(reference_graph, inputs, starts, ends)
        if mode == "prepared":
            prepared = scheduler.prepare_execution(graph, inputs, starts, ends)
            call = prepared.run_snapshot
        else:
            def call():
                return scheduler.execute(graph, inputs, starts, ends)
        first, later = call(), call()
        assert_same_outputs(first, reference)
        assert_same_outputs(later, reference)
        assert chunk_sum(first, "fused_pointwise_calls") == 1
        assert chunk_sum(first, "pointwise_tiles") == 3
        assert chunk_sum(first, "numeric_arena_bytes") == 0
        assert chunk_sum(first, "root_copy_bytes") == 0
        assert chunk_sum(first, "direct_output_bytes") == sum(output.values[0].nbytes for output in first.outputs)
        # This is a workspace/Planner check, not a measured heap/RSS assertion.
        full_returns_bytes = 513 * 17 * 3 * 8
        assert first.plan.estimated_worker_scratch_bytes < full_returns_bytes
        for chunk in first.audit["native_chunks"]:
            assert chunk["operator_workspace_capacity_bytes"] <= first.plan.estimated_worker_scratch_bytes
        for output, new in zip(first.outputs, later.outputs, strict=True):
            assert not np.shares_memory(output.values[0], new.values[0])
            assert all(not np.shares_memory(output.values[0], value) for value in inputs.values())
            assert not output.values[0].flags.writeable
            if output.values[0].ndim == 2:
                assert output.order == order
                assert output.values[0].flags.f_contiguous if order == "F" else output.values[0].flags.c_contiguous
        for name, original in originals.items():
            ieee_equal(inputs[name], original)


@pytest.mark.parametrize("selection", ["period", "contribution", "mixed", "public_returns", "independent", "isolate"])
def test_private_projection_admission_uses_exact_selected_consumer_closure(selection):
    inputs = projection_fixture()
    expressions = projection_expressions(3, 5)
    if selection == "period":
        expressions = {"period_0": expressions["period_0"]}
    elif selection == "contribution":
        expressions = {"contribution_0_1": expressions["contribution_0_1"]}
    elif selection == "mixed":
        expressions = {name: expressions[name] for name in ("period_1", "contribution_0_2", "contribution_1_0")}
    elif selection == "public_returns":
        expressions["returns"] = "expm1(affine_contract(x,transpose(beta),initial))"
    elif selection == "independent":
        expressions["independent"] = "mean(x)"
    policy = "isolate" if selection == "isolate" else "raise"
    graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", error_policy=policy)
    reference_graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", error_policy="isolate")
    starts, ends = np.array([0], np.int64), np.array([16], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, selected_inputs(graph, inputs), starts, ends)
        reference = scheduler.execute(reference_graph, selected_inputs(reference_graph, inputs), starts, ends)
    assert_same_outputs(result, reference)
    if selection in ("period", "contribution", "mixed"):
        assert chunk_sum(result, "fused_pointwise_calls") == 1
        assert chunk_sum(result, "pointwise_tiles") == 1
        assert chunk_sum(result, "numeric_arena_bytes") == 0
    else:
        # Publishing R, adding an unrelated operator root or requesting isolate
        # makes the whole closed region use the existing canonical graph path.
        assert chunk_sum(result, "pointwise_tiles") == 0
        assert chunk_sum(result, "numeric_arena_bytes") > 0
    if selection == "public_returns":
        actual = result.named_outputs["returns"].values[0]
        np.testing.assert_allclose(actual, stable_expm1(ordered_affine(inputs["x"], inputs["beta"].T, inputs["initial"])), rtol=2e-14, atol=2e-16)


@pytest.mark.parametrize("steps,paths,factors,assets", [(0, 3, 2, 3), (3, 0, 2, 3), (3, 4, 0, 3), (3, 4, 2, 0)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_private_projection_empty_dimensions_keep_period_geometry(steps, paths, factors, assets, order):
    inputs = projection_fixture(steps, paths, factors, assets)
    expressions = projection_expressions(assets, paths)
    if steps == 0 or paths == 0:
        # sum(empty) is not a valid scalar contributor. Period-only roots still
        # have known empty geometry, and must not form invalid input offsets.
        expressions = {name: value for name, value in expressions.items() if name.startswith("period")}
    graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", output_order=order)
    reference_graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", output_order=order, error_policy="isolate")
    starts, ends = np.array([0], np.int64), np.array([steps], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
        reference = scheduler.execute(reference_graph, inputs, starts, ends)
    assert_same_outputs(result, reference)
    assert chunk_sum(result, "numeric_arena_bytes") == 0
    for name in ("period_0", "period_1"):
        output = result.named_outputs[name]
        assert output.values[0].shape == (paths, steps)
        np.testing.assert_array_equal(output.root_statuses, [0])
        if assets == 0:
            np.testing.assert_array_equal(output.values[0], np.zeros((paths, steps)))


def test_empty_projection_contribution_fails_without_poisoning_isolated_period():
    inputs = projection_fixture(steps=0, paths=3, assets=1)
    expressions = projection_expressions(1, 3)
    starts = ends = np.array([0], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        strict = GraphCompiler(declarations()).compile(expressions, result_format="typed")
        with pytest.raises((ValueError, RuntimeError), match="INSUFFICIENT_SAMPLE"):
            scheduler.execute(strict, inputs, starts, ends)
        isolated = GraphCompiler(declarations()).compile(expressions, result_format="typed", error_policy="isolate")
        result = scheduler.execute(isolated, inputs, starts, ends)
    for name, output in result.named_outputs.items():
        if name.startswith("period"):
            assert output.values[0].shape == (3, 0)
            np.testing.assert_array_equal(output.root_statuses, [0])
        else:
            assert output.root_statuses[0] != 0


@pytest.mark.parametrize("layout", ["c", "f", "negative"])
def test_private_projection_preserves_extreme_ieee_values_and_original_fold(layout):
    x = np.array([[[1e16, -1e16], [np.inf, 0.]],
                  [[-0., -0.], [1e-20, -1e-20]],
                  [[710., 0.], [-np.inf, 0.]],
                  [[np.nan, 0.], [-745., 0.]]])
    values = dict(x=x, beta=np.array([[1., 1.], [0., 0.], [1., 0.]]),
                  initial=np.array([1., -0., 0.]), w0=np.array([1., 0., -1.]),
                  w1=np.array([0., -1., 1.]))
    inputs = {name: layout_view(value, layout) for name, value in values.items()}
    expressions = projection_expressions(3, 2)
    graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", output_order="F")
    # Keep the same strict error policy: isolate intentionally marks a
    # non-finite scalar root as status 4. Publishing R instead forces the
    # canonical path while preserving the original selected expressions.
    reference_expressions = dict(expressions)
    reference_expressions["audit_returns"] = "expm1(affine_contract(x,transpose(beta),initial))"
    reference_graph = GraphCompiler(declarations()).compile(reference_expressions, result_format="typed", output_order="F")
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
        reference = scheduler.execute(reference_graph, inputs, starts, ends)
    for name, output in result.named_outputs.items():
        target = reference.named_outputs[name]
        assert output.axes == target.axes and output.dtype == target.dtype
        np.testing.assert_array_equal(output.root_statuses, target.root_statuses)
        ieee_equal(output.values[0], target.values[0])
    assert chunk_sum(reference, "pointwise_tiles") == 0
    assert chunk_sum(reference, "numeric_arena_bytes") > 0
    assert chunk_sum(result, "fused_pointwise_calls") == 1
    assert all(not np.any(output.root_statuses) for output in result.outputs)
    assert np.isnan(result.named_outputs["period_0"].values[0]).any()
    # This compares classifications and finite bits; it does not promise NaN
    # payload identity or the platform's first-fault prefix/status protocol.


def test_projection_isolate_keeps_existing_nonfinite_typed_policy_and_healthy_root():
    inputs = dict(x=np.full((2, 2, 1), 710.), beta=np.ones((1, 1)),
                  initial=np.zeros(1), w0=np.ones(1), w1=np.zeros(1))
    expressions = projection_expressions(1, 2)
    expressions["healthy"] = "mean(x)"
    graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([2], np.int64))
    assert np.isposinf(result.named_outputs["period_0"].values[0]).all()
    assert np.isnan(result.named_outputs["period_1"].values[0]).all()
    # The existing typed publication loop marks every non-finite float64
    # element status 4 and aggregates it into root_statuses (graph.cpp). It
    # keeps array IEEE payloads; a failed scalar ancestor publishes NaN.
    for name in ("period_0", "period_1", "contribution_0_0", "contribution_1_0"):
        np.testing.assert_array_equal(result.named_outputs[name].root_statuses, [4])
        np.testing.assert_array_equal(result.named_outputs[name].statuses[0], np.full(result.named_outputs[name].values[0].shape, 4, np.int16))
    np.testing.assert_array_equal(result.named_outputs["healthy"].root_statuses, [0])
    assert result.named_outputs["healthy"].values[0] == 710.
    assert chunk_sum(result, "pointwise_tiles") == 0


def projection_lane_configuration(lane):
    return PlannerConfig(thread_work_units=1 if lane == "thread" else 1e100,
                         process_work_units=1 if lane.startswith("process") else 1e100,
                         shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
                         min_rows_per_worker=1, max_processes=2)


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_private_projection_existing_scheduler_lanes_transport_complete_outputs(lane, order):
    inputs = projection_fixture(steps=18, paths=5, layout="negative")
    expressions = projection_expressions(3, 5)
    graph = GraphCompiler(declarations()).compile(expressions, result_format="typed", output_order=order)
    starts, ends = np.array([0, 6], np.int64), np.array([6, 18], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as single:
        reference = single.execute(graph, inputs, starts, ends)
    with AdaptiveScheduler(cpu_budget=2, config=projection_lane_configuration(lane)) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
        later = scheduler.execute(graph, inputs, starts, ends)
        assert result.plan.lane == ("process" if lane.startswith("process") else "thread")
        assert result.audit["cpu_tokens"] <= 2
        assert result.audit["python_worker_callbacks"] == 0
        assert_same_outputs(result, reference)
        assert_same_outputs(later, reference)
        assert chunk_sum(result, "numeric_arena_bytes") == 0
        assert chunk_sum(result, "fused_pointwise_calls") == 2
        if lane == "process_shared":
            assert result.plan.use_shared_memory
            assert result.audit["output_copy_bytes"] == 0
        for old, new in zip(result.outputs, later.outputs, strict=True):
            for old_value, new_value in zip(old.values, new.values, strict=True):
                assert not np.shares_memory(old_value, new_value)
                assert not old_value.flags.writeable
                assert all(not np.shares_memory(old_value, value) for value in inputs.values())
                if old_value.ndim == 2:
                    assert old_value.flags.f_contiguous if order == "F" else old_value.flags.c_contiguous
    del inputs, later
    gc.collect()
    assert_same_outputs(result, reference)


def test_private_projection_planner_admits_bounded_workspace_and_checks_total_budget():
    graph = GraphCompiler(declarations()).compile(projection_expressions(3, 17), result_format="typed")
    starts = np.array([0], np.int64)
    large = projection_fixture(steps=513, paths=17)
    larger = projection_fixture(steps=1026, paths=17)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first_plan = scheduler.plan(graph, large, starts, np.array([513], np.int64))
        plan = scheduler.plan(graph, larger, starts, np.array([1026], np.int64))
        # Same bounded tile, graph and contributor state; only independent
        # input/result byte requirements grow. Planner is not a heap profiler.
        assert plan.estimated_worker_scratch_bytes == first_plan.estimated_worker_scratch_bytes
        assert plan.estimated_output_bytes > first_plan.estimated_output_bytes
        with pytest.raises(MemoryError):
            scheduler.execute(graph, larger, starts, np.array([1026], np.int64), memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        result = scheduler.execute(graph, larger, starts, np.array([1026], np.int64), memory_budget_bytes=plan.estimated_total_memory_bytes)
        assert chunk_sum(result, "numeric_arena_bytes") == 0
        assert all(chunk["operator_workspace_capacity_bytes"] <= plan.estimated_worker_scratch_bytes for chunk in result.audit["native_chunks"])


@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_projection_bad_weight_geometry_remains_a_structural_error(policy):
    inputs = projection_fixture(assets=3)
    expressions = projection_expressions(3, 5)
    expressions["contribution_0_0"] = expressions["contribution_0_0"].rsplit("/", 1)[0] + "/denominator"
    types = declarations()
    types["denominator"] = dict(kind="scalar")
    graph = GraphCompiler(types).compile(expressions, error_policy=policy, result_format="typed")
    inputs["w1"] = np.ones(4)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises((ValueError, RuntimeError), match="symbolic dimensions|SHAPE"):
            scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([16], np.int64), parameters={"denominator": 0.})


@pytest.mark.parametrize("mode", ["ordinary", "prepared"])
def test_private_projection_rereads_coefficients_weights_and_scalar_divisor(mode):
    inputs = projection_fixture(steps=3, paths=2, assets=2)
    # Mutations preserve object, shape, stride and declaration identity.
    inputs = {name: value.copy() for name, value in inputs.items()}
    expressions = projection_expressions(2, 2)
    expressions["contribution_0_0"] = expressions["contribution_0_0"].rsplit("/", 1)[0] + "/denominator"
    types = declarations()
    types["denominator"] = dict(kind="scalar")
    graph = GraphCompiler(types).compile(expressions, result_format="typed")
    starts, ends = np.array([0], np.int64), np.array([3], np.int64)
    parameters = np.array([2.])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if mode == "prepared":
            prepared = scheduler.prepare_execution(graph, inputs, starts, ends, parameters=parameters)
            call = prepared.run_snapshot
        else:
            def call():
                return scheduler.execute(graph, inputs, starts, ends, parameters=parameters)
        previous = call()
        saved = [output.values[0].copy() for output in previous.outputs]
        inputs["beta"][:] *= 2.
        inputs["w0"][:] *= -3.
        parameters[0] = 4.
        current = call()
        expected = dict(zip(projection_expressions(2, 2), projection_expected(inputs["x"], inputs["beta"], inputs["initial"], [inputs["w0"], inputs["w1"]]), strict=True))
        expected["contribution_0_0"] /= 2.
        for name, output in current.named_outputs.items():
            np.testing.assert_allclose(output.values[0], expected[name], rtol=2e-14, atol=2e-16)
        for output, snapshot in zip(previous.outputs, saved, strict=True):
            ieee_equal(output.values[0], snapshot)
        assert chunk_sum(current, "numeric_arena_bytes") == 0
        parameters[0] = 0.
        with pytest.raises((ValueError, RuntimeError), match="DIVIDE_BY_ZERO"):
            call()


def failed_affine_seed_expression(coefficients="stack(w,w,0)", initial="0"):
    seed = "stack(block_apply(mean(x),1/z),block_apply(mean(x),1/z),1)"
    return f"iterate(affine_contract(iterate_x,{coefficients},{initial}),{seed},1e-6,2)"


@pytest.mark.parametrize("policy", ["raise", "isolate"])
@pytest.mark.parametrize("multihop", [False, True])
def test_unknown_affine_iterate_seed_preserves_original_numeric_error(policy, multihop):
    types = {"x": "series", "w": dict(kind="vector", axes=["component"], shape=["2"]), "z": "scalar"}
    expression = failed_affine_seed_expression()
    if multihop:
        projected = "sum_axis(affine_contract(iterate_x,stack(w,w,0),0),1)"
        seed = "stack(block_apply(mean(x),1/z),block_apply(mean(x),1/z),1)"
        expression = f"iterate(stack({projected},{projected},1),{seed},1e-6,2)"
    graph = GraphCompiler(types).compile({"failed": expression, "healthy": "mean(x)"}, error_policy=policy, result_format="typed")
    inputs = {"x": np.arange(4., dtype=np.float64), "w": np.ones(2)}
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if policy == "raise":
            with pytest.raises((ValueError, RuntimeError), match="DIVIDE_BY_ZERO"):
                scheduler.execute(graph, inputs, starts, ends, parameters={"z": 0.})
        else:
            result = scheduler.execute(graph, inputs, starts, ends, parameters={"z": 0.})
            np.testing.assert_array_equal(result.named_outputs["failed"].root_statuses, [2])
            assert result.named_outputs["healthy"].values[0] == 1.5
            np.testing.assert_array_equal(result.named_outputs["healthy"].root_statuses, [0])


@pytest.mark.parametrize("broken", ["coefficients", "initial"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_failed_affine_seed_cannot_hide_independent_known_input_geometry(broken, policy):
    types = {"x": "series", "z": "scalar",
             "coefficients": dict(kind="matrix", axes=["component", "component"], shape=["2", "2"]),
             "initial": dict(kind="vector", axes=["component"], shape=["2"])}
    graph = GraphCompiler(types).compile(failed_affine_seed_expression("coefficients", "initial"), error_policy=policy, result_format="typed")
    inputs = {"x": np.arange(4., dtype=np.float64), "coefficients": np.eye(2), "initial": np.zeros(2)}
    inputs[broken] = np.ones((2, 3)) if broken == "coefficients" else np.ones(3)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises((ValueError, RuntimeError), match="SHAPE|dimensions|dimension"):
            scheduler.execute(graph, inputs, np.array([0], np.int64), np.array([4], np.int64), parameters={"z": 0.})


def strict_projection_reference_graph(expressions, types=None, order="C"):
    roots = dict(expressions)
    roots["audit_returns"] = "expm1(affine_contract(x,transpose(beta),initial))"
    return GraphCompiler(types or declarations()).compile(roots, result_format="typed", output_order=order)


def assert_selected_outputs_exact(result, reference):
    for name, output in result.named_outputs.items():
        expected = reference.named_outputs[name]
        assert output.axes == expected.axes and output.dtype == expected.dtype
        np.testing.assert_array_equal(output.root_statuses, expected.root_statuses)
        for value, target in zip(output.values, expected.values, strict=True):
            ieee_equal(value, target)


@pytest.mark.parametrize("order", ["C", "F"])
def test_projection_scan_merge_duplicate_roots_do_not_double_count(order):
    inputs = projection_fixture(steps=5, paths=3)
    roots = projection_expressions(3, 3)
    roots["duplicate_period"] = roots["period_0"]
    roots["duplicate_contribution"] = roots["contribution_0_1"]
    graph = GraphCompiler(declarations()).compile(roots, result_format="typed", output_order=order)
    canonical = strict_projection_reference_graph(roots, order=order)
    starts, ends = np.array([0], np.int64), np.array([5], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
        reference = scheduler.execute(canonical, inputs, starts, ends)
    assert_selected_outputs_exact(result, reference)
    assert chunk_sum(result, "numeric_arena_bytes") == 0
    for first, duplicate in (("period_0", "duplicate_period"), ("contribution_0_1", "duplicate_contribution")):
        original = result.named_outputs[first].values[0]
        another = result.named_outputs[duplicate].values[0]
        ieee_equal(original, another)
        assert not np.shares_memory(original, another)


def test_projection_scan_merge_partial_contribution_selection_keeps_unmatched_weights():
    inputs = projection_fixture(steps=7, paths=3)
    all_roots = projection_expressions(3, 3)
    roots = {name: all_roots[name] for name in ("period_1", "contribution_0_0", "contribution_0_2", "contribution_1_1")}
    graph = GraphCompiler(declarations()).compile(roots, result_format="typed")
    canonical = strict_projection_reference_graph(roots)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        starts, ends = np.array([0], np.int64), np.array([7], np.int64)
        result = scheduler.execute(graph, selected_inputs(graph, inputs), starts, ends)
        reference = scheduler.execute(canonical, selected_inputs(canonical, inputs), starts, ends)
    assert_selected_outputs_exact(result, reference)
    assert chunk_sum(result, "numeric_arena_bytes") == 0
    assert chunk_sum(result, "fused_pointwise_calls") == 1


def test_projection_scan_merge_same_content_weights_retain_separate_input_identity():
    inputs = {name: value.copy() for name, value in projection_fixture(steps=5, paths=3).items()}
    inputs["w1"][:] = inputs["w0"]
    all_roots = projection_expressions(3, 3)
    roots = {name: all_roots[name] for name in ("period_0", "contribution_1_0", "contribution_1_1")}
    graph = GraphCompiler(declarations()).compile(roots, result_format="typed")
    canonical = strict_projection_reference_graph(roots)
    starts, ends = np.array([0], np.int64), np.array([5], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        previous = scheduler.execute(graph, inputs, starts, ends)
        saved = {name: output.values[0].copy() for name, output in previous.named_outputs.items()}
        inputs["w1"][:] *= 3.
        changed = scheduler.execute(graph, inputs, starts, ends)
        reference = scheduler.execute(canonical, inputs, starts, ends)
        assert_selected_outputs_exact(changed, reference)
        ieee_equal(changed.named_outputs["period_0"].values[0], saved["period_0"])
        for name in ("contribution_1_0", "contribution_1_1"):
            assert changed.named_outputs[name].values[0] != saved[name]
        inputs["w0"][:] *= -2.
        changed_period = scheduler.execute(graph, inputs, starts, ends)
        reference = scheduler.execute(canonical, inputs, starts, ends)
        assert_selected_outputs_exact(changed_period, reference)
        for name in ("contribution_1_0", "contribution_1_1"):
            ieee_equal(changed_period.named_outputs[name].values[0], changed.named_outputs[name].values[0])
        for name, output in previous.named_outputs.items():
            ieee_equal(output.values[0], saved[name])


def test_projection_scan_merge_shared_sum_preserves_each_constant_and_parameter_divisor():
    inputs = projection_fixture(steps=5, paths=3)
    original = projection_expressions(3, 3)
    numerator = original["contribution_0_0"].rsplit("/", 1)[0]
    roots = {"period_0": original["period_0"], "by_three": numerator + "/3",
             "by_six": numerator + "/6", "by_parameter": numerator + "/denominator"}
    types = declarations()
    types["denominator"] = dict(kind="scalar")
    graph = GraphCompiler(types).compile(roots, result_format="typed")
    canonical = strict_projection_reference_graph(roots, types)
    starts, ends = np.array([0], np.int64), np.array([5], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        for divisor in (3., 12.):
            result = scheduler.execute(graph, inputs, starts, ends, parameters={"denominator": divisor})
            reference = scheduler.execute(canonical, inputs, starts, ends, parameters={"denominator": divisor})
            assert_selected_outputs_exact(result, reference)
            assert chunk_sum(result, "numeric_arena_bytes") == 0
            ieee_equal(result.named_outputs["by_six"].values[0], result.named_outputs["by_three"].values[0] / 2.)


@pytest.mark.parametrize("order", ["C", "F"])
def test_projection_scan_merge_reversed_period_root_order_keeps_contribution_folds(order):
    inputs = projection_fixture(steps=5, paths=3)
    original = projection_expressions(3, 3)
    roots = {name: original[name] for name in ("period_1", "period_0")}
    roots.update({name: expression for name, expression in original.items() if name.startswith("contribution")})
    graph = GraphCompiler(declarations()).compile(roots, result_format="typed", output_order=order)
    canonical = strict_projection_reference_graph(original, order=order)
    starts, ends = np.array([0], np.int64), np.array([5], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, inputs, starts, ends)
        reference = scheduler.execute(canonical, inputs, starts, ends)
    assert_selected_outputs_exact(result, reference)
    assert chunk_sum(result, "numeric_arena_bytes") == 0
