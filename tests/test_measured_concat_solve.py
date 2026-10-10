"""Terminal private concat LU: local backward envelope, replay and ownership."""

import os
import pickle
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle
from calmetrics_engine import operators as op

from .test_guarded_lu_graph import COUNTEREXAMPLES, count
from .test_solve_library import residual, same_ieee
from .test_typed_results import config

TYPES = {
    k: dict(kind="matrix", axes=["asset", "asset"], shape=shape)
    for k, shape in {
        "h": ["N", "N"],
        "g": ["M", "N"],
        "z": ["M", "M"],
        "b": ["N", "K"],
        "zr": ["M", "K"],
    }.items()
}
LIBRARY = int(
    os.environ.get(
        "CALMETRICS_TEST_EXPECT_ACCELERATE",
        str(
            int(
                platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and int(platform.mac_ver()[0].split(".")[0] or 0) >= 15
                and os.environ.get("CME_EXPECT_ACCELERATE") != "0"
            )
        ),
    )
)

COEFFICIENT = "concatenate(concatenate(h,transpose(g),1),concatenate(g,z,1),0)"
RHS = "concatenate(b,zr,0)"
EXPR = f"solve_multi({COEFFICIENT},{RHS})"


def fixture(n=1024, mode="c", columns=2, constraints=16):
    rng = np.random.default_rng(731)
    f = rng.normal(size=(n, 8)) * 0.01
    h = f @ f.T + np.diag(np.linspace(0.5, 1.0, n))
    g = np.zeros((constraints, n))
    if constraints:
        g[0] = 1.0
    for i in range(1, constraints):
        g[i, n - i] = -1.0
    values = dict(
        h=h,
        g=g,
        z=np.zeros((constraints, constraints)),
        b=rng.normal(size=(n, columns)),
        zr=np.zeros((constraints, columns)),
    )
    if mode == "f":
        values = {k: np.asfortranarray(v) for k, v in values.items()}
    elif mode == "reverse":
        values = {k: v[::-1, ::-1] for k, v in values.items()}
    elif mode == "zero":
        values["b"] = np.broadcast_to(values["b"][:, :1], values["b"].shape)
    for v in values.values():
        v.flags.writeable = False
    return values


def original(data):
    h, g, z, b, zr = [data[k] for k in TYPES]
    return np.block([[h, g.T], [g, z]]), np.concatenate((b, zr), axis=0)


def run(data, order="C", lane="single", expression=EXPR, **kwargs):
    graph = GraphCompiler(TYPES).compile(
        expression, result_format="typed", output_order=order, **kwargs
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        return scheduler.execute(graph, data, bounds, bounds)


@pytest.mark.parametrize("n", [1008, 1024, 1031])
@pytest.mark.parametrize("mode", ["c", "f", "reverse", "zero"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_non_dd_math_ownership_and_rectangles(n, mode, order):
    data = fixture(n, mode)
    a, b = original(data)
    expected = op.solve_multi(a, b)
    result = run(data, order)
    value = result.outputs[0].values[0]
    np.testing.assert_allclose(value, expected, rtol=1e-10, atol=4.9e-5)
    residual(a, value, b)
    assert count(result, "accelerate_lu_attempts") == LIBRARY
    if LIBRARY and mode != "reverse":
        assert count(result, "accelerate_lu_calls") == 1
        assert count(result, "solve_residual_materialization_bytes") == a.nbytes
    else:
        assert count(result, "accelerate_lu_calls") == 0
        assert count(result, "accelerate_lu_fallbacks") == LIBRARY
        same_ieee(value, expected)
    assert count(result, "input_copy_bytes") == 0
    assert value.flags.c_contiguous if order == "C" else value.flags.f_contiguous
    assert not value.flags.writeable
    for input in data.values():
        assert not np.shares_memory(input, value)


@pytest.mark.parametrize("entry", COUNTEREXAMPLES)
@pytest.mark.parametrize("order", ["C", "F"])
def test_original_threshold_counterexamples(entry, order):
    data = fixture(constraints=0, columns=1)
    h = np.eye(1024) * 2.0
    h[:2, :2] = entry["a"]
    data["h"] = h
    a, b = original(data)
    try:
        expected = op.solve_multi(a, b)
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            run(data, order)
        assert str(actual.value) == str(error)
    else:
        result = run(data, order)
        same_ieee(result.outputs[0].values[0], expected)
        assert count(result, "accelerate_lu_calls") == 0
        assert count(result, "accelerate_lu_fallbacks") == LIBRARY


@pytest.mark.parametrize(
    "fault", ["a_nan", "b_nan", "both", "singular", "zero_rhs", "tiny_rhs", "large_rhs"]
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_input_errors_replay_and_zero_results(fault, order):
    data = {k: v.copy() for k, v in fixture(constraints=0).items()}
    if fault in ["a_nan", "both"]:
        data["h"][-1, -1] = np.nan
    if fault in ["b_nan", "both"]:
        data["b"][-1, -1] = np.nan
    if fault == "singular":
        data["h"][-1] = 0.0
    if fault == "zero_rhs":
        data["b"][:] = 0.0
    if fault == "tiny_rhs":
        data["b"][:] = 1e-300
    if fault == "large_rhs":
        data["b"][:] = 1e308
    a, b = original(data)
    try:
        expected = op.solve_multi(a, b)
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            run(data, order)
        assert str(actual.value) == str(error)
    else:
        result = run(data, order)
        same_ieee(result.outputs[0].values[0], expected)
        assert count(result, "accelerate_lu_calls") == 0


@pytest.mark.parametrize("mode", ["downstream", "isolate", "extra"])
def test_consumer_and_error_policy_exclusions(mode):
    data = fixture()
    a, b = original(data)
    if mode == "downstream":
        expr = EXPR + "+0"
        kwargs = {}
    elif mode == "isolate":
        expr = EXPR
        kwargs = dict(error_policy="isolate")
    else:
        expr = [EXPR, "sum_axis(b,0)"]
        kwargs = {}
    result = run(data, expression=expr, **kwargs)
    if mode == "extra":
        assert count(result, "accelerate_lu_calls") == LIBRARY
    else:
        assert count(result, "accelerate_lu_calls") == 0
        same_ieee(
            result.outputs[0].values[0],
            op.add(op.solve_multi(a, b), 0.0) if mode == "downstream" else op.solve_multi(a, b),
        )


def test_retained_generations_and_changed_contents():
    data = {k: v.copy() for k, v in fixture(mode="f").items()}
    graph = GraphCompiler(TYPES).compile(EXPR, result_format="typed", output_order="F")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, data, bounds, bounds)
        old = first.outputs[0].values[0].copy()
        data["b"] *= 2
        second = scheduler.execute(graph, data, bounds, bounds)
        prepared = scheduler.prepare_execution(graph, data, bounds, bounds)
        third = prepared.run_snapshot()
        assert (
            count(first, "accelerate_lu_calls") == count(second, "accelerate_lu_calls") == LIBRARY
        )
        assert count(third, "accelerate_lu_calls") == LIBRARY
    same_ieee(first.outputs[0].values[0], old)
    np.testing.assert_allclose(second.outputs[0].values[0], 2 * old, rtol=1e-10, atol=4.9e-5)
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    np.testing.assert_allclose(
        second.outputs[0].values[0], third.outputs[0].values[0], rtol=1e-10, atol=4.9e-5
    )


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("fault", [False, True], ids=["healthy", "singular"])
def test_concat_transport_and_lifetime(lane, order, fault):
    data = fixture(mode="f", columns=1)
    if fault:
        data["g"] = np.zeros_like(data["g"])
        data["g"].flags.writeable = False
    saved = {k: v.copy() for k, v in data.items()}
    a, b = original(data)
    graph = pickle.loads(
        pickle.dumps(GraphCompiler(TYPES).compile(EXPR, result_format="typed", output_order=order))
    )
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:

        def execute(inputs):
            return scheduler.execute(graph, inputs, bounds, bounds)

        def check_input(inputs):
            if fault:
                with pytest.raises(ValueError) as reference:
                    op.solve_multi(a, b)
                error_type = RuntimeError if lane.startswith("process") else ValueError
                with pytest.raises(error_type) as actual:
                    execute(inputs)
                expected = ("native worker: " if lane.startswith("process") else "") + str(
                    reference.value
                )
                assert str(actual.value) == expected
                return None
            return execute(inputs), execute(inputs)

        if lane == "process_shared":
            with SharedInputBundle.from_inputs(data) as bundle:
                results = check_input(bundle)
        else:
            results = check_input(data)
    for k, v in data.items():
        same_ieee(v, saved[k])
    if fault:
        return
    first, second = results
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    assert count(first, "accelerate_lu_calls") == 2 * LIBRARY
    for result in results:
        for row in range(2):
            value = result.outputs[0].values[row]
            residual(a, value, b)
            assert not value.flags.writeable
            for input in data.values():
                assert not np.shares_memory(input, value)
    for x in first.outputs[0].values:
        for y in second.outputs[0].values:
            assert not np.shares_memory(x, y)
    assert not np.shares_memory(*first.outputs[0].values)
