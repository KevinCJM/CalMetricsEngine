"""Deferred private C arrays must survive slot reuse without altering results."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_typed_results import config


def matrix_schema():
    return dict(kind="matrix", axes=["asset", "factor"], shape=["N", "D"])


def compiler(extra_indices=False):
    specs = {
        "x": matrix_schema(),
        "indices": dict(kind="vector", dtype="int64", axes=["asset"], shape=["M"]),
    }
    if extra_indices:
        specs["other_indices"] = dict(kind="vector", dtype="int64", axes=["asset"], shape=["M"])
    return GraphCompiler(specs)


def fixture(layout="C", n=96, d=17, m=37):
    x = np.random.default_rng(5701).normal(size=(n, d)) + 0.4
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1, ::-1]
    indices = np.random.default_rng(5757).integers(0, n, size=m, dtype=np.int64)
    x.flags.writeable = indices.flags.writeable = False
    return {"x": x, "indices": indices}


def execute(expressions, data, *, order="C", lane="single", isolate=False):
    graph = compiler("other_indices" in data).compile(
        expressions,
        result_format="typed",
        output_order=order,
        error_policy="isolate" if isolate else "raise",
    )
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as engine:
        result = engine.execute(graph, data, bounds, bounds)
        assert engine.peak_active_cpu_tokens <= 2
        return result


def standardization(first="-", second="/", source_left=True, pair_left=True):
    source = "take_axis(x,indices,0)"
    mean = f"broadcast_like(mean_axis({source},0),{source})"
    scale = f"broadcast_like(sqrt(variance_axis({source},0,1))+2,{source})"
    middle = f"({source}){first}{mean}" if source_left else f"{mean}{first}({source})"
    return f"({middle}){second}{scale}" if pair_left else f"{scale}{second}({middle})"


@pytest.mark.parametrize(
    "operations", [("+", "+"), ("-", "/"), ("*", "-"), ("-", "-"), ("*", "/"), ("+", "*")]
)
@pytest.mark.parametrize("source_left", [False, True])
@pytest.mark.parametrize("pair_left", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "reverse"])
def test_private_array_and_short_snapshot_preserve_math_and_ownership(
    operations, source_left, pair_left, layout
):
    data = fixture(layout)
    expression = standardization(*operations, source_left, pair_left)
    order = "F" if layout == "F" else "C"
    before = {key: value.copy() for key, value in data.items()}
    actual = execute(expression, data, order=order)
    # Isolated execution deliberately retains original materialization and errors.
    control = execute(expression, data, order=order, isolate=True)
    for value, expected in zip(actual.outputs[0].values, control.outputs[0].values, strict=False):
        np.testing.assert_array_equal(value.ravel().view(np.uint8), expected.ravel().view(np.uint8))
        assert not value.flags.writeable and not np.shares_memory(value, data["x"])
    assert not np.shares_memory(*actual.outputs[0].values)
    for key, value in data.items():
        np.testing.assert_array_equal(value, before[key])
    chunk = actual.audit["native_chunks"][0]
    selected = data["x"][data["indices"]]
    assert chunk["fused_pointwise_calls"] > 0
    assert chunk["numeric_arena_bytes"] <= selected.nbytes + 12 * selected.shape[1] * 8
    assert control.audit["native_chunks"][0]["numeric_arena_bytes"] >= selected.nbytes * 2


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_private_pair_transport_has_two_independent_generations(lane, order):
    data = fixture(n=256, d=32, m=128)
    graph = compiler().compile(standardization(), result_format="typed", output_order=order)
    bounds = np.zeros(4, np.int64)
    selected = data["x"][data["indices"]]
    expected = (selected - selected.mean(0)) / (selected.std(0, ddof=1) + 2)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as engine:
        old = engine.execute(graph, data, bounds, bounds)
        saved = [value.copy() for value in old.outputs[0].values]
        new = engine.execute(graph, data, bounds, bounds)
        assert engine.peak_active_cpu_tokens <= 2
        assert new.audit["lane"] == ("process" if lane.startswith("process") else lane)
        for previous, snapshot, current in zip(
            old.outputs[0].values, saved, new.outputs[0].values, strict=False
        ):
            np.testing.assert_array_equal(previous, snapshot)
            np.testing.assert_allclose(current, expected, rtol=1e-13, atol=1e-13)
            assert not np.shares_memory(previous, current)
            assert not current.flags.writeable
            assert current.flags.f_contiguous if order == "F" else current.flags.c_contiguous
        assert all(chunk["fused_pointwise_calls"] > 0 for chunk in new.audit["native_chunks"])


def test_intervening_writer_of_dead_input_slot_prevents_deferral():
    data = fixture()
    data["other_indices"] = np.zeros(len(data["indices"]), np.int64)
    expression = "(take_axis(x,indices,0)-1)+take_axis(x,other_indices,0)"
    actual = execute(expression, data)
    expected = (data["x"][data["indices"]] - 1) + data["x"][data["other_indices"]]
    for value in actual.outputs[0].values:
        np.testing.assert_array_equal(value, expected)
    assert actual.audit["native_chunks"][0]["fused_pointwise_calls"] == 0


@pytest.mark.parametrize("view", ["slice", "diag"])
def test_borrowed_private_source_retains_canonical_path(view):
    data = fixture(n=96, d=37, m=37)
    source = "take_axis(x,indices,0)"
    if view == "slice":
        source = f"slice_axis({source},0,0,37,1)"
        expected = (data["x"][data["indices"]] - 1) + 2
    else:
        data = fixture(n=37, d=37, m=37)
        source = f"diag({source})"
        expected = (np.diag(data["x"][data["indices"]]) - 1) + 2
    if view == "diag":
        specs = {
            "x": dict(kind="matrix", axes=["asset", "asset"], shape=[37, 37]),
            "indices": dict(kind="vector", dtype="int64", axes=["asset"], shape=[37]),
        }
        graph = GraphCompiler(specs).compile(f"({source}-1)+2", result_format="typed")
        bounds = np.zeros(2, np.int64)
        with AdaptiveScheduler(cpu_budget=1) as engine:
            actual = engine.execute(graph, data, bounds, bounds)
    else:
        actual = execute(f"({source}-1)+2", data)
    for value in actual.outputs[0].values:
        np.testing.assert_array_equal(value, expected)
    assert actual.audit["native_chunks"][0]["fused_pointwise_calls"] == 0


@pytest.mark.parametrize("shape", [(0, 5), (1, 5), (2, 5), (5, 0), (7, 1)])
def test_private_geometry_handles_empty_and_short_shapes(shape):
    m, d = shape
    data = fixture(n=16, d=d, m=m)
    actual = execute("(take_axis(x,indices,0)-1)+2", data)
    for value in actual.outputs[0].values:
        np.testing.assert_array_equal(value, (data["x"][data["indices"]] - 1) + 2)


@pytest.mark.parametrize("order", ["C", "F"])
def test_another_late_consumer_preserves_private_source(order):
    data = fixture()
    actual = execute([standardization(), "sum_axis(take_axis(x,indices,0),0)"], data, order=order)
    control = execute(
        [standardization(), "sum_axis(take_axis(x,indices,0),0)"], data, order=order, isolate=True
    )
    for output, reference in zip(actual.outputs, control.outputs, strict=False):
        for value, expected in zip(output.values, reference.values, strict=False):
            np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_rereads_indices_and_input_without_overwriting_old_results(order):
    data = fixture()
    data["x"] = data["x"].copy()
    data["indices"] = data["indices"].copy()
    graph = compiler().compile(standardization(), result_format="typed", output_order=order)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        prepared = engine.prepare_execution(graph, data, bounds, bounds)
        old = prepared.run_snapshot().outputs[0].values[0]
        saved = old.copy()
        data["indices"][:] = data["indices"][::-1]
        data["x"] *= np.linspace(1, 3, data["x"].shape[1])
        data["x"] += np.arange(len(data["x"]))[:, None] * 0.2
        new = prepared.run_snapshot().outputs[0].values[0]
        selected = data["x"][data["indices"]]
        np.testing.assert_array_equal(old, saved)
        np.testing.assert_allclose(
            new,
            (selected - selected.mean(0)) / (selected.std(0, ddof=1) + 2),
            rtol=1e-13,
            atol=1e-13,
        )
        assert not np.shares_memory(old, new)


@pytest.mark.parametrize("order", ["C", "F"])
def test_divide_fault_and_independent_isolated_result_remain_original(order):
    data = fixture()
    data["x"] = np.ones_like(data["x"])
    expression = "(take_axis(x,indices,0)-1)/0"
    with pytest.raises((ValueError, RuntimeError), match="DIVIDE_BY_ZERO"):
        execute(expression, data, order=order)
    actual = execute([expression, "sum_axis(x,0)"], data, order=order, isolate=True)
    assert actual.outputs[0].root_statuses[0] != 0
    assert actual.outputs[1].root_statuses[0] == 0
    np.testing.assert_array_equal(actual.outputs[1].values[0], data["x"].sum(0))


def test_unused_nonfinite_rows_do_not_become_new_failures():
    data = fixture()
    data["indices"] = np.arange(37, dtype=np.int64)
    data["x"] = data["x"].copy()
    data["x"][77, 2] = np.nan
    data["x"].flags.writeable = data["indices"].flags.writeable = False
    actual = execute(standardization(), data)
    control = execute(standardization(), data, isolate=True)
    np.testing.assert_array_equal(actual.outputs[0].values[0], control.outputs[0].values[0])
