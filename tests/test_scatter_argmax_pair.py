"""Private scatter/argmax fusion: integer decisions, failures and ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_typed_results import config


def graph(data, rows, fill, *, control=False, isolate=False, computed=False, axis=1):
    defs = {}
    for name, a in data.items():
        defs[name] = {
            "kind": {1: "vector", 2: "matrix", 3: "tensor"}[a.ndim],
            "axes": ["asset", "factor", "scenario"][: a.ndim],
            "shape": [str(x) for x in a.shape],
            "dtype": str(a.dtype),
        }
    # Axis 1 is already a graph constant before scatter; take preserves all
    # selected rows including nonfinite data and arbitrary column strides.
    source = "take_axis(x,cols,1)" if axis == 1 else "x"
    indices = "argsort(score)" if computed else "ix"
    scatter = (
        f"scatter_rows({source},{indices},{rows},{fill})"
        if fill is not None
        else f"scatter_rows({source},{indices},{rows})"
    )
    expr = f"argmax_axis({scatter},{axis})"
    return GraphCompiler(defs).compile(
        [expr, scatter] if control else expr,
        result_format="typed",
        error_policy="isolate" if isolate else "raise",
    )


def sample(layout="c", shape=(4, 7)):
    rng = np.random.default_rng(8347)
    x = rng.normal(size=shape)
    x[:, 0] = 0.0
    x[:, 1] = -0.0
    x[:, 2] = 2.0
    x[:, 3] = 2.0
    if layout == "f":
        x = np.asfortranarray(x)
    elif layout == "negative":
        x = x[::-1, ::-1].copy()[::-1, ::-1]
    elif layout == "zero":
        x = np.broadcast_to(x[:1], x.shape)
    cols = np.arange(shape[1], dtype=np.int64)
    ix = np.array([7, 1, 5, 3], dtype=np.int64)
    for a in (x, cols, ix):
        a.flags.writeable = False
    return {"x": x, "cols": cols, "ix": ix}


def run(g, data, *, lane="single", prepared=False):
    starts = np.zeros(2, np.int64)
    ends = starts.copy()
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as s:
        f = (
            s.prepare_execution(g, data, starts, ends).run_snapshot
            if prepared
            else lambda: s.execute(g, data, starts, ends)
        )
        first = f()
        saved = [a.copy() for a in first.outputs[0].values]
        second = f()
        for old, snapshot, new in zip(first.outputs[0].values, saved, second.outputs[0].values, strict=True):
            np.testing.assert_array_equal(old, snapshot)
            assert not np.shares_memory(old, new)
            assert not np.shares_memory(old, data["ix"])
        return first


@pytest.mark.parametrize("layout", ["c", "f", "negative", "zero"])
@pytest.mark.parametrize("fill", ["0", "1", "-2"])
@pytest.mark.parametrize("prepared", [False, True])
def test_exact_decisions_and_retained_results(layout, fill, prepared):
    data = sample(layout)
    expected = run(graph(data, 9, fill, control=True), data, prepared=prepared)
    actual = run(graph(data, 9, fill), data, prepared=prepared)
    for a, b in zip(actual.outputs[0].values, expected.outputs[0].values, strict=True):
        assert a.dtype == np.int64 and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert actual.audit["native_chunks"][0]["fused_pointwise_calls"] > 0
    assert actual.audit["native_chunks"][0]["numeric_arena_bytes"] <= 4 * 7 * 8
    assert (
        expected.audit["native_chunks"][0]["direct_output_bytes"]
        > actual.audit["native_chunks"][0]["direct_output_bytes"]
    )
    assert data["x"].flags.writeable is False


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
def test_transport(lane):
    data = sample("negative")
    a = run(graph(data, 9, "0"), data, lane=lane)
    b = run(graph(data, 9, "0", control=True), data, lane=lane)
    for x, y in zip(a.outputs[0].values, b.outputs[0].values, strict=True):
        assert x.tobytes() == y.tobytes()


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "negative",
        "upper",
        "nan",
        "inf",
        "nan_fill",
        "inf_fill",
        "default_fill",
        "zero_width",
    ],
)
def test_exception_type_message_and_recovery(fault):
    data = sample()
    data = {k: a.copy() for k, a in data.items()}
    fill = "0"
    if fault == "duplicate":
        data["ix"][1] = data["ix"][0]
    elif fault == "negative":
        data["ix"][1] = -1
    elif fault == "upper":
        data["ix"][1] = 9
    elif fault == "nan":
        data["x"][1, 4] = np.nan
    elif fault == "inf":
        data["x"][2, 4] = np.inf
    elif fault == "nan_fill":
        fill = "0/0"
    elif fault == "inf_fill":
        fill = "1/0"
    elif fault == "default_fill":
        fill = None
    elif fault == "zero_width":
        data["x"] = np.empty((4, 0))
        data["cols"] = np.empty(0, np.int64)
    errors = []
    for control in (False, True):
        with pytest.raises(Exception) as error:
            run(graph(data, 9, fill, control=control), data)
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]
    good = sample()
    a = run(graph(good, 9, "0"), good)
    b = run(graph(good, 9, "0", control=True), good)
    assert a.outputs[0].values[0].tobytes() == b.outputs[0].values[0].tobytes()


@pytest.mark.parametrize("fill", [None, "0/0", "1/0"])
def test_unused_fill_preserves_original_expression_evaluation(fill):
    # Division fills fail while evaluating their own nodes; a default NaN
    # fill is overwritten when every output row is covered and must succeed.
    data = sample()
    data["ix"] = np.arange(4, dtype=np.int64)
    if fill is None:
        a = run(graph(data, 4, fill), data)
        b = run(graph(data, 4, fill, control=True), data)
        assert a.outputs[0].values[0].tobytes() == b.outputs[0].values[0].tobytes()
    else:
        for control in (False, True):
            with pytest.raises(Exception, match="DIVIDE_BY_ZERO"):
                run(graph(data, 4, fill, control=control), data)


@pytest.mark.parametrize("shape,rows", [((0, 7), 0), ((0, 0), 0), ((0, 7), 5), ((4, 0), 0)])
def test_empty_geometry(shape, rows):
    data = {
        "x": np.empty(shape),
        "ix": np.zeros(shape[0], np.int64),
        "cols": np.arange(shape[1], dtype=np.int64),
    }
    if shape[0]:
        for control in (False, True):
            with pytest.raises((ValueError, RuntimeError)):
                run(graph(data, rows, "0", control=control), data)
    else:
        a = run(graph(data, rows, "0"), data)
        b = run(graph(data, rows, "0", control=True), data)
        assert a.outputs[0].values[0].tobytes() == b.outputs[0].values[0].tobytes()


def test_computed_indices_do_not_alias_integer_destination():
    data = sample()
    data["ix"] = np.arange(4, dtype=np.int64)
    data["score"] = np.array([2.0, 4.0, 1.0, 3.0])
    a = run(graph(data, 4, "0", computed=True), data)
    b = run(graph(data, 4, "0", computed=True, control=True), data)
    assert a.outputs[0].values[0].tobytes() == b.outputs[0].values[0].tobytes()
    assert a.audit["native_chunks"][0]["numeric_arena_bytes"] >= 4 * 7 * 8


@pytest.mark.parametrize("shape,axis", [((4, 7, 3), 1), ((4, 7), 0)])
def test_other_geometry_axes_stay_canonical(shape, axis):
    data = {
        "x": np.arange(np.prod(shape), dtype=np.float64).reshape(shape),
        "ix": np.array([7, 1, 5, 3], np.int64),
        "cols": np.arange(shape[1], dtype=np.int64),
    }
    a = run(graph(data, 9, "0", axis=axis), data)
    b = run(graph(data, 9, "0", control=True, axis=axis), data)
    assert a.outputs[0].values[0].tobytes() == b.outputs[0].values[0].tobytes()
    assert a.audit["native_chunks"][0]["numeric_arena_bytes"] >= 9 * np.prod(shape[1:]) * 8


def test_isolate_stays_canonical():
    data = sample()
    a = run(graph(data, 9, "0", isolate=True), data)
    b = run(graph(data, 9, "0", isolate=True, control=True), data)
    assert a.outputs[0].values[0].tobytes() == b.outputs[0].values[0].tobytes()
    assert a.audit["native_chunks"][0]["numeric_arena_bytes"] >= 9 * 7 * 8
