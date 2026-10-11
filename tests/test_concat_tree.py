"""Private nested concatenate lowering preserves shapes, payload and ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_guarded_lu_graph import count
from .test_solve_library import same_ieee
from .test_typed_results import config


def declaration(rank, dtype="float64", prefix="D", vary_axis=None):
    vary_axis = rank - 1 if vary_axis is None else vary_axis
    return dict(
        kind=["vector", "matrix", "tensor"][rank - 1],
        dtype=dtype,
        axes=["asset"] * rank,
        shape=[f"{prefix}{i}" if i == vary_axis else f"D{i}" for i in range(rank)],
    )


def execute(graph, data, lane="single", rows=1, scheduler=None):
    bounds = np.zeros(rows, np.int64)
    if scheduler is not None:
        return scheduler.execute(graph, data, bounds, bounds)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as owner:
        return owner.execute(graph, data, bounds, bounds)


def layout(a, mode):
    if mode == "f":
        return np.asfortranarray(a)
    if mode == "reverse":
        return a[(slice(None, None, -1),) * a.ndim]
    if mode == "zero":
        return np.broadcast_to(a.take([0], axis=-1), a.shape)
    return a


@pytest.mark.parametrize("rank,axis", [(r, a) for r in [1, 2, 3] for a in range(r)])
@pytest.mark.parametrize("mode", ["c", "f", "reverse", "zero"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_rectangles_and_bit_patterns(rank, axis, mode, order):
    shape = (3, 4, 5)[:rank]
    data = {
        k: layout(np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + i * 100.0, mode)
        for i, k in enumerate(["x", "y", "z"])
    }
    if mode != "zero":
        data["x"].flat[0] = -0.0
        data["y"].flat[0] = np.nan
        data["z"].flat[0] = np.inf
    saved = {k: v.copy() for k, v in data.items()}
    for v in data.values():
        v.flags.writeable = False
    expr = f"concatenate(concatenate(x,y,{axis}),z,{axis})"
    graph = GraphCompiler({k: declaration(rank, prefix=k, vary_axis=axis) for k in data}).compile(
        expr, output_order=order, result_format="typed"
    )
    result = execute(graph, data)
    actual = result.outputs[0].values[0]
    expected = np.concatenate(
        (np.concatenate((data["x"], data["y"]), axis=axis), data["z"]), axis=axis
    )
    same_ieee(actual, expected)
    assert not actual.flags.writeable
    assert actual.flags.c_contiguous if order == "C" else actual.flags.f_contiguous
    if order == "C":
        assert count(result, "numeric_arena_bytes") == 0
    for k, v in data.items():
        same_ieee(v, saved[k])
        assert not np.shares_memory(v, actual)


@pytest.mark.parametrize(
    "rank,axis,empty",
    [
        (r, a, e)
        for r in [1, 2, 3]
        for a in range(r)
        for e in ["first", "last", "all", "non_axis"]
        if e != "non_axis" or r > 1
    ],
)
def test_empty_leaf_geometry(rank, axis, empty):
    shape = list((2, 3, 4)[:rank])
    data = {k: np.ones(shape) for k in ["x", "y", "z"]}
    for k in ["x", "y", "z"]:
        dims = shape.copy()
        if empty == "all" or (empty == "first" and k == "x") or (empty == "last" and k == "z"):
            dims[axis] = 0
        elif empty == "non_axis":
            dims[(axis + 1) % rank] = 0
        data[k] = np.ones(dims)
    graph = GraphCompiler({k: declaration(rank, prefix=k, vary_axis=axis) for k in data}).compile(
        f"concatenate(concatenate(x,y,{axis}),z,{axis})", result_format="typed"
    )
    result = execute(graph, data)
    actual = result.outputs[0].values[0]
    same_ieee(actual, np.concatenate(tuple(data.values()), axis=axis))
    assert count(result, "numeric_arena_bytes") == 0


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("case", ["public_inner", "shared_inner", "computed_leaf", "isolate"])
def test_retained_or_computed_branches_keep_real_values(order, case):
    data = {
        k: np.asfortranarray(np.arange(12.0).reshape(3, 4) + i * 20)
        for i, k in enumerate(["x", "y", "z"])
    }
    inner = "concatenate(x,y,1)"
    outer = f"concatenate({inner},z,1)"
    expressions = [outer]
    if case == "public_inner":
        expressions += [inner]
    if case == "shared_inner":
        expressions += [f"sum_axis({inner},0)"]
    if case == "computed_leaf":
        expressions = ["concatenate(concatenate(add(x,1),y,1),z,1)"]
    graph = GraphCompiler({k: declaration(2, prefix=k) for k in data}).compile(
        expressions,
        output_order=order,
        result_format="typed",
        error_policy="isolate" if case == "isolate" else "raise",
    )
    result = execute(graph, data)
    expected = np.concatenate(
        (data["x"] + (1 if case == "computed_leaf" else 0), data["y"], data["z"]), axis=1
    )
    same_ieee(result.outputs[0].values[0], expected)
    if case == "public_inner":
        same_ieee(result.outputs[1].values[0], np.concatenate((data["x"], data["y"]), axis=1))
    if case == "shared_inner":
        same_ieee(
            result.outputs[1].values[0], np.concatenate((data["x"], data["y"]), axis=1).sum(axis=0)
        )
    if case != "public_inner" or order == "F":
        assert count(result, "numeric_arena_bytes") > 0


@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_other_dtypes_keep_canonical_materialization(dtype):
    data = {k: np.arange(6).reshape(2, 3).astype(dtype) for k in ["x", "y", "z"]}
    types = {k: declaration(2, dtype, prefix=k) for k in data}
    if dtype == "bool":
        for t in types.values():
            t["semantic_dimension"] = "mask"
    graph = GraphCompiler(types).compile("concatenate(concatenate(x,y,1),z,1)")
    result = execute(graph, data)
    np.testing.assert_array_equal(
        result.outputs[0].values[0], np.concatenate(tuple(data.values()), axis=1)
    )


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_transports_two_generations_and_old_owners(lane, order):
    data = {
        k: np.asfortranarray(np.arange(512.0).reshape(32, 16) + i * 1000)
        for i, k in enumerate(["x", "y", "z"])
    }
    saved = {k: v.copy() for k, v in data.items()}
    for v in data.values():
        v.flags.writeable = False
    graph = GraphCompiler({k: declaration(2, prefix=k) for k in data}).compile(
        "concatenate(concatenate(x,y,1),z,1)", output_order=order, result_format="typed"
    )
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = execute(graph, data, lane, 2, scheduler)
        second = execute(graph, data, lane, 2, scheduler)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    retained = [a for r in [first, second] for a in r.outputs[0].values]
    for i, a in enumerate(retained):
        same_ieee(a, np.concatenate(tuple(data.values()), axis=1))
        assert all(not np.shares_memory(a, b) for b in retained[i + 1 :])
        assert all(not np.shares_memory(a, v) for v in data.values())
    for k, v in data.items():
        same_ieee(v, saved[k])


def test_shared_dimensions_fail_before_independent_graph_error():
    types = {k: declaration(2, prefix=k) for k in ["x", "y", "z"]}
    data = {"x": np.ones((2, 3)), "y": np.ones((4, 3)), "z": np.ones((2, 3))}
    # Input dimension binding fails before the independent logarithm error.
    graph = GraphCompiler(types).compile(
        ["concatenate(concatenate(x,y,1),z,1)", "log(-1)"], result_format="typed"
    )
    with pytest.raises(ValueError, match="symbolic dimensions"):
        execute(graph, data)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("order", ["C", "F"])
def test_different_axis_subtrees(rank, order):
    shape = (7, 11, 5)[:rank]
    data = {
        k: np.asfortranarray(np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + i * 1000)
        for i, k in enumerate(["x", "y", "z", "w"])
    }
    types = {
        k: dict(
            kind="matrix" if rank == 2 else "tensor",
            axes=["asset"] * rank,
            shape=["N", "K", "L"][:rank],
        )
        for k in data
    }
    graph = GraphCompiler(types).compile(
        f"concatenate(concatenate(x,y,0),concatenate(z,w,0),{rank - 1})",
        output_order=order,
        result_format="typed",
    )
    result = execute(graph, data)
    same_ieee(
        result.outputs[0].values[0],
        np.concatenate(
            (
                np.concatenate((data["x"], data["y"]), axis=0),
                np.concatenate((data["z"], data["w"]), axis=0),
            ),
            axis=rank - 1,
        ),
    )
    if order == "C":
        assert count(result, "numeric_arena_bytes") == 0


@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_mutation_and_retained_results(order):
    data = {k: np.arange(12.0).reshape(3, 4) + i * 100 for i, k in enumerate(["x", "y", "z"])}
    graph = GraphCompiler({k: declaration(2, prefix=k) for k in data}).compile(
        "concatenate(concatenate(x,y,1),z,1)", output_order=order, result_format="typed"
    )
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, data, bounds, bounds)
        first = prepared.run_snapshot()
        old = first.outputs[0].values[0].copy()
        data["y"] *= 2
        second = prepared.run_audit()
        same_ieee(second.outputs[0].values[0], np.concatenate(tuple(data.values()), axis=1))
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    same_ieee(first.outputs[0].values[0], old)


@pytest.mark.parametrize("rows", [31, 32, 33, 63, 65])
@pytest.mark.parametrize("cols", [31, 32, 33, 67])
@pytest.mark.parametrize("mode", ["f", "f_padded", "f_reverse"])
def test_large_f_rectangles_cross_tile_boundaries(rows, cols, mode):
    data = {}
    for i, k in enumerate(["x", "y", "z", "w"]):
        physical = np.asfortranarray(
            np.arange((rows + 17) * cols, dtype=np.float64).reshape(rows + 17, cols) + i * 10000
        )
        value = physical[:rows] if mode == "f_padded" else np.asfortranarray(physical[:rows])
        value[0, 0] = -0.0
        value[-1, -1] = np.nan
        if mode == "f_reverse":
            value = value[::-1, ::-1]
        value.flags.writeable = False
        data[k] = value
    types = {k: dict(kind="matrix", axes=["asset", "asset"], shape=["N", "K"]) for k in data}
    graph = GraphCompiler(types).compile(
        "concatenate(concatenate(x,y,0),concatenate(z,w,0),1)", result_format="typed"
    )
    result = execute(graph, data)
    actual = result.outputs[0].values[0]
    expected = np.block([[data["x"], data["z"]], [data["y"], data["w"]]])
    same_ieee(actual, expected)
    assert count(result, "numeric_arena_bytes") == 0
    assert all(not np.shares_memory(actual, v) for v in data.values())
