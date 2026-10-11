"""Private decision tiling preserves exact comparisons, errors and independent outputs."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_typed_results import config


def fixture(dtype="float64", stride=1, n=12317):
    rng = np.random.default_rng(581)
    x = rng.integers(-3, 9, n, dtype=np.int64)
    if dtype == "float64":
        x = x.astype(np.float64)
        x[::307] = np.nan
        x[::509] = np.inf
        x[::701] = -np.inf
    else:
        x[: min(6, n)] = [2**53 + 1, 2**53 + 2, 2**63 - 1, -(2**63), 0, -1][: min(6, n)]
    valid = rng.random(n) > 0.2
    # The existing Python float64 time-series boundary requires contiguous storage.
    selected = x[::stride]
    data = dict(
        x=np.ascontiguousarray(selected) if dtype == "float64" else selected, valid=valid[::stride]
    )
    for a in data.values():
        a.flags.writeable = False
    return data


def definitions(data):
    return {
        name: dict(kind="series", dtype=str(a.dtype), axes=["time"], shape=["N"])
        for name, a in data.items()
    }


def expression(op="==", previous_side=2, scalar_first=False, k=8):
    result = "-1"
    for s in range(k):
        predicate = f"{s}{op}x" if scalar_first else f"x{op}{s}"
        result = (
            f"state_select({predicate},{s},{result},valid)"
            if previous_side == 2
            else f"state_select({predicate},{result},{s},valid)"
        )
    return result


def reference(data, op="==", previous_side=2, scalar_first=False, k=8):
    functions = {
        "==": np.equal,
        "!=": np.not_equal,
        "<": np.less,
        "<=": np.less_equal,
        ">": np.greater,
        ">=": np.greater_equal,
    }
    result = np.full(len(data["x"]), -1, np.int64)
    for s in range(k):
        mask = functions[op](s, data["x"]) if scalar_first else functions[op](data["x"], s)
        result = np.where(mask, s, result) if previous_side == 2 else np.where(mask, result, s)
        result = np.where(data["valid"], result, -1)
    return result


def execute(g, data, lane="single", prepared=False):
    bounds = np.zeros(3, np.int64)
    ends = np.full(3, len(next(iter(data.values()))), np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as engine:
        run = (
            engine.prepare_execution(g, data, bounds, ends).run_snapshot
            if prepared
            else lambda: engine.execute(g, data, bounds, ends)
        )
        old = run()
        snapshots = [v.copy() for out in old.outputs for v in out.values]
        new = run()
        assert engine.peak_active_cpu_tokens <= 2
        for a, b, s in zip(
            [v for out in old.outputs for v in out.values],
            [v for out in new.outputs for v in out.values],
            snapshots,
            strict=True,
        ):
            np.testing.assert_array_equal(a, s)
            assert not a.flags.writeable and not np.shares_memory(a, b)
            assert not any(np.shares_memory(a, v) for v in data.values())
        return old


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
@pytest.mark.parametrize("stride", [1, 2, -1])
@pytest.mark.parametrize("previous_side", [1, 2])
@pytest.mark.parametrize("scalar_first", [False, True])
def test_numeric_math_and_tile_boundaries(op, stride, previous_side, scalar_first):
    data = fixture(stride=stride)
    expr = expression(op, previous_side, scalar_first)
    g = GraphCompiler(definitions(data)).compile(expr, result_format="typed")
    out = execute(g, data)
    for a in out.outputs[0].values:
        np.testing.assert_array_equal(a, reference(data, op, previous_side, scalar_first))
    for chunk in out.audit["native_chunks"]:
        assert chunk["fused_pointwise_calls"] > 0 and chunk["mask_arena_bytes"] == 0
        assert chunk["numeric_arena_bytes"] == 0
        assert chunk["pointwise_tiles"] > 0


@pytest.mark.parametrize("op", ["==", "!="])
@pytest.mark.parametrize("stride", [1, -1])
@pytest.mark.parametrize("previous_side", [1, 2])
@pytest.mark.parametrize("scalar_first", [False, True])
def test_int64_never_converts_through_float(op, stride, previous_side, scalar_first):
    data = fixture("int64", stride)
    g = GraphCompiler(definitions(data)).compile(
        expression(op, previous_side, scalar_first), result_format="typed"
    )
    out = execute(g, data)
    for a in out.outputs[0].values:
        np.testing.assert_array_equal(a, reference(data, op, previous_side, scalar_first))
    assert all(
        c["mask_arena_bytes"] == 0 and c["numeric_arena_bytes"] == 0
        for c in out.audit["native_chunks"]
    )


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
def test_transport_and_prepared_keep_independent_generations(lane, prepared, order):
    data = fixture("int64", -1)
    g = GraphCompiler(definitions(data)).compile(
        expression(), result_format="typed", output_order=order
    )
    g = pickle.loads(pickle.dumps(g))
    if prepared and lane != "single":
        with pytest.raises(ValueError, match="single-lane"):
            execute(g, data, lane, prepared)
        return
    out = execute(g, data, lane, prepared)
    for a in out.outputs[0].values:
        np.testing.assert_array_equal(a, reference(data))
    assert out.audit["lane"] == ("process" if lane.startswith("process") else lane)
    assert all(c["mask_arena_bytes"] == 0 for c in out.audit["native_chunks"])


@pytest.mark.parametrize("n", [0, 1, 2, 4095, 4096, 4097, 8192, 8193])
def test_empty_short_and_exact_tiles(n):
    data = fixture("int64", n=n)
    g = GraphCompiler(definitions(data)).compile(expression(k=2), result_format="typed")
    out = execute(g, data)
    for a in out.outputs[0].values:
        np.testing.assert_array_equal(a, reference(data, k=2))


@pytest.mark.parametrize("exposure", ["predicate", "intermediate", "source"])
def test_observable_intermediates_have_canonical_fallback(exposure):
    data = fixture("int64", n=91)
    extra = {"predicate": "x==7", "intermediate": expression(k=4), "source": "x"}[exposure]
    g = GraphCompiler(definitions(data)).compile([expression(), extra], result_format="typed")
    out = execute(g, data)
    for a in out.outputs[0].values:
        np.testing.assert_array_equal(a, reference(data))
    if exposure != "source":
        assert all(c["mask_arena_bytes"] > 0 for c in out.audit["native_chunks"])


@pytest.mark.parametrize("fault", ["threshold", "state", "shape"])
def test_original_failure_matches_isolated_materialization(fault):
    data = fixture("int64", n=51)
    expr = expression()
    if fault == "threshold":
        expr = expr.replace("x==3", "x==3.5")
    elif fault == "state":
        expr = expr.replace("x==3,3", "x==3,-2")
    elif fault == "shape":
        data["valid"] = np.ones(47, np.bool_)
    errors = []
    for isolate in [False, True]:
        try:
            g = GraphCompiler(definitions(data)).compile(
                expr, result_format="typed", error_policy="isolate" if isolate else "raise"
            )
            execute(g, data)
        except Exception as e:
            errors.append((type(e), str(e)))
    assert len(errors) == 2 and errors[0] == errors[1]


def test_computed_source_arena_is_retained_through_elided_writes():
    data = fixture("int64", n=10003)
    x = np.column_stack([data["x"] % 8, 7 - data["x"] % 8]).astype(np.float64)
    x.flags.writeable = False
    data = dict(matrix=x, valid=data["valid"])
    defs = definitions({"valid": data["valid"]})
    defs["matrix"] = dict(kind="matrix", axes=["time", "factor"], shape=["N", "2"])
    expr = expression().replace("x==", "argmax_axis(matrix,1)==")
    g = GraphCompiler(defs).compile(expr, result_format="typed")
    out = execute(g, data)
    expected = reference(dict(x=np.argmax(x, axis=1).astype(np.int64), valid=data["valid"]))
    for a in out.outputs[0].values:
        np.testing.assert_array_equal(a, expected)
    chunks = out.audit["native_chunks"]
    assert all(
        c["mask_arena_bytes"] == 0 and c["numeric_arena_bytes"] == len(x) * 8 for c in chunks
    )


@pytest.mark.parametrize(
    "mode", ["isolate", "minimum", "computed_valid", "one_step", "array_branch"]
)
def test_unproved_regions_keep_original_materialization(mode):
    data = fixture("int64", n=101)
    expr = expression(k=1 if mode == "one_step" else 8)
    kwargs = {}
    if mode == "isolate":
        kwargs["error_policy"] = "isolate"
    if mode == "minimum":
        kwargs["minimum_observations"] = 1
    if mode == "computed_valid":
        expr = expr.replace(",valid)", ",logical_and(valid,x==x))")
    if mode == "array_branch":
        expr = expr.replace("x==0,0,-1", "x==0,x,-1")
        # An int64 branch needs the public state semantic and valid code domain.
        data["x"] = np.arange(101, dtype=np.int64) % 8
    defs = definitions(data)
    if mode == "array_branch":
        defs["x"]["semantic_dimension"] = "state"
    g = GraphCompiler(defs).compile(expr, result_format="typed", **kwargs)
    out = execute(g, data)
    for a in out.outputs[0].values:
        np.testing.assert_array_equal(a, reference(data, k=1 if mode == "one_step" else 8))
    assert all(
        c["mask_arena_bytes"] > 0 and c["fused_pointwise_calls"] == 0
        for c in out.audit["native_chunks"]
    )


def test_two_distinct_decision_roots_share_no_results():
    data = fixture("int64", n=8301)
    exprs = [expression("=="), expression("!=", 1)]
    g = GraphCompiler(definitions(data)).compile(exprs, result_format="typed")
    out = execute(g, data)
    for j, (op, side) in enumerate([("==", 2), ("!=", 1)]):
        for a in out.outputs[j].values:
            np.testing.assert_array_equal(a, reference(data, op, side))
    assert not np.shares_memory(out.outputs[0].values[0], out.outputs[1].values[0])
    assert all(c["mask_arena_bytes"] == 0 for c in out.audit["native_chunks"])


def test_variable_temporal_intervals_keep_original_shapes():
    data = fixture("int64", n=9101)
    g = GraphCompiler(definitions(data)).compile(expression(), result_format="typed")
    starts = np.array([0, 3, 100, 4096, 9101], np.int64)
    ends = np.array([0, 8, 8501, 9101, 9101], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config("thread")) as engine:
        actual = engine.execute(g, data, starts, ends)
    full = reference(data)
    for a, b, e in zip(actual.outputs[0].values, starts, ends, strict=True):
        np.testing.assert_array_equal(a, full[b:e])
    assert all(c["mask_arena_bytes"] == 0 for c in actual.audit["native_chunks"])
