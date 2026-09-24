"""Physical optimizations preserve logical roots, ownership and IEEE semantics."""

import itertools
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler


def schema(shape):
    return {
        "kind": {1: "vector", 2: "matrix", 3: "tensor"}[len(shape)],
        "axes": ["scenario", "path", "asset"][-len(shape) :],
        "shape": ["S", "P", "N"][-len(shape) :],
    }


@pytest.mark.parametrize("shape", [(0,), (1,), (4097,), (35, 117), (17, 13, 19)])
@pytest.mark.parametrize("layout", ["C", "negative", "F"])
def test_fused_roots_shared_ancestors_duplicates_and_snapshots(shape, layout):
    rng = np.random.default_rng(123)
    base = rng.normal(size=shape)
    if base.size:
        base.flat[0] = np.nan
    if base.size > 2:
        base.flat[1:3] = [np.inf, -np.inf]
    x = base if layout == "C" else base[::-1] if layout == "negative" else np.asfortranarray(base)
    x.flags.writeable = False
    graph = GraphCompiler({"x": schema(shape)}).compile(
        {"a": "x*2+1", "b": "x*2+1>3", "again": "x*2+1", "input": "x"}
    )
    graph = pickle.loads(pickle.dumps(graph))
    with AdaptiveScheduler(cpu_budget=1) as engine:
        prepared = engine.prepare_execution(
            graph, {"x": x}, np.array([0], np.int64), np.array([1], np.int64)
        )
        first = prepared.run_snapshot()
        second = prepared.run_snapshot()
        expected = [x * 2 + 1, x * 2 + 1 > 3, x * 2 + 1, x]
        for i, wanted in enumerate(expected):
            actual = first.outputs[i].values[0]
            np.testing.assert_array_equal(actual, wanted)
            np.testing.assert_array_equal(second.outputs[i].values[0], wanted)
            assert not np.shares_memory(actual, x)
            assert not np.shares_memory(actual, second.outputs[i].values[0])
        audit = first.audit["native_chunks"][0]
        if x.flags.c_contiguous and x.size and layout != "negative":
            assert audit["fused_pointwise_calls"] == 1
            assert audit["numeric_arena_bytes"] <= 2048 * 8
            assert audit["input_copy_bytes"] == 0
        # Duplicated roots must be independent physical fields.
        assert not np.shares_memory(first.outputs[0].values[0], first.outputs[2].values[0])


@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_direct_root_destination_keeps_borrowed_consumers(policy):
    x = np.arange(12.0, dtype=np.float64)
    graph = GraphCompiler({"x": schema(x.shape)}).compile(
        {"root": "x+1", "child": "sum(x+1)", "other": "(x+1)*2"}, error_policy=policy
    )
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x}, np.array([0], np.int64), np.array([1], np.int64))
    np.testing.assert_array_equal(result.outputs[0].values[0], x + 1)
    assert result.outputs[1].values[0] == np.sum(x + 1)
    np.testing.assert_array_equal(result.outputs[2].values[0], (x + 1) * 2)
    assert result.audit["native_chunks"][0]["direct_output_bytes"] == x.nbytes * 2


def test_tensor_temporal_rows_keep_slice_boundaries():
    x = np.arange(5 * 3 * 7.0).reshape(5, 3, 7)
    kind = {"kind": "tensor", "axes": ["time", "path", "asset"], "shape": ["T", "P", "N"]}
    graph = GraphCompiler({"x": kind}).compile({"value": "x*2+1", "mask": "x>3"})
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(
            graph, {"x": x}, np.array([0, 1, 4], np.int64), np.array([0, 4, 5], np.int64)
        )
    for row, (begin, end) in enumerate([(0, 0), (1, 4), (4, 5)]):
        np.testing.assert_array_equal(result.outputs[0].values[row], x[begin:end] * 2 + 1)
        np.testing.assert_array_equal(result.outputs[1].values[row], x[begin:end] > 3)


@pytest.mark.parametrize("size", [1, 4097])
@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5])
def test_iteration_alternating_buffers_preserve_final_state(size, steps):
    x = np.full(size, 1e307)
    expr = f"iterate(iterate_x*2,x,0,{steps})"
    graph = GraphCompiler({"x": schema(x.shape)}).compile(
        {"value": expr, "status": f"iteration_status({expr})", "count": f"iteration_count({expr})"}
    )
    with AdaptiveScheduler(cpu_budget=1) as engine:
        prepared = engine.prepare_execution(
            graph, {"x": x}, np.array([0], np.int64), np.array([1], np.int64)
        )
        results = [prepared.run_snapshot() for _ in range(3)]
    for result in results:
        np.testing.assert_array_equal(result.outputs[0].values[0], x * 2 ** min(steps, 4))
        assert result.outputs[1].values[0] == (5 if steps == 5 else 1)
        assert result.outputs[2].values[0] == steps
    np.testing.assert_array_equal(x, np.full(size, 1e307))


@pytest.mark.parametrize("cpu", [1, 2, 4])
@pytest.mark.parametrize("temporal", [False, True])
def test_tensor_partition_uses_one_budget_and_disjoint_root_slots(cpu, temporal):
    from calmetrics_engine import PlannerConfig

    shape = (65, 33, 17)
    x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    kind = schema(shape)
    if temporal:
        kind = dict(kind, axes=["time", "path", "asset"], shape=["T", "P", "N"])
    graph = GraphCompiler({"x": kind}).compile({"value": "x*2+1", "mask": "x*2+1>3"})
    starts, ends = np.array([0], np.int64), np.array([len(x) if temporal else 1], np.int64)
    config = PlannerConfig(thread_work_units=1, process_work_units=1e100)
    with AdaptiveScheduler(cpu_budget=cpu, config=config) as engine:
        result = engine.execute(graph, {"x": x}, starts, ends)
    np.testing.assert_array_equal(result.outputs[0].values[0], x * 2 + 1)
    np.testing.assert_array_equal(result.outputs[1].values[0], x * 2 + 1 > 3)
    assert result.audit["cpu_tokens"] <= cpu
    if cpu > 1:
        assert result.plan.lane == "thread"
        assert result.plan.parallel_dimension == "tensor"
        assert result.audit["native_threads"] == cpu
    assert (
        sum(chunk["direct_output_bytes"] for chunk in result.audit["native_chunks"]) == x.size * 9
    )


def test_short_work_defaults_to_single_even_with_four_cpu_tokens():
    from calmetrics_engine import PlannerConfig

    graph = GraphCompiler({"x": "series"}).compile("mean(x)")
    x = np.ones(100)
    config = PlannerConfig()
    with AdaptiveScheduler(cpu_budget=4, config=config) as engine:
        result = engine.execute(graph, {"x": x}, np.zeros(32, np.int64), np.ones(32, np.int64))
    assert result.plan.lane == "single"
    assert result.audit["native_processes"] == 0


@pytest.mark.parametrize("ops", list(itertools.product(["+", "-", "*"], repeat=2)))
@pytest.mark.parametrize("sides", list(itertools.product([False, True], repeat=2)))
@pytest.mark.parametrize("layout", ["C", "negative", "zero"])
def test_register_chain_preserves_rounding_order_roots_and_strides(ops, sides, layout):
    base = np.resize(np.array([1.0 - 2.0**-27, -0.0, 0.0, np.nan, np.inf, -np.inf, 2.0]), (3, 5, 9))
    x = (
        base
        if layout == "C"
        else base[:, ::-1, ::-1]
        if layout == "negative"
        else np.broadcast_to(base[:, :, :1], base.shape)
    )
    x.flags.writeable = False
    first = f"c{ops[0]}x" if sides[0] else f"x{ops[0]}c"
    second = f"d{ops[1]}({first})" if sides[1] else f"({first}){ops[1]}d"
    graph = GraphCompiler({"x": schema(x.shape), "c": "scalar", "d": "scalar"}).compile(
        {
            "first": first,
            "second": second,
            "mask": f"({second})>0",
            "input": "x",
            "duplicate": second,
        }
    )
    fn = {"+": np.add, "-": np.subtract, "*": np.multiply}
    c, d = 1.0 + 2.0**-27, -1.0
    with np.errstate(invalid="ignore", over="ignore"):
        a = fn[ops[0]](c, x) if sides[0] else fn[ops[0]](x, c)
        b = fn[ops[1]](d, a) if sides[1] else fn[ops[1]](a, d)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(
            graph,
            {"x": x},
            np.array([0], np.int64),
            np.array([1], np.int64),
            parameters={"c": c, "d": d},
        )
    for field, expected in zip(result.outputs, [a, b, b > 0, x, b], strict=True):
        np.testing.assert_array_equal(field.values[0], expected)
        if expected.dtype == np.float64:
            zero = expected == 0
            np.testing.assert_array_equal(
                np.signbit(field.values[0][zero]), np.signbit(expected[zero])
            )
    assert result.audit["native_chunks"][0]["numeric_arena_bytes"] == 0


@pytest.mark.parametrize("predicate", ["==", "!=", "<", "<=", ">", ">="])
@pytest.mark.parametrize("reverse", [False, True])
def test_register_predicate_only_and_scalar_operand_order(predicate, reverse):
    x = np.array([np.nan, -np.inf, -0.0, 0.0, 1.0, np.inf])
    expression = f"0{predicate}x" if reverse else f"x{predicate}0"
    graph = GraphCompiler({"x": schema(x.shape)}).compile({"mask": expression})
    compare = {
        "==": np.equal,
        "!=": np.not_equal,
        "<": np.less,
        "<=": np.less_equal,
        ">": np.greater,
        ">=": np.greater_equal,
    }[predicate]
    expected = compare(0, x) if reverse else compare(x, 0)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x}, np.array([0], np.int64), np.array([1], np.int64))
    np.testing.assert_array_equal(result.outputs[0].values[0], expected)


@pytest.mark.parametrize(
    "layout",
    ["inner_reverse", "all_reverse", "outer_reverse", "transpose", "zero", "step", "singleton"],
)
def test_register_chain_strided_runs_cross_cache_tiles(layout):
    base = np.arange(7 * 17 * 19.0, dtype=np.float64).reshape(7, 17, 19)
    layouts = {
        "inner_reverse": base[:, ::-1, ::-1],
        "all_reverse": base[::-1, ::-1, ::-1],
        "outer_reverse": base[::-1],
        "transpose": base.transpose(2, 0, 1),
        "zero": np.broadcast_to(base[:, :1, :1], base.shape),
        "step": base[:, :, ::2],
        "singleton": base.reshape(1, 7, 17 * 19)[:, ::-1, ::-1],
    }
    x = layouts[layout]
    x.flags.writeable = False
    graph = GraphCompiler({"x": schema(x.shape)}).compile({"value": "x*2+1", "mask": "x*2+1>3000"})
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": x}, np.array([0], np.int64), np.array([1], np.int64))
    np.testing.assert_array_equal(result.outputs[0].values[0], x * 2 + 1)
    np.testing.assert_array_equal(result.outputs[1].values[0], x * 2 + 1 > 3000)
    assert result.audit["native_chunks"][0]["input_copy_bytes"] == 0


def test_iteration_finiteness_proof_rechecks_mutable_coefficients():
    from tests.test_iteration_protocol import outputs

    x = np.resize(np.array([8.0, -8.0, -0.0, np.finfo(float).max]), 4097)
    x.flags.writeable = False
    expr = "iterate(iterate_x*k,x,0,4)"
    graph = GraphCompiler({"x": schema(x.shape), "k": "scalar"}).compile(outputs(expr))
    params = np.array([0.5])
    starts, ends = np.array([0], np.int64), np.array([1], np.int64)
    saved = []
    with AdaptiveScheduler(cpu_budget=1) as engine:
        for factor in [0.5, 2.0, -1.0, np.inf, 0.0, np.nan, 1.0, -0.5, 1.00001]:
            params[0] = factor
            state = x.copy()
            residual, status, steps = np.inf, 1, 0
            with np.errstate(over="ignore", invalid="ignore"):
                for step_index in range(1, 5):
                    steps = step_index
                    candidate = state * factor
                    if not np.all(np.isfinite(candidate)):
                        status, residual = 5, np.inf
                        break
                    residual = np.max(np.abs(candidate - state))
                    state = candidate
                    if residual <= 0:
                        status = 0
                        break
            result = engine.execute(graph, {"x": x}, starts, ends, parameters=params)
            fields = result.named_outputs
            np.testing.assert_array_equal(fields["value"].values[0], state)
            assert fields["count"].values[0] == steps
            assert fields["status"].values[0] == status
            assert fields["residual"].values[0] == residual
            if np.isfinite(factor) and abs(factor) <= 1:
                assert (
                    result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"] < x.nbytes
                )
            saved.append((result, state.copy()))
    for result, expected in saved:
        np.testing.assert_array_equal(result.outputs[0].values[0], expected)
        assert not np.shares_memory(result.outputs[0].values[0], x)


def test_ordinary_plan_cache_revalidates_geometry_and_request_policy():
    graph = GraphCompiler({"x": "series"}).compile("mean(x)")
    x = np.arange(32.0)
    starts, ends = np.array([0], np.int64), np.array([16], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        a = engine.execute(graph, {"x": x}, starts, ends)
        b = engine.execute(graph, {"x": x}, starts, ends)
        assert a.plan is b.plan
        assert not np.shares_memory(a.values, b.values)
        ends[0] = 8
        c = engine.execute(graph, {"x": x}, starts, ends)
        assert c.plan is not b.plan and c.values[0, 0] == 3.5
        with pytest.raises(ValueError, match="stale"):
            engine.execute(graph, {"x": x}, starts, ends, plan=a.plan)
        ends[0] = 99
        with pytest.raises(ValueError):
            engine.execute(graph, {"x": x}, starts, ends)
        ends[0] = 16
        d = engine.execute(graph, {"x": x}, starts, ends, hard_stop=True, timeout=30)
        assert d.plan.lane == "process"
        e = engine.execute(graph, {"x": x}, starts, ends)
        assert e.plan.lane == "single" and e.values[0, 0] == a.values[0, 0]


@pytest.mark.parametrize("layout", ["C", "negative", "zero", "transpose", "step"])
@pytest.mark.parametrize("shape", [(0, 3, 5), (1, 3, 5), (7, 17, 39)])
@pytest.mark.parametrize("retained", [False, True])
def test_segmented_chain_keeps_external_consumers_and_liveness(layout, shape, retained):
    base = np.resize(np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0 - 2.0**-27, 2.0]), shape)
    x = {
        "C": base,
        "negative": base[:, ::-1, ::-1],
        "zero": np.broadcast_to(base[:, :, :1], shape),
        "transpose": base.transpose(2, 0, 1),
        "step": base[:, :, ::2],
    }[layout]
    x.flags.writeable = False
    types = {"x": schema(x.shape), "a": "scalar", "b": "scalar"}
    first, second = "x*a", "(x*a)+b"
    expr = f"(((({second})*3)-2)*4)+1"
    expressions = {"value": expr, "mask": f"{expr}>0", "duplicate": expr}
    if retained:
        expressions.update(first=first, external=f"{first}-7", second=second)
    graph = pickle.loads(pickle.dumps(GraphCompiler(types).compile(expressions)))
    params = np.array([1.0 + 2.0**-27, -1.0])
    with AdaptiveScheduler(cpu_budget=1) as engine:
        prepared = engine.prepare_execution(
            graph, {"x": x}, np.array([0], np.int64), np.array([1], np.int64), parameters=params
        )
        saved = []
        for a, b in [(1.0 + 2.0**-27, -1.0), (-0.5, 0.0)]:
            params[:] = a, b
            with np.errstate(invalid="ignore", over="ignore"):
                first_value = x * a
                second_value = first_value + b
                value = (((second_value * 3) - 2) * 4) + 1
                expected = [value, value > 0, value]
                if retained:
                    expected += [first_value, first_value - 7, second_value]
            result = prepared.run_snapshot()
            for output, wanted in zip(result.outputs, expected, strict=True):
                actual = output.values[0]
                np.testing.assert_array_equal(actual, wanted)
                if wanted.dtype == np.float64:
                    np.testing.assert_array_equal(
                        np.signbit(actual[wanted == 0]), np.signbit(wanted[wanted == 0])
                    )
                assert not np.shares_memory(actual, x)
            saved.append((result, expected))
            audit = result.audit["native_chunks"][0]
            assert audit["numeric_arena_bytes"] <= graph.metadata()["numeric_slots"] * 2048 * 8
            assert audit["input_copy_bytes"] == 0
        for result, expected in saved:
            for output, wanted in zip(result.outputs, expected, strict=True):
                np.testing.assert_array_equal(output.values[0], wanted)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_segmented_branch_fusion_transports_roots_and_shared_dependencies(lane):
    from calmetrics_engine import PlannerConfig

    x = np.linspace(-3, 4, 4097)
    y = x[::-1]
    types = {"x": schema(x.shape), "y": schema(y.shape)}
    lhs, rhs = "((x*2+1)*3)-2", "((y*4-3)*2)+1"
    graph = GraphCompiler(types).compile(
        {
            "sum": f"({lhs})+({rhs})",
            "shared": f"(({lhs})+({rhs}))*2+3",
            "mask": f"(({lhs})+({rhs}))*2+3>0",
        }
    )
    config = PlannerConfig(
        thread_work_units=1 if lane == "thread" else 1e100,
        process_work_units=1 if lane.startswith("process") else 1e100,
        min_rows_per_worker=1,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
    )
    with AdaptiveScheduler(cpu_budget=2 if lane != "single" else 1, config=config) as engine:
        result = engine.execute(
            graph, {"x": x, "y": y}, np.zeros(4, np.int64), np.ones(4, np.int64)
        )
    value = ((x * 2 + 1) * 3 - 2) + ((y * 4 - 3) * 2 + 1)
    for field, expected in zip(
        result.outputs, [value, value * 2 + 3, value * 2 + 3 > 0], strict=True
    ):
        for actual in field.values:
            np.testing.assert_array_equal(actual, expected)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)


@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_long_generic_chain_preserves_masks_scalar_expressions_and_branches(policy):
    x = np.linspace(-2, 3, 8193)
    y = x[::-1]
    scalar = np.array([0.125])
    types = {"x": schema(x.shape), "y": schema(y.shape), "a": "scalar"}
    expressions = {
        "value": "absolute(((x+y)*2+1)-y)*3+2",
        "mask": "absolute(((x+y)*2+1)-y)*3+2>=4",
        "scalar_expr": "(x*(a+1)+2)*3-1",
        "mask_branch": "x>0",
        "raw": "x",
    }
    graph = GraphCompiler(types).compile(expressions, error_policy=policy)
    expected = np.abs(((x + y) * 2 + 1) - y) * 3 + 2
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(
            graph,
            {"x": x, "y": y},
            np.array([0], np.int64),
            np.array([1], np.int64),
            parameters=scalar,
        )
    for field, wanted in zip(
        result.outputs,
        [expected, expected >= 4, (x * (scalar[0] + 1) + 2) * 3 - 1, x > 0, x],
        strict=True,
    ):
        np.testing.assert_array_equal(field.values[0], wanted)
