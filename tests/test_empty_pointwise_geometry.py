"""Public empty-axis chains retain array rank through layout and worker planning.

These cases target the pre-fix capacity heuristic: scalar size=1 must not replace
an empty vector/matrix/tensor descriptor before sum_axis/transpose/affine. The
expected values follow empty-axis sums and zero-factor affine identity; no engine
operator or copied implementation supplies the oracle. Execution is required
to establish PASS; this file itself is only a regression specification.
"""

import gc

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, SharedInputBundle

from .test_typed_results import config

INITIAL = (5.0, -2.0, 0.125)


def declaration(kind, dtype="float64"):
    axes, shape = {
        "vector": (["component"], ["N"]),
        "matrix": (["time", "factor"], ["T", "F"]),
        "tensor": (["time", "path", "factor"], ["T", "P", "F"]),
    }[kind]
    result = dict(kind=kind, dtype=dtype, axes=axes, shape=shape)
    if dtype == "bool":
        result["semantic_dimension"] = "mask"
    return result


def model(kind, mode, order, policy, *, fault=False, only_empty_sum=False):
    shape = {"vector": (0,), "matrix": (6, 0), "tensor": (6, 4, 0)}[kind]
    source = "m" if mode == "where" else "x"
    dtype = "bool" if mode == "where" else "float64"
    variables = {source: declaration(kind, dtype)}
    inputs = {source: np.empty(shape, dtype=dtype, order=order)}
    if mode == "left":
        value = "add(7,multiply(2,x))"
        empty_divide = "divide(0," + value + ")"
    elif mode == "right":
        value = "add(multiply(x,2),7)"
        empty_divide = "divide(" + value + ",0)"
    else:
        value = "where(m,7,-2)"
        empty_divide = "divide(" + value + ",0)"
    predicate = "less_than(" + value + ",0)"
    axis = {"vector": 0, "matrix": 1, "tensor": 2}[kind]
    roots = {
        "empty": value,
        "empty_divide": empty_divide,
        "predicate": predicate,
        "axis_sum": "sum_axis(" + value + "," + str(axis) + ")",
        "total": "sum(" + value + ")",
        "predicate_count": "count_true(" + predicate + ")",
    }
    if kind == "matrix":
        roots["transposed"] = "transpose(" + value + ")"
    if kind != "vector":
        variables.update(
            k=dict(kind="matrix", dtype="float64", axes=["factor", "asset"], shape=["F", "A"]),
            initial=dict(kind="vector", dtype="float64", axes=["asset"], shape=["A"]),
        )
        inputs.update(k=np.empty((0, 3), dtype=np.float64, order=order),
                      initial=np.array(INITIAL, dtype=np.float64))
        projection = "affine_contract(" + value + ",k,initial)"
        roots["projection"] = projection
        roots["projected_sum"] = "sum_axis(" + projection + "," + str(axis) + ")"
    if fault:
        # Dependencies remain eager even when the selected mask has no elements.
        # Failure of one scalar branch must not corrupt independent empty roots.
        assert mode == "where" and kind != "vector"
        bad = "divide(sum(initial),0)"
        roots["bad_scalar"] = bad
        roots["bad_where"] = "where(m," + bad + ",0)"
    if only_empty_sum:
        roots = {"total": roots["total"]}
    elif policy == "raise":
        # Canonical sum rejects empty samples. A healthy strict-mode chain must
        # be a separate request from this independently failing scalar root.
        del roots["total"]
    for value in inputs.values():
        value.flags.writeable = False
    graph = GraphCompiler(variables).compile(roots, result_format="typed",
                                             error_policy=policy, output_order=order)
    return graph, inputs


def expected(kind, length):
    shape = {"vector": (0,), "matrix": (length, 0), "tensor": (length, 4, 0)}[kind]
    prefix = shape[:-1]
    result = {
        "empty": np.empty(shape, np.float64),
        "empty_divide": np.empty(shape, np.float64),
        "predicate": np.empty(shape, np.bool_),
        "axis_sum": np.zeros(prefix, np.float64),
        "predicate_count": np.asarray(0.0),
    }
    if kind == "matrix":
        result["transposed"] = np.empty((0, length), np.float64)
    if kind != "vector":
        # No factors contribute. Each row/path has exactly the supplied initial
        # value; the sum 5 - 2 + 1/8 = 25/8 is exactly representable in binary.
        result["projection"] = np.broadcast_to(np.asarray(INITIAL), prefix + (3,))
        result["projected_sum"] = np.full(prefix, 3.125, np.float64)
    return result


def intervals(kind):
    return ((0, 0), (0, 0)) if kind == "vector" else ((0, 3), (3, 6))


def check_healthy(result, kind, policy, order):
    starts, ends = intervals(kind)
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for name, wanted in expected(kind, end - start).items():
            output = result.named_outputs[name]
            value = output.values[row]
            assert value.shape == wanted.shape, name
            assert value.dtype == wanted.dtype, name
            assert output.shape_known[row], name
            assert output.root_statuses[row] == 0, name
            assert not value.flags.writeable, name
            np.testing.assert_array_equal(value, wanted, err_msg=name)
            if value.ndim >= 2 and value.size:
                assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
            if policy == "isolate":
                np.testing.assert_array_equal(output.statuses[row], np.zeros(wanted.shape, np.int16))
            else:
                assert output.statuses[row] is None
    assert result.audit["python_operator_calls"] == 0
    assert not result.audit["python_fallback"]
    assert result.audit["result_lifetime"] == "independent"
    if policy == "isolate":
        output = result.named_outputs["total"]
        for row in range(len(starts)):
            value = output.values[row]
            assert value.shape == () and value.dtype == np.float64
            assert output.shape_known[row]
            assert output.root_statuses[row] == 4  # INSUFFICIENT_SAMPLE
            assert np.isnan(value) and not value.flags.writeable
            np.testing.assert_array_equal(output.statuses[row], np.asarray(4, np.int16))


def exercise(kind, mode, lane, order, policy):
    graph, inputs = model(kind, mode, order, policy)
    failure_graph = (model(kind, mode, order, policy, only_empty_sum=True)[0]
                     if policy == "raise" else None)
    originals = {name: value.copy(order="K") for name, value in inputs.items()}
    first = second = None
    bundle = SharedInputBundle.from_inputs(inputs) if lane == "process_shared" else None
    transported = bundle.arrays if bundle is not None else inputs
    starts, ends = intervals(kind)
    starts, ends = np.asarray(starts, np.int64), np.asarray(ends, np.int64)
    try:
        with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
            first = scheduler.execute(graph, transported, starts, ends, timeout=20)
            saved = {name: [value.copy(order="K") for value in output.values]
                     for name, output in first.named_outputs.items()}
            second = scheduler.execute(graph, transported, starts, ends, timeout=20)
            for result in (first, second):
                check_healthy(result, kind, policy, order)
                assert result.plan.lane == ("process" if lane.startswith("process") else lane)
                assert result.audit["cpu_tokens"] <= 2
                if lane.startswith("process"):
                    assert result.audit["native_processes"] > 0
                    assert result.plan.use_shared_memory == (lane == "process_shared")
                    assert result.audit["python_worker_callbacks"] == 0
                if lane == "process_shared":
                    assert result.audit["output_copy_bytes"] == 0
            for name, output in first.named_outputs.items():
                for row, value in enumerate(output.values):
                    np.testing.assert_array_equal(value, saved[name][row], err_msg=name)
                    assert not np.shares_memory(value, second.named_outputs[name].values[row])
                    for source in transported.values():
                        assert not np.shares_memory(value, source)
            # Nonempty scalar reductions and matrix/tensor projections make the
            # owner checks meaningful even when other roots contain zero bytes.
            assert first.named_outputs["predicate_count"].values[0].size == 1
            if kind != "vector":
                assert first.named_outputs["projection"].values[0].size > 0
            if failure_graph is not None:
                with pytest.raises((ValueError, RuntimeError), match="INSUFFICIENT_SAMPLE"):
                    scheduler.execute(failure_graph, transported, starts, ends, timeout=20)
    finally:
        if bundle is not None:
            bundle.release()
    for name, original in originals.items():
        np.testing.assert_array_equal(inputs[name], original)
        assert not inputs[name].flags.writeable
    del graph, failure_graph, inputs, transported, second, bundle
    gc.collect()
    for name, output in first.named_outputs.items():
        for row, value in enumerate(output.values):
            np.testing.assert_array_equal(value, saved[name][row], err_msg=name)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_empty_vector_scalar_broadcast_then_axis_sum(side, order, policy):
    exercise("vector", side, "single", order, policy)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_empty_vector_boolean_where_keeps_empty_numeric_shape(order, policy):
    exercise("vector", "where", "single", order, policy)


@pytest.mark.parametrize("kind", ["matrix", "tensor"])
@pytest.mark.parametrize("mode", ["left", "right", "where"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_empty_pointwise_followed_by_axis_affine_and_native_transport(kind, mode, order, policy, lane):
    exercise(kind, mode, lane, order, policy)


@pytest.mark.parametrize("kind", ["matrix", "tensor"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
@pytest.mark.parametrize("bad", ["dtype", "shape"])
def test_empty_chain_does_not_bypass_input_dtype_or_symbolic_shape_checks(kind, policy, bad):
    graph, inputs = model(kind, "right", "C", policy)
    if bad == "dtype":
        changed = np.empty(inputs["x"].shape, np.float32)
        message = "dtype|float64"
    else:
        shape = list(inputs["x"].shape)
        shape[-1] = 1  # k still fixes F=0; this input cannot satisfy that symbol.
        changed = np.ones(shape, np.float64)
        message = "symbolic|shape|SHAPE"
    changed.flags.writeable = False
    inputs["x"] = changed
    starts, ends = intervals(kind)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises((TypeError, ValueError), match=message):
            scheduler.execute(graph, inputs, np.asarray(starts, np.int64), np.asarray(ends, np.int64))


@pytest.mark.parametrize("kind", ["matrix", "tensor"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_empty_where_preserves_eager_branch_error_and_independent_healthy_roots(kind, policy):
    graph, inputs = model(kind, "where", "C", policy, fault=True)
    starts, ends = intervals(kind)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if policy == "raise":
            with pytest.raises((ValueError, RuntimeError), match="DIVIDE_BY_ZERO"):
                scheduler.execute(graph, inputs, np.asarray(starts, np.int64), np.asarray(ends, np.int64))
        else:
            result = scheduler.execute(graph, inputs, np.asarray(starts, np.int64), np.asarray(ends, np.int64))
            check_healthy(result, kind, policy, "C")
            for name in ("bad_scalar", "bad_where"):
                assert np.all(result.named_outputs[name].root_statuses != 0)
            assert np.isnan(result.named_outputs["bad_scalar"].values[0])
