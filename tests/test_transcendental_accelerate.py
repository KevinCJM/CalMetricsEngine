"""Independent vForce roots preserve errors, logical axes and owned results."""

import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import settings
from .test_nonnegative_matmul_accelerate import ACCELERATE, active


def declaration(rank):
    if rank == 1:
        return "series"
    return dict(kind=["value", "vector", "matrix", "tensor"][rank],
                axes=["time"] + ["asset"] * (rank - 1),
                shape=["T"] + ["D", "K"][:rank - 1])


def payload(name, rank=2, order="C", n=1031):
    shape = (n, 3, 2)[:rank]
    x = np.random.default_rng(20261001).uniform(.01, 10, shape)
    if name == "exp":
        x -= 5
    x = np.array(x, order=order)
    x.flags.writeable = False
    return x


def graph(name, rank=2, order="auto", *, expression=None, isolate=False):
    return GraphCompiler({"x": declaration(rank)}).compile(
        expression or f"{name}(x)", output_order=order, result_format="typed",
        error_policy="isolate" if isolate else "raise",
    )


def calls(result):
    return sum(chunk["accelerate_vforce_calls"] for chunk in result.audit["native_chunks"])


def compare(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    for predicate in (np.isfinite, np.isnan, np.signbit, lambda y: y == 0):
        np.testing.assert_array_equal(predicate(actual), predicate(expected))


@active
@pytest.mark.parametrize("name", ["exp", "log"])
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("source_order", ["C", "F"])
@pytest.mark.parametrize("order", ["C", "F", "auto"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_graph_layout_transport_and_independent_ownership(name, rank, source_order, order, lane):
    x = payload(name, rank, source_order)
    before = x.copy(order="K")
    compiled = pickle.loads(pickle.dumps(graph(name, rank, order)))
    starts, ends = np.zeros(2, np.int64), np.full(2, len(x), np.int64)
    expected = op.call(name, x, simd="scalar")
    chosen = "F" if rank >= 2 and (order == "F" or order == "auto" and source_order == "F") else "C"
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        first = scheduler.execute(compiled, {"x": x}, starts, ends)
        second = scheduler.execute(compiled, {"x": x}, starts, ends)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    assert first.outputs[0].order == chosen
    eligible = rank == 1 or chosen == source_order
    assert calls(first) == (2 if eligible else 0)
    for row, value in enumerate(first.outputs[0].values):
        compare(value, expected)
        assert not value.flags.writeable
        assert value.flags.f_contiguous if chosen == "F" else value.flags.c_contiguous
        assert not np.shares_memory(value, x)
        assert not np.shares_memory(value, second.outputs[0].values[row])
    assert not np.shares_memory(*first.outputs[0].values)
    np.testing.assert_array_equal(x.view(np.uint64), before.view(np.uint64))
    for chunk in first.audit["native_chunks"]:
        assert chunk["root_copy_bytes"] == chunk["numeric_arena_bytes"] == 0
        assert chunk["algorithm_copy_bytes"] == chunk["input_copy_bytes"] == 0
        assert chunk["operator_workspace_capacity_bytes"] == 0


@pytest.mark.parametrize("name", ["exp", "log"])
@pytest.mark.parametrize("n", [0, 1, 63, 1023, 1024, 131073])
def test_direct_threshold_canaries_and_workspace(name, n):
    x = payload(name, 1, n=n)
    owner = np.full(n + 2, 912.)
    actual, audit = op.call(name, x, out=owner[1:-1], audit=True)
    compare(actual, op.call(name, x, simd="scalar"))
    assert owner[0] == owner[-1] == 912.
    assert audit["accelerate_vforce_calls"] == int(ACCELERATE and n >= 1024)
    assert audit["workspace_bytes"] == audit["input_copy_bytes"] == 0


@pytest.mark.parametrize("name,fault", [("exp", 701.), ("exp", 710.), ("exp", np.nan),
                                      ("exp", np.inf), ("log", 0.), ("log", -1.),
                                      ("log", np.nan), ("log", np.inf),
                                      ("log", np.nextafter(0., 1.))])
@pytest.mark.parametrize("position", [3, 1030])
def test_rejection_retains_exact_canonical_written_prefix(name, fault, position):
    x = payload(name, 1).copy()
    x[position] = fault
    x.flags.writeable = False
    outputs, exceptions = [], []
    for isa in ("scalar", "auto"):
        out = np.full(len(x), 912.)
        try:
            _, audit = op.call(name, x, out=out, simd=isa, audit=True)
            assert audit["accelerate_vforce_calls"] == 0
            exceptions.append(None)
        except ValueError as exc:
            exceptions.append(str(exc))
        outputs.append(out)
    assert exceptions[0] == exceptions[1]
    np.testing.assert_array_equal(outputs[0].view(np.uint64), outputs[1].view(np.uint64))


@active
@pytest.mark.parametrize("name", ["exp", "log"])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_partial_f_window_prepared_rereads_payload(name, rank, lane):
    x = payload(name, rank, "F", n=2059)
    starts, ends = np.array([3, 8], np.int64), np.array([1034, 1041], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        compiled = graph(name, rank)
        if lane == "single":
            prepared = scheduler.prepare_execution(compiled, {"x": x}, starts, ends)
            execute = prepared.run_snapshot
        else:
            def execute():
                return scheduler.execute(compiled, {"x": x}, starts, ends)
        first = execute()
        saved = [value.copy() for value in first.outputs[0].values]
        assert calls(first) == 2 * np.prod(x.shape[1:])
        for row, value in enumerate(first.outputs[0].values):
            compare(value, op.call(name, x[starts[row]:ends[row]], simd="scalar"))
        x.flags.writeable = True
        # Outside both windows: must not disable the eligible consumed payload.
        x[-1] = np.nan
        x.flags.writeable = False
        second = execute()
        assert calls(second) == calls(first)
        x.flags.writeable = True
        x[10] = .5
        x.flags.writeable = False
        third = execute()
        assert calls(third) == calls(first)
        for row, value in enumerate(first.outputs[0].values):
            np.testing.assert_array_equal(value, saved[row])
            compare(third.outputs[0].values[row], op.call(name, x[starts[row]:ends[row]], simd="scalar"))
            assert not np.shares_memory(value, third.outputs[0].values[row])


@pytest.mark.parametrize("name", ["exp", "log"])
@pytest.mark.parametrize("scope", ["consumer", "multiple", "isolate", "reverse"])
def test_unsafe_graph_contexts_keep_canonical_math(name, scope):
    x = payload(name, 2)
    expression = f"{name}(x)>0" if scope == "consumer" else f"{name}(x)"
    if scope == "multiple":
        expression = [expression, "x+1"]
    if scope == "reverse":
        x = x[::-1]
    compiled = graph(name, 2, expression=expression, isolate=scope == "isolate")
    starts, ends = np.zeros(1, np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(compiled, {"x": x}, starts, ends)
    assert calls(result) == 0
    expected = op.call(name, x, simd="scalar")
    if scope == "consumer":
        expected = expected > 0
    np.testing.assert_array_equal(result.outputs[0].values[0], expected)


@pytest.mark.parametrize("name", ["exp", "log"])
def test_direct_rejects_alias_before_library(name):
    x = payload(name, 1).copy()
    before = x.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIASES_INPUT"):
        op.call(name, x, out=x)
    np.testing.assert_array_equal(x, before)


@active
@pytest.mark.parametrize("name,pattern", [("exp", "range"), ("exp", "zeros"),
                                        ("log", "range"), ("log", "unit")])
def test_qualified_extremes_and_zero_classification(name, pattern):
    if pattern == "zeros":
        x = np.tile(np.array([-0., 0.]), 2048)
    elif pattern == "unit":
        one = np.array([1.]).view(np.uint64)[0]
        x = np.arange(one - 2048, one + 2049, dtype=np.uint64).view(np.float64)
    elif name == "exp":
        x = np.linspace(-700., 700., 4097)
    else:
        # Avoid geomspace's intermediate overflow when forming the fixture.
        x = np.exp2(np.linspace(-1022, 1023, 4097))
        x[-1] = np.finfo(float).max
    x.flags.writeable = False
    actual, audit = op.call(name, x, audit=True)
    compare(actual, op.call(name, x, simd="scalar"))
    assert audit["accelerate_vforce_calls"] == 1


@pytest.mark.parametrize("name", ["exp", "log"])
def test_padded_f_late_outlier_uses_only_consumed_columns(name):
    owner = np.full((1034, 3), np.nan, order="F")
    x = owner[:1031]
    x[:] = payload(name)
    x[-1, -1] = 701. if name == "exp" else np.nextafter(0., 1.)
    owner.flags.writeable = x.flags.writeable = False
    starts, ends = np.array([0], np.int64), np.array([len(x)], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph(name, order="F"), {"x": x}, starts, ends)
    assert calls(result) == (2 if ACCELERATE else 0)
    expected = op.call(name, x, simd="scalar")
    compare(result.outputs[0].values[0], expected)
    assert result.outputs[0].values[0][-1, -1].view(np.uint64) == expected[-1, -1].view(np.uint64)
