"""Column gathering preserves scatter values/decisions without a mapped matrix."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_typed_results import config


def fixture(order="c", n=13, d=7, m=5):
    rng = np.random.default_rng(517)
    x = rng.normal(size=(n, d))
    if d >= 3:
        x[:, 0] = 2.0
        x[:, 1] = 2.0
        x[:, 2] = -0.0
    if order == "f":
        x = np.asfortranarray(x)
    elif order == "reverse":
        x = x[::-1, ::-1].copy()[::-1, ::-1]
    elif order == "zero":
        x = np.broadcast_to(x[:1], x.shape)
    cols = np.array([3, 0, 1, 0, 2], np.int64) if d >= 4 else np.arange(d, dtype=np.int64)
    score = rng.normal(size=d)
    rows = np.array([7, 1, 5, 3, 11], np.int64)[:m]
    selected = np.arange(m, dtype=np.int64)[::-1].copy()
    data = dict(x=x, cols=cols, score=score, ix=rows, selected=selected)
    for a in data.values():
        a.flags.writeable = False
    return data


def graph(
    data,
    *,
    maximum=False,
    private=False,
    computed=False,
    control=False,
    rows=19,
    fill="0",
    axis=1,
    isolate=False,
    output_order="C",
):
    definitions = {
        n: dict(
            kind={1: "vector", 2: "matrix", 3: "tensor"}[a.ndim],
            axes=["asset", "factor", "scenario"][: a.ndim],
            shape=[str(i) for i in a.shape],
            dtype=str(a.dtype),
        )
        for n, a in data.items()
    }
    source = "take_axis(x,selected,0)" if private else "slice_axis(x,0,0,5,1)"
    columns = "argsort(score)" if computed else "cols"
    mapped = f"take_axis({source},{columns},{axis})"
    scatter = f"scatter_rows({mapped},ix,{rows},{fill})"
    result = f"argmax_axis({scatter},1)" if maximum else scatter
    # Put scalar options first, as in the complete platform chain. Control
    # exposes the mapped matrix so it must be materialized independently.
    expressions = [str(rows), fill, result] + ([mapped] if control else [])
    return GraphCompiler(definitions).compile(
        expressions,
        result_format="typed",
        error_policy="isolate" if isolate else "raise",
        output_order=output_order,
    )


def run(g, data, lane="single", prepared=False):
    starts = np.zeros(2, np.int64)
    ends = starts.copy()
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        execute = (
            scheduler.prepare_execution(g, data, starts, ends).run_snapshot
            if prepared
            else lambda: scheduler.execute(g, data, starts, ends)
        )
        a = execute()
        saved = [v.copy() for o in a.outputs for v in o.values]
        b = execute()
        assert scheduler.peak_active_cpu_tokens <= 2
        for old, copy, new in zip(
            [v for o in a.outputs for v in o.values],
            saved,
            [v for o in b.outputs for v in o.values],
            strict=True,
        ):
            np.testing.assert_array_equal(old, copy)
            assert not old.flags.writeable
            assert not np.shares_memory(old, new)
            assert not any(np.shares_memory(old, x) for x in data.values())
        return a


@pytest.mark.parametrize("order", ["c", "f", "reverse", "zero"])
@pytest.mark.parametrize("maximum", [False, True])
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("computed", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_exact_values_decisions_and_owned_generations(order, maximum, private, computed, prepared):
    data = fixture(order)
    actual = run(
        graph(data, maximum=maximum, private=private, computed=computed), data, prepared=prepared
    )
    expected = run(
        graph(data, maximum=maximum, private=private, computed=computed, control=True),
        data,
        prepared=prepared,
    )
    for a, b in zip(actual.outputs[2].values, expected.outputs[2].values, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    chunks = actual.audit["native_chunks"]
    assert sum(c["fused_pointwise_calls"] for c in chunks) > 0
    # A control root writes its mapped matrix outside the arena. Comparing
    # arenas alone with that control would omit its independent result bytes.
    # The computed-index/argmax case can reuse the dead column-index slot and
    # deliberately retains canonical take execution instead of aliasing it.
    if not computed or not maximum:
        limit = (5 * 7 * 8 if private else 0) + (7 * 8 if computed else 0)
        assert max(c["numeric_arena_bytes"] for c in chunks) <= limit


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("maximum", [False, True])
def test_transport_exact_typed_results(lane, maximum):
    data = fixture("reverse")
    actual = run(graph(data, maximum=maximum, private=True, computed=True), data, lane)
    expected = run(
        graph(data, maximum=maximum, private=True, computed=True, control=True), data, lane
    )
    for a, b in zip(actual.outputs[2].values, expected.outputs[2].values, strict=True):
        assert a.tobytes() == b.tobytes()
    assert actual.audit["lane"] == ("process" if lane.startswith("process") else lane)


@pytest.mark.parametrize(
    "fault",
    ["column_negative", "column_upper", "row_negative", "row_upper", "row_duplicate", "nonfinite"],
)
@pytest.mark.parametrize("maximum", [False, True])
def test_original_faults_and_unselected_values(fault, maximum):
    data = {k: v.copy() for k, v in fixture().items()}
    if fault == "column_negative":
        data["cols"][2] = -1
    elif fault == "column_upper":
        data["cols"][2] = data["x"].shape[1]
    elif fault == "row_negative":
        data["ix"][2] = -1
    elif fault == "row_upper":
        data["ix"][2] = 19
    elif fault == "row_duplicate":
        data["ix"][2] = data["ix"][0]
    elif fault == "nonfinite":
        data["x"][2, 0] = np.nan
    exceptions = []
    results = []
    for control in [False, True]:
        g = graph(data, maximum=maximum, private=True, control=control)
        try:
            results.append(run(g, data))
        except Exception as e:
            exceptions.append((type(e), str(e)))
    if exceptions:
        assert len(exceptions) == 2 and exceptions[0] == exceptions[1]
    else:
        assert fault == "nonfinite" and not maximum
        np.testing.assert_array_equal(
            results[0].outputs[2].values[0], results[1].outputs[2].values[0]
        )


@pytest.mark.parametrize("maximum", [False, True])
def test_only_selected_columns_are_consumed(maximum):
    data = {k: v.copy() for k, v in fixture().items()}
    data["x"][:, 6] = np.nan
    actual = run(graph(data, maximum=maximum, private=True), data)
    expected = run(graph(data, maximum=maximum, private=True, control=True), data)
    np.testing.assert_array_equal(actual.outputs[2].values[0], expected.outputs[2].values[0])


@pytest.mark.parametrize("maximum", [False, True])
@pytest.mark.parametrize("shape", [(5, 0), (5, 1), (0, 7)])
def test_empty_short_shapes(shape, maximum):
    data = fixture(n=max(5, shape[0]), d=shape[1])
    data["x"] = data["x"][: shape[0]]
    if not shape[0]:
        data["ix"] = np.empty(0, np.int64)
        data["selected"] = np.empty(0, np.int64)
    # Avoid a five-row source slice for the genuinely empty case.
    g = graph(data, maximum=maximum, private=True)
    control = graph(data, maximum=maximum, private=True, control=True)
    errors = []
    results = []
    for candidate in [g, control]:
        try:
            results.append(run(candidate, data))
        except Exception as e:
            errors.append((type(e), str(e)))
    if errors:
        assert len(errors) == 2 and errors[0] == errors[1]
    else:
        np.testing.assert_array_equal(
            results[0].outputs[2].values[0], results[1].outputs[2].values[0]
        )


def test_isolation_retains_canonical_graph_and_healthy_output():
    data = {k: v.copy() for k, v in fixture().items()}
    data["x"][2, 0] = np.nan
    result = run(graph(data, maximum=True, private=True, isolate=True), data)
    assert result.outputs[0].values[0] == 19
    assert result.outputs[2].root_statuses[0] != 0
    assert sum(c["fused_pointwise_calls"] for c in result.audit["native_chunks"]) == 0


@pytest.mark.parametrize("maximum", [False, True])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("axis", [0, 1])
def test_nonmatrix_and_other_axis_fallback(rank, axis, maximum):
    data = fixture()
    if rank == 3:
        data["x"] = np.stack([data["x"], data["x"] + 0.5], axis=2)
    actual = run(graph(data, maximum=maximum, private=True, axis=axis), data)
    expected = run(graph(data, maximum=maximum, private=True, axis=axis, control=True), data)
    np.testing.assert_array_equal(actual.outputs[2].values[0], expected.outputs[2].values[0])
    if rank == 3:
        assert sum(c["fused_pointwise_calls"] for c in actual.audit["native_chunks"]) == 0


@pytest.mark.parametrize("maximum", [False, True])
def test_intervening_scalar_operation_keeps_canonical_take(maximum):
    data = fixture()
    actual = run(graph(data, private=True, maximum=maximum, fill="-2"), data)
    expected = run(graph(data, private=True, maximum=maximum, fill="-2", control=True), data)
    np.testing.assert_array_equal(actual.outputs[2].values[0], expected.outputs[2].values[0])


@pytest.mark.parametrize("maximum", [False, True])
def test_prepared_rereads_computed_order_without_overwriting_results(maximum):
    data = {k: v.copy() for k, v in fixture().items()}
    g = graph(data, private=True, computed=True, maximum=maximum)
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(g, data, bounds, bounds)
        old = prepared.run_snapshot().outputs[2].values[0]
        saved = old.copy()
        data["score"] *= -1
        data["selected"][:] = data["selected"][::-1]
        data["x"] += np.arange(data["x"].shape[0])[:, None] * 0.03
        changed = prepared.run_snapshot().outputs[2].values[0]
        source = data["x"][data["selected"]][:, np.argsort(data["score"], kind="stable")]
        reference = np.zeros((19, source.shape[1]))
        reference[data["ix"]] = source
        if maximum:
            reference = np.argmax(reference, axis=1)
        np.testing.assert_array_equal(changed, reference)
        np.testing.assert_array_equal(old, saved)
        assert not np.shares_memory(old, changed)


@pytest.mark.parametrize("maximum", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
def test_explicit_independent_root_layout(order, maximum):
    data = fixture("f")
    actual = run(graph(data, private=True, maximum=maximum, output_order=order), data)
    expected = run(
        graph(data, private=True, maximum=maximum, output_order=order, control=True), data
    )
    result = actual.outputs[2].values[0]
    np.testing.assert_array_equal(result, expected.outputs[2].values[0])
    assert result.flags.f_contiguous if order == "F" else result.flags.c_contiguous
