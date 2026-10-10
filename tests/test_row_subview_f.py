"""Continuous F rows preserve mathematical order and independent ownership."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler

from . import latent_chain as chain
from .test_latent_workspace_loan import selected
from .test_row_subview import data, digest, exact, exercise, graph
from .test_row_subview_budget import original_budget, settings


@pytest.mark.parametrize(
    "range_kind", ["first", "middle", "last", "unused_bad", "swap", "duplicate"]
)
@pytest.mark.parametrize("output_order", ["C", "F"])
def test_f_ranges_preserve_canonical_values_and_snapshots(range_kind, output_order):
    x = data("F", 120001)
    n = len(x["raw"])
    nt = n * 3 // 4
    p = x["positions"].copy()
    if range_kind == "middle":
        p[:nt] += n // 8
    elif range_kind == "last":
        p[:nt] += n - nt
    elif range_kind == "unused_bad":
        p[nt + 2] = -100
    elif range_kind == "swap":
        p[4], p[5] = p[5], p[4]
    elif range_kind == "duplicate":
        p[5] = p[4]
    p.flags.writeable = False
    x["positions"] = p
    gain = range_kind not in ("swap", "duplicate")
    result = exercise(x, graph(x, order=output_order), gain)
    assert (
        result.outputs[0].values[0].flags.f_contiguous
        if output_order == "F"
        else result.outputs[0].values[0].flags.c_contiguous
    )


@pytest.mark.parametrize("layout", ["f_padded", "c_padded", "reverse", "overlap", "zero"])
def test_unsupported_backing_retains_materialization(layout):
    x = data("F", 120001)
    n = len(x["raw"])
    raw = x["raw"]
    if layout == "f_padded":
        z = np.empty((n + 7, 4), order="F")
        z[:n] = raw
        raw = z[:n]
    elif layout == "c_padded":
        z = np.empty((n, 7))
        z[:, :4] = raw
        raw = z[:, :4]
    elif layout == "reverse":
        raw = raw[::-1]
    elif layout == "overlap":
        z = np.random.default_rng(90).normal(size=n + 3 * 65)
        raw = np.lib.stride_tricks.as_strided(z, shape=(n, 4), strides=(8, 65 * 8), writeable=False)
    else:
        raw = np.broadcast_to(raw[:, 0:1], (n, 4))
    raw.flags.writeable = False
    x["raw"] = raw
    exercise(x, graph(x), False)


@pytest.mark.parametrize("bad", ["negative_index", "large_index", "nan", "overflow"])
def test_consumed_errors_match_canonical_and_recover(bad):
    x = data("F", 120001)
    g = graph(x)
    bounds = np.array([0], np.int64)
    ends = np.array([len(x["raw"])], np.int64)
    invalid = {k: np.array(v, order="F" if v.ndim == 2 else "C", copy=True) for k, v in x.items()}
    if bad.endswith("index"):
        invalid["positions"][10] = -1 if bad == "negative_index" else len(x["raw"])
    elif bad == "nan":
        invalid["raw"][10, 3] = np.nan
    else:
        invalid["raw"][10, 3] = np.finfo(np.float64).max
    for v in invalid.values():
        v.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as s:
        old = s.execute(g, x, bounds, ends)
        hashes = [v.values[0].tobytes() for v in old.outputs]
        prepared = s.prepare_execution(g, invalid, bounds, ends)
        with pytest.raises(Exception) as reference:
            prepared.run_snapshot()
        with pytest.raises(type(reference.value)) as actual:
            s.execute(g, invalid, bounds, ends)
        assert str(actual.value) == str(reference.value)
        new = s.execute(g, x, bounds, ends)
        exact(new, old)
        assert hashes == [v.values[0].tobytes() for v in old.outputs]


@pytest.mark.parametrize("extra,public", [(True, False), (False, True)])
def test_other_consumers_and_public_selected_storage_remain_independent(extra, public):
    x = data("F", 120001)
    r = exercise(x, graph(x, extra=extra, public=public))
    if public:
        assert not np.shares_memory(r.outputs[0].values[0], r.outputs[-1].values[0])


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_f_native_lanes_and_original_geometry_budget(lane):
    x = data("F", 120001)
    g = graph(x)
    rows = 1 if lane == "single" else 2
    starts = np.zeros(rows, np.int64)
    ends = np.full(rows, len(x["raw"]), np.int64)
    before = digest(x)
    with AdaptiveScheduler(cpu_budget=1) as s:
        canonical = s.prepare_execution(g, x, starts, ends).run_snapshot()
    with AdaptiveScheduler(cpu_budget=rows, config=settings(lane)) as s:
        plan = s.plan(g, x, starts, ends)
        budget = original_budget(g, plan)
        first = s.execute(g, x, starts, ends)
        snapshots = [v.values[0].tobytes() for v in first.outputs]
        second = s.execute(g, x, starts, ends, memory_budget_bytes=budget)
        exact(first, canonical)
        exact(second, canonical)
        assert snapshots == [v.values[0].tobytes() for v in first.outputs]
        for a, b in zip(first.outputs, second.outputs, strict=True):
            assert all(not np.shares_memory(u, v) for u, v in zip(a.values, b.values, strict=True))
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        assert second.plan.estimated_total_memory_bytes <= budget
    assert before == digest(x)


@pytest.mark.parametrize("d,k,streaming", [(8, 8, True), (4, 12, False)])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_complete_hmm_preserves_all_outputs_and_removes_f_copy(d, k, streaming, layout):
    x = chain.inputs(240000, d, k)
    x["raw"] = np.array(x["raw"], order=layout, copy=True)
    x["raw"].flags.writeable = False
    g, _ = chain.graph_for(x, k, 3, 17, "hmm")
    x = selected(x, g)
    starts = np.zeros(1, np.int64)
    ends = np.full(1, 240000, np.int64)
    before = digest(x)
    with AdaptiveScheduler(cpu_budget=1) as s:
        canonical = s.prepare_execution(g, x, starts, ends).run_snapshot()
        first = s.execute(g, x, starts, ends)
        saved = [v.values[0].tobytes() for v in first.outputs]
        second = s.execute(g, x, starts, ends)
        exact(first, canonical)
        exact(second, canonical)
    assert len(first.outputs) == 13 and saved == [v.values[0].tobytes() for v in first.outputs]
    for r in [first, second]:
        chunk = r.audit["native_chunks"][0]
        # Original three-history preservation bound applies to every model.
        beta_copies = 3 * len(x["raw"]) * 3 // 4 * k * 8
        assert chunk["algorithm_copy_bytes"] < beta_copies
        assert (
            chunk["operator_workspace_capacity_bytes"]
            < canonical.audit["native_chunks"][0]["operator_workspace_capacity_bytes"]
        )
        # Only the D8/K8 case had the identified F-layout copy regression.
        if d == 8 and k == 8:
            assert chunk["algorithm_copy_bytes"] < 8 * 1024 * 1024
        assert (chunk["gaussian_filter_streams"] > 0) == streaming
        assert all(
            not np.shares_memory(a.values[0], b.values[0])
            for a, b in zip(first.outputs, second.outputs, strict=True)
        )
    assert before == digest(x)
