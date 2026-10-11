"""Private HMM last-row materialization preserves recurrence and ownership."""

import hashlib

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_typed_results import config


def fixture(n=90000, k=12, order="c"):
    rng = np.random.default_rng(583)
    data = {
        "em": rng.normal(size=(n, k)),
        "initial": rng.uniform(0.5, 1.0, k),
        "transition": rng.uniform(0.1, 1.0, (k, k)),
    }
    data["initial"] /= data["initial"].sum()
    data["transition"] /= data["transition"].sum(axis=1)[:, None]
    for name, a in data.items():
        if order == "f" and a.ndim == 2:
            a = np.asfortranarray(a)
        elif order == "reverse":
            rev = (slice(None, None, -1),) * a.ndim
            a = a[rev].copy()[rev]
        elif order == "zero" and name == "em" and n:
            a = np.broadcast_to(a[:1], a.shape)
        a.flags.writeable = False
        data[name] = a
    return data


def graph(data, continuation=0, *, control=False, public=False, isolate=False, start=None):
    n, k = data["em"].shape
    defs = {
        "em": {"kind": "matrix", "axes": ["time", "component"], "shape": [str(n), str(k)]},
        "initial": {"kind": "vector", "axes": ["component"], "shape": [str(k)]},
        "transition": {
            "kind": "matrix",
            "axes": ["component", "component"],
            "shape": [str(k), str(k)],
        },
    }
    begin = max(0, n - 1) if start is None else start
    fitted = f"hmm_filter(em,initial,transition,{continuation})"
    last = f"sum_axis(slice_axis(filtered,0,{begin},{n},1),0)"
    roots = {"last": last}
    if control:
        roots["first"] = f"sum_axis(slice_axis(filtered,0,0,{min(1,n)},1),0)"
    if public:
        roots["full"] = "filtered"
    return GraphCompiler(defs).compile(
        roots,
        root_bindings=[{"filtered": fitted}] * len(roots),
        result_format="typed",
        error_policy="isolate" if isolate else "raise",
    )


def run(data, **options):
    lane = options.pop("lane", "single")
    prepared = options.pop("prepared", False)
    g = graph(data, **options)
    starts = np.zeros(2, np.int64)
    ends = np.full(2, data["em"].shape[0], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as s:
        call = (
            s.prepare_execution(g, data, starts, ends).run_snapshot
            if prepared
            else lambda: s.execute(g, data, starts, ends)
        )
        plan = s.plan(g, data, starts, ends)
        first = call()
        assert plan.estimated_worker_scratch_bytes >= max(
            c["numeric_arena_bytes"] + c["operator_workspace_capacity_bytes"]
            for c in first.audit["native_chunks"]
        )
        saved = [x.copy() for x in first.outputs[0].values]
        second = call()
        for old, value, new in zip(
            first.outputs[0].values, saved, second.outputs[0].values, strict=True
        ):
            assert old.tobytes() == value.tobytes() == new.tobytes()
            assert not np.shares_memory(old, new)
    return first


@pytest.mark.parametrize("order", ["c", "f", "reverse", "zero"])
@pytest.mark.parametrize("continuation", [0, 1])
def test_tail_exact_reference_and_private_arena(order, continuation):
    data = fixture(order=order)
    before = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in data.items()}
    full = op.hmm_filter(*data.values(), continuation)
    control = run(data, continuation=continuation, control=True)
    actual = run(data, continuation=continuation)
    for value in actual.outputs[0].values:
        assert value.tobytes() == full[-1].tobytes() == control.outputs[0].values[0].tobytes()
    a = actual.audit["native_chunks"][0]
    c = control.audit["native_chunks"][0]
    assert a["numeric_arena_bytes"] <= 12 * 8
    assert c["numeric_arena_bytes"] >= full.nbytes
    assert a["algorithm_copy_bytes"] >= 12 * 8
    assert before == {
        name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in data.items()
    }
    assert all(not a.flags.writeable for a in data.values())


@pytest.mark.parametrize(
    "lane,prepared",
    [("thread", False), ("process_inline", False), ("process_shared", False), ("single", True)],
)
def test_tail_transport_and_snapshot(lane, prepared):
    data = fixture(order="reverse")
    expected = run(data, control=True)
    actual = run(data, lane=lane, prepared=prepared)
    for x, y in zip(actual.outputs[0].values, expected.outputs[0].values, strict=True):
        assert x.tobytes() == y.tobytes()
    assert all(c["numeric_arena_bytes"] <= 12 * 8 for c in actual.audit["native_chunks"])


@pytest.mark.parametrize(
    "kind", ["em_nan", "em_inf", "initial_negative", "transition_zero", "transition_inf"]
)
def test_tail_first_error_and_recovery(kind):
    good = fixture()
    data = {name: value.copy() for name, value in good.items()}
    if kind.startswith("em_"):
        data["em"][0, 2] = np.nan if kind == "em_nan" else np.inf
    elif kind == "initial_negative":
        data["initial"][0] = -1
    elif kind == "transition_zero":
        data["transition"][1] = 0
    else:
        data["transition"][0, 0] = np.inf
    errors = []
    for control in (False, True):
        with pytest.raises(Exception) as error:
            run(data, control=control)
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]
    actual = run(good)
    expected = run(good, control=True)
    assert actual.outputs[0].values[0].tobytes() == expected.outputs[0].values[0].tobytes()


@pytest.mark.parametrize(
    "options", [{"control": True}, {"public": True}, {"isolate": True}, {"start": 89998}]
)
def test_tail_keeps_shared_public_isolated_and_other_slices(options):
    data = fixture()
    actual = run(data, **options)
    full = op.hmm_filter(*data.values(), 0)
    start = options.get("start", 89999)
    expected = full[start:].sum(axis=0)
    assert actual.outputs[0].values[0].tobytes() == expected.tobytes()
    a = actual.audit["native_chunks"][0]
    if not options.get("public"):
        assert a["numeric_arena_bytes"] >= full.nbytes
    else:
        assert actual.outputs[1].values[0].tobytes() == full.tobytes()


@pytest.mark.parametrize("n,k", [(0, 12), (1, 1), (10, 3), (1048575, 1), (1048576, 1)])
def test_tail_empty_small_and_admission_boundary(n, k):
    data = fixture(n, k)
    actual = run(data)
    full = op.hmm_filter(*data.values(), 0)
    expected = full[-1] if n else np.zeros(k)
    assert actual.outputs[0].values[0].tobytes() == expected.tobytes()
    a = actual.audit["native_chunks"][0]
    if n * k >= 1048576:
        assert a["numeric_arena_bytes"] <= k * 8
    else:
        assert a["numeric_arena_bytes"] == full.nbytes


def test_tail_public_slice_and_multiple_consumers_keep_lifetime():
    data = fixture(order="f")
    n, k = data["em"].shape
    types = {
        "em": {"kind": "matrix", "axes": ["time", "component"], "shape": [str(n), str(k)]},
        "initial": {"kind": "vector", "axes": ["component"], "shape": [str(k)]},
        "transition": {
            "kind": "matrix",
            "axes": ["component", "component"],
            "shape": [str(k), str(k)],
        },
    }
    bindings = {
        "filtered": "hmm_filter(em,initial,transition)",
        "last": f"slice_axis(filtered,0,{n-1},{n},1)",
    }
    roots = {"last": "last", "sum": "sum_axis(last,0)", "scaled": "last*2"}
    g = GraphCompiler(types).compile(
        roots, root_bindings=[bindings] * len(roots), result_format="typed"
    )
    full = op.hmm_filter(*data.values())
    expected = [full[-1:], full[-1], full[-1:] * 2]
    with AdaptiveScheduler(cpu_budget=1) as s:
        starts, ends = np.array([0], np.int64), np.array([n], np.int64)
        first = s.execute(g, data, starts, ends)
        second = s.execute(g, data, starts, ends)
        for old, new, value in zip(first.outputs, second.outputs, expected, strict=True):
            assert old.values[0].tobytes() == new.values[0].tobytes() == value.tobytes()
            assert not np.shares_memory(old.values[0], new.values[0])
        assert first.audit["native_chunks"][0]["numeric_arena_bytes"] < full.nbytes


def test_tail_failed_request_restores_shared_scheduler_and_old_result():
    data = fixture()
    g = graph(data)
    starts, ends = np.array([0], np.int64), np.array([90000], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        old = s.execute(g, data, starts, ends)
        saved = old.outputs[0].values[0].copy()
        prepared = s.prepare_execution(g, data, starts, ends)
        for row in (0, 89999):
            bad = dict(data)
            bad["em"] = data["em"].copy()
            bad["em"][row, 0] = np.nan
            bad["em"].flags.writeable = False
            with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT"):
                s.execute(g, bad, starts, ends)
            recovered = s.execute(g, data, starts, ends)
            assert (
                recovered.outputs[0].values[0].tobytes()
                == old.outputs[0].values[0].tobytes()
                == saved.tobytes()
            )
            assert prepared.run_snapshot().outputs[0].values[0].tobytes() == saved.tobytes()
