"""Private Gaussian/filter replay: geometry, errors, ownership and transport."""

import hashlib
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_typed_results import config


def fixture(n=43691, d=4, k=3, order="c"):
    rng = np.random.default_rng(857)
    data = {
        "x": rng.normal(size=(n, d)),
        "means": rng.normal(size=(k, d)),
        "variances": rng.uniform(0.2, 2, (k, d)),
        "initial": rng.uniform(0.1, 1, k),
        "transition": rng.uniform(0.1, 1, (k, k)),
    }
    data["initial"] /= data["initial"].sum()
    data["transition"] /= data["transition"].sum(axis=1, keepdims=True)
    for name, a in data.items():
        if order == "f" and a.ndim == 2:
            a = np.asfortranarray(a)
        elif order == "reverse":
            rev = (slice(None, None, -1),) * a.ndim
            a = a[rev].copy()[rev]
        elif order == "padded" and a.ndim == 2:
            backing = np.full((a.shape[0], a.shape[1] * 2 + 1), -999.0)
            backing[:, : a.shape[1] * 2 : 2] = a
            a = backing[:, : a.shape[1] * 2 : 2]
        elif order == "zero" and name == "x" and n:
            a = np.broadcast_to(a[:1], a.shape)
        a.flags.writeable = False
        data[name] = a
    return data


def declarations(data):
    n, d = data["x"].shape
    k = data["means"].shape[0]
    return {
        "x": {"kind": "matrix", "axes": ["time", "factor"], "shape": [str(n), str(d)]},
        "means": {"kind": "matrix", "axes": ["component", "factor"], "shape": [str(k), str(d)]},
        "variances": {"kind": "matrix", "axes": ["component", "factor"], "shape": [str(k), str(d)]},
        "initial": {"kind": "vector", "axes": ["component"], "shape": [str(k)]},
        "transition": {
            "kind": "matrix",
            "axes": ["component", "component"],
            "shape": [str(k), str(k)],
        },
    }


def graph(
    data,
    continuation=0,
    *,
    control=False,
    public=False,
    isolate=False,
    start=None,
    interleave=False,
):
    n = len(data["x"])
    k = len(data["means"])
    defs = declarations(data)
    bindings = {
        "em": "gaussian_log_density(x,means,variances)",
        "filtered": f"hmm_filter(em,initial,transition,{continuation})",
    }
    if interleave:
        defs["initial_rows"] = {
            "kind": "matrix",
            "axes": ["time", "component"],
            "shape": [str(n), str(k)],
        }
        bindings["em"] = "gaussian_log_density(x+0,means,variances)"
        bindings["filtered"] = (
            f"hmm_filter(em,sum_axis(initial_rows+0,0),transition,{continuation})"
        )
    begin = max(0, n - 1) if start is None else start
    roots = {"last": f"sum_axis(slice_axis(filtered,0,{begin},{n},1),0)"}
    if control:
        roots["density"] = "em"
    if public:
        roots["history"] = "filtered"
    return GraphCompiler(defs).compile(
        roots,
        root_bindings=[bindings] * len(roots),
        result_format="typed",
        error_policy="isolate" if isolate else "raise",
    )


def run(data, lane="single", prepared=False, **options):
    g = graph(data, **options)
    starts, ends = np.zeros(2, np.int64), np.full(2, len(data["x"]), np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as s:
        plan = s.plan(g, data, starts, ends)
        call = (
            s.prepare_execution(g, data, starts, ends).run_snapshot
            if prepared
            else lambda: s.execute(g, data, starts, ends)
        )
        first = call()
        assert plan.estimated_worker_scratch_bytes >= max(
            c["numeric_arena_bytes"] + c["operator_workspace_capacity_bytes"]
            for c in first.audit["native_chunks"]
        )
        saved = [x.copy() for x in first.outputs[0].values]
        second = call()
        for old, copy, new in zip(
            first.outputs[0].values, saved, second.outputs[0].values, strict=True
        ):
            assert old.tobytes() == copy.tobytes() == new.tobytes()
            assert not np.shares_memory(old, new)
        return first, plan


@pytest.mark.parametrize("order", ["c", "f", "reverse", "padded", "zero"])
@pytest.mark.parametrize("continuation", [0, 1])
def test_exact_recurrence_and_removed_private_arrays(order, continuation):
    data = fixture(order=order)
    before = {key: hashlib.sha256(a.tobytes()).hexdigest() for key, a in data.items()}
    density = op.gaussian_log_density(data["x"], data["means"], data["variances"])
    full = op.hmm_filter(density, data["initial"], data["transition"], continuation)
    actual, plan = run(data, continuation=continuation)
    control, _ = run(data, continuation=continuation, control=True)
    for out in actual.outputs[0].values:
        assert out.tobytes() == full[-1].tobytes() == control.outputs[0].values[0].tobytes()
        assert not out.flags.writeable
    chunks = actual.audit["native_chunks"]
    assert sum(c["gaussian_filter_streams"] for c in chunks) == 2
    assert all(c["numeric_arena_bytes"] < density.nbytes for c in chunks)
    assert all(c["operator_workspace_capacity_bytes"] >= 131 * 3 * 8 for c in chunks)
    assert "gaussian_filter_stream_replay_work_accounted" in plan.reason_codes
    assert before == {key: hashlib.sha256(a.tobytes()).hexdigest() for key, a in data.items()}


@pytest.mark.parametrize(
    "lane,prepared",
    [("single", True), ("thread", False), ("process_inline", False), ("process_shared", False)],
)
def test_native_transport_and_prepared_snapshot(lane, prepared):
    data = fixture(order="reverse")
    actual, _ = run(data, lane=lane, prepared=prepared)
    control, _ = run(data, control=True)
    assert sum(c["gaussian_filter_streams"] for c in actual.audit["native_chunks"]) == 2
    for a, b in zip(actual.outputs[0].values, control.outputs[0].values, strict=True):
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize(
    "n,d,k,eligible",
    [
        (0, 4, 3, False),
        (1, 1, 1, False),
        (43690, 4, 3, False),
        (43691, 4, 3, True),
        (32767, 4, 4, False),
        (32768, 4, 4, True),
        (131072, 1, 1, True),
        (43691, 5, 3, True),
        (32768, 4, 5, True),
    ],
)
def test_geometry_thresholds(n, d, k, eligible):
    data = fixture(n, d, k)
    actual, plan = run(data)
    density = op.gaussian_log_density(data["x"], data["means"], data["variances"])
    full = op.hmm_filter(density, data["initial"], data["transition"])
    expected = full[-1] if n else np.zeros(k)
    assert actual.outputs[0].values[0].tobytes() == expected.tobytes()
    assert sum(c["gaussian_filter_streams"] for c in actual.audit["native_chunks"]) == (
        2 if eligible else 0
    )
    assert ("gaussian_filter_stream_replay_work_accounted" in plan.reason_codes) == eligible


@pytest.mark.parametrize(
    "options", [{"control": True}, {"public": True}, {"isolate": True}, {"start": 43689}]
)
def test_public_shared_isolated_and_other_slices_keep_normal_payload(options):
    data = fixture()
    actual, _ = run(data, **options)
    density = op.gaussian_log_density(data["x"], data["means"], data["variances"])
    full = op.hmm_filter(density, data["initial"], data["transition"])
    expected = op.sum_axis(full[options.get("start", 43690) :], 0)
    assert actual.outputs[0].values[0].tobytes() == expected.tobytes()
    assert sum(c["gaussian_filter_streams"] for c in actual.audit["native_chunks"]) == 0


@pytest.mark.parametrize(
    "kind",
    [
        "x_nan",
        "x_inf",
        "mean_nan",
        "variance_inf",
        "variance_negative",
        "initial_negative",
        "transition_zero",
        "transition_inf",
        "late_overflow_bad_initial",
        "late_overflow_bad_transition",
        "bad_shape",
    ],
)
def test_error_order_and_independent_validation(kind):
    good = fixture()
    data = {name: a.copy() for name, a in good.items()}
    if kind.startswith("x_"):
        data["x"][-1, -1] = np.nan if kind == "x_nan" else np.inf
    elif kind == "mean_nan":
        data["means"][0, 0] = np.nan
    elif kind == "variance_inf":
        data["variances"][0, 0] = np.inf
    elif kind == "variance_negative":
        data["variances"][0, 0] = -1
    elif kind == "initial_negative":
        data["initial"][0] = -1
    elif kind == "transition_zero":
        data["transition"][0] = 0
    elif kind == "transition_inf":
        data["transition"][0, 0] = np.inf
    elif kind == "bad_shape":
        data["transition"] = np.ones((3, 2))
    else:
        data["x"][-1, 0] = np.finfo(np.float64).max
        if kind.endswith("initial"):
            data["initial"][0] = -1
        else:
            data["transition"][0, 0] = -1
    errors = []
    for control in (False, True):
        with pytest.raises(Exception) as e:
            run(data, control=control)
        errors.append((type(e.value), str(e.value)))
    assert errors[0] == errors[1]
    if kind.startswith("late_overflow"):
        assert "NON_FINITE_RESULT" in errors[0][1]
    actual, _ = run(good)
    control, _ = run(good, control=True)
    assert actual.outputs[0].values[0].tobytes() == control.outputs[0].values[0].tobytes()


def test_intervening_arena_writer_declines_stale_backing_replay():
    data = fixture()
    data["initial_rows"] = np.tile(data["initial"] / len(data["x"]), (len(data["x"]), 1))
    data["initial_rows"].flags.writeable = False
    actual, _ = run(data, interleave=True)
    control, _ = run(data, interleave=True, control=True)
    assert actual.outputs[0].values[0].tobytes() == control.outputs[0].values[0].tobytes()
    assert sum(c["gaussian_filter_streams"] for c in actual.audit["native_chunks"]) == 0


@pytest.mark.parametrize("epoch", [24, 25, 26, 27, 28, 29])
def test_packaged_worker_rejects_old_request_before_payload_decode(epoch):
    import os

    import calmetrics_engine

    worker = Path(calmetrics_engine.__file__).with_name(
        "calmetrics_worker.exe" if os.name == "nt" else "calmetrics_worker"
    )
    # An impossible blob length would fail differently if decoding preceded the
    # protocol check. Two frames also verify that rejection keeps framing intact.
    payload = struct.pack("<QQ", 0x434D453300000000 | epoch, 2**64 - 1)
    frame = struct.pack("<Q", len(payload)) + payload
    completed = subprocess.run(
        [str(worker), "--stdio"], input=frame * 2, capture_output=True, timeout=10, check=True
    )
    cursor = 0
    for _ in range(2):
        (size,) = struct.unpack_from("<Q", completed.stdout, cursor)
        cursor += 8
        response = completed.stdout[cursor : cursor + size]
        cursor += size
        magic, failed, text_size = struct.unpack_from("<QQQ", response)
        assert magic == 0x434D45530000001D and failed == 1
        assert response[24 : 24 + text_size] == b"unsupported native worker request version"
        assert size == 24 + text_size
    assert cursor == len(completed.stdout)


def iteration_graph(data, *, isolate=False, nested=False, control=False):
    n = len(data["x"])
    density = "gaussian_log_density(x,means,variances)"
    tail = f"sum_axis(slice_axis(hmm_filter({density},initial,transition),0,{n - 1},{n},1),0)"
    body = f"sum(slice_axis({tail},0,0,1,1))"
    if control:
        body += f"+sum(sum_axis({density},0))*0"
    iteration = f"iterate({body},0,0,2)"
    if nested:
        iteration = f"iterate({iteration},0,0,2)"
    roots = {
        "value": iteration,
        "status": f"iteration_status({iteration})",
        "count": f"iteration_count({iteration})",
        "independent": "7",
    }
    return GraphCompiler(declarations(data)).compile(
        roots, result_format="typed", error_policy="isolate" if isolate else "raise"
    )


@pytest.mark.parametrize("lane", ["single", "process_inline", "process_shared"])
@pytest.mark.parametrize("isolate", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_iteration_subscope_streaming_and_retained_outputs(lane, isolate, nested):
    data = fixture()
    before = {key: a.tobytes() for key, a in data.items()}
    starts, ends = np.zeros(2, np.int64), np.full(2, len(data["x"]), np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        actual = scheduler.execute(
            iteration_graph(data, isolate=isolate, nested=nested), data, starts, ends
        )
        saved = [[a.copy() for a in root.values] for root in actual.outputs]
        control = scheduler.execute(
            iteration_graph(data, isolate=isolate, nested=nested, control=True), data, starts, ends
        )
    full = op.hmm_filter(
        op.gaussian_log_density(data["x"], data["means"], data["variances"]),
        data["initial"],
        data["transition"],
    )
    assert actual.named_outputs["value"].values[0].tobytes() == full[-1, :1].tobytes()
    assert sum(c["gaussian_filter_streams"] for c in actual.audit["native_chunks"]) == (
        8 if nested else 4
    )
    assert sum(c["gaussian_filter_streams"] for c in control.audit["native_chunks"]) == 0
    for root, reference, frozen in zip(actual.outputs, control.outputs, saved, strict=True):
        for a, b, old in zip(root.values, reference.values, frozen, strict=True):
            assert a.tobytes() == b.tobytes() == old.tobytes()
            assert not np.shares_memory(a, b)
    assert before == {key: a.tobytes() for key, a in data.items()}


@pytest.mark.parametrize("lane", ["single", "process_inline", "process_shared"])
@pytest.mark.parametrize("fault", ["late_overflow", "invalid_initial"])
def test_iteration_fault_replay_preserves_diagnostics_and_independent_root(lane, fault):
    data = {key: a.copy() for key, a in fixture().items()}
    if fault == "late_overflow":
        data["x"][-1, 0] = np.finfo(np.float64).max
    else:
        data["initial"][0] = -1
    starts, ends = np.zeros(2, np.int64), np.full(2, len(data["x"]), np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        actual = scheduler.execute(iteration_graph(data, isolate=True), data, starts, ends)
        control = scheduler.execute(
            iteration_graph(data, isolate=True, control=True), data, starts, ends
        )
    for root, reference in zip(actual.outputs, control.outputs, strict=True):
        assert root.root_statuses.tolist() == reference.root_statuses.tolist()
        for a, b in zip(root.values, reference.values, strict=True):
            assert a.tobytes() == b.tobytes()
    assert all(a == 5 for a in actual.named_outputs["status"].values)
    assert all(a == 1 for a in actual.named_outputs["count"].values)
    assert all(a == 7 for a in actual.named_outputs["independent"].values)
    assert not actual.named_outputs["independent"].root_statuses.any()


@pytest.mark.parametrize("lane", ["single", "thread", "process_shared"])
def test_streaming_scratch_is_admitted_before_execution(lane):
    data = fixture()
    g = graph(data)
    starts, ends = np.zeros(2, np.int64), np.full(2, len(data["x"]), np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        base = scheduler.plan(g, data, starts, ends)
        budget = base.estimated_total_memory_bytes
        plan = scheduler.plan(g, data, starts, ends, memory_budget_bytes=budget)
        assert plan.estimated_total_memory_bytes <= budget
        actual = scheduler.execute(g, data, starts, ends, plan=plan)
        assert all(
            c["numeric_arena_bytes"] + c["operator_workspace_capacity_bytes"]
            <= plan.estimated_worker_scratch_bytes
            for c in actual.audit["native_chunks"]
        )
        assert sum(c["gaussian_filter_streams"] for c in actual.audit["native_chunks"]) == 2
        # Binding input/output storage already exhausts this budget: no scratch
        # is available. Admission must reject instead of executing over budget.
        tiny = plan.estimated_input_bytes + plan.estimated_output_bytes + 32
        with pytest.raises(MemoryError):
            scheduler.execute(g, data, starts, ends, memory_budget_bytes=tiny)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("isolate", [False, True])
def test_nested_replay_work_matches_observed_extra_computation(nested, isolate):
    data = fixture()
    starts = np.zeros(2, np.int64)
    ends = np.full(2, len(data["x"]), np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        g = iteration_graph(data, nested=nested, isolate=isolate)
        control = iteration_graph(data, nested=nested, isolate=isolate, control=True)
        plan = scheduler.plan(g, data, starts, ends)
        original = scheduler.plan(control, data, starts, ends)
        actual = scheduler.execute(g, data, starts, ends)
        observed = scheduler.execute(control, data, starts, ends)
    # Observing density requires a small sum but avoids a second density pass.
    # Recomputed Gaussian rows cost more, also inside nested healthy scopes.
    assert plan.estimated_work_units > original.estimated_work_units
    assert sum(c["gaussian_filter_streams"] for c in actual.audit["native_chunks"]) > 0
    assert sum(c["gaussian_filter_streams"] for c in observed.audit["native_chunks"]) == 0
    for root, reference in zip(actual.outputs, observed.outputs, strict=True):
        for a, b in zip(root.values, reference.values, strict=True):
            assert a.tobytes() == b.tobytes()
            assert not np.shares_memory(a, b)
