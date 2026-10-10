"""Per-call row proofs, canonical errors and independent result ownership."""

import hashlib

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler


def data(order="C", n=300000):
    rng = np.random.default_rng(7391)
    x = rng.normal(size=(n, 4))
    if order == "F":
        x = np.asfortranarray(x)
    if order == "reverse":
        x = x[::-1].copy()[::-1]
    out = {"raw": x, "positions": np.arange(n, dtype=np.int64)}
    for v in out.values():
        v.flags.writeable = False
    return out


def graph(x, policy="raise", extra=False, public=False, order="C"):
    n = len(x["raw"])
    nt = n * 3 // 4
    types = {
        "raw": {"kind": "matrix", "axes": ["time", "factor"], "shape": [str(n), "4"]},
        "positions": {
            "kind": "series",
            "axes": ["time"],
            "shape": [str(n)],
            "dtype": "int64",
            "semantic_dimension": "index",
        },
    }
    b = {
        "idx": f"slice_axis(positions,0,0,{nt},1)",
        "train": "take_axis(raw,idx,0)",
        "m": "mean_axis(train,0)",
        "s": "maximum(sqrt(variance_axis(train,0,0)),1e-10)",
        "full": "(raw-broadcast_like(m,raw))/broadcast_like(s,raw)",
        "centered": "train-broadcast_like(m,train)",
        "selected": "centered/broadcast_like(s,train)",
    }
    roots = {"full": "full", "summary": "sum_axis(selected,0)"}
    if extra:
        roots["centered"] = "centered"
    if public:
        roots["selected"] = "selected"
    return GraphCompiler(types).compile(
        roots,
        root_bindings=[b] * len(roots),
        result_format="typed",
        error_policy=policy,
        output_order=order,
    )


def digest(x):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in x.items()}


def exact(a, b):
    for x, y in zip(a.outputs, b.outputs, strict=True):
        np.testing.assert_array_equal(x.root_statuses, y.root_statuses)
        for u, v in zip(x.values, y.values, strict=True):
            assert u.shape == v.shape and u.dtype == v.dtype
            np.testing.assert_array_equal(
                u.reshape(-1).view(np.uint8), v.reshape(-1).view(np.uint8)
            )


def exercise(x, g, gain=None):
    starts = np.array([0], np.int64)
    ends = np.array([len(x["raw"])], np.int64)
    inputs = {n: x[n] for n in g.input_names}
    before = digest(x)
    with AdaptiveScheduler(cpu_budget=1) as s:
        control = s.prepare_execution(g, inputs, starts, ends).run_snapshot()
        one = s.execute(g, inputs, starts, ends)
        hashes = [digest({str(i): v for i, v in enumerate(o.values)}) for o in one.outputs]
        two = s.execute(g, inputs, starts, ends)
        exact(one, control)
        exact(two, control)
        for a, b in zip(one.outputs, two.outputs, strict=True):
            for u, v in zip(a.values, b.values, strict=True):
                assert not u.flags.writeable and not np.shares_memory(u, v)
                assert all(not np.shares_memory(u, z) for z in x.values())
        assert hashes == [digest({str(i): v for i, v in enumerate(o.values)}) for o in one.outputs]
        actual = max(c["numeric_arena_bytes"] for c in one.audit["native_chunks"])
        original = max(c["numeric_arena_bytes"] for c in control.audit["native_chunks"])
        if gain is True:
            assert actual < original - 2 * 1024**2
        if gain is False:
            assert actual == original
    exact(one, control)
    assert before == digest(x)
    return one


def test_dense_joint_reuse_and_old_outputs():
    x = data()
    exercise(x, graph(x), True)


@pytest.mark.parametrize("order,gain", [("F", True), ("reverse", False)])
def test_input_layout_keeps_canonical_math_and_ownership(order, gain):
    x = data(order)
    exercise(x, graph(x), gain)


def test_explicit_f_root_remains_f_with_canonical_selected_math():
    x = data()
    r = exercise(x, graph(x, order="F"))
    assert r.outputs[0].values[0].flags.f_contiguous


@pytest.mark.parametrize("mode", ["shift", "swap", "duplicate", "unused_bad"])
def test_each_call_proves_current_index_contents(mode):
    x = data()
    n = len(x["raw"])
    nt = n * 3 // 4
    positions = x["positions"].copy()
    if mode == "shift":
        positions[:nt] += 1
    if mode == "swap":
        positions[4], positions[5] = positions[5], positions[4]
    if mode == "duplicate":
        positions[5] = positions[4]
    if mode == "unused_bad":
        positions[nt + 2] = -100
    positions.flags.writeable = False
    x["positions"] = positions
    exercise(x, graph(x), True if mode in ("shift", "unused_bad") else False)


@pytest.mark.parametrize("bad", [-1, 300000])
def test_invalid_index_same_exception_and_recovery(bad):
    x = data()
    g = graph(x)
    starts = np.array([0], np.int64)
    ends = np.array([len(x["raw"])], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as s:
        old = s.execute(g, x, starts, ends)
        old_hash = [digest({str(i): v for i, v in enumerate(o.values)}) for o in old.outputs]
        invalid = dict(x)
        p = x["positions"].copy()
        p[10] = bad
        p.flags.writeable = False
        invalid["positions"] = p
        prepared = s.prepare_execution(g, invalid, starts, ends)
        with pytest.raises(Exception) as canonical:
            prepared.run_snapshot()
        with pytest.raises(type(canonical.value)) as actual:
            s.execute(g, invalid, starts, ends)
        assert str(actual.value) == str(canonical.value)
        recovered = s.execute(g, x, starts, ends)
        exact(recovered, old)
        assert old_hash == [
            digest({str(i): v for i, v in enumerate(o.values)}) for o in old.outputs
        ]


def test_shared_subtraction_output_is_not_elided():
    x = data()
    exercise(x, graph(x, extra=True))


def test_public_selected_matrix_retains_own_storage():
    x = data()
    r = exercise(x, graph(x, public=True))
    assert not np.shares_memory(r.outputs[0].values[0], r.outputs[-1].values[0])


def test_isolate_keeps_original_arena_and_statuses():
    x = data()
    exercise(x, graph(x, policy="isolate"), False)


def test_small_extent_uses_canonical():
    x = data(n=3000)
    exercise(x, graph(x), False)
