"""SIMD lanes are independent row states, preserving each chronological fold."""

import numpy as np
import pytest

from .test_row_batch_execution import Y
from .test_scan_reduce_fusion import execute, ieee_equal


def expression(seed=1, peak_seed=1, constant=1):
    nav = f"cumulative_product_axis(add(1,{Y}),1,{seed!r})"
    peak = f"cumulative_max_axis({nav},1,{peak_seed!r})"
    return f"max_axis(subtract({constant!r},divide({nav},{peak})),1)", nav


@pytest.mark.parametrize("rows", [128, 129, 193])
@pytest.mark.parametrize("length", [127, 128, 129, 252, 257, 2520])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "padded"])
def test_packet_scan_keep_exact_fold_inputs_tail_and_old_owners(rows, length, layout):
    rng = np.random.default_rng(919)
    w = rng.uniform(0.1, 1.0, (rows, 7))
    w /= w.sum(axis=1, keepdims=True)
    r = rng.normal(scale=0.01, size=(length, 7))
    if layout == "F":
        w = np.array(w, order="F")
        r = np.array(r, order="F")
    elif layout == "negative":
        w = w[::-1, ::-1]
        r = r[::-1, ::-1]
    elif layout == "padded":
        a = np.full((rows, 14), np.nan)
        a[:, ::2] = w
        w = a[:, ::2]
        b = np.full((length, 14), np.nan)
        b[:, ::2] = r
        r = b[:, ::2]
    data = dict(w=w, r=r, p=np.array([0.05]))
    saved = {n: x.copy() for n, x in data.items()}
    for x in data.values():
        x.flags.writeable = False
    root, nav = expression()
    first = execute([root], data, length)
    reference = execute([root, nav], data, length)
    second = execute([root], data, length)
    ieee_equal(first.outputs[0].values[0], reference.outputs[0].values[0])
    ieee_equal(first.outputs[0].values[0], second.outputs[0].values[0])
    assert not first.outputs[0].values[0].flags.writeable
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
    assert all(not np.shares_memory(first.outputs[0].values[0], x) for x in data.values())
    observed = max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] + c["operator_workspace_capacity_bytes"]
        for c in first.audit["native_chunks"]
    )
    assert first.plan.metadata()["estimated_worker_scratch_bytes"] >= observed
    for n, x in data.items():
        np.testing.assert_array_equal(x.view(np.uint64), saved[n].view(np.uint64))


@pytest.mark.parametrize(
    "seed,peak_seed", [(0, 0), (1, 0), (-1, 0), (-0.0, -0.0), (-1, -1), (2, 1)]
)
@pytest.mark.parametrize("mode", ["zeros", "underflow", "overflow", "mixed"])
def test_packet_scan_preserves_zero_exception_and_original_error_order(seed, peak_seed, mode):
    w = np.ones((193, 1))
    r = np.zeros((257, 1))
    if mode == "underflow":
        r.fill(-0.9999999999999999)
    elif mode == "overflow":
        r.fill(1e308)
    elif mode == "mixed":
        r.fill(1e308)
        w[130:] = 0.0
    data = dict(w=w, r=r, p=np.array([0.05]))
    for x in data.values():
        x.flags.writeable = False
    root, nav = expression(seed, peak_seed, 2.5)
    independent = f"mean_axis(divide(1,{Y}),1)"
    for roots in [[root], [root, independent], [independent, root]]:
        try:
            reference = execute(roots + [nav], data, 257)
        except ValueError as error:
            with pytest.raises(ValueError) as caught:
                execute(roots, data, 257)
            assert str(caught.value) == str(error)
        else:
            actual = execute(roots, data, 257)
            for a, b in zip(actual.outputs, reference.outputs[: len(roots)], strict=True):
                ieee_equal(a.values[0], b.values[0])
