"""Consumption counts and logical-C packing for native order statistics."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_order_selection_library import (
    assert_bits,
    cached_sort_values,
    finite_values,
    sorted_value,
)


def layout_view(flat, layout, shape):
    flat = np.asarray(flat)
    if layout in ("C", "F"):
        return np.array(flat.reshape(shape), order=layout, copy=True)
    if layout == "padded":
        owner = np.empty(3 * flat.size, dtype=flat.dtype)
        if flat.dtype == np.dtype(bool):
            # Invalid bytes outside the logical mask must never be validated.
            owner.view(np.uint8)[:] = 2
        else:
            owner[:] = np.nan
        owner[::3] = flat
        return owner[::3].reshape(shape)
    if layout == "negative":
        owner = flat[::-1].copy()
        return owner[::-1].reshape(shape)
    owner = flat.copy()
    item = flat.itemsize
    if layout == "singleton2":
        return np.ndarray(shape, flat.dtype, buffer=owner, strides=(item, -37 * item))
    assert layout == "singleton3"
    return np.ndarray(shape, flat.dtype, buffer=owner, strides=(-19 * item, item, 0))


@pytest.mark.parametrize(
    "x_layout,m_layout,shape",
    [
        ("C", "C", (2, 1025)),
        ("C", "padded", (2, 1025)),
        ("padded", "C", (2, 1025)),
        ("F", "C", (2, 1025)),
        ("negative", "negative", (2, 1025)),
        ("singleton2", "singleton2", (2050, 1)),
        ("singleton3", "singleton3", (1, 2050, 1)),
    ],
)
@pytest.mark.parametrize("sparse", [False, True])
def test_packing_layout_and_actual_copy_count(x_layout, m_layout, shape, sparse):
    flat = finite_values(2050)
    selected = np.arange(flat.size) % 3 == 1 if sparse else np.ones(flat.size, bool)
    values = layout_view(flat, x_layout, shape)
    mask = layout_view(selected, m_layout, shape)
    values.setflags(write=False)
    mask.setflags(write=False)
    before = values.copy()
    mask_before = mask.view(np.uint8).copy()
    consumed = flat[selected]
    workspace = op.Workspace()
    for name, probability in (("median_where", None), ("quantile_where", 0.37)):
        arguments = (values, mask) if probability is None else (values, mask, probability)
        owner = np.full(3, 912.0)
        actual, audit = op.call(
            name, *arguments, workspace=workspace, out=owner[1:2].reshape(()), audit=True
        )
        assert_bits(actual, sorted_value(consumed, probability))
        assert audit["algorithm_copy_bytes"] == consumed.size * 8
        assert audit["workspace_bytes"] == consumed.size * 8
        assert audit["input_copy_bytes"] == 0
        assert owner[0] == owner[-1] == 912.0
    assert_bits(values, before)
    np.testing.assert_array_equal(mask.view(np.uint8), mask_before)


@pytest.mark.parametrize("count", [1023, 1024, 1025])
@pytest.mark.parametrize("m_layout", ["C", "padded"])
def test_threshold_is_consumed_count_not_input_size(count, m_layout):
    values = np.full(3 * count + 7, np.nan)
    chosen = finite_values(count)
    values[: 3 * count : 3] = chosen
    selected = np.zeros(values.size, bool)
    selected[: 3 * count : 3] = True
    mask = layout_view(selected, m_layout, values.shape)
    values.setflags(write=False)
    mask.setflags(write=False)
    for name, probability in (("median_where", None), ("quantile_where", 0.5)):
        args = (values, mask) if probability is None else (values, mask, probability)
        actual, audit = op.call(name, *args, audit=True)
        assert_bits(actual, sorted_value(chosen, probability))
        assert audit["algorithm_copy_bytes"] == count * 8
        assert audit["workspace_bytes"] == count * 8
        assert op.get(name).requirements(*args)["scratch_doubles"] == values.size


@pytest.mark.parametrize(
    "pattern",
    [
        "unconsumed",
        "mixed_zero",
        "negative_zero",
        "qnan",
        "snan",
    ],
)
def test_only_consumed_ieee_bits_affect_selection(pattern):
    chosen = finite_values(1025)
    if pattern == "mixed_zero":
        chosen.view(np.uint64)[::2] = 0x8000000000000000
        chosen.view(np.uint64)[1::2] = 0
    elif pattern == "negative_zero":
        chosen.view(np.uint64)[:] = 0x8000000000000000
    elif pattern == "qnan":
        chosen.view(np.uint64)[::2] = 0x7FF8000000000013
        chosen.view(np.uint64)[1::2] = 0xFFF8000000000029
    elif pattern == "snan":
        chosen.view(np.uint64)[::2] = 0x7FF0000000000013
        chosen.view(np.uint64)[1::2] = 0xFFF0000000000029
    # Even signaling NaNs and opposite zero signs are irrelevant when unselected.
    padding = np.array([0x7FF0000000000037, 0xFFF8000000000051, 0x8000000000000000, 0], np.uint64)
    bits = np.resize(padding, chosen.size * 3)
    bits[::3] = chosen.view(np.uint64)
    values = bits.view(np.float64)
    mask = np.zeros(values.size, bool)
    mask[::3] = True
    values.setflags(write=False)
    mask.setflags(write=False)
    before = bits.copy()
    # The unchanged multi-consumer native sort is the exceptional-tie oracle.
    # .5 has an integral rank here, so both interpolation operands use one payload.
    median, quantile = cached_sort_values(chosen, 0.5)
    for name, expected, extra in (
        ("median_where", median, ()),
        ("quantile_where", quantile, (0.5,)),
    ):
        actual, audit = op.call(name, values, mask, *extra, audit=True)
        assert_bits(actual, expected)
        assert audit["algorithm_copy_bytes"] == chosen.nbytes
    np.testing.assert_array_equal(bits, before)


@pytest.mark.parametrize("empty", [False, True])
def test_empty_consumption_keeps_output_and_workspace_reusable(empty):
    values = np.empty(0) if empty else finite_values(1025)
    mask = np.zeros(values.shape, bool)
    values.setflags(write=False)
    mask.setflags(write=False)
    before = values.copy()
    workspace = op.Workspace()
    for name, extra in (("median_where", ()), ("quantile_where", (0.5,))):
        owner = np.full(3, 912.0)
        with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
            op.call(
                name,
                values,
                mask,
                *extra,
                workspace=workspace,
                out=owner[1:2].reshape(()),
                audit=True,
            )
        np.testing.assert_array_equal(owner, 912.0)
        recovered, audit = op.call(
            name,
            np.array([2.0, 6.0, 10.0]),
            np.array([True, False, True]),
            *extra,
            workspace=workspace,
            audit=True,
        )
        assert_bits(recovered, 6.0)
        assert audit["algorithm_copy_bytes"] == 2 * 8
    assert_bits(values, before)


@pytest.mark.parametrize("bad_index", [0, 2049])
def test_invalid_bool_byte_is_rejected_before_value_or_output_write(bad_index):
    values = np.full(2050, np.nan)
    mask = np.ones(values.shape, bool)
    mask.view(np.uint8)[bad_index] = 2
    values.setflags(write=False)
    mask.setflags(write=False)
    before, mask_before = values.copy(), mask.view(np.uint8).copy()
    owner = np.full(3, 912.0)
    with pytest.raises(ValueError, match="INVALID_MASK"):
        op.call("quantile_where", values, mask, 0.5, out=owner[1:2].reshape(()), audit=True)
    assert_bits(values, before)
    np.testing.assert_array_equal(mask.view(np.uint8), mask_before)
    np.testing.assert_array_equal(owner, 912.0)


@pytest.mark.parametrize("value", [3.0, np.array(3.0), np.array([3.0]).reshape(())])
def test_public_rank_zero_reduction_remains_rejected(value):
    # Raw reduce_value/Value inline storage is not exposed by the Python API.
    owner = np.full(3, 912.0)
    with pytest.raises(ValueError, match="RANK_MISMATCH"):
        op.call("median", value, out=owner[1:2].reshape(()))
    np.testing.assert_array_equal(owner, 912.0)


def graph_copy_bytes(result):
    return sum(chunk["algorithm_copy_bytes"] for chunk in result.audit["native_chunks"])


@pytest.mark.parametrize(
    "name,extra",
    [
        ("median_where", ()),
        ("quantile_where", (0.37,)),
    ],
)
@pytest.mark.parametrize("reserved", [False, True])
def test_current_count_and_retained_workspace_capacity(name, extra, reserved):
    values = finite_values(4096)
    mask = np.zeros(values.shape, bool)
    workspace = op.Workspace()
    if reserved:
        workspace.reserve(doubles=values.size)
    high_water = values.size if reserved else 0
    owner = np.full(3, 912.0)
    for count in (7, values.size, 1, 0, 1025):
        mask[:] = False
        mask[:count] = True
        args = (values, mask, *extra)
        assert op.get(name).requirements(*args)["scratch_doubles"] == values.size
        if not count:
            owner[1] = 912.0
            with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
                op.call(name, *args, workspace=workspace, out=owner[1:2].reshape(()), audit=True)
            np.testing.assert_array_equal(owner, 912.0)
            continue
        actual, audit = op.call(
            name, *args, workspace=workspace, out=owner[1:2].reshape(()), audit=True
        )
        high_water = max(high_water, count)
        assert_bits(actual, sorted_value(values[:count], extra[0] if extra else None))
        assert audit["workspace_bytes"] == count * 8
        assert audit["workspace_capacity_bytes"] == high_water * 8
        assert owner[0] == owner[-1] == 912.0


def test_masked_counts_and_shared_unmasked_sort_coexist():
    values = finite_values(4100)
    mask = np.arange(values.size) % 3 != 0
    values.setflags(write=False)
    mask.setflags(write=False)
    graph = GraphCompiler(
        {"x": "series", "m": dict(kind="series", dtype="bool", semantic_dimension="mask")}
    ).compile(
        ["median(x)", "quantile(x,.75)", "median_where(x,m)", "quantile_where(x,m,.37)"],
        result_format="typed",
    )
    starts, ends = np.array([0, 2050], np.int64), np.array([2050, 4100], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": values, "m": mask}, starts, ends)
    assert sum(chunk["order_stat_sorts"] for chunk in result.audit["native_chunks"]) == 2
    assert graph_copy_bytes(result) == (values.size + 2 * np.count_nonzero(mask)) * 8
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        part = values[start:end]
        consumed = part[mask[start:end]]
        expected = (
            sorted_value(part),
            sorted_value(part, 0.75),
            sorted_value(consumed),
            sorted_value(consumed, 0.37),
        )
        for field, wanted in zip(result.outputs, expected, strict=True):
            assert_bits(field.values[row], wanted)
            assert not field.values[row].flags.writeable
            assert not np.shares_memory(field.values[row], values)


def test_prepared_rereads_current_mask_and_preserves_snapshots_after_failure():
    values = finite_values(2049)
    mask = np.zeros(values.shape, bool)
    values.setflags(write=False)
    mask.setflags(write=False)
    graph = GraphCompiler(
        {"x": "series", "m": dict(kind="series", dtype="bool", semantic_dimension="mask")}
    ).compile("quantile_where(x,m,.37)", result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(
            graph,
            {"x": values, "m": mask},
            np.array([0], np.int64),
            np.array([values.size], np.int64),
        )
        saved = []
        high_water = 0
        for count in (1023, 1024, 1025, values.size, 7):
            mask.setflags(write=True)
            mask[:] = False
            mask[:count] = True
            mask.setflags(write=False)
            result = prepared.run_snapshot()
            wanted = sorted_value(values[:count], 0.37)
            assert_bits(result.outputs[0].values[0], wanted)
            assert graph_copy_bytes(result) == count * 8
            high_water = max(high_water, count)
            assert (
                sum(
                    chunk["operator_workspace_capacity_bytes"]
                    for chunk in result.audit["native_chunks"]
                )
                == high_water * 8
            )
            saved.append((result, wanted))
        mask.setflags(write=True)
        mask[:] = False
        mask.setflags(write=False)
        with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
            prepared.run_snapshot()
        mask.setflags(write=True)
        mask[1::2] = True
        mask.setflags(write=False)
        borrowed = prepared.run()
        assert_bits(borrowed.outputs[0].values[0], sorted_value(values[1::2], 0.37))
        assert graph_copy_bytes(borrowed) == 1024 * 8
        for result, wanted in saved:
            assert_bits(result.outputs[0].values[0], wanted)
            assert not np.shares_memory(result.outputs[0].values[0], borrowed.outputs[0].values[0])
