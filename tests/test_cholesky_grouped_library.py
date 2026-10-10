"""Four independent Cholesky rows retain ordered arithmetic and error replay."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_cholesky_panel_library import DECL, assert_bits, assert_dispatch, spd


def layout_input(values, layout):
    if layout == "f":
        return np.asfortranarray(values), None
    if layout == "negative":
        owner = np.array(values[::-1, ::-1], order="C")
        return owner[::-1, ::-1], None
    if layout == "padded":
        n = len(values)
        owner = np.full((n + 3, n + 5), np.nan)
        owner[:n, :n] = values
        return owner[:n, :n], owner
    return values, None


def guarded_output(n):
    storage = np.full(n * n + 2, -913.25)
    return storage, storage[1:-1].reshape(n, n)


@pytest.mark.parametrize("n,layout", [
    (1026, "c"), (1027, "f"), (1028, "negative"),
    (1032, "padded"), (1039, "c"), (1040, "f"),
])
def test_grouped_dense_rows_and_unequal_triangular_tails(n, layout):
    # Dense off-diagonal entries exercise every preceding factor. These sizes
    # cover terminal panels of 2/3/4/8/15/16 rows without a layout cross-product.
    a, owner = layout_input(spd(n), layout)
    before = a.copy()
    owner_before = owner.copy() if owner is not None else None
    a.flags.writeable = False
    expected = op.cholesky(a, simd="scalar")
    storage, out = guarded_output(n)
    actual, audit = op.cholesky(a, out=out, audit=True)
    assert actual is out
    assert_bits(actual, expected)
    assert_dispatch(audit, n)
    assert storage[0] == storage[-1] == -913.25
    assert not np.shares_memory(actual, a)
    assert_bits(a, before)
    if owner is not None:
        assert_bits(owner, owner_before)


def common_tile_values(case, n=1032):
    a = np.eye(n)
    if case == "cancellation":
        # The (20,8) residual is (2**54+4)-2**54-1 = 3. Combining
        # the two products first would lose the 1 at this scale.
        for row in (8, 20):
            a[0, row] = a[row, 0] = 2. ** 27
            a[1, row] = a[row, 1] = 1.
        a[8, 8] = 2. ** 54 + 8.
        a[8, 20] = a[20, 8] = 2. ** 54 + 4.
        a[20, 20] = 2. ** 54 + 16.
    elif case == "subnormal":
        a[8, 20] = a[20, 8] = np.nextafter(0., 1.)
    else:
        a[a == 0.] = -0.
    return a


@pytest.mark.parametrize("case", ["cancellation", "subnormal", "signed_zero"])
def test_grouped_common_tile_preserves_consumed_special_values(case):
    # Column 8 of row 20 lies in a common complete tile, rather than the
    # diagonal or the final row's ungrouped suffix.
    a = common_tile_values(case)
    before = a.copy()
    a.flags.writeable = False
    expected = op.cholesky(a, simd="scalar")
    actual, audit = op.cholesky(a, audit=True)
    assert_bits(actual, expected)
    assert_dispatch(audit, len(a))
    assert_bits(a, before)
    if case == "cancellation":
        assert expected[20, 8] == 3. / np.sqrt(7.)
    elif case == "subnormal":
        assert expected[20, 8] == np.nextafter(0., 1.)
    else:
        assert expected[20, 8] == 0. and np.signbit(expected[20, 8])


def shared_tile_overflow(n, bad_row=16, overflow_row=20):
    a = np.eye(n)
    a[8, 8] = 2e300
    a[0, 8] = a[8, 0] = 1e150
    a[bad_row, bad_row] = -1.
    a[0, overflow_row] = a[overflow_row, 0] = 1e200
    return a


@pytest.mark.parametrize("bad_row,overflow_row,layout", [
    (16, 20, "c"), (19, 20, "f"),
    (20, 24, "c"), (27, 28, "f"),
])
def test_grouped_shared_tile_overflow_replays_earlier_bad_pivot(
        bad_row, overflow_row, layout):
    # All inputs/factors are finite. The shared tile's column 8 overflows
    # before packed execution reaches the earlier negative pivot in row order.
    a, _ = layout_input(shared_tile_overflow(1028, bad_row, overflow_row), layout)
    before = a.copy()
    a.flags.writeable = False
    errors = []
    for isa in ("scalar", "auto"):
        storage, out = guarded_output(len(a))
        with pytest.raises((ValueError, RuntimeError), match="DOMAIN_ERROR") as error:
            op.cholesky(a, out=out, simd=isa)
        errors.append(error.value)
        assert storage[0] == storage[-1] == -913.25
        assert_bits(a, before)
    assert type(errors[0]) is type(errors[1])
    assert str(errors[0]) == str(errors[1])


@pytest.mark.parametrize("value", [np.nan, np.inf], ids=["nan", "infinity"])
def test_grouped_consumed_nonfinite_keeps_input_validation_priority(value):
    a = np.eye(1028)
    a[16, 16] = -1.
    a[8, 20] = a[20, 8] = value
    before = a.copy()
    a.flags.writeable = False
    errors = []
    for isa in ("scalar", "auto"):
        storage, out = guarded_output(len(a))
        with pytest.raises((ValueError, RuntimeError), match="NON_FINITE_RESULT") as error:
            op.cholesky(a, out=out, simd=isa)
        errors.append(error.value)
        assert storage[0] == storage[-1] == -913.25
        assert_bits(a, before)
    assert type(errors[0]) is type(errors[1])
    assert str(errors[0]) == str(errors[1])


def test_grouped_f_snapshot_survives_shared_tile_failure_and_workspace_reuse():
    n = 1028
    good = spd(n)
    a = np.asfortranarray(good)
    a.flags.writeable = False
    expected = op.cholesky(a, simd="scalar")
    graph = GraphCompiler(DECL).compile("cholesky(a)", output_order="F")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a}, bounds, bounds)
        first = prepared.run_snapshot()
        first_value = first.outputs[0].values[0]
        assert_bits(first_value, expected)
        a.flags.writeable = True
        a[:] = shared_tile_overflow(n)
        a.flags.writeable = False
        failed_input = a.copy()
        with pytest.raises(ValueError, match="DOMAIN_ERROR"):
            prepared.run_snapshot()
        assert_bits(a, failed_input)
        assert_bits(first_value, expected)
        a.flags.writeable = True
        a[:] = good * 4.
        a.flags.writeable = False
        second = prepared.run_snapshot()
    second_value = second.outputs[0].values[0]
    assert_bits(second_value, op.cholesky(a, simd="scalar"))
    assert_bits(first_value, expected)
    assert not np.shares_memory(first_value, second_value)
    assert not np.shares_memory(first_value, a)
    for value in (first_value, second_value):
        assert value.flags.f_contiguous and not value.flags.writeable
    for chunk in second.audit["native_chunks"]:
        assert chunk["numeric_arena_bytes"] == chunk["root_copy_bytes"] == 0
        assert chunk["direct_output_bytes"] == a.nbytes
        assert chunk["input_copy_bytes"] == 0
