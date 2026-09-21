"""Native type, memory, ISA, workspace and concurrency contracts."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import numpy as np
import pytest

from calmetrics_engine import operators as op


def test_registry_is_an_immutable_source_of_truth():
    entry = op.catalog()[0]
    entry["id"] = "changed"
    assert op.catalog()[0]["id"] == "add"
    with pytest.raises((ValueError, TypeError)):
        op.get("__import__")
    for code in (0, 126, -1, 65536):
        with pytest.raises((ValueError, TypeError)):
            op.get_by_opcode(code)
    assert "rolling_apply" not in {item["id"] for item in op.catalog()}
    assert "rolling_window" not in {item["id"] for item in op.catalog()}


def test_named_arguments_defaults_and_preparation():
    x = np.arange(8, dtype=np.float64)
    np.testing.assert_array_equal(op.add(lhs=x, rhs=2.0), x + 2)
    np.testing.assert_allclose(op.std(values=x, ddof=0), np.std(x))
    np.testing.assert_allclose(
        op.rolling_std(values=x, window=3, min_periods=1),
        op.rolling_std(x, 3, 0, 1),
        equal_nan=True,
    )
    assert op.get("std").requirements(x)["shape"] == []
    assert op.get("transpose").requirements(x.reshape(4, 2))["shape"] == [2, 4]
    assert op.get("transpose").requirements(x.reshape(4, 2))["borrowed_output"]
    with pytest.raises(TypeError, match="Multiple"):
        op.add(x, 1.0, lhs=x)
    with pytest.raises(TypeError, match="Missing"):
        op.add(rhs=x)
    with pytest.raises(TypeError, match="Unknown"):
        op.std(x, unknown=1)
    with pytest.raises((ValueError, TypeError), match="ARITY"):
        op.add(x, 1.0, 2.0)
    with pytest.raises(TypeError, match="audit"):
        op.sum(x, audit=1)


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 2.0],
        np.ones(3, dtype=np.float32),
        np.ones(3, dtype=np.int64),
        np.ones(3, dtype="c16"),
        np.ones(3, dtype=object),
        np.ones(3, dtype=">f8"),
    ],
)
def test_never_casts_array_like_values(value):
    with pytest.raises(TypeError):
        op.sum(value)


@pytest.mark.parametrize("value", [np.float32(1.0), 2**60, np.int64(2**60)])
def test_rejects_inexact_or_wrong_scalar_types(value):
    with pytest.raises((ValueError, TypeError)):
        op.add(value, 1.0)


@pytest.mark.parametrize("value", [1.0, np.asarray(1.0), np.ones((1, 1, 1))])
def test_reduction_rejects_wrong_rank(value):
    with pytest.raises(ValueError, match="RANK"):
        op.mean(value)


@pytest.mark.parametrize("name", ["add", "subtract", "multiply", "divide", "greater_than"])
def test_no_implicit_matrix_vector_broadcast(name):
    with pytest.raises(ValueError, match="SHAPE"):
        getattr(op, name)(np.ones((4, 3)), np.ones(3))


def test_matrix_shape_validation_happens_before_access():
    a = np.ones((2, 3))
    for name, args in (
        ("matmul", (a, np.ones((2, 2)))),
        ("matvec", (a, np.ones(2))),
        ("solve", (a, np.ones(3))),
        ("dot", (np.ones(2), np.ones(3))),
        ("covariance", (np.ones(2), np.ones(3))),
        ("quadratic_form", (np.ones(2), a)),
        ("sum_where", (np.ones(2), np.ones(3, dtype=np.uint8))),
    ):
        with pytest.raises(ValueError, match="SHAPE"):
            getattr(op, name)(*args)


def test_numeric_masks_are_not_interchangeable():
    with pytest.raises(ValueError, match="DTYPE"):
        op.logical_and(np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="DTYPE"):
        op.add(np.ones(3, dtype=np.uint8), 1.0)
    with pytest.raises(ValueError, match="INVALID_MASK"):
        op.count_true(np.array([0, 1, 2], dtype=np.uint8))
    with pytest.raises(ValueError, match="INVALID_MASK"):
        op.logical_not(np.uint8(255))
    with pytest.raises(ValueError, match="SHAPE"):
        op.logical_and(True, np.ones(3, dtype=np.uint8))
    np.testing.assert_array_equal(op.logical_not(np.array([True, False])), [0, 1])
    np.testing.assert_array_equal(op.where(np.array([True, False]), 3.0, 7.0), [3.0, 7.0])
    np.testing.assert_array_equal(op.where(True, np.arange(3.0), 0.0), np.arange(3.0))


def test_domain_missing_and_tie_contracts():
    missing = np.array([1.0, np.nan, 3.0])
    assert np.isnan(op.mean(missing))  # Ordinary reduction does not silently nanmean.
    assert op.mean_where(missing, np.array([1, 0, 1], dtype=np.uint8)) == 2.0
    np.testing.assert_array_equal(op.new_high_mask(np.array([1.0, 1.0, 2.0, 2.0])), [1, 0, 1, 0])
    interval = op.last_drawdown_interval(np.array([0.0, -0.2, 0.0, -0.2, -0.2, 0.0]))
    assert interval == (2.0, 4.0, 5.0, 1.0)
    assert op.interval_start(interval) == 2.0
    assert op.interval_trough(interval) == 4.0
    assert op.interval_recovery(interval) == 5.0
    assert np.isnan(op.interval_recovery(op.last_drawdown_interval(np.array([0.0, -0.2]))))
    assert op.last_drawdown_interval(np.array([0.0, np.nan]))[3] == -1.0
    with pytest.raises(ValueError, match="RECORD_TYPE"):
        op.fit_slope(interval)
    with pytest.raises(ValueError, match="RECORD_TYPE"):
        op.interval_start(op.linear_fit(np.arange(3.0)))
    assert op.days_between(100.0, 110.0) == 10.0
    assert np.isnan(op.value_at(missing, 1.0))


@pytest.mark.parametrize("name", ["std", "variance"])
@pytest.mark.parametrize("degrees", [-1, np.nan, np.inf, 1.5, 1e100])
def test_ddof_is_validated_before_integer_conversion(name, degrees):
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        getattr(op, name)(np.arange(8.0), degrees)


@pytest.mark.parametrize("name", ["rolling_mean", "rolling_std", "rolling_min", "rolling_max"])
def test_window_and_minimum_constraints(name):
    x = np.arange(8.0)
    for window in (0, -1, 0.5, np.nan, np.inf, 1e100):
        with pytest.raises(ValueError, match="INVALID_PARAMETER"):
            getattr(op, name)(x, window)
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        getattr(op, name)(x, 3, min_periods=4)


@pytest.mark.parametrize(
    "name,args",
    [
        ("add", (np.arange(8.0), 2.0)),
        ("greater_than", (np.arange(8.0), 2.0)),
        ("cumulative_sum", (np.arange(8.0),)),
    ],
)
def test_out_shape_dtype_writeability_and_alias(name, args):
    function = getattr(op, name)
    x = args[0]
    dtype = np.uint8 if name == "greater_than" else np.float64
    with pytest.raises(ValueError, match="OUTPUT"):
        function(*args, out=np.empty(7, dtype=dtype))
    with pytest.raises((ValueError, TypeError)):
        function(*args, out=np.empty(8, dtype=np.float32))
    readonly = np.empty(8, dtype=dtype)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="OUTPUT"):
        function(*args, out=readonly)
    with pytest.raises(ValueError, match="OUTPUT"):
        function(*args, out=np.empty(16, dtype=dtype)[::2])
    if dtype == np.float64:
        with pytest.raises(ValueError, match="ALIASES"):
            function(*args, out=x)
        storage = np.arange(16.0)
        with pytest.raises(ValueError, match="ALIASES"):
            function(storage[:8], *args[1:], out=storage[1:9])


def test_scalar_out_and_record_out():
    x = np.arange(8.0)
    output = np.empty((), dtype=np.float64)
    assert op.sum(x, out=output) is output
    assert output.item() == 28.0
    boolean = np.empty((), dtype=np.uint8)
    assert op.logical_not(False, out=boolean) is boolean
    assert boolean.item() == 1
    with pytest.raises(TypeError, match="record outputs"):
        op.linear_fit(x, out=np.empty(5))


@pytest.mark.parametrize("name", ["lag", "transpose", "diag"])
def test_view_owner_survives_input_deletion(name):
    owner = np.arange(24.0)
    source = owner if name == "lag" else owner.reshape(6, 4)
    pointer = source.__array_interface__["data"][0]
    result, audit = getattr(op, name)(source, audit=True)
    assert np.shares_memory(result, owner)
    assert result.__array_interface__["data"][0] == pointer
    assert not result.flags.writeable
    assert audit["output_is_view"]
    assert audit["scheduler_backend"] == "process_wide_cpp_cpu_admission"
    assert audit["cpu_tokens"] == 0
    saved = result.copy()
    ref = weakref.ref(owner)
    del source, owner
    gc.collect()
    assert ref() is not None
    np.testing.assert_array_equal(result, saved)
    del result
    gc.collect()
    assert ref() is None


def test_strided_zero_stride_zero_length_and_unaligned():
    vector = np.broadcast_to(np.array([2.0]), (8,))
    np.testing.assert_array_equal(op.add(vector, 1.0), np.full(8, 3.0))
    with pytest.raises(ValueError, match="UNALIGNED"):
        op.sum(np.ndarray((3,), dtype=np.float64, buffer=bytearray(25), offset=1))
    with pytest.raises(ValueError, match="BYTE_STRIDE"):
        op.sum(np.ndarray((3,), dtype=np.float64, buffer=bytearray(32), strides=(9,)))
    bad = np.lib.stride_tricks.as_strided(np.ones(1), shape=(3,), strides=(2**62,))
    with pytest.raises(ValueError, match="STRIDE_OVERFLOW"):
        op.sum(bad)
    assert op.length(np.empty(0)) == 0
    np.testing.assert_array_equal(op.add(np.empty(0), 2.0), np.empty(0))
    assert op.transpose(np.empty((0, 3))).shape == (3, 0)
    assert op.diag(np.empty((0, 3))).shape == (0,)


@pytest.mark.parametrize(
    "name",
    ["median", "quantile", "solve", "rolling_min", "rolling_max", "covariance", "quadratic_form"],
)
def test_workspace_reuse_and_copy_accounting(name):
    x = np.arange(20.0)
    matrix = np.eye(4) * 2 + 0.25
    args = {
        "median": (x,),
        "quantile": (x, 0.3),
        "solve": (matrix, np.ones(4)),
        "rolling_min": (x, 3),
        "rolling_max": (x, 3),
        "covariance": (np.arange(20.0).reshape(5, 4),),
        "quadratic_form": (np.ones(4), matrix),
    }[name]
    workspace = op.Workspace()
    requirements = op.get(name).requirements(*args)
    workspace.reserve(
        doubles=requirements["scratch_doubles"], indices=requirements["scratch_indices"]
    )
    before = workspace.capacity_bytes
    first, audit = op.call(name, *args, workspace=workspace, audit=True)
    snapshot = np.array(first, copy=True)
    second = op.call(name, *args, workspace=workspace)
    assert workspace.capacity_bytes == before
    np.testing.assert_allclose(first, snapshot, equal_nan=True)
    np.testing.assert_allclose(second, snapshot, equal_nan=True)
    assert audit["workspace_bytes"] == requirements["workspace_bytes"]
    assert audit["input_copy_bytes"] == 0
    assert audit["scheduler_backend"] == "process_wide_cpp_cpu_admission"
    assert audit["cpu_tokens"] == 1
    if name == "solve":
        assert audit["algorithm_copy_bytes"] == (16 + 4) * 8
    elif name in {"median", "quantile"}:
        assert audit["algorithm_copy_bytes"] == x.size * 8
        assert requirements["scratch_doubles"] == x.size
        assert requirements["scratch_indices"] == 0
    else:
        assert audit["algorithm_copy_bytes"] == 0
    if name in {"rolling_min", "rolling_max"}:
        assert requirements["scratch_indices"] == 3  # bounded by window, not history


def test_parallel_callers_and_workspace_exclusion():
    x = np.random.default_rng(14).normal(size=25000)
    x.flags.writeable = False
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: op.std(x), range(16)))
    np.testing.assert_array_equal(results, np.repeat(op.std(x), 16))

    # Sorting a large numeric scratch buffer guarantees useful overlap without modifying x.
    large = np.random.default_rng(15).normal(size=1_000_000)
    workspace = op.Workspace()
    started = threading.Event()

    def calculate():
        started.set()
        return op.quantile(large, 0.5, workspace=workspace)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(calculate)
        assert started.wait(timeout=5)
        saw_busy = False
        deadline = time.monotonic() + 5
        while not future.done() and time.monotonic() < deadline:
            try:
                workspace.reserve()
            except ValueError as exc:
                assert "WORKSPACE_BUSY" in str(exc)
                saw_busy = True
                break
            time.sleep(0.001)
        assert saw_busy
        np.testing.assert_allclose(future.result(timeout=10), np.quantile(large, 0.5))
    assert np.isfinite(op.quantile(large[:8], 0.5, workspace=workspace))


def test_shared_memory_can_be_borrowed_without_copy():
    storage = shared_memory.SharedMemory(create=True, size=32 * 8)
    try:
        array = np.ndarray((32,), dtype=np.float64, buffer=storage.buf)
        array[:] = np.arange(32.0)
        array.flags.writeable = False
        view = array[8:24]
        result, audit = op.sum(view, audit=True)
        assert result == sum(range(8, 24))
        assert audit["input_addresses"] == [view.__array_interface__["data"][0]]
        assert audit["input_copy_bytes"] == 0
        del view, array
    finally:
        storage.close()
        storage.unlink()


SIMD_NAMES = [
    spec["id"] for spec in op.catalog() if spec["simd_eligible"] and spec["family"] == "elementwise"
]


@pytest.mark.parametrize("name", SIMD_NAMES)
@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 7, 17])
def test_simd_tails_scalar_parity_and_real_lane(name, length):
    # Offset by one double: dtype-aligned but not necessarily SIMD-width aligned.
    x = np.arange(length + 1, dtype=np.float64)[1:] + 1.0
    y = np.arange(length + 1, dtype=np.float64)[1:] + 2.0
    args = (x,) if op.get(name).spec["min_args"] == 1 else (x, y)
    reference = op.call(name, *args, simd="scalar")
    for isa in op.available_simd():
        result, audit = op.call(name, *args, simd=isa, audit=True)
        np.testing.assert_array_equal(result, reference)
        if isa != "scalar" and length >= 4:
            assert audit["isa"] == isa
            assert audit["vector_elements"] > 0
        assert audit["vector_elements"] <= length


@pytest.mark.parametrize(
    "shape",
    [(5, 7, 9), (8, 8, 8), (3, 10, 6), (4, 0, 8), (0, 5, 8), (5, 7, 0), (1, 1, 1), (17, 17, 17)],
)
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "negative_left", "negative_right"])
def test_matrix_simd_blocks_tails_and_layouts(shape, layout):
    rows, inner, cols = shape
    rng = np.random.default_rng(102)
    lhs = rng.normal(size=(rows, inner))
    rhs = rng.normal(size=(inner, cols))
    if layout == "fortran":
        lhs, rhs = np.asfortranarray(lhs), np.asfortranarray(rhs)
    elif layout == "negative_left":
        lhs = lhs[::-1]
    elif layout == "negative_right":
        rhs = rhs[::-1]
    expected = lhs @ rhs
    for isa in op.available_simd():
        result, audit = op.matmul(lhs, rhs, simd=isa, audit=True)
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-14)
        assert audit["input_copy_bytes"] == 0
        assert audit["algorithm_copy_bytes"] == 0
        if isa != "scalar" and rows >= 4 and cols >= 8 and inner > 0 and rhs.strides[1] == 8:
            assert audit["isa"] == isa
            assert audit["vector_elements"] > 0


def test_detectable_unsafe_views_and_explicit_none_are_rejected():
    owner = np.arange(4.0)
    bad = np.lib.stride_tricks.as_strided(owner, shape=(8,), strides=(8,))
    with pytest.raises(ValueError, match="OUT_OF_BOUNDS"):
        op.sum(bad)
    before = np.lib.stride_tricks.as_strided(owner, shape=(4,), strides=(-8,))
    with pytest.raises(ValueError, match="OUT_OF_BOUNDS"):
        op.sum(before)
    with pytest.raises(TypeError, match="Multiple"):
        op.linear_fit(None, owner, values=owner)
    with pytest.raises(TypeError):
        op.rolling_std(owner, 3, None)


def test_simd_nan_masks_and_signed_zero():
    x = np.array([-0.0, 0.0, np.nan, np.inf, -np.inf, 2.0, -1.0, np.nan])
    y = np.array([0.0, -0.0, 1.0, np.inf, 1.0, np.nan, -1.0, np.nan])
    for name in ("minimum", "maximum", "equal", "not_equal", "less_than", "greater_equal"):
        reference = op.call(name, x, y, simd="scalar")
        for isa in op.available_simd():
            result = op.call(name, x, y, simd=isa)
            np.testing.assert_array_equal(result, reference)
            zero = (result == 0) & (reference == 0)
            np.testing.assert_array_equal(np.signbit(result[zero]), np.signbit(reference[zero]))
    np.testing.assert_array_equal(op.finite_mask(x), np.isfinite(x).astype(np.uint8))
    for isa in {"sse2", "avx2", "neon"} - set(op.available_simd()):
        with pytest.raises(ValueError, match="UNSUPPORTED_ISA"):
            op.add(np.arange(8.0), 1.0, simd=isa)
    with pytest.raises(ValueError, match="INVALID_SIMD"):
        op.add(np.arange(8.0), 1.0, simd="avx512_not_installed")
