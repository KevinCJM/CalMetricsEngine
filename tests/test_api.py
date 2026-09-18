import gc
import importlib
import importlib.machinery
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import my_ctools as mc
from my_ctools import _api, _core

NAMES = [name for name in mc.__all__ if name.startswith("cal_")]


def invoke(name, values, *, n_threads=1):
    rows, cols = values.shape
    days = np.datetime64("2020-01-01", "ns") + np.arange(rows).astype("timedelta64[D]")
    starts = np.zeros(cols, dtype=np.int64)
    ends = np.full(cols, rows - 1, dtype=np.int64)
    if name == "cal_cpr":
        args = [np.arange(cols, dtype=np.int32) % 2, values]
    elif name == "cal_rolling_gain_loss":
        args = ["1M", values, starts, ends, days]
    elif name in {"cal_max_dd", "cal_all_largest_indicators", "cal_all_longest_indicators"}:
        args = [values, days]
    else:
        args = [values]
    return getattr(mc, name)(*args, n_threads=n_threads)


def arrays_in(result):
    if isinstance(result, np.ndarray):
        return [result]
    if isinstance(result, dict):
        result = result.values()
    return [item for item in result if isinstance(item, np.ndarray)]


def assert_same(actual, expected):
    if isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        for key in actual:
            assert_same(actual[key], expected[key])
    elif isinstance(actual, tuple):
        for a, b in zip(actual, expected, strict=True):
            assert_same(a, b)
    else:
        assert actual == expected


def test_installed_metadata_and_one_native_extension():
    assert mc.__version__ == importlib.metadata.version("my_ctools")
    root = Path(mc.__file__).parent
    extensions = [
        path
        for path in root.iterdir()
        if any(str(path).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    ]
    assert len(extensions) == 1
    assert extensions[0].name.startswith("_core.")
    assert (root / "py.typed").is_file()
    info = mc.build_info()
    assert info["cpu_policy"] == "baseline"
    assert info["cxx_standard"] == 17


@pytest.mark.parametrize("name", NAMES)
def test_legacy_imports(name):
    module_name = "cal_std_mean" if name == "cal_std_mean_simd" else name
    module = importlib.import_module(f"my_ctools.{module_name}")
    assert getattr(module, name) is getattr(mc, name)
    assert callable(getattr(mc, name))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (1, 0), (5, 0), (1, 3)])
def test_empty_and_single_row_shapes(name, shape):
    result = invoke(name, np.zeros(shape))
    for array in arrays_in(result):
        assert array.shape[-1] == shape[1]
        assert array.flags.owndata
        assert array.dtype in (np.dtype(np.float64), np.dtype(np.int64))


def test_empty_semantics():
    values = np.empty((0, 2))
    assert np.isnan(mc.cal_std_mean(values)).all()
    assert np.isnan(mc.cal_std_mean_simd(values)).all()
    assert np.isnan(mc.cal_cpr([1, 1], values)).all()
    np.testing.assert_array_equal(mc.cal_longest_dd_recover(values), [0, 0])
    drawdown, days, recovery = mc.cal_max_dd(values, [])
    assert np.isnan(drawdown).all()
    assert days == ["", ""]
    np.testing.assert_array_equal(recovery, [1_000_000, 1_000_000])
    assert all(np.isnan(output).all() for output in invoke("cal_rolling_gain_loss", values))


def test_statistics_numpy_reference():
    values = np.random.default_rng(42).normal(size=(80, 7))
    values[::3, 2] = np.nan
    means_stds = mc.cal_std_mean_simd(values)
    np.testing.assert_allclose(means_stds[0], np.nanmean(values, axis=0), rtol=1e-12)
    np.testing.assert_allclose(means_stds[1], np.nanstd(values, axis=0, ddof=1), rtol=1e-12)


@pytest.mark.parametrize(
    "layout", ["fortran", "strided", "negative", "float32", "unaligned", "big-endian"]
)
@pytest.mark.parametrize("name", NAMES)
def test_normalizes_layout_without_changing_results(name, layout):
    base = np.arange(480, dtype=np.float64).reshape(120, 4) / 100000 - 0.002
    if layout == "fortran":
        values = np.asfortranarray(base)
    elif layout == "strided":
        values = base[:, ::2]
    elif layout == "negative":
        values = base[::-1]
    elif layout == "float32":
        values = base.astype(np.float32)
    elif layout == "big-endian":
        values = base.astype(">f8")
    else:
        values = np.ndarray(
            base.shape, dtype=np.float64, buffer=bytearray(base.nbytes + 1), offset=1
        )
        values[:] = base
        assert not values.flags.aligned
    expected = invoke(name, np.array(values, dtype=np.float64, order="C", copy=True))
    assert_same(invoke(name, values), expected)


def test_compliant_inputs_are_borrowed():
    values = np.arange(24, dtype=np.float64).reshape(6, 4)
    assert _api._values(values) is values
    integers = np.arange(6, dtype=np.int64)
    assert _api._integers(integers, np.dtype(np.int64), "indices") is integers
    dates = integers.view("datetime64[ns]")
    assert np.shares_memory(_api._dates(dates), dates)
    values.setflags(write=False)
    assert _api._values(values) is values


@pytest.mark.parametrize("name", NAMES)
def test_readonly_input_and_output_lifetime(name):
    values = np.full((80, 4), 0.001)
    values.setflags(write=False)
    output = invoke(name, values, n_threads=2)
    saved = [array.copy() for array in arrays_in(output)]
    del values
    gc.collect()
    for actual, expected in zip(arrays_in(output), saved, strict=True):
        assert actual.flags.owndata
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [-1, 257, 2**100])
def test_invalid_thread_count(value):
    with pytest.raises(ValueError, match="n_threads"):
        mc.cal_std_mean([[1.0]], n_threads=value)


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, "2", None])
def test_noninteger_thread_count(value):
    with pytest.raises(TypeError, match="n_threads"):
        mc.cal_std_mean([[1.0]], n_threads=value)


@pytest.mark.parametrize("values", [np.ones(3), np.ones((2, 2, 2)), np.asarray(1.0)])
def test_wrong_dimensions(values):
    with pytest.raises(ValueError, match="dimensions"):
        mc.cal_std_mean(values)


@pytest.mark.parametrize("values", [[[1j]], [["1"]], [[None]]])
def test_nonreal_values(values):
    with pytest.raises(TypeError, match="real numeric"):
        mc.cal_std_mean(values)


def test_strict_core_does_not_copy_or_cast():
    with pytest.raises(TypeError):
        _core.cal_std_mean([[1.0]])
    with pytest.raises(TypeError, match="dtype"):
        _core.cal_std_mean(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="dtype"):
        _core.cal_std_mean(np.ones((2, 2), dtype=">f8"))
    with pytest.raises(ValueError, match="C-contiguous"):
        _core.cal_std_mean(np.ones((2, 4))[:, ::2])
    unaligned = np.ndarray((2, 2), dtype=np.float64, buffer=bytearray(33), offset=1)
    with pytest.raises(ValueError, match="aligned"):
        _core.cal_std_mean(unaligned)


def test_lengths_and_integer_bounds():
    values = np.ones((2, 2))
    with pytest.raises(ValueError, match="f_type length"):
        mc.cal_cpr([1], values)
    with pytest.raises(ValueError, match="dates length"):
        mc.cal_max_dd(values, [0])
    with pytest.raises(ValueError, match="range"):
        mc.cal_cpr([2**40, 1], values)
    with pytest.raises(ValueError, match="range"):
        mc.cal_max_dd(values, np.array([0, 2**63], dtype=np.uint64))
    with pytest.raises(TypeError, match="integers"):
        mc.cal_cpr([1.5, 1], values)
    with pytest.raises(TypeError, match="integers"):
        mc.cal_max_dd(values, [0.5, 1.5])


def test_nat_and_wrong_date_unit():
    values = np.ones((2, 1))
    with pytest.raises(ValueError, match="NaT"):
        mc.cal_max_dd(values, np.array(["NaT", "2020-01-01"], dtype="datetime64[ns]"))
    with pytest.raises(TypeError, match=r"datetime64\[ns\]"):
        mc.cal_max_dd(values, np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"))


@pytest.mark.parametrize(
    "date", ["1700-02-28", "1960-02-29", "1969-12-31", "1970-01-01", "2000-02-29", "2262-04-11"]
)
def test_portable_calendar(date):
    ns = np.datetime64(date, "ns").astype(np.int64)
    _, output_dates, _ = mc.cal_max_dd([[0.0], [-0.1]], [int(ns), int(ns)])
    assert output_dates == [date.replace("-", "")]


@pytest.mark.parametrize("rows", [0, 1, 2, 10])
def test_rolling_window_longer_than_sample(rows):
    output = invoke("cal_rolling_gain_loss", np.ones((rows, 2)) * 0.001)
    assert all(np.isnan(array).all() for array in output)


@pytest.mark.parametrize("period", ["bad", "0M", "0Y", "999999999999999999999999Y", "2147483647Y"])
def test_invalid_periods(period):
    with pytest.raises(ValueError, match="i_code"):
        mc.cal_rolling_gain_loss(period, [[0.0]], [0], [0], [0])


def test_invalid_modes():
    for func in (mc.cal_all_largest_indicators, mc.cal_all_longest_indicators):
        with pytest.raises(ValueError, match="i_code"):
            func([[0.1]], [0], "invalid")


def test_rolling_indices_and_sorting():
    for starts, ends in [([1], [0]), ([-2], [0]), ([0], [2])]:
        with pytest.raises(ValueError, match="indices"):
            mc.cal_rolling_gain_loss("1M", [[0.1], [0.2]], starts, ends, [0, 1])
    with pytest.raises(ValueError, match="lengths"):
        mc.cal_rolling_gain_loss("1M", [[0.0]], [], [], [0])
    with pytest.raises(ValueError, match="sorted"):
        mc.cal_rolling_gain_loss("1M", [[0.1], [0.2]], [0], [1], [1, 0])
    output = mc.cal_rolling_gain_loss("1M", [[0.1], [0.2]], [-1], [-1], [0, 1])
    assert all(np.isnan(array).all() for array in output)


def test_non_native_datetime_byte_order():
    days = np.array(["2020-02-28", "2020-02-29"], dtype="datetime64[ns]")
    swapped = days.astype(days.dtype.newbyteorder("S"))
    values = [[0.0], [-0.1]]
    assert_same(mc.cal_max_dd(values, swapped), mc.cal_max_dd(values, days))


def test_parallel_callers_share_readonly_inputs_safely():
    values = np.random.default_rng(13).normal(size=(300, 12)) * 0.001
    values.setflags(write=False)
    expected = mc.cal_std_mean(values)
    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(lambda _: mc.cal_std_mean(values, n_threads=2), range(16)))
    for output in outputs:
        np.testing.assert_array_equal(output, expected)
