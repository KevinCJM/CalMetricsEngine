import gc
import importlib
import importlib.machinery
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import calmetrics_engine as engine
from calmetrics_engine import _api, _native

NAMES = [name for name in engine.__all__ if name.startswith("cal_")]


def f64(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def i32(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int32)


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
    return getattr(engine, name)(*args, n_threads=n_threads)


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
        for left, right in zip(actual, expected, strict=True):
            assert_same(left, right)
    else:
        assert actual == expected


def test_installed_metadata_and_one_native_extension():
    assert engine.__version__ == importlib.metadata.version("calmetrics-engine")
    root = Path(engine.__file__).parent
    extensions = [
        path
        for path in root.iterdir()
        if any(str(path).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    ]
    assert len(extensions) == 1
    assert extensions[0].name.startswith("_native.")
    assert (root / "py.typed").is_file()
    info = engine.build_info()
    assert info["cpu_policy"] == "baseline"
    assert info["cxx_standard"] == 17
    assert info["execution_backend"] == "pybind11_aot"
    assert info["input_memory_policy"] == "exact_dtype_strided_zero_copy"


@pytest.mark.parametrize("name", NAMES)
def test_public_submodule_imports(name):
    module_name = "cal_std_mean" if name == "cal_std_mean_simd" else name
    module = importlib.import_module(f"calmetrics_engine.{module_name}")
    assert getattr(module, name) is getattr(engine, name)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (1, 0), (5, 0), (1, 3)])
def test_empty_and_single_row_shapes(name, shape):
    result = invoke(name, np.zeros(shape, dtype=np.float64))
    for array in arrays_in(result):
        assert array.shape[-1] == shape[1]
        assert array.flags.owndata
        assert array.dtype in (np.dtype(np.float64), np.dtype(np.int64))


def test_empty_semantics():
    values = np.empty((0, 2), dtype=np.float64)
    assert np.isnan(engine.cal_std_mean(values)).all()
    assert np.isnan(engine.cal_std_mean_simd(values)).all()
    assert np.isnan(engine.cal_cpr(i32([1, 1]), values)).all()
    np.testing.assert_array_equal(engine.cal_longest_dd_recover(values), [0, 0])
    drawdown, days, recovery = engine.cal_max_dd(values, i64([]))
    assert np.isnan(drawdown).all()
    assert days == ["", ""]
    np.testing.assert_array_equal(recovery, [1_000_000, 1_000_000])
    assert all(np.isnan(output).all() for output in invoke("cal_rolling_gain_loss", values))


def test_statistics_numpy_reference():
    values = np.random.default_rng(42).normal(size=(80, 7))
    values[::3, 2] = np.nan
    means_stds = engine.cal_std_mean_simd(values)
    np.testing.assert_allclose(means_stds[0], np.nanmean(values, axis=0), rtol=1e-12)
    np.testing.assert_allclose(means_stds[1], np.nanstd(values, axis=0, ddof=1), rtol=1e-12)


@pytest.mark.parametrize("layout", ["fortran", "strided-columns", "strided-rows", "negative"])
@pytest.mark.parametrize("name", NAMES)
def test_strided_zero_copy_layouts_match_contiguous_reference(name, layout):
    base = np.arange(960, dtype=np.float64).reshape(120, 8) / 100000 - 0.002
    if layout == "fortran":
        values = np.asfortranarray(base)
    elif layout == "strided-columns":
        values = base[:, ::2]
    elif layout == "strided-rows":
        values = base[::2, :]
    else:
        values = base[::-1, :]
    assert _api._values(values) is values
    expected = invoke(name, np.array(values, dtype=np.float64, order="C", copy=True))
    assert_same(invoke(name, values), expected)


def test_strided_integer_and_date_views_are_borrowed():
    types_owner = np.array([0, 99, 1, 99], dtype=np.int32)
    types = types_owner[::2]
    assert _api._integers(types, np.dtype(np.int32), "f_type") is types

    date_owner = np.array([0, 99, 1, 99], dtype=np.int64)
    dates = date_owner[::2]
    assert _api._dates(dates) is dates

    values = f64([[0.01, 0.02], [0.03, 0.01]])
    engine.cal_cpr(types, values)
    engine.cal_max_dd(values, dates)


def test_datetime_view_is_zero_copy():
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    converted = _api._dates(dates)
    assert np.shares_memory(converted, dates)
    assert converted.dtype == np.int64


@pytest.mark.parametrize("name", NAMES)
def test_readonly_input_and_output_lifetime(name):
    values = np.full((80, 4), 0.001, dtype=np.float64)
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
        engine.cal_std_mean(f64([[1.0]]), n_threads=value)


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, "2", None])
def test_noninteger_thread_count(value):
    with pytest.raises(TypeError, match="n_threads"):
        engine.cal_std_mean(f64([[1.0]]), n_threads=value)


@pytest.mark.parametrize(
    "value", [[[1.0]], np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=">f8")]
)
def test_python_boundary_never_coerces(value):
    with pytest.raises(TypeError, match="NumPy ndarray|exact native dtype"):
        engine.cal_std_mean(value)


@pytest.mark.parametrize("values", [np.ones(3), np.ones((2, 2, 2)), np.asarray(1.0)])
def test_wrong_dimensions(values):
    with pytest.raises(ValueError, match="ndim"):
        engine.cal_std_mean(values)


def test_unaligned_input_is_rejected_without_copy():
    values = np.ndarray((2, 2), dtype=np.float64, buffer=bytearray(33), offset=1)
    assert not values.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        engine.cal_std_mean(values)


def test_native_boundary_does_not_copy_or_cast():
    with pytest.raises(TypeError):
        _native.cal_std_mean([[1.0]])
    with pytest.raises(TypeError, match="dtype"):
        _native.cal_std_mean(np.ones((2, 2), dtype=np.float32))

    view = np.arange(16, dtype=np.float64).reshape(4, 4)[:, ::2]
    np.testing.assert_array_equal(
        _native.cal_std_mean(view),
        engine.cal_std_mean(view),
    )


def test_lengths_and_exact_integer_dtypes():
    values = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="f_type length"):
        engine.cal_cpr(i32([1]), values)
    with pytest.raises(ValueError, match="dates length"):
        engine.cal_max_dd(values, i64([0]))
    with pytest.raises(TypeError, match="exact native dtype"):
        engine.cal_cpr(i64([1, 1]), values)
    with pytest.raises(TypeError, match="exact native dtype"):
        engine.cal_max_dd(values, np.array([0, 1], dtype=np.uint64))


def test_nat_and_wrong_date_unit():
    values = np.ones((2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="NaT"):
        engine.cal_max_dd(
            values,
            np.array(["NaT", "2020-01-01"], dtype="datetime64[ns]"),
        )
    with pytest.raises(TypeError, match="exact native dtype"):
        engine.cal_max_dd(
            values,
            np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        )


@pytest.mark.parametrize(
    "date",
    ["1700-02-28", "1960-02-29", "1969-12-31", "1970-01-01", "2000-02-29", "2262-04-11"],
)
def test_portable_calendar(date):
    ns = np.datetime64(date, "ns").astype(np.int64)
    _, output_dates, _ = engine.cal_max_dd(
        f64([[0.0], [-0.1]]),
        np.array([ns, ns], dtype=np.int64),
    )
    assert output_dates == [date.replace("-", "")]


@pytest.mark.parametrize("rows", [0, 1, 2, 10])
def test_rolling_window_longer_than_sample(rows):
    output = invoke(
        "cal_rolling_gain_loss",
        np.ones((rows, 2), dtype=np.float64) * 0.001,
    )
    assert all(np.isnan(array).all() for array in output)


@pytest.mark.parametrize(
    "period",
    ["bad", "0M", "0Y", "999999999999999999999999Y", "2147483647Y"],
)
def test_invalid_periods(period):
    with pytest.raises(ValueError, match="i_code"):
        engine.cal_rolling_gain_loss(
            period,
            f64([[0.0]]),
            i64([0]),
            i64([0]),
            i64([0]),
        )


def test_invalid_modes():
    values = f64([[0.1]])
    dates = i64([0])
    for func in (engine.cal_all_largest_indicators, engine.cal_all_longest_indicators):
        with pytest.raises(ValueError, match="i_code"):
            func(values, dates, "invalid")


def test_rolling_indices_and_sorting():
    values = f64([[0.1], [0.2]])
    dates = i64([0, 1])
    for starts, ends in [(i64([1]), i64([0])), (i64([-2]), i64([0])), (i64([0]), i64([2]))]:
        with pytest.raises(ValueError, match="indices"):
            engine.cal_rolling_gain_loss("1M", values, starts, ends, dates)

    with pytest.raises(ValueError, match="lengths"):
        engine.cal_rolling_gain_loss(
            "1M",
            f64([[0.0]]),
            i64([]),
            i64([]),
            i64([0]),
        )
    with pytest.raises(ValueError, match="sorted"):
        engine.cal_rolling_gain_loss("1M", values, i64([0]), i64([1]), i64([1, 0]))

    output = engine.cal_rolling_gain_loss(
        "1M",
        values,
        i64([-1]),
        i64([-1]),
        dates,
    )
    assert all(np.isnan(array).all() for array in output)


def test_non_native_datetime_byte_order_is_rejected_not_copied():
    days = np.array(["2020-02-28", "2020-02-29"], dtype="datetime64[ns]")
    swapped = days.astype(days.dtype.newbyteorder("S"))
    with pytest.raises(TypeError, match="exact native dtype"):
        engine.cal_max_dd(f64([[0.0], [-0.1]]), swapped)


def test_parallel_callers_share_readonly_inputs_safely():
    values = np.random.default_rng(13).normal(size=(300, 12)) * 0.001
    values.setflags(write=False)
    expected = engine.cal_std_mean(values)
    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(lambda _: engine.cal_std_mean(values, n_threads=2), range(16)))
    for output in outputs:
        np.testing.assert_array_equal(output, expected)
