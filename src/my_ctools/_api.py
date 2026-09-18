"""Stable, normalizing Python API over the strict native extension.

Native aligned C-contiguous arrays of the expected dtype are borrowed, not copied.
Callers must not mutate an input while a calculation is running without the GIL.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _values(value: ArrayLike) -> FloatArray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("values must contain real numeric data")
    return np.require(array, dtype=np.float64, requirements=["C", "A"])


def _integers(value: ArrayLike, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(value)
    # An untyped empty list is safe to convert; nonempty fractional indices are not.
    if array.size and array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    if array.size and array.dtype != dtype:
        limits = np.iinfo(dtype)
        if int(array.min()) < limits.min or int(array.max()) > limits.max:
            raise ValueError(f"{name} values are out of range for {dtype.name}")
    return np.require(array, dtype=dtype, requirements=["C", "A"])


def _dates(value: ArrayLike) -> IntArray:
    array = np.asarray(value)
    if array.dtype.kind == "M":
        if np.datetime_data(array.dtype) != ("ns", 1):
            raise TypeError("dates must use datetime64[ns], or int64 nanoseconds")
        if not array.dtype.isnative:
            array = array.astype("datetime64[ns]")
        array = array.view(np.int64)
    return _integers(array, np.dtype(np.int64), "dates")


def _threads(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("n_threads must be an integer")
    if not 0 <= value <= 256:
        raise ValueError("n_threads must be between 0 and 256")
    return int(value)


def cal_std_mean(input: ArrayLike, *, n_threads: int = 1) -> FloatArray:
    """Return column sample std (ddof=1), ignoring NaN; historical name retained."""
    return _core.cal_std_mean(_values(input), n_threads=_threads(n_threads))


def cal_std_mean_simd(input: ArrayLike, *, n_threads: int = 1) -> FloatArray:
    """Return [means, sample stds], shape (2, N); both NaN when count <= 1.

    The historical suffix is retained, but does not require AVX or promise SIMD.
    """
    return _core.cal_std_mean_simd(_values(input), n_threads=_threads(n_threads))


def cal_cpr(f_type: ArrayLike, funds_value: ArrayLike, *, n_threads: int = 1) -> FloatArray:
    """Return persistence ratios relative to each type's daily median."""
    return _core.cal_cpr(
        _integers(f_type, np.dtype(np.int32), "f_type"),
        _values(funds_value),
        n_threads=_threads(n_threads),
    )


def cal_longest_dd_recover(funds_val: ArrayLike, *, n_threads: int = 1) -> IntArray:
    """Return longest drawdown-recovery periods, using int64 on every platform."""
    return _core.cal_longest_dd_recover(_values(funds_val), n_threads=_threads(n_threads))


def cal_max_dd(
    funds_val: ArrayLike, day_arr: ArrayLike, *, n_threads: int = 1
) -> tuple[FloatArray, list[str], IntArray]:
    """Return maximum drawdown, YYYYMMDD dates and recovery observation counts."""
    return _core.cal_max_dd(_values(funds_val), _dates(day_arr), n_threads=_threads(n_threads))


def cal_all_largest_indicators(
    array_value: ArrayLike,
    dates: ArrayLike,
    i_code: str = "positive",
    *,
    n_threads: int = 1,
) -> dict[str, FloatArray | IntArray | list[str]]:
    """Return legacy largest-streak fields r, p, s, l; modes positive/negative."""
    return _core.cal_all_largest_indicators(
        _values(array_value), _dates(dates), i_code, n_threads=_threads(n_threads)
    )


def cal_all_longest_indicators(
    a_value: ArrayLike,
    dates: ArrayLike,
    i_code: str = "positive",
    *,
    n_threads: int = 1,
) -> tuple[FloatArray, list[str], list[str], IntArray]:
    """Return legacy longest-streak return, start/end dates and observation counts."""
    return _core.cal_all_longest_indicators(
        _values(a_value), _dates(dates), i_code, n_threads=_threads(n_threads)
    )


def cal_rolling_gain_loss(
    i_code: str,
    funds_val: ArrayLike,
    start_idx: ArrayLike,
    end_idx: ArrayLike,
    day_arr: ArrayLike,
    *,
    n_threads: int = 1,
) -> tuple[FloatArray, ...]:
    """Return legacy rolling mean, median, win rate and six gain/loss buckets.

    Period codes contain a positive number followed by M or Y. Indices are
    inclusive; (-1, -1) marks an inactive column. Dates must be sorted.
    """
    return _core.cal_rolling_gain_loss(
        i_code,
        _values(funds_val),
        _integers(start_idx, np.dtype(np.int64), "start_idx"),
        _integers(end_idx, np.dtype(np.int64), "end_idx"),
        _dates(day_arr),
        n_threads=_threads(n_threads),
    )


def build_info() -> dict[str, str | int]:
    """Return native build version, compiler, CPU policy and thread backend."""
    return _core.build_info()
