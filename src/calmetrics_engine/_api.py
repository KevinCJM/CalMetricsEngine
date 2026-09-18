"""Strict zero-copy Python boundary for the CalMetricsEngine native backend.

The engine never coerces calculation inputs. Callers normalize dtype once at the
platform data boundary, then CalMetricsEngine borrows NumPy memory directly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _native

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _array(value: np.ndarray, *, dtype: np.dtype, ndim: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray; CalMetricsEngine does not copy inputs")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if value.dtype != dtype or not value.dtype.isnative:
        raise TypeError(f"{name} must use exact native dtype {dtype.name}")
    if not value.flags.aligned:
        raise ValueError(f"{name} must be aligned")
    return value


def _values(value: np.ndarray) -> FloatArray:
    return _array(
        value,
        dtype=np.dtype(np.float64),
        ndim=2,
        name="values",
    )


def _integers(value: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    return _array(value, dtype=dtype, ndim=1, name=name)


def _dates(value: np.ndarray) -> IntArray:
    if not isinstance(value, np.ndarray):
        raise TypeError("dates must be a NumPy ndarray; CalMetricsEngine does not copy inputs")
    if value.ndim != 1:
        raise ValueError("dates must have ndim=1")
    if value.dtype == np.dtype("datetime64[ns]") and value.dtype.isnative:
        result = value.view(np.int64)
        if value.size and not np.shares_memory(result, value):
            raise RuntimeError("datetime64[ns] view unexpectedly copied memory")
        return result
    return _integers(value, np.dtype(np.int64), "dates")


def _threads(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("n_threads must be an integer")
    if not 0 <= value <= 256:
        raise ValueError("n_threads must be between 0 and 256")
    return int(value)


def cal_std_mean(input: np.ndarray, *, n_threads: int = 1) -> FloatArray:
    """Return column sample std (ddof=1), ignoring NaN."""
    return _native.cal_std_mean(_values(input), n_threads=_threads(n_threads))


def cal_std_mean_simd(input: np.ndarray, *, n_threads: int = 1) -> FloatArray:
    """Return [means, sample stds], shape (2, N).

    The historical suffix is retained for calculation compatibility; the public
    wheel does not require a particular SIMD ISA.
    """
    return _native.cal_std_mean_simd(_values(input), n_threads=_threads(n_threads))


def cal_cpr(
    f_type: np.ndarray,
    funds_value: np.ndarray,
    *,
    n_threads: int = 1,
) -> FloatArray:
    """Return persistence ratios relative to each type's daily median."""
    return _native.cal_cpr(
        _integers(f_type, np.dtype(np.int32), "f_type"),
        _values(funds_value),
        n_threads=_threads(n_threads),
    )


def cal_longest_dd_recover(
    funds_val: np.ndarray,
    *,
    n_threads: int = 1,
) -> IntArray:
    """Return longest drawdown-recovery periods."""
    return _native.cal_longest_dd_recover(
        _values(funds_val),
        n_threads=_threads(n_threads),
    )


def cal_max_dd(
    funds_val: np.ndarray,
    day_arr: np.ndarray,
    *,
    n_threads: int = 1,
) -> tuple[FloatArray, list[str], IntArray]:
    """Return maximum drawdown, YYYYMMDD dates and recovery observation counts."""
    return _native.cal_max_dd(
        _values(funds_val),
        _dates(day_arr),
        n_threads=_threads(n_threads),
    )


def cal_all_largest_indicators(
    array_value: np.ndarray,
    dates: np.ndarray,
    i_code: str = "positive",
    *,
    n_threads: int = 1,
) -> dict[str, FloatArray | IntArray | list[str]]:
    """Return legacy largest-streak fields r, p, s and l."""
    return _native.cal_all_largest_indicators(
        _values(array_value),
        _dates(dates),
        i_code,
        n_threads=_threads(n_threads),
    )


def cal_all_longest_indicators(
    a_value: np.ndarray,
    dates: np.ndarray,
    i_code: str = "positive",
    *,
    n_threads: int = 1,
) -> tuple[FloatArray, list[str], list[str], IntArray]:
    """Return longest-streak return, start/end dates and observation counts."""
    return _native.cal_all_longest_indicators(
        _values(a_value),
        _dates(dates),
        i_code,
        n_threads=_threads(n_threads),
    )


def cal_rolling_gain_loss(
    i_code: str,
    funds_val: np.ndarray,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    day_arr: np.ndarray,
    *,
    n_threads: int = 1,
) -> tuple[FloatArray, ...]:
    """Return rolling mean, median, win rate and six gain/loss buckets."""
    return _native.cal_rolling_gain_loss(
        i_code,
        _values(funds_val),
        _integers(start_idx, np.dtype(np.int64), "start_idx"),
        _integers(end_idx, np.dtype(np.int64), "end_idx"),
        _dates(day_arr),
        n_threads=_threads(n_threads),
    )


def build_info() -> dict[str, str | int]:
    """Return immutable native build and memory-contract metadata."""
    return _native.build_info()
