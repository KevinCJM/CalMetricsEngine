"""Compatibility import path; implementation lives in the shared native core."""

from ._api import cal_std_mean, cal_std_mean_simd

__all__ = ["cal_std_mean", "cal_std_mean_simd"]
