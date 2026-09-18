"""Portable, precompiled C++ financial analytics. No runtime compilation."""

from . import cal_std_mean as cal_std_mean_module  # noqa: F401 - historical module alias
from ._api import build_info
from ._core import __version__
from .cal_all_largest_indicators import cal_all_largest_indicators
from .cal_all_longest_indicators import cal_all_longest_indicators
from .cal_cpr import cal_cpr
from .cal_longest_dd_recover import cal_longest_dd_recover
from .cal_max_dd import cal_max_dd
from .cal_rolling_gain_loss import cal_rolling_gain_loss
from .cal_std_mean import cal_std_mean, cal_std_mean_simd

__all__ = [
    "__version__",
    "build_info",
    "cal_std_mean",
    "cal_std_mean_simd",
    "cal_cpr",
    "cal_longest_dd_recover",
    "cal_max_dd",
    "cal_all_largest_indicators",
    "cal_all_longest_indicators",
    "cal_rolling_gain_loss",
]
