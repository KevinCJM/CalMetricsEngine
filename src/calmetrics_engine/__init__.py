"""CalMetricsEngine: AOT native calculation backend for financial research."""

from . import cal_std_mean as cal_std_mean_module  # noqa: F401
from . import operators
from ._api import build_info
from ._native import __version__
from .cal_all_largest_indicators import cal_all_largest_indicators
from .cal_all_longest_indicators import cal_all_longest_indicators
from .cal_cpr import cal_cpr
from .cal_longest_dd_recover import cal_longest_dd_recover
from .cal_max_dd import cal_max_dd
from .cal_rolling_gain_loss import cal_rolling_gain_loss
from .cal_std_mean import cal_std_mean, cal_std_mean_simd
from .graph import CompiledGraph, GraphCompileError, GraphCompiler
from .planner import AdaptivePlanner, ExecutionPlan, PlannerConfig
from .runtime import AdaptiveScheduler, GraphExecutionResult, PreparedGraphExecution
from .shared import SharedArrayDescriptor, SharedArrayOwner, SharedInputBundle

__all__ = [
    "__version__",
    "build_info",
    "operators",
    "cal_std_mean",
    "cal_std_mean_simd",
    "cal_cpr",
    "cal_longest_dd_recover",
    "cal_max_dd",
    "cal_all_largest_indicators",
    "cal_all_longest_indicators",
    "cal_rolling_gain_loss",
    "CompiledGraph",
    "GraphCompileError",
    "GraphCompiler",
    "AdaptivePlanner",
    "ExecutionPlan",
    "PlannerConfig",
    "AdaptiveScheduler",
    "GraphExecutionResult",
    "PreparedGraphExecution",
    "SharedArrayDescriptor",
    "SharedArrayOwner",
    "SharedInputBundle",
]
