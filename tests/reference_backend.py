"""Explicit execution mode for independent test references, never production.

NJIT remains the default for existing benchmark callers. Portable wheel tests
can explicitly interpret the same frozen bodies where Numba has no wheel.
Missing/disabled NJIT is an error, never an automatic Python fallback.
"""

import os

BACKEND = os.environ.get("CALMETRICS_TEST_REFERENCE_BACKEND", "njit")
if BACKEND not in {"njit", "python"}:
    raise RuntimeError("CALMETRICS_TEST_REFERENCE_BACKEND must be njit or python")

if BACKEND == "njit":
    from numba import config, njit

    if config.DISABLE_JIT:
        raise RuntimeError("NJIT reference requires NUMBA_DISABLE_JIT=0")
else:
    def njit(**options):
        def decorate(function):
            return function
        return decorate


def require_njit_reference():
    """Fail before any performance measurement with an interpreted oracle."""
    if BACKEND != "njit":
        raise RuntimeError("Performance measurement requires the njit reference backend")
    from numba import config

    if config.DISABLE_JIT:
        raise RuntimeError("NJIT reference requires NUMBA_DISABLE_JIT=0")
