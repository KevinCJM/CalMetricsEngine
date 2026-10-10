"""Test-only continuous allocation accounting and warm RSS observation.

The heap epoch spans setup, first result release, warmup and warm execution.
Peaks already include live setup/workspace; stage peaks must never be added.
RSS remains a supplementary page-level observation, not an allocation counter.
"""

from __future__ import annotations

import ctypes
import gc
import threading
import time


class HeapStats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "allocations", "frees", "allocated_bytes", "live_bytes", "peak_bytes",
        "untracked_frees", "table_overflows",
    )]


def measure_heap(setup, library, repeats=21):
    """Return owner/callable plus measured stage totals, retaining no result."""
    probe = ctypes.CDLL(library)
    stages = {name: HeapStats() for name in (
        "setup_heap", "cold_heap", "cold_retained_heap", "warm_baseline_heap",
        "warm_heap", "warm_retained_heap",
    )}
    final = HeapStats()
    gc.collect()
    owner = function = result = None
    probe.cme_heap_begin()
    try:
        owner, function = setup()
        probe.cme_heap_snapshot(ctypes.byref(stages["setup_heap"]))
        probe.cme_heap_reset_peak()
        result = function()
        probe.cme_heap_snapshot(ctypes.byref(stages["cold_heap"]))
        del result
        probe.cme_heap_snapshot(ctypes.byref(stages["cold_retained_heap"]))
        for _ in range(5):
            function()
        gc.collect()
        probe.cme_heap_snapshot(ctypes.byref(stages["warm_baseline_heap"]))
        probe.cme_heap_reset_peak()
        result = None
        for _ in range(repeats):
            result = function()
        probe.cme_heap_snapshot(ctypes.byref(stages["warm_heap"]))
        del result
        probe.cme_heap_snapshot(ctypes.byref(stages["warm_retained_heap"]))
    except BaseException:
        probe.cme_heap_end(ctypes.byref(final))
        if owner:
            owner.close()
        raise
    probe.cme_heap_end(ctypes.byref(final))
    if final.table_overflows:
        if owner:
            owner.close()
        raise RuntimeError("allocation tracker overflow")
    report = {
        key: {name: getattr(stats, name) for name, _ in stats._fields_}
        for key, stats in stages.items()
    }
    report["heap_schema"] = "continuous-live-1"
    report["cold_peak_bytes"] = stages["cold_heap"].peak_bytes
    report["warm_peak_bytes"] = stages["warm_heap"].peak_bytes
    return owner, function, report


def measure_rss(function, repeats=100):
    """Warm monitor before baseline; report an idle control without subtracting it."""
    import psutil

    process = psutil.Process()
    stop, ready = threading.Event(), threading.Event()
    peak = [0]

    def monitor():
        peak[0] = process.memory_info().rss
        ready.set()
        while not stop.wait(0.0005):
            peak[0] = max(peak[0], process.memory_info().rss)

    gc.collect()
    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        ready.wait()
        # Allow thread/psutil allocator initialization before the control window.
        time.sleep(0.01)
        idle_base = process.memory_info().rss
        peak[0] = idle_base
        time.sleep(0.01)
        idle_delta = max(0, peak[0] - idle_base)
        base = process.memory_info().rss
        peak[0] = base
        result = None
        for _ in range(repeats):
            result = function()
        observed = max(peak[0], process.memory_info().rss)
    finally:
        stop.set()
        thread.join()
    del result
    return {"rss_baseline": base, "rss_peak": observed,
            "rss_delta": observed - base, "rss_idle_delta": idle_delta,
            "rss_schema": "warm-monitor-control-1"}
