"""Measure explicit export costs; not a C++/NJIT numerical performance gate."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from calmetrics_engine import GraphCompiler, build_info, excel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("small_scalar", GraphCompiler({"x": "series"}).compile("mean(x)"),
         {"x": np.arange(1., 64.)}),
        ("shared_timeseries", GraphCompiler({"x": "series"}).compile(
            {"mean": "rolling_mean(x,20)", "std": "rolling_std(x,20)",
             "drawdown": "drawdown_series(x)", "mask": "x>100"}, result_format="typed"),
         {"x": np.arange(1., 253.)}),
    ]
    results = []
    for name, graph, inputs in cases:
        start = time.perf_counter()
        plan = excel.plan(graph, inputs)
        planned = time.perf_counter()
        result = excel.export(plan, args.output / (name + ".xlsx"))
        finished = time.perf_counter()
        results.append({"case": name, "plan_seconds": planned - start,
                        "reference_and_write_seconds": finished - planned,
                        "total_seconds": finished - start,
                        "cells": result["cells"], "formula_characters": result["formula_characters"],
                        "snapshot_bytes": result["snapshot_bytes"], "xlsx_bytes": result["bytes"],
                        "upper_bound": result["upper_bound"]})
    report = {"build": build_info(), "cases": results,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                        (1 if sys.platform == "darwin" else 1024),
              "verification": "NOT_RECALCULATED"}
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
