"""Replay successful regression inputs for every canonical direct operator.

Collection is a separate pytest run; measurement never includes test collection.
Only load the local pickle produced by this tool. Cases retain values/dtypes but
are copied to contiguous storage. These are native scalar/auto smoke timings,
NOT a warmed NJIT comparison or proof of large-workload performance coverage.
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import pickle
import statistics
import time
from pathlib import Path

import numpy as np

from calmetrics_engine import build_info
from calmetrics_engine import operators as op


class Collector:
    def __init__(self):
        self.cases = {}
        self.coverage = {}
        self.current_test = ""
        self.originals = {}

    def pytest_runtest_setup(self, item):
        self.current_test = item.nodeid

    def capture(self, name, args, kwargs, result):
        tests = self.coverage.setdefault(name, set())
        tests.add(self.current_test)
        if any(key in kwargs for key in ("out", "workspace")):
            return
        if kwargs.get("audit"):
            result = result[0]
        count = sum(x.size for x in args if isinstance(x, np.ndarray))
        count += sum(x.size for x in kwargs.values() if isinstance(x, np.ndarray))
        # Bound snapshots; avoid replacing useful nonempty samples by giant tests.
        if count > 131072 or count <= self.cases.get(name, {}).get("elements", -1):
            return
        try:
            values = copy.deepcopy((args, {k: v for k, v in kwargs.items() if k not in {"simd", "audit"}}, result))
            pickle.dumps(values)
        except (TypeError, ValueError, RuntimeError):
            return
        self.cases[name] = dict(elements=count, test=self.current_test, values=values)

    def wrap(self, function, name):
        @functools.wraps(function)
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            self.capture(name, args, kwargs, result)
            return result
        return call

    def install(self):
        collector = self

        class Handle:
            def __init__(self, original):
                self.original = original

            def __getattr__(self, name):
                return getattr(self.original, name)

            def __call__(self, *args, **kwargs):
                result = self.original(*args, **kwargs)
                collector.capture(self.original.spec["id"], args, kwargs, result)
                return result

        for name in [s["id"] for s in op.catalog()]:
            self.originals[name] = getattr(op, name)
            setattr(op, name, self.wrap(getattr(op, name), name))
        for name in ("get", "get_by_opcode"):
            original = getattr(op, name)
            self.originals[name] = original
            setattr(op, name, lambda key, original=original: Handle(original(key)))
        original_call = op.call
        self.originals["call"] = original_call

        def call(name, *args, **kwargs):
            result = original_call(name, *args, **kwargs)
            self.capture(name, args, kwargs, result)
            return result

        op.call = call

    def restore(self):
        for name, original in self.originals.items():
            setattr(op, name, original)


def equivalent(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            equivalent(a, e)
        return
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape and a.dtype == e.dtype
    if e.dtype.kind in "biu":
        np.testing.assert_array_equal(a, e)
    else:
        np.testing.assert_allclose(a, e, rtol=2e-12, atol=2e-14, equal_nan=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        import pytest
        collector = Collector()
        collector.install()
        try:
            status = pytest.main(["tests", "-q"], plugins=[collector])
        finally:
            collector.restore()
        args.fixtures.write_bytes(pickle.dumps(collector.cases))
        args.output.write_text(json.dumps({
            "pytest_exit_code": int(status), "build": build_info(),
            "coverage": {k: sorted(v) for k, v in collector.coverage.items()},
            "replayable": sorted(collector.cases),
        }, indent=2))
        raise SystemExit(status)
    fixtures = pickle.loads(args.fixtures.read_bytes())
    rows = []
    for spec in op.catalog():
        name = spec["id"]
        row = {"operator": name, "opcode": spec["opcode"]}
        if name not in fixtures:
            row["status"] = "NO_REPLAYABLE_DIRECT_CASE"
            rows.append(row)
            continue
        case = fixtures[name]
        values, kwargs, expected = case["values"]
        function = op.get(name)
        row.update(test=case["test"], elements=case["elements"])
        try:
            for policy in ("scalar", "auto"):
                run = functools.partial(function, *values, **kwargs, simd=policy)
                equivalent(run(), expected)
                start = time.perf_counter_ns()
                run()
                pilot = time.perf_counter_ns() - start
                loops = max(1, min(1000, int(2e6 / max(1, pilot))))
                durations = []
                for _ in range(15):
                    start = time.perf_counter_ns()
                    for _ in range(loops):
                        run()
                    durations.append((time.perf_counter_ns() - start) / loops)
                row[policy + "_median_ns"] = statistics.median(durations)
            row["status"] = "PASS"
        except Exception as error:
            row.update(status="FAIL", error=str(error))
        rows.append(row)
    args.output.write_text(json.dumps({
        "build": build_info(), "scope": "contiguous native scalar/auto replay; no NJIT speed claim",
        "operators": rows,
    }, indent=2))
    print({state: sum(r["status"] == state for r in rows)
           for state in sorted({r["status"] for r in rows})})
    raise SystemExit(0 if rows and all(r["status"] == "PASS" for r in rows) else 1)


if __name__ == "__main__":
    main()
