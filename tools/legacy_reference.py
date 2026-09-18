"""Regenerate regression fixtures from immutable pre-upgrade Git sources.

Run from a development environment with pybind11/NumPy and a C++17 compiler.
The historical SIMD selector is forced to its existing scalar fallback solely
inside a temporary directory, so generating fixtures does not require AVX.
Both generations disable floating-point contraction for compiler-independent
comparison; no historical code is imported into the published package.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "d1ced04"
STATISTICS_BASELINE = "1bf35be^"
MODULES = (
    "cal_std_mean",
    "cal_cpr",
    "cal_max_dd",
    "cal_longest_dd_recover",
    "cal_all_largest_indicators",
    "cal_all_longest_indicators",
    "cal_rolling_gain_loss",
)


def encode(value):
    if isinstance(value, np.ndarray):
        data = value.astype(object)
        if value.dtype.kind == "f":
            data[np.isnan(value)] = None
            data[np.isposinf(value)] = "inf"
            data[np.isneginf(value)] = "-inf"
        return {"array": data.tolist(), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, tuple):
        return {"tuple": [encode(item) for item in value]}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    return value


def load_legacy(directory: Path) -> dict:
    if os.name == "nt":
        raise SystemExit(
            "Fixture generation uses a Unix compiler; checked-in fixtures test Windows."
        )
    includes = shlex.split(
        subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True)
    )
    loaded = {}
    for name in MODULES:
        ref = STATISTICS_BASELINE if name == "cal_std_mean" else BASELINE
        source = subprocess.check_output(
            ["git", "show", f"{ref}:my_ctools/{name}.cpp"], cwd=ROOT, text=True
        )
        if name == "cal_std_mean":
            source = source.replace("#if defined(__x86_64__) || defined(_M_X64)", "#if 0")
            source = source.replace("#elif defined(__aarch64__)", "#elif 0")
        cpp = directory / f"{name}.cpp"
        extension = directory / f"{name}{sysconfig.get_config_var('EXT_SUFFIX')}"
        cpp.write_text(source)
        command = [
            os.environ.get("CXX", "c++"),
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-include",
            "cmath",
            *includes,
            str(cpp),
            "-o",
            str(extension),
        ]
        if platform.system() == "Darwin":
            command.extend(["-undefined", "dynamic_lookup"])
        subprocess.run(command, check=True, capture_output=True, text=True)
        spec = importlib.util.spec_from_file_location(name, extension)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def make_cases(legacy: dict) -> list:
    rng = np.random.default_rng(20260918)
    seeded = rng.normal(0.0005, 0.015, (400, 6))
    seeded[rng.random(seeded.shape) < 0.12] = np.nan
    seeded[:, 0] = np.nan
    seeded[:, 1] = 0.005
    edges = np.array(
        [
            [np.nan, 0.0, 0.1, -0.1, np.nan],
            [0.1, 0.0, 0.1, 0.1, np.nan],
            [np.nan, 0.0, -0.1, 0.1, 0.5],
            [-0.2, 0.0, 0.1, 0.0, np.nan],
            [0.3, 0.0, 0.1, -0.2, np.nan],
            [np.nan, 0.0, -0.1, 0.2, np.nan],
        ]
    )
    arrays = [seeded, edges, np.array([[0.1, np.nan, -0.2, 0.0]])]
    cases = []
    for dataset, values in enumerate(arrays):
        rows, cols = values.shape
        days = (
            np.datetime64("2020-01-01", "ns") + np.arange(rows).astype("timedelta64[D]")
        ).astype(np.int64)
        groups = (np.arange(cols) % 2).astype(np.int32)
        calls = [
            ("cal_std_mean", [values]),
            ("cal_std_mean_simd", [values]),
            ("cal_cpr", [groups, values]),
            ("cal_max_dd", [values, days]),
            ("cal_longest_dd_recover", [values]),
        ]
        for mode in ("positive", "negative"):
            calls.extend(
                [
                    ("cal_all_largest_indicators", [values, days, mode]),
                    ("cal_all_longest_indicators", [values, days, mode]),
                ]
            )
        if rows >= 400:
            start = np.zeros(cols, dtype=np.int64)
            end = np.full(cols, rows - 1, dtype=np.int64)
            start[-1], end[-1] = 5, rows - 10
            for period in ("1M", "3M", "1Y"):
                calls.append(("cal_rolling_gain_loss", [period, values, start, end, days]))
        for name, args in calls:
            module_name = "cal_std_mean" if name == "cal_std_mean_simd" else name
            result = getattr(legacy[module_name], name)(*args)
            cases.append(
                {
                    "id": f"dataset-{dataset}-{name}-{len(cases)}",
                    "name": name,
                    "args": [encode(arg) for arg in args],
                    "expected": encode(result),
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Explicitly replace the fixture file")
    args = parser.parse_args()
    if not args.write:
        parser.error("Regeneration requires --write; normal tests use checked-in fixtures.")
    with tempfile.TemporaryDirectory(prefix="my-ctools-legacy-") as temp:
        cases = make_cases(load_legacy(Path(temp)))
    target = ROOT / "tests/data/legacy_reference.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {"baseline": BASELINE, "statistics_baseline": STATISTICS_BASELINE, "cases": cases},
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    print(f"Wrote {len(cases)} historical API cases to {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
