"""Validate CalMetricsEngine build archives without importing the checkout."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "calmetrics_engine"
DISTRIBUTION = "calmetrics-engine"
PYTHON_FILES = {
    f"{PACKAGE}/__init__.py",
    f"{PACKAGE}/_api.py",
    f"{PACKAGE}/operators.py",
    f"{PACKAGE}/graph.py",
    f"{PACKAGE}/planner.py",
    f"{PACKAGE}/runtime.py",
    f"{PACKAGE}/shared.py",
    f"{PACKAGE}/excel.py",
    f"{PACKAGE}/py.typed",
    f"{PACKAGE}/cal_std_mean.py",
    f"{PACKAGE}/cal_cpr.py",
    f"{PACKAGE}/cal_max_dd.py",
    f"{PACKAGE}/cal_longest_dd_recover.py",
    f"{PACKAGE}/cal_all_largest_indicators.py",
    f"{PACKAGE}/cal_all_longest_indicators.py",
    f"{PACKAGE}/cal_rolling_gain_loss.py",
}


def check_wheel(path: Path, version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = PYTHON_FILES - names
        if missing:
            raise ValueError(f"{path.name}: missing Python files {sorted(missing)}")

        extensions = [name for name in names if name.endswith((".so", ".pyd"))]
        if len(extensions) != 1 or not extensions[0].startswith(f"{PACKAGE}/_native."):
            raise ValueError(f"{path.name}: expected exactly one _native extension")

        worker = f"{PACKAGE}/calmetrics_worker" + (".exe" if "-win" in path.name else "")
        if worker not in names:
            raise ValueError(f"{path.name}: missing standalone native worker")
        if (
            not worker.endswith(".exe")
            and not (archive.getinfo(worker).external_attr >> 16) & 0o111
        ):
            raise ValueError(f"{path.name}: native worker is not executable")

        if any(
            name.endswith((".cpp", ".hpp", ".def")) or name.startswith(("tests/", "tools/"))
            for name in names
        ):
            raise ValueError(f"{path.name}: development sources leaked into wheel")

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"{path.name}: invalid metadata count")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata["Version"] != version:
            raise ValueError(f"{path.name}: version mismatch")
        if metadata["Name"].lower().replace("_", "-") != DISTRIBUTION:
            raise ValueError(f"{path.name}: incorrect distribution name")
        if not any(item.startswith("numpy") for item in metadata.get_all("Requires-Dist", [])):
            raise ValueError(f"{path.name}: missing NumPy runtime dependency")


def check_sdist(path: Path, version: str) -> None:
    prefix = f"{PACKAGE}-{version}/"
    with tarfile.open(path) as archive:
        names = {name.removeprefix(prefix) for name in archive.getnames()}

    required = {
        "pyproject.toml",
        "CMakeLists.txt",
        "README.md",
        "cpp/bindings.cpp",
        "cpp/operator_bindings.cpp",
        "cpp/graph_bindings.cpp",
        "cpp/graph.cpp",
        "cpp/compiler.cpp",
        "cpp/excel.cpp",
        "cpp/excel_budget.cpp",
        "cpp/excel_operators.cpp",
        "cpp/excel_state.cpp",
        "cpp/excel_scopes.cpp",
        "cpp/excel_internal.hpp",
        "cpp/excel_bindings.cpp",
        "cpp/include/calmetrics_engine/excel.hpp",
        "tests/excel_native_tests.cpp",
        "tests/test_excel_export.py",
        "tests/test_excel_parity_tool.py",
        "tests/excel_cases.py",
        "tools/check_excel_parity.py",
        "docs/excel-export.md",
        "cpp/planner.cpp",
        "cpp/native_runtime.cpp",
        "cpp/native_process.cpp",
        "cpp/native_worker_main.cpp",
        "cpp/native_api_bindings.cpp",
        "cpp/graph_binding_utils.hpp",
        "cpp/shared_memory.cpp",
        "cpp/scheduler.cpp",
        "cpp/include/calmetrics_engine/compiler.hpp",
        "cpp/include/calmetrics_engine/planner.hpp",
        "cpp/include/calmetrics_engine/native_runtime.hpp",
        "cpp/include/calmetrics_engine/native_process.hpp",
        "cpp/include/calmetrics_engine/shared_memory.hpp",
        "cpp/include/calmetrics_engine/graph.hpp",
        "cpp/include/calmetrics_engine/operators.hpp",
        "cpp/include/calmetrics_engine/operators.def",
        "cpp/operators/registry.cpp",
        "cpp/operators/elementwise.cpp",
        "cpp/operators/reduction.cpp",
        "cpp/operators/sequence.cpp",
        "cpp/operators/matrix.cpp",
        "cpp/operators/state.cpp",
        "cpp/operators/simd.cpp",
        "cpp/operators/simd_avx2.cpp",
        "cpp/operators/simd_loop.hpp",
        "cpp/include/calmetrics_engine/array_view.hpp",
        "cpp/include/calmetrics_engine/calendar.hpp",
        "cpp/include/calmetrics_engine/finance.hpp",
        "cpp/include/calmetrics_engine/numeric.hpp",
        "cpp/include/calmetrics_engine/parallel.hpp",
        "cpp/include/calmetrics_engine/scheduler.hpp",
        "cpp/finance/statistics.cpp",
        "cpp/finance/drawdown.cpp",
        "cpp/finance/streaks.cpp",
        "cpp/finance/rolling.cpp",
        "tests/test_api.py",
        "tests/test_regression.py",
        "tests/native_tests.cpp",
        "tests/operator_native_tests.cpp",
        "tests/graph_native_tests.cpp",
        "tests/compiler_native_tests.cpp",
        "tests/runtime_native_tests.cpp",
        "tests/test_cpp_first_runtime.py",
        "tests/test_planner_physical_dag.py",
        "tests/test_operator_reference.py",
        "tests/test_phase2_graph_runtime.py",
        "tests/test_operator_independent_parity.py",
        "tests/test_operator_contracts.py",
        "tests/data/canonical_reference.json",
        "tests/data/legacy_reference.json",
        "tools/check_dist.py",
        "tools/capture_canonical_reference.py",
        "tools/benchmark_operators.py",
        "tools/benchmark_operator_memory.py",
        "tools/benchmark_multiworkload.py",
        "tools/benchmark_multiworkload.cpp",
        "tools/benchmark_phase2_graph.py",
        "tools/benchmark_phase2_micro.py",
        "tools/benchmark_cpp_first_control.py",
        "tools/benchmark_cpp_vs_njit_matrix.py",
        "tools/check_phase2_performance.py",
        "docs/architecture.md",
        "docs/cpp-first-design.md",
        "docs/cpp-first-acceptance.md",
        "docs/cpp-vs-njit-benchmark-matrix-2026-09-20.md",
        "docs/planner-physical-dag-optimization-design-2026-09-20.md",
        "docs/planner-physical-dag-optimization-acceptance-2026-09-20.md",
        "docs/phase2-execution-graph-design.md",
        "docs/phase2-execution-graph-acceptance.md",
        "docs/canonical-operators-design.md",
        "docs/canonical-operators.md",
        "docs/canonical-operators-acceptance.md",
    }
    required |= {f"src/{name}" for name in PYTHON_FILES}
    if missing := required - names:
        raise ValueError(f"{path.name}: missing source files {sorted(missing)}")

    forbidden = (".venv", ".build", "docs/.ai-hermes-user-memory", "dist/", ".git/")
    if any(
        name.startswith(forbidden)
        or "__pycache__" in Path(name).parts
        or name.endswith((".pyc", ".pyo"))
        for name in names
    ):
        raise ValueError(f"{path.name}: local/private files leaked into sdist")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag", default="", help="Optional release tag; must equal v<version>")
    args = parser.parse_args()

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if args.tag and args.tag != f"v{version}":
        raise SystemExit(f"Release tag must be v{version}, got {args.tag!r}")

    archives = sorted(args.directory.glob("*.whl")) + sorted(args.directory.glob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"No distribution archives found in {args.directory}")

    for archive in archives:
        if archive.name.endswith(".whl"):
            check_wheel(archive, version)
        else:
            check_sdist(archive, version)
        print(f"OK {archive.name}")


if __name__ == "__main__":
    main()
