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

        if any(
            name.endswith((".cpp", ".hpp")) or name.startswith(("tests/", "tools/"))
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
        "cpp/include/calmetrics_engine/array_view.hpp",
        "cpp/include/calmetrics_engine/calendar.hpp",
        "cpp/include/calmetrics_engine/finance.hpp",
        "cpp/include/calmetrics_engine/numeric.hpp",
        "cpp/include/calmetrics_engine/parallel.hpp",
        "cpp/finance/statistics.cpp",
        "cpp/finance/drawdown.cpp",
        "cpp/finance/streaks.cpp",
        "cpp/finance/rolling.cpp",
        "tests/test_api.py",
        "tests/test_regression.py",
        "tests/native_tests.cpp",
        "tests/data/legacy_reference.json",
        "tools/check_dist.py",
        "docs/architecture.md",
    }
    required |= {f"src/{name}" for name in PYTHON_FILES}
    if missing := required - names:
        raise ValueError(f"{path.name}: missing source files {sorted(missing)}")

    forbidden = (".venv", ".build", "docs/.ai-hermes-user-memory", "dist/", ".git/")
    if any(name.startswith(forbidden) for name in names):
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
