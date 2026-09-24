import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from platform_inventory import scan, main


def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "kernels.py").write_text("""
from numba import njit
@njit('float64(float64)')
def compiled(x):
    return x * x
def wrapper(x):
    return compiled(x)
def undecorated(x):
    return x.sum()
def factory():
    def local(x):
        return x + 1
    return njit(local)
def repeated(x):
    return x
def repeated(x):
    return x+1
def repeated(x):
    return x+2
registered = njit(undecorated)
""")
    (tmp_path / "caller.py").write_text("from kernels import wrapper as kernel\ndef call(x):\n    return kernel(x)\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_fixture.py").write_text("def helper():\n    return 3\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    return tmp_path


def test_inventory_retains_uncompiled_dynamic_nested_and_redefined_functions(tmp_path):
    result = scan(repository(tmp_path))
    rows = result["functions"]
    assert len(rows) == 9
    assert len({row["source_id"] for row in rows}) == len(rows)
    assert sum(row["compiled_decorator"] for row in rows) == 1
    assert len(result["dynamic_sites"]) == 2  # The static decorator is not a dynamic compilation.
    call = next(row for row in rows if row["symbol"] == "call")
    assert call["calls"][0]["resolved"] == ["kernels.py::wrapper"]
    assert all(row["service_verified"] == "not_verified" for row in rows)
    assert next(row for row in rows if row["symbol"] == "undecorated")["numeric_evidence"]
    assert not any(row["source_path"].startswith("tests/") for row in rows)


def test_source_drift_fails_even_when_head_did_not_change(tmp_path, monkeypatch):
    source = repository(tmp_path)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(scan(source)))
    monkeypatch.setattr(sys, "argv", ["platform_inventory", str(source), "--check", str(inventory)])
    main()
    with (source / "kernels.py").open("a") as output:
        output.write("\ndef newly_added(x):\n    return x*x\n")
    with pytest.raises(SystemExit, match="drift"):
        main()


def test_compiler_import_aliases_are_recognized(tmp_path):
    source = repository(tmp_path)
    (source / "kernels.py").write_text("""
from numba import njit as native
import numba as nb
@native('float64(float64)')
def first(x):
    return x + 1
@nb.njit('float64(float64)')
def second(x):
    return x + 2
def factory():
    return native(first)
registered = native(second)
""")
    result = scan(source)
    assert sum(row["compiled_decorator"] for row in result["functions"]) == 2
    assert any(row["source_id"].endswith("::factory") for row in result["dynamic_sites"])
    assert any(row["source_id"] == "kernels.py::<module>" for row in result["dynamic_sites"])
