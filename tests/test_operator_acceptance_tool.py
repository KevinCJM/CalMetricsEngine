"""Acceptance reports must fail closed for missing or invalid replay evidence."""

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "operator_acceptance_tool",
    Path(__file__).resolve().parents[1] / "tools/check_all_operator_cases.py",
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.mark.parametrize("case", ["pass", "missing", "dtype", "value"])
def test_replay_exit_matches_evidence(case, tmp_path, monkeypatch):
    expected = np.array([1], dtype=np.int64)
    actual = np.array([1 if case != "value" else 2],
                      dtype=np.float64 if case == "dtype" else np.int64)
    fixtures = {} if case == "missing" else {
        "example": {"test": "fixture", "elements": 1,
                    "values": ((expected,), {}, expected)},
    }
    fixture_file, output = tmp_path / "fixture.pkl", tmp_path / "report.json"
    fixture_file.write_bytes(pickle.dumps(fixtures))
    monkeypatch.setattr(sys, "argv", ["check", "--fixtures", str(fixture_file),
                                     "--output", str(output)])
    monkeypatch.setattr(tool, "build_info", lambda: {})
    monkeypatch.setattr(tool.op, "catalog", lambda: [{"id": "example", "opcode": 1}])
    monkeypatch.setattr(tool.op, "get", lambda name: lambda *args, **kwargs: actual.copy())
    with pytest.raises(SystemExit) as result:
        tool.main()
    assert result.value.code == (0 if case == "pass" else 1)
    row = json.loads(output.read_text())["operators"][0]
    assert row["status"] == {
        "pass": "PASS", "missing": "NO_REPLAYABLE_DIRECT_CASE",
        "dtype": "FAIL", "value": "FAIL",
    }[case]
