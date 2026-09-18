"""Golden outputs were produced by the actual pre-upgrade C++ implementations."""

import json
from pathlib import Path

import numpy as np
import pytest

import my_ctools

CASES = json.loads((Path(__file__).parent / "data/legacy_reference.json").read_text())["cases"]


def decode(value):
    if isinstance(value, dict):
        if "array" in value:
            return np.asarray(value["array"], dtype=value["dtype"]).reshape(value["shape"])
        if "tuple" in value:
            return tuple(decode(item) for item in value["tuple"])
        return {key: decode(item) for key, item in value.items()}
    return value


def assert_result(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        if expected.dtype.kind in "iu":
            assert actual.dtype == np.int64
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual.dtype == np.float64
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14, equal_nan=True)
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            assert_result(got, want)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_result(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("threads", [1, 2, 0])
def test_legacy_numerical_contract(case, threads):
    args = [decode(value) for value in case["args"]]
    snapshots = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]
    result = getattr(my_ctools, case["name"])(*args, n_threads=threads)
    assert_result(result, decode(case["expected"]))
    for arg, snapshot in zip(args, snapshots, strict=True):
        if isinstance(arg, np.ndarray):
            np.testing.assert_array_equal(arg, snapshot)
