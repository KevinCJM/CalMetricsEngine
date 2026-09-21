"""Parity against frozen outputs from the actual BetterSaaTaa NJIT implementations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from calmetrics_engine import operators as op

REFERENCE = json.loads((Path(__file__).parent / "data/canonical_reference.json").read_text())
CASES = REFERENCE["cases"]


def decode(value):
    if isinstance(value, dict):
        if "array" in value:
            return np.asarray([decode(x) for x in value["array"]], dtype=value["dtype"]).reshape(
                value["shape"]
            )
        if "tuple" in value:
            return tuple(decode(x) for x in value["tuple"])
        if "scalar" in value:
            return np.dtype(value["dtype"]).type(value["scalar"])
    if isinstance(value, str) and value in {"nan", "inf", "-inf"}:
        return float(value)
    return value


def assert_reference(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_reference(a, e)
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14, equal_nan=True)
    a, e = np.asarray(actual), np.asarray(expected)
    zero = (a == 0) & (e == 0)
    if np.any(zero):
        np.testing.assert_array_equal(np.signbit(a[zero]), np.signbit(e[zero]))


def test_registry_matches_pinned_source_and_stable_opcodes():
    specs = op.catalog()
    # Frozen external-source evidence remains immutable. New native capabilities
    # have independent cases in test_operator_extensions.py.
    assert len(specs) == 146
    assert [item["id"] for item in specs[:118]] == REFERENCE["canonical_names"]
    assert [item["opcode"] for item in specs] == list(range(1, 147))
    assert len({case["id"] for case in CASES}) == len(CASES)
    successful = {case["operator"] for case in CASES if "expected" in case}
    assert successful == set(REFERENCE["canonical_names"])
    assert (
        REFERENCE["source_sha256"]["typed_numba_kernels.py"]
        == "757251922914384fb4fdb295036ad2550179107d4ca14e88b211cca521e104fb"
    )
    for spec in specs:
        assert callable(getattr(op, spec["id"]))
        assert op.get(spec["id"]).spec == spec
        assert op.get_by_opcode(spec["opcode"]).spec == spec
        assert spec["input_policy"] == "exact_native_dtype_readonly_strided_no_copy"
        assert spec["execution_backend"] == "pybind11_aot"
        if spec["opcode"] > 118:
            continue
        for signature in spec["signatures"]:
            assert (
                signature["parameters"]
                == REFERENCE["parameter_names"][spec["id"]][str(signature["arity"])]
            )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("simd", ["scalar", "auto"])
def test_original_njit_contract(case, simd):
    args = [decode(x) for x in case["args"]]
    arrays = [x for x in args if isinstance(x, np.ndarray)]
    snapshots = [x.copy() for x in arrays]
    function = op.get(case["operator"]) if simd == "scalar" else getattr(op, case["operator"])
    if "error" in case:
        with pytest.raises(ValueError, match=re.escape(case["error"])):
            function(*args, simd=simd)
    else:
        actual = function(*args, simd=simd)
        assert_reference(actual, decode(case["expected"]))
    for array, original in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(array, original)


SUCCESS_CASES = [case for case in CASES if "expected" in case]
ARRAY_CASES = [
    case
    for case in SUCCESS_CASES
    if isinstance(case["expected"], dict) and "array" in case["expected"]
]


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=lambda case: case["id"])
def test_source_named_argument_contract(case):
    args = [decode(x) for x in case["args"]]
    names = REFERENCE["parameter_names"][case["operator"]][str(len(args))]
    kwargs = dict(zip(names, args, strict=True))
    result = op.call(case["operator"], **kwargs)
    assert_reference(result, decode(case["expected"]))


def layout_view(array, layout):
    if layout == "fortran":
        view = np.asfortranarray(array)
    elif layout == "strided":
        shape = list(array.shape)
        shape[-1] *= 2
        owner = np.empty(shape, dtype=array.dtype)
        view = owner[..., ::2]
        view[...] = array
    elif layout == "negative":
        owner = np.empty_like(array)
        view = owner[::-1]
        view[...] = array
    else:
        view = array.view()
    view.flags.writeable = False
    return view


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("layout", ["fortran", "strided", "negative", "readonly"])
def test_zero_copy_layout_matches_source(case, layout):
    args = [decode(x) for x in case["args"]]
    args = [layout_view(x, layout) if isinstance(x, np.ndarray) else x for x in args]
    arrays = [x for x in args if isinstance(x, np.ndarray)]
    expected_addresses = [x.__array_interface__["data"][0] for x in arrays]
    result, audit = op.call(case["operator"], *args, audit=True)
    assert_reference(result, decode(case["expected"]))
    assert audit["input_copy_bytes"] == 0
    assert audit["input_addresses"] == expected_addresses
    assert audit["python_fallback"] == 0
    for actual, encoded in zip(args, case["args"], strict=True):
        if isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, decode(encoded))


@pytest.mark.parametrize("case", ARRAY_CASES, ids=lambda case: case["id"])
def test_preallocated_output_matches_source(case):
    args = [decode(x) for x in case["args"]]
    expected = decode(case["expected"])
    output = np.empty(expected.shape, dtype=expected.dtype, order="C")
    address = output.__array_interface__["data"][0]
    result, audit = op.call(case["operator"], *args, out=output, audit=True)
    assert result is output
    assert result.__array_interface__["data"][0] == address
    assert audit["input_copy_bytes"] == 0
    assert_reference(result, expected)
