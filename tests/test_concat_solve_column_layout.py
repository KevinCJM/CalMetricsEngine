"""Asymmetric coefficient and out-of-domain replay of private F construction."""

import numpy as np
import pytest

from .test_measured_concat_solve import LIBRARY, count, fixture, original, run


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_asymmetric_coefficients_keep_matrix_orientation_and_old_results(layout, order):
    data = fixture(n=1024, columns=3, constraints=1)
    h = np.eye(1024) * 2.0
    h[np.arange(1, 1024), np.arange(1023)] = 0.125
    data["h"] = np.array(h, order=layout)
    data["g"] = np.full((1, 1024), 0.05, order=layout)
    for value in data.values():
        value.flags.writeable = False
    a, b = original(data)
    expected = np.linalg.solve(a, b)
    first = run(data, order)
    saved = first.outputs[0].values[0].copy()
    second = run(data, order)
    for result in (first, second):
        value = result.outputs[0].values[0]
        np.testing.assert_allclose(value, expected, atol=4.9e-5, rtol=1e-11)
        np.testing.assert_allclose(a @ value, b, atol=1e-10, rtol=1e-11)
        assert count(result, "accelerate_lu_calls") == LIBRARY
        assert value.flags.c_contiguous if order == "C" else value.flags.f_contiguous
        assert not value.flags.writeable
        assert all(not np.shares_memory(value, x) for x in data.values())
    np.testing.assert_array_equal(first.outputs[0].values[0], saved)
    assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])


@pytest.mark.parametrize("columns", [0, 65])
@pytest.mark.parametrize("order", ["C", "F"])
def test_rhs_geometry_outside_admission_keeps_canonical_ownership(columns, order):
    data = fixture(n=1024, columns=columns, constraints=1)
    for value in data.values():
        value.flags.writeable = False
    a, b = original(data)
    expected = np.linalg.solve(a, b) if columns else np.empty((a.shape[0], 0))
    result = run(data, order)
    value = result.outputs[0].values[0]
    np.testing.assert_allclose(value, expected, atol=4.9e-5, rtol=1e-11)
    assert count(result, "accelerate_lu_attempts") == 0
    assert value.shape == expected.shape and not value.flags.writeable
    assert all(not np.shares_memory(value, x) for x in data.values())


@pytest.mark.parametrize("dominant", ["C", "F", "padded_F", "padded_C"])
def test_actual_leaf_layout_selects_private_construction(dominant):
    from .test_concat_solve_workspace import (
        compact_factor_copies,
    )
    from .test_concat_solve_workspace import (
        fixture as block_fixture,
    )
    from .test_concat_solve_workspace import (
        graph as block_graph,
    )
    from .test_concat_solve_workspace import (
        original as block_original,
    )
    from .test_concat_solve_workspace import (
        run as block_run,
    )

    # The large H block determines the layout even when thin leaves disagree.
    values = block_fixture("c" if dominant.endswith("F") else "f", active=8)
    order = "F" if dominant.endswith("F") else "C"
    if dominant.startswith("padded"):
        storage = np.zeros((1028, 1028), order=order)
        values["h"] = storage[2:-2, 2:-2]
        values["h"][:] = np.diag(np.linspace(0.5, 1.0, 1024) + 1e-10)
    else:
        values["h"] = np.array(values["h"], order=order)
    saved = {key: value.copy() for key, value in values.items()}
    for value in values.values():
        value.flags.writeable = False
    coefficient, rhs, expected = block_original(values)
    result = block_run(block_graph("F"), values)
    actual = result.outputs[0].values[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=4.9e-5)
    transpose_bytes = 0 if order == "F" else coefficient.nbytes
    expected_copies = (
        LIBRARY * (transpose_bytes + compact_factor_copies(coefficient.shape[0]))
        + (1 + LIBRARY) * rhs.nbytes
    )
    assert count(result, "algorithm_copy_bytes") == expected_copies
    assert count(result, "accelerate_lu_calls") == LIBRARY
    assert count(result, "input_copy_bytes") == 0
    assert actual.flags.f_contiguous and not actual.flags.writeable
    for key, value in values.items():
        np.testing.assert_array_equal(value, saved[key])
        assert not np.shares_memory(actual, value)
