"""Measured representation must retain full-rank and awkward basis domains."""

import numpy as np
import pytest

from calmetrics_engine import operators as op

from .test_guarded_lu_graph import count
from .test_measured_concat_solve import LIBRARY, fixture, original, run
from .test_solve_library import same_ieee


def coefficients(case, n=1031):
    rng = np.random.default_rng(8491)
    if case == "rank64":
        factors = rng.normal(size=(n, 64)) * 0.004
        h = factors @ factors.T + np.diag(np.linspace(0.5, 1.0, n))
    elif case == "dense_asymmetric":
        h = rng.normal(size=(n, n)) * 0.001
        h[np.diag_indices(n)] += 2.0
    elif case == "missed_samples":
        vector = np.zeros(n)
        vector[533:611] = rng.normal(size=78) * 0.02
        vector[71:98] = rng.normal(size=27) * 0.02
        h = np.outer(vector, vector) + np.diag(np.linspace(0.5, 1.0, n))
    elif case == "permuted_rows":
        factors = rng.normal(size=(n, 8)) * 0.01
        h = factors @ factors.T + np.diag(np.linspace(0.5, 1.0, n))
        h = h[np.roll(np.arange(n), 23)]
    elif case == "scaled_small":
        factors = rng.normal(size=(n, 8)) * 0.01
        h = (factors @ factors.T + np.diag(np.linspace(0.5, 1.0, n))) * 1e-5
    else:
        raise AssertionError(case)
    return h


@pytest.mark.parametrize(
    "case", ["rank64", "dense_asymmetric", "missed_samples", "permuted_rows", "scaled_small"]
)
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_full_rank_and_rank_admission_math_and_input_ownership(case, layout, order):
    data = fixture(n=1031, columns=2)
    data["h"] = np.array(coefficients(case), order=layout)
    for value in data.values():
        value.flags.writeable = False
    saved = {key: value.copy() for key, value in data.items()}
    a, b = original(data)
    try:
        expected = op.solve_multi(a, b)
    except ValueError as expected_error:
        with pytest.raises(ValueError) as actual_error:
            run(data, order)
        assert str(actual_error.value) == str(expected_error)
        for key, original_value in data.items():
            same_ieee(original_value, saved[key])
        return
    result = run(data, order)
    value = result.outputs[0].values[0]
    np.testing.assert_allclose(value, expected, rtol=1e-10, atol=4.9e-5)
    # Componentwise original-equation residual remains meaningful when scaling
    # produces large directions and cancellation in the zero constraint RHS.
    equation = np.einsum("ij,jk->ik", a, value, optimize=False) - b
    magnitude = np.einsum("ij,jk->ik", np.abs(a), np.abs(value), optimize=False) + np.abs(b)
    assert np.all(np.abs(equation) <= 1e-10 * magnitude + 1e-13)
    assert count(result, "accelerate_lu_attempts") == LIBRARY
    assert (
        count(result, "accelerate_lu_calls") + count(result, "accelerate_lu_fallbacks") == LIBRARY
    )
    assert not value.flags.writeable
    assert value.flags.c_contiguous if order == "C" else value.flags.f_contiguous
    for key, original_value in data.items():
        same_ieee(original_value, saved[key])
        assert not np.shares_memory(original_value, value)
