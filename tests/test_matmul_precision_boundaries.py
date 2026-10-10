"""Private precision admission must preserve domain and budget boundaries."""

import numpy as np
import pytest

from .test_matmul_terminal_float import execute
from .test_nonnegative_matmul_accelerate import ACCELERATE, chunks


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "case",
    [
        "grid_min",
        "grid_nextabove",
        "grid_nextbelow",
        "maximum",
        "cancellation",
        "near_budget",
        "over_budget",
        "negative_zero",
        "positive_zero",
    ],
)
def test_private_float_domain_and_complete_budget_boundaries(order, case):
    a = np.full((512, 256), 0.01, order=order)
    b = np.full((256, 512), 0.01, order=order)
    expected = True
    if case.startswith("grid_"):
        value = 2.0**-36
        if case == "grid_nextabove":
            value = np.nextafter(value, np.inf)
        elif case == "grid_nextbelow":
            value = np.nextafter(value, 0.0)
            expected = False
        a.fill(value)
    elif case == "maximum":
        a.fill(1.0)
        b.fill(0.001)
    elif case == "cancellation":
        a.fill(1.0)
        a[:, 1::2] = -1.0
        b.fill(0.001)
    elif case in ("near_budget", "over_budget"):
        a.fill(1.0)
        b.fill(0.004 if case == "near_budget" else 0.0065)
        expected = case == "near_budget"
    else:
        a[2, 3] = -0.0 if case == "negative_zero" else 0.0
        expected = False
    a.flags.writeable = b.flags.writeable = False
    before = a.copy(), b.copy()
    raw = a @ b
    result = execute(a, b, order, ["matmul(a,b)*0.5+4.0", "matmul(a,b)*0.25-4.0"])
    for output, want in zip(result.outputs, [raw * 0.5 + 4.0, raw * 0.25 - 4.0], strict=True):
        np.testing.assert_allclose(
            output.values[0], want, rtol=0, atol=0.000049 if expected and ACCELERATE else 1e-11
        )
    np.testing.assert_array_equal(a, before[0])
    np.testing.assert_array_equal(b, before[1])
    if ACCELERATE:
        copies = (a.size + b.size) * 4 + raw.size * 8 if expected else 0
        assert all(c["algorithm_copy_bytes"] == copies for c in chunks(result))
