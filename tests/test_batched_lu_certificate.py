"""Large KKT roots formerly replayed after a mixed-scale error envelope."""
import hashlib

import numpy as np
import pytest

from .test_concat_solve_workspace import LIBRARY, graph, run
from .test_guarded_lu_graph import count


@pytest.mark.parametrize('layout', ['c', 'f'])
def test_large_mixed_scale_root_uses_guarded_library_and_retains_results(layout):
    n, active = 8176, 16
    rng = np.random.default_rng(20261004)
    factors = rng.normal(size=(n, 8)) * .01
    coefficient = factors @ factors.T
    coefficient[np.diag_indices(n)] += np.linspace(.5, 1., n) + 1e-10
    constraints = np.zeros((active, n))
    constraints[0] = 1.
    for row in range(1, active):
        constraints[row, n-row] = -1.
    values = {name: np.array(value, order=layout.upper()) for name, value in dict(
        h=coefficient, g=constraints, z=np.zeros((active, active)),
        b=-rng.normal(size=(n, 1)), zr=np.zeros((active, 1))).items()}
    del coefficient, constraints, factors

    def digest(value):
        return hashlib.sha256(value.tobytes(order='A')).hexdigest()

    before = {name: digest(value) for name, value in values.items()}
    for value in values.values():
        value.flags.writeable = False
    first = run(graph(layout.upper()), values)
    old = first.outputs[0].values[0]
    saved = old.copy()
    assert count(first, 'accelerate_lu_calls') == LIBRARY
    assert count(first, 'accelerate_lu_fallbacks') == 0
    assert first.audit['boundary_copy_bytes'] == 0
    assert count(first, 'operator_workspace_capacity_bytes') < values['h'].nbytes

    changed = dict(values, b=values['b'] * 2.)
    changed['b'].flags.writeable = False
    second = run(graph(layout.upper()), changed)
    current = second.outputs[0].values[0]
    assert count(second, 'accelerate_lu_calls') == LIBRARY
    np.testing.assert_array_equal(old, saved)
    assert not np.shares_memory(current, old)
    np.testing.assert_allclose(current, old * 2., rtol=1e-10, atol=4.9e-5)
    for inputs, result in ((values, old), (changed, current)):
        direction, multiplier = result[:n], result[n:]
        stationarity = np.einsum('ij,jk->ik', inputs['h'], direction, optimize=False)
        stationarity += np.einsum('ji,jk->ik', inputs['g'], multiplier, optimize=False)
        feasibility = np.einsum('ij,jk->ik', inputs['g'], direction, optimize=False)
        np.testing.assert_allclose(stationarity, inputs['b'], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(feasibility, inputs['zr'], rtol=0., atol=1e-10)
        assert result.flags.f_contiguous if layout == 'f' else result.flags.c_contiguous
    assert {name: digest(value) for name, value in values.items()} == before
