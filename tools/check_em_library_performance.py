"""Supplemental GMM/HMM old/new/NJIT comparison with complete public projections.

Design: preserve the 24 cases (16/252/2048/32768 rows, direct/ordinary/prepared,
GMM/HMM), explicit identical starting parameters, 12 iterations and tolerance 0.
Use the same public projections and axis declarations as check_latent_performance,
plus em_converged so arrays, int64 iteration counts and bool state all participate.
No nominal model is a public DAG root and no output is flattened or packed merely
for comparison. Direct calls include the model plus every projection; ordinary and
prepared calls return the same heterogeneous outputs through a shared-CSE graph.
Both access every public output field inside the measured callable and retain
the raw Result plus the projected tuple in bench.MaterializedResult. Thus lazy
ndarray/scalar wrappers are included in timing, heap and RSS on native entries,
matching the already materialized direct/NJIT projection return contract.

Boundary: the reference reuses the frozen EM algorithm and only appends its exact
False convergence state for tolerance 0. NJIT .compile() prepares the actual readonly
signature without numerical execution. Inputs/imports/compilation precede memory
measurement; graph setup and the FIRST numerical invocation stay inside the shared
heap/RSS epoch. All return allocation/projection work remains inside timed calls.
Float parity uses the existing EM rtol=atol=1e-9; integer/bool fields remain exact.
This repository supplemental tool does not replace the full latent/Excel acceptance.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
# The shared harness sets CPU limits before importing numerical packages.
import check_large_matrix_performance as bench  # noqa: E402
import numpy as np  # noqa: E402

_references = {}
MODELS = {
    'gmm': (
        'gaussian_mixture_em', ('x', 'w', 'm', 'v'),
        ('mixture_weights', 'mixture_means', 'mixture_variances'),
    ),
    'hmm': (
        'gaussian_hmm_em', ('x', 'w', 'a', 'm', 'v'),
        ('hmm_initial', 'hmm_transition', 'hmm_means', 'hmm_variances'),
    ),
}
DIAGNOSTICS = ('em_iterations', 'em_log_likelihood', 'em_converged')


def cases():
    return [
        dict(name=name, n=n, d=d, k=k, iterations=12, layout='c', entry=entry,
             rtol=1e-9, atol=1e-9)
        for name in ('gmm', 'hmm')
        for n, d, k in ((16, 2, 2), (252, 5, 3), (2048, 8, 4), (32768, 8, 4))
        for entry in ('ordinary', 'direct', 'prepared')
    ]


def inputs(case):
    rng = np.random.default_rng(20260927)
    n, d, k = case['n'], case['d'], case['k']
    x = rng.normal(size=(n, d)) + (np.arange(n) % k)[:, None] * 2
    data = dict(x=x, w=np.full(k, 1 / k), m=x[:k].copy(),
                v=np.tile(x.var(axis=0), (k, 1)))
    if case['name'] == 'hmm':
        data['a'] = np.full((k, k), .1 / (k - 1))
        np.fill_diagonal(data['a'], .9)
    for array in data.values():
        array.flags.writeable = False
    return data


def setup(case, backend, data):
    name, keys, fields = MODELS[case['name']]
    projections = fields + DIAGNOSTICS
    iterations = case['iterations']
    args = tuple(data[key] for key in keys)
    if backend == 'njit':
        from numba import njit, typeof

        if not _references:
            sys.path.insert(0, str(ROOT))
            from tests.latent_reference import gmm_em_reference, hmm_em_reference

            @njit(nogil=True)
            def gmm(x, w, m, v, count):
                weights, means, variances, completed, likelihood = gmm_em_reference(
                    x, w, m, v, count, 0.0
                )
                return weights, means, variances, completed, likelihood, False

            @njit(nogil=True)
            def hmm(x, w, a, m, v, count):
                initial, transition, means, variances, completed, likelihood = hmm_em_reference(
                    x, w, a, m, v, count, 0.0
                )
                return initial, transition, means, variances, completed, likelihood, False

            _references.update(gmm=gmm, hmm=hmm)
        function = _references[case['name']]
        signature = tuple(typeof(item) for item in (*args, iterations))
        if signature not in function.signatures:
            function.compile(signature)
        return None, lambda: function(*args, iterations)

    from calmetrics_engine import AdaptiveScheduler, GraphCompiler
    from calmetrics_engine import operators as op

    if case['entry'] == 'direct':
        def call():
            model = op.call(name, *args, iterations, 0.0)
            return tuple(getattr(op, projection)(model) for projection in projections)

        return None, call
    definitions = {
        'x': dict(kind='matrix', axes=['scenario', 'factor'], shape=['N', 'D']),
        'w': dict(kind='vector', axes=['component'], shape=['K']),
        'm': dict(kind='matrix', axes=['component', 'factor'], shape=['K', 'D']),
        'v': dict(kind='matrix', axes=['component', 'factor'], shape=['K', 'D']),
        'a': dict(kind='matrix', axes=['component', 'component'], shape=['K', 'K']),
    }
    expression = f"{name}({','.join(keys)},{iterations},0)"
    graph = GraphCompiler({key: definitions[key] for key in keys}).compile(
        [f'{projection}({expression})' for projection in projections], result_format='typed'
    )
    engine = AdaptiveScheduler(cpu_budget=1)
    bounds = np.zeros(1, np.int64)
    selected = {key: data[key] for key in keys}
    try:
        if case['entry'] == 'prepared':
            prepared = engine.prepare_execution(graph, selected, bounds, bounds)
            return engine, lambda: bench.MaterializedResult(prepared.run_snapshot())
        return engine, lambda: bench.MaterializedResult(
            engine.execute(graph, selected, bounds, bounds)
        )
    except BaseException:
        engine.close()
        raise


def invariants(case, data, outputs):
    _, _, fields = MODELS[case['name']]
    if not isinstance(outputs, tuple) or len(outputs) != len(fields) + len(DIAGNOSTICS):
        raise AssertionError('incomplete EM projection results')
    arrays = tuple(np.asarray(value) for value in outputs)
    k, d = case['k'], case['d']
    expected_shapes = (
        ((k,), (k, d), (k, d), (), (), ()) if case['name'] == 'gmm'
        else ((k,), (k, k), (k, d), (k, d), (), (), ())
    )
    for value, shape in zip(arrays, expected_shapes, strict=True):
        assert value.shape == shape
        assert np.all(np.isfinite(value))
    completed, likelihood, converged = arrays[-3:]
    assert completed.dtype == np.dtype(np.int64)
    assert converged.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(completed, case['iterations'])
    np.testing.assert_array_equal(converged, False)
    for value in arrays[:-3] + (likelihood,):
        assert value.dtype == np.dtype(np.float64)
    assert np.all(arrays[-4] >= 1e-5)
    assert np.all(arrays[0] >= 0)
    np.testing.assert_allclose(arrays[0].sum(), 1.0, rtol=1e-9, atol=1e-9)
    if case['name'] == 'hmm':
        assert np.all(arrays[1] >= 0)
        np.testing.assert_allclose(arrays[1].sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)
    return dict(finite_model=True, exact_iteration_count=True, exact_convergence_state=True,
                nonnegative_variances=True, projection_count=len(outputs))


def audit(case, backend, data, result):
    if backend == 'njit':
        return {}
    if case['entry'] != 'direct':
        return result.raw.audit
    from calmetrics_engine import operators as op

    name, keys, _ = MODELS[case['name']]
    _, report = op.call(
        name, *(data[key] for key in keys), case['iterations'], 0.0, audit=True
    )
    # Diagnostic model invocation is outside timing and allocation epochs.
    return report


bench.cases = cases
bench.inputs = inputs
bench.setup = setup
bench.invariants = invariants
bench.audit = audit
bench.__file__ = __file__
if __name__ == '__main__':
    bench.main()
