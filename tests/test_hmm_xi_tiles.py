"""HMM time tiles retain scalar libm, stopping, failures and buffer ownership."""

import ctypes
import platform
from contextlib import contextmanager

import numpy as np
import pytest

from calmetrics_engine import operators as op

from . import latent_reference as ref
from .test_latent_operators import fixture


def bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def sample(n, d, k, layout="c"):
    values = fixture(n, d, k)
    owners = []
    arrays = []
    for value in values:
        if layout == "f" and value.ndim == 2:
            value = np.asfortranarray(value)
        elif layout == "negative":
            reverse = (slice(None, None, -1),) * value.ndim
            # Preserve the logical sample while giving every axis a negative stride.
            value = value[reverse].copy()[reverse]
        elif layout == "padded":
            owner = np.full(tuple(size + 2 for size in value.shape), -991.0)
            interior = (slice(1, -1),) * value.ndim
            owner[interior] = value
            value = owner[interior]
            owner.flags.writeable = False
            owners.append(owner)
        value.flags.writeable = False
        arrays.append(value)
    return tuple(arrays), owners


def fit(values, iterations=5, tolerance=0.0, **kwargs):
    return op.gaussian_hmm_em(*values, iterations, float(tolerance), **kwargs)


def compare_models(actual, expected, d):
    # Width one excludes existing feature-library paths: all model bits matter.
    # Wider models retain their existing numerical tolerance and exact decisions.
    if d == 1:
        bits(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert op.em_iterations(actual) == op.em_iterations(expected)
    assert bool(op.em_converged(actual)) == bool(op.em_converged(expected))


# Cover every tile boundary and state gate without a Cartesian layout/shape suite.
BOUNDARIES = [
    (1, 1, 1, "c", 3),
    (127, 1, 1, "c", 3),
    (128, 1, 1, "f", 3),
    (129, 1, 2, "negative", 3),
    (127, 1, 3, "c", 3),
    (128, 1, 3, "c", 3),
    (129, 1, 3, "c", 3),
    (255, 1, 3, "c", 3),
    (256, 1, 3, "c", 3),
    (257, 1, 3, "c", 3),
    (258, 1, 3, "c", 3),
    (128, 1, 12, "c", 1),
    (129, 1, 12, "f", 5),
    (257, 1, 12, "negative", 5),
    (258, 1, 12, "padded", 5),
    (128, 4, 3, "c", 5),
    (129, 4, 3, "f", 5),
    (255, 4, 3, "negative", 5),
    (256, 4, 12, "padded", 5),
    (257, 4, 2, "c", 5),
    (258, 4, 1, "c", 5),
]


@pytest.mark.parametrize("n,d,k,layout,iterations", BOUNDARIES)
def test_hmm_time_boundaries_workspace_budget_and_canaries(n, d, k, layout, iterations):
    values, owners = sample(n, d, k, layout)
    originals = [value.copy() for value in values]
    owner_originals = [owner.copy() for owner in owners]
    expected, scalar = fit(values, iterations, simd="scalar", audit=True)
    assert scalar["isa"] == "scalar"

    storage = np.full(expected.size + 2, -889.0)
    out = storage[1:-1].reshape(expected.shape)
    work = op.Workspace()
    scratch = 3 * n * k + 2 * k * k + 4 * k
    if n >= 128 and k >= 3:
        scratch += min(n - 1, 128)
    work.reserve(doubles=scratch + 19)
    capacity = work.capacity_bytes
    actual, audit = fit(values, iterations, out=out, workspace=work, audit=True)
    assert actual is out
    compare_models(actual, expected, d)
    assert storage[0] == storage[-1] == -889.0
    assert audit["workspace_bytes"] == scalar["workspace_bytes"] == scratch * 8
    assert work.capacity_bytes == capacity
    assert audit["workspace_capacity_bytes"] == capacity
    assert audit["input_copy_bytes"] == scalar["input_copy_bytes"] == 0
    assert audit["algorithm_copy_bytes"] == scalar["algorithm_copy_bytes"]

    retained = fit(values, iterations, workspace=work)
    saved = retained.copy()
    later = fit(values, 1, workspace=work)
    assert not np.shares_memory(actual, retained)
    assert not np.shares_memory(retained, later)
    bits(retained, saved)
    compare_models(actual, expected, d)
    assert storage[0] == storage[-1] == -889.0
    for value, original in zip(values, originals, strict=True):
        bits(value, original)
        assert not np.shares_memory(retained, value)
    for owner, original in zip(owners, owner_originals, strict=True):
        bits(owner, original)


@pytest.mark.parametrize("n,d,k", [(129, 1, 3), (258, 4, 12)])
def test_hmm_tiled_model_matches_independent_reference_and_inference(n, d, k):
    values, _ = sample(n, d, k)
    actual = fit(values, 3)
    expected = ref.hmm_em_reference(*values, 3, 0.0)
    projections = (op.hmm_initial, op.hmm_transition, op.hmm_means, op.hmm_variances)
    for project, target in zip(projections, expected[:-2], strict=True):
        np.testing.assert_allclose(project(actual), target, rtol=1e-10, atol=1e-10)
    assert op.em_iterations(actual) == expected[-2]
    assert not op.em_converged(actual)
    np.testing.assert_allclose(op.em_log_likelihood(actual), expected[-1], rtol=1e-12)

    scalar = fit(values, 3, simd="scalar")
    posteriors = []
    for model in (scalar, actual):
        emissions = op.gaussian_log_density(values[0], op.hmm_means(model), op.hmm_variances(model))
        posteriors.append(op.hmm_smooth(emissions, op.hmm_initial(model), op.hmm_transition(model)))
    np.testing.assert_allclose(posteriors[1], posteriors[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(op.argmax_axis(posteriors[1], 1), op.argmax_axis(posteriors[0], 1))


@pytest.mark.parametrize("n,k,step", [(129, 3, 2), (258, 12, 2), (129, 3, 5), (258, 12, 5)])
def test_hmm_time_tiles_keep_strict_stopping_at_tolerance_neighbors(n, k, step):
    values, _ = sample(n, 1, k)
    previous = fit(values, step - 1, simd="scalar")
    current = fit(values, step, simd="scalar")
    difference = abs(op.em_log_likelihood(current) - op.em_log_likelihood(previous))
    assert np.isfinite(difference) and difference > 0.0
    for tolerance in (np.nextafter(difference, 0.0), difference,
                      np.nextafter(difference, np.inf)):
        expected = fit(values, step + 1, tolerance, simd="scalar")
        actual = fit(values, step + 1, tolerance)
        compare_models(actual, expected, 1)
        if step == 2:
            assert op.em_iterations(actual) == (2 if tolerance > difference else 3)


@contextmanager
def rounding_mode(name):
    system, machine = platform.system(), platform.machine().lower()
    if system in {"Darwin", "Linux"} and machine in {"arm64", "aarch64"}:
        modes = dict(nearest=0, downward=0x00800000, upward=0x00400000, towardzero=0x00C00000)
    elif system in {"Darwin", "Linux"} and machine in {"x86_64", "amd64", "i386", "i686"}:
        modes = dict(nearest=0, downward=0x0400, upward=0x0800, towardzero=0x0C00)
    else:
        pytest.skip("C fenv constants are not established for this platform")
    libc = ctypes.CDLL(None)
    try:
        get_environment, set_environment = libc.fegetenv, libc.fesetenv
        get_rounding, set_rounding = libc.fegetround, libc.fesetround
    except AttributeError:
        pytest.skip("C fenv functions are unavailable")
    get_environment.argtypes = set_environment.argtypes = (ctypes.c_void_p,)
    get_rounding.argtypes = ()
    set_rounding.argtypes = (ctypes.c_int,)
    for function in (get_environment, set_environment, get_rounding, set_rounding):
        function.restype = ctypes.c_int
    # Overallocated, suitably aligned storage keeps the platform fenv opaque.
    environment = (ctypes.c_longdouble * 32)()
    assert get_environment(ctypes.byref(environment)) == 0
    try:
        assert set_rounding(modes[name]) == 0
        assert get_rounding() == modes[name]
        yield get_rounding, modes[name]
    finally:
        assert set_environment(ctypes.byref(environment)) == 0


@pytest.mark.parametrize("mode", ["nearest", "downward", "upward", "towardzero"])
@pytest.mark.parametrize("n,k", [(129, 3), (258, 12)])
def test_hmm_time_tiles_preserve_all_rounding_modes(mode, n, k):
    # Build inputs before changing fenv; compare the same native libm under it.
    values, _ = sample(n, 1, k)
    with rounding_mode(mode) as (get_rounding, expected_mode):
        expected = fit(values, 3, simd="scalar")
        assert get_rounding() == expected_mode
        actual = fit(values, 3)
        assert get_rounding() == expected_mode
    compare_models(actual, expected, 1)


@pytest.mark.parametrize("kind", ["zero_posterior", "zero_probabilities", "constant", "large_finite"])
def test_hmm_time_tiles_keep_floors_underflow_and_extreme_finite_models(kind):
    values = list(fixture(258, 1, 3))
    x, weights, transition, means, variances = values
    if kind == "zero_posterior":
        means[1:] = 1000.0
        variances[:] = 1.0
    elif kind == "zero_probabilities":
        weights[:] = [1.0, 0.0, 0.0]
        transition[:] = np.eye(3)
    elif kind == "constant":
        x[:] = means[:] = 2.0
        variances[:] = 0.0
    else:
        x *= 1e140
        means *= 1e140
        variances *= 1e280
    for value in values:
        value.flags.writeable = False
    expected = fit(values, 5, simd="scalar")
    actual = fit(values, 5)
    compare_models(actual, expected, 1)
    assert np.isfinite(actual).all()
    assert np.all(op.hmm_variances(actual) >= 1e-5)


@pytest.mark.parametrize("kind,code", [
    ("nan", "NON_FINITE_RESULT"),
    ("inf", "NON_FINITE_RESULT"),
    ("negative_inf", "NON_FINITE_RESULT"),
    ("negative_variance", "DOMAIN_ERROR"),
    ("zero_weights", "DOMAIN_ERROR"),
    ("zero_transition_row", "DOMAIN_ERROR"),
    ("empty", "INSUFFICIENT_SAMPLE"),
    ("transition_shape", "SHAPE_MISMATCH"),
    ("density_overflow", "NON_FINITE_RESULT"),
    ("forward_overflow", "NON_FINITE_RESULT"),
])
def test_hmm_time_tiles_preserve_failure_code_output_and_workspace_recovery(kind, code, capfd):
    values = list(fixture(258, 1, 3))
    if kind in {"nan", "inf", "negative_inf"}:
        values[0][128, 0] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[kind]
    elif kind == "negative_variance":
        values[4][1, 0] = -1.0
    elif kind == "zero_weights":
        values[1][:] = 0.0
    elif kind == "zero_transition_row":
        values[2][1] = 0.0
    elif kind == "empty":
        values[0] = values[0][:0]
    elif kind == "transition_shape":
        values[2] = np.ones((3, 4))
    elif kind == "density_overflow":
        values[0][128, 0] = np.finfo(float).max
    else:
        # Finite individual emissions overflow only in the cumulative forward path.
        values[0][:] = 2e153
        values[3][:] = 0.0
        values[4][:] = 1.0
    originals = [value.copy() for value in values]
    for value in values:
        value.flags.writeable = False
    healthy, _ = sample(129, 1, 3)
    target = fit(healthy, 2, simd="scalar")
    outputs, errors, stderrs = [], [], []
    capfd.readouterr()
    for simd in ("scalar", "auto"):
        work = op.Workspace()
        storage = np.full(3 * (3 + 2 + 4) + 2, 772.0)
        out = storage[1:-1].reshape(3, 9)
        with pytest.raises((ValueError, RuntimeError), match=code) as error:
            fit(values, 5, simd=simd, out=out, workspace=work)
        errors.append((type(error.value), str(error.value)))
        stderrs.append(capfd.readouterr().err)
        outputs.append(out.copy())
        assert storage[0] == storage[-1] == 772.0
        recovered = fit(healthy, 2, simd=simd, workspace=work)
        compare_models(recovered, target, 1)
        assert not np.shares_memory(recovered, out)
    bits(outputs[0], outputs[1])
    assert errors[0] == errors[1]
    assert stderrs[0] == stderrs[1]
    for value, original in zip(values, originals, strict=True):
        bits(value, original)


def test_hmm_workspace_growth_and_other_operator_reuse_do_not_mutate_models():
    small, _ = sample(128, 1, 3)
    large, _ = sample(258, 1, 12, "padded")
    work = op.Workspace()
    first = fit(small, 3, workspace=work)
    saved = first.copy()
    original_capacity = work.capacity_bytes
    large_model = fit(large, 3, workspace=work)
    assert work.capacity_bytes > original_capacity
    assert not np.shares_memory(first, large_model)
    bits(first, saved)
    # Reuse scratch through a different public operator, then fit again.
    op.median(np.arange(4096, dtype=np.float64), workspace=work)
    again = fit(small, 3, workspace=work)
    bits(again, saved)
    bits(first, saved)
    assert not np.shares_memory(first, again)
