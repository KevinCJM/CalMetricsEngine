"""Shared ordered LU library with distinct solve/solve_multi contracts.

The reference below transcribes the pre-optimization per-k/per-row arithmetic
using separate NumPy multiply/subtract calls. It is test-only, not a fallback.
simd='scalar' is checked as a request contract, never used as the oracle: old
solve_multi had no separate Isa branch. Existing matvec dispatch reports the
shared Eigen/platform capability without an external test environment override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig
from calmetrics_engine import operators as op

MIN_N = 128
LIBRARY_ISA = "eigen-array-blocked-lu"
SENTINEL = -913.25


@dataclass
class Reference:
    output: np.ndarray
    error: str | None
    pivots: list[tuple[int, int]]


def canonical(name, a, b):
    """Preserve old pivot comparisons, error order and explicit-out prefix."""
    n = a.shape[0]
    out = np.full(b.shape, SENTINEL, dtype=np.float64)
    pivots = []
    multi = name == "solve_multi"
    with np.errstate(all="ignore"):
        if multi:
            out[...] = b  # Original copy_finite writes the entire RHS first.
            if not np.isfinite(out).all():
                return Reference(out, "NON_FINITE_RESULT", pivots)
            m = np.array(a, dtype=np.float64, order="C", copy=True)
            if not np.isfinite(m).all():
                return Reference(out, "NON_FINITE_RESULT", pivots)
            rhs = out
        else:
            m = np.array(a, dtype=np.float64, order="C", copy=True)
            rhs = np.array(b, dtype=np.float64, copy=True)
        for k in range(n):
            pivot = k
            for row in range(k + 1, n):
                if abs(m[row, k]) > abs(m[pivot, k]):
                    pivot = row
            magnitude = abs(m[pivot, k])
            singular = (not magnitude > 1e-14) if multi else magnitude <= 1e-14
            if singular:
                return Reference(out, "SINGULAR_MATRIX", pivots)
            pivots.append((k, pivot))
            if pivot != k:
                begin = k if multi else 0
                m[[k, pivot], begin:] = m[[pivot, k], begin:]
                rhs[[k, pivot]] = rhs[[pivot, k]]
            for row in range(k + 1, n):
                factor = m[row, k] / m[k, k]
                if not multi:
                    m[row, k] = 0.0
                # Do not skip factor==0; retain 0*Inf/NaN propagation. Each
                # coefficient sees k ascending and a rounded product first.
                m[row, k + 1:] -= factor * m[k, k + 1:]
                rhs[row] -= factor * rhs[k]
        for row in range(n - 1, -1, -1):
            diagonal = m[row, row]
            if multi:
                if not np.isfinite(diagonal):
                    return Reference(out, "NON_FINITE_RESULT", pivots)
                if abs(diagonal) <= 1e-14:
                    return Reference(out, "SINGULAR_MATRIX", pivots)
                for k in range(row + 1, n):
                    rhs[row] -= m[row, k] * rhs[k]
                # Original loop divides and finite-checks each RHS column in
                # order; later columns retain their predivision values on error.
                for column in range(rhs.shape[1]):
                    rhs[row, column] /= diagonal
                    if not np.isfinite(rhs[row, column]):
                        return Reference(out, "NON_FINITE_RESULT", pivots)
            else:
                total = rhs[row]
                for k in range(row + 1, n):
                    total -= m[row, k] * out[k]
                if abs(diagonal) <= 1e-14:
                    return Reference(out, "SINGULAR_MATRIX", pivots)
                out[row] = total / diagonal
    return Reference(out, None, pivots)


def same_ieee(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    numeric = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[numeric].view(np.uint64), expected[numeric].view(np.uint64))


def residual(a, x, b):
    reconstructed = np.einsum("ij,j...->i...", a, x, optimize=False)
    np.testing.assert_allclose(reconstructed, b, rtol=1e-11, atol=1e-11)


def native_against_reference(name, a, b, *, simd="auto", workspace=None):
    reference = canonical(name, a, b)
    out = np.full(b.shape, SENTINEL, dtype=np.float64)
    if reference.error:
        with pytest.raises(ValueError, match=reference.error):
            op.call(name, a, b, out=out, simd=simd, workspace=workspace)
        audit = None
    else:
        returned, audit = op.call(name, a, b, out=out, simd=simd,
                                  workspace=workspace, audit=True)
        assert np.shares_memory(out, returned) or out.size == 0
    same_ieee(out, reference.output)
    return out, audit, reference


def sample(name, n, columns=3, layout="c"):
    rng = np.random.default_rng(472 + n)
    a = rng.normal(size=(n, n))
    a[np.diag_indices(n)] += np.abs(a).sum(axis=1) + 1.0
    b = rng.normal(size=(n, columns) if name == "solve_multi" else n)
    if layout == "f":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif layout == "negative":
        a, b = a[::-1, ::-1], b[::-1]
    elif layout == "sliced":
        storage = np.zeros((n * 2, n * 2))
        storage[::2, ::2] = a
        a = storage[::2, ::2]
        storage = np.zeros(tuple(2 * size for size in b.shape))
        selection = (slice(None, None, 2),) * b.ndim
        storage[selection] = b
        b = storage[selection]
    elif layout == "zero_rhs_stride":
        if b.ndim == 1:
            b = np.broadcast_to(b[:1], b.shape)
        else:
            b = np.broadcast_to(b[:1, :1], b.shape)
    return a, b


@pytest.fixture(scope="module")
def eigen_backend_available():
    # Matvec uses the same build flag and macOS/arm64 availability guard. Probe
    # that existing independent kernel instead of inferring capability from
    # solve_multi's result or requiring manual ON/OFF environment settings.
    a = np.ones((128, 8))
    b = np.ones(8)
    _, audit = op.matvec(a, b, audit=True)
    assert audit["eigen_matvec_calls"] in (0, 1)
    return bool(audit["eigen_matvec_calls"])


def assert_dispatch(name, audit, n, simd, eigen_backend_available):
    if simd == "scalar" or n < MIN_N or not eigen_backend_available:
        assert audit["isa"] == "scalar"
    else:
        assert audit["isa"] == LIBRARY_ISA


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
@pytest.mark.parametrize("n", [128, 129, 257])
@pytest.mark.parametrize("layout", ["c", "f", "sliced", "negative", "zero_rhs_stride"])
def test_layout_tail_readonly_workspace_and_exact_result(name, n, layout):
    a, b = sample(name, n, layout=layout)
    before_a, before_b = a.copy(), b.copy()
    a.flags.writeable = b.flags.writeable = False
    workspace = op.Workspace()
    actual, audit, _ = native_against_reference(name, a, b, workspace=workspace)
    residual(a, actual, b)
    requirements = op.get(name).requirements(a, b)
    expected_scratch = n * n + (n if name == "solve" else 0)
    assert requirements["scratch_doubles"] == expected_scratch
    assert audit["workspace_bytes"] == expected_scratch * 8
    assert audit["algorithm_copy_bytes"] == (a.size + b.size) * 8
    assert audit["input_copy_bytes"] == 0
    capacity = audit["workspace_capacity_bytes"]
    independent, again = op.call(name, a, b, workspace=workspace, audit=True)
    same_ieee(independent, actual)
    assert not np.shares_memory(independent, actual)
    assert not np.shares_memory(independent, a) and not np.shares_memory(independent, b)
    assert again["workspace_capacity_bytes"] == capacity
    same_ieee(a, before_a)
    same_ieee(b, before_b)


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
@pytest.mark.parametrize("n", sorted({max(1, MIN_N - 1), MIN_N, MIN_N + 1}))
def test_admission_and_explicit_scalar_request(name, n, eigen_backend_available):
    a, b = sample(name, n, columns=8)
    workspace = op.Workspace()
    automatic, audit, reference = native_against_reference(name, a, b, workspace=workspace)
    scalar, scalar_audit = op.call(name, a, b, simd="scalar", audit=True, workspace=workspace)
    same_ieee(scalar, reference.output)
    same_ieee(automatic, scalar)
    assert_dispatch(name, audit, n, "auto", eigen_backend_available)
    assert_dispatch(name, scalar_audit, n, "scalar", eigen_backend_available)
    assert audit["workspace_capacity_bytes"] == scalar_audit["workspace_capacity_bytes"]


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
@pytest.mark.parametrize("pattern", ["second_panel_multiplier_swap", "multiple_swaps", "tie"])
def test_pivots_preserve_prior_multipliers_and_first_equal_candidate(name, pattern):
    n = max(MIN_N, 129)
    a, b = sample(name, n, columns=8)
    if pattern == "second_panel_multiplier_swap":
        a = np.eye(n) * 2
        ix = np.array([32, 33, 65])
        a[np.ix_(ix, ix)] = [[4, 1, 2], [1, 1, 7], [2, 5, 3]]
        # With panel32, pivot 33->65 must swap the multiplier from column32.
        b = np.eye(n)[:, ix] if name == "solve_multi" else np.eye(n)[:, 32]
    elif pattern == "multiple_swaps":
        ix = np.array([0, 32, 65, n - 1])
        a[ix] = a[np.roll(ix, 1)]
    else:
        a[:, 0] *= 0.001
        a[0, 0], a[1, 0], a[65, 0] = 1.0, -1.0, 1.0
    actual, _, reference = native_against_reference(name, a, b)
    if pattern == "second_panel_multiplier_swap":
        assert (33, 65) in reference.pivots
    elif pattern == "multiple_swaps":
        assert sum(k != row for k, row in reference.pivots) >= 3
        assert any(k // 32 != row // 32 for k, row in reference.pivots)
    else:
        assert reference.pivots[0] == (0, 0)
    residual(a, actual, b)


def test_minimal_three_by_three_second_panel_pivot_reference():
    # For the standalone temporary probe with panel=2, this is the minimal
    # multiplier-swap counterexample. The installed public API has no panel arg.
    a = np.array([[4.0, 1, 2], [1, 1, 7], [2, 5, 3]])
    b = np.eye(3)
    actual, _, reference = native_against_reference("solve_multi", a, b)
    assert reference.pivots == [(0, 0), (1, 2), (2, 2)]
    residual(a, actual, b)
    np.testing.assert_allclose(actual, np.linalg.solve(a, b), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
def test_separate_multiply_subtract_not_fma(name):
    n = max(MIN_N, 129)
    a = np.eye(n) * 2
    a[32, 32] = 1
    a[33, 32] = 1 - 2.0**-27
    a[32, 65] = 1 + 2.0**-27
    a[33, 65] = 1
    b = np.zeros((n, 1) if name == "solve_multi" else n)
    b[33], b[65] = 1, 2.0**55
    actual, _, _ = native_against_reference(name, a, b)
    assert np.asarray(actual[33]).item() == 0.5


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
@pytest.mark.parametrize("direction", [-np.inf, None, np.inf])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_pivot_threshold_adjacent_representable_values(name, direction, sign):
    n = max(MIN_N, 129)
    a = np.eye(n) * 2
    pivot = 1e-14 if direction is None else np.nextafter(1e-14, direction)
    a[-1, -1] = sign * pivot
    b = np.ones((n, 3) if name == "solve_multi" else n)
    actual, _, reference = native_against_reference(name, a, b)
    assert reference.error == (None if direction == np.inf else "SINGULAR_MATRIX")
    if not reference.error:
        residual(a, actual, b)


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
@pytest.mark.parametrize("fault", [
    "rhs_nan", "matrix_nan", "matrix_inf", "rhs_nan_and_singular", "elimination_overflow",
    "backsolve_overflow", "zero_times_intermediate_inf",
])
def test_nonfinite_and_intermediate_overflow_keep_error_and_written_prefix(name, fault):
    n = max(MIN_N, 129)
    a = np.eye(n) * 2
    b = np.ones((n, 3) if name == "solve_multi" else n)
    maximum = np.finfo(np.float64).max
    if fault.startswith("rhs_nan"):
        b[-1] = np.nan
        if fault == "rhs_nan_and_singular":
            a[0, 0] = 0
    elif fault == "matrix_nan":
        a[-1, -1] = np.nan
    elif fault == "matrix_inf":
        a[-1, -1] = np.inf
    elif fault == "elimination_overflow":
        a[-2:, -2:] = [[maximum, maximum], [-maximum, maximum]]
    elif fault == "backsolve_overflow":
        a[-2, -1] = maximum
        b[-1] = 8
        if name == "solve_multi":
            # First RHS column remains finite; failure on the second must not
            # divide later columns or back-substitute earlier rows first.
            b[-1, 0] = 0
            b[-1, 2] = 0
    else:
        a[0, 0], a[1, 0] = 1, -1
        a[0, -1] = a[1, -1] = maximum
    before_a, before_b = a.copy(), b.copy()
    actual, _, reference = native_against_reference(name, a, b)
    if fault == "backsolve_overflow" and name == "solve_multi":
        assert reference.error == "NON_FINITE_RESULT"
        assert actual[-2, 0] == 0.5 and np.isneginf(actual[-2, 1])
        assert actual[-2, 2] == 1.0  # Unvisited column must remain undivided.
    same_ieee(a, before_a)
    same_ieee(b, before_b)


@pytest.mark.parametrize("n", [0, 3, MIN_N])
@pytest.mark.parametrize("fault", ["none", "singular", "nonfinite"])
def test_empty_rhs_does_not_skip_coefficient_contract(n, fault):
    a = np.eye(n) * 2
    if n and fault != "none":
        a[-1, -1] = 0.0 if fault == "singular" else np.nan
    b = np.empty((n, 0))
    actual, _, reference = native_against_reference("solve_multi", a, b)
    assert actual.shape == (n, 0)
    if n and fault != "none":
        assert reference.error == ("SINGULAR_MATRIX" if fault == "singular" else "NON_FINITE_RESULT")


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
def test_zero_coefficient_strides_preserve_singularity(name):
    n = 129
    a = np.broadcast_to(np.ones((1, 1)), (n, n))
    b = np.ones((n, 3) if name == "solve_multi" else n)
    _, _, reference = native_against_reference(name, a, b)
    assert reference.error == "SINGULAR_MATRIX"


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
def test_readonly_coefficient_and_rhs_may_share_storage(name):
    a, _ = sample(name, max(MIN_N, 129))
    b = a[:, :3] if name == "solve_multi" else a[:, 0]
    before = a.copy()
    assert np.shares_memory(a, b)
    a.flags.writeable = b.flags.writeable = False
    actual, _, _ = native_against_reference(name, a, b)
    residual(a, actual, b)
    same_ieee(a, before)


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
@pytest.mark.parametrize("alias", ["matrix", "rhs"])
def test_output_alias_rejected_before_input_mutation(name, alias):
    a, b = sample(name, 129)
    out = b if alias == "rhs" else a.reshape(-1)[:b.size].reshape(b.shape)
    before_a, before_b = a.copy(), b.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIASES_INPUT"):
        op.call(name, a, b, out=out)
    same_ieee(a, before_a)
    same_ieee(b, before_b)


def declarations(name):
    return {
        "a": dict(kind="matrix", axes=["asset", "asset"], shape=["N", "N"]),
        "b": (dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K"])
              if name == "solve_multi" else dict(kind="vector", axes=["asset"], shape=["N"])),
    }


def config(lane):
    return PlannerConfig(
        thread_work_units=1e100 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1, max_processes=2,
    )


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
def test_prepared_refresh_preserves_retained_independent_snapshots(name):
    a, b = sample(name, max(MIN_N, 129), columns=8)
    expression = f"{name}(a,b)"
    graph = GraphCompiler(declarations(name)).compile([expression, expression + ">0", "b"])
    bounds = np.zeros(1, np.int64)
    first_expected = canonical(name, a, b).output
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, dict(a=a, b=b), bounds, bounds)
        first = prepared.run_snapshot()
        saved = [output.values[0].copy() for output in first.outputs]
        same_ieee(saved[0], first_expected)
        a[np.diag_indices(a.shape[0])] += 1.0
        b *= 2.0
        second_expected = canonical(name, a, b).output
        second = prepared.run_snapshot()
        same_ieee(second.outputs[0].values[0], second_expected)
        np.testing.assert_array_equal(second.outputs[1].values[0], second_expected > 0)
        same_ieee(second.outputs[2].values[0], b)
    for original, current, snapshot in zip(first.outputs, second.outputs, saved, strict=True):
        np.testing.assert_array_equal(original.values[0], snapshot)
        assert not np.shares_memory(original.values[0], current.values[0])
        assert not np.shares_memory(current.values[0], a)
        assert not np.shares_memory(current.values[0], b)


@pytest.mark.parametrize("name", ["solve", "solve_multi"])
@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_multiroot_cse_error_isolation_and_independent_worker_results(name, lane):
    a, b = sample(name, max(MIN_N, 129), columns=8)
    types = declarations(name)
    types.update({
        "bad": dict(kind="matrix", axes=["asset", "asset"], shape=["D", "D"]),
        "bad_b": (dict(kind="matrix", axes=["asset", "factor"], shape=["D", "J"])
                  if name == "solve_multi" else dict(kind="vector", axes=["asset"], shape=["D"])),
    })
    expression, broken = f"{name}(a,b)", f"{name}(bad,bad_b)"
    graph = GraphCompiler(types).compile(
        [expression, expression + ">0", broken, broken + ">0", "b"], error_policy="isolate"
    )
    assert graph.metadata()["cse_eliminated_nodes"] > 0
    expected = canonical(name, a, b).output
    # Fail after several completed panels, so worker propagation also covers
    # the optimized elimination path rather than only the small scalar path.
    bad = np.eye(129)
    bad[-1, -1] = 0.0
    data = dict(a=a, b=b, bad=bad,
                bad_b=np.ones((129, 3) if name == "solve_multi" else 129))
    bounds = np.zeros(2, np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        result = scheduler.execute(graph, data, bounds, bounds)
    assert result.plan.lane == ("process" if lane.startswith("process") else lane)
    for row in range(2):
        same_ieee(result.outputs[0].values[row], expected)
        np.testing.assert_array_equal(result.outputs[1].values[row], expected > 0)
        assert result.outputs[0].root_statuses[row] == result.outputs[1].root_statuses[row] == 0
        assert result.outputs[2].root_statuses[row] != 0
        assert result.outputs[2].root_statuses[row] == result.outputs[3].root_statuses[row]
        assert result.outputs[4].root_statuses[row] == 0
        same_ieee(result.outputs[4].values[row], b)
        for index in (0, 1, 4):
            output = result.outputs[index].values[row]
            assert not output.flags.writeable
            assert not np.shares_memory(output, a) and not np.shares_memory(output, b)
    if lane == "process_shared":
        assert result.audit["output_copy_bytes"] == 0
