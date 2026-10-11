"""Contract, packing, ownership and transport checks for native library rotations."""

import ctypes
import os
import platform
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matrix_array_foundation import matrix, run
from .test_typed_results import config

PACKING_PLATFORM = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and int(platform.mac_ver()[0].split(".")[0] or 0) >= 15
)


def sample(n):
    # A dense, deterministic block exercises several sweeps while the larger
    # diagonal keeps tests quick and gives repeated-eigenvalue coverage.
    a = np.eye(n)
    k = min(n, 7)
    rng = np.random.default_rng(8841)
    x = rng.normal(size=(k, k))
    a[:k, :k] = x + x.T
    return a


@pytest.mark.parametrize("n", [0, 1, 17, 255, 256, 257, 384, 512])
@pytest.mark.parametrize("layout", ["c", "f", "reverse"])
def test_packed_jacobi_contract_and_workspace(n, layout):
    a = sample(n)
    if layout == "f":
        a = np.asfortranarray(a)
    elif layout == "reverse":
        a = a[::-1, ::-1]
    before = a.copy()
    a.flags.writeable = False
    work = op.Workspace()
    pair, audit = op.call("symmetric_eigh", a, workspace=work, audit=True)
    packed = PACKING_PLATFORM and n >= 256 and n % 128 == 0
    required = (n * (n + 1) // 2 if packed else n * n) * 8
    indices = n * np.dtype(np.uintp).itemsize if n >= 128 else 0
    assert audit["workspace_bytes"] == required + indices
    assert audit["workspace_capacity_bytes"] == required + indices
    assert audit["algorithm_copy_bytes"] == required
    assert audit["input_copy_bytes"] == 0
    if not packed:
        assert audit["eigen_rotation_calls"] == 0
    elif os.environ.get("CME_EXPECT_EIGEN") is not None:
        assert bool(audit["eigen_rotation_calls"]) == bool(
            int(os.environ["CME_EXPECT_EIGEN"])
        )
    w, v = op.eigenvalues(pair), op.eigenvectors(pair)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(a), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose((v * w) @ v.T, a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(v.T @ v, np.eye(n), rtol=1e-12, atol=1e-12)
    assert np.all(w[1:] >= w[:-1])
    if n:
        assert np.all(v[np.argmax(abs(v), axis=0), np.arange(n)] >= 0)
    assert not np.shares_memory(pair, a)
    snapshot = pair.copy()
    op.symmetric_eigh(a * 2, workspace=work)
    np.testing.assert_array_equal(pair, snapshot)
    np.testing.assert_array_equal(a, before)


@pytest.mark.parametrize("lane", ["single", "thread", "process", "process_shared"])
def test_library_usage_survives_cse_and_worker_transport(lane):
    a = sample(256)
    direct, audit = op.call("symmetric_eigh", a, audit=True)
    graph = GraphCompiler({"a": matrix()}).compile(
        ["eigenvalues(symmetric_eigh(a))", "eigenvectors(symmetric_eigh(a))"]
    )
    result = run(graph, {"a": a}, lane)
    calls = sum(chunk["eigen_rotation_calls"] for chunk in result.audit["native_chunks"])
    assert calls == 2 * audit["eigen_rotation_calls"]  # two rows, one CSE solve per row
    for row in range(2):
        np.testing.assert_allclose(
            result.outputs[0].values[row], op.eigenvalues(direct), atol=1e-12
        )
        np.testing.assert_allclose(
            result.outputs[1].values[row], op.eigenvectors(direct), atol=1e-12
        )
    before = result.outputs[1].values[0].copy()
    run(graph, {"a": a * 2}, lane)
    np.testing.assert_array_equal(result.outputs[1].values[0], before)


@pytest.mark.parametrize("fault", ["nan", "inf", "asymmetric", "nonconvergence"])
def test_packed_errors_keep_independent_roots_and_thread_setting(fault):
    a = sample(256)
    if fault == "nan":
        a[-1, -1] = np.nan
    if fault == "inf":
        a[-1, -1] = np.inf
    if fault == "asymmetric":
        a[-1, 0] = 0.2
    iterations = 1 if fault == "nonconvergence" else 32
    error = (
        "NON_CONVERGENCE"
        if iterations == 1
        else ("DOMAIN_ERROR" if fault == "asymmetric" else "NON_FINITE_RESULT")
    )
    with pytest.raises(ValueError, match=error):
        op.symmetric_eigh(a, 1e-12, iterations)
    graph = GraphCompiler({"a": matrix()}).compile(
        [f"eigenvalues(symmetric_eigh(a,1e-12,{iterations}))", "7"], error_policy="isolate"
    )
    result = run(graph, {"a": a})
    assert result.outputs[0].root_statuses[0] != 0
    assert result.outputs[1].values[0] == 7


def test_prepare_snapshots_reread_values_and_budget():
    a = sample(256)
    graph = GraphCompiler({"a": matrix()}).compile("eigenvalues(symmetric_eigh(a))")
    bounds = np.array([0], np.int64)
    with AdaptiveScheduler(cpu_budget=1, config=config("single")) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"a": a}, bounds, bounds)
        first = prepared.run_snapshot()
        before = first.outputs[0].values[0].copy()
        a *= 2
        second = prepared.run_snapshot()
        np.testing.assert_allclose(second.outputs[0].values[0], before * 2, atol=1e-12)
        np.testing.assert_array_equal(first.outputs[0].values[0], before)
        plan = scheduler.plan(graph, {"a": a}, bounds, bounds)
        with pytest.raises(MemoryError):
            scheduler.plan(
                graph,
                {"a": a},
                bounds,
                bounds,
                memory_budget_bytes=plan.estimated_total_memory_bytes - 1,
            )


@pytest.mark.skipif(
    platform.system() != "Darwin" or int(platform.mac_ver()[0].split(".")[0] or 0) < 15,
    reason="Apple thread-local BLAS API requires macOS 15",
)
def test_caller_blas_thread_policy_unchanged_on_success_and_exception():
    lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    lib.BLASGetThreading.restype = ctypes.c_uint
    lib.BLASSetThreading.argtypes = [ctypes.c_uint]

    def call():
        before = lib.BLASGetThreading()
        for policy in (0, 1):
            assert lib.BLASSetThreading(policy) == 0
            try:
                a = sample(256)
                _, audit = op.call("symmetric_eigh", a, audit=True)
                assert lib.BLASGetThreading() == policy
                with pytest.raises(ValueError, match="NON_CONVERGENCE"):
                    op.symmetric_eigh(a, 1e-12, 1)
                assert lib.BLASGetThreading() == policy
            finally:
                lib.BLASSetThreading(before)
        return audit["eigen_rotation_calls"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        counts = list(pool.map(lambda _: call(), range(3)))
    assert counts[0] == counts[1] == counts[2]


@pytest.mark.parametrize("cross", [np.nextafter(1e-12, 0), 1e-12, np.nextafter(1e-12, np.inf)])
def test_exact_stopping_boundary(cross):
    a = np.eye(256)
    a[0, 1] = a[1, 0] = cross
    result, audit = op.call("symmetric_eigh", a, 1e-12, 1, audit=True)
    if cross <= 1e-12:
        np.testing.assert_array_equal(op.eigenvalues(result), np.ones(256))
        np.testing.assert_array_equal(op.eigenvectors(result), np.eye(256))
        assert audit["eigen_rotation_calls"] == 0
    else:
        assert op.eigenvalues(result)[0] < 1
        assert op.eigenvalues(result)[-1] > 1


def test_nested_iteration_keeps_library_audit_and_shared_input():
    a = sample(256)
    projected = run(GraphCompiler({"a": matrix()}).compile("eigenvalues(symmetric_eigh(a))"), {"a": a})
    reference_calls = sum(c["eigen_rotation_calls"] for c in projected.audit["native_chunks"])
    g = GraphCompiler(
        {"a": matrix(), "seed": {"kind": "vector", "axes": ["component"], "shape": ["N"]}}
    ).compile("iterate(eigenvalues(symmetric_eigh(a))+iterate_x*0,seed,1e-12,2)")
    result = run(g, {"a": a, "seed": np.zeros(256)})
    np.testing.assert_allclose(result.outputs[0].values[0], np.linalg.eigvalsh(a), atol=1e-10)
    calls = sum(c["eigen_rotation_calls"] for c in result.audit["native_chunks"])
    assert calls == 2 * reference_calls  # two steps in each of the same two rows
