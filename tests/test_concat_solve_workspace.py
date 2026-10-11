"""The private block coefficient is consumed by the canonical solve, not copied."""

import os
import platform

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_guarded_lu_graph import count
from .test_solve_library import residual, same_ieee

LIBRARY = int(
    os.environ.get(
        "CALMETRICS_TEST_EXPECT_ACCELERATE",
        str(
            int(
                platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and int(platform.mac_ver()[0].split(".")[0] or 0) >= 15
                and os.environ.get("CME_EXPECT_ACCELERATE") != "0"
            )
        ),
    )
)

COEFFICIENT = "concatenate(concatenate(h,transpose(g),1),concatenate(g,z,1),0)"
RHS = "concatenate(b,zr,0)"
SHAPES = {"h": ["N", "N"], "g": ["M", "N"], "z": ["M", "M"], "b": ["N", "K"], "zr": ["M", "K"]}
TYPES = {
    key: dict(kind="matrix", axes=["asset", "asset"], shape=shape) for key, shape in SHAPES.items()
}


def fixture(layout="c", n=1024, active=8):
    h = np.diag(np.linspace(0.5, 1.0, n) + 1e-10)
    g = np.zeros((active, n))
    g[0] = 1.0
    for row in range(1, active):
        g[row, n - row] = -1.0
    b = np.random.default_rng(948).normal(size=(n, 1))
    return {
        key: np.array(value, order=layout.upper())
        for key, value in dict(
            h=h, g=g, z=np.zeros((active, active)), b=b, zr=np.zeros((active, 1))
        ).items()
    }


def original(values):
    a = np.block([[values["h"], values["g"].T], [values["g"], values["z"]]])
    b = np.concatenate((values["b"], values["zr"]), axis=0)
    return a, b, op.solve_multi(a, b, simd="scalar")


def graph(order="C", *, public=False, **kwargs):
    roots = [f"solve_multi({COEFFICIENT},{RHS})"]
    if public:
        roots.append(COEFFICIENT)
    return GraphCompiler(TYPES).compile(roots, result_format="typed", output_order=order, **kwargs)


def run(g, values, scheduler=None):
    bounds = np.zeros(1, np.int64)
    if scheduler is not None:
        return scheduler.execute(g, values, bounds, bounds)
    with AdaptiveScheduler(cpu_budget=1) as owner:
        return owner.execute(g, values, bounds, bounds)


def compact_factor_copies(n):
    # Local backward envelopes require no QR samples or reconstructed factors.
    return 0


@pytest.mark.parametrize("layout", ["c", "f"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("active", [1, 8])
def test_kkt_block_graph_private_coefficient_and_bounded_workspace(layout, order, active):
    values = fixture(layout, active=active)
    saved = {k: v.copy() for k, v in values.items()}
    for value in values.values():
        value.flags.writeable = False
    coefficient, rhs, expected = original(values)
    result = run(graph(order), values)
    actual = result.outputs[0].values[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=4.9e-5)
    residual(coefficient, actual, rhs)
    assert count(result, "accelerate_lu_calls") == LIBRARY
    assert count(result, "accelerate_lu_attempts") == LIBRARY
    # Logical concat output is outside algorithm_copy_bytes. Only F-dominant
    # leaves eliminate the private transpose; C leaves keep the old path.
    transpose_bytes = coefficient.nbytes if layout == "c" else 0
    assert (
        count(result, "algorithm_copy_bytes")
        == LIBRARY * (transpose_bytes + compact_factor_copies(coefficient.shape[0]))
        + (1 + LIBRARY) * rhs.nbytes
    )
    if LIBRARY:
        assert 0 < count(result, "operator_workspace_capacity_bytes") < coefficient.nbytes
    else:
        assert count(result, "operator_workspace_capacity_bytes") == 0
        same_ieee(actual, expected)
    assert count(result, "input_copy_bytes") == 0
    assert not actual.flags.writeable
    assert actual.flags.c_contiguous if order == "C" else actual.flags.f_contiguous
    for key, value in values.items():
        same_ieee(value, saved[key])
        assert not np.shares_memory(actual, value)


@pytest.mark.parametrize("order", ["C", "F"])
def test_public_coefficient_and_two_generations_remain_owned(order):
    values = fixture("f")
    coefficient, rhs, expected = original(values)
    g = graph(order, public=True)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = run(g, values, scheduler)
        second = run(g, values, scheduler)
    for result in [first, second]:
        same_ieee(result.outputs[0].values[0], expected)
        same_ieee(result.outputs[1].values[0], coefficient)
        assert count(result, "algorithm_copy_bytes") == coefficient.nbytes + rhs.nbytes
    for a in first.outputs:
        for b in second.outputs:
            assert not np.shares_memory(a.values[0], b.values[0])


@pytest.mark.parametrize("fault", ["first_rhs_nan", "coefficient_inf", "singular"])
def test_original_error_order_and_input_ownership(fault):
    values = fixture("f")
    if fault == "first_rhs_nan":
        values["b"][-1, 0] = np.nan
        values["g"][:] = 0.0
    elif fault == "coefficient_inf":
        values["h"][-1, -1] = np.inf
    else:
        values["g"][:] = 0.0
    saved = {k: v.copy() for k, v in values.items()}
    with pytest.raises(ValueError) as reference:
        original(values)
    with pytest.raises(ValueError, match=str(reference.value).split(":")[0]):
        run(graph("F"), values)
    for key, value in values.items():
        same_ieee(value, saved[key])


def test_prepared_mutation_recovery_and_old_results():
    values = fixture("f")
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph("F"), values, bounds, bounds)
        first = prepared.run_snapshot()
        saved = first.outputs[0].values[0].copy()
        values["b"] *= 2.0
        values["h"][np.diag_indices(1024)] += 0.5
        second = prepared.run_audit()
        coefficient, rhs, expected = original(values)
        np.testing.assert_allclose(second.outputs[0].values[0], expected, rtol=1e-10, atol=4.9e-5)
        residual(coefficient, second.outputs[0].values[0], rhs)
        old = values["g"].copy()
        values["g"][:] = 0.0
        with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
            prepared.run_audit()
        values["g"][:] = old
        recovered = prepared.run_snapshot().outputs[0].values[0]
        np.testing.assert_allclose(recovered, expected, rtol=1e-10, atol=4.9e-5)
        residual(coefficient, recovered, rhs)
    same_ieee(first.outputs[0].values[0], saved)


@pytest.mark.parametrize("order", ["C", "F"])
def test_negative_rhs_preserves_readonly_leaf(order):
    values = fixture("f")
    values["b"] = values["b"][::-1]
    saved = values["b"].copy()
    values["b"].flags.writeable = False
    coefficient, rhs, expected = original(values)
    result = run(graph(order), values)
    np.testing.assert_allclose(result.outputs[0].values[0], expected, rtol=1e-10, atol=4.9e-5)
    residual(coefficient, result.outputs[0].values[0], rhs)
    same_ieee(values["b"], saved)
    # Logical concat output is outside algorithm_copy_bytes. Direct F
    # construction eliminates the formerly counted private transpose.
    assert (
        count(result, "algorithm_copy_bytes")
        == LIBRARY * compact_factor_copies(coefficient.shape[0]) + (1 + LIBRARY) * rhs.nbytes
    )


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("case", ["later_consumer", "isolate"])
def test_private_reuse_exclusions_preserve_consumers(order, case):
    values = fixture("f")
    coefficient, rhs, expected = original(values)
    if case == "later_consumer":
        g = GraphCompiler(TYPES).compile(
            [f"solve_multi({COEFFICIENT},{RHS})", f"sum_axis({COEFFICIENT},0)"],
            result_format="typed",
            output_order=order,
        )
    else:
        g = graph(order, error_policy="isolate")
    result = run(g, values)
    same_ieee(result.outputs[0].values[0], expected)
    assert count(result, "algorithm_copy_bytes") == coefficient.nbytes + rhs.nbytes
    if case == "later_consumer":
        same_ieee(result.outputs[1].values[0], op.sum_axis(coefficient, 0))
