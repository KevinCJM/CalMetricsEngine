"""Large model reservation and free-history loan without changing math/ownership."""

import numpy as np

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from . import latent_chain as chain
from .test_hmm_xi_tiles import sample
from .test_latent_workspace_loan import exact, selected


def test_large_direct_graph_without_history_loan_fits_two_matrix_budget():
    # Just beyond the loan's NK threshold; this graph has no future result or
    # matrix-sized arena. A hidden third allocation would violate the estimate.
    n, d, k = 174763, 1, 12
    values, owners = sample(n, d, k)
    before = [a.tobytes() for a in values + tuple(owners)]
    expected, audit = op.gaussian_hmm_em(*values, 2, 0.0, audit=True)
    beta_copies = 2 * n * k * 8
    initial_copies = (k * (1 + 2 * d) + k * k) * 8
    assert audit["algorithm_copy_bytes"] == beta_copies + initial_copies
    required = (2 * n * k + 2 * k * k + 4 * k + 512 + 128 * k) * 8
    assert audit["workspace_bytes"] == required
    types = {
        "x": dict(kind="matrix", axes=["scenario", "factor"], shape=["N", "D"]),
        "w": dict(kind="vector", axes=["component"], shape=["K"]),
        "a": dict(kind="matrix", axes=["component", "component"], shape=["K", "K"]),
        "m": dict(kind="matrix", axes=["component", "factor"], shape=["K", "D"]),
        "v": dict(kind="matrix", axes=["component", "factor"], shape=["K", "D"]),
    }
    fields = [
        "hmm_initial",
        "hmm_transition",
        "hmm_means",
        "hmm_variances",
        "em_iterations",
        "em_log_likelihood",
        "em_converged",
    ]
    graph = GraphCompiler(types).compile(
        [f"{field}(gaussian_hmm_em(x,w,a,m,v,2,0))" for field in fields], result_format="typed"
    )
    inputs = dict(zip(types, values, strict=True))
    bounds = np.zeros(1, np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, inputs, bounds, bounds)
        saved = first.outputs[0].values[0].tobytes()
        second = scheduler.execute(graph, inputs, bounds, bounds)
        for field, a, b in zip(fields, first.outputs, second.outputs, strict=True):
            target = np.asarray(getattr(op, field)(expected))
            assert a.values[0].shape == b.values[0].shape == target.shape
            assert a.values[0].dtype == b.values[0].dtype == target.dtype
            # Projection views may be strided; compare logical bytes without
            # requiring a C layout or altering the public result contract.
            assert a.values[0].tobytes() == b.values[0].tobytes() == target.tobytes()
        assert first.outputs[0].values[0].tobytes() == saved
        assert not np.shares_memory(first.outputs[0].values[0], second.outputs[0].values[0])
        assert not first.outputs[0].values[0].flags.writeable
        for result in (first, second):
            report = result.audit["native_chunks"][0]
            assert report["operator_workspace_capacity_bytes"] == required
            assert report["algorithm_copy_bytes"] >= beta_copies + initial_copies
            assert (
                report["algorithm_copy_bytes"] < beta_copies + initial_copies + 4 * expected.nbytes
            )
    assert [a.tobytes() for a in values + tuple(owners)] == before


def test_complete_chain_uses_free_history_instead_of_copying_beta_again():
    data = chain.inputs(240000, 4, 12)
    graph, names = chain.graph_for(data, 12, 3, 17, "hmm")
    starts = np.zeros(1, np.int64)
    ends = np.full(1, 240000, np.int64)
    beta_copies = 3 * len(data["training_positions"]) * 12 * 8
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        inputs = selected(data, graph)
        canonical = scheduler.prepare_execution(graph, inputs, starts, ends).run_snapshot()
        first = scheduler.execute(graph, inputs, starts, ends)
        second = scheduler.execute(graph, inputs, starts, ends)
        exact(first, canonical)
        exact(second, canonical)
        exact(first, second)
        baseline = canonical.audit["native_chunks"][0]
        report = first.audit["native_chunks"][0]
        assert baseline["algorithm_copy_bytes"] >= beta_copies
        # Existing live saves/restores are still counted. Adding another beta
        # history copy on top of them would exceed this bound at three E steps.
        assert report["algorithm_copy_bytes"] < beta_copies
        assert (
            report["operator_workspace_capacity_bytes"]
            < baseline["operator_workspace_capacity_bytes"]
        )
        for a, b in zip(first.outputs, second.outputs, strict=True):
            assert not a.values[0].flags.writeable and not np.shares_memory(
                a.values[0], b.values[0]
            )
