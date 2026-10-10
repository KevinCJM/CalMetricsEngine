"""Large short-horizon state batches through ordinary native execution.

Compare every selected root bitwise against the unchanged public canonical DAG.
The smaller independent-oracle fixtures and fair warmed NJIT whole-chain
performance comparison are maintained separately; these are contract tests.
"""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, SharedInputBundle

from .test_compact_path_execution import (
    assert_canonical,
    assert_private,
    check_ownership,
    chunk_sum,
    compare_selected,
    execute,
    saved_outputs,
)
from .test_mc_compact_graph import compile_compact, ieee_bits
from .test_raw_factor_prefix import (
    public_factor_spec,
    raw_factor_spec,
    raw_parameters,
    small_raw_tape,
)
from .test_typed_results import config


def large_short_fixture(student, order):
    spec = raw_factor_spec(5, 22000, 32, 32, student)
    inputs = small_raw_tape(steps=5, paths=22000, factors=32, innovations=4,
                            assets=32, combinations=32, student=student, layout=order)
    return spec, inputs


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_shared"])
def test_large_short_batches_preserve_all_roots_transport_and_ownership(student, order, lane):
    spec, inputs = large_short_fixture(student, order)
    originals = {name: value.copy() for name, value in inputs.items()}
    graph = compile_compact(spec, order=order)
    canonical = compile_compact(public_factor_spec(spec), order=order)
    parameters = raw_parameters(spec)
    rows = 1 if lane == "single" else 2
    starts, ends = np.zeros(rows, np.int64), np.full(rows, 5, np.int64)
    bundle = SharedInputBundle.from_inputs(inputs) if lane == "process_shared" else None
    transported = bundle.arrays if bundle is not None else inputs
    try:
        with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
            reference = execute(scheduler, canonical, transported, starts, ends, parameters, timeout=60.)
            assert_canonical(reference)
            first = execute(scheduler, graph, transported, starts, ends, parameters, timeout=60.)
            saved = saved_outputs(first)
            later = execute(scheduler, graph, transported, starts, ends, parameters, timeout=60.)
            for result in (first, later):
                assert result.plan.lane == ("process" if lane == "process_shared" else lane)
                assert_private(result)
                compare_selected(result, reference)
                assert chunk_sum(result, "compact_path_batches") == rows * 8
                assert chunk_sum(result, "compact_path_groups") == rows * 32
                assert chunk_sum(result, "compact_path_raw_rows") == rows * 8 * 5 * 22000
                assert chunk_sum(result, "compact_path_raw_scale_elements") == (
                    rows * 5 * 22000 if student else 0
                )
                for chunk in result.audit["native_chunks"]:
                    assert chunk["operator_workspace_capacity_bytes"] <= result.plan.estimated_worker_scratch_bytes
                assert result.audit["cpu_tokens"] <= 2
                if lane == "process_shared":
                    assert result.audit["output_copy_bytes"] == 0
                    assert result.audit["output_ownership"] == "native_shared_region"
            check_ownership(first, later, saved, transported)
            for name, original in originals.items():
                ieee_bits(transported[name], original)
                assert not transported[name].flags.writeable
    finally:
        if bundle is not None:
            bundle.release()
    for name, original in originals.items():
        ieee_bits(inputs[name], original)


@pytest.mark.parametrize("fault", ["zero_initial", "bad_post_probability", "bad_scale"])
def test_large_short_batch_errors_match_canonical_and_scheduler_recovers(fault):
    spec, inputs = large_short_fixture(True, "C")
    graph = compile_compact(spec)
    canonical = compile_compact(public_factor_spec(spec))
    parameters = raw_parameters(spec)
    starts, ends = np.array([0], np.int64), np.array([5], np.int64)
    bad_inputs, bad_parameters = dict(inputs), dict(parameters)
    if fault == "zero_initial":
        bad_parameters["n0"] = 0.
    elif fault == "bad_post_probability":
        bad_inputs["qreturn"] = np.array([.05, 1.1, .95], np.float64)
        bad_inputs["qreturn"].flags.writeable = False
    else:
        bad_parameters["df"] = 0.
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        messages = []
        for compiled in (canonical, graph):
            with pytest.raises((ValueError, RuntimeError)) as error:
                execute(scheduler, compiled, bad_inputs, starts, ends, bad_parameters, timeout=60.)
            messages.append((type(error.value), str(error.value)))
        assert messages[0] == messages[1]
        reference = execute(scheduler, canonical, inputs, starts, ends, parameters, timeout=60.)
        recovered = execute(scheduler, graph, inputs, starts, ends, parameters, timeout=60.)
        compare_selected(recovered, reference)
        assert chunk_sum(recovered, "compact_path_batches") == 8
