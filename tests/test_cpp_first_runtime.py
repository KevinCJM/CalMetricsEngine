"""Acceptance of the C++-only compiler/planner/thread/process path.

These tests exercise native ownership and failures, not only happy-path parity.
Python pools below are test drivers, never part of the engine's numerical path.
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import gc
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pytest

from calmetrics_engine import (
    AdaptivePlanner,
    AdaptiveScheduler,
    GraphCompileError,
    GraphCompiler,
    PlannerConfig,
    SharedArrayDescriptor,
    SharedArrayOwner,
    SharedInputBundle,
)
from calmetrics_engine.shared import attach_shared_array


def config(lane: str) -> PlannerConfig:
    return PlannerConfig(
        thread_work_units=1.0 if lane != "single" else 1e100,
        process_work_units=1.0 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )


def fixture(rows=8, length=128):
    values = np.ascontiguousarray(1.1 + np.sin(np.arange(rows * length) * 0.031) * 0.01)
    starts = np.arange(rows, dtype=np.int64) * length
    ends = starts + length
    return values, starts, ends


def graph():
    return GraphCompiler({"x": "series"}).compile(
        ["mean(x)", "std(x,1)", "median(x)", "quantile(x,0.75)"]
    )


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_sync_request_never_uses_python_parser_or_numerical_pools(monkeypatch, lane):
    values, starts, ends = fixture()
    expected = np.asarray(
        [
            [a.mean(), a.std(ddof=1), np.median(a), np.quantile(a, 0.75)]
            for a in values.reshape(8, -1)
        ]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Python compilation/partition/pool used in native request")

    with monkeypatch.context() as patch:
        patch.setattr(ast, "parse", forbidden)
        patch.setattr(concurrent.futures, "ThreadPoolExecutor", forbidden)
        patch.setattr(concurrent.futures, "ProcessPoolExecutor", forbidden)
        patch.setattr(multiprocessing, "get_context", forbidden)
        patch.setattr(np, "unique", forbidden)
        patch.setattr(np, "cumsum", forbidden)
        compiled = graph()
        with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
            plan = scheduler.plan(compiled, {"x": values}, starts, ends)
            result = scheduler.execute(compiled, {"x": values}, starts, ends, plan=plan)
    np.testing.assert_allclose(result.values, expected, rtol=2e-12, atol=2e-14)
    assert compiled.metadata()["compiler_backend"] == "cpp"
    assert plan.metadata()["planner_backend"] == "cpp"
    assert result.audit["scheduler_backend"] == "cpp"
    assert result.audit["python_worker_callbacks"] == 0
    assert result.audit["python_native_transitions"] == 1
    if lane.startswith("process"):
        assert result.audit["native_processes"] == 2
        assert plan.use_shared_memory == (lane == "process_shared")
    else:
        assert result.audit["native_processes"] == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("-2**2", -4),
        ("2**-2", 0.25),
        ("2**3**2", 512),
        ("(-2)**2", 4),
        ("1e-2 + .25", 0.26),
        ("1 + 2*3", 7),
        ("(1+2)*3", 9),
        ("4 # ignored", 4),
    ],
)
def test_native_parser_precedence(source, expected):
    compiled = GraphCompiler({"x": "series"}).compile(source)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            compiled, {"x": np.ones(1)}, np.array([0], np.int64), np.array([1], np.int64)
        )
    np.testing.assert_allclose(result.values, [[expected]], rtol=1e-14)


def test_low_level_program_execute_uses_native_cpu_admission():
    compiled = GraphCompiler({"x": "series"}).compile("mean(x)")
    values = np.arange(8.0)
    starts = np.asarray([0], dtype=np.int64)
    ends = np.asarray([values.size], dtype=np.int64)
    output = np.empty((1, 1), dtype=np.float64)
    audit = compiled._program.execute((values,), starts, ends, output)

    assert output[0, 0] == values.mean()
    assert audit["scheduler_backend"] == "process_wide_cpp_cpu_admission"
    assert audit["cpu_tokens"] == 1


def test_native_parser_unicode_pickle_and_limits():
    compiled = GraphCompiler({"净值": "series"}).compile(["mean(净值)", "std(净值, 1)"])
    restored = pickle.loads(pickle.dumps(compiled))
    assert restored.fingerprint == compiled.fingerprint
    assert restored.native_nodes() == compiled.native_nodes()
    for expression in ("(" * 200 + "1" + ")" * 200, "x " * (1 << 20), "where(x,x,x,x,x)"):
        with pytest.raises(GraphCompileError):
            GraphCompiler({"x": "series"}).compile(expression)


def test_borrowed_view_liveness_extends_owning_arena():
    # lag's backing add(x,1) must survive both the unrelated branch and its reuse.
    compiled = GraphCompiler({"x": "series"}).compile(
        [
            "mean(lag(x+1,1)) + mean(x*4) + std(lag(x+1,1),1)",
            "mean(lag(lag(x+1,1),1)) + mean(x+5)",
        ]
    )
    x = np.arange(1.0, 65.0)
    expected = [
        (x[:-1] + 1).mean() + (x * 4).mean() + (x[:-1] + 1).std(ddof=1),
        (x[:-2] + 1).mean() + (x + 5).mean(),
    ]
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            compiled, {"x": x}, np.array([0], np.int64), np.array([x.size], np.int64)
        )
    np.testing.assert_allclose(result.values[0], expected, rtol=1e-13)
    assert compiled.metadata()["liveness_policy"] == "borrowed_view_aware"


@pytest.mark.parametrize(
    "mutation", ["input_dtype", "input_shape", "parameter_dtype", "interval_shape"]
)
@pytest.mark.filterwarnings(
    "ignore:Setting the (dtype|shape) on a NumPy array has been deprecated:DeprecationWarning"
)
def test_prepared_rejects_changed_array_geometry(mutation):
    # Deliberately mutate the SAME ndarray to verify stale-pointer rejection.
    # NumPy 2.5 warns about these setters; a new view would not test that hazard.
    compiled = GraphCompiler({"x": "series", "k": "scalar"}).compile("mean(x)+k")
    x = np.arange(8.0)
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    params = np.array([0.2])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(compiled, {"x": x}, starts, ends, parameters=params)
        prepared.run()
        if mutation == "input_dtype":
            x.dtype = np.int64
        elif mutation == "input_shape":
            x.shape = (2, 4)
        elif mutation == "parameter_dtype":
            params.dtype = np.int64
        else:
            starts.shape = (1, 1)
        with pytest.raises(ValueError, match="PREPARED_INPUT_CHANGED"):
            prepared.run()


def test_mutated_intervals_reject_stale_plan_but_replanning_updates_audit():
    compiled = graph()
    values, starts, ends = fixture(rows=1, length=16)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(compiled, {"x": values}, starts, ends)
        first = scheduler.execute(compiled, {"x": values}, starts, ends, plan=plan)
        ends[0] = 8
        with pytest.raises(ValueError, match="stale execution plan"):
            scheduler.execute(compiled, {"x": values}, starts, ends, plan=plan)
        second = scheduler.execute(compiled, {"x": values}, starts, ends)
    assert first.audit["native_chunks"][0]["max_window"] == 16
    assert second.audit["native_chunks"][0]["max_window"] == 8
    assert not np.shares_memory(first.values, second.values)


def test_cache_revalidates_parameter_keys_and_reads_mutated_values():
    compiled = GraphCompiler({"x": "series", "k": "scalar"}).compile("mean(x)+k")
    x = np.arange(8.0)
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    parameters = np.array([1.0])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(compiled, {"x": x}, starts, ends)
        a = scheduler.execute(compiled, {"x": x}, starts, ends, parameters=parameters, plan=plan)
        parameters[0] = 2
        x[:] += 3
        b = scheduler.execute(compiled, {"x": x}, starts, ends, parameters=parameters, plan=plan)
        with pytest.raises(ValueError, match="keys mismatch"):
            scheduler.execute(
                compiled, {"x": x}, starts, ends, parameters={"k": 2, "extra": 1}, plan=plan
            )
    assert a.values[0, 0] == 4.5
    assert b.values[0, 0] == 8.5


def test_prepared_output_reuse_and_closed_engine():
    compiled = graph()
    values, starts, ends = fixture(rows=1, length=16)
    scheduler = AdaptiveScheduler(cpu_budget=1)
    prepared = scheduler.prepare_execution(compiled, {"x": values}, starts, ends)
    a = prepared.run()
    address = a.ctypes.data
    values[:] += 0.1
    b = prepared.run()
    assert address == b.ctypes.data
    scheduler.close()
    scheduler.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.run()


@pytest.mark.parametrize("lane", ["thread", "process_inline", "process_shared"])
def test_worker_error_drains_and_engine_can_be_reused(lane):
    compiled = GraphCompiler({"x": "series"}).compile(["mean(log(x))", "median(x)"])
    values, starts, ends = fixture()
    values[0] = -1
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        with pytest.raises((RuntimeError, ValueError), match="DOMAIN_ERROR"):
            scheduler.execute(compiled, {"x": values}, starts, ends, timeout=10)
        values[0] = 1.1
        result = scheduler.execute(compiled, {"x": values}, starts, ends, timeout=10)
        assert scheduler.peak_active_cpu_tokens <= 2
    assert np.isfinite(result.values).all()


def test_missing_worker_path_fails_without_losing_cpu_tokens(tmp_path):
    compiled = graph()
    values, starts, ends = fixture()
    with AdaptiveScheduler(
        cpu_budget=2, config=config("process_shared"), worker_path=tmp_path / "missing-worker"
    ) as scheduler:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="worker"):
                scheduler.execute(compiled, {"x": values}, starts, ends, timeout=3)


def test_thread_timeout_and_close_are_resource_safe():
    compiled = graph()
    values, starts, ends = fixture(rows=16, length=4096)
    with AdaptiveScheduler(cpu_budget=2, config=config("thread")) as scheduler:
        with pytest.raises(TimeoutError):
            scheduler.execute(compiled, {"x": values}, starts, ends, timeout=0)
        result = scheduler.execute(compiled, {"x": values}, starts, ends, timeout=5)
        assert np.isfinite(result.values).all()


def test_concurrent_requests_share_native_cpu_admission():
    compiled = graph()
    values, starts, ends = fixture(rows=8, length=2048)
    with AdaptiveScheduler(cpu_budget=2, config=config("thread")) as scheduler:

        def run(_):
            return scheduler.execute(compiled, {"x": values}, starts, ends).values

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as callers:
            outputs = list(callers.map(run, range(18)))
        assert 1 <= scheduler.peak_active_cpu_tokens <= 2
    for actual in outputs[1:]:
        np.testing.assert_array_equal(actual, outputs[0])
        assert not np.shares_memory(actual, outputs[0])


def test_async_request_is_only_an_interface_bridge():
    compiled = graph()
    values, starts, ends = fixture()

    async def run():
        with AdaptiveScheduler(cpu_budget=2, config=config("thread")) as scheduler:
            outputs = await scheduler.execute_many_async(
                [
                    dict(graph=compiled, inputs={"x": values}, starts=starts, ends=ends)
                    for _ in range(6)
                ]
            )
            assert scheduler.peak_active_cpu_tokens <= 2
            return outputs

    outputs = asyncio.run(run())
    for result in outputs:
        assert result.audit["scheduler_backend"] == "cpp"
        assert result.plan.async_orchestration
        np.testing.assert_array_equal(result.values, outputs[0].values)


def test_shared_view_keeps_mapping_alive_after_owner_release():
    owner = SharedArrayOwner.from_array(np.arange(16.0))
    view = owner.array
    descriptor = owner.descriptor
    owner.release()
    gc.collect()
    np.testing.assert_array_equal(view, np.arange(16.0))
    with pytest.raises(RuntimeError, match="released"):
        _ = owner.array
    attached_owner, attached = attach_shared_array(descriptor)
    assert not attached.flags.writeable
    with pytest.raises(ValueError):
        attached.setflags(write=True)
    attached_owner.close()
    np.testing.assert_array_equal(attached, view)
    del attached, attached_owner, view
    gc.collect()
    with pytest.raises(RuntimeError):
        attach_shared_array(descriptor)


def test_shared_bundle_release_is_idempotent_and_invalidates_interface():
    bundle = SharedInputBundle.from_inputs({"x": np.arange(8.0)})
    views = bundle.arrays
    descriptor = bundle.descriptors["x"]
    assert descriptor.readonly
    assert not views["x"].flags.writeable
    with pytest.raises(ValueError):
        views["x"].setflags(write=True)
    attached_owner, attached = attach_shared_array(descriptor, readonly=False)
    assert not attached.flags.writeable
    with pytest.raises(ValueError):
        attached.setflags(write=True)
    attached_owner.close()
    bundle.release()
    bundle.release()
    np.testing.assert_array_equal(views["x"], np.arange(8.0))
    with pytest.raises(RuntimeError, match="released"):
        _ = bundle.descriptors


def test_shared_rejects_objects_and_out_of_bounds_descriptor():
    with pytest.raises(TypeError):
        SharedArrayOwner.from_array(np.array([object()], dtype=object))
    with SharedArrayOwner.from_array(np.arange(8.0)) as owner:
        descriptor = SharedArrayDescriptor(owner.descriptor.name, (10000,), "<f8")
        with pytest.raises((ValueError, RuntimeError)):
            attach_shared_array(descriptor)
    with pytest.raises((ValueError, OverflowError)):
        SharedArrayOwner.empty((-1,), np.float64)


def test_shared_process_output_is_zero_copy_and_survives_engine_close():
    compiled = graph()
    values, starts, ends = fixture()
    with AdaptiveScheduler(cpu_budget=2, config=config("process_shared")) as scheduler:
        first = scheduler.execute(compiled, {"x": values}, starts, ends)
        snapshot = first.values.copy()
        second = scheduler.execute(compiled, {"x": values}, starts, ends)
    assert first.audit["output_copy_bytes"] == 0
    assert first.audit["output_ownership"] == "native_shared_region"
    assert first.values.base is not None
    assert not np.shares_memory(first.values, second.values)
    view = first.values
    del first, second
    gc.collect()
    np.testing.assert_array_equal(view, snapshot)


def test_native_worker_binary_is_installed():
    import calmetrics_engine

    suffix = ".exe" if __import__("sys").platform == "win32" else ""
    worker = Path(calmetrics_engine.__file__).parent / ("calmetrics_worker" + suffix)
    assert worker.is_file()
    assert worker.stat().st_size > 0


def test_supplied_plan_cannot_bypass_cpu_or_hard_stop_contract():
    compiled = graph()
    values, starts, ends = fixture()
    planner = AdaptivePlanner(config("thread"))
    plan = planner.plan(compiled, {"x": values}, starts, ends, cpu_budget=4)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises(ValueError, match="CPU budget"):
            scheduler.execute(compiled, {"x": values}, starts, ends, plan=plan)
    with AdaptiveScheduler(cpu_budget=4) as scheduler:
        with pytest.raises(ValueError, match="hard_stop"):
            scheduler.execute(compiled, {"x": values}, starts, ends, plan=plan, hard_stop=True)


@pytest.mark.parametrize('dictionary_type', [dict, type('InputDict', (dict,), {})])
def test_cached_entry_preserves_mapping_checks_and_independent_results(dictionary_type):
    compiled = GraphCompiler({'x': 'series', 'k': 'scalar'}).compile('mean(x)+k')
    x = np.arange(8.)
    inputs, parameters = dictionary_type(x=x), dictionary_type(k=1.)
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(compiled, inputs, starts, ends)
        first = scheduler.execute(compiled, inputs, starts, ends, parameters=parameters, plan=plan)
        cached = scheduler.execute(compiled, inputs, starts, ends, parameters=parameters, plan=plan)
        assert cached.audit['prepared_cached']
        parameters['k'] = 2.
        updated = scheduler.execute(compiled, inputs, starts, ends, parameters=parameters, plan=plan)
        assert first.values[0, 0] == cached.values[0, 0] == 4.5
        assert updated.values[0, 0] == 5.5
        assert not np.shares_memory(first.values, cached.values)
        inputs['wrong'] = inputs.pop('x')
        with pytest.raises(ValueError, match='missing graph input'):
            scheduler.execute(compiled, inputs, starts, ends, parameters=parameters, plan=plan)
        inputs['x'] = inputs.pop('wrong')
        parameters['wrong'] = parameters.pop('k')
        with pytest.raises(ValueError, match='missing graph parameter'):
            scheduler.execute(compiled, inputs, starts, ends, parameters=parameters, plan=plan)


@pytest.mark.parametrize('field', ['start', 'end', 'product'])
def test_plan_recheck_rejects_in_place_descriptor_changes(field):
    compiled = GraphCompiler({'x': 'series'}).compile('mean(x)')
    x = np.arange(8.)
    starts, ends, products = np.array([0], np.int64), np.array([8], np.int64), np.array([0], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(compiled, {'x': x}, starts, ends, product_ids=products)
        before = prepared.run_snapshot()
        target = {'start': starts, 'end': ends, 'product': products}[field]
        original = target[0]
        target[0] = original + (1 if field != 'end' else -1)
        with pytest.raises(ValueError, match='stale execution plan'):
            prepared.run_snapshot()
        target[0] = original
        np.testing.assert_allclose(prepared.run_snapshot().values, before.values)
