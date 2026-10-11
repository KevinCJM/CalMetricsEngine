"""Paired fixed transformed-tape MC projection subchain; never full MC proof.

Execute only under the root task's serial numerical-run authority. A selected
native installation is loaded via --install/CME_INSTALL, with no source-tree
fallback. Independent complete period and contribution outputs are retained on
both sides; two-result replacement, preparation and actual build audit are
reported. Same CPU budget defaults to one; the NJIT reference is explicitly
serial even when an exploratory native CPU ceiling above one is requested.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

STAGE = Path(__file__).resolve().parent
DEFAULT_INSTALL = STAGE / "install"
SCOPE = "healthy_fixed_transformed_tape_selected_assets_complete_period_and_contributions"
GAPS = ["primitive_random_draw_generation", "old_seed_equivalence", "covariance_root_algorithm",
        "ordered_first_fault_prefix_diagnostics", "axis_nav_peak_drawdown", "fan_tail_statistics",
        "platform_http_business_integration"]


def verified_source_identity(manifest, package):
    helper = Path('/private/tmp/calmetrics-platform-latent-closure/memory.py')
    spec = importlib.util.spec_from_file_location('projection_identity_check', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.identity(manifest, package)
    if not result['build_id_verified']:
        raise AssertionError('current source and selected wheel do not match the manifest')
    result['identity_helper_sha256'] = hashlib.sha256(helper.read_bytes()).hexdigest()
    return result


def bounded_execution_audit(actual, output_bytes):
    chunks = actual.audit['native_chunks']
    if (not chunks or any(chunk['fused_pointwise_calls'] <= 0 or
            chunk['pointwise_tiles'] <= 0 or chunk['numeric_arena_bytes'] != 0
            for chunk in chunks) or
            sum(chunk['direct_output_bytes'] for chunk in chunks) != output_bytes):
        raise AssertionError('bounded projection did not write every selected output directly')


def load_runtime(install=None, cpu=1):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(cpu)
    os.environ["NUMBA_NUM_THREADS"] = str(cpu)
    path = Path(install or os.environ.get("CME_INSTALL", DEFAULT_INSTALL)).resolve()
    if not (path / "calmetrics_engine").is_dir():
        raise RuntimeError("selected installed package is missing: " + str(path))
    sys.path.insert(0, str(path))
    sys.path.insert(0, str(STAGE))
    import numpy as np
    import calmetrics_engine as package
    from reference import service_chain, shared_chain
    if not Path(package._native.__file__).resolve().is_relative_to(path):
        raise RuntimeError("native module was already loaded from a different installation")
    return np, package, service_chain, shared_chain, path


def inputs(np, n=120, factors=4, assets=3, combinations=2, steps=None, seed=17):
    if min(n, factors, assets, combinations) < 1:
        raise ValueError("all dimensions must be positive")
    if n * factors > 8_000_000:
        raise ValueError("fixture exceeds source-shaped 8M transformed-factor budget")
    if steps is None:
        steps = min(200, n)
        if n % steps:
            steps = 1
    if steps < 1 or n % steps:
        raise ValueError("n must be divisible by positive steps")
    paths = n // steps
    rng = np.random.default_rng(seed)
    x = rng.normal(0., .01, size=(steps, paths, factors))
    beta = rng.normal(0., .08, size=(assets, factors))
    initial = rng.normal(0., .0002, size=assets)
    weights = rng.uniform(.1, 1., size=(combinations, assets))
    for combination in range(combinations):
        # This is deterministic fixture preparation, not native/NJIT hot work.
        weights[combination] /= weights[combination].sum()
    for value in (x, beta, initial, weights):
        value.flags.writeable = False
    return dict(x=x, beta=beta, initial=initial, weights=weights, steps=steps,
                paths=paths, n=n, factors=factors, assets=assets, combinations=combinations)


def graph_for(package, data, order="C", error_policy="raise"):
    types = {
        "x": dict(kind="tensor", axes=["time", "path", "factor"], shape=["T", "P", "F"]),
        "beta": dict(kind="matrix", axes=["asset", "factor"], shape=["A", "F"]),
        "initial": dict(kind="vector", axes=["asset"], shape=["A"]),
    }
    roots = {}
    simple = "expm1(affine_contract(x,transpose(beta),initial))"
    for combination in range(data["combinations"]):
        name = f"w{combination}"
        types[name] = dict(kind="vector", axes=["asset"], shape=["A"])
        weighted = f"multiply(broadcast_like({name},{simple}),{simple})"
        roots[f"period_{combination}"] = f"swap_axes(sum_axis({weighted},2),0,1)"
        for asset in range(data["assets"]):
            weight = f"first(slice_axis({name},0,{asset},{asset + 1},1))"
            value = f"slice_axis({simple},2,{asset},{asset + 1},1)"
            roots[f"contribution_{combination}_{asset}"] = f"sum(multiply({weight},{value}))/{data['paths']}"
    return package.GraphCompiler(types).compile(roots, result_format="typed", output_order=order,
                                                error_policy=error_policy), list(roots)


def setup(np, package, data, mode="ordinary", cpu=1, order="C", reference=None):
    graph, names = graph_for(package, data, order)
    selected = {name: data[name] for name in ("x", "beta", "initial")}
    for combination in range(data["combinations"]):
        selected[f"w{combination}"] = data["weights"][combination]
    starts = np.array([0], np.int64)
    ends = np.array([data["steps"]], np.int64)
    owner = package.AdaptiveScheduler(cpu_budget=cpu)
    try:
        if mode == "prepared":
            prepared = owner.prepare_execution(graph, selected, starts, ends)
            native = prepared.run_snapshot
        else:
            native = lambda: owner.execute(graph, selected, starts, ends)
    except BaseException:
        owner.close()
        raise
    baseline = None if reference is None else lambda: reference(data["x"], data["beta"], data["initial"], data["weights"])
    return owner, native, baseline, graph, names


def reference_values(np, result):
    periods, contributions = result
    values = []
    for period, contribution in zip(periods, contributions, strict=True):
        values.append(period)
        values.extend(np.asarray(value, dtype=np.float64) for value in contribution)
    return values


def compare(np, actual, expected, names):
    targets = reference_values(np, expected)
    if len(actual.outputs) != len(targets) or len(targets) != len(names):
        raise AssertionError("complete selected output schema mismatch")
    details = {}
    for output, target, name in zip(actual.outputs, targets, names, strict=True):
        value = output.values[0]
        np.testing.assert_array_equal(output.root_statuses, [0])
        if value.shape != target.shape or value.dtype != target.dtype:
            raise AssertionError(name + ": output shape/dtype mismatch")
        np.testing.assert_allclose(value, target, rtol=2e-12, atol=2e-13, equal_nan=True, err_msg=name)
        np.testing.assert_array_equal(np.isnan(value), np.isnan(target))
        np.testing.assert_array_equal(np.isposinf(value), np.isposinf(target))
        np.testing.assert_array_equal(np.isneginf(value), np.isneginf(target))
        np.testing.assert_array_equal(value <= -1., target <= -1.)
        delta = np.abs(value - target)
        details[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                             max_absolute_error=float(delta.max(initial=0)), nbytes=value.nbytes)
    # Independent invariant: contributions summed over assets reconstruct mean
    # cumulative period return. These differ in reduction order, so use a
    # predeclared floating tolerance rather than requiring identical bits.
    for period, contribution in zip(expected[0], expected[1], strict=True):
        np.testing.assert_allclose(contribution.sum(), period.sum() / period.shape[0], rtol=2e-12, atol=2e-13)
    if actual.audit["result_lifetime"] != "independent":
        raise AssertionError("benchmark requires independent native output")
    for key in ("python_fallback", "python_operator_calls", "request_time_compilation"):
        if actual.audit[key]:
            raise AssertionError("unexpected native runtime callback/compile")
    return details


def identity(package):
    from reference import FROZEN_SOURCE_DATE, FROZEN_SOURCE_SHA256, FROZEN_SOURCE_SYMBOL
    binary = Path(package._native.__file__).resolve()
    return dict(native_path=str(binary), native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                frozen_source_date=FROZEN_SOURCE_DATE, frozen_source_sha256=FROZEN_SOURCE_SHA256,
                frozen_source_symbol=FROZEN_SOURCE_SYMBOL,
                reference_sha256=hashlib.sha256((STAGE / "reference.py").read_bytes()).hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--implementation", choices=("both", "cpp", "njit"), default="both")
    parser.add_argument("--reference", choices=("service", "shared"), default="service")
    parser.add_argument("--mode", choices=("ordinary", "prepared"), default="ordinary")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--factors", type=int, default=4)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--combinations", type=int, default=2)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cpu < 1 or args.rounds < 2:
        parser.error("cpu>=1 and rounds>=2 required")
    np, package, service, shared, install = load_runtime(args.install, args.cpu)
    before_identity = verified_source_identity(args.identity_manifest, package)
    from threadpoolctl import threadpool_info, threadpool_limits
    from numba import set_num_threads
    set_num_threads(args.cpu)
    reference = service if args.reference == "service" else shared
    with threadpool_limits(limits=args.cpu):
        start = time.perf_counter_ns()
        data = inputs(np, args.n, args.factors, args.assets, args.combinations, args.steps, args.seed)
        input_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        owner, native, baseline, graph, names = setup(np, package, data, args.mode, args.cpu, args.order, reference)
        setup_ms = (time.perf_counter_ns() - start) / 1e6
        retained = None
        try:
            # Same readonly signature, tiny geometry: machine compilation is
            # explicit preparation and excluded from both warmed timings.
            warm = inputs(np, 12, args.factors, args.assets, args.combinations, 2, args.seed)
            start = time.perf_counter_ns()
            reference(warm["x"], warm["beta"], warm["initial"], warm["weights"])
            compile_ms = (time.perf_counter_ns() - start) / 1e6
            del warm
            retained = {"cpp": native(), "njit": baseline()}
            parity = compare(np, retained["cpp"], retained["njit"], names)
            output_bytes = sum(p['nbytes'] for p in parity.values())
            bounded_execution_audit(retained['cpp'], output_bytes)
            if retained['cpp'].audit['engine_build_id'] != before_identity['manifest_build_id']:
                raise AssertionError('execution audit differs from the verified source identity')
            # Ownership proof is outside timing: two calls keep independent full
            # results, and the old result remains equal to the frozen reference.
            second_native = native()
            second_reference = baseline()
            compare(np, second_native, second_reference, names)
            for old, new in zip(retained['cpp'].outputs, second_native.outputs, strict=True):
                if np.shares_memory(old.values[0], new.values[0]):
                    raise AssertionError('native results alias across independent snapshots')
            for old, new in zip(reference_values(np, retained['njit']),
                                reference_values(np, second_reference), strict=True):
                if np.shares_memory(old, new):
                    raise AssertionError('reference results alias across calls')
            compare(np, retained['cpp'], retained['njit'], names)
            del second_native, second_reference
            audit = dict(retained["cpp"].audit)
            samples = {"cpp_ms": [], "njit_ms": []}
            if not args.correctness_only:
                functions = {"cpp": native, "njit": baseline}
                backends = ["cpp", "njit"] if args.implementation == "both" else [args.implementation]
                for round_index in range(args.rounds):
                    order = backends if round_index % 2 == 0 else backends[::-1]
                    for backend in order:
                        start = time.perf_counter_ns()
                        retained[backend] = functions[backend]()
                        samples[backend + "_ms"].append((time.perf_counter_ns() - start) / 1e6)
                compare(np, retained["cpp"], retained["njit"], names)
                bounded_execution_audit(retained['cpp'], output_bytes)
            after_identity = verified_source_identity(args.identity_manifest, package)
            if before_identity != after_identity:
                raise AssertionError('source/wheel identity changed during timing')
            medians = {name: statistics.median(values) if values else None for name, values in samples.items()}
            report = dict(scope=SCOPE, remaining_gaps=GAPS, install=str(install),
                          shape={key: data[key] for key in ("n", "steps", "paths", "factors", "assets", "combinations")},
                          mode=args.mode, reference=args.reference, implementation=args.implementation,
                          cpu_budget=args.cpu, reference_lane="serial", njit_parallel=False,
                          same_cpu_budget_primary_evidence=args.cpu == 1,
                          output_order=args.order, output_count=len(names), output_bytes=sum(p["nbytes"] for p in parity.values()),
                          two_result_lifecycle="independent_previous_plus_new_per_backend",
                          input_preparation_ms=input_ms, graph_or_prepared_construction_ms=setup_ms,
                          njit_tiny_signature_compile_and_warm_ms_excluded=compile_ms,
                          rounds=0 if args.correctness_only else args.rounds, samples=samples, medians_ms=medians,
                          cpp_over_njit=(medians["cpp_ms"] / medians["njit_ms"] if medians["cpp_ms"] is not None and medians["njit_ms"] else None),
                          parity=parity, audit=audit, build_info=package.build_info(),
                          identity=identity(package), graph=graph.metadata(), threadpools=threadpool_info(),
                          source_identity=before_identity, source_unchanged=True,
                          independent_two_result_ownership_verified=True,
                          notes=["Fixed transformed tape, all supplied assets selected; no production random/covariance/available selection work is timed.",
                                 "Frozen service reference preserves original ordered projection checks; full first-fault output protocol is not represented by this DAG.",
                                 "All complete period matrices and per-asset contribution roots are returned on both sides; no full L/R audit root is selected.",
                                 "Service reference repeats affine/expm1 per combination; shared is an additional bounded-memory optimized NJIT comparator.",
                                 "Prepared uses run_snapshot, never borrowed output; previous and new full results remain independent.",
                                 "The paired time process retains one result from each backend; this is per-backend two-result replacement, not a two-result total-RSS claim.",
                                 "No unrelated numerical tasks should run concurrently; this driver performs paired calls sequentially.",
                                 "CPU>1 is exploratory: references have no prange; do not present those runs as same-effective-core speedup.",
                                 "This time driver is not an incremental heap/RSS measurement; use the existing memory helper separately."])
            text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
            if args.output:
                args.output.write_text(text + "\n")
            print(text, flush=True)
        finally:
            if retained is not None:
                retained.clear()
            owner.close()


if __name__ == "__main__":
    main()
