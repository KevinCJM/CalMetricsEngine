"""Fresh-child memory measurement of bench.py's identical thirteen outputs.

Uses the existing continuous heap/RSS helpers without copying their machinery.
The parent CLI runs one fresh child per metric, sequentially. Numeric execution
is intentionally performed only when the root agent launches this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STAGE = Path(__file__).resolve().parent
ENGINE = Path("/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine")
DEFAULT_PROBE = Path("/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib")
DEFAULT_IDENTITY = ENGINE / "docs/physical-execution-evidence/2026-10-03-solve-pair-integration/identity.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(manifest_path, package):
    # build_info has no build hash. Resolve the real installed binary by SHA and
    # the complete source manifest, rather than inventing it from package labels.
    native_path = Path(package._native.__file__).resolve()
    actual_binary = sha(native_path)
    manifest = json.loads(manifest_path.read_text())
    expected_binary = manifest.get("installed_binaries", {}).get("calmetrics_engine/" + native_path.name)
    sources = {name: sha(ENGINE / name) for name in manifest["source"]}
    mismatches = [name for name, digest in sources.items() if digest != manifest["source"][name]]
    actual_names = {str(path.relative_to(ENGINE)) for path in (ENGINE / "cpp").rglob("*") if path.is_file()}
    actual_names.update(("CMakeLists.txt", "pyproject.toml"))
    extra = sorted(actual_names - set(manifest["source"]))
    missing = sorted(set(manifest["source"]) - actual_names)
    verified = not mismatches and not extra and not missing and actual_binary == expected_binary
    return {"manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
            "manifest_build_id": manifest["build_id"], "build_id_verified": verified,
            "native_path": str(native_path), "native_sha256": actual_binary,
            "native_matches_manifest": actual_binary == expected_binary,
            "source_count": len(sources), "source_sha256": sources,
            "source_mismatches": mismatches, "unlisted_sources": extra, "missing_sources": missing}


class NoOwner:
    def close(self):
        pass


def child(args):
    sys.path.insert(0, str(STAGE))
    import bench
    import calmetrics_engine as package
    sys.path.insert(0, str(ENGINE / "tools"))
    from benchmark_memory import measure_heap, measure_rss
    from threadpoolctl import threadpool_info, threadpool_limits

    before_identity = identity(args.identity_manifest, package)
    data = bench.inputs(args.n, args.d, args.k, args.training_count, args.seed)
    # Tiny input with identical dtype/layout/readonly signature warms machine
    # compilation before the epoch on BOTH sides. Shape and option values are
    # runtime arguments of this controlled lazy-NJIT reference.
    warm_data = bench.inputs(max(120, args.k * 10), args.d, args.k, seed=args.seed)
    start = time.perf_counter_ns()
    warm = bench.chain_reference(warm_data["raw"], warm_data["training_positions"],
                                warm_data["classify_positions"], args.k, 1,
                                args.seed, args.model == "hmm")
    del warm, warm_data
    compile_warm_ms = (time.perf_counter_ns() - start) / 1e6
    observed = {}
    prep_ms = [0.0]

    def capture(call):
        def wrapped():
            result = call()
            if not observed:
                if args.backend == "njit":
                    values = [bench.np.asarray(value) for value in result]
                else:
                    values = [output.values[0] for output in result.outputs]
                    observed["native_audit"] = dict(result.audit)
                    observed["root_statuses"] = [list(map(int, output.root_statuses)) for output in result.outputs]
                observed.update(output_count=len(values),
                                output_bytes=sum(value.nbytes for value in values),
                                actual_iterations=int(values[6]), pretrain_iterations=int(values[7]),
                                output_shapes=[list(value.shape) for value in values],
                                output_dtypes=[str(value.dtype) for value in values])
            return result
        return wrapped

    def construct():
        start = time.perf_counter_ns()
        if args.backend == "njit":
            owner = NoOwner()
            call = lambda: bench.chain_reference(data["raw"], data["training_positions"],
                                                data["classify_positions"], args.k,
                                                args.iterations, args.seed, args.model == "hmm")
        else:
            owner, call, _, _, _ = bench.setup(data, args.k, args.iterations, args.seed,
                                             args.model, args.backend)
        prep_ms[0] = (time.perf_counter_ns() - start) / 1e6
        return owner, capture(call)

    owner = function = None
    with threadpool_limits(limits=1):
        try:
            if args.metric == "heap":
                if os.environ.get("DYLD_INSERT_LIBRARIES") != str(args.heap_probe.resolve()):
                    raise RuntimeError("heap child must be launched with the specified interposition library")
                if args.phase == "execution":
                    owner, function = construct()
                    setup = lambda: (owner, function)
                else:
                    setup = construct
                owner, function, measurement = measure_heap(setup, str(args.heap_probe), args.repeats)
            else:
                owner, function = construct()
                # A fresh RSS child avoids inheriting the heap child's warm malloc
                # cache. The monitor is warmed, but the first full-size request is
                # inside the measurement (followed by retained-result replacement).
                measurement = measure_rss(function, args.repeats)
        except BaseException:
            if owner:
                owner.close()
            raise
        try:
            report = {"backend": args.backend, "metric": args.metric, "phase": args.phase,
                      "scope": "finite_fixed_realtime_random_init_state_count_equals_k",
                      "shape": {"n": args.n, "d": args.d, "k": args.k,
                                "training": data["training_positions"].size,
                                "classify": data["classify_positions"].size},
                      "model": args.model, "max_iterations": args.iterations,
                      "repeats": args.repeats, "cpu_budget": 1, "fresh_pid": os.getpid(),
                      "reference_compile_and_tiny_warm_ms_excluded": compile_warm_ms,
                      "graph_or_callable_construction_ms": prep_ms[0],
                      "measurement": measurement, "outputs": observed,
                      "measurement_scope": ("heap_epoch_includes_graph_construction" if args.metric == "heap" and args.phase == "setup"
                                            else "execution_with_graph_owner_constructed_before_observation"),
                      "build_info": package.build_info(), "threadpools": threadpool_info(),
                      "identity": before_identity,
                      "post_measurement_source_unchanged": identity(args.identity_manifest, package) == before_identity,
                      "probe_sha256": sha(args.heap_probe) if args.metric == "heap" else None,
                      "helper_sha256": sha(ENGINE / "tools/benchmark_memory.py"),
                      "reference_sha256": sha(ENGINE / "tests/latent_reference.py"),
                      "bench_sha256": sha(STAGE / "bench.py"), "driver_sha256": sha(__file__),
                      "measurement_notes": [
                          "All thirteen service/audit outputs remain independent; old/new result replacement is measured by the shared helper.",
                          "Common readonly input construction and tiny-signature NJIT compile/warm are before both measurement epochs.",
                          "Heap setup phase covers native graph construction, first execution, five helper warmups and repeated retained replacement; do not add stage peaks.",
                          "Heap execution phase excludes already-created graph/owner bytes; setup and execution reports are separate scopes, not additive peaks.",
                          "RSS is sampled at 0.5ms after monitor initialization, includes first full-size request, and reports un-subtracted idle control.",
                          "Heap interposition may miss Python arena suballocations, mappings and allocations bypassing intercepted malloc APIs; RSS contains imported-runtime/JIT baseline.",
                          "No worker processes are used by cpu_budget=1; no shared-page duplication or process-tree aggregation is claimed.",
                          "First-result scalar/schema/native audit capture is included, retains no extra arrays and is diagnostic overhead.",
                      ]}
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
        finally:
            del function
            if owner:
                owner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ordinary", "prepared", "njit"), required=True)
    parser.add_argument("--model", choices=("gmm", "hmm"), default="gmm")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--training-count", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--phase", choices=("setup", "execution"), default="setup")
    parser.add_argument("--metric", choices=("heap", "rss", "all"), default="all")
    parser.add_argument("--heap-probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--identity-manifest", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or args.iterations < 1:
        parser.error("repeats>=2 (retained old/new replacement), iterations>=1 required")
    if args.child:
        if args.metric == "all":
            parser.error("child requires a single metric")
        child(args)
        return
    if os.environ.get("DYLD_INSERT_LIBRARIES"):
        parser.error("parent must not inherit allocation instrumentation")
    reports = []
    for metric in (("heap", "rss") if args.metric == "all" else (args.metric,)):
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--backend", args.backend,
                   "--model", args.model, "--n", str(args.n), "--d", str(args.d), "--k", str(args.k),
                   "--iterations", str(args.iterations), "--seed", str(args.seed), "--repeats", str(args.repeats),
                   "--phase", args.phase, "--metric", metric, "--heap-probe", str(args.heap_probe.resolve()),
                   "--identity-manifest", str(args.identity_manifest.resolve())]
        if args.training_count is not None:
            command += ["--training-count", str(args.training_count)]
        env = dict(os.environ)
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
            env[key] = "1"
        if metric == "heap":
            env["DYLD_INSERT_LIBRARIES"] = str(args.heap_probe.resolve())
        completed = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
        reports.append(json.loads(completed.stdout))
    text = json.dumps(reports, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
