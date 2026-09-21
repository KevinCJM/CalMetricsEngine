# Platform execution contracts

The native graph compiler accepts optional `error_policy="isolate"`, one
`root_bindings` mapping and one opaque `source_contracts` string per root, and
`minimum_observations`. The default error policy remains `raise`.

## Isolation

An isolated result exposes read-only int16 `statuses`, shaped like `values`.
Numerical errors set NaN and propagate through actual dependencies; independent
roots and subsequent intervals continue. Codes: 0 OK, 1 insufficient observations,
2 guarded scalar division by zero, 3 guarded scalar domain error, 4 unavailable numeric result (including operator sample/parameter exceptions), 7 invalid positional lookup,
9 missing interval, 10 unrecovered interval. Structural/input/resource/process
failures remain request failures. Status collection works for single, thread,
process/shared-memory and DAG-branch execution. Series isolation is per root and
interval when an operator throws; individual nonfinite output points receive 4.

## Native interval bindings

Bindings are restricted expressions parsed by the C++ compiler, with cycle,
depth and expansion limits. They support scalar and aligned-series roots. Physical inputs
share the NAV axis; bindings may derive shorter series inside each interval.
`interval_tail(x)` is a compiler-only borrowed view exposed solely inside a
binding, so `interval_tail(nav)/lag(nav)-1` retains the exact return arithmetic.
It extends backing-buffer liveness and is not an additional canonical operator.
Per-root parameter names permit shared expressions with different parameter
values. Source-contract namespaces prevent cross-version operation CSE and enter
the graph fingerprint. The platform owns historical financial DSL validation.

The binary program format is v4; decoding v1–v3 preserves their original contracts,
including strict-error defaults for v1/v2. The worker request/response protocol is v3
and requires an exact version match. Use the worker shipped
with the same native package; old workers are rejected. Status transport uses
bounded IPC frames; large numeric inputs/outputs retain existing shared-memory
transport. Planner memory estimates include status chunks, Python result copies
and IPC buffers. The 256 MiB frame limit remains enforced.

## Ownership and execution evidence

`execute()` owns independent numeric output. Prepared `run()` and `run_audit()`
borrow reusable output until the next run. `run_snapshot()` executes directly into
independent read-only output under the prepared execution lock; no post-unlock
copy race. Snapshots and their status arrays remain valid after later runs and
scheduler closure. Float64 time-series graph inputs remain C-contiguous; typed
matrix/vector, int64 and mask inputs support validated strides. All paths permit
read-only data and retain input owners without silently normalizing layout.

Result audit schema `cpp-aot-execution-1` records engine version, source/toolchain
build identity, plan fingerprint, canonical operator/IR versions, CPU budget,
dtype, source contracts and result lifetime. It explicitly reports zero Python
operator/worker callbacks, fallback and request-time machine-code compilation.
These are internal provenance records, not an authentication protocol.

Validation: `tests/test_execution_contracts.py`, the full installed-wheel test
suite, native CTest, and `tools/check_phase2_performance.py`. Platform acceptance
is recorded in the sibling platform's `docs/cpp-aot-contracts-acceptance.md`.
