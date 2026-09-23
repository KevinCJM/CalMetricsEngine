# Platform execution contracts

For an end-to-end introduction, see the [user guide](user-guide.md); for strategy selection,
see the [execution guide](execution-guide.md). This page describes the current integration boundary.

The native graph compiler accepts optional `error_policy="isolate"`, one
`root_bindings` mapping and one opaque `source_contracts` string per root, and
`minimum_observations`. The default error policy remains `raise`.

## Isolation

An isolated compatible result exposes read-only int16 `statuses`, shaped like `values`.
Typed results expose matching arrays per root/interval through `outputs`, together with
`root_statuses` and `shape_known`; unknown failure geometry is distinct from a known empty result.
Numerical errors set NaN for float64, or False/0 placeholders for bool/int64, and propagate through actual dependencies; independent
roots and subsequent intervals continue. Codes: 0 OK, 1 insufficient observations,
2 guarded scalar division by zero, 3 guarded scalar domain error, 4 unavailable numeric result (including operator sample/parameter exceptions), 7 invalid positional lookup,
9 missing interval, 10 unrecovered interval. Structural/input/resource/process
failures remain request failures. Status collection works for single, thread,
process/shared-memory and DAG-branch execution. Actual rolling/group/segment exceptions retain per-position
failure provenance through mapped downstream dependencies; operators without an exact positional
dependency map conservatively invalidate their dependent node. Independent roots and healthy
mapped segments continue. Nonfinite float64 output points receive 4. Normal missing/warmup values
do not themselves become captured execution exceptions. See [stateful failure propagation](stateful-series-design.md#分段异常向下游传播).

Strict mode preserves the ordinary `rolling_apply` compatibility rule: without a segment-failure
context, a numerical body exception becomes NaN for that window; strict `group_apply` still raises.
Choose `isolate` when downstream bool/int64 results must retain explicit window/group failure status.

## Native interval bindings

Bindings are restricted expressions parsed by the C++ compiler, with cycle,
depth and expansion limits. They support public scalar/series/vector/matrix roots, including heterogeneous typed results. Physical inputs
share the NAV axis; bindings may derive shorter series inside each interval.
`interval_tail(x)` is a compiler-only borrowed view exposed solely inside a
binding, so `interval_tail(nav)/lag(nav)-1` retains the exact return arithmetic.
It extends backing-buffer liveness and is not an additional canonical operator.
Per-root parameter names permit shared expressions with different parameter
values. Source-contract namespaces prevent cross-version operation CSE and enter
the graph fingerprint. The platform owns historical financial DSL validation.

The binary program format is v6 with per-root dtype/rank schemas; decoding v1–v5 preserves
their original output contracts, including strict-error defaults for v1/v2. The worker request/response protocol is v5
and requires an exact version match. Use the worker shipped
with the same native package; old workers are rejected. Status transport uses
bounded IPC frames; large numeric inputs/outputs retain existing shared-memory
transport. Planner memory estimates include status chunks, Python result copies
and IPC buffers. The 256 MiB frame limit remains enforced.

## Ownership and execution evidence

`execute()` owns independent numeric output. Prepared execution currently requires a single-lane plan;
thread/process plans are rejected. Prepared `run()` and `run_audit()`
borrow reusable output until the next run. `run_snapshot()` executes directly into
independent read-only output under the prepared execution lock; no post-unlock
copy race. Snapshots and their status arrays remain valid after later runs and
scheduler closure. Float64 time-series graph inputs remain C-contiguous; typed
matrix/vector, int64 and mask inputs support validated strides. All paths permit
read-only data and retain input owners without silently normalizing layout.

Compatible results retain audit schema `cpp-aot-execution-1`; typed results use
`cpp-aot-execution-2` with `result_protocol="typed-results-1"`, `output_dtype="per_output"`,
per-root schemas and native capacity/layout information. A platform accepting only the first schema
must explicitly adapt and validate the new protocol before use; engine tests do not certify that integration.
Both schemas record engine version, source/toolchain
build identity, plan fingerprint, canonical operator/IR versions, CPU budget,
dtype, source contracts and result lifetime. It explicitly reports zero Python
operator/worker callbacks, fallback and request-time machine-code compilation.
These are internal provenance records, not an authentication protocol.

Validation: `tests/test_execution_contracts.py`, the full installed-wheel test
suite, native CTest, and `tools/check_phase2_performance.py`. Platform acceptance
is recorded in the sibling platform's `docs/cpp-aot-contracts-acceptance.md`.
