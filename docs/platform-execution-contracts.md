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

The binary program format is v8 with per-root output-order policies; all-C programs may retain
v7 encoding. Decoding v1–v7 preserves their C output contract, including strict-error defaults for
v1/v2. The worker request/response protocol is v23, with request magic `0x434d453300000017`
and response magic `0x434d455300000017`; it transports actual raw factor-prefix
row/scale and coefficient-packing counters, guarded-LU attempts/successes/replays and
residual-panel materialization counters, and requires an exact version match. Use the worker shipped
with the same native package; old workers are rejected. Status transport uses
bounded IPC frames; large numeric inputs/outputs retain existing shared-memory
transport. Planner memory estimates include status chunks, Python result copies
and IPC buffers. The 256 MiB frame limit remains enforced.

IPC21 introduced `highway_vqsort_calls`, the actual completed Highway numeric-line
sort calls from `quantiles_axis` and admitted compact consumers; current v23 transport
retains this field. The existing
`native_library_workspace_reserve_bytes` records a chunk's maximum library resource estimate,
not measured allocation or RSS. Sorting eligibility and the estimated 16KiB allowance per active
worker are described in the [execution guide](execution-guide.md#按轴扫描与紧凑消费者执行).

## Ownership and execution evidence

`execute()` owns independent numeric output. Prepared execution currently requires a single-lane plan;
thread/process plans are rejected. Prepared `run()` and `run_audit()`
borrow reusable output until the next run. `run_snapshot()` executes directly into
independent read-only output under the prepared execution lock; no post-unlock
copy race. Snapshots and their status arrays remain valid after later runs and
scheduler closure. Float64 time-series graph inputs remain C-contiguous; typed
matrix/vector/tensor, int64 and mask inputs support validated strides. All paths permit
read-only data and retain input owners without silently normalizing layout.

A complete identity compact branch may share only immutable `const ExecutionMetadata` and its
physical proof with the main Program after full field, mapping, slot and constant-bit checks.
Validation still runs and worker budgets remain unchanged. This does not share request Scratch,
mutable state or result buffers, and does not alter public graph definitions or serialization.

Private raw factor-prefix execution retains the editable logical chain: zero-initialized
`affine_contract` rotation, normal/Student scale, means addition, then asset projection and
`expm1`. The entire prefix and compact-consumer dependency closure must be proved private.
Published intermediates, extra consumers, isolation, observers or unsupported scopes use
the canonical graph. For T steps, P paths, F transformed factors, A assets and C groups,
`B0=min(C,max(1,T/3))` uses integer division; all C states are admitted only when
`(C-B0)*(3*P+A)<=T*P*F`. Each admitted T×P row then performs one shared rotation and asset
projection across the groups. State, row scratch, Student scale, bounded coefficient packs,
child Scratch and independent output are all included in the native memory budget.

Raw inputs retain their supported C/F layouts and signed strides; binding does not normalize
the complete tape to C. Input owners remain pinned and read-only. Canonical Student scale
passes retain CME domain/error order; this does not certify the platform's IEEE behavior or
business-stage first fault. The planner's empty-array rank fix prevents a scalar size of one
from replacing a known empty array descriptor; it adds no broadcasting or input-type contract.

The v22 audit fields `compact_path_raw_rows`, `compact_path_raw_scale_elements` and
`compact_path_raw_pack_bytes` report actual rotation rows, scale elements and rotation/beta
packing bytes. Healthy admitted intervals count ΣT×P rows, with zero scale elements for
normal and ΣT×P for Student. Packing is already included in `algorithm_copy_bytes`, so it
must not be added again. These counters do not establish performance acceptance. Current
parity evidence covers immutable fixed-tape replay, whose NJIT copy is distinct from the
platform's already-owned writable RNG tape. Owned RNG generation, covariance-root construction,
coverage, stage-first-fault, limits/JSON and complete Excel budget/recalculation acceptance
remain separate integration work; see [the raw-prefix design](mc-axis-primitives.md).

Compatible results retain audit schema `cpp-aot-execution-1`; typed results use
`cpp-aot-execution-2`, `output_dtype="per_output"`, per-root schemas and native capacity/layout
information. New graphs default to `output_order="auto"`; auto/F typed graphs declare
`result_protocol="typed-results-2"`, while explicit C and legacy graphs retain `typed-results-1`.
Each typed output exposes its actual `order`; each interval view has its actual shape and byte
strides. Numeric C/F views retain independent result ownership; positional statuses remain in
logical C order. Consumers must address both by logical coordinates, not zip their physical bytes.
Auto currently uses a bounded set of native layout rules and selects C for isolated requests;
explicit F remains supported. See [layout design and measured scope](physical-layout-design.md#10-图结果布局阶段2026-09-30).
A platform accepting only an earlier schema or result protocol must explicitly adapt and validate
the new protocol before use; engine tests do not certify that integration.
Both schemas record engine version, source/toolchain
build identity, plan fingerprint, canonical operator/IR versions, CPU budget,
dtype, source contracts and result lifetime. It explicitly reports zero Python
operator/worker callbacks, fallback and request-time machine-code compilation.
These are internal provenance records, not an authentication protocol.

Validation: `tests/test_execution_contracts.py`, the full installed-wheel test
suite, native CTest, and `tools/check_phase2_performance.py`. Platform acceptance
is recorded in the sibling platform's `docs/cpp-aot-contracts-acceptance.md`.
