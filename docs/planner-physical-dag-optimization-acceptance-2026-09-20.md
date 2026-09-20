# Planner Physical-DAG Optimization Acceptance — 2026-09-20

## Scope

This acceptance covers the three Planner/DAG optimizations implemented after the audit:

1. fusion-aware physical execution cost;
2. physical-work-weighted product/interval partitioning;
3. coarse DAG branch fork/join for one/few-row heavy graphs.

Process transport, SharedMemory ownership, canonical operator semantics and the public Python API were not redesigned.

## Implementation

- Compiler now emits PhysicalCost for the full graph and for independent root components.
- Roots sharing any executable operation ancestor remain in one branch.
- Roots participating in the same summary/order-stat fusion group remain in one branch.
- Every branch is compiled once into a dependency-closed native sub-program.
- Branch numeric/mask arena slots are compacted independently.
- Planner exposes estimated_logical_work_units and estimated_work_units (physical).
- Product and interval chunks are cut using physical row work rather than equal row counts.
- Heavy branch tasks are only enabled when row-level parallelism cannot fill the CPU budget.
- Runtime executes branch tasks in bounded waves so actual concurrency cannot exceed the acquired CPU lease.
- Memory Budget can reduce branch thread_count without dropping branch tasks or breaking correctness.

## Correctness evidence

- Heavy branch execution is bitwise-equivalent to serial full-DAG execution in the tested branch workload.
- Shared-prefix and fusion-dependent roots are not split across physical branches.
- Borrowed-view liveness and existing CSE/arena contracts remain unchanged.
- Multi-process SharedMemory transport remains unchanged.

## Performance evidence

Final 15-scenario mixed matrix: 15/15 Native/NJIT < 1.0.

Representative final ratios:

| Workload | Native/NJIT |
| --- | ---: |
| 1 product × 1 interval × 1 metric × 63 obs | 0.872× |
| 1 product × 16 intervals × 5 metrics | 0.507× |
| 10 products × 2 intervals × 16 metrics | 0.573× |
| 1 product × 1 interval × 16 metrics × 200k obs, dag_branch | 0.407× |
| 500 products × 12 intervals × 16 metrics | 0.621× |
| 1000 products × 12 intervals × 16 metrics | 0.608× |

The dedicated single-row heavy DAG benchmark also measured branch execution at about 2.09× the throughput of forced serial full-DAG execution on the current ARM64 host.

## SharedMemory regression

Four-process audit with a 32 KiB input:

| Mode | boundary_copy_bytes | output_copy_bytes | Output owner |
| --- | ---: | ---: | --- |
| inline small-input IPC | 131,072 | 192 | NumPy owned |
| automatic SharedMemory | 32,896 | 0 | native shared region |
| pre-shared input | 128 | 0 | native shared region |

Automatic/shared modes still use one parent shared input block that all workers attach read-only; the Planner optimization did not alter this contract.

## Regression evidence

- Python full suite: 4034 passed.
- Native ARM64 Release: 5/5 CTest passed.
- ARM64 ASan/UBSan: 5/5 CTest passed.
- x86_64/Rosetta: 5/5 CTest passed.
- Standard NJIT performance gate: PASS.
- Ruff / format / git diff check: PASS.

## Remaining boundaries

- DAG branch concurrency is coarse root-component fork/join, not node-per-task scheduling.
- Matrix-growing interval-DAG nodes remain out of scope.
- Process × thread hybrid numerical execution remains disabled.
- Linux/Windows real runtime CI is still required.

Design: docs/planner-physical-dag-optimization-design-2026-09-20.md
Benchmark matrix: docs/cpp-vs-njit-benchmark-matrix-2026-09-20.md
