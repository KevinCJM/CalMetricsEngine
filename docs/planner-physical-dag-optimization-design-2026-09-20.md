# Planner Physical-DAG Optimization Design — 2026-09-20

## 1. Scope

This change addresses exactly three findings from the Planner/DAG audit:

1. Planner estimated logical DAG work even when the executor physically fuses reductions/order statistics.
2. Interval-mode chunks were equal-row rather than estimated-work weighted.
3. One/few rows could leave CPUs idle even when the DAG contained independent expensive root branches.

Four-question scope check:

- Current requirement: yes.
- Blocking if unchanged: yes for fully DAG-aware planning.
- Minimal change: compiler metadata + planner policy + native thread runtime only.
- Contract change: no numerical/operator/process/shared-memory API semantics change.

Out of scope: matrix-growing DAGs, process×thread hybridism, arbitrary node-per-task scheduling, and process transport redesign.

## 2. Target flow

Restricted AST → Logical DAG → Lowering/CSE → Borrow-aware liveness → Physical execution metadata → Adaptive Planner → Native Scheduler.

Physical execution metadata contains fusion-aware cost plus dependency-closed independent branch sub-programs.

## 3. Fusion-aware physical cost

CompiledGraph now stores PhysicalCost with:

- constant_per_row
- linear_per_observation
- sort_nlogn

Planner evaluates:

work = constant_per_row × rows + linear_per_observation × sum(T_i) + sort_nlogn × sum(T_i log2(max(T_i,2))).

Cost generation uses the same summary/order fusion eligibility as the executor. Compatible summary consumers over one source are charged as one physical scan; mean_absolute_deviation adds its required second pass. Median/quantile consumers sharing one source are charged as one sort.

Plan preserves both estimated_logical_work_units and estimated_work_units (physical) for auditability.

## 4. Independent branch construction

Roots are unioned into one branch if they share any executable operation ancestor. Read-only input/parameter/constant nodes alone do not connect roots.

Roots are also unioned when executor fusion requires them to remain together:

- summary-fusion consumers sharing one source;
- median/quantile consumers sharing one source.

For every resulting root component Compiler builds one immutable dependency-closed branch Program. Parent ids are remapped once at compile time. Numeric/mask arena slots are compacted per branch.

This guarantees shared executable prefixes and fusion groups are never duplicated across branch tasks.

## 5. DAG branch scheduling

Node-per-task scheduling is deliberately rejected.

DAG branch fork/join is considered only when:

- process isolation is not selected;
- cpu_budget > 1;
- row count is smaller than CPU budget;
- at least two independent branches exist;
- at least two branches are non-trivial;
- total physical work exceeds dag_branch_work_units (default 5,000,000).

When selected:

- lane = thread
- parallel_dimension = dag_branch
- thread_count = min(cpu_budget, rows × branch_count)

Planner creates (row, branch) tasks and sorts them by estimated physical work descending. Runtime executes precompiled branch programs and writes only the corresponding root columns.

If Memory Budget reduces thread_count below branch_task_count, Runtime executes tasks in bounded waves. Therefore actual branch concurrency never exceeds the CPU tokens owned by that request.

## 6. Weighted interval/product partition

Geometry retains every interval length. For one row:

row_weight = PhysicalCost(T_i).

Product mode sums row weights per product. Interval mode uses each row weight directly. Contiguous cuts are selected near equal cumulative physical work while preserving order and at least one unit per remaining worker.

This replaces equal-row interval partitioning and raw-observation-only product weighting.

## 7. Planner priority

1. Hard Stop / process-isolation policy.
2. Heavy independent DAG branches when row-level parallelism cannot fill CPU.
3. Product/interval thread parallelism.
4. Single native execution.

Process SharedMemory policy is unchanged.

## 8. SharedMemory invariants

For process SharedMemory mode, parent creates at most one shared region per ordinary input and copies the input once. Every worker receives the same descriptor and attaches the same region read-only. Pre-shared input reuses existing descriptors and removes that repeated input copy. Output is one shared region split into disjoint row ranges and can be returned with output_copy_bytes=0.

Small process inputs may still use inline IPC by policy because SharedMemory setup can cost more than copying small buffers.

## 9. Validation requirements

- fusion-heavy graph physical cost must be below logical cost;
- shared-prefix/fusion roots must remain in one branch;
- branch execution must equal serial full-DAG output exactly;
- heavy single-row graph must select dag_branch;
- branch concurrency must obey CPU and Memory Budget;
- interval weighted partition must isolate a very long interval from many short intervals;
- existing product grouping must remain intact;
- process SharedMemory and hard-stop contracts must remain unchanged;
- full Python/native/ASan/x86 regressions must remain green;
- mixed NJIT performance matrix must have no regression.

## 10. Initial measured effect

On the current ARM64 host, a 1-product × 1-interval × 1,000,000-observation × 12-heavy-root graph compiles into four independent physical branches. Branch execution is bitwise-equivalent to serial full-DAG execution and measured about 2.09× faster (approximately 10.2 ms vs 21.4 ms in the audit run).

This number is evidence for the tested shape only and is not a universal performance claim.
