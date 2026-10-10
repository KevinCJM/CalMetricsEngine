# HMM/EM private workspace loan — isolated design

2026-10-04, based on tested 7366 scatter candidate; production remains 7aba.
Objective: eliminate simultaneous NK-matrix arena + 3NK owned Workspace footprint.
No operator/API/IR/IPC/Excel formula/math-contract change; no new threads or libraries.

## Dependencies, liveness and ownership

Only ordinary strict typed, flat mixed graphs without scopes, observers, unavailable inputs or
prebound borrowed results. EM/HMM matrix scratch at least 16 MiB. Scan future edges and roots at
the fit node; chase canonical view backing to past materialized owners. Copy only live arena
owner ranges to exclusive Workspace, including parents and model projections needed later.
The arena can then loan contiguous NK blocks; a future independent float64 scatter root may
loan one NK region. Future root's own producer must fully initialize it after the fit; graph
order proves it has not been written or observed. ResultSlot and row enforce call/row isolation.
Never borrow caller inputs, past roots, previous call outputs, shared live branch outputs or
Prepared.run output. Canonical execution remains when geometry/ownership proof is insufficient.

Workspace additionally holds current fit output if its original destination is arena-backed,
plus existing small EM arrays and Xi normalization tile. Fit arguments retain exact strides and
are rebound only where their pointers fall in preserved live ranges. Copy back original live
ranges and fit output on normal and exception exit. Values outside arena keep original owners.
No private state survives a call except exclusively owned capacity; shrink obsolete large
Workspace capacity before a loan. Canonical full workspace budget remains a conservative upper
bound during screening; no admission reduction yet. Actual capacities and copies stay audited.

## Single mathematical implementation

An optional internal LatentFitScratch describes emission/alpha/beta and small scratch. Common
execute retains output/order/ISA/data validation; canonical and loan paths invoke one shared
latent body. Rebind pointers only: density, forward/backward, ascending Xi sums, Eigen columns,
M-step, stops, diagnostics, finite/probability checks retain exact evaluation order. Small pointer
segments are disjoint and do not escape the fit. No numerical inputs or fitting work precomputed
outside ordinary execution; no Python production planning or callback.

## Validation and adoption

Compare all 13 old/new HMM outputs byte-for-byte and frozen warmed NJIT floats at existing 1e-9,
discrete fields exactly; readonly inputs and two independent result generations. Errors then
recovery, C/F/strides, multiple rows, thread/process and unsupported scope/prepared paths. Native
scratch differential plus ASan/UBSan before adoption. Pair complete 20k/500k finite realtime
branches at CPU1; continuous requested heap and three fresh RSS processes include full fixture,
graph, binding/planning, allocations/copies, returns and two retained result generations.
Compilation/prewarm excluded, fixture/compile excluded only from timer and included in memory.
Prototype counts and arena geometry do not prove full-request memory success. Do not integrate
root source until fair full-chain gain and correctness/sanitizer evidence establish adoption.

## Geometry observation

Audit numeric_arena_bytes includes int64 as well as double buffers. The exact AOT geometry
probe has 96,002,368 double bytes; actual aggregate is 105,002,464 (9,000,096 int64 bytes).
The earlier Python probe assertion compared double-only with aggregate audit and is not a
physical shape defect. Do not infer loan pointers from declared Typed IR shapes alone.

Excel logical DAG and mathematics remain unchanged. No formula/expansion/budget changes and no
new Microsoft Excel recalculation claim; native physical buffers cannot be Excel cell reuse.

## Ownership admission refinement

Initial focused run had 690 pass / 6 fail: three scalar reference fixtures needed
np.asarray; three workspace checks showed public Prepared.run_snapshot uses the
ordinary graph executor and is not Graph's nested prebound_inputs route. Preserve
that evidence. Native ingress now grants workspace loans explicitly for ordinary
fresh outputs; public Prepared paths remain canonical in-process. The per-thread
graph permission restores on every exit. Workers can loan private inline storage
or a newly created transport result region, which is never a previously returned
public array; failed transport does not finish/copy into Prepared caller storage.
No wire field or version change: worker ownership follows its existing construction.
This distinction must be tested before any integration. Canonical fit mathematics
and invalid-input errors were already equal in the failed fixtures.
