# Private shared coefficient solve pair — isolated candidate

Goal: remove a full N² clone and a duplicate LU factorization when two terminal solve_multi roots consume one private arithmetic coefficient. Existing logical roots and public operator contracts remain unchanged.

Proof: typed float64 rank-two outputs; no scopes/isolation/observer/deferred-error execution; exactly two terminal roots; same private add/subtract/multiply producer, exactly two uses, no public coefficient; producer parents are persistent non-operation leaves and not deferred/fused; only static RHS inputs between solves. No runtime numerical cache. Both independent destinations belong to the current result allocation.

Admission: use the existing guarded Accelerate domain and canonical certificate/residual: N 1024..4096, nonempty combined RHS <=64 columns, dense coefficient, round-to-nearest, finite strict DD and identity pivots. All rejection/failure paths restore the original physical coefficient order from still-live leaves before the original two solves. A later RHS error must not overtake the first solve. Small/portable/OFF/unsupported graphs retain canonical execution.

Lifetime: pack the two RHS matrices in an NK prefix of the existing exclusive Workspace, then invoke the same guarded LU core with offset NK. The prefix keeps original B until residual checks complete; the factor arena becomes residual workspace as before. Pack/core/split movement is counted. Actual workspace is 2NK+5N doubles; current copy-root admission conservatively covers canonical fallback N² and vendor reserve. Do not reduce worst-case admission merely because successful execution borrows the private arena.

Layout: private coefficient keeps its fast dense C/F producer order. Root destinations use the requested C/F result layout; each result is split only after common certificate/residual validation. F canonical staging remains available on fallback. No public dtype/shape/layout or Excel mathematical changes.

Validation planned: both producer layouts/operators, C/F outputs, different/maximum RHS columns, readonly/negative RHS strides, singular/NaN/error ordering/certificate fallback, selected roots, third consumers, transport/pickle/prepared/two generations, native/full/SAN/OFF regressions. Formal end-to-end paired timing and heap/RSS after all concurrent validation stops; CPU1, warmed readonly NJIT, owned preparation/output retention included. Experiment is not adopted until evidence supports production integration.
