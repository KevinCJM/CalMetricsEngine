# V62 public references and applicability

Read online on2026-10-07. Only primary documentation is used for conclusions.

- [Eigen: using BLAS/LAPACK](https://libeigen.gitlab.io/eigen/docs-nightly/TopicUsingBlasLapack.html)
  describes vendor backends for dense products/decompositions. The current
  project already bundles Eigen3.4.0 and calls Accelerate directly. No library
  version upgrade, new dependency, Python callback or global thread pool.
- [Eigen: lazy evaluation and aliasing](https://libeigen.gitlab.io/eigen/docs-nightly/TopicLazyEvaluation.html)
  explains expression fusion, temporary arrays and alias proofs. Candidates
  use coefficient-wise Array Maps for existing scalar pairs and an independent
  root for GEMM, with exact per-position in-place arithmetic only after the
  original source is dead. No matrix-product alias is assumed safe.
- [Netlib DGEMM](https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html)
  specifies alpha*A*B+beta*C and that C need not be initialized when beta=0.
  This is a library input contract; it does not repeal Graph's existing failed
  buffer/padding contract. Initial typed clearing is retained in every version.

Nightly Eigen documentation is method guidance, not evidence that the bundled
3.4 implementation has upgraded. Attempts to open the3.4 documentation mirror
failed; no inaccessible source is cited as verified. Apple documentation's JS
page supplied no substantive text, so it is not used as numerical proof.

Actual adoption must use full requests, matched CPU/output/preparation/owners,
warm NJIT and measured malloc/RSS. No advertised or isolated GEMM throughput
substitutes for those measurements. Scope/unknown-shape Planner qualification,
full regression/SAN/OFF and any adoption remain separate uncompleted gates.
