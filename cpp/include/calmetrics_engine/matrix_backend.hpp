#pragma once
#include <cstddef>

namespace calmetrics_engine::ops {
struct Audit;
struct Shape;
// Byte-preserving layout copy for disjoint, already-validated F/C rectangles.
// Returns false on unqualified platforms/geometries; never allocates or retains.
bool try_f_rectangle_copy(const double *source, std::size_t source_stride,
                         double *target, std::size_t target_stride,
                         std::size_t rows, std::size_t columns);
// A shape-only bound shared by geometry validation, Planner and execution.
std::size_t matmul_scratch_doubles(const Shape &, const Shape &);
// Admission estimate for guarded system BLAS plus live partition scratch; allocation is lazy.
std::size_t matmul_accelerate_reserve_doubles(const Shape &, const Shape &,
                                               bool expanded_geometry = true);
std::size_t covariance_accelerate_reserve_doubles(const Shape &);
std::size_t transcendental_accelerate_scratch_doubles(const Shape &);
std::size_t cholesky_scratch_doubles(std::size_t order);
// Graph-only mature LU path, with canonical replay owned by the caller.
std::size_t solve_accelerate_scratch_doubles(std::size_t n, std::size_t k);
std::size_t solve_accelerate_copy_reserve_doubles(std::size_t n, std::size_t k);
// Synchronous source owned by the running graph. Writes a checked dense C
// rectangle; neither context nor target is retained after the call.
struct SolveCoefficientBlocks {
    const void *context;
    void (*copy)(const void *, std::size_t, std::size_t, std::size_t, std::size_t, double *);
};
std::size_t solve_concat_reserve_doubles(std::size_t n, std::size_t k);
struct Value;
// Guard statistics only; all-zero/negative-zero and unusual values retain the
// existing scalar classifier. No input, output or dynamic buffer is modified.
enum class MatmulBoundsStatus { unavailable, scalar_required, rejected, valid };
struct MatmulInputBounds { double maximum = 0.0; bool negative = false; };
MatmulBoundsStatus library_matmul_input_bounds(const Value &, MatmulInputBounds &);
// Conservative finite/zero-separation proof over a borrowed contiguous piece.
// Unavailable libraries or ambiguous values return false without modifying data.
bool library_numeric_zero_separated(const double *, std::size_t, double);
struct Workspace;
struct Prepared;
struct Output;
enum class Isa;
bool try_dense_elementwise(const Prepared &, Output &, Workspace &, Audit &, bool private_arena);
bool try_solve_accelerate_private(const Prepared &, const Prepared &, Output &, Workspace &, Audit &, double *);
// Caller proves exclusive coefficient ownership and two independent terminal roots.
bool try_solve_accelerate_private_pair(const Prepared &, const Prepared &, const Prepared &,
                                      Output &, Output &, Workspace &, Audit &, double *);
// Completes an admitted terminal graph root, including canonical replay.
bool try_solve_accelerate_copy(const Prepared &, Output &, Workspace &, Audit &);
// May change private coefficients on rejection. Caller reconstructs and replays.
bool try_solve_accelerate_concat(const Prepared &, const SolveCoefficientBlocks &,
                                Output &, Workspace &, Audit &, double *);

// Physical statistics of a private matmul; logical reducers remain unchanged.
struct MatmulStatisticsAccuracy {
    bool allowed = false;
    std::size_t maximum_ddof = 0;
    double linear_scale[3]{}; // sum / mean / variance consumer closure
    double sqrt_scale[3]{};
};
struct MatmulQuantileOutput {
    const Value *probabilities = nullptr;
    double *values = nullptr;
    Output *result = nullptr; // this call's independent typed root, if proved
};
std::size_t matmul_quantiles_scratch_doubles(const Shape &, const Shape &);
std::size_t matmul_quantiles(const Prepared &, double *totals, double *squared,
    double *scratch, const MatmulStatisticsAccuracy &, double quantile_scale,
    const MatmulQuantileOutput *, std::size_t outputs, Audit &, void *context,
    void (*checkpoint)(void *));
std::size_t matmul_statistics_scratch_doubles(const Shape &, const Shape &);
std::size_t matmul_statistics(const Prepared &, double *totals, double *squared,
    double *scratch, const MatmulStatisticsAccuracy &, Audit &, void *context,
    void (*checkpoint)(void *));
bool matmul_packed_eligible(const Value &, const Value &);
// Physical F output reuses the same ordered kernel and panel bound.
bool library_matmul_packed(const Value &, const Value &, double *, Workspace &, Audit &,
                           bool f_output = false);
// Synchronous graph-only proof. The graph scans every original input and
// certifies the COMPLETE exclusive arithmetic chain before creating it. Never
// cache across executions. Only the exact Prepared instance can consume it;
// intermediate zero/sign checks are deferred to the graph's terminal replay.
struct MatmulTerminalProof {
    const Prepared *source = nullptr;
    MatmulInputBounds left, right;
};
// Caller proves an independent float root; rejected input never writes output.
// The graph owns the independent destination. Late rejection is replayed fully
// in the current canonical implementation before returning true.
bool try_accelerate_matmul_root(const Prepared &, Output &, Workspace &, Audit &,
                               void *context = nullptr, void (*checkpoint)(void *) = nullptr,
                               bool expanded_geometry = true,
                               const MatmulTerminalProof *terminal_proof = nullptr);
// Same-call graph-only bounded binary32 candidate. The caller certifies every
// terminal scalar branch. Packs exclusively into unpublished independent output
// bytes, expands backwards to double, and never retains numerical state.
bool try_accelerate_matmul_terminal_float(const Prepared &, Output &, Workspace &, Audit &,
    void *, void (*)(void *), bool, const MatmulTerminalProof *);
// Complete an eligible independent root, including canonical replay. A paired
// correlation output requires both covariance and normalized error bounds.
bool try_accelerate_covariance(const Prepared &, Output &, Workspace &, Audit &,
                              bool correlation_consumer = false);
// Graph-only terminal F root. Shares canonical Cholesky arithmetic and keeps
// the mirrored C upper triangle as the final F lower triangle.
void cholesky_f_root(const Prepared &, Output &, Workspace &, Isa, Audit &);
// Direct calls precheck all values before any write. A graph caller may admit
// block-local scalar replay only for an independent, unobserved terminal root.
bool try_accelerate_transcendental(const Prepared &, Output &, Workspace &, Audit &,
                                  bool independent_root = false);
// Completes a triangular solve after the canonical caller copies/validates RHS.
void triangular_solve_finish(const Prepared &, Output &, Workspace &, Isa, Audit &);
// Writes only the upper triangle. The last output row owns temporary means
// until its diagonal is written; the caller mirrors only after normalization.
bool library_covariance_upper(const Value &, double *, Workspace &, Audit &);
// Borrows inputs and writes only the independent result.
bool library_matvec(const Value &, const Value &, double *, Audit &);
// Graph-only ordered product/max/divide/subtract/max closure. Four independent
// rows share existing Eigen packets; no allocation, cache or retained pointer.
bool library_scan_reduce_quad(const Value &, std::size_t row, double initial,
    double peak_initial, double subtraction, double *totals, unsigned &errors,
    void *context, void (*checkpoint)(void *));
// Graph-only canonical left folds across four independent rows. Operands are
// already prepared/validated; rejection writes nothing, and no buffer is owned.
bool library_where_sum_quad(const Prepared &, std::size_t row, double *totals,
    void *context, void (*checkpoint)(void *));
bool library_square_sum_quad(const Prepared &, const Value &mask, double otherwise,
    std::size_t row, double *totals, void *context, void (*checkpoint)(void *));
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
// Internal C/F-compatible finite covariance pieces. Caller admits geometry.
void covariance_library_means(const Value &, double *);
// Complete centered F panel, N >= 32 and even D; consumes workspace before release.
void covariance_library_centered(const Value &, Output &, Workspace &, Audit &);
double covariance_library_pair(const Value &, std::size_t, std::size_t, const double *);
// Admitted C/F-compatible matmul; void entries permit caller tail dispatch.
void library_matmul_f(const Value &, const Value &, double *, Audit &);
void library_matmul_f_root(const Value &, const Value &, double *, Audit &);
void library_matmul_c(const Value &, const Value &, double *, Audit &);
void library_matmul_c_blocked(const Value &, const Value &, double *, Audit &);
void library_matmul_f_blocked(const Value &, const Value &, double *, Audit &);
void library_matmul_repeated_rows(const Value &, const Value &, double *, Audit &);
#endif
// A library call runs on the Scheduler's admitted thread. No process-global
// thread setting, numerical cache or separately owned result is introduced.
class MatrixRotation {
public:
    explicit MatrixRotation(bool eligible) noexcept : active_(eligible) {}
    MatrixRotation(const MatrixRotation &) = delete;
    MatrixRotation &operator=(const MatrixRotation &) = delete;
    void apply(double *x, double *y, std::size_t count, double c, double s,
               Audit &audit) const;
private:
    [[maybe_unused]] bool active_;
};
} // namespace calmetrics_engine::ops
