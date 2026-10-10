#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>
#include <cfenv>
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
#include <Accelerate/Accelerate.h>
#endif
#include <numeric>
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#endif

#include "column_library.hpp"

namespace calmetrics_engine::ops {
namespace {
using O = Op;
std::size_t plus(std::size_t a, std::size_t b) {
  require(a <= std::size_t(PTRDIFF_MAX) / 8 &&
              b <= std::size_t(PTRDIFF_MAX) / 8 - a,
          "SHAPE_OVERFLOW");
  return a + b;
}
std::size_t times(std::size_t a, std::size_t b) {
  require(!b || a <= std::size_t(PTRDIFF_MAX) / 8 / b, "SHAPE_OVERFLOW");
  return a * b;
}
void arg(const Prepared &p, std::size_t i, int lo, int hi,
         Kind kind = Kind::number) {
  require(p.args[i].kind == kind, "DTYPE_MISMATCH");
  if (p.geometry(i))
    require(p.args[i].shape.rank >= lo && p.args[i].shape.rank <= hi,
            "RANK_MISMATCH");
}
bool option(Prepared &p, std::size_t i, double lo, double hi,
            bool integer = true) {
  arg(p, i, 0, 0);
  if (!p.payload(i))
    return false;
  auto x = p.args[i].scalar;
  bool ok = std::isfinite(x) && x >= lo && x <= hi &&
            (!integer || x == std::floor(x));
  if (!ok && p.structure_only) {
    p.payload_available_mask &= ~(1u << i);
    return false;
  }
  require(ok, "INVALID_PARAMETER");
  return true;
}
void same(const Prepared &p, std::size_t i, std::size_t j) {
  if (p.geometry(i) && p.geometry(j))
    require(p.args[i].shape == p.args[j].shape, "SHAPE_MISMATCH");
}
void finite(double x) { require(std::isfinite(x), "NON_FINITE_RESULT"); }
void finite_array(const Value &v) {
  if (v.contiguous()) {
    const auto *data = static_cast<const double *>(v.data);
    bool valid = true;
    for (std::size_t i = 0; i < v.size(); ++i)
      valid &= std::isfinite(data[i]);
    require(valid, "NON_FINITE_RESULT");
  } else if (v.shape.rank == 2) {
    for (std::size_t i = 0; i < v.shape.dim[0]; ++i)
      for (std::size_t j = 0; j < v.shape.dim[1]; ++j)
        finite(v.at(i, j));
  } else
    for (std::size_t i = 0; i < v.size(); ++i)
      finite(v.f(i));
}
void probabilities(const Value &v, bool rows = false) {
  finite_array(v);
  const auto groups = rows ? v.shape.dim[0] : 1,
             width = rows ? v.shape.dim[1] : v.size();
  for (std::size_t i = 0; i < groups; ++i) {
    double sum = 0;
    for (std::size_t j = 0; j < width; ++j) {
      auto x = v.f(i * width + j);
      require(x >= 0, "DOMAIN_ERROR");
      sum += x;
    }
    finite(sum);
    require(sum > 0, "DOMAIN_ERROR");
  }
}
void validate_density_inputs(const Value &x, const Value &means,
                             const Value &vars, bool fit) {
  require(means.shape.dim[0] > 0 && x.shape.dim[1] > 0 &&
              (!fit || x.shape.dim[0] > 0), "INSUFFICIENT_SAMPLE");
  finite_array(x);
  finite_array(means);
  finite_array(vars);
  for (std::size_t i = 0; i < vars.size(); ++i)
    require(vars.f(i) >= 0, "DOMAIN_ERROR");
}
void density_variance_logs(const Value &vars, double *logs) {
  for (std::size_t c = 0; c < vars.shape.dim[0]; ++c) {
    double sum = 0;
    for (std::size_t j = 0; j < vars.shape.dim[1]; ++j)
      sum += std::log(std::max(vars.at(c, j), 1e-12));
    logs[c] = sum;
  }
}
bool strictly_ordered(const Value &v) {
  if (v.size() < 2) return true;
  const auto *codes = static_cast<const std::int64_t *>(v.data);
  const auto step = v.stride[0];
  bool increasing = true, decreasing = true;
  for (std::size_t j = 1; j < v.size(); ++j) {
    const auto a = codes[std::ptrdiff_t(j - 1) * step];
    const auto b = codes[std::ptrdiff_t(j) * step];
    increasing &= a < b;
    decreasing &= a > b;
    if (!increasing && !decreasing) return false;
  }
  return true;
}
bool indices(const Prepared &p, std::size_t i, std::size_t limit, bool bounded,
             bool unique, bool missing = false,
             std::size_t *scratch = nullptr) {
  if (!p.payload(i))
    return false;
  const auto &v = p.args[i];
  const auto *codes = static_cast<const std::int64_t *>(v.data);
  const auto step = v.stride[0];
  bool valid = true;
  const auto count = v.size();
  for (std::size_t j = 0; j < count; ++j) {
    const auto x = codes[std::ptrdiff_t(j) * step];
    valid &= x >= (missing ? -1 : 0);
    if (bounded && x >= 0) valid &= std::uint64_t(x) < limit;
  }
  require(valid, "INDEX_OUT_OF_BOUNDS");
  if (unique && v.size() > 1 && !strictly_ordered(v)) {
    // Independent payload validation must also run after another input fails.
    // This bounded validation copy is included in the Planner workspace bound.
    std::vector<std::size_t> owned;
    if (!scratch) {
      owned.resize(v.size());
      scratch = owned.data();
    }
    for (std::size_t j = 0; j < v.size(); ++j)
      scratch[j] = std::size_t(codes[std::ptrdiff_t(j) * step]);
    std::sort(scratch, scratch + v.size());
    for (std::size_t j = 1; j < v.size(); ++j)
      require(scratch[j] != scratch[j - 1], "DUPLICATE_INDEX");
    return true;
  }
  return false;
}
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
// Keep the vector exponential's constants out of the generic axis-loop frame.
__attribute__((noinline))
void eigen_softmax_row(const double *source, double *output,
                       std::size_t count, double maximum) {
  using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
  const auto width = static_cast<Eigen::Index>(count);
  Eigen::Map<Row, Eigen::Unaligned> target(output, width);
  target = (Eigen::Map<const Row, Eigen::Unaligned>(source, width) - maximum).exp();
  double sum = 0;
  // Ascending reduction and true division preserve the operator's arithmetic
  // structure; the caller-owned result is the only numerical temporary.
  for (std::size_t c = 0; c < count; ++c) sum += target[c];
  target /= sum;
}
// This driver owns only eligible contiguous axis rows. Keep the generic axis
// template unchanged so unselected layouts do not pay for library selection.
__attribute__((noinline))
void eigen_softmax_rows(const Prepared &p, Output &out, Audit &audit) {
  const auto &input = p.args[0];
  const auto count = input.shape.dim[std::size_t(p.args[1].scalar)];
  const auto minimum_shift = std::log(double(count)) - 700.0;
  const auto *data = static_cast<const double *>(input.data);
  auto *output = static_cast<double *>(out.data);
  for (std::size_t first = 0; first < input.size(); first += count) {
    const auto *row = data + input.offset(first);
    double maximum = -std::numeric_limits<double>::infinity();
    double minimum = std::numeric_limits<double>::infinity();
    bool valid = true;
    for (std::size_t c = 0; c < count; ++c) {
      const auto value = row[c];
      valid &= std::isfinite(value);
      if (value > maximum) maximum = value;
      if (value < minimum) minimum = value;
    }
    require(valid, "NON_FINITE_RESULT");
    // Eigen clamps very negative exp arguments. This count-adjusted domain
    // also keeps the normalized probability above the subnormal boundary.
    if (minimum - maximum >= minimum_shift) {
      eigen_softmax_row(row, output + first, count, maximum);
      audit.isa = "eigen-array";
    } else {
      double sum = 0;
      for (std::size_t c = 0; c < count; ++c) sum += std::exp(row[c] - maximum);
      for (std::size_t c = 0; c < count; ++c)
        output[first + c] = std::exp(row[c] - maximum) / sum;
    }
  }
}
#endif
template<O op>
void axis_reduce(const Prepared &p, Output &out) {
  const auto &a = p.args;
  const auto axis = std::size_t(a[1].scalar), count = a[0].shape.dim[axis];
  std::size_t outer = 1, inner = 1;
    for (std::size_t j = 0; j < axis; ++j)
      outer *= a[0].shape.dim[j];
    for (int j = int(axis) + 1; j < a[0].shape.rank; ++j)
      inner *= a[0].shape.dim[j];
    for (std::size_t i = 0; i < outer; ++i)
      for (std::size_t j = 0; j < inner; ++j) {
        auto index = [&](std::size_t c) { return (i * count + c) * inner + j; };
        const auto *source =
            static_cast<const double *>(a[0].data) + a[0].offset(index(0));
        const auto step = a[0].stride[axis];
        auto at = [&](std::size_t c) {
          return source[std::ptrdiff_t(c) * step];
        };
        double sum = 0, m = -std::numeric_limits<double>::infinity();
        std::int64_t winner = 0;
        bool valid = true;
        for (std::size_t c = 0; c < count; ++c) {
          auto x = at(c);
          valid &= std::isfinite(x);
          if constexpr (op == O::variance_axis) sum += x;
          else if (x > m) {
            m = x;
            if constexpr (op == O::argmax_axis) winner = std::int64_t(c);
          }
        }
        require(valid, "NON_FINITE_RESULT");
        if constexpr (op == O::argmax_axis) {
          out.set_integer(i * inner + j, winner);
          continue;
        }
        if constexpr (op == O::max_axis) {
          out.set(i * inner + j, m);
          continue;
        }
        if constexpr (op == O::variance_axis) {
          double mean = sum / double(count), s = 0;
          for (std::size_t c = 0; c < count; ++c) {
            auto delta = at(c) - mean;
            s += delta * delta;
          }
          sum = s / (double(count) - a[2].scalar);
        } else {
          sum = 0;
          for (std::size_t c = 0; c < count; ++c)
            sum += std::exp(at(c) - m);
          if constexpr (op == O::softmax_axis) {
            for (std::size_t c = 0; c < count; ++c)
              out.set(index(c), std::exp(at(c) - m) / sum);
            continue;
          }
          sum = m + std::log(std::max(sum, 1e-300));
        }
        finite(sum);
        out.set(i * inner + j, sum);
      }
}
bool is_hmm_projection(O op) {
  return op >= O::hmm_initial && op <= O::hmm_variances;
}
std::size_t model_dimensions(const Shape &s, bool hmm) {
  const auto overhead = plus(hmm ? s.dim[0] : 0, 4);
  require(s.dim[0] > 0 && s.dim[1] > overhead && (s.dim[1] - overhead) % 2 == 0,
          "SHAPE_MISMATCH");
  return (s.dim[1] - overhead) / 2;
}
double lse(const double *x, std::size_t k) {
  double m = x[0];
  for (std::size_t i = 1; i < k; ++i)
    m = std::max(m, x[i]);
  double s = 0;
  for (std::size_t i = 0; i < k; ++i)
    // exp(0) is exactly one; preserve the original summation order.
    s += x[i] == m ? 1.0 : std::exp(x[i] - m);
  const auto value = m + std::log(std::max(s, 1e-300));
  finite(value);
  return value;
}
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
template<int States>
__attribute__((noinline))
void eigen_density_state_groups(const Value &x, const Value &means,
    const Value &vars, double *out, const double *logs, double constant) {
    using Components = Eigen::Array<double, States, 1>;
    using Column = Eigen::Map<const Components, Eigen::Unaligned, Eigen::InnerStride<>>;
    const auto n=x.shape.dim[0], d=x.shape.dim[1], k=means.shape.dim[0];
    const auto *center=static_cast<const double *>(means.data);
    const auto *variance=static_cast<const double *>(vars.data);
    for (std::size_t i=0; i<n; ++i)
        for (std::size_t first=0; first<k; first+=States) {
            Components q=Components::Zero();
            for (std::size_t j=0; j<d; ++j) {
                const Column m(center+std::ptrdiff_t(first)*means.stride[0]+
                    std::ptrdiff_t(j)*means.stride[1], Eigen::InnerStride<>(means.stride[0]));
                const Column v(variance+std::ptrdiff_t(first)*vars.stride[0]+
                    std::ptrdiff_t(j)*vars.stride[1], Eigen::InnerStride<>(vars.stride[0]));
                const Components delta=x.at(i,j)-m;
                // Independent component lanes; each keeps its feature reduction order.
                q+=(delta*delta)/v.max(1e-12);
            }
            const Eigen::Map<const Components, Eigen::Unaligned> log_values(logs+first);
            const Components result=-0.5*((constant+log_values)+q);
            for (int lane=0; lane<States; ++lane) {
                out[i*k+first+std::size_t(lane)]=result[lane];
                finite(result[lane]);
            }
        }
}

constexpr std::size_t eigen_density_column_state_capacity = 16;

__attribute__((noinline))
std::size_t eigen_density_columns(const Value &x, const Value &means,
    const Value &vars, double *out, const double *logs, double constant,
    std::size_t output_stride = 0) {
    using Rows=Eigen::Array<double,8,1>;
    const auto n=x.shape.dim[0],d=x.shape.dim[1],k=means.shape.dim[0];
    if (!output_stride) output_stride = k;
    const auto *input=static_cast<const double *>(x.data);
    double tile[8 * eigen_density_column_state_capacity];
    std::size_t first=0;
    for(;n-first>=8;first+=8) {
        for(std::size_t c=0;c<k;++c) {
            Rows q=Rows::Zero();
            for(std::size_t j=0;j<d;++j) {
                const Eigen::Map<const Rows> values(input+first+j*x.stride[1]);
                const auto delta=values-means.at(c,j);
                q+=(delta*delta)/std::max(vars.at(c,j),1e-12);
            }
            q=-0.5*((constant+logs[c])+q);
            for(std::size_t row=0;row<8;++row) tile[row*k+c]=q[row];
        }
        // Preserve the original public output prefix on a numerical exception.
        for(std::size_t row=0;row<8;++row) for(std::size_t c=0;c<k;++c) {
            out[(first+row)*output_stride+c]=tile[row*k+c];
            finite(out[(first+row)*output_stride+c]);
        }
    }
    return first;
}
#endif

void density_rows(const Value &x, const Value &means, const Value &vars,
                  double *out, const double *logs, std::size_t selection_n,
                  Isa requested, Audit &audit, std::size_t output_stride = 0,
                  std::size_t layout_rows = 0) {
  const auto n = x.shape.dim[0], d = x.shape.dim[1], k = means.shape.dim[0];
  if (!output_stride) output_stride = k;
  if (!layout_rows) layout_rows = n;
  const double constant =
      double(d) * std::log(2.0 * 3.14159265358979323846264338327950288);
  std::size_t first=0;
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
  if (__builtin_available(macOS 15.0, *)) {
    if (requested!=Isa::scalar && selection_n>=4096 && d>=32 && k>0 && k%4==0 &&
        x.stride[1]==1 && output_stride==k) {
      if (k%8==0) eigen_density_state_groups<8>(x,means,vars,out,logs,constant);
      else eigen_density_state_groups<4>(x,means,vars,out,logs,constant);
      ++audit.eigen_density_calls;
      audit.isa="eigen-array-state-density";
      return;
    }
    if (requested!=Isa::scalar && selection_n>=32 && d>=8 && k>0 &&
        k<=eigen_density_column_state_capacity &&
        x.stride[0]==1 && x.stride[1]>=static_cast<std::ptrdiff_t>(layout_rows)) {
      first=eigen_density_columns(x,means,vars,out,logs,constant,output_stride);
      ++audit.eigen_density_calls;
      audit.isa="eigen-array-column-density";
    }
  }
#else
  (void)requested; (void)audit; (void)selection_n;
#endif
  for (std::size_t i = first; i < n; ++i)
    for (std::size_t c = 0; c < k; ++c) {
      double q = 0;
      for (std::size_t j = 0; j < d; ++j) {
        auto delta = x.at(i, j) - means.at(c, j);
        q += delta * delta / std::max(vars.at(c, j), 1e-12);
      }
      out[i * output_stride + c] = -0.5 * (constant + logs[c] + q);
      finite(out[i * output_stride + c]);
    }
}
void density(const Value &x, const Value &means, const Value &vars, double *out,
             double *logs, Isa requested, Audit &audit) {
  density_variance_logs(vars, logs);
  density_rows(x, means, vars, out, logs, x.shape.dim[0], requested, audit);
}

#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
// xi is dead before transition accumulation. Batch only independent states;
// keep each log-sum-exp's source-state reduction in its original order.
template<bool Forward>
bool library_hmm_row(const double *previous, const double *emission,
    const double *log_a, std::size_t k, double *row, double *temp,
    double *xi, Audit &audit) {
  constexpr auto bound = std::numeric_limits<double>::max() / 32.0;
  bool valid = true;
  for (std::size_t group = 0; group < k; ++group)
    for (std::size_t lane = 0; lane < k; ++lane) {
      const auto value = Forward ? previous[lane] + log_a[lane*k+group] :
          log_a[group*k+lane] + emission[lane] + previous[lane];
      xi[group*k+lane] = value;
      valid &= std::abs(value) <= bound;
    }
  if (!valid) return false;
  for (std::size_t group = 0; group < k; ++group) {
    double maximum = xi[group*k];
    for (std::size_t lane = 1; lane < k; ++lane)
      maximum = std::max(maximum, xi[group*k+lane]);
    row[group] = maximum;
    for (std::size_t lane = 0; lane < k; ++lane)
      xi[group*k+lane] -= maximum;
  }
  const int cells = int(k*k);
  vvexp(xi, xi, &cells);
  for (std::size_t group = 0; group < k; ++group) {
    double sum = 0.0;
    for (std::size_t lane = 0; lane < k; ++lane)
      sum += xi[group*k+lane];
    temp[group] = std::max(sum, 1e-300);
  }
  // Retain the original ordered sum and scalar logarithm.
  // The expensive K*K exponentials remain a single native library batch.
  for (std::size_t group = 0; group < k; ++group)
    temp[group] = std::log(temp[group]);
  ++audit.accelerate_vforce_calls;
  for (std::size_t group = 0; group < k; ++group) {
    const auto value = row[group] + temp[group];
    finite(value);
    row[group] = Forward ? emission[group] + value : value;
    if constexpr (Forward) finite(row[group]);
  }
  return true;
}
// Factor the same log recurrence into positive matrix-vector products.
// A bounded positive transition matrix prevents a zero or overflowing dot.
// Only fixed-iteration checkpoint sweeps pass this cache; full/strict sweeps
// retain the existing ordered expressions.
template<bool Forward>
bool library_factored_hmm_row(const double *previous, const double *emission,
    const double *positive_a, std::size_t k, double *row, double *temp,
    double *row_work, Audit &audit) {
  constexpr double bound=1e8;
  bool valid=true;
  double maximum=-std::numeric_limits<double>::infinity();
  for (std::size_t lane=0;lane<k;++lane) {
    valid &= std::abs(previous[lane])<=bound && std::abs(emission[lane])<=bound;
    const auto value=Forward ? previous[lane] : emission[lane]+previous[lane];
    row_work[lane]=value;
    maximum=std::max(maximum,value);
  }
  if (!valid) return false;
  for (std::size_t lane=0;lane<k;++lane) row_work[lane]-=maximum;
  const int length=int(k);
  vvexp(row_work,row_work,&length);
  cblas_dgemv(CblasRowMajor,Forward ? CblasTrans : CblasNoTrans,
      length,length,1.0,positive_a,length,row_work,1,0.0,temp,1);
  ++audit.accelerate_vforce_calls;
  ++audit.accelerate_matmul_calls;
  for (std::size_t lane=0;lane<k;++lane) {
    const auto value=maximum+std::log(temp[lane]);
    row[lane]=Forward ? emission[lane]+value : value;
    finite(row[lane]);
  }
  return true;
}
#endif

void forward_row(const double *previous, const double *emission,
                 const double *log_a, std::size_t k, double *row, double *temp,
                 bool state_library, double *row_work, Audit &audit,
                 const double *positive_a=nullptr) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
  if (positive_a && library_factored_hmm_row<true>(previous, emission,
      positive_a, k, row, temp, row_work, audit)) return;
  if (state_library && library_hmm_row<true>(previous, emission, log_a, k,
      row, temp, row_work, audit)) return;
#else
  (void)state_library; (void)row_work; (void)audit; (void)positive_a;
#endif
  for (std::size_t t = 0; t < k; ++t) {
    for (std::size_t c = 0; c < k; ++c)
      temp[c] = previous[c] + log_a[c * k + t];
    row[t] = emission[t] + lse(temp, k);
    finite(row[t]);
  }
}
void forward(const double *em, std::size_t n, std::size_t k,
             const double *initial, const double *log_a, double *alpha,
             double *temp, bool state_library, double *xi, Audit &audit) {
  for (std::size_t c = 0; c < k; ++c)
    alpha[c] = std::log(std::max(initial[c], 1e-12)) + em[c];
  for (std::size_t i = 1; i < n; ++i)
    forward_row(alpha+(i-1)*k, em+i*k, log_a, k, alpha+i*k, temp,
                state_library, xi, audit);
}
// The same backward expression writes either an independent history or rows
// whose emission was already consumed by every source state. The initial and
// effective-count K buffers are dead until gamma/M and serve as rolling beta.
template<bool InPlace, bool CheckBound = false>
bool backward(const double *em, std::size_t n, std::size_t k,
              const double *log_a, double *beta, double *temp, Audit &audit,
              double *next = nullptr, double *current = nullptr,
              bool state_library = false, double *xi = nullptr,
              const double *positive_a = nullptr) {
  if constexpr (!InPlace) (void)audit;
  if constexpr (InPlace) std::fill_n(next, k, 0.0);
  else std::fill(beta + (n - 1) * k, beta + n * k, 0.0);
  bool bounded = true;
  for (std::size_t q = 1; q < n; ++q) {
    const auto i = n - 1 - q;
    const auto *previous = InPlace ? next : beta + (i + 1) * k;
    auto *row = InPlace ? current : beta + i * k;
    bool batched = false;
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
    if (positive_a) batched = library_factored_hmm_row<false>(previous, em+(i+1)*k,
        positive_a, k, row, temp, xi, audit);
    if (!batched && state_library) batched = library_hmm_row<false>(previous, em+(i+1)*k,
        log_a, k, row, temp, xi, audit);
#else
    (void)state_library; (void)xi; (void)positive_a;
#endif
    if (!batched)
      for (std::size_t s = 0; s < k; ++s) {
        for (std::size_t t = 0; t < k; ++t)
          temp[t] = log_a[s * k + t] + em[(i + 1) * k + t] + previous[t];
        row[s] = lse(temp, k);
      }
    if constexpr (CheckBound)
      for (std::size_t s = 0; s < k; ++s)
        bounded &= std::abs(row[s]) <= std::numeric_limits<double>::max() / 32.0;
    if constexpr (InPlace) {
      std::copy_n(next, k, beta + (i + 1) * k);
      audit.algorithm_copy_bytes += k * sizeof(double);
      std::swap(next, current);
    }
  }
  if constexpr (InPlace) {
    std::copy_n(next, k, beta);
    audit.algorithm_copy_bytes += k * sizeof(double);
  }
  return bounded;
}

// Both full history and a private last-row projection use this ordered recurrence.
template<class Emission>
void filter_row_values(const Value *a, std::size_t i, std::size_t k,
                       const double *previous, double *current,
                       const Emission &emission) {
  double m = emission(0);
  for (std::size_t c = 1; c < k; ++c)
    m = std::max(m, emission(c));
  double total = 0;
  for (std::size_t t = 0; t < k; ++t) {
    double predicted = 0;
    if (!i && a[3].scalar == 0)
      predicted = a[1].f(t);
    else
      for (std::size_t s = 0; s < k; ++s)
        predicted += (i ? previous[s] : a[1].f(s)) * a[2].at(s, t);
    current[t] = predicted * std::exp(emission(t) - m);
    total += current[t];
  }
  finite(total);
  for (std::size_t t = 0; t < k; ++t)
    current[t] /= std::max(total, 1e-300);
}

void filter_row(const Value *a, std::size_t i, std::size_t k,
                const double *previous, double *current) {
  filter_row_values(a, i, k, previous, current,
                    [&](std::size_t c) { return a[0].at(i, c); });
}

// Row normalizers are independent. Interleave those libm chains while keeping
// every row's source/target order and every xi cell's ascending-time reduction.
// Block=1 is the same mathematical body for small and explicitly scalar fits.
void ordered_hmm_transition_tile(const double *alpha, const double *beta_next,
    const double *em_next, const double *log_a, std::size_t count, std::size_t k,
    double *xi, double *normalizers) {
  std::fill(normalizers, normalizers + count,
            -std::numeric_limits<double>::infinity());
  for (std::size_t s = 0; s < k; ++s)
    for (std::size_t t = 0; t < k; ++t)
      for (std::size_t row = 0; row < count; ++row) {
        const auto v = alpha[row * k + s] + log_a[s * k + t] +
                       em_next[row * k + t] + beta_next[row * k + t];
        auto &z = normalizers[row];
        if (z == -std::numeric_limits<double>::infinity()) z = v;
        else if (v > z) z = v + std::log1p(std::exp(z - v));
        else z += std::log1p(std::exp(v - z));
      }
  for (std::size_t s = 0; s < k; ++s)
    for (std::size_t t = 0; t < k; ++t)
      for (std::size_t row = 0; row < count; ++row)
        xi[s * k + t] += std::exp(alpha[row * k + s] + log_a[s * k + t] +
            em_next[row * k + t] + beta_next[row * k + t] - normalizers[row]);
}

template<std::size_t Block>
void accumulate_hmm_transitions(const double *alpha, const double *beta,
    const double *em, const double *log_a, std::size_t n, std::size_t k,
    double *xi, double *normalizers) {
  std::fill(xi, xi + k * k, 0.0);
  for (std::size_t first = 0; first + 1 < n; first += Block)
    ordered_hmm_transition_tile(alpha + first * k, beta + (first + 1) * k,
        em + (first + 1) * k, log_a, std::min(Block, n - 1 - first), k, xi, normalizers);
}


#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
// Before replay destroys emission, scan alpha/em/logA and track beta bounds
// during backward. With an independent beta history, scan all three after the
// sweeps instead. Reject the whole library route before any transition writes.
template<bool CheckAlpha = true>
bool bounded_hmm_library_values(const double *alpha, const double *beta,
    const double *em,const double *log_a,std::size_t n,std::size_t k) {
  constexpr auto limit=std::numeric_limits<double>::max()/32.0;
  bool valid=true;
  for(std::size_t i=0;i<n*k;++i)
    valid &= (!CheckAlpha || std::abs(alpha[i])<=limit) && std::abs(em[i])<=limit &&
             (!beta || std::abs(beta[i])<=limit);
  for(std::size_t i=0;i<k*k;++i)valid &= std::abs(log_a[i])<=limit;
  return valid;
}

// Batch independent times only. Preserve the original online s,t normalizer
// and every xi cell's ascending time sum. Remember each conditional maximum,
// instead of reading/recomputing the transition expression after library calls.
// All three 128-double arrays are included in the planned small scratch tail.
void library_hmm_transition_tile(const double *alpha, const double *beta_next,
    const double *em_next, const double *log_a, std::size_t count, std::size_t k,
    double *xi, double *normalizers, double *input, double *output,
    double *base, Audit &audit) {
    const int length=int(count);
    for(std::size_t s=0;s<k;++s) for(std::size_t t=0;t<k;++t) {
      if(s==0 && t==0) {
        for(std::size_t row=0;row<count;++row) {
          normalizers[row]=alpha[row*k]+log_a[0]+em_next[row*k]+beta_next[row*k];
        }
        continue;
      }
      for(std::size_t row=0;row<count;++row) {
        const auto v=alpha[row*k+s]+log_a[s*k+t]+em_next[row*k+t]+beta_next[row*k+t];
        const auto z=normalizers[row];
        base[row]=v>z ? v : z;
        input[row]=v>z ? z-v : v-z;
      }
      vvexp(output,input,&length);
      vvlog1p(input,output,&length);
      audit.accelerate_vforce_calls+=2;
      for(std::size_t row=0;row<count;++row) normalizers[row]=base[row]+input[row];
    }
    for(std::size_t s=0;s<k;++s) for(std::size_t t=0;t<k;++t) {
      for(std::size_t row=0;row<count;++row) {
        input[row]=alpha[row*k+s]+log_a[s*k+t]+em_next[row*k+t]+beta_next[row*k+t]-normalizers[row];
      }
      vvexp(output,input,&length);
      ++audit.accelerate_vforce_calls;
      for(std::size_t row=0;row<count;++row) xi[s*k+t]+=output[row];
    }
}
#endif

// Save the same full forward values at block boundaries, never just sampled
// validation. The first complete density still precedes every recurrence.
bool checkpoint_forward(const double *em, std::size_t n, std::size_t k,
    const double *initial, const double *log_a, double *checkpoint, double *last,
    double *rolling, double *temp, bool state_library, double *row_work, Audit &audit,
    const double *positive_a) {
  constexpr auto block = latent_hmm_alpha_checkpoint_block;
  constexpr auto bound = std::numeric_limits<double>::max()/32.0;
  auto *previous = rolling, *current = rolling+k;
  bool bounded = true;
  for (std::size_t c=0; c<k; ++c) {
    previous[c] = std::log(std::max(initial[c],1e-12)) + em[c];
    bounded &= std::abs(previous[c])<=bound;
  }
  std::copy_n(previous,k,checkpoint);
  for (std::size_t i=1; i<n; ++i) {
    forward_row(previous,em+i*k,log_a,k,current,temp,state_library,row_work,audit,positive_a);
    for (std::size_t c=0;c<k;++c) bounded &= std::abs(current[c])<=bound;
    std::swap(previous,current);
    if (i%block==0) std::copy_n(previous,k,checkpoint+(i/block)*k);
  }
  std::copy_n(previous,k,last);
  audit.algorithm_copy_bytes += (((n-1)/block+1)*k+k)*sizeof(double);
  return bounded;
}

double checkpoint_hmm_step(const Value &x, const Value &means, const Value &vars,
    double *history, double *small, Isa requested, bool state_library, Audit &audit) {
  constexpr auto block = latent_hmm_alpha_checkpoint_block;
  const auto n=x.shape.dim[0],k=means.shape.dim[0];
  auto *log_a=small, *xi=log_a+k*k, *temp=xi+k*k, *effective=temp+k,
       *initial=effective+k, *logs=initial+k, *normalizers=logs+k;
  auto *checkpoint=normalizers+512, *last=checkpoint+((n-1)/block+1)*k,
       *alpha=last+k, *emission=alpha+block*k, *row_work=emission+(block+8)*k;
  const double *positive_a=nullptr;
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
  if (state_library) {
    const auto lower=std::log(1e-12);
    bool admitted=true;
    for (std::size_t cell=0;cell<k*k;++cell)
      admitted &= log_a[cell]>=lower && log_a[cell]<=0.0;
    if (admitted) {
      auto *cache=row_work+k*k;
      for (std::size_t cell=0;cell<k*k;++cell) cache[cell]=std::exp(log_a[cell]);
      positive_a=cache;
    }
  }
#endif
  const bool alpha_bound = checkpoint_forward(history,n,k,initial,log_a,
      checkpoint,last,alpha,temp,state_library,row_work,audit,positive_a);
  bool library = false;
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
  library = requested!=Isa::scalar && n>=4096 && k>=3 &&
      std::fegetround()==FE_TONEAREST && alpha_bound &&
      bounded_hmm_library_values<false>(nullptr,nullptr,history,log_a,n,k);
#else
  (void)alpha_bound;
#endif
  if (library) library=backward<true,true>(history,n,k,log_a,history,temp,audit,
                                         initial,effective,state_library,row_work,positive_a);
  else backward<true>(history,n,k,log_a,history,temp,audit,initial,effective,
                      state_library,row_work,positive_a);
  // Keep likelihood validation after the complete backward pass.
  const double ll=lse(last,k);
  std::fill_n(xi,k*k,0.0);
  for (std::size_t first=0;first<n;first+=block) {
    const auto count=std::min(block,n-first), transitions=std::min(count,n-1-first);
    auto input=x;
    input.data=static_cast<const double *>(x.data)+std::ptrdiff_t(first)*x.stride[0];
    // Include the next original eight-row group so the boundary emission does
    // not become a scalar tail when it was vectorized in the full density.
    input.shape.dim[0]=std::min(block+8,n-first);
    density_rows(input,means,vars,emission,logs,n,requested,audit);
    std::copy_n(checkpoint+(first/block)*k,k,alpha);
    audit.algorithm_copy_bytes += k*sizeof(double);
    for (std::size_t row=1;row<count;++row)
      forward_row(alpha+(row-1)*k,emission+row*k,log_a,k,alpha+row*k,temp,
                  state_library,row_work,audit,positive_a);
    if (transitions) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
      if (library)
        library_hmm_transition_tile(alpha,history+(first+1)*k,emission+k,log_a,
            transitions,k,xi,normalizers,normalizers+128,normalizers+256,
            normalizers+384,audit);
      else
#endif
        ordered_hmm_transition_tile(alpha,history+(first+1)*k,emission+k,
            log_a,transitions,k,xi,normalizers);
    }
    // Every beta row written here has had its last ascending xi consumer.
    for (std::size_t row=0;row<count;++row) {
      auto *destination=history+(first+row)*k;
      for (std::size_t c=0;c<k;++c) temp[c]=alpha[row*k+c]+destination[c];
      const auto z=lse(temp,k);
      for (std::size_t c=0;c<k;++c) destination[c]=std::exp(temp[c]-z);
    }
  }
  return ll;
}

// M-step features are independent, while each feature's observation reduction
// must retain its original order. Use the model output as the accumulator;
// no old mean/variance is consumed after the complete E-step has finished.
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
template<int Features, bool Variance>
void eigen_weighted_feature_sums(const Value &x, const double *alpha, std::size_t k,
                                std::size_t component, const double *means,
                                double *sums, bool reset = true) {
  using FeatureArray = Eigen::Array<double, Features, 1>;
  const auto size = static_cast<Eigen::Index>(x.shape.dim[1]);
  Eigen::Map<FeatureArray> totals(sums, size);
  if (reset) totals.setZero();
  const auto *source = static_cast<const double *>(x.data);
  for (std::size_t i = 0; i < x.shape.dim[0]; ++i) {
    const Eigen::Map<const FeatureArray> row(
        source + static_cast<std::ptrdiff_t>(i) * x.stride[0], size);
    const double weight = alpha[i * k + component];
    if constexpr (Variance) {
      const Eigen::Map<const FeatureArray> center(means, size);
      const auto delta = row - center;
      // Keep (alpha * delta) * delta, including when alpha is zero.
      totals += (weight * delta) * delta;
    } else {
      totals += weight * row;
    }
  }
}
#endif

#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
// Isolate wide-map code generation from the common latent operator frame.
template<bool Variance>
__attribute__((noinline))
void eigen_wide_feature_sums(const Value &x, const double *alpha, std::size_t k,
                             std::size_t component, const double *means,
                             double *sums, bool reset = true) {
  switch (x.shape.dim[1]) {
#define WIDE_EM_FEATURES(count) case count: \
    eigen_weighted_feature_sums<count, Variance>(x, alpha, k, component, means, sums, reset); return;
    WIDE_EM_FEATURES(32)
    WIDE_EM_FEATURES(64)
    WIDE_EM_FEATURES(128)
    WIDE_EM_FEATURES(256)
#undef WIDE_EM_FEATURES
  }
}
#endif

#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
// Eight independent feature accumulators keep observation order. Strided Maps
// read the original column layout directly; there is no packed input or heap.
template<bool Variance>
__attribute__((noinline))
void eigen_column_feature_sums(const Value &x, const double *alpha, std::size_t k,
                              std::size_t component, const double *means,
                              double *sums, bool reset = true) {
  using Features = Eigen::Array<double, 8, 1>;
  const auto *data = static_cast<const double *>(x.data);
  const auto n = x.shape.dim[0], d = x.shape.dim[1];
  for (std::size_t first = 0; first < d; first += 8) {
    Eigen::Map<Features> total(sums + first);
    if (reset) total.setZero();
    for (std::size_t i = 0; i < n; ++i) {
      const Eigen::Map<const Features, Eigen::Unaligned, Eigen::InnerStride<>> value(
          data + i + first * x.stride[1], 8, Eigen::InnerStride<>(x.stride[1]));
      const double weight = alpha[i * k + component];
      if constexpr (Variance) {
        const Eigen::Map<const Features> center(means + first);
        const auto delta = value - center;
        total += (weight * delta) * delta;
      } else {
        total += weight * value;
      }
    }
  }
}
#endif

template<bool Variance>
void weighted_feature_sums(const Value &x, const double *alpha, std::size_t k,
                           std::size_t component, const double *means,
                           double *sums, Isa requested, Audit &audit, bool reset = true,
                           std::size_t selection_n = 0) {
  const auto n = x.shape.dim[0], d = x.shape.dim[1];
  if (!selection_n) selection_n = n;
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
  if (__builtin_available(macOS 15.0, *)) {
    if (requested != Isa::scalar && selection_n >= 32 && d >= 8 && d % 8 == 0 &&
        x.stride[0] == 1 && x.stride[1] >= static_cast<std::ptrdiff_t>(n)) {
      eigen_column_feature_sums<Variance>(x, alpha, k, component, means, sums, reset);
      audit.isa = "eigen-array-column-em";
      return;
    }
    if (requested != Isa::scalar && selection_n >= 32 && x.stride[1] == 1) {
      // Fixed-size Map exposes packet/tail sizes once per component, instead
      // of doing dynamic packet alignment/tail dispatch for every observation.
      // These are non-owning maps, never fixed-size numerical stack arrays.
      switch (d) {
#define EM_FEATURES(count) case count: \
        eigen_weighted_feature_sums<count, Variance>(x, alpha, k, component, means, sums, reset); \
        audit.isa = "eigen-array"; return;
        EM_FEATURES(4)
        EM_FEATURES(5)
        EM_FEATURES(8)
        EM_FEATURES(9)
        EM_FEATURES(16)
      case 32: case 64: case 128: case 256:
        eigen_wide_feature_sums<Variance>(x, alpha, k, component, means, sums, reset);
        audit.isa = "eigen-array"; return;
#undef EM_FEATURES
      default:
        // Dynamic feature maps regressed measured whole-model latency.
        // Unmeasured widths use the shared scalar reduction below.
        break;
      }
    }
  }
#else
  (void)requested;
  (void)audit;
#endif
  // General signed/zero feature strides, small shapes and portable/OFF builds
  // retain the same per-feature scalar reduction with no extra storage.
  for (std::size_t j = 0; j < d; ++j) {
    double sum = reset ? 0 : sums[j];
    for (std::size_t i = 0; i < n; ++i) {
      if constexpr (Variance) {
        const double delta = x.at(i, j) - means[j];
        sum += alpha[i * k + component] * delta * delta;
      } else {
        sum += alpha[i * k + component] * x.at(i, j);
      }
    }
    sums[j] = sum;
  }
}

// Measured Eigen CPU route: keep small/component-light/wide fits on the
// canonical full posterior. Replay carries exact sample-order centered sums.
std::size_t gmm_replay_scratch(std::size_t n, std::size_t k, std::size_t d) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
  if (__builtin_available(macOS 15.0, *)) {
    if (k < 16 || d < 8 || d > 32) return 0;
  } else {
    return 0;
  }
#else
  return 0;
#endif
  const auto cells = times(n, k);
  const auto row_storage = plus(128, plus(times(2, d), 3));
  const auto storage = plus(times(row_storage, k), times(128, d));
  if (cells < latent_gmm_replay_min_elements || storage >= cells) return 0;
  return storage;
}

// Non-owning existing Eigen row-density kernel, at most 128D doubles.
// The full NK array is internal, so component tiles never publish model state.
void gmm_tile_density(const Value &block, const Value &means, const Value &vars,
                      double *tile, const double *logs, std::size_t selection_n,
                      double *input_tile, Isa requested, Audit &audit) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
  if (__builtin_available(macOS 15.0, *)) {
    if (requested != Isa::scalar && block.shape.dim[1] >= 8) {
      auto columns = block;
      const auto n = block.shape.dim[0], d = block.shape.dim[1], k = means.shape.dim[0];
      if (block.stride[0] != 1 || block.stride[1] < std::ptrdiff_t(n)) {
        for (std::size_t j = 0; j < d; ++j)
          for (std::size_t i = 0; i < n; ++i) input_tile[j * 128 + i] = block.at(i, j);
        columns.data = input_tile;
        columns.stride = {1, 128, 1};
        audit.algorithm_copy_bytes += n * d * sizeof(double);
      }
      for (std::size_t first = 0; first < k; first += 8) {
        auto center = means, variance = vars;
        const auto states = std::min(std::size_t{8}, k - first);
        center.shape.dim[0] = variance.shape.dim[0] = states;
        center.data = static_cast<const double *>(means.data) + std::ptrdiff_t(first) * means.stride[0];
        variance.data = static_cast<const double *>(vars.data) + std::ptrdiff_t(first) * vars.stride[0];
        // Each component group writes its own columns in the same C tile.
        // No density_tile -> posterior_tile copy or second tile allocation.
        density_rows(columns, center, variance, tile + first, logs + first,
                     selection_n, requested, audit, k);
      }
      return;
    }
  }
#endif
  density_rows(block, means, vars, tile, logs, selection_n, requested, audit);
}

double gmm_replay_step(const Value &x, const Value &means, const Value &vars,
                       double *y, std::size_t width, double *scratch,
                       Isa requested, Audit &audit) {
  const auto n = x.shape.dim[0], d = x.shape.dim[1], k = means.shape.dim[0];
  const auto features = times(k, d);
  auto *tile = scratch;
  auto *old_means = tile + 128 * k;
  auto *sums = old_means + features;
  auto *counts = sums + features;
  auto *variance_logs = counts + k;
  auto *weight_logs = variance_logs + k;
  auto *input_tile = weight_logs + k;
  for (std::size_t c = 0; c < k; ++c) {
    counts[c] = 0;
    weight_logs[c] = std::log(std::max(y[c * width], 1e-12));
    double log_sum = 0;
    for (std::size_t j = 0; j < d; ++j) {
      old_means[c * d + j] = means.at(c, j);
      log_sum += std::log(std::max(vars.at(c, j), 1e-12));
    }
    variance_logs[c] = log_sum;
  }
  audit.algorithm_copy_bytes += features * sizeof(double);
  auto saved_means = means;
  saved_means.data = old_means;
  saved_means.stride = {std::ptrdiff_t(d), 1, 1};
  std::fill_n(sums, features, 0.0);
  double likelihood = 0;
  for (std::size_t first = 0; first < n; first += 128) {
    auto block = x;
    block.data = static_cast<const double *>(x.data) + std::ptrdiff_t(first) * x.stride[0];
    block.shape.dim[0] = std::min(std::size_t{128}, n - first);
    gmm_tile_density(block, saved_means, vars, tile, variance_logs, n, input_tile,
                     requested, audit);
    for (std::size_t i = 0; i < block.shape.dim[0]; ++i) {
      auto *row = tile + i * k;
      for (std::size_t c = 0; c < k; ++c) row[c] += weight_logs[c];
      const auto z = lse(row, k);
      likelihood += z;
      for (std::size_t c = 0; c < k; ++c) {
        row[c] = std::exp(row[c] - z);
        counts[c] += row[c];
      }
    }
    for (std::size_t c = 0; c < k; ++c)
      weighted_feature_sums<false>(block, tile, k, c, nullptr, sums + c * d,
                                   requested, audit, false, n);
  }
  finite(likelihood);
  for (std::size_t c = 0; c < k; ++c) {
    counts[c] = std::max(counts[c], 1e-8);
    y[c * width] = counts[c] / double(n);
    auto *center = y + c * width + 1;
    std::copy_n(sums + c * d, d, center);
    audit.algorithm_copy_bytes += d * sizeof(double);
    for (std::size_t j = 0; j < d; ++j) {
      center[j] /= counts[c];
      finite(center[j]);
    }
  }
  std::fill_n(sums, features, 0.0);
  // Replay the same OLD E-step parameters. Variances are published only after
  // every tile has consumed them; the new means are used only by centered M.
  for (std::size_t first = 0; first < n; first += 128) {
    auto block = x;
    block.data = static_cast<const double *>(x.data) + std::ptrdiff_t(first) * x.stride[0];
    block.shape.dim[0] = std::min(std::size_t{128}, n - first);
    gmm_tile_density(block, saved_means, vars, tile, variance_logs, n, input_tile,
                     requested, audit);
    for (std::size_t i = 0; i < block.shape.dim[0]; ++i) {
      auto *row = tile + i * k;
      for (std::size_t c = 0; c < k; ++c) row[c] += weight_logs[c];
      const auto z = lse(row, k);
      for (std::size_t c = 0; c < k; ++c) row[c] = std::exp(row[c] - z);
    }
    for (std::size_t c = 0; c < k; ++c)
      weighted_feature_sums<true>(block, tile, k, c, y + c * width + 1,
                                  sums + c * d, requested, audit, false, n);
  }
  for (std::size_t c = 0; c < k; ++c) {
    auto *variance = y + c * width + 1 + d;
    std::copy_n(sums + c * d, d, variance);
    audit.algorithm_copy_bytes += d * sizeof(double);
    for (std::size_t j = 0; j < d; ++j) {
      variance[j] = std::max(variance[j] / counts[c], 1e-5);
      finite(variance[j]);
    }
  }
  return likelihood;
}

// Isolate the column scratch from the common latent dispatcher. Both paths use
// the same three eight-double buffers and retain the original failure prefix.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void variance_columns(const Value &x, double ddof, Output &out,
                      Isa requested, Audit &audit) {
  const auto count = x.shape.dim[0], width = x.shape.dim[1];
  const auto *data = static_cast<const double *>(x.data);
  bool use_eigen = false;
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
  if (__builtin_available(macOS 15.0, *))
    use_eigen = requested != Isa::scalar && count >= 32 && width >= 8;
#else
  (void)requested;
#endif
  for (std::size_t first = 0; first < width; first += 8) {
    const auto lanes = std::min<std::size_t>(8, width - first);
    double sums[8]{}, means[8]{}, squared[8]{};
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (use_eigen && lanes == 8) {
      using Block = Eigen::Array<double, 8, 1>;
      Eigen::Map<Block, Eigen::Unaligned> total(sums), mean(means), square(squared);
      bool valid = true;
      for (std::size_t i = 0; i < count; ++i) {
        const Eigen::Map<const Block, Eigen::Unaligned> row(data + i * width + first);
        valid &= row.isFinite().all();
        total += row;
      }
      require(valid, "NON_FINITE_RESULT");
      mean = total / double(count);
      for (std::size_t i = 0; i < count; ++i) {
        const Eigen::Map<const Block, Eigen::Unaligned> row(data + i * width + first);
        const auto delta = row - mean;
        square += delta * delta;
      }
    } else
#endif
    {
      bool valid = true;
      for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < lanes; ++j) {
          const auto value = data[i * width + first + j];
          valid &= std::isfinite(value);
          sums[j] += value;
        }
      require(valid, "NON_FINITE_RESULT");
      for (std::size_t j = 0; j < lanes; ++j) means[j] = sums[j] / double(count);
      for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < lanes; ++j) {
          const auto delta = data[i * width + first + j] - means[j];
          squared[j] += delta * delta;
        }
    }
    for (std::size_t j = 0; j < lanes; ++j) {
      const auto result = squared[j] / (double(count) - ddof);
      finite(result);
      out.set(first + j, result);
    }
  }
  if (use_eigen) audit.isa = "eigen-array";
}
} // namespace

void hmm_filter_tail(const Prepared &p, Output &out, Workspace &work, Audit &audit) {
  require(p.spec->op == O::hmm_filter && !p.borrowed, "FILTER_TAIL_CONTRACT");
  const auto *a = p.args.data();
  const auto n = a[0].shape.dim[0], k = a[1].size();
  require(n > 0 && k > 0 && out.kind == Kind::number && out.data &&
              out.order == MemoryOrder::c && out.shape == matrix_shape(1, k),
          "FILTER_TAIL_CONTRACT");
  // Same validation order as full-history inference, before any recurrence write.
  finite_array(a[0]);
  probabilities(a[1]);
  probabilities(a[2], true);
  work.ensure(times(2, k), 0);
  audit = {};
  audit.scratch_bytes = times(2, k) * sizeof(double);
  auto *previous = work.doubles.data(), *current = previous + k;
  for (std::size_t i = 0; i < n; ++i) {
    filter_row(a, i, k, previous, current);
    std::swap(previous, current);
  }
  std::copy_n(previous, k, static_cast<double *>(out.data));
  audit.algorithm_copy_bytes = k * sizeof(double);
}

// Physical Gaussian producer: retain its full numerical validation/error
// position while discarding private rows. Its logical descriptor has no payload.
void gaussian_density_stream_validate(const Prepared &p, Workspace &work,
                                      Isa requested, Audit &audit) {
  require(p.spec->op == O::gaussian_log_density && !p.borrowed && p.count == 3,
          "GAUSSIAN_STREAM_CONTRACT");
  const auto &a = p.args;
  const auto n = a[0].shape.dim[0], d = a[0].shape.dim[1], k = a[1].shape.dim[0];
  require(n > 0 && d <= latent_gaussian_stream_max_features &&
              k <= latent_gaussian_stream_max_components &&
              p.output_shape == matrix_shape(n, k), "GAUSSIAN_STREAM_CONTRACT");
  validate_density_inputs(a[0], a[1], a[2], false);
  work.ensure(times(129, k), 0);
  audit = {};
  audit.scratch_bytes = times(129, k) * sizeof(double);
  auto *tile = work.doubles.data(), *logs = tile + 128 * k;
  density_variance_logs(a[2], logs);
  for (std::size_t first = 0; first < n; first += 128) {
    auto block = a[0];
    block.data = static_cast<const double *>(a[0].data) +
        std::ptrdiff_t(first) * a[0].stride[0];
    block.shape.dim[0] = std::min(std::size_t{128}, n - first);
    density_rows(block, a[1], a[2], tile, logs, n, requested, audit, 0, n);
  }
}
// Its Gaussian producer already checked every row. Recompute bounded tiles
// with the original parameters and feed the same ordered filter recurrence.
void gaussian_filter_tail(const Prepared &gaussian, const Prepared &filter,
                          Output &out, Workspace &work, Isa requested, Audit &audit) {
  const auto &g = gaussian.args;
  const auto *a = filter.args.data();
  const auto n = g[0].shape.dim[0], k = g[1].shape.dim[0];
  require(gaussian.spec->op == O::gaussian_log_density &&
      filter.spec->op == O::hmm_filter && !filter.borrowed && n > 0 &&
      g[0].shape.dim[1] <= latent_gaussian_stream_max_features && k > 0 &&
      k <= latent_gaussian_stream_max_components &&
      filter.output_shape == gaussian.output_shape &&
      out.kind == Kind::number && out.data && out.order == MemoryOrder::c &&
      out.shape == matrix_shape(1, k), "GAUSSIAN_STREAM_CONTRACT");
  probabilities(a[1]);
  probabilities(a[2], true);
  work.ensure(times(131, k), 0);
  audit = {};
  audit.scratch_bytes = times(131, k) * sizeof(double);
  auto *tile = work.doubles.data(), *logs = tile + 128 * k;
  auto *previous = logs + k, *current = previous + k;
  density_variance_logs(g[2], logs);
  for (std::size_t first = 0; first < n; first += 128) {
    auto block = g[0];
    block.data = static_cast<const double *>(g[0].data) +
        std::ptrdiff_t(first) * g[0].stride[0];
    block.shape.dim[0] = std::min(std::size_t{128}, n - first);
    density_rows(block, g[1], g[2], tile, logs, n, requested, audit, 0, n);
    for (std::size_t row = 0; row < block.shape.dim[0]; ++row) {
      filter_row_values(a, first + row, k, previous, current,
                        [&](std::size_t c) { return tile[row * k + c]; });
      std::swap(previous, current);
    }
  }
  // The tail may reuse dead Gaussian input storage only after all reads finish.
  std::copy_n(previous, k, static_cast<double *>(out.data));
  audit.algorithm_copy_bytes = k * sizeof(double);
}

std::size_t latent_hmm_full_scratch(std::size_t n,std::size_t k) {
  const auto matrices=latent_fit_matrix_count(O::gaussian_hmm_em,n,k);
  const auto tail=matrices==2 ? plus(512,times(128,k)) :
      n>=128 && k>=3 ? std::min(n-1,std::size_t{128}) : 0;
  return plus(times(matrices,times(n,k)),
      plus(plus(times(2,times(k,k)),times(4,k)),tail));
}
std::size_t latent_hmm_alpha_checkpoint_tail(std::size_t n,std::size_t k) {
  require(n>0 && k>0 && k<=128,"FIT_SCRATCH_CONTRACT");
  constexpr auto block=latent_hmm_alpha_checkpoint_block;
  return plus(plus(times(4,times(k,k)),times(4,k)),
      plus(512,plus(times((n-1)/block+1,k),times(2*block+9,k))));
}
double hmm_alpha_checkpoint_step(const Value &x,const Value &means,const Value &vars,
    double *history,double *small,std::size_t small_count,Isa requested,
    bool state_library,Audit &audit) {
  const auto n=x.shape.dim[0],k=means.shape.dim[0];
  require(history && small && small_count>=latent_hmm_alpha_checkpoint_tail(n,k),
          "FIT_SCRATCH_CONTRACT");
  return checkpoint_hmm_step(x,means,vars,history,small,requested,state_library,audit);
}

std::size_t latent_scratch(Op op, const Value *a) {
  const auto n = a[0].shape.dim[0];
  if (op == O::gaussian_mixture_em) {
    auto k = a[1].size();
    const auto replay = gmm_replay_scratch(n, k, a[0].shape.dim[1]);
    return replay ? replay : plus(times(n, k), times(3, k));
  }
  if (op == O::gaussian_hmm_em) {
    const auto k=a[1].size();
    if (a[6].scalar==0.0 && latent_hmm_alpha_checkpoint_eligible(n,k,a[0].shape.dim[1]))
      return plus(times(n,k),latent_hmm_alpha_checkpoint_tail(n,k));
    return latent_hmm_full_scratch(n,k);
  }
  if (op == O::gaussian_log_density)
    return a[1].shape.dim[0];
  if (op == O::hmm_smooth) {
    auto k = a[1].size();
    return plus(times(k, k), times(4, k));
  }

  return 0;
}

void prepare_latent(Prepared &p) {
  auto &a = p.args;
  const auto op = p.spec->op;
  if (op >= O::variance_axis && op <= O::softmax_axis) {
    arg(p, 0, 1, 3);
    const bool axis = option(p, 1, 0, p.geometry(0) ? a[0].shape.rank - 1 : 2);
    if (op == O::variance_axis) {
      if (p.count < 3)
        a[2] = Value::number(0);
      option(p, 2, 0, 9007199254740991.0);
    }
    p.output_kind = op == O::argmax_axis ? Kind::integer : Kind::number;
    p.output_geometry_known = p.geometry(0) && axis;
    if (p.output_geometry_known) {
      p.output_shape = a[0].shape;
      if (op != O::softmax_axis) {
        auto ax = std::size_t(a[1].scalar);
        --p.output_shape.rank;
        for (int j = int(ax); j < p.output_shape.rank; ++j)
          p.output_shape.dim[j] = a[0].shape.dim[j + 1];
        for (int j = p.output_shape.rank; j < 3; ++j)
          p.output_shape.dim[j] = 1;
      }
    }
    return;
  }
  if (op >= O::mixture_weights && op <= O::hmm_variances) {
    arg(p, 0, 2, 2);
    const bool diag = op >= O::em_iterations && op <= O::em_converged;
    p.output_kind = op == O::em_iterations  ? Kind::integer
                    : op == O::em_converged ? Kind::mask
                                            : Kind::number;
    p.output_geometry_known = diag || p.geometry(0);
    if (!p.geometry(0))
      return;
    const auto k = a[0].shape.dim[0], w = a[0].shape.dim[1];
    if (diag) {
      require(k > 0 && w >= 6, "SHAPE_MISMATCH");
      return;
    }
    const bool hmm = is_hmm_projection(op);
    const auto d = model_dimensions(a[0].shape, hmm);
    std::size_t offset = 0;
    if (op == O::mixture_weights || op == O::hmm_initial)
      p.output_shape = vector_shape(k);
    else if (op == O::hmm_transition) {
      offset = 1;
      p.output_shape = matrix_shape(k, k);
    } else {
      offset = 1 + (hmm ? k : 0) +
               ((op == O::mixture_variances || op == O::hmm_variances) ? d : 0);
      p.output_shape = matrix_shape(k, d);
    }
    if (p.payload(0)) {
      p.borrowed = true;
      p.view = a[0];
      p.view.shape = p.output_shape;
      p.view.data = static_cast<const double *>(a[0].data) +
                    std::ptrdiff_t(offset) * a[0].stride[1];
    }
    return;
  }
  if (op == O::transition_counts || op == O::one_hot) {
    arg(p, 0, 1, 1, Kind::integer);
    const bool k = option(p, 1, 1, 9007199254740991.0);
    if (p.structure_only)
      indices(p, 0, k ? std::size_t(a[1].scalar) : 0, k, false, true);
    p.output_geometry_known =
        k && (op == O::transition_counts || p.geometry(0));
    if (p.output_geometry_known) {
      auto count = std::size_t(a[1].scalar);
      p.output_shape =
          matrix_shape(op == O::one_hot ? a[0].size() : count, count);
    }
    return;
  }
  if (op == O::sample_unique_indices) {
    arg(p, 0, 1, 3);
    const bool count = option(p, 1, 0, 9007199254740991.0);
    if (p.count < 3)
      a[2] = Value::number(0);
    option(p, 2, -9007199254740991.0, 9007199254740991.0);
    if (count && p.geometry(0))
      require(a[1].scalar <= double(a[0].shape.dim[0]), "SHAPE_MISMATCH");
    p.output_geometry_known = count;
    p.output_kind = Kind::integer;
    if (count)
      p.output_shape = vector_shape(std::size_t(a[1].scalar));
    return;
  }
  if (op == O::partition_mean || op == O::scatter_rows) {
    arg(p, 0, op == O::partition_mean ? 2 : 1, op == O::partition_mean ? 2 : 3);
    arg(p, 1, 1, 1, Kind::integer);
    const bool size =
        option(p, 2, op == O::partition_mean ? 1 : 0, 9007199254740991.0);
    if (op == O::scatter_rows) {
      if (p.count < 4)
        a[3] = Value::number(std::numeric_limits<double>::quiet_NaN());
      arg(p, 3, 0, 0);
    }
    if (p.geometry(0) && p.geometry(1))
      require(a[0].shape.dim[0] == a[1].size(), "SHAPE_MISMATCH");
    const bool bounded = op == O::partition_mean ? p.geometry(0) : size;
    const auto limit = op == O::partition_mean
                           ? a[0].shape.dim[0]
                           : (size ? std::size_t(a[2].scalar) : 0);
    if (p.structure_only)
      indices(p, 1, limit, bounded, true);
    if (p.geometry(1) && (!p.payload(1) || !strictly_ordered(a[1])))
      p.scratch_indices = a[1].size();
    if (op == O::partition_mean && p.geometry(0) && size)
      require(a[2].scalar <= double(a[0].shape.dim[0]), "SHAPE_MISMATCH");
    p.output_geometry_known = p.geometry(0) && size;
    if (p.output_geometry_known) {
      p.output_shape = a[0].shape;
      p.output_shape.dim[0] = std::size_t(a[2].scalar);
    }
    return;
  }
  const bool mixture = op == O::gaussian_mixture_em,
             hmm = op == O::gaussian_hmm_em;
  const bool inference = op == O::hmm_filter || op == O::hmm_smooth;
  arg(p, 0, 2, 2);
  std::size_t mean_id = op == O::gaussian_log_density ? 1 : hmm ? 3 : 2;
  if (inference || mixture || hmm)
    arg(p, 1, 1, 1);
  if (inference || hmm)
    arg(p, 2, 2, 2);
  if (!inference) {
    arg(p, mean_id, 2, 2);
    arg(p, mean_id + 1, 2, 2);
    same(p, mean_id, mean_id + 1);
  }
  if (op == O::hmm_filter) {
    if (p.count < 4)
      a[3] = Value::number(0);
    option(p, 3, 0, 1);
  }
  if (mixture || hmm) {
    auto i = hmm ? 5u : 4u;
    if (p.count <= i)
      a[i] = Value::number(100);
    if (p.count <= i + 1)
      a[i + 1] = Value::number(hmm ? 1e-5 : 1e-6);
    option(p, i, 1, 10000);
    option(p, i + 1, 0, 1e100, false);
  }
  if (inference || hmm) {
    if (p.geometry(2))
      require(a[2].shape.dim[0] == a[2].shape.dim[1], "SHAPE_MISMATCH");
    if (p.geometry(1) && p.geometry(2))
      require(a[1].size() == a[2].shape.dim[0], "SHAPE_MISMATCH");
  }
  if (inference) {
    if (p.geometry(0) && p.geometry(1))
      require(a[0].shape.dim[1] == a[1].size(), "SHAPE_MISMATCH");
    if (p.geometry(0) && p.geometry(2))
      require(a[0].shape.dim[1] == a[2].shape.dim[0], "SHAPE_MISMATCH");
  } else {
    for (auto id : {mean_id, mean_id + 1}) {
      if (p.geometry(0) && p.geometry(id))
        require(a[0].shape.dim[1] == a[id].shape.dim[1], "SHAPE_MISMATCH");
      if ((mixture || hmm) && p.geometry(1) && p.geometry(id))
        require(a[1].size() == a[id].shape.dim[0], "SHAPE_MISMATCH");
      if (hmm && p.geometry(2) && p.geometry(id))
        require(a[2].shape.dim[0] == a[id].shape.dim[0], "SHAPE_MISMATCH");
    }
  }
  p.output_geometry_known = inference ? p.geometry(0)
                            : op == O::gaussian_log_density
                                ? (p.geometry(0) && p.geometry(1))
                                : p.geometry(mean_id);
  if (p.output_geometry_known) {
    if (inference)
      p.output_shape = a[0].shape;
    else if (op == O::gaussian_log_density)
      p.output_shape = matrix_shape(a[0].shape.dim[0], a[1].shape.dim[0]);
    else {
      const auto k = a[mean_id].shape.dim[0], d = a[mean_id].shape.dim[1];
      p.output_shape = matrix_shape(k, plus(plus(times(2, d), hmm ? k : 0), 4));
    }
  }
  bool known = true;
  for (std::size_t i = 0; i < (inference ? 3 : hmm ? 5 : mixture ? 4 : 3); ++i)
    known = known && p.geometry(i);
  if (known)
    p.scratch_doubles = latent_scratch(op, a.data());
}

void validate_scatter_indices(const Prepared &p, Workspace &work, Audit &audit) {
  require(p.spec->op==O::scatter_rows,"INVALID_OPERATOR");
  work.ensure(0,p.scratch_indices);
  const bool copied=indices(p,1,std::size_t(p.args[2].scalar),true,true,false,work.indices.data());
  audit.algorithm_copy_bytes=copied ? p.args[1].size()*sizeof(std::size_t) : 0;
  audit.scratch_bytes=p.scratch_indices*sizeof(std::size_t);
}

void scatter_argmax_pair(const Prepared &scatter, const Prepared &reduce, Output &out,
                         const Prepared *take) {
  require(scatter.spec->op==O::scatter_rows && reduce.spec->op==O::argmax_axis &&
      scatter.output_shape.rank==2 && reduce.args[1].scalar==1,"INVALID_OPERATOR");
  if (!out.shape.size()) return;
  const auto width=scatter.output_shape.dim[1];
  require(width>0,"INSUFFICIENT_SAMPLE");
  // -1 is internal coverage only. It cannot be a returned argmax index.
  for (std::size_t row=0;row<out.shape.size();++row) out.set_integer(row,-1);
  for (std::size_t row=0;row<scatter.args[1].size();++row) {
    double maximum=-std::numeric_limits<double>::infinity();
    std::int64_t winner=0;
    bool valid=true;
    for (std::size_t col=0;col<width;++col) {
      const auto x=take ? take->args[0].at(row,static_cast<std::size_t>(take->args[1].i(col)))
                        : scatter.args[0].at(row,col);
      valid &= std::isfinite(x);
      if (x>maximum) {maximum=x;winner=static_cast<std::int64_t>(col);}
    }
    require(valid,"NON_FINITE_RESULT");
    out.set_integer(static_cast<std::size_t>(scatter.args[1].i(row)),winner);
  }
  const bool finite_fill=std::isfinite(scatter.args[3].scalar);
  const auto *result=static_cast<const std::int64_t *>(out.data);
  for (std::size_t row=0;row<out.shape.size();++row)
    if (result[row]<0) {
      require(finite_fill,"NON_FINITE_RESULT");
      out.set_integer(row,0);
    }
}

// Shared canonical fill/copy semantics. A validated private take changes only
// the source coordinate; it does not allocate or reorder floating point math.
static void scatter_values(const Prepared &p, Output &out, Audit &audit,
                           const Prepared *take=nullptr) {
  if (!out.shape.size()) return;
  auto *y=static_cast<double *>(out.data);
  const auto &a=p.args;
  std::fill(y,y+out.shape.size(),a[3].scalar);
  const auto width=a[0].shape.dim[0] ? a[0].size()/a[0].shape.dim[0] : 0;
  const bool contiguous=!take && a[0].contiguous() && out.order==MemoryOrder::c;
  const auto *source=static_cast<const double *>(a[0].data);
  for (std::size_t i=0;i<a[1].size();++i) {
    const auto destination=std::size_t(a[1].i(i))*width;
    if (contiguous) std::copy_n(source+i*width,width,y+destination);
    else for (std::size_t j=0;j<width;++j) {
      const auto value=take ? take->args[0].at(i,static_cast<std::size_t>(take->args[1].i(j))) : a[0].f(i*width+j);
      out.set(destination+j,value);
    }
  }
  audit.algorithm_copy_bytes+=a[0].size()*sizeof(double);
}

void scatter_take_pair(const Prepared &take, const Prepared &scatter, Output &out,
                       Workspace &work, Audit &audit) {
  require(take.spec->op==O::take_axis && take.args[2].scalar==1 &&
      take.output_shape.rank==2 && scatter.spec->op==O::scatter_rows,"INVALID_OPERATOR");
  validate_scatter_indices(scatter,work,audit);
  scatter_values(scatter,out,audit,&take);
}

void latent_execute(const Prepared &p, Output &out, Workspace &work,
                    Isa requested, Audit &audit, const LatentFitScratch *fit_scratch) {
  const auto op = p.spec->op;
  const auto &a = p.args;
  auto *y = static_cast<double *>(out.data);
  if (op == O::transition_counts || op == O::one_hot)
    indices(p, 0, std::size_t(a[1].scalar), true, false, true);
  if (op == O::scatter_rows) {
    validate_scatter_indices(p,work,audit);
  } else if (op == O::partition_mean) {
    const bool copied = indices(p, 1,
            op == O::partition_mean ? a[0].shape.dim[0]
                                    : std::size_t(a[2].scalar),
            true, true, false, work.indices.data());
    audit.algorithm_copy_bytes = copied ? a[1].size() * sizeof(std::size_t) : 0;
  }
  if (latent_view(op))
    return; // handled by the common borrowed-output path
  if (op >= O::em_iterations && op <= O::em_converged) {
    auto w = a[0].shape.dim[1];
    auto value = a[0].at(0, w - (op == O::em_iterations       ? 3
                                 : op == O::em_log_likelihood ? 2
                                                              : 1));
    finite(value);
    if (op == O::em_iterations) {
      require(value >= 1 && value <= 10000 && value == std::floor(value),
              "INVALID_PARAMETER");
      out.set_integer(0, std::int64_t(value));
    } else if (op == O::em_converged) {
      require(value == 0 || value == 1, "INVALID_MASK");
      out.set_mask(0, std::uint8_t(value));
    } else
      out.set(0, value);
    return;
  }
  if (op == O::one_hot || op == O::transition_counts) {
    if (!out.shape.size())
      return;
    const auto k = std::size_t(a[1].scalar);
    const auto *codes = static_cast<const std::int64_t *>(a[0].data);
    const auto step = a[0].stride[0];
    std::fill(y, y + out.shape.size(), 0.0);
    for (std::size_t i = 0; i < a[0].size(); ++i) {
      auto code = codes[std::ptrdiff_t(i) * step];
      if (code < 0)
        continue;
      if (op == O::one_hot)
        y[i * k + std::size_t(code)] = 1;
      else if (i && codes[std::ptrdiff_t(i - 1) * step] >= 0)
        y[std::size_t(codes[std::ptrdiff_t(i - 1) * step]) * k +
          std::size_t(code)] += 1;
    }
    return;
  }
  if (op == O::sample_unique_indices) {
    const auto n = a[0].shape.dim[0], k = out.shape.size();
    auto *z = static_cast<std::int64_t *>(out.data);
    auto state = std::int64_t(a[2].scalar) % 2147483646LL;
    if (state < 0)
      state += 2147483646LL;
    ++state;
    for (std::size_t i = 0; i < k; ++i) {
      state = (state * 48271) % 2147483647LL;
      auto row = std::size_t(state) % n;
      bool used = true;
      while (used) {
        used = false;
        for (std::size_t j = 0; j < i; ++j)
          if (z[j] == std::int64_t(row)) {
            row = (row + 1) % n;
            used = true;
            break;
          }
      }
      z[i] = std::int64_t(row);
    }
    return;
  }
  if (op == O::scatter_rows) {
    scatter_values(p,out,audit);
    return;
  }
  if (op == O::partition_mean) {
    finite_array(a[0]);
    auto n = a[0].shape.dim[0], d = a[0].shape.dim[1], k = out.shape.dim[0];
    std::size_t end = 0, remainder = 0;
    for (std::size_t c = 0; c < k; ++c) {
      auto start = end;
      end += n / k;
      remainder += n % k;
      if (remainder >= k) {
        ++end;
        remainder -= k;
      }
      for (std::size_t first = 0; first < d; first += 8) {
        const auto lanes = std::min<std::size_t>(8, d - first);
        double sums[8]{};
        for (auto i = start; i < end; ++i) {
          const auto row = std::size_t(a[1].i(i));
          for (std::size_t j = 0; j < lanes; ++j)
            sums[j] += a[0].at(row, first + j);
        }
        for (std::size_t j = 0; j < lanes; ++j) {
          out.set(c * d + first + j, sums[j] / double(end - start));
          finite(sums[j]);
        }
      }
    }
    return;
  }
  if (op >= O::variance_axis && op <= O::softmax_axis) {
    if (!out.shape.size())
      return;
    const auto axis = std::size_t(a[1].scalar), count = a[0].shape.dim[axis];
    require(count > 0, "INSUFFICIENT_SAMPLE");
    if (op == O::variance_axis)
      require(a[2].scalar < double(count), "INSUFFICIENT_SAMPLE");
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
    if (op == O::variance_axis && axis == 0 && requested != Isa::scalar &&
        eigen_ordered_rows<Op::variance_axis>(a[0],double(count)-a[2].scalar,out,audit)) return;
#endif
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (op == O::variance_axis && axis == 0 && a[0].shape.rank == 2 && requested != Isa::scalar && count >= 32 && a[0].shape.dim[1] >= 8 && (a[0].stride[0] == 1 || a[0].stride[0] == -1) && a[0].stride[1] >= std::ptrdiff_t(count)) {
      if (__builtin_available(macOS 15.0, *)) {
        eigen_ordered_columns<Op::variance_axis>(a[0],double(count)-a[2].scalar,out,audit);return;
      }
    }
#endif
    if (op == O::variance_axis && axis == 0 && a[0].shape.rank == 2 && a[0].contiguous()) {
      variance_columns(a[0], a[2].scalar, out, requested, audit);
      return;
    }
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (op == O::softmax_axis && requested != Isa::scalar && a[0].size() >= 256 &&
        count >= 8 && a[0].stride[axis] == 1) {
      if (__builtin_available(macOS 15.0, *)) {
        std::size_t inner = 1;
        for (int j = int(axis) + 1; j < a[0].shape.rank; ++j) inner *= a[0].shape.dim[j];
        if (inner == 1) {
          eigen_softmax_rows(p, out, audit);
          return;
        }
      }
    }
#endif
    switch (op) {
#define AXIS(name) case O::name: axis_reduce<O::name>(p, out); break;
    AXIS(variance_axis) AXIS(max_axis) AXIS(argmax_axis) AXIS(logsumexp_axis) AXIS(softmax_axis)
#undef AXIS
    default: break;
    }
    return;
  }
  const auto n = a[0].shape.dim[0];
  if (op == O::hmm_filter || op == O::hmm_smooth) {
    const auto k = a[1].size();
    require(k > 0, "INSUFFICIENT_SAMPLE");
    finite_array(a[0]);
    probabilities(a[1]);
    probabilities(a[2], true);
    if (!n)
      return;
    if (op == O::hmm_filter) {
      for (std::size_t i = 0; i < n; ++i) {
        filter_row(a.data(), i, k, i ? y + (i - 1) * k : nullptr, y + i * k);
      }
      return;
    }
    auto *log_a = work.doubles.data(), *beta = log_a + k * k, *next = beta + k,
         *temp = next + k, *init = temp + k;
    for (std::size_t s = 0; s < k; ++s) {
      init[s] = a[1].f(s);
      beta[s] = 0;
      for (std::size_t t = 0; t < k; ++t)
        log_a[s * k + t] = std::log(std::max(a[2].at(s, t), 1e-12));
    }
    for (std::size_t c = 0; c < k; ++c)
      y[c] = std::log(std::max(init[c], 1e-12)) + a[0].at(0, c);
    for (std::size_t i = 1; i < n; ++i)
      for (std::size_t t = 0; t < k; ++t) {
        for (std::size_t s = 0; s < k; ++s)
          temp[s] = y[(i - 1) * k + s] + log_a[s * k + t];
        y[i * k + t] = a[0].at(i, t) + lse(temp, k);
      }
    for (std::size_t q = 0; q < n; ++q) {
      auto i = n - 1 - q;
      for (std::size_t c = 0; c < k; ++c)
        temp[c] = y[i * k + c] + beta[c];
      auto z = lse(temp, k);
      for (std::size_t c = 0; c < k; ++c)
        y[i * k + c] = std::exp(temp[c] - z);
      if (i) {
        for (std::size_t s = 0; s < k; ++s) {
          for (std::size_t t = 0; t < k; ++t)
            temp[t] = log_a[s * k + t] + a[0].at(i, t) + beta[t];
          next[s] = lse(temp, k);
        }
        std::swap(beta, next);
      }
    }
    return;
  }
  const bool hmm = op == O::gaussian_hmm_em,
             fit = hmm || op == O::gaussian_mixture_em;
  const std::size_t mi = fit ? (hmm ? 3 : 2) : 1, k = a[mi].shape.dim[0],
                    d = a[0].shape.dim[1];
  validate_density_inputs(a[0], a[mi], a[mi + 1], fit);
  if (!fit) {
    density(a[0], a[mi], a[mi + 1], y, work.doubles.data(), requested, audit);
    return;
  }
  probabilities(a[1]);
  if (hmm)
    probabilities(a[2], true);
  const auto width = out.shape.dim[1], base = 1 + (hmm ? k : 0);
  for (std::size_t c = 0; c < k; ++c) {
    y[c * width] = a[1].f(c);
    if (hmm)
      for (std::size_t j = 0; j < k; ++j)
        y[c * width + 1 + j] = a[2].at(c, j);
    for (std::size_t j = 0; j < d; ++j) {
      y[c * width + base + j] = a[mi].at(c, j);
      y[c * width + base + d + j] = a[mi + 1].at(c, j);
    }
  }
  audit.algorithm_copy_bytes = (k * (1 + 2 * d) + (hmm ? k * k : 0)) * 8;
  const bool replay_gmm = !hmm && !fit_scratch && gmm_replay_scratch(n, k, d);
  auto *em = fit_scratch ? fit_scratch->em : work.doubles.data();
  const bool compact_hmm = hmm && !fit_scratch && a[6].scalar==0.0 &&
      latent_hmm_alpha_checkpoint_eligible(n,k,d);
  auto *alpha = fit_scratch ? fit_scratch->alpha : hmm && !compact_hmm ? em + n * k : em;
  const bool replay_density = hmm && !compact_hmm && latent_fit_matrix_count(op, n, k) == 2 &&
      (!fit_scratch || fit_scratch->beta == fit_scratch->em);
  auto *beta = fit_scratch ? fit_scratch->beta : hmm && !replay_density && !compact_hmm ? alpha + n * k : em;
  auto *small = fit_scratch ? fit_scratch->small :
      compact_hmm ? em+n*k : hmm ? (replay_density ? alpha : beta) + n * k : replay_gmm ? em : em + n * k;
  auto *log_a = hmm ? small : em, *xi = hmm ? log_a + k * k : em,
       *temp = hmm ? xi + k * k : small, *effective = temp + k,
       *initial = effective + k, *logs = hmm ? initial + k : initial;
  double prior = -std::numeric_limits<double>::infinity(), ll = prior;
  bool converged = false;
  std::size_t completed = 0;
  const auto it = std::size_t(a[hmm ? 5 : 4].scalar);
  const auto tolerance = a[hmm ? 6 : 5].scalar;
  Value means_view = a[mi], vars_view = a[mi + 1];
  means_view.data = y + base;
  vars_view.data = y + base + d;
  means_view.stride = vars_view.stride = {std::ptrdiff_t(width), 1, 1};
  for (std::size_t iteration = 1; iteration <= it; ++iteration) {
    completed = iteration;
    if (replay_gmm) {
      ll = gmm_replay_step(a[0], means_view, vars_view, y, width, em, requested, audit);
      converged = std::abs(ll - prior) < tolerance;
      if (converged) break;
      prior = ll;
      continue;
    }
    density(a[0], means_view, vars_view, em, logs, requested, audit);
    if (!hmm) {
      for (std::size_t c = 0; c < k; ++c)
        logs[c] = std::log(std::max(y[c * width], 1e-12));
      ll = 0;
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < k; ++c)
          em[i * k + c] += logs[c];
        auto z = lse(em + i * k, k);
        ll += z;
        for (std::size_t c = 0; c < k; ++c)
          em[i * k + c] = std::exp(em[i * k + c] - z);
      }
    } else {
      for (std::size_t c = 0; c < k; ++c) {
        initial[c] = y[c * width];
        for (std::size_t t = 0; t < k; ++t)
          log_a[c * k + t] = std::log(std::max(y[c * width + 1 + t], 1e-12));
      }
      bool state_library = false;
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
      if (__builtin_available(macOS 15.0, *))
        state_library = requested != Isa::scalar && std::fegetround() == FE_TONEAREST &&
            n >= 4096 && k >= 8 && k <= 128 && tolerance == 0.0;
      // Vector exp can change a strict stopping boundary. Only fixed-budget
      // fits are admitted; every positive-tolerance fit keeps scalar sweeps.
#endif
      if (compact_hmm)
        ll=hmm_alpha_checkpoint_step(a[0],means_view,vars_view,em,small,
            p.scratch_doubles-n*k,requested,state_library,audit);
      else {
      forward(em, n, k, initial, log_a, alpha, temp, state_library, xi, audit);
      bool library = false;
      if (replay_density) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
        library = requested != Isa::scalar && std::fegetround() == FE_TONEAREST &&
            bounded_hmm_library_values(alpha, nullptr, em, log_a, n, k);
#endif
        if (library) library = backward<true, true>(em, n, k, log_a, beta, temp, audit, initial, effective, state_library, xi);
        else backward<true>(em, n, k, log_a, beta, temp, audit, initial, effective, state_library, xi);
      } else {
        backward<false>(em, n, k, log_a, beta, temp, audit,
                        nullptr, nullptr, state_library, xi);
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
        library = requested != Isa::scalar && n >= 4096 && k >= 3 &&
            std::fegetround() == FE_TONEAREST &&
            bounded_hmm_library_values(alpha, beta, em, log_a, n, k);
#endif
      }
      ll = lse(alpha + (n - 1) * k, k);
      if (replay_density || library) {
        std::fill(xi, xi + k * k, 0.0);
        for (std::size_t first = 0; first + 1 < n; first += 128) {
          const auto count = std::min(std::size_t{128}, n - 1 - first);
          const double *emission = em + (first + 1) * k;
          if (replay_density) {
            auto block = a[0];
            block.data = static_cast<const double *>(a[0].data) +
                std::ptrdiff_t(first + 1) * a[0].stride[0];
            block.shape.dim[0] = count;
            auto *replayed = logs + k + 512;
            // Parameters/logs still belong to this E step, before M.
            density_rows(block, means_view, vars_view, replayed, logs, n, requested, audit);
            emission = replayed;
          }
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
          if (library)
            library_hmm_transition_tile(alpha + first * k, beta + (first + 1) * k,
                emission, log_a, count, k, xi, logs + k, logs + k + 128,
                logs + k + 256, logs + k + 384, audit);
          else
#endif
            ordered_hmm_transition_tile(alpha + first * k, beta + (first + 1) * k,
                emission, log_a, count, k, xi, logs + k);
        }
      } else if (requested != Isa::scalar && n >= 128 && k >= 3)
        accumulate_hmm_transitions<128>(alpha, beta, em, log_a, n, k, xi, logs + k);
      else
        accumulate_hmm_transitions<1>(alpha, beta, em, log_a, n, k, xi, temp);
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < k; ++c)
          temp[c] = alpha[i * k + c] + beta[i * k + c];
        auto z = lse(temp, k);
        for (std::size_t c = 0; c < k; ++c)
          alpha[i * k + c] = std::exp(temp[c] - z);
      }
      }
      double total = 0;
      for (std::size_t c = 0; c < k; ++c) {
        y[c * width] = std::max(alpha[c], 1e-8);
        total += y[c * width];
      }
      for (std::size_t c = 0; c < k; ++c)
        y[c * width] /= total;
      for (std::size_t s = 0; s < k; ++s) {
        total = 0;
        for (std::size_t t = 0; t < k; ++t) {
          y[s * width + 1 + t] = std::max(xi[s * k + t], 1e-8);
          total += y[s * width + 1 + t];
        }
        for (std::size_t t = 0; t < k; ++t)
          y[s * width + 1 + t] /= total;
      }
    }
    finite(ll);
    for (std::size_t c = 0; c < k; ++c) {
      double count = 0;
      for (std::size_t i = 0; i < n; ++i)
        count += alpha[i * k + c];
      effective[c] = std::max(count, 1e-8);
      if (!hmm)
        y[c * width] = effective[c] / double(n);
      auto *means = y + c * width + base;
      weighted_feature_sums<false>(a[0], alpha, k, c, nullptr, means, requested, audit);
      for (std::size_t j = 0; j < d; ++j) {
        means[j] /= effective[c];
        finite(means[j]);
      }
    }
    for (std::size_t c = 0; c < k; ++c) {
      const auto *means = y + c * width + base;
      auto *variances = y + c * width + base + d;
      weighted_feature_sums<true>(a[0], alpha, k, c, means, variances, requested, audit);
      for (std::size_t j = 0; j < d; ++j) {
        variances[j] = std::max(variances[j] / effective[c], 1e-5);
        finite(variances[j]);
      }
    }
    converged = std::abs(ll - prior) < tolerance;
    if (converged)
      break;
    prior = ll;
  }
  for (std::size_t c = 0; c < k; ++c) {
    y[c * width + width - 3] = double(completed);
    y[c * width + width - 2] = ll;
    y[c * width + width - 1] = converged ? 1 : 0;
  }
}
} // namespace calmetrics_engine::ops
