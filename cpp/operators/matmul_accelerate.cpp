#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include "matmul_partitioned.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <climits>
#include <cmath>
#include <limits>
#include <cstring>
#include <new>

#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace calmetrics_engine::ops {
namespace {
std::size_t native_matmul_reserve_doubles(const Shape &a,const Shape &b,bool expanded_geometry=true) {
    // Constrained Plans preserve the original optional-library geometry.
    if(!expanded_geometry && !(a.rank==2&&b.rank==2&&a.dim[0]>=512&&a.dim[0]<=8192&&
        b.dim[1]>=512&&b.dim[1]<=8192&&a.dim[1]>=256&&a.dim[1]<=8192))return 0;
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if(__builtin_available(macOS 15.0,*)) {
        if(a.rank==2&&b.rank==2&&a.dim[1]==b.dim[0]&&
           a.dim[0]&&a.dim[0]<=INT_MAX&&b.dim[1]&&b.dim[1]<=INT_MAX&&
           a.dim[1]&&a.dim[1]<=8192) {
            // Preserve the previous minimum work, without restricting aspect ratio.
            constexpr std::size_t minimum_work=512*256*512;
            const auto columns=(minimum_work+a.dim[1]-1)/a.dim[1];
            const auto rows=(columns+b.dim[1]-1)/b.dim[1];
            if(a.dim[0]<rows)return 0;
            const auto signs=std::max(a.dim[0],b.dim[1]);
            // Existing geometries keep exactly the previous reserve; longer
            // axes account for the independent byte-sized sign metadata.
            return 1024*1024+512*a.dim[1]+(signs>8192?(signs+7)/8:0);
        }
    }
#else
    (void)a;(void)b;
#endif
    return 0;
}
std::size_t partitioned_matmul_scratch(const Shape &a,const Shape &b,bool expanded_geometry=true) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(CALMETRICS_HAVE_EIGEN_ROTATION)
    if(native_matmul_reserve_doubles(a,b,expanded_geometry))return 320*a.dim[1]+72;
#else
    (void)a;(void)b;(void)expanded_geometry;
#endif
    return 0;
}
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(CALMETRICS_HAVE_EIGEN_ROTATION)
bool partitioned_matmul_root(const Prepared &prepared,Output &output,Workspace &work,Audit &audit,
                            double ma,double mb,void *context,void (*checkpoint)(void*)) {
    const auto &a=prepared.args[0],&b=prepared.args[1];
    const auto cap=partitioned_matmul_scratch(a.shape,b.shape);
    if(!cap)return false;
    const partitioned::Input lhs{static_cast<const double*>(a.data),a.shape.dim[0],a.shape.dim[1],a.stride[0],a.stride[1]};
    const partitioned::Input rhs{static_cast<const double*>(b.data),b.shape.dim[0],b.shape.dim[1],b.stride[0],b.stride[1]};
    partitioned::Plan plan;
    {
        // Preflight can reject. Keep the caller's retained workspace intact;
        // this bounded norm-only temporary dies before numerical growth.
        const auto lines=std::max(lhs.rows,rhs.columns);
        const auto count=lines>(cap-8)/2?cap:2*lines+8;
        std::vector<double> norms(count);
        if(!partitioned::make_plan(lhs,rhs,ma,mb,plan,norms.data(),norms.size(),context,checkpoint,output.order==MemoryOrder::f))return false;
    }
    const auto previous=BLASGetThreading();
    if(previous!=BLAS_THREADING_MULTI_THREADED&&previous!=BLAS_THREADING_SINGLE_THREADED)return false;
    const auto restore=[&] {
        require(BLASSetThreading(previous)==0&&BLASGetThreading()==previous,
                "NATIVE_LIBRARY_THREAD_RESTORE_FAILED");
    };
    if(BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)!=0||BLASGetThreading()!=BLAS_THREADING_SINGLE_THREADED) {
        restore();return false;
    }
    partitioned::Stats stats;
    bool accepted=false;
    try {
        if(work.doubles.capacity()>cap)std::vector<double>().swap(work.doubles);
        std::vector<std::size_t>().swap(work.indices);
        work.ensure(cap,0);
        accepted=partitioned::calculate(lhs,rhs,plan,static_cast<double*>(output.data),
            output.order==MemoryOrder::f,work.doubles,stats,context,checkpoint);
    } catch(...) {restore();throw;}
    restore();
    if(checkpoint)checkpoint(context);
    audit={};
    if(!accepted) {
        // Late rejection may have written private cells. Replay ALL cells;
        // ops::execute is the sole canonical implementation, including F layout.
        execute(prepared,output,work,Isa::automatic,audit);
        if(checkpoint)checkpoint(context);
        audit.isa="accelerate-partitioned-canonical-replay";
    }else {
        audit.isa="accelerate-partitioned-matmul";
        audit.vector_elements=output.shape.size();
    }
    audit.accelerate_matmul_calls+=stats.calls;
    audit.algorithm_copy_bytes+=stats.output_copy_bytes;
    audit.scratch_bytes=std::max(audit.scratch_bytes,work.capacity_bytes());
    audit.native_library_workspace_reserve_bytes=std::max(audit.native_library_workspace_reserve_bytes,
        native_matmul_reserve_doubles(a.shape,b.shape)*sizeof(double));
    return true;
}
#endif
} // namespace

std::size_t matmul_accelerate_reserve_doubles(const Shape &a,const Shape &b,bool expanded_geometry) {
    // Both buffers can be simultaneously live while DGEMM packs internally.
    return native_matmul_reserve_doubles(a,b,expanded_geometry)+partitioned_matmul_scratch(a,b,expanded_geometry);
}

bool try_accelerate_matmul_terminal_float(const Prepared &p, Output &output,
    Workspace &work, Audit &audit, void *context, void (*checkpoint)(void *),
    bool expanded_geometry, const MatmulTerminalProof *proof) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (__builtin_available(macOS 15.0,*)) {
        if (!proof || proof->source!=&p || std::fegetround()!=FE_TONEAREST ||
            !p.spec || p.spec->op!=Op::matmul || p.count!=2 ||
            output.kind!=Kind::number || !output.data ||
            (output.order!=MemoryOrder::c && output.order!=MemoryOrder::f)) return false;
        const auto &a=p.args[0],&b=p.args[1];
        if (a.kind!=Kind::number || b.kind!=Kind::number || !a.data || !b.data ||
            a.shape.rank!=2 || b.shape.rank!=2 || a.shape.dim[1]!=b.shape.dim[0] ||
            output.shape.rank!=2 || output.shape.dim[0]!=a.shape.dim[0] ||
            output.shape.dim[1]!=b.shape.dim[1] ||
            !std::isfinite(proof->left.maximum) || !std::isfinite(proof->right.maximum) ||
            proof->left.maximum<=0.0 || proof->right.maximum<=0.0 ||
            proof->left.maximum>1.0 || proof->right.maximum>1.0) return false;
        const auto reserve=native_matmul_reserve_doubles(a.shape,b.shape,expanded_geometry);
        if (!reserve) return false;
        const auto m=a.shape.dim[0],k=a.shape.dim[1],n=b.shape.dim[1];
        const auto cells=output.shape.size(),left=a.shape.size(),right=b.shape.size();
        // All geometries are already byte-bounded by native preparation.
        // Half of the double output holds raw float C; its other half holds
        // both packed inputs. No new allocation or planner capacity is needed.
        if (left>cells || right>cells-left) return false;
        const bool f=output.order==MemoryOrder::f;
        const auto describe=[f](const Value &v,CBLAS_TRANSPOSE &trans,int &leading) {
            if (v.stride[1]==1 && v.stride[0]==std::ptrdiff_t(v.shape.dim[1])) {
                trans=f?CblasTrans:CblasNoTrans;leading=int(v.shape.dim[1]);return true;
            }
            if (v.stride[0]==1 && v.stride[1]==std::ptrdiff_t(v.shape.dim[0])) {
                trans=f?CblasNoTrans:CblasTrans;leading=int(v.shape.dim[0]);return true;
            }
            return false;
        };
        CBLAS_TRANSPOSE ta,tb;int lda,ldb;
        if (!describe(a,ta,lda) || !describe(b,tb,ldb)) return false;
        const auto valid=[&](const Value &v) {
            const auto *data=static_cast<const double *>(v.data);
            for (std::size_t i=0;i<v.shape.size();++i) {
                if (checkpoint && i%65536==0) checkpoint(context);
                const double value=data[i],magnitude=std::abs(value);
                if (!std::isfinite(value) || magnitude>1.0 ||
                    (value==0.0 ? std::signbit(value) : magnitude<0x1p-36)) return false;
            }
            return true;
        };
        // Preflight precedes all output writes. A rounded nonzero input has
        // exponent >= -36, hence float input grid >=2^-59 and product grid
        // >=2^-118. Nonzero products/intermediate sums cannot be subnormal;
        // k<=8192 and |x|<=1 rule out float overflow.
        if (!valid(a) || !valid(b)) return false;
        const auto previous=BLASGetThreading();
        if (previous!=BLAS_THREADING_MULTI_THREADED && previous!=BLAS_THREADING_SINGLE_THREADED) return false;
        const auto restore=[&] { require(BLASSetThreading(previous)==0 &&
            BLASGetThreading()==previous,"NATIVE_LIBRARY_THREAD_RESTORE_FAILED"); };
        if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)!=0 ||
            BLASGetThreading()!=BLAS_THREADING_SINGLE_THREADED) { restore();return false; }
        std::vector<double>().swap(work.doubles);
        std::vector<std::size_t>().swap(work.indices);
        auto *bytes=static_cast<unsigned char *>(output.data);
        auto *product=reinterpret_cast<float *>(bytes);
        auto *packed_a=reinterpret_cast<float *>(bytes+cells*sizeof(float));
        auto *packed_b=reinterpret_cast<float *>(bytes+(cells+left)*sizeof(float));
        // Explicit placement starts float object lifetimes in private raw
        // storage; default initialization writes nothing. After the synchronous
        // GEMM inputs are dead and backwards expansion restores double objects.
        try {
            product=::new(static_cast<void *>(product)) float[cells];
            const auto pack=[&](const Value &v,float *target) {
                const auto *data=static_cast<const double *>(v.data);
                target=::new(static_cast<void *>(target)) float[v.shape.size()];
                for (std::size_t i=0;i<v.shape.size();++i) {
                    if (checkpoint && i%65536==0) checkpoint(context);
                    target[i]=static_cast<float>(data[i]);
                }
            };
            pack(a,packed_a);pack(b,packed_b);
            if (checkpoint) checkpoint(context);
            cblas_sgemm(f?CblasColMajor:CblasRowMajor,ta,tb,int(m),int(n),int(k),
                1.0f,packed_a,lda,packed_b,ldb,0.0f,product,int(f?m:n));
            restore();
            if (checkpoint) checkpoint(context);
            // Restore the complete indexable double array before expansion.
            // Default initialization does not write its object representation;
            // read the initialized GEMM bytes through character storage only.
            // Retain the new-expression pointer rather than a stale typed view.
            auto *restored=::new(output.data) double[cells];
            output.data=restored;
            for (std::size_t remaining=cells;remaining;) {
                const std::size_t i=--remaining;
                if (checkpoint && i%65536==0) checkpoint(context);
                unsigned char representation[sizeof(float)];
                std::memcpy(representation,bytes+i*sizeof(float),sizeof(float));
                float value;
                std::memcpy(&value,representation,sizeof(float));
                restored[i]=static_cast<double>(value);
            }
        } catch (...) {
            // Reestablish double lifetimes before the graph's existing
            // unpublished-root error guard clears this region.
            auto *restored=::new(output.data) double[cells];
            output.data=restored;
            std::fill_n(restored,cells,0.0);
            restore();
            throw;
        }
        audit={};audit.isa="accelerate-private-bounded-float-matmul";
        audit.accelerate_matmul_calls=1;audit.vector_elements=cells;
        audit.algorithm_copy_bytes=(left+right)*sizeof(float)+cells*sizeof(double);
        audit.native_library_workspace_reserve_bytes=reserve*sizeof(double);
        return true;
    }
#else
    (void)p;(void)output;(void)work;(void)audit;(void)context;(void)checkpoint;
    (void)expanded_geometry;(void)proof;
#endif
    return false;
}

bool try_accelerate_matmul_root(const Prepared &p, Output &output,
                                         Workspace &work, Audit &audit, void *context, void (*checkpoint)(void*),bool expanded_geometry,
                                         const MatmulTerminalProof *terminal_proof) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (__builtin_available(macOS 15.0, *)) {
        if (p.spec->op != Op::matmul || p.count != 2 ||
            (output.order != MemoryOrder::c && output.order != MemoryOrder::f) ||
            output.kind != Kind::number ||
            !output.data) return false;
        const auto &a = p.args[0], &b = p.args[1];
        const auto reserve = native_matmul_reserve_doubles(a.shape, b.shape,expanded_geometry);
        if (!reserve || a.kind != Kind::number || b.kind != Kind::number ||
            !a.data || !b.data) return false;
        if(checkpoint)checkpoint(context);
        const bool f_output = output.order == MemoryOrder::f;
        const auto describe = [f_output](const Value &v, CBLAS_TRANSPOSE &trans, int &ld) {
            const auto rows = static_cast<std::ptrdiff_t>(v.shape.dim[0]);
            const auto cols = static_cast<std::ptrdiff_t>(v.shape.dim[1]);
            if (v.stride[1] == 1 && v.stride[0] >= cols && v.stride[0] <= INT_MAX) {
                trans = f_output ? CblasTrans : CblasNoTrans;
                ld = static_cast<int>(v.stride[0]); return true;
            }
            if (v.stride[0] == 1 && v.stride[1] >= rows && v.stride[1] <= INT_MAX) {
                trans = f_output ? CblasNoTrans : CblasTrans;
                ld = static_cast<int>(v.stride[1]); return true;
            }
            return false;
        };
        CBLAS_TRANSPOSE ta, tb;
        int lda, ldb;
        if (!describe(a, ta, lda) || !describe(b, tb, ldb)) return false;
        if (terminal_proof && (terminal_proof->source != &p ||
            std::fegetround() != FE_TONEAREST ||
            !std::isfinite(terminal_proof->left.maximum) ||
            !std::isfinite(terminal_proof->right.maximum) ||
            terminal_proof->left.maximum < 0x1p-128 ||
            terminal_proof->right.maximum < 0x1p-128 ||
            terminal_proof->left.maximum > 0x1p128 ||
            terminal_proof->right.maximum > 0x1p128)) return false;
        bool has_negative = false, has_zero = false, mixed_sign = false;
        const auto scan = [&](const Value &v, bool by_row, bool &all_zero, double &maximum) {
            // Detect the existing cancellation-free domain and bound the
            // general signed domain independently of physical C/F storage.
            std::array<std::int8_t, 8192> short_signs{};
            std::vector<std::int8_t> long_signs;
            const auto count=v.shape.dim[by_row?0:1];
            if(count>short_signs.size())long_signs.resize(count);
            auto *signs=long_signs.empty()?short_signs.data():long_signs.data();
            const bool column = v.stride[0] == 1;
            const auto rows = v.shape.dim[0], cols = v.shape.dim[1];
            const auto *data = static_cast<const double *>(v.data);
            all_zero = true; maximum = 0.0;
            for (std::size_t outer = 0; outer < (column ? cols : rows); ++outer) {
                if (checkpoint && outer % 64 == 0) checkpoint(context);
                for (std::size_t inner = 0; inner < (column ? rows : cols); ++inner) {
                    const auto i = column ? inner : outer, j = column ? outer : inner;
                    const auto value = data[static_cast<std::ptrdiff_t>(i) * v.stride[0] +
                                            static_cast<std::ptrdiff_t>(j) * v.stride[1]];
                    if (!std::isfinite(value)) return false;
                    if (value == 0.0) {
                        if (std::signbit(value)) return false;
                        has_zero = true;
                        continue;
                    }
                    const auto magnitude = std::abs(value);
                    if (magnitude < 0x1p-128 || magnitude > 0x1p128)
                        return false;
                    const std::int8_t sign = std::signbit(value) ? -1 : 1;
                    auto &previous = signs[by_row ? i : j];
                    if (previous && previous != sign) mixed_sign = true;
                    previous = sign;
                    maximum = std::max(maximum, magnitude);
                    has_negative = has_negative || sign < 0;
                    all_zero = false;
                }
            }
            return true;
        };
        bool zero_a, zero_b;
        double max_a, max_b;
        const auto proves_mixed = [](const Value &v, bool by_row) {
            const auto count = std::min<std::size_t>(64, v.shape.dim[by_row ? 1 : 0]);
            const auto stride = v.stride[by_row ? 1 : 0];
            const auto *data = static_cast<const double *>(v.data);
            int previous = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const double value = data[std::ptrdiff_t(i) * stride];
                if (!std::isfinite(value)) return false;
                if (value == 0.0) continue;
                const int sign = std::signbit(value) ? -1 : 1;
                if (previous && previous != sign) return true;
                previous = sign;
            }
            return false;
        };
        // A witnessed opposite-sign pair within one A row or B column proves
        // the original mixed domain; sampling never proves validity or absence.
        const bool witnessed = !terminal_proof && (proves_mixed(a, true) || proves_mixed(b, false));
        if (witnessed && std::fegetround() != FE_TONEAREST) return false;
        bool bounded = false;
        if (terminal_proof) {
            // The same-call graph has already checked all finite nonzero data,
            // their magnitudes and the final consumer closure. No new scan,
            // input packing or mathematical cache is introduced here.
            max_a=terminal_proof->left.maximum; max_b=terminal_proof->right.maximum;
            has_negative=terminal_proof->left.negative || terminal_proof->right.negative;
            mixed_sign=true; zero_a=zero_b=false; bounded=true;
        }
        if (witnessed) {
            const auto input_bounds=[&](const Value &value,std::size_t axis,MatmulInputBounds &bounds) {
                constexpr std::size_t block=8192;
                if(value.shape.dim[axis]<=block)return library_matmul_input_bounds(value,bounds);
                MatmulInputBounds combined;
                const auto *data=static_cast<const double*>(value.data);
                for(std::size_t begin=0;begin<value.shape.dim[axis];begin+=block) {
                    if(checkpoint)checkpoint(context);
                    auto part=value;
                    part.shape.dim[axis]=std::min(block,value.shape.dim[axis]-begin);
                    part.data=data+std::ptrdiff_t(begin)*value.stride[axis];
                    MatmulInputBounds piece;
                    const auto state=library_matmul_input_bounds(part,piece);
                    if(state!=MatmulBoundsStatus::valid)return state;
                    combined.maximum=std::max(combined.maximum,piece.maximum);
                    combined.negative=combined.negative||piece.negative;
                }
                bounds=combined;
                return MatmulBoundsStatus::valid;
            };
            MatmulInputBounds left, right;
            const auto left_status = input_bounds(a,0,left);
            if (left_status == MatmulBoundsStatus::rejected) return false;
            if (left_status == MatmulBoundsStatus::valid) {
                const auto right_status = input_bounds(b,1,right);
                if (right_status == MatmulBoundsStatus::rejected) return false;
                if (right_status == MatmulBoundsStatus::valid) {
                    max_a = left.maximum; max_b = right.maximum;
                    has_negative = left.negative || right.negative;
                    mixed_sign = true; zero_a = zero_b = false; bounded = true;
                }
            }
        }
        if (!bounded && (!scan(a, true, zero_a, max_a) || !scan(b, false, zero_b, max_b)))
            return false;
        if (has_negative && std::fegetround() != FE_TONEAREST) return false;
        double error_bound = 0.0;
        if (mixed_sign && !zero_a && !zero_b) {
            // Standard binary64 dot-product bound for both the canonical fold
            // and GEMM. Round each positive operation outward; no data packing.
            const auto up = [](double x) {
                return std::nextafter(x, std::numeric_limits<double>::infinity());
            };
            const double ku = 2.0 * double(a.shape.dim[1]) * 0x1p-53;
            const double gamma = up(ku / std::nextafter(1.0 - ku, 0.0));
            error_bound = up(up(up(up(2.0 * gamma) * double(a.shape.dim[1])) * max_a) * max_b);
            if (error_bound > 0.000049) {
                if (terminal_proof) return false;
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION)
                return partitioned_matmul_root(p,output,work,audit,max_a,max_b,context,checkpoint);
#else
                return false;
#endif
            }
        }
        if (zero_a || zero_b) {
            std::vector<double>().swap(work.doubles);
            std::vector<std::size_t>().swap(work.indices);
            audit = {};
            std::fill_n(static_cast<double *>(output.data), output.shape.size(), 0.0);
            audit.isa = "std-fill-matmul-zero";
            audit.matmul_zero_calls = 1;
            return true;
        }
        const auto previous = BLASGetThreading();
        if (previous != BLAS_THREADING_MULTI_THREADED &&
            previous != BLAS_THREADING_SINGLE_THREADED) return false;
        const auto restore = [&] {
            const auto status = BLASSetThreading(previous);
            require(status == 0 && BLASGetThreading() == previous,
                    "NATIVE_LIBRARY_THREAD_RESTORE_FAILED");
        };
        const auto status = BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        if (status != 0 || BLASGetThreading() != BLAS_THREADING_SINGLE_THREADED) {
            restore(); return false;
        }
        // Release retained numerical panels only after successful admission.
        std::vector<double>().swap(work.doubles);
        std::vector<std::size_t>().swap(work.indices);
        audit = {};
        try {
            cblas_dgemm(f_output ? CblasColMajor : CblasRowMajor, ta, tb, static_cast<int>(a.shape.dim[0]),
                        static_cast<int>(b.shape.dim[1]), static_cast<int>(a.shape.dim[1]),
                        1.0, static_cast<const double *>(a.data), lda,
                        static_cast<const double *>(b.data), ldb, 0.0,
                        static_cast<double *>(output.data),
                        static_cast<int>(f_output ? a.shape.dim[0] : b.shape.dim[1]));
            restore();
            if (checkpoint) checkpoint(context);
        } catch (...) {
            // This producer is not a published logical root. A synchronous
            // checkpoint/library exception cannot expose its intermediate.
            if (terminal_proof)
                std::fill_n(static_cast<double *>(output.data),output.shape.size(),0.0);
            throw;
        }
        if (mixed_sign && !terminal_proof) {
            // Only ambiguous zero/sign cells need the existing exact fold.
            // Independent root admission excludes every in-graph consumer.
            auto *result = static_cast<double *>(output.data);
            const auto rows = a.shape.dim[0], cols = b.shape.dim[1];
            for (std::size_t outer = 0; outer < (f_output ? cols : rows); ++outer)
                for (std::size_t inner = 0; inner < (f_output ? rows : cols); ++inner) {
                    const auto row = f_output ? inner : outer, col = f_output ? outer : inner;
                    auto &value = result[f_output ? col * rows + row : row * cols + col];
                    if (!std::isfinite(value) || std::abs(value) <= error_bound)
                        value = dot_value(a.row(row), b.column(col));
                }
        }
        // Under RN, canonical +0 accumulation gives +0 for an all-zero dot.
        // The admitted magnitude bounds exclude nonzero underflow/cancellation.
        if (has_negative && has_zero) {
            auto *result = static_cast<double *>(output.data);
            std::replace(result, result + output.shape.size(), 0.0, 0.0);
        }
        audit.isa = terminal_proof ? "accelerate-private-terminal-matmul"
                              : mixed_sign ? "accelerate-bounded-signed-matmul"
                              : has_negative ? "accelerate-sign-consistent-matmul"
                                : "accelerate-nonnegative-matmul";
        audit.accelerate_matmul_calls = 1;
        audit.vector_elements = output.shape.size();
        audit.native_library_workspace_reserve_bytes = reserve * sizeof(double);
        return true;
    }
#else
    (void)p; (void)output; (void)work; (void)audit; (void)context; (void)checkpoint; (void)expanded_geometry; (void)terminal_proof;
#endif
    return false;
}
} // namespace calmetrics_engine::ops
