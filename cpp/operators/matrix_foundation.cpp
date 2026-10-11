#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include "solve_multi_library.hpp"
#include "cholesky_library.hpp"
#include <string_view>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "column_library.hpp"
#include "axis_primitives.hpp"

namespace calmetrics_engine::ops {
namespace {
// Only a consumed private covariance needs its producer to restore the input
// before canonical error replay. Ordinary out-of-place operators replay here.
struct CholeskyReplay : Error { using Error::Error; };
// Dense column prefixes suffer cache-set conflicts at these row pitches.
// Odd sizes measured slower after packing, so do not select by size alone.
bool packed_jacobi(std::size_t n) {
#if defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) return n>=256 && n%128==0;
#endif
    (void)n;
    return false; // Other platforms keep their measured native layout.
}
std::size_t add_size(std::size_t a, std::size_t b) {
    require(a <= static_cast<std::size_t>(PTRDIFF_MAX) &&
            b <= static_cast<std::size_t>(PTRDIFF_MAX) - a, "SHAPE_OVERFLOW");
    return a + b;
}
void array_arg(const Prepared &p, std::size_t i, int lo, int hi, bool numeric = false) {
    require(p.args[i].kind == Kind::number || (!numeric &&
            (p.args[i].kind == Kind::integer || p.args[i].kind == Kind::mask)), "DTYPE_MISMATCH");
    if (p.geometry(i)) require(p.args[i].shape.rank >= lo && p.args[i].shape.rank <= hi, "RANK_MISMATCH");
}
bool scalar_arg(Prepared &p, std::size_t i, double lo, double hi, bool integer = true) {
    array_arg(p, i, 0, 0, true);
    if (!p.payload(i)) return false;
    const auto x = p.args[i].scalar;
    const bool valid=std::isfinite(x) && x >= lo && x <= hi && (!integer || x == std::floor(x));
    if (!valid && p.structure_only) { p.payload_available_mask &= ~(1u<<i); return false; }
    require(valid, "INVALID_PARAMETER");
    return true;
}
bool axis_arg(Prepared &p, std::size_t i, int rank) {
    return scalar_arg(p, i, 0, rank >= 0 ? rank - 1 : 2);
}
void shape_from(Prepared &p, std::size_t i) {
    p.output_geometry_known = p.geometry(i);
    if (p.geometry(i)) p.output_shape = p.args[i].shape;
}
std::ptrdiff_t multiply_stride(std::ptrdiff_t stride, std::size_t count) {
    if (stride == 0 || count == 0) return 0;
    const auto magnitude = stride < 0 ? std::size_t(-(stride + 1)) + 1 : std::size_t(stride);
    require(count <= std::size_t(PTRDIFF_MAX) / magnitude, "STRIDE_OVERFLOW");
    const auto value = static_cast<std::ptrdiff_t>(magnitude * count);
    return stride < 0 ? -value : value;
}
void move_data(Value &v, std::ptrdiff_t offset) {
    if (v.kind == Kind::integer) v.data = static_cast<const std::int64_t *>(v.data) + offset;
    else if (v.kind == Kind::mask) v.data = static_cast<const std::uint8_t *>(v.data) + offset;
    else v.data = static_cast<const double *>(v.data) + offset;
}
void finite(double x) { require(std::isfinite(x), "NON_FINITE_RESULT"); }
void solve_back_substitute(const double *m,double *y,std::size_t n,std::size_t columns) {
    for(std::size_t q=0;q<n;++q) { const auto i=n-1-q;
        finite(m[i*n+i]); require(std::abs(m[i*n+i])>1e-14,"SINGULAR_MATRIX");
        for(std::size_t k=i+1;k<n;++k)
            for(std::size_t j=0;j<columns;++j) y[i*columns+j]-=m[i*n+k]*y[k*columns+j];
        for(std::size_t j=0;j<columns;++j) { y[i*columns+j]/=m[i*n+i]; finite(y[i*columns+j]); }
    }
}
void copy_finite(const Value &a, double *destination) {
    const auto count=a.size();
    bool valid=true;
    if (a.contiguous()) {
        const auto *source=static_cast<const double *>(a.data);
        for (std::size_t i=0;i<count;++i) {
            const auto value=source[i]; destination[i]=value; valid &= std::isfinite(value);
        }
    } else {
        for (std::size_t i=0;i<count;++i) {
            const auto value=a.f(i); destination[i]=value; valid &= std::isfinite(value);
        }
    }
    require(valid,"NON_FINITE_RESULT");
}
// Distinct matrix rows never overlap. Expose that fact to the vectorizer while
// retaining each destination's original sequence of subtraction operations.
void subtract_scaled(double *__restrict destination, const double *__restrict source,
                     std::size_t count, double factor) {
    for (std::size_t j=0;j<count;++j) destination[j]-=factor*source[j];
}

double symmetric_scale(const Value &a) {
    double scale = 0;
    for (std::size_t i = 0; i < a.shape.dim[0]; ++i)
        for (std::size_t j = 0; j < a.shape.dim[1]; ++j) {
            const auto x = a.at(i,j); finite(x);
            scale = std::max(scale, std::abs(x));
        }
    for (std::size_t i = 0; i < a.shape.dim[0]; ++i)
        for (std::size_t j = i + 1; j < a.shape.dim[1]; ++j)
            require(a.at(i,j) == a.at(j,i), "DOMAIN_ERROR");
    return scale;
}
void copy_element(Output &out, std::size_t destination, const Value &source, std::size_t index) {
    if (source.kind == Kind::integer) out.set_integer(destination, source.i(index));
    else if (source.kind == Kind::mask) out.set_mask(destination, source.u(index));
    else out.set(destination, source.f(index));
}
// Decode the non-reduced axes once per line, not once per element. Signed
// element strides retain negative/zero-stride input contracts.
std::ptrdiff_t line_offset(const Value &a, std::size_t axis,
                           std::size_t outer, std::size_t inner) {
    std::ptrdiff_t offset=0;
    for(int d=int(axis)-1;d>=0;--d) {
        offset+=static_cast<std::ptrdiff_t>(outer%a.shape.dim[d])*a.stride[d];
        outer/=a.shape.dim[d];
    }
    for(int d=a.shape.rank-1;d>int(axis);--d) {
        offset+=static_cast<std::ptrdiff_t>(inner%a.shape.dim[d])*a.stride[d];
        inner/=a.shape.dim[d];
    }
    return offset;
}
template<class T>
void take_lines(Output &out, const Value &a, const Value &indices,
                std::size_t axis, std::size_t outer, std::size_t inner) {
    auto *destination=static_cast<T *>(out.data);
    const auto *source=static_cast<const T *>(a.data);
    for(std::size_t o=0;o<outer;++o) for(std::size_t j=0;j<inner;++j) {
        const auto base=line_offset(a,axis,o,j);
        for(std::size_t i=0;i<indices.size();++i)
            destination[(o*indices.size()+i)*inner+j]=
                source[base+static_cast<std::ptrdiff_t>(indices.i(i))*a.stride[axis]];
    }
}
template<bool Packed, bool ValuesOnly = false, bool Split = false>
void symmetric_jacobi(const Prepared &p, Output &out, Workspace &work, Audit &audit,
                      double *vectors = nullptr) {
    const auto &a=p.args[0]; const auto n=a.shape.dim[0];
    auto *y=static_cast<double *>(out.data);
    const auto scale=symmetric_scale(a); const auto width=ValuesOnly ? std::size_t(1) : n+1;
    const auto result=[&](std::size_t i,std::size_t k) -> double & {
        if constexpr (Split) return k ? vectors[i*n+k-1] : y[i];
        else return y[i*width+k];
    };
    auto *m=work.doubles.data();
    const auto row_start=[n](std::size_t i) {
        if constexpr (Split) return i*(n+1);
        else if constexpr (Packed) return i*n-(i ? i*(i-1)/2 : 0);
        else return i*n;
    };
    const auto at=[&](std::size_t i,std::size_t j) -> double & {
        return m[row_start(i)+j-((Packed && !Split) ? i : 0)];
    };
    MatrixRotation rotation(Packed);
    const auto rotate=[&](double *x,double *z,std::size_t count,double c,double s) {
        if constexpr (Packed) rotation.apply(x,z,count,c,s,audit);
        else for(std::size_t k=0;k<count;++k) {
            const auto u=x[k],v=z[k]; x[k]=c*u-s*v; z[k]=s*u+c*v;
        }
    };
    for (std::size_t i=0;i<n;++i) {
        if constexpr (!ValuesOnly) result(i,0)=0;
        for (std::size_t j=0;j<n;++j) {
            if (!Packed || j>=i) at(i,j)=scale==0 ? 0 : a.at(i,j)/scale;
            if constexpr (!ValuesOnly) result(i,j+1)=i==j ? 1 : 0;
        }
    }
    // A deferred producer's dead arena may become the projection destination.
    // Finish copying its source before writing any projected value.
    if constexpr (ValuesOnly) std::fill_n(y,n,0.0);
    audit.algorithm_copy_bytes=p.scratch_doubles*sizeof(double);
    bool lower_state=false;
    const auto tolerance=p.args[1].scalar;
    const auto off_diagonal=[&] {
        double maximum=0;
        for (std::size_t i=0;i<n;++i) for (std::size_t j=i+1;j<n;++j)
            maximum=std::max(maximum,std::abs(Split && lower_state ? at(j,i) : at(i,j)));
        return maximum;
    };
    for (std::size_t sweep=0;sweep<std::size_t(p.args[2].scalar) && off_diagonal()>tolerance;++sweep) {
        if constexpr (Split) {
            if (lower_state) {
                for (std::size_t i=0;i<n;++i) for (std::size_t j=i+1;j<n;++j) at(i,j)=at(j,i);
                audit.algorithm_copy_bytes+=n*(n-1)/2*sizeof(double);
                lower_state=false;
            }
        }
        for (std::size_t i=0;i<n;++i) {
          for (std::size_t j=i+1;j<n;++j) {
            const auto cross=at(i,j); if (std::abs(cross)<=tolerance) continue;
            const auto tau=(at(j,j)-at(i,i))/(2*cross);
            const auto t=std::copysign(1.0/(std::abs(tau)+std::hypot(1.0,tau)),tau);
            const auto c=1.0/std::sqrt(1+t*t), s=t*c;
            at(i,i)-=t*cross; at(j,j)+=t*cross; at(i,j)=0;
            // Only the upper triangle is read by subsequent rotations.
            // Keep rotation order, but remove mirrored column writes and
            // separate the contiguous tail from the strided prefixes.
            if constexpr (Split) {
                if (i) rotate(&at(i,0),&at(j,0),i,c,s);
            } else for (std::size_t k=0;k<i;++k) {
                    const auto u=at(k,i),v=at(k,j);
                    at(k,i)=c*u-s*v; at(k,j)=s*u+c*v;
                }
            for (std::size_t k=i+1;k<j;++k) {
                const auto u=at(i,k),v=at(k,j);
                at(i,k)=c*u-s*v; at(k,j)=s*u+c*v;
            }
            // The upper-triangle tail and eigenvectors are contiguous and
            // disjoint. Eigen Map borrows them without packing or workspace.
            const auto count=n-j-1;
            if (count) rotate(&at(i,j+1),&at(j,j+1),count,c,s);
            if constexpr (!ValuesOnly) {
                if constexpr (Split) rotate(vectors+i*n,vectors+j*n,n,c,s);
                else rotate(y+i*width+1,y+j*width+1,n,c,s);
            }
          }
          if constexpr (Split) for (std::size_t j=i+1;j<n;++j) at(j,i)=at(i,j);
        }
        if constexpr (Split) {
            audit.algorithm_copy_bytes+=n*(n-1)/2*sizeof(double);
            lower_state=true;
        }
    }
    require(off_diagonal()<=tolerance,"NON_CONVERGENCE");
    for (std::size_t i=0;i<n;++i) {
        result(i,0)=at(i,i)*scale; finite(result(i,0));
        if constexpr (!ValuesOnly) {
            std::size_t pivot=1;
            for (std::size_t j=2;j<width;++j)
                if (std::abs(result(i,j))>std::abs(result(i,pivot))) pivot=j;
            if (result(i,pivot)<0) for (std::size_t j=1;j<width;++j) result(i,j)=-result(i,j);
        }
    }
    if constexpr (!ValuesOnly) {
        if (n>=128) {
            // The original row number makes equal-value ordering stable.
            auto *order=work.indices.data();
            std::iota(order,order+n,std::size_t(0));
            std::sort(order,order+n,[&](std::size_t i,std::size_t j) {
                return result(i,0)<result(j,0) || (result(i,0)==result(j,0) && i<j);
            });
            // The scaled matrix is dead after all eigenvalues are extracted.
            for (std::size_t i=0;i<n;++i) {
                if (order[i]==i) continue;
                if constexpr (Split) {
                    m[0]=y[i]; std::copy_n(vectors+i*n,n,m+1);
                    audit.algorithm_copy_bytes+=width*sizeof(double);
                } else std::copy_n(y+i*width,width,m);
                auto current=i;
                while (order[current]!=i) {
                    const auto source=order[current];
                    if constexpr (Split) {
                        y[current]=y[source]; std::copy_n(vectors+source*n,n,vectors+current*n);
                        audit.algorithm_copy_bytes+=width*sizeof(double);
                    } else std::copy_n(y+source*width,width,y+current*width);
                    order[current]=current;current=source;
                }
                if constexpr (Split) {
                    y[current]=m[0]; std::copy_n(m+1,n,vectors+current*n);
                    audit.algorithm_copy_bytes+=width*sizeof(double);
                } else std::copy_n(m,width,y+current*width);
                order[current]=current;
            }
            return;
        }
    }
    // Stable insertion sorting keeps equal eigenvalues in their deterministic sweep order.
    for (std::size_t i=1;i<n;++i) for (std::size_t j=i;j>0 && result(j,0)<result(j-1,0);--j)
        for (std::size_t k=0;k<width;++k) std::swap(result(j,k),result(j-1,k));
}
} // namespace

std::size_t symmetric_eigh_split_scratch_doubles(std::size_t n) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (n<512 || n%256) return 0;
        // The padded row pitch avoids the measured power-of-two cache conflict.
        const auto count=matrix_shape(n,add_size(n,1)).size();
        require(count<=static_cast<std::size_t>(PTRDIFF_MAX)/sizeof(double),"SHAPE_OVERFLOW");
        return count;
    }
#endif
    (void)n;
    return 0;
}

std::size_t cholesky_scratch_doubles(std::size_t n) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (n<1024) return 0;
        constexpr auto max=static_cast<std::size_t>(PTRDIFF_MAX)/sizeof(double);
        require(n<=(max-128)/16-17,"SHAPE_OVERFLOW");
        return 16*(n+17)+128;
    }
#endif
    (void)n;
    return 0;
}

void prepare_foundation(Prepared &p) {
    const auto op = p.spec->op;
    auto &a = p.args;
    if (op <= Op::eigenvectors) {
        array_arg(p,0,2,2,true);
        if (op == Op::eigenvalues || op == Op::eigenvectors) {
            p.output_geometry_known = p.geometry(0);
            if (!p.geometry(0)) return;
            const auto n = a[0].shape.dim[0];
            require(a[0].shape.dim[1] == add_size(n,1), "SHAPE_MISMATCH");
            p.output_shape = op == Op::eigenvalues ? vector_shape(n) : matrix_shape(n,n);
            if (p.payload(0)) {
                p.borrowed = true; p.view = a[0]; p.view.shape = p.output_shape;
                if (op == Op::eigenvalues) p.view.stride = {a[0].stride[0],1,1};
                else {
                    if (n) move_data(p.view,a[0].stride[1]);
                    p.view.stride = {a[0].stride[1],a[0].stride[0],1};
                }
            }
            return;
        }
        if (op == Op::solve_multi || op == Op::triangular_solve)
            array_arg(p,1,op == Op::solve_multi ? 2 : 1,2,true);
        if (op == Op::symmetric_eigh) {
            if (p.count < 2) a[1] = Value::number(1e-12);
            if (p.count < 3) a[2] = Value::number(32);
            scalar_arg(p,1,1e-15,1e-3,false); scalar_arg(p,2,1,128);
        }
        if (op == Op::triangular_solve) {
            if (p.count < 3) a[2] = Value::number(1);
            if (p.count < 4) a[3] = Value::number(0);
            scalar_arg(p,2,0,1); scalar_arg(p,3,0,1);
        }
        if (p.geometry(0)) require(a[0].shape.dim[0] == a[0].shape.dim[1], "SHAPE_MISMATCH");
        if (op == Op::solve_multi || op == Op::triangular_solve) {
            if (p.geometry(0) && p.geometry(1))
                require(a[0].shape.dim[0] == a[1].shape.dim[0], "SHAPE_MISMATCH");
            shape_from(p,1);
        } else shape_from(p,0);
        if (op == Op::symmetric_eigh && p.geometry(0)) {
            const auto n = a[0].shape.dim[0];
            p.output_shape = matrix_shape(n,add_size(n,1));
            p.scratch_doubles = packed_jacobi(n) ? p.output_shape.size()/2 : a[0].size();
            p.scratch_indices = n>=128 ? n : 0;
        } else if (op == Op::solve_multi && p.geometry(0))
            p.scratch_doubles = a[0].size();
        else if (op == Op::cholesky && p.geometry(0))
            p.scratch_doubles = cholesky_scratch_doubles(a[0].shape.dim[0]);
        return;
    }
    const bool stacking = op == Op::stack;
    array_arg(p,0,stacking ? 0 : 1,stacking ? 2 : 3);
    p.output_kind = a[0].kind;
    if (op == Op::sum_axis || op == Op::mean_axis) array_arg(p,0,1,3,true);
    const int rank = p.geometry(0) ? a[0].shape.rank : -1;
    const bool binary = op == Op::concatenate || stacking || op == Op::broadcast_like;
    if (binary) {
        array_arg(p,1,stacking ? 0 : 1,stacking ? 2 : 3);
        require(a[0].kind == a[1].kind,"DTYPE_MISMATCH");
        if (p.geometry(0) && p.geometry(1) && op != Op::broadcast_like)
            require(a[0].shape.rank == a[1].shape.rank,"RANK_MISMATCH");
    }
    if (op == Op::take_axis) {
        require(a[1].kind == Kind::integer,"DTYPE_MISMATCH");
        if (p.geometry(1)) require(a[1].shape.rank == 1,"RANK_MISMATCH");
        if (p.payload(1)) for (std::size_t i=0;i<a[1].size();++i)
            require(a[1].i(i)>=0,"INDEX_OUT_OF_BOUNDS");
    }
    bool known = p.geometry(0);
    std::size_t axis=0, second=0;
    if (op != Op::broadcast_like) {
        const auto ai = (op == Op::take_axis || binary) ? 2u : 1u;
        const auto axis_known = axis_arg(p,ai,rank < 0 ? -1 : rank + (stacking ? 1 : 0));
        known = known && axis_known;
        if (axis_known) axis = static_cast<std::size_t>(a[ai].scalar);
    }
    if (op == Op::slice_axis) {
        bool config = true;
        for (std::size_t i=2;i<5;++i)
            config = scalar_arg(p,i,i==4 ? 1 : 0,9007199254740991.0) && config;
        known = known && config;
    }
    if (op == Op::swap_axes) {
        const auto valid = axis_arg(p,2,rank);
        known = known && valid;
        if (valid) second = static_cast<std::size_t>(a[2].scalar);
    }
    if (binary || op == Op::take_axis) known = known && p.geometry(1);
    p.output_geometry_known = known;
    if (!known) return;
    p.output_shape = a[0].shape;
    if (op == Op::take_axis) {
        if (p.payload(1)) for (std::size_t i=0;i<a[1].size();++i)
            require(std::uint64_t(a[1].i(i)) < a[0].shape.dim[axis],"INDEX_OUT_OF_BOUNDS");
        p.output_shape.dim[axis]=a[1].size();
    } else if (op == Op::slice_axis) {
        const auto start=std::size_t(a[2].scalar), stop=std::size_t(a[3].scalar), step=std::size_t(a[4].scalar);
        require(start<=stop && stop<=a[0].shape.dim[axis],"INDEX_OUT_OF_BOUNDS");
        p.output_shape.dim[axis]=(stop-start)/step+((stop-start)%step != 0);
        if (p.payload(0)) {
            p.borrowed=true; p.view=a[0]; p.view.shape=p.output_shape;
            if (p.view.size()) move_data(p.view,multiply_stride(a[0].stride[axis],start));
            p.view.stride[axis]=p.view.shape.dim[axis]>1 ? multiply_stride(a[0].stride[axis],step) : 0;
        }
    } else if (op == Op::concatenate || stacking) {
        for (int i=0;i<rank;++i)
            if (stacking || std::size_t(i)!=axis)
                require(a[0].shape.dim[i]==a[1].shape.dim[i],"SHAPE_MISMATCH");
        if (stacking) {
            p.output_shape.rank=rank+1;
            for (int i=rank;i>=0;--i)
                p.output_shape.dim[i]=std::size_t(i)==axis ? 2 : a[0].shape.dim[i-(std::size_t(i)>axis)];
        } else p.output_shape.dim[axis]=add_size(a[0].shape.dim[axis],a[1].shape.dim[axis]);
    } else if (op == Op::broadcast_like) {
        require(rank<=a[1].shape.rank,"RANK_MISMATCH");
        p.output_shape=a[1].shape;
        p.view=a[0]; p.view.shape=p.output_shape; p.view.stride={0,0,0};
        const auto offset=a[1].shape.rank-rank;
        for (int i=0;i<rank;++i) {
            require(a[0].shape.dim[i]==1 || a[0].shape.dim[i]==a[1].shape.dim[i+offset],"SHAPE_MISMATCH");
            p.view.stride[i+offset]=a[0].shape.dim[i]==1 ? 0 : a[0].stride[i];
        }
        p.borrowed=p.payload(0);
    } else if (op == Op::swap_axes) {
        std::swap(p.output_shape.dim[axis],p.output_shape.dim[second]);
        if (p.payload(0)) {
            p.borrowed=true; p.view=a[0]; p.view.shape=p.output_shape;
            std::swap(p.view.stride[axis],p.view.stride[second]);
        }
    } else {
        --p.output_shape.rank;
        for (int i=int(axis);i<p.output_shape.rank;++i) p.output_shape.dim[i]=a[0].shape.dim[i+1];
        for (int i=p.output_shape.rank;i<3;++i) p.output_shape.dim[i]=1;
    }
}

static void solve_multi_execute(const Prepared &p, Output &out, Workspace &work,
                                Isa isa, Audit &audit, double *private_coefficients = nullptr) {
    const auto &a=p.args[0], &b=p.args[1];
    const auto n=a.shape.dim[0], columns=b.shape.rank==1 ? 1 : b.shape.dim[1];
    auto *y=static_cast<double *>(out.data);
    copy_finite(b,y);
    audit.algorithm_copy_bytes=b.size()*sizeof(double);
    auto *m=private_coefficients ? private_coefficients : work.doubles.data();
    if (private_coefficients) {
        // Preserve the full A scan after B validation without a second N² buffer.
        bool valid=true;
        for (std::size_t i=0;i<a.size();++i) valid &= std::isfinite(m[i]);
        require(valid,"NON_FINITE_RESULT");
    } else {
        copy_finite(a,m);
        audit.algorithm_copy_bytes+=a.size()*sizeof(double);
    }
    if (isa == Isa::scalar || !library_eliminate_multi(m,y,n,columns,audit))
    for (std::size_t k=0;k<n;++k) {
        auto pivot=k;
        for (std::size_t i=k+1;i<n;++i)
            if (std::abs(m[i*n+k])>std::abs(m[pivot*n+k])) pivot=i;
        require(std::abs(m[pivot*n+k])>1e-14,"SINGULAR_MATRIX");
        if (pivot!=k) {
            for (std::size_t j=k;j<n;++j) std::swap(m[k*n+j],m[pivot*n+j]);
            for (std::size_t j=0;j<columns;++j) std::swap(y[k*columns+j],y[pivot*columns+j]);
        }
        for (std::size_t i=k+1;i<n;++i) {
            const double factor=m[i*n+k]/m[k*n+k];
            subtract_scaled(m+i*n+k+1,m+k*n+k+1,n-k-1,factor);
            subtract_scaled(y+i*columns,y+k*columns,columns,factor);
        }
    }
    solve_back_substitute(m,y,n,columns);
}

template <bool f_cholesky>
static void matrix_foundation_execute_impl(const Prepared &p, Output &out, Workspace &work,
                                           Isa isa, Audit &audit) {
    const auto op=p.spec->op; const auto &a=p.args[0]; const auto &b=p.args[1];
    const auto n=a.shape.dim[0];
    auto *y=static_cast<double *>(out.data);
    if (op==Op::cholesky) {
        symmetric_scale(a);
        const bool inplace=a.data==out.data && a.contiguous();
        const auto finish_row=[&](std::size_t i,std::size_t first) {
            auto *row=y+i*n;
            // Upper cells mirror finished columns; lower cells keep k order.
            for (std::size_t k=first;k<i;++k) {
                finite(row[k]);
                const auto factor=row[k]/y[k*n+k]; finite(factor);
                row[k]=factor; y[k*n+i]=factor;
                subtract_scaled(row+k+1,y+k*n+k+1,i-k,factor);
            }
            finite(row[i]); require(row[i]>0,"DOMAIN_ERROR"); row[i]=std::sqrt(row[i]);
        };
        bool packed=false;
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
        if (isa==Isa::automatic && p.scratch_doubles) {
            require(work.doubles.size()>=p.scratch_doubles,"WORKSPACE_TOO_SMALL");
            try {
                cholesky_detail::factor_rows(a,y,work,finish_row);
                packed=true;
                audit.isa="eigen-array-panel-cholesky";
            } catch (const Error &error) {
                // Independent later rows may overflow before an earlier bad
                // pivot is visited. Replay only numerical failures in row order.
                const std::string_view code(error.what());
                if (code!="NON_FINITE_RESULT" && code!="DOMAIN_ERROR") throw;
                if (inplace) throw CholeskyReplay(error.what());
            }
        }
#endif
        if (!packed) for (std::size_t i=0;i<n;++i) {
            if (!inplace) for (std::size_t j=0;j<=i;++j) y[i*n+j]=a.at(i,j);
            finish_row(i,0);
        }
        // Completed factors are mirrored bit-for-bit. Keeping the C upper
        // triangle instead publishes the same lower factor through F strides.
        for (std::size_t i=0;i<n;++i) {
            if (f_cholesky) std::fill(y+i*n,y+i*n+i,0.0);
            else std::fill(y+i*n+i+1,y+(i+1)*n,0.0);
        }
        audit.algorithm_copy_bytes=n*(n+1)/2*sizeof(double)*(packed ? 2 : inplace ? 0 : 1);
        return;
    }
    if (op==Op::solve_multi) return solve_multi_execute(p,out,work,isa,audit);
    if (op==Op::triangular_solve) {
        copy_finite(b,y);
        audit.algorithm_copy_bytes=b.size()*sizeof(double);
#if defined(__clang__)
        [[clang::musttail]]
#endif
        return triangular_solve_finish(p,out,work,isa,audit);
    }
    if (op==Op::symmetric_eigh) {
        if (packed_jacobi(n)) symmetric_jacobi<true>(p,out,work,audit);
        else symmetric_jacobi<false>(p,out,work,audit);
        return;
    }
    // Empty selected output may still have very large non-selected extents.
    // Structural validation has already run; do not iterate empty logical lines.
    if(out.shape.size()==0) return;
    const auto axis=std::size_t(p.args[(op==Op::take_axis || op==Op::concatenate || op==Op::stack) ? 2 : 1].scalar);
    std::size_t inner=1, outer=1;
    for (int i=int(axis)+1;i<a.shape.rank;++i) inner*=a.shape.dim[i];
    for (std::size_t i=0;i<axis;++i) outer*=a.shape.dim[i];
    if (op==Op::stack) {
        inner=1;
        for (int i=int(axis);i<a.shape.rank;++i) inner*=a.shape.dim[i];
        for (std::size_t o=0;o<outer;++o) for (std::size_t source=0;source<2;++source)
            for (std::size_t j=0;j<inner;++j)
                copy_element(out,(o*2+source)*inner+j,p.args[source],o*inner+j);
    } else if (op==Op::take_axis) {
        if(a.kind==Kind::integer) take_lines<std::int64_t>(out,a,b,axis,outer,inner);
        else if(a.kind==Kind::mask) take_lines<std::uint8_t>(out,a,b,axis,outer,inner);
        else take_lines<double>(out,a,b,axis,outer,inner);
    } else if (op==Op::concatenate) {
        const auto left=a.shape.dim[axis], right=b.shape.dim[axis];
        for (std::size_t o=0;o<outer;++o) {
            for (std::size_t j=0;j<left*inner;++j) copy_element(out,o*(left+right)*inner+j,a,o*left*inner+j);
            for (std::size_t j=0;j<right*inner;++j) copy_element(out,(o*(left+right)+left)*inner+j,b,o*right*inner+j);
        }
    } else if (op==Op::sum_axis || op==Op::mean_axis) {
        const auto count=a.shape.dim[axis];
        if (op==Op::mean_axis && out.shape.size()) require(count>0,"INSUFFICIENT_SAMPLE");
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
        if (axis==0 && isa!=Isa::scalar &&
            (op==Op::mean_axis ? eigen_ordered_rows<Op::mean_axis>(a,double(count),out,audit) :
                                eigen_ordered_rows<Op::sum_axis>(a,1.,out,audit))) return;
#endif
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
        if (axis==0 && a.shape.rank==2 && isa != Isa::scalar && count>=32 && a.shape.dim[1]>=8 && (a.stride[0]==1 || a.stride[0]==-1) && a.stride[1]>=std::ptrdiff_t(count)) {
          if (__builtin_available(macOS 15.0, *)) {
            if (op==Op::mean_axis) eigen_ordered_columns<Op::mean_axis>(a,double(count),out,audit);
            else eigen_ordered_columns<Op::sum_axis>(a,1.,out,audit);
            return;
          }
        }
#endif
        if (axis==0 && a.shape.rank==2 && a.contiguous()) {
            const auto *source=static_cast<const double *>(a.data);
            for (std::size_t first=0;first<inner;first+=8) {
                const auto lanes=std::min<std::size_t>(8,inner-first);
                double totals[8]{};
                for (std::size_t k=0;k<count;++k)
                    for (std::size_t j=0;j<lanes;++j) totals[j]+=source[k*inner+first+j];
                for (std::size_t j=0;j<lanes;++j)
                    out.set(first+j,op==Op::mean_axis ? totals[j]/double(count) : totals[j]);
            }
            return;
        }
        for (std::size_t o=0;o<outer;++o) for (std::size_t j=0;j<inner;++j) {
            double total=0;
            const auto base=line_offset(a,axis,o,j);
            const auto *source=static_cast<const double *>(a.data);
            for (std::size_t k=0;k<count;++k)
                total=axis_path_detail::ordered_sum_step(total,
                    source[base+static_cast<std::ptrdiff_t>(k)*a.stride[axis]]);
            out.set(o*inner+j,op==Op::mean_axis ? total/double(count) : total);
        }
    } else throw Error("UNKNOWN_OPERATOR");
}
void matrix_foundation_execute(const Prepared &p, Output &out, Workspace &work,
                               Isa isa, Audit &audit) {
    matrix_foundation_execute_impl<false>(p,out,work,isa,audit);
}

bool solve_multi_pair(const Prepared &first,const Prepared &second,
                      Output &out1,Output &out2,Workspace &work,Audit &audit) {
    const auto n=first.args[0].shape.dim[0];
    if (!library_eliminate_available(n)) return false;
    require(first.spec->op==Op::solve_multi && second.spec->op==Op::solve_multi &&
        out1.order==MemoryOrder::c && out2.order==MemoryOrder::c &&
        out1.shape==first.output_shape && out2.shape==second.output_shape &&
        first.args[0].shape==second.args[0].shape,"INVALID_PHYSICAL_PAIR");
    audit={};
    work.ensure(first.scratch_doubles,first.scratch_indices);
    auto *m=work.doubles.data();
    const auto columns1=first.args[1].shape.dim[1],columns2=second.args[1].shape.dim[1];
    auto *y1=static_cast<double *>(out1.data),*y2=static_cast<double *>(out2.data);
    try {
        copy_finite(first.args[1],y1);
        copy_finite(first.args[0],m);
        copy_finite(second.args[1],y2);
        const std::array<SolveRhsSpan,2> spans{{{y1,columns1},{y2,columns2}}};
        library_eliminate_spans(m,spans,n,audit);
        // Complete the first result before checking the second's back solve.
        solve_back_substitute(m,y1,n,columns1);
        solve_back_substitute(m,y2,n,columns2);
    } catch (const Error &error) {
        const std::string_view code(error.what());
        if (code!="SINGULAR_MATRIX" && code!="NON_FINITE_RESULT") throw;
        // Restore the still-unexecuted root before original error/output replay.
        if (out2.shape.size()) std::fill_n(y2,out2.shape.size(),0.0);
        return false;
    }
    audit.algorithm_copy_bytes=(first.args[0].size()+first.args[1].size()+second.args[1].size())*sizeof(double);
    audit.scratch_bytes=first.scratch_doubles*sizeof(double)+first.scratch_indices*sizeof(std::size_t);
    return true;
}

void solve_multi_private(const Prepared &p, Output &out, Workspace &work,
                         Audit &audit, double *coefficients) {
    // Only the executor's certified, dead, materialized arena may enter here.
    // solve_multi has no mask/index data-structure validation in registry execute.
    require(p.spec && p.spec->op==Op::solve_multi && p.count==2 &&
        !p.borrowed && coefficients && p.args[0].data==coefficients &&
        p.args[0].contiguous() && out.order==MemoryOrder::c &&
        out.kind==p.output_kind && out.shape==p.output_shape && !p.scratch_indices,
        "PRIVATE_SOLVE_CONTRACT");
    require(out.shape.size()==0 || out.data,"NULL_OUTPUT");
    audit={};
    work.ensure(0,0);
    solve_multi_execute(p,out,work,Isa::automatic,audit,coefficients);
}

void cholesky_f_root(const Prepared &p, Output &out, Workspace &work,
                     Isa isa, Audit &audit) {
    // This private entry does not relax registry/direct out order admission.
    // Cholesky prepare accepts only a numeric matrix, so registry's mask/index
    // data-structure checks have no work for this operation.
    require(p.spec && p.spec->op==Op::cholesky && p.count==1 &&
        !p.borrowed && out.order==MemoryOrder::f && out.kind==p.output_kind &&
        out.shape==p.output_shape,"OUTPUT_MISMATCH");
    require(out.shape.size()==0 || out.data,"NULL_OUTPUT");
    require(supports_isa(isa),"UNSUPPORTED_ISA");
    audit={};
    const auto doubles=isa==Isa::automatic ? p.scratch_doubles : 0;
    work.ensure(doubles,p.scratch_indices);
    audit.scratch_bytes=doubles*sizeof(double)+p.scratch_indices*sizeof(std::size_t);
    matrix_foundation_execute_impl<true>(p,out,work,isa,audit);
}

void covariance_cholesky_pair(const Prepared &first, const Prepared &second,
                              Output &out, Workspace &work, Audit &audit,
                              bool independent_root) {
    require(first.spec->op==Op::covariance && first.count==1 &&
        second.spec->op==Op::cholesky && second.count==1 &&
        first.output_shape==second.output_shape &&
        (out.order==MemoryOrder::c || (out.order==MemoryOrder::f && independent_root)),
        "INVALID_PHYSICAL_PAIR");
    audit={};
    // Both numerical backends and error replay keep their original C layout.
    // The independent result is not exposed until the successful F conversion.
    auto physical=out;
    physical.order=MemoryOrder::c;
    if (!(independent_root &&
          try_accelerate_covariance_cholesky(first,physical,work,audit))) {
        const auto collect=[&](const Audit &current) {
            audit.scratch_bytes=std::max(audit.scratch_bytes,current.scratch_bytes);
            audit.eigen_covariance_calls+=current.eigen_covariance_calls;
            audit.algorithm_copy_bytes+=current.algorithm_copy_bytes;
            audit.vector_elements+=current.vector_elements;
        };
        const auto produce=[&] {
            Audit current;
            execute(first,physical,work,Isa::automatic,current);
            collect(current);
        };
        produce();
        auto consumer=second;
        consumer.args[0].data=physical.data;
        consumer.args[0].set_contiguous_strides();
        Audit current;
        try {
            execute(consumer,physical,work,Isa::automatic,current);
        } catch (const CholeskyReplay &) {
            collect(current);
            produce();
            execute(consumer,physical,work,Isa::scalar,current);
        }
        collect(current);
    }
    if (out.order==MemoryOrder::f) {
        const auto n=first.output_shape.dim[0];
        auto *factor=static_cast<double *>(out.data);
        constexpr std::size_t tile=32;
        for (std::size_t row=0;row<n;row+=tile)
            for (std::size_t column=0;column<=row;column+=tile)
                for (std::size_t i=row;i<std::min(row+tile,n);++i)
                    for (std::size_t j=column;j<std::min(column+tile,i);++j)
                        std::swap(factor[i*n+j],factor[j*n+i]);
        // Two destination writes per off-diagonal swap; diagonal is unchanged.
        audit.algorithm_copy_bytes+=(first.output_shape.size()-n)*sizeof(double);
    }
}

void symmetric_eigh_values_pair(const Prepared &first, Output &out,
                                Workspace &work, Audit &audit) {
    require(first.spec && first.spec->op==Op::symmetric_eigh &&
        out.kind==Kind::number && out.shape==vector_shape(first.args[0].shape.dim[0]) &&
        (out.shape.size()==0 || out.data), "INVALID_PHYSICAL_PAIR");
    audit={};
    work.ensure(first.scratch_doubles,0);
    audit.scratch_bytes=first.scratch_doubles*sizeof(double);
    if (packed_jacobi(first.args[0].shape.dim[0]))
        symmetric_jacobi<true,true>(first,out,work,audit);
    else symmetric_jacobi<false,true>(first,out,work,audit);
}

void symmetric_eigh_split_pair(const Prepared &first, Output &values, Output &vectors,
                               Workspace &work, Audit &audit) {
    require(first.spec && first.spec->op==Op::symmetric_eigh,"INVALID_PHYSICAL_PAIR");
    const auto n=first.args[0].shape.dim[0];
    const auto capacity=symmetric_eigh_split_scratch_doubles(n);
    require(capacity && values.kind==Kind::number && values.shape==vector_shape(n) && values.data &&
        vectors.kind==Kind::number && vectors.shape==matrix_shape(n,n) && vectors.data &&
        vectors.order==MemoryOrder::f,"INVALID_PHYSICAL_PAIR");
    audit={};
    work.ensure(capacity,first.scratch_indices);
    audit.scratch_bytes=capacity*sizeof(double)+first.scratch_indices*sizeof(std::size_t);
    symmetric_jacobi<true,false,true>(first,values,work,audit,static_cast<double *>(vectors.data));
}

void take_reduce(const Prepared &take, const Prepared &reduce, Output &out) {
    const auto &a=take.args[0], &indices=take.args[1];
    const auto axis=std::size_t(take.args[2].scalar), count=indices.size();
    if (!out.shape.size()) return;
    const bool mean=reduce.spec->op==Op::mean_axis;
    if (mean) require(count>0,"INSUFFICIENT_SAMPLE");
    std::size_t outer=1,inner=1;
    for (std::size_t j=0;j<axis;++j) outer*=a.shape.dim[j];
    for (int j=int(axis)+1;j<a.shape.rank;++j) inner*=a.shape.dim[j];
    const auto *source=static_cast<const double *>(a.data);
    for (std::size_t o=0;o<outer;++o) for (std::size_t j=0;j<inner;++j) {
        const auto base=line_offset(a,axis,o,j);
        double total=0;
        for (std::size_t k=0;k<count;++k)
            total+=source[base+std::ptrdiff_t(indices.i(k))*a.stride[axis]];
        out.set(o*inner+j,mean ? total/double(count) : total);
    }
}
} // namespace calmetrics_engine::ops
