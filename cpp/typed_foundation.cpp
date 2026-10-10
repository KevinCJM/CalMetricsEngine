#include "calmetrics_engine/typed_ir.hpp"
#include <algorithm>
#include <cmath>

namespace calmetrics_engine::typed {
namespace {
void check(bool test, const char *message) { if (!test) throw Error(message); }
ValueType shaped(ValueType out, std::vector<std::string> axes, std::vector<std::string> shape) {
    out.axes=std::move(axes); out.shape=std::move(shape); out.record_tag.clear(); out.fields.clear();
    out.kind=out.axes.empty() ? (out.dtype==DType::float64 ? ValueKind::scalar : ValueKind::value)
        : out.axes.size()==3 ? ValueKind::tensor : out.axes.size()==2 ? ValueKind::matrix
        : out.axes[0]=="time" ? ValueKind::series : ValueKind::vector;
    out.validate(); return out;
}
}
ValueType infer_foundation(const ops::Spec &spec, const std::vector<ValueType> &v,
                          const std::vector<double> &constants) {
    using O=ops::Op; const auto op=spec.op;
    auto out=v[0];
    const auto literal=[&](std::size_t i, double maximum=2) {
        check(i<v.size() && v[i].is_scalar() && v[i].is_numeric() &&
              (v[i].semantic_dimension=="dimensionless" || v[i].semantic_dimension=="count"),
              "TYPE_MISMATCH: axis/layout parameter must be a numeric scalar");
        check(i<constants.size() && std::isfinite(constants[i]) && constants[i]>=0 &&
              constants[i]<=maximum && constants[i]==std::floor(constants[i]),
              "CONSTANT_LAYOUT_REQUIRED: axis/slice parameters must be nonnegative integer literals");
        return std::size_t(constants[i]);
    };
    if (op==O::eigenvalues || op==O::eigenvectors) {
        check(out.kind==ValueKind::matrix && out.record_tag=="symmetric_eigh" &&
              out.dtype==DType::float64 && out.fields==std::vector<std::string>{"eigenvalues","eigenvectors"},
              "TYPE_MISMATCH: expected symmetric_eigh decomposition");
        if (op==O::eigenvalues) return shaped(out,{"component"},{out.shape[0]});
        auto result=shaped(out,{out.axes[1],"component"},{out.shape[0],out.shape[0]});
        return result.with_semantics("dimensionless");
    }
    if (op<=O::symmetric_eigh) {
        check(out.kind==ValueKind::matrix && out.is_numeric(),"TYPE_MISMATCH: expected float64 matrix");
        check(out.shape[0]==out.shape[1],"SHAPE_MISMATCH: expected square matrix");
        if (op==O::cholesky || op==O::symmetric_eigh) {
            check(out.axes[0]==out.axes[1],"AXIS_MISMATCH: symmetric matrix requires matching axes");
            if (op==O::cholesky) {
                auto semantic=out.semantic_dimension;
                if (semantic.rfind("squared:",0)==0) semantic=semantic.substr(8);
                else if (semantic!="dimensionless") semantic="derived:sqrt:"+semantic;
                return out.with_semantics(semantic);
            }
            for (std::size_t i=1;i<v.size();++i)
                check(v[i].is_scalar() && v[i].is_numeric() && v[i].semantic_dimension=="dimensionless",
                      "TYPE_MISMATCH: eigensolver options must be dimensionless scalars");
            out=shaped(out,{"component",v[0].axes[0]},{out.shape[0],"("+out.shape[0]+"+1)"});
            out.record_tag="symmetric_eigh"; out.fields={"eigenvalues","eigenvectors"}; return out;
        }
        check(v[1].is_numeric() && (v[1].rank()==2 || (op==O::triangular_solve && v[1].rank()==1)),
              "TYPE_MISMATCH: invalid right hand side");
        check(v[1].axes[0]==out.axes[0],"AXIS_MISMATCH: RHS equation axis differs");
        check(v[1].shape[0]==out.shape[0],"SHAPE_MISMATCH: RHS length differs");
        for (std::size_t i=2;i<v.size();++i)
            check(v[i].is_scalar() && v[i].is_numeric() && v[i].semantic_dimension=="dimensionless",
                  "TYPE_MISMATCH: triangular options must be dimensionless scalars");
        auto axes=v[1].axes; axes[0]=out.axes[1];
        auto result=shaped(v[1],axes,v[1].shape);
        return result.with_semantics(v[0].semantic_dimension=="dimensionless" ? v[1].semantic_dimension
            : "derived:"+v[1].semantic_dimension+"/"+v[0].semantic_dimension);
    }
    check(out.record_tag.empty() && out.kind!=ValueKind::record && out.kind!=ValueKind::window,
          "TYPE_MISMATCH: array operation cannot consume a record/window");
    check(out.rank()>0 || op==O::stack,"RANK_MISMATCH: expected array");
    if (op==O::broadcast_like) {
        check(v[1].record_tag.empty() && v[1].kind!=ValueKind::record && v[1].kind!=ValueKind::window &&
              v[1].rank()>=out.rank() && v[1].dtype==out.dtype,"TYPE_MISMATCH: invalid broadcast template");
        const auto offset=v[1].rank()-out.rank();
        for (std::size_t i=0;i<out.rank();++i) {
            check(out.axes[i]==v[1].axes[offset+i],"AXIS_MISMATCH: broadcast axes differ");
            check(out.shape[i]=="1" || out.shape[i]==v[1].shape[offset+i],"SHAPE_MISMATCH: explicit broadcast mismatch");
        }
        return shaped(out,v[1].axes,v[1].shape);
    }
    const bool binary=op==O::concatenate || op==O::stack;
    const auto axis=literal(op==O::take_axis || binary ? 2 : 1,out.rank()-(op==O::stack ? 0 : 1));
    auto axes=out.axes, shape=out.shape;
    if (op==O::take_axis) {
        check(v[1].rank()==1 && v[1].dtype==DType::int64 && v[1].record_tag.empty(),
              "TYPE_MISMATCH: take_axis requires int64 indices");
        shape[axis]=v[1].shape[0];
    } else if (op==O::slice_axis) {
        const auto start=literal(2,9007199254740991.0), stop=literal(3,9007199254740991.0),
                   step=literal(4,9007199254740991.0);
        check(step>0 && start<=stop,"INVALID_PARAMETER: invalid slice bounds/step");
        shape[axis]=std::to_string((stop-start)/step+((stop-start)%step!=0));
    } else if (binary) {
        check(out.dtype==v[1].dtype && out.axes==v[1].axes && out.record_tag==v[1].record_tag &&
              out.semantic_dimension==v[1].semantic_dimension && out.price_basis==v[1].price_basis,
              "TYPE_MISMATCH: concatenate/stack require identical dtype/axes/semantics");
        for (std::size_t i=0;i<shape.size();++i)
            if (op==O::stack || i!=axis) check(shape[i]==v[1].shape[i],"SHAPE_MISMATCH: unmatched array dimensions");
        if (op==O::stack) {
            check(out.rank()<3,"RANK_MISMATCH: stack supports rank up to two");
            axes.insert(axes.begin()+axis,"component"); shape.insert(shape.begin()+axis,"2");
        } else shape[axis]="("+shape[axis]+"+"+v[1].shape[axis]+")";
    } else if (op==O::swap_axes) {
        const auto other=literal(2,out.rank()-1);
        std::swap(axes[axis],axes[other]); std::swap(shape[axis],shape[other]);
    } else {
        check(out.is_numeric(),"TYPE_MISMATCH: axis reduction requires float64");
        axes.erase(axes.begin()+axis); shape.erase(shape.begin()+axis);
    }
    return shaped(out,axes,shape);
}
} // namespace calmetrics_engine::typed
