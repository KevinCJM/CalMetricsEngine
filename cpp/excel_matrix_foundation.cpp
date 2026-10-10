#include "excel_internal.hpp"

namespace calmetrics_engine::excel {
Array Builder::foundation_operation(Op op, const std::vector<Array> &a, Array out) {
  auto cell=[&](const F &f) { return formula("IFERROR("+f+","+err("NON_FINITE_RESULT")+")"); };
  auto A=[&](std::size_t i,std::size_t j=0) { return a.at(i).at(j); };
  F failure=q("");
  auto guard=[&](const F &test,const F &code) {
    failure=cell(iff(failure+"<>\"\"",failure,iff(test,q(""),err(code))));
  };
  auto finish=[&]() {
    out.failure=failure;
    for(auto &v:out.refs) v=cell(iff(failure+"<>\"\"",failure,v));
    return out;
  };
  const auto n=a[0].shape.dim[0];
  if(op==Op::eigenvalues || op==Op::eigenvectors) {
    for(std::size_t i=0;i<n;++i) {
      if(op==Op::eigenvalues) out.refs[i]=A(0,i*(n+1));
      else for(std::size_t j=0;j<n;++j) out.refs[j*n+i]=A(0,i*(n+1)+j+1);
    }
    return out;
  }
  if(op==Op::cholesky || op==Op::symmetric_eigh) {
    std::vector<F> checks;
    for(std::size_t i=0;i<a[0].size();++i) checks.push_back(fin(A(0,i)));
    guard(all(checks),"NON_FINITE_RESULT"); checks.clear();
    for(std::size_t i=0;i<n;++i) for(std::size_t j=i+1;j<n;++j)
      checks.push_back(A(0,i*n+j)+"="+A(0,j*n+i));
    guard(all(checks),"DOMAIN_ERROR");
  }
  if(op==Op::cholesky) {
    for(std::size_t i=0;i<n;++i) {
      for(std::size_t j=0;j<=i;++j) {
        F v=A(0,i*n+j);
        for(std::size_t k=0;k<j;++k) v=cell(v+"-"+out.refs[i*n+k]+"*"+out.refs[j*n+k]);
        if(i==j) { guard(v+">0","DOMAIN_ERROR"); out.refs[i*n+j]=cell(iff(v+">0",fn("SQRT",v),"0")); }
        else out.refs[i*n+j]=cell(v+"/"+out.refs[j*n+j]);
      }
      for(std::size_t j=i+1;j<n;++j) out.refs[i*n+j]="0";
    }
    return finish();
  }
  if(op==Op::solve_multi || op==Op::triangular_solve) {
    auto m=a[0].refs, rhs=a[1].refs;
    const auto columns=a[1].shape.rank==1 ? 1 : a[1].shape.dim[1];
    const bool lower=op==Op::solve_multi || bound(a[2],"lower",1)==1;
    const bool unit=op==Op::triangular_solve && bound(a[3],"unit diagonal",1)==1;
    std::vector<F> checks;
    for(auto &v:rhs) checks.push_back(fin(v));
    for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j)
      if(op==Op::solve_multi || (i==j ? !unit : (lower ? j<i : j>i))) checks.push_back(fin(m[i*n+j]));
    guard(all(checks),"NON_FINITE_RESULT");
    if(op==Op::solve_multi) {
      for(std::size_t k=0;k<n;++k) {
        F pivot=std::to_string(k), mag=cell(fn("ABS",m[k*n+k]));
        for(std::size_t i=k+1;i<n;++i) {
          const auto test=cell(fn("ABS",m[i*n+k]));
          pivot=cell(iff(test+">"+mag,std::to_string(i),pivot));
          mag=cell("MAX("+mag+","+test+")");
        }
        guard(mag+">1E-14","SINGULAR_MATRIX");
        auto old=m, oldrhs=rhs;
        for(std::size_t j=k;j<n;++j) {
          std::vector<F> col;
          for(std::size_t i=0;i<n;++i) col.push_back(old[i*n+j]);
          m[k*n+j]=choose(col,pivot);
          for(std::size_t i=k+1;i<n;++i) m[i*n+j]=cell(iff(pivot+"="+std::to_string(i),old[k*n+j],old[i*n+j]));
        }
        for(std::size_t j=0;j<columns;++j) {
          std::vector<F> col;
          for(std::size_t i=0;i<n;++i) col.push_back(oldrhs[i*columns+j]);
          rhs[k*columns+j]=choose(col,pivot);
          for(std::size_t i=k+1;i<n;++i) rhs[i*columns+j]=cell(iff(pivot+"="+std::to_string(i),oldrhs[k*columns+j],oldrhs[i*columns+j]));
        }
        for(std::size_t i=k+1;i<n;++i) {
          const auto factor=cell(iff(mag+">1E-14",m[i*n+k]+"/"+m[k*n+k],"0"));
          for(std::size_t j=k+1;j<n;++j) m[i*n+j]=cell(m[i*n+j]+"-"+factor+"*"+m[k*n+j]);
          for(std::size_t j=0;j<columns;++j) rhs[i*columns+j]=cell(rhs[i*columns+j]+"-"+factor+"*"+rhs[k*columns+j]);
        }
      }
    }
    const bool forward=op==Op::triangular_solve && lower;
    for(std::size_t qv=0;qv<n;++qv) {
      const auto i=forward ? qv : n-1-qv;
      const auto diagonal=unit ? F("1") : m[i*n+i];
      guard(fn("ABS",diagonal)+">1E-14","SINGULAR_MATRIX");
      for(std::size_t j=0;j<columns;++j) {
        F v=rhs[i*columns+j];
        for(std::size_t k=forward ? 0 : i+1;k<(forward ? i : n);++k)
          v=cell(v+"-"+m[i*n+k]+"*"+out.refs[k*columns+j]);
        out.refs[i*columns+j]=cell(iff(fn("ABS",diagonal)+">1E-14",v+"/"+diagonal,"0"));
      }
    }
    return finish();
  }
  if(op==Op::symmetric_eigh) {
    const auto sweeps=bound(a[2],"eigen sweeps",128);
    auto m=a[0].refs; std::vector<F> vectors(n*n,"0");
    F scale="0";
    for(auto &v:m) scale=cell("MAX("+scale+",ABS("+v+"))");
    for(auto &v:m) v=cell(iff(scale+"=0","0",v+"/"+scale));
    for(std::size_t i=0;i<n;++i) vectors[i*n+i]="1";
    guard("AND(ISNUMBER("+A(1)+"),"+A(1)+">=1E-15,"+A(1)+"<=0.001)","INVALID_PARAMETER");
    for(std::size_t sweep=0;sweep<sweeps;++sweep) {
      for(std::size_t i=0;i<n;++i) for(std::size_t j=i+1;j<n;++j) {
        const auto cross=m[i*n+j];
        const auto active=cell(fn("ABS",cross)+">"+A(1));
        const auto tau=cell(iff(active,"("+m[j*n+j]+"-"+m[i*n+i]+")/(2*"+cross+")","0"));
        const auto t=cell(iff(active,iff(tau+"<0","-1","1")+"/(ABS("+tau+")+SQRT(1+"+tau+"^2))","0"));
        const auto c=cell("1/SQRT(1+"+t+"^2)"), s=cell(t+"*"+c);
        m[i*n+i]=cell(m[i*n+i]+"-"+t+"*"+cross); m[j*n+j]=cell(m[j*n+j]+"+"+t+"*"+cross);
        m[i*n+j]=m[j*n+i]=cell(iff(active,"0",cross));
        for(std::size_t k=0;k<n;++k) {
          if(k!=i && k!=j) {
            const auto u=m[k*n+i],v=m[k*n+j];
            m[k*n+i]=m[i*n+k]=cell(c+"*"+u+"-"+s+"*"+v);
            m[k*n+j]=m[j*n+k]=cell(s+"*"+u+"+"+c+"*"+v);
          }
          const auto u=vectors[i*n+k],v=vectors[j*n+k];
          vectors[i*n+k]=cell(c+"*"+u+"-"+s+"*"+v);
          vectors[j*n+k]=cell(s+"*"+u+"+"+c+"*"+v);
        }
      }
    }
    std::vector<F> converged;
    for(std::size_t i=0;i<n;++i) for(std::size_t j=i+1;j<n;++j) converged.push_back(fn("ABS",m[i*n+j])+"<="+A(1));
    guard(all(converged),"NON_CONVERGENCE");
    for(std::size_t i=0;i<n;++i) {
      out.refs[i*(n+1)]=cell(m[i*n+i]+"*"+scale);
      F sign=vectors[i*n], mag=cell(fn("ABS",sign));
      for(std::size_t j=1;j<n;++j) {
        const auto test=cell(fn("ABS",vectors[i*n+j]));
        sign=cell(iff(test+">"+mag,vectors[i*n+j],sign)); mag=cell("MAX("+mag+","+test+")");
      }
      for(std::size_t j=0;j<n;++j) out.refs[i*(n+1)+j+1]=cell(iff(sign+"<0","-"+vectors[i*n+j],vectors[i*n+j]));
    }
    // Fixed insertion network, stable on equal eigenvalues. No C++ values enter formulas.
    for(std::size_t i=1;i<n;++i) for(std::size_t j=i;j>0;--j) {
      const auto swap=cell(out.refs[j*(n+1)]+"<"+out.refs[(j-1)*(n+1)]);
      for(std::size_t k=0;k<n+1;++k) {
        const auto l=out.refs[(j-1)*(n+1)+k],r=out.refs[j*(n+1)+k];
        out.refs[(j-1)*(n+1)+k]=cell(iff(swap,r,l)); out.refs[j*(n+1)+k]=cell(iff(swap,l,r));
      }
    }
    return finish();
  }
  const auto axis=op==Op::broadcast_like ? 0 : bound(a[(op==Op::take_axis || op==Op::concatenate || op==Op::stack) ? 2 : 1],"axis",3);
  const auto start=op==Op::slice_axis ? bound(a[2],"slice start",9007199254740991ULL) : 0;
  if(op==Op::slice_axis) bound(a[3],"slice stop",9007199254740991ULL);
  const auto step=op==Op::slice_axis ? bound(a[4],"slice step",9007199254740991ULL) : 1;
  const auto other=op==Op::swap_axes ? bound(a[2],"second axis",2) : 0;
  auto index=[](const ops::Shape &shape,const std::array<std::size_t,3> &coordinate) {
    std::size_t result=0; for(int d=0;d<shape.rank;++d) result=result*shape.dim[d]+coordinate[d]; return result;
  };
  for(std::size_t i=0;i<out.size();++i) {
    std::array<std::size_t,3> coordinate{}; auto remaining=i;
    for(int d=out.shape.rank-1;d>=0;--d) {coordinate[d]=remaining%out.shape.dim[d];remaining/=out.shape.dim[d];}
    auto source=coordinate; std::size_t operand=0;
    if(op==Op::stack) { operand=coordinate[axis]; for(int d=int(axis);d<a[0].shape.rank;++d) source[d]=coordinate[d+1]; }
    else if(op==Op::concatenate && coordinate[axis]>=a[0].shape.dim[axis]) {operand=1;source[axis]-=a[0].shape.dim[axis];}
    else if(op==Op::slice_axis) source[axis]=start+source[axis]*step;
    else if(op==Op::swap_axes) std::swap(source[axis],source[other]);
    else if(op==Op::broadcast_like) {
      const auto offset=out.shape.rank-a[0].shape.rank;
      for(int d=0;d<a[0].shape.rank;++d) source[d]=a[0].shape.dim[d]==1 ? 0 : coordinate[d+offset];
    }
    if(op==Op::take_axis) {
      std::vector<F> candidates;
      for(std::size_t k=0;k<a[0].shape.dim[axis];++k) {source[axis]=k;candidates.push_back(A(0,index(a[0].shape,source)));}
      out.refs[i]=choose(candidates,A(1,coordinate[axis]),err("INDEX_OUT_OF_BOUNDS"));
    } else if(op==Op::sum_axis || op==Op::mean_axis) {
      for(int d=a[0].shape.rank-1;d>int(axis);--d) source[d]=coordinate[d-1];
      std::vector<F> selected;
      for(std::size_t k=0;k<a[0].shape.dim[axis];++k) {source[axis]=k;selected.push_back(A(0,index(a[0].shape,source)));}
      if(selected.empty()) out.refs[i]=op==Op::sum_axis ? "0" : err("INSUFFICIENT_SAMPLE");
      else {
        const auto total=reduce(Op::sum,selected);
        out.refs[i]=op==Op::sum_axis ? total : cell(iff(fin(total),total+"/"+std::to_string(selected.size()),q("NaN")));
      }
    } else out.refs[i]=A(operand,index(a[operand].shape,source));
  }
  return out;
}
} // namespace calmetrics_engine::excel
