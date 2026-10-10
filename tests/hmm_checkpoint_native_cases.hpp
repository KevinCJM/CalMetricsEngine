#pragma once
#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace hmm_checkpoint_native_cases {
namespace o=calmetrics_engine::ops;

o::Value matrix(const double *p,std::size_t n,std::size_t d,std::ptrdiff_t a,std::ptrdiff_t b) {
  o::Value v;v.data=p;v.shape=o::matrix_shape(n,d);v.stride={a,b,1};return v;
}
o::Value vector(const double *p,std::size_t n) {
  o::Value v;v.data=p;v.shape=o::vector_shape(n);return v;
}
std::vector<double> run(const char *name,const std::vector<o::Value> &a,o::Isa isa) {
  auto p=o::prepare(o::lookup(name),a.data(),a.size());std::vector<double> v(p.output_shape.size());
  o::Output out;out.kind=p.output_kind;out.shape=p.output_shape;out.data=v.data();
  o::Workspace w;o::Audit audit;o::execute(p,out,w,isa,audit);return v;
}
double lse(const double *a,std::size_t k) {
  double maximum=*std::max_element(a,a+k),total=0.;
  for(std::size_t j=0;j<k;++j)total+=std::exp(a[j]-maximum);
  return maximum+std::log(std::max(total,1e-300));
}
template<class Check>
void run(Check check) {
 std::size_t cases=0;double maximum_error=0;
 for(auto n:{1u,127u,128u,129u,255u,256u,257u})
 for(auto k:{3u,8u,12u,32u,128u}) for(auto d:{1u,8u}) for(int layout=0;layout<4;++layout)
 for(auto isa:{o::Isa::scalar,o::Isa::automatic}) {
  const auto stride=layout==3 ? d+3 : d;
  std::vector<double> storage(n*(d+3),-887.0),means(k*d),vars(k*d),weights(k),transition(k*k);
  auto x=matrix(storage.data(),n,d,stride,1);
  if(layout==1)x.stride={1,std::ptrdiff_t(n),1};
  if(layout==2){x.data=storage.data()+(n-1)*d;x.stride[0]=-std::ptrdiff_t(d);}
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<d;++j)
   const_cast<double *>(static_cast<const double *>(x.data))[std::ptrdiff_t(i)*x.stride[0]+std::ptrdiff_t(j)*x.stride[1]]=
    .2*double(int(i%7)-3)+.1*double(j);
  for(std::size_t c=0;c<k;++c){weights[c]=1./double(k);for(std::size_t j=0;j<d;++j){means[c*d+j]=.31*c-.08*j;vars[c*d+j]=.2+.07*c+.04*j;}
    for(std::size_t t=0;t<k;++t)transition[c*k+t]=c==t?.9:.1/double(k-1);}
  const auto saved=storage;
  auto mv=matrix(means.data(),k,d,d,1),vv=matrix(vars.data(),k,d,d,1),av=matrix(transition.data(),k,k,k,1);
  auto density=run("gaussian_log_density",{x,mv,vv},isa);
  auto expected=run("hmm_smooth",{matrix(density.data(),n,k,k,1),vector(weights.data(),k),av},isa);
  std::vector<double> alpha(n*k),beta(n*k,0.),temporary(k),logs(k*k),xi(k*k,0.);
  for(std::size_t c=0;c<k;++c){alpha[c]=std::log(std::max(weights[c],1e-12))+density[c];
    for(std::size_t t=0;t<k;++t)logs[c*k+t]=std::log(std::max(transition[c*k+t],1e-12));}
  for(std::size_t i=1;i<n;++i)for(std::size_t t=0;t<k;++t){
    for(std::size_t c=0;c<k;++c)temporary[c]=alpha[(i-1)*k+c]+logs[c*k+t];
    alpha[i*k+t]=density[i*k+t]+lse(temporary.data(),k);}
  for(std::size_t q=1;q<n;++q){auto i=n-1-q;for(std::size_t c=0;c<k;++c){
    for(std::size_t t=0;t<k;++t)temporary[t]=logs[c*k+t]+density[(i+1)*k+t]+beta[(i+1)*k+t];
    beta[i*k+c]=lse(temporary.data(),k);}}
  const auto expected_ll=lse(alpha.data()+(n-1)*k,k);
  // Independent textbook xi, row-major observations rather than batched rows.
  for(std::size_t i=0;i+1<n;++i){double z=-INFINITY;
    for(std::size_t c=0;c<k;++c)for(std::size_t t=0;t<k;++t){
      const double value=alpha[i*k+c]+logs[c*k+t]+density[(i+1)*k+t]+beta[(i+1)*k+t];
      if(z==-INFINITY)z=value;else if(value>z)z=value+std::log1p(std::exp(z-value));else z+=std::log1p(std::exp(value-z));}
    for(std::size_t c=0;c<k;++c)for(std::size_t t=0;t<k;++t)
      xi[c*k+t]+=std::exp(alpha[i*k+c]+logs[c*k+t]+density[(i+1)*k+t]+beta[(i+1)*k+t]-z);}
  const auto tail=o::latent_hmm_alpha_checkpoint_tail(n,k);
  std::vector<double> history(n*k+2),scratch(tail+2);
  for(int replay=0;replay<2;++replay){
    std::fill(history.begin(),history.end(),-119.0);std::fill(scratch.begin(),scratch.end(),7777.0);
    std::copy(density.begin(),density.end(),history.begin()+1);auto *small=scratch.data()+1;
    std::copy(logs.begin(),logs.end(),small);
    auto *initial=small+2*k*k+2*k,*gaussian_logs=initial+k;
    std::copy(weights.begin(),weights.end(),initial);
    for(std::size_t c=0;c<k;++c){double total=0;for(std::size_t j=0;j<d;++j)total+=std::log(std::max(vars[c*d+j],1e-12));gaussian_logs[c]=total;}
    o::Audit audit;double ll=o::hmm_alpha_checkpoint_step(x,mv,vv,history.data()+1,small,tail,isa,isa!=o::Isa::scalar,audit);
    check(std::abs(ll-expected_ll)<.000049,"LL mismatch");
    for(std::size_t i=0;i<n*k;++i){double error=std::abs(history[i+1]-expected[i]);maximum_error=std::max(maximum_error,error);check(error<.000049,"gamma mismatch");}
    for(std::size_t i=0;i<k*k;++i)check(std::abs(small[k*k+i]-xi[i])<.000049,"xi mismatch");
    check(history.front()==-119.0&&history.back()==-119.0,"history canary");
    check(scratch.front()==7777.0&&scratch.back()==7777.0,"scratch canary");
    check(storage==saved,"input mutated");
  }
  ++cases;
 }
 check(cases==560,"checkpoint test geometry coverage");
 check(maximum_error<.000049,"checkpoint gamma absolute budget");
}
template<class Check>
void gaussian_stream(Check check) {
 std::size_t cases=0;
 for(auto n:{1u,127u,128u,129u,255u,256u,257u})
 for(auto k:{1u,3u,4u}) for(auto d:{1u,4u}) for(int layout=0;layout<4;++layout)
 for(auto isa:{o::Isa::scalar,o::Isa::automatic}) for(int continuation=0;continuation<2;++continuation) {
  std::vector<double> storage(n*(d+3),-88),means(k*d),vars(k*d),initial(k,1./k),trans(k*k,1./k);
  auto x=matrix(storage.data(),n,d,d,1);
  if(layout==1)x.stride={1,std::ptrdiff_t(n),1};
  if(layout==2){x.data=storage.data()+(n-1)*d;x.stride[0]=-std::ptrdiff_t(d);}
  if(layout==3)x.stride[0]=d+3;
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<d;++j)
   const_cast<double *>(static_cast<const double *>(x.data))[std::ptrdiff_t(i)*x.stride[0]+std::ptrdiff_t(j)*x.stride[1]]=.03*double(int(i%11)-5)+.12*j;
  for(std::size_t i=0;i<k*d;++i){means[i]=.07*i;vars[i]=.2+.03*i;}
  const auto saved=storage;
  std::vector<o::Value> ga={x,matrix(means.data(),k,d,d,1),matrix(vars.data(),k,d,d,1)};
  auto density=run("gaussian_log_density",ga,isa);
  std::vector<o::Value> fa={matrix(density.data(),n,k,k,1),vector(initial.data(),k),
      matrix(trans.data(),k,k,k,1),o::Value::number(continuation)};
  auto expected=run("hmm_filter",fa,isa);
  const auto gp=o::prepare(o::lookup("gaussian_log_density"),ga.data(),ga.size());
  fa[0].data=nullptr;
  const auto fp=o::prepare_geometry(o::lookup("hmm_filter"),fa.data(),fa.size());
  o::Workspace w;w.ensure(131*k+1,0);
  const auto capacity=w.doubles.capacity();const auto *workspace=w.doubles.data();
  std::vector<double> tail(k+2,-1234.);
  o::Output out;out.shape=o::matrix_shape(1,k);out.data=tail.data()+1;
  for(int repeat=0;repeat<2;++repeat){
   o::Audit validation,inference;
   o::gaussian_density_stream_validate(gp,w,isa,validation);
   check(w.doubles.size()==129*k,"Gaussian validation live workspace size");
   o::gaussian_filter_tail(gp,fp,out,w,isa,inference);
   check(std::memcmp(tail.data()+1,expected.data()+(n-1)*k,k*sizeof(double))==0,"Gaussian stream exact tail");
   check(tail.front()==-1234.&&tail.back()==-1234.,"Gaussian stream output canary");
   // ensure() changes the live vector size. Spare capacity is not an
   // indexable canary; ASan retains its container redzone beyond size().
   check(w.doubles.size()==131*k,"Gaussian inference live workspace size");
   check(w.doubles.capacity()==capacity&&w.doubles.data()==workspace,"Gaussian stream bounded workspace reuse");
   check(validation.scratch_bytes==129*k*sizeof(double)&&inference.scratch_bytes==131*k*sizeof(double),"Gaussian stream scratch accounting");
   check(storage==saved,"Gaussian stream input mutation");
  }
  if(n==257&&k==3&&d==4&&layout==0&&continuation==0){
   storage[(n-1)*d]=std::numeric_limits<double>::max();initial[0]=-1;
   std::string error;
   try{o::Audit a;o::gaussian_density_stream_validate(gp,w,isa,a);o::gaussian_filter_tail(gp,fp,out,w,isa,a);}
   catch(const std::exception &e){error=e.what();}
   check(error=="NON_FINITE_RESULT","Late Gaussian overflow must precede invalid probability");
  }
  ++cases;
 }
 check(cases==672,"Gaussian stream tile/layout/ISA/continuation coverage");
}
} // namespace hmm_checkpoint_native_cases
