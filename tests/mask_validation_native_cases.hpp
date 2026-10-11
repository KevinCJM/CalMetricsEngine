#pragma once
#include "calmetrics_engine/operators.hpp"
#include "../cpp/operators/mask_validation.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mask_validation_native_cases {
namespace o = calmetrics_engine::ops;
inline void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
inline void run() {
  for (std::size_t n : {0U,1U,15U,16U,17U,127U,128U,129U,255U,256U,257U,4097U,65537U}) {
    std::vector<std::uint8_t> input(n,1);
    o::Value value; value.kind=o::Kind::mask; value.shape=o::vector_shape(n);
    value.data=input.data();
    const auto p=o::prepare(o::lookup("count_true"),&value,1);
    o::Output out; out.kind=p.output_kind; out.shape=p.output_shape;
    o::Workspace work; o::Audit audit;
    o::execute(p,out,work,o::Isa::automatic,audit);
    check(out.scalar==double(n),"mask count changed or overflowed");
    check(std::all_of(input.begin(),input.end(),[](auto x){return x==1;}),"mask input mutated");
  }
#if CALMETRICS_EXPECT_HIGHWAY_ORDER
  for (std::size_t n : {0U,1U,15U,16U,17U,127U,128U,129U,255U,256U,257U,4097U,65537U})
    for (std::size_t offset : {0U,1U,15U}) {
      // Exact allocation end permits ASan to catch any tail over-read;
      // invalid leading bytes must never be read as part of this view.
      std::unique_ptr<std::uint8_t[]> owner(new std::uint8_t[n+offset]);
      std::fill_n(owner.get(),n+offset,std::uint8_t(255));
      auto *data=owner.get()+offset;
      std::fill_n(data,n,std::uint8_t(1));
      if (n>1) for (std::size_t i=1;i<n;i+=3) data[i]=0;
      const std::size_t expected=n ? std::count(data,data+n,std::uint8_t(1)) : 0;
      std::size_t selected=777;
      check(o::mask_detail::validate_mask_bytes_highway(data,n,&selected)&&selected==expected,
            "Highway mask count/validity or wide count overflow");
      check(o::mask_detail::validate_mask_bytes_highway(data,n,nullptr),"mask predicate-only validation");
      for (std::size_t i=0;i<offset;++i) check(owner[i]==255,"mask changed leading canary");
      if (n) for (std::size_t position : {std::size_t(0),n/2,n-1}) {
        const auto old=data[position];
        for (unsigned bad=2;bad<=255;++bad) {
          data[position]=std::uint8_t(bad); selected=777;
          check(!o::mask_detail::validate_mask_bytes_highway(data,n,&selected)&&selected==777,
                "mask invalid byte accepted or failed count committed");
          check(!o::mask_detail::validate_mask_bytes_highway(data,n,nullptr),"mask invalid predicate accepted");
        }
        data[position]=old;
      }
    }
#endif
}
} // namespace mask_validation_native_cases
