#include "mask_validation.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mask_validation_highway.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace calmetrics_engine::ops::mask_detail {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <bool Counting>
bool check_bytes(const std::uint8_t *data, std::size_t count,
                 std::size_t *selected) {
  const hn::ScalableTag<std::uint8_t> d;
  const auto lanes = hn::Lanes(d);
  const auto limit = hn::Set(d, std::uint8_t(2));
  const auto one = hn::Set(d, std::uint8_t(1));
  const auto stop = count - count % lanes;
  std::size_t total = 0, i = 0;
  for (; i < stop; i += lanes) {
    const auto bytes = hn::LoadU(d, data + i);
    if (!hn::AllTrue(d, hn::Lt(bytes, limit))) return false;
    if constexpr (Counting) total += hn::CountTrue(d, hn::Eq(bytes, one));
  }
  for (; i < count; ++i) {
    if (data[i] > 1) return false;
    if constexpr (Counting) total += data[i];
  }
  if constexpr (Counting) *selected = total;
  return true;
}

bool ValidateMaskBytesTarget(const std::uint8_t *data, std::size_t count,
                             std::size_t *selected) {
  return selected ? check_bytes<true>(data, count, selected)
                  : check_bytes<false>(data, count, nullptr);
}
} // namespace HWY_NAMESPACE
} // namespace calmetrics_engine::ops::mask_detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace calmetrics_engine::ops::mask_detail {
HWY_EXPORT(ValidateMaskBytesTarget);
bool validate_mask_bytes_highway(const std::uint8_t *data, std::size_t count,
                                 std::size_t *selected) noexcept {
  return HWY_DYNAMIC_DISPATCH(ValidateMaskBytesTarget)(data, count, selected);
}
} // namespace calmetrics_engine::ops::mask_detail
#endif
