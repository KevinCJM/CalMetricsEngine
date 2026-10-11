#pragma once
#include <cstddef>
#include <cstdint>

namespace calmetrics_engine::ops::mask_detail {
// Called only by an enabled native backend. Borrowed bytes are never modified;
// selected is optional and is committed only when every byte is valid.
bool validate_mask_bytes_highway(const std::uint8_t *data, std::size_t count,
                                 std::size_t *selected) noexcept;
}
