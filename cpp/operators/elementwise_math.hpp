#pragma once

namespace calmetrics_engine::ops::elementwise_detail {

// Shared elementary bodies. The canonical operator checks its domain before
// division; physical callers must retain that check or original-node replay.
inline double subtract_step(double left, double right) noexcept {
    return left - right;
}
inline double multiply_step(double left, double right) noexcept {
    return left * right;
}
inline double nonzero_divide_step(double left, double right) noexcept {
    return left / right;
}

} // namespace calmetrics_engine::ops::elementwise_detail
