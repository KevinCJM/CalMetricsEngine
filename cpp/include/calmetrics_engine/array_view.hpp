#pragma once

#include <cstddef>
#include <cstdint>

namespace calmetrics_engine {

template <class T>
struct VectorView {
    const T* data = nullptr;
    std::size_t size = 0;
    std::ptrdiff_t stride = 1;

    const T& operator[](std::size_t index) const noexcept {
        return data[static_cast<std::ptrdiff_t>(index) * stride];
    }
};

struct MatrixView {
    const double* data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::ptrdiff_t row_stride = 0;
    std::ptrdiff_t col_stride = 1;

    double operator()(std::size_t row, std::size_t col) const noexcept {
        return data[static_cast<std::ptrdiff_t>(row) * row_stride
                  + static_cast<std::ptrdiff_t>(col) * col_stride];
    }
};

}  // namespace calmetrics_engine
