#pragma once
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace masked_reduce_native_cases {
inline void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
inline void run() {
  namespace o = calmetrics_engine::ops;
  constexpr std::size_t rows = 7, count = 129;
  const auto same = [](double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0 || (std::isnan(a) && std::isnan(b));
  };
  const auto expected_dispatch = [](bool used) {
#if defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
      const auto expected = std::getenv("CALMETRICS_TEST_EXPECT_EIGEN");
      if (expected) check(used == (std::strcmp(expected, "1") == 0), "masked sum actual dispatch");
    }
#else
    check(!used, "unsupported masked sum dispatch");
#endif
  };
  for (int layout = 0; layout < 5; ++layout) {
    std::vector<double> left(rows * count * 2), right(rows * count * 2);
    std::vector<std::uint8_t> masks(rows * count * 2);
    for (std::size_t i = 0; i < left.size(); ++i) {
      const double values[]{1e16, 1.0, -1e16, 2.0, -0.0};
      left[i] = values[i % 5]; right[i] = double(int(i % 7) - 3) / 8;
      masks[i] = i % 3 == 0 ? 0 : i % 3 == 1 ? 1 : 255;
    }
    const auto view = [&](const void *data, o::Kind kind) {
      o::Value v; v.kind = kind; v.shape = o::matrix_shape(rows, count); v.data = data;
      v.stride[0] = layout == 1 ? 1 : layout == 4 ? 0 : (layout == 3 ? 2 : 1) * count;
      v.stride[1] = layout == 1 ? rows : layout == 3 ? 2 : 1;
      if (layout == 2) { v.stride[0] = -std::ptrdiff_t(count); v.stride[1] = -1; }
      return v;
    };
    const auto offset = layout == 2 ? rows * count - 1 : 0;
    auto av = view(left.data() + offset, o::Kind::number);
    auto bv = view(right.data() + offset, o::Kind::number);
    auto mask = view(masks.data() + offset, o::Kind::mask);
    const auto left_before = left, right_before = right;
    const auto mask_before = masks;
    for (int broadcast = 0; broadcast < 5; ++broadcast) {
      o::Prepared where; where.count = 3; where.output_shape = av.shape;
      where.args[0] = mask; where.args[1] = av; where.args[2] = bv;
      if (broadcast == 1 || broadcast == 3) where.args[1] = o::Value::number(-0.0);
      if (broadcast == 2 || broadcast == 3) where.args[2] = o::Value::number(.5);
      if (broadcast == 4) { where.args[0].shape = {}; where.args[0].data = nullptr; where.args[0].scalar = 1; }
      o::Prepared subtraction; subtraction.count = 2; subtraction.output_shape = av.shape;
      subtraction.args[0] = where.args[1]; subtraction.args[1] = where.args[2];
      const auto value = [&](const o::Value &v, std::size_t i) {
        return v.shape.rank ? static_cast<const double *>(v.data)[v.offset(i)] : v.scalar;
      };
      const auto selected = [&](std::size_t i) {
        const auto &v = where.args[0];
        return v.shape.rank ? static_cast<const std::uint8_t *>(v.data)[v.offset(i)] != 0 : v.scalar != 0;
      };
      for (bool square : {false, true}) {
        std::vector<double> output(rows + 2, -987);
        std::size_t calls = 0;
        const auto checkpoint = [](void *p) { ++*static_cast<std::size_t *>(p); };
        const bool used = square
            ? o::library_square_sum_quad(subtraction, where.args[0], .25, 2, output.data(), &calls, checkpoint)
            : o::library_where_sum_quad(where, 2, output.data(), &calls, checkpoint);
        expected_dispatch(used);
        for (std::size_t row = 0; row < output.size(); ++row) {
          if (!used || row < 2 || row >= 6) {
            check(output[row] == -987, "masked sum wrote outside four rows"); continue;
          }
          double total = 0;
          for (std::size_t j = 0; j < count; ++j) {
            const auto i = row * count + j;
            const auto a = value(where.args[1], i), b = value(where.args[2], i);
            const auto delta = a - b;
            total += square ? (selected(i) ? delta * delta : .25) : (selected(i) ? a : b);
          }
          check(same(output[row], total), "masked sum changed canonical left fold");
        }
        check(calls == (used ? 1u : 0u), "masked sum checkpoint count");
        if (used) {
          std::fill(output.begin(), output.end(), -987);
          try {
            const auto fail = [](void *) { throw std::runtime_error("masked-checkpoint"); };
            if (square) o::library_square_sum_quad(subtraction, where.args[0], .25, 2, output.data(), nullptr, fail);
            else o::library_where_sum_quad(where, 2, output.data(), nullptr, fail);
            check(false, "masked sum lost cancellation");
          } catch (const std::runtime_error &e) {
            check(std::strcmp(e.what(), "masked-checkpoint") == 0, "masked sum changed cancellation");
          }
          for (auto x : output) check(x == -987, "masked sum wrote cancelled result");
        }
        std::fill(output.begin(), output.end(), -987);
        calls = 0;
        for (std::size_t row : {std::size_t(4), rows + 1}) {
          const bool rejected = square
              ? !o::library_square_sum_quad(subtraction, where.args[0], .25, row, output.data(), &calls, checkpoint)
              : !o::library_where_sum_quad(where, row, output.data(), &calls, checkpoint);
          check(rejected, "masked sum admitted short or invalid row range");
        }
        auto short_where = where; short_where.output_shape = o::matrix_shape(rows, 127);
        auto short_subtraction = subtraction; short_subtraction.output_shape = short_where.output_shape;
        check(!o::library_where_sum_quad(short_where, 0, output.data(), &calls, checkpoint), "masked sum admitted short axis");
        check(!o::library_square_sum_quad(short_subtraction, where.args[0], .25, 0, output.data(), &calls, checkpoint), "square sum admitted short axis");
        for (auto x : output) check(x == -987, "masked sum rejection wrote output");
        check(calls == 0, "masked sum rejection called checkpoint");
      }
    }
    check(left == left_before && right == right_before && masks == mask_before, "masked sum mutated borrowed input");
  }
  // Selection must not perform arithmetic on its unselected NaN/Inf branch.
  std::vector<double> data(rows * count, std::numeric_limits<double>::quiet_NaN());
  std::vector<std::uint8_t> mask(rows * count, 0);
  o::Prepared where; where.count = 3; where.output_shape = o::matrix_shape(rows, count);
  where.args[0].kind = o::Kind::mask; where.args[0].shape = where.output_shape;
  where.args[0].data = mask.data(); where.args[0].set_contiguous_strides();
  where.args[1].shape = where.output_shape; where.args[1].data = data.data();
  where.args[1].set_contiguous_strides(); where.args[2] = o::Value::number(.5);
  double output[rows]{};
  const bool used = o::library_where_sum_quad(where, 1, output, nullptr, nullptr);
  expected_dispatch(used);
  if (used) for (std::size_t row = 1; row < 5; ++row)
    check(output[row] == double(count) / 2, "unselected NaN contaminated masked sum");

  // Cancellation after arithmetic has started still cannot publish this quad.
  constexpr std::size_t long_count = 2049;
  data.assign(rows * long_count, .125); mask.assign(rows * long_count, 1);
  where.output_shape = o::matrix_shape(rows, long_count);
  where.args[0].shape = where.output_shape; where.args[0].data = mask.data();
  where.args[0].set_contiguous_strides();
  where.args[1].shape = where.output_shape; where.args[1].data = data.data();
  where.args[1].set_contiguous_strides();
  o::Prepared subtraction; subtraction.count = 2; subtraction.output_shape = where.output_shape;
  subtraction.args[0] = where.args[1]; subtraction.args[1] = o::Value::number(.0625);
  for (bool square : {false, true}) {
    std::fill_n(output, rows, -987);
    std::size_t checkpoints = 0;
    const auto cancel = [](void *p) {
      if (++*static_cast<std::size_t *>(p) == 2) throw std::runtime_error("masked-midpoint");
    };
    try {
      const bool active = square
          ? o::library_square_sum_quad(subtraction, where.args[0], 0, 1, output, &checkpoints, cancel)
          : o::library_where_sum_quad(where, 1, output, &checkpoints, cancel);
      check(!active, "masked sum missed midpoint cancellation");
    } catch (const std::runtime_error &e) {
      check(std::strcmp(e.what(), "masked-midpoint") == 0 && checkpoints == 2,
          "masked sum midpoint cancellation changed");
    }
    for (auto x : output) check(x == -987, "masked sum published cancelled quad");
  }
}
}
