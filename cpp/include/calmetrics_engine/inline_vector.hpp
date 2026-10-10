#pragma once
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace calmetrics_engine {
// Small execution metadata, with ordinary vector storage for larger batches.
// Deliberately limited to the operations used by native result/audit containers.
template <class T, std::size_t N> class InlineVector {
  static_assert(N > 0 && std::is_nothrow_move_constructible_v<T> &&
                std::is_nothrow_move_assignable_v<T>);
  // The storage alternatives overlap: unused inline space must not remain
  // attached to a heap-backed result or inflate every retained result twice.
  std::variant<std::array<T, N>, std::vector<T>> storage_;
  std::size_t size_ = 0;
public:
  InlineVector() = default;
  InlineVector(const InlineVector &) = default;
  InlineVector &operator=(const InlineVector &) = default;
  InlineVector(InlineVector &&other) noexcept
      : storage_(std::move(other.storage_)),
        size_(std::exchange(other.size_, 0)) {}
  InlineVector &operator=(InlineVector &&other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return !size_; }
  T *data() noexcept {
    return storage_.index() == 0 ? std::get<0>(storage_).data() : std::get<1>(storage_).data();
  }
  const T *data() const noexcept {
    return storage_.index() == 0 ? std::get<0>(storage_).data() : std::get<1>(storage_).data();
  }
  T *begin() noexcept { return data(); }
  const T *begin() const noexcept { return data(); }
  T *end() noexcept { return size_ ? data() + size_ : data(); }
  const T *end() const noexcept { return size_ ? data() + size_ : data(); }
  T &operator[](std::size_t i) noexcept { return data()[i]; }
  const T &operator[](std::size_t i) const noexcept { return data()[i]; }
  const T &at(std::size_t i) const {
    if (i >= size_) throw std::out_of_range("execution metadata index");
    return data()[i];
  }
  void resize(std::size_t n) {
    const auto old = size_;
    if (n > N) {
      if (storage_.index() == 1) std::get<1>(storage_).resize(n);
      else {
        std::vector<T> next(n);
        auto &local = std::get<0>(storage_);
        std::move(local.begin(), local.begin() + old, next.begin());
        storage_.template emplace<1>(std::move(next));
      }
    } else {
      if (storage_.index() == 1) {
        std::array<T, N> next{};
        auto &heap = std::get<1>(storage_);
        std::move(heap.begin(), heap.begin() + std::min(n, old), next.begin());
        storage_.template emplace<0>(std::move(next));
      } else {
        auto &local = std::get<0>(storage_);
        for (auto i = n; i < old; ++i) local[i] = T{};
        for (auto i = old; i < n; ++i) local[i] = T{};
      }
    }
    size_ = n;
  }
  void clear() { resize(0); }
  void push_back(T value) { const auto i = size_; resize(i + 1); data()[i] = std::move(value); }
};

// A single inline element needs no separate length: the variant identifies
// that element, and the vector owns the size for both empty and larger batches.
template <class T> class InlineVector<T, 1> {
  static_assert(std::is_nothrow_move_constructible_v<T> &&
                std::is_nothrow_move_assignable_v<T>);
  std::variant<T, std::vector<T>> storage_{std::in_place_index<1>};
public:
  InlineVector() = default;
  InlineVector(const InlineVector &) = default;
  InlineVector &operator=(const InlineVector &other) {
    if (this != &other) {
      InlineVector next(other);
      *this = std::move(next);
    }
    return *this;
  }
  InlineVector(InlineVector &&other) noexcept
      : storage_(std::move(other.storage_)) { other.clear(); }
  InlineVector &operator=(InlineVector &&other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      other.clear();
    }
    return *this;
  }
  std::size_t size() const noexcept {
    return storage_.index() == 0 ? 1 : std::get<1>(storage_).size();
  }
  bool empty() const noexcept { return !size(); }
  T *data() noexcept {
    return storage_.index() == 0 ? std::addressof(std::get<0>(storage_))
                                 : std::get<1>(storage_).data();
  }
  const T *data() const noexcept {
    return storage_.index() == 0 ? std::addressof(std::get<0>(storage_))
                                 : std::get<1>(storage_).data();
  }
  T *begin() noexcept { return data(); }
  const T *begin() const noexcept { return data(); }
  T *end() noexcept { return empty() ? data() : data() + size(); }
  const T *end() const noexcept { return empty() ? data() : data() + size(); }
  T &operator[](std::size_t i) noexcept { return data()[i]; }
  const T &operator[](std::size_t i) const noexcept { return data()[i]; }
  const T &at(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("execution metadata index");
    return data()[i];
  }
  void resize(std::size_t n) {
    if (n == size()) return;
    if (n == 0) { clear(); return; }
    if (n == 1) {
      auto &heap = std::get<1>(storage_);
      // Construct before replacing storage: a throwing default constructor
      // must leave the empty container usable, never a valueless variant.
      T next = heap.empty() ? T{} : std::move(heap.front());
      storage_.template emplace<0>(std::move(next));
    } else if (storage_.index() == 1) {
      std::get<1>(storage_).resize(n);
    } else {
      // Allocation/default construction may fail; move the live owner only
      // after those operations succeed. All remaining moves are nonthrowing.
      std::vector<T> next(n);
      next[0] = std::move(std::get<0>(storage_));
      storage_.template emplace<1>(std::move(next));
    }
  }
  void clear() noexcept { storage_.template emplace<1>(); }
  void push_back(T value) {
    const auto i = size();
    resize(i + 1);
    data()[i] = std::move(value);
  }
};
} // namespace calmetrics_engine
