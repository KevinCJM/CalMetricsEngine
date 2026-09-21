#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace calmetrics_engine::ops {

inline constexpr const char *registry_version = "canonical-native-1";
inline constexpr std::size_t operator_count = 125;

enum class Op : std::uint16_t {
#define OP(id, name, family, lo, hi, params) name = id,
#include "calmetrics_engine/operators.def"
#undef OP
};
enum class Family {
    elementwise,
    reduction,
    sequence,
    rolling,
    matrix,
    regression,
    state,
    composite
};
enum class Kind { number, mask, fit, interval, integer };
enum class Isa { automatic, scalar, sse2, avx2, neon };

struct Error : std::runtime_error {
    explicit Error(const char *code) : std::runtime_error(code) {}
};
inline void require(bool condition, const char *code) {
    if (!condition)
        throw Error(code);
}

struct Shape {
    int rank = 0;
    std::array<std::size_t, 2> dim{1, 1};
    std::size_t size() const {
        if (rank == 0)
            return 1;
        if (rank == 1)
            return dim[0];
        require(dim[1] == 0 || dim[0] <= static_cast<std::size_t>(PTRDIFF_MAX) / dim[1],
                "SHAPE_OVERFLOW");
        return dim[0] * dim[1];
    }
    bool operator==(const Shape &other) const noexcept {
        return rank == other.rank && (rank == 0 || dim[0] == other.dim[0]) &&
               (rank < 2 || dim[1] == other.dim[1]);
    }
};
inline Shape vector_shape(std::size_t n) { return {1, {n, 1}}; }
inline Shape matrix_shape(std::size_t rows, std::size_t cols) { return {2, {rows, cols}}; }

// Non-owning, read-only metadata. Strides are signed ELEMENT strides, never bytes.
// Scalars/records live inline, so moving a Value cannot invalidate its own storage.
struct Value {
    Kind kind = Kind::number;
    Shape shape{};
    const void *data = nullptr;
    std::array<std::ptrdiff_t, 2> stride{1, 1};
    double scalar = 0.0;
    std::int64_t integer = 0;
    std::array<double, 5> record{};

    static Value number(double x) {
        Value v;
        v.scalar = x;
        return v;
    }
    std::size_t size() const { return shape.size(); }
    std::ptrdiff_t offset(std::size_t i) const noexcept {
        if (shape.rank == 0)
            return 0;
        if (shape.rank == 1)
            return static_cast<std::ptrdiff_t>(i) * stride[0];
        return static_cast<std::ptrdiff_t>(i / shape.dim[1]) * stride[0] +
               static_cast<std::ptrdiff_t>(i % shape.dim[1]) * stride[1];
    }
    double f(std::size_t i) const noexcept {
        return shape.rank == 0 ? scalar : static_cast<const double *>(data)[offset(i)];
    }
    std::uint8_t u(std::size_t i) const noexcept {
        return shape.rank == 0 ? static_cast<std::uint8_t>(scalar)
                               : static_cast<const std::uint8_t *>(data)[offset(i)];
    }
    std::int64_t i(std::size_t index) const noexcept {
        return shape.rank == 0 ? integer
                               : static_cast<const std::int64_t *>(data)[offset(index)];
    }
    double at(std::size_t row, std::size_t col) const noexcept {
        return static_cast<const double *>(data)[static_cast<std::ptrdiff_t>(row) * stride[0] +
                                                 static_cast<std::ptrdiff_t>(col) * stride[1]];
    }
    bool contiguous() const noexcept {
        return shape.rank == 0 ||
               (shape.rank == 1
                    ? stride[0] == 1
                    : stride[1] == 1 && stride[0] == static_cast<std::ptrdiff_t>(shape.dim[1]));
    }
    Value row(std::size_t i) const;
    Value column(std::size_t i) const;
};

struct Spec {
    Op op;
    const char *name;
    Family family;
    std::uint8_t min_args;
    std::uint8_t max_args;
    const char *parameters;
};
const std::array<Spec, operator_count> &registry();
const Spec &lookup(std::string_view name);
const Spec &lookup(std::uint16_t opcode);
const char *family_name(Family family);
const char *shape_rule(const Spec &spec);
const char *missing_policy(const Spec &spec);
const char *composition(const Spec &spec);
const char *default_rule(const Spec &spec);
std::vector<std::string> parameter_names(const Spec &spec, std::size_t arity);

struct Prepared {
    const Spec *spec = nullptr;
    std::array<Value, 8> args{};
    std::size_t count = 0;
    Kind output_kind = Kind::number;
    Shape output_shape{};
    bool borrowed = false;
    Value view{};
    std::size_t scratch_doubles = 0;
    std::size_t scratch_indices = 0;
};
Prepared prepare(const Spec &spec, const Value *args, std::size_t count);

struct Output {
    Kind kind = Kind::number;
    Shape shape{};
    void *data = nullptr; // caller-owned contiguous output, null for inline scalar/record
    double scalar = 0.0;
    std::int64_t integer = 0;
    std::array<double, 5> record{};
    void set(std::size_t i, double x) noexcept {
        if (shape.rank == 0) {
            scalar = x;
            if (data)
                *static_cast<double *>(data) = x;
        } else
            static_cast<double *>(data)[i] = x;
    }
    void set_mask(std::size_t i, std::uint8_t x) noexcept {
        if (shape.rank == 0) {
            scalar = x;
            if (data)
                *static_cast<std::uint8_t *>(data) = x;
        } else
            static_cast<std::uint8_t *>(data)[i] = x;
    }
    void set_integer(std::size_t i, std::int64_t x) noexcept {
        if (shape.rank == 0) {
            integer = x;
            if (data)
                *static_cast<std::int64_t *>(data) = x;
        } else
            static_cast<std::int64_t *>(data)[i] = x;
    }
};

// Caller-owned and noncopyable. Bindings use try_lock, never wait with the GIL held.
struct Workspace {
    std::mutex mutex;
    std::vector<double> doubles;
    std::vector<std::size_t> indices;
    void ensure(std::size_t d, std::size_t i) {
        doubles.resize(d);
        indices.resize(i);
    }
    std::size_t capacity_bytes() const noexcept {
        return doubles.capacity() * sizeof(double) + indices.capacity() * sizeof(std::size_t);
    }
};
struct Audit {
    const char *isa = "scalar";
    std::size_t vector_elements = 0;
    std::size_t scratch_bytes = 0;
    std::size_t algorithm_copy_bytes = 0;
};
void execute(const Prepared &plan, Output &output, Workspace &work, Isa isa, Audit &audit);

// Family kernels. They never import/call Python or create worker threads.
void elementwise(const Prepared &, Output &, Workspace &, Isa, Audit &);
void reduction(const Prepared &, Output &, Workspace &, Audit &);
void sequence(const Prepared &, Output &, Workspace &, Audit &);
void matrix(const Prepared &, Output &, Workspace &, Isa, Audit &);
void regression(const Prepared &, Output &, Workspace &, Audit &);
void state(const Prepared &, Output &, Workspace &, Audit &);
void composite(const Prepared &, Output &, Workspace &, Isa, Audit &);

double scalar_math(Op op, double x, double y = 0.0, double z = 0.0);
double reduce_value(Op op, const Value &x, const Value *mask, std::size_t ddof, double probability,
                    Workspace &work);
std::array<double, 5> fit_value(const Value &x, const Value &y, bool implicit_x);
double product_value(const Value &x, bool add_one, Output *prefix = nullptr);
double dot_value(const Value &x, const Value &y);
void matvec_value(const Value &a, const Value &x, double *out);

bool supports_isa(Isa isa);
const char *isa_name(Isa isa);
Isa parse_isa(std::string_view name);
Isa selected_isa(Isa requested);
bool simd_eligible(Op op);
// Returns the number of elements actually processed using vector instructions.
std::size_t simd_transform(Op op, const Value &x, const Value &y, Output &out, std::size_t count,
                           Isa requested, Audit &audit);
// If nonzero, the entire result (including scalar tails) has been written.
std::size_t simd_matmul(const Value &lhs, const Value &rhs, Output &out, Isa requested,
                        Audit &audit);

} // namespace calmetrics_engine::ops
