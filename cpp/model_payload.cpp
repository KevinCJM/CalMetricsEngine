#include "calmetrics_engine/model_payload.hpp"
#include "calmetrics_engine/planner.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>

namespace calmetrics_engine::native {
namespace {
constexpr std::size_t max_fields = 4096, max_text = 4096;
constexpr std::uint64_t magic = 0x434d454d00000001ull;
using Bytes = std::vector<std::uint8_t>;
void put(Bytes &out, std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
}
void text(Bytes &out, const std::string &value) {
  ops::require(value.size() <= max_text, "MODEL_METADATA_LIMIT");
  put(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}
struct Reader {
  const Bytes &data;
  std::size_t cursor = 0;
  std::uint64_t get() {
    ops::require(cursor <= data.size() && data.size() - cursor >= 8, "MODEL_TRUNCATED");
    std::uint64_t out = 0;
    for (unsigned i = 0; i < 8; ++i) out |= std::uint64_t(data[cursor++]) << (i * 8);
    return out;
  }
  std::string string() {
    const auto count = get();
    ops::require(count <= max_text && count <= data.size() - cursor, "MODEL_METADATA_LIMIT");
    std::string out(data.begin() + cursor, data.begin() + cursor + count);
    cursor += count;
    return out;
  }
};
bool identifier(const std::string &s) {
  if (s.empty() || s.size() > 256 || (s[0] >= '0' && s[0] <= '9')) return false;
  return std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
  });
}
Bytes header(const std::vector<typed::Variable> &schema,
             const std::vector<ops::Value> &fields,
             const std::map<std::string, std::string> &metadata) {
  Bytes out;
  put(out, magic);
  put(out, metadata.size());
  for (const auto &entry : metadata) { text(out, entry.first); text(out, entry.second); }
  put(out, schema.size());
  for (std::size_t i = 0; i < schema.size(); ++i) {
    text(out, schema[i].name);
    const auto &type = schema[i].type;
    put(out, static_cast<unsigned>(type.kind));
    put(out, static_cast<unsigned>(type.dtype));
    text(out, type.semantic_dimension); text(out, type.price_basis);
    put(out, type.rank());
    for (std::size_t axis = 0; axis < type.rank(); ++axis) {
      text(out, type.axes[axis]);
      put(out, fields[i].shape.dim[axis]);
    }
  }
  return out;
}
} // namespace

std::shared_ptr<ModelPayload> ModelPayload::create(
    const std::vector<typed::Variable> &schema, const std::vector<ops::Value> &fields,
    const std::map<std::string, std::string> &metadata) {
  const std::uint16_t endian = 1;
  ops::require(*reinterpret_cast<const std::uint8_t *>(&endian) == 1, "MODEL_ENDIAN_UNSUPPORTED");
  ops::require(!schema.empty() && schema.size() <= max_fields && schema.size() == fields.size(), "MODEL_SCHEMA");
  ops::require(metadata.size() <= 32, "MODEL_METADATA_LIMIT");
  for (const auto *key : {"schema", "algorithm", "algorithm_version", "source_identity", "training_options", "random_version"}) {
    const auto found = metadata.find(key);
    ops::require(found != metadata.end() && !found->second.empty(), "MODEL_IDENTITY_REQUIRED");
  }
  for (const auto &entry : metadata)
    ops::require(!entry.first.empty() && entry.first.size() <= max_text && entry.second.size() <= max_text, "MODEL_METADATA_LIMIT");
  auto result = std::shared_ptr<ModelPayload>(new ModelPayload);
  result->schema_ = schema;
  result->metadata_ = metadata;
  std::set<std::string> names;
  std::map<std::string, std::size_t> dimensions;
  std::vector<std::size_t> offsets;
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < schema.size(); ++i) {
    auto &type = result->schema_[i].type;
    const auto &field = fields[i];
    type.validate();
    ops::require(identifier(schema[i].name) && names.insert(schema[i].name).second, "MODEL_FIELD_NAME");
    ops::require(type.kind != typed::ValueKind::record && type.kind != typed::ValueKind::window &&
        type.record_tag.empty() && std::find(type.axes.begin(), type.axes.end(), "time") == type.axes.end(), "MODEL_STATIC_FIELDS_REQUIRED");
    ops::require(field.shape.rank == static_cast<int>(type.rank()), "MODEL_FIELD_RANK");
    const auto kind = type.dtype == typed::DType::float64 ? ops::Kind::number :
        type.dtype == typed::DType::int64 ? ops::Kind::integer : ops::Kind::mask;
    ops::require(field.kind == kind, "MODEL_FIELD_DTYPE");
    ops::require(field.shape.rank == 0 || field.size() == 0 || field.data != nullptr, "MODEL_NULL_FIELD");
    if (type.is_scalar()) type.kind = typed::ValueKind::value;
    for (std::size_t axis = 0; axis < type.rank(); ++axis) {
      const auto &dimension = type.shape[axis];
      if (!dimension.empty() && std::all_of(dimension.begin(), dimension.end(), [](char c) { return c >= '0' && c <= '9'; }))
        ops::require(std::stoull(dimension) == field.shape.dim[axis], "MODEL_FIELD_SHAPE");
      else {
        const auto entry = dimensions.emplace(dimension, field.shape.dim[axis]);
        ops::require(entry.second || entry.first->second == field.shape.dim[axis], "MODEL_SYMBOLIC_SHAPE");
      }
      type.shape[axis] = std::to_string(field.shape.dim[axis]);
    }
    offsets.push_back(bytes);
    const auto raw = planner::checked_mul(field.size(), kind == ops::Kind::mask ? 1 : 8);
    bytes = planner::checked_add(bytes, planner::checked_mul(planner::checked_add(raw, 7) / 8, 8));
  }
  auto owner = SharedRegion::create(bytes);
  if (bytes) std::memset(owner->data(), 0, bytes); // deterministic alignment padding
  result->fields_.reserve(fields.size());
  for (std::size_t i = 0; i < fields.size(); ++i) {
    auto field = fields[i];
    auto *destination = static_cast<std::uint8_t *>(owner->data()) + offsets[i];
    for (std::size_t j = 0; j < field.size(); ++j) {
      if (field.kind == ops::Kind::mask) {
        const auto value = field.u(j); ops::require(value <= 1, "INVALID_MASK");
        destination[j] = value;
      } else if (field.kind == ops::Kind::integer) {
        const auto value = field.i(j); std::memcpy(destination + j * 8, &value, 8);
      } else {
        const auto value = field.f(j); std::memcpy(destination + j * 8, &value, 8);
      }
    }
    field.data = destination;
    field.set_contiguous_strides();
    result->fields_.push_back(field);
  }
  // Stable content identity, not an authentication signature. Hash all metadata,
  // shapes/dtypes and exact bytes; include algorithm and random-stream versions.
  auto prefix = header(result->schema_, result->fields_, metadata);
  std::uint64_t a = 14695981039346656037ull, b = 7809847782465536322ull;
  auto consume = [&](std::uint8_t x) { a = (a ^ x) * 1099511628211ull; b = (b ^ x) * 14029467366897019727ull; };
  for (auto x : prefix) consume(x);
  for (std::size_t i = 0; i < bytes; ++i) consume(static_cast<const std::uint8_t *>(owner->data())[i]);
  std::ostringstream id;
  id << "model-1-" << std::hex << std::setfill('0') << std::setw(16) << a << std::setw(16) << b;
  result->identity_ = id.str();
  owner->make_readonly();
  result->owner_ = std::move(owner);
  return result;
}

std::vector<std::uint8_t> ModelPayload::encode() const {
  auto out = header(schema_, fields_, metadata_);
  put(out, bytes());
  const auto *data = static_cast<const std::uint8_t *>(owner_->data());
  out.insert(out.end(), data, data + bytes());
  return out;
}

std::shared_ptr<ModelPayload> ModelPayload::decode(const std::vector<std::uint8_t> &bytes) {
  Reader in{bytes};
  ops::require(in.get() == magic, "MODEL_PROTOCOL_VERSION");
  const auto metadata_count = in.get();
  ops::require(metadata_count <= 32, "MODEL_METADATA_LIMIT");
  std::map<std::string, std::string> metadata;
  for (std::size_t i = 0; i < metadata_count; ++i) {
    const auto key = in.string(), value = in.string();
    ops::require(metadata.emplace(key, value).second, "MODEL_DUPLICATE_METADATA");
  }
  const auto count = in.get();
  ops::require(count > 0 && count <= max_fields, "MODEL_SCHEMA");
  std::vector<typed::Variable> schema;
  std::vector<ops::Value> fields;
  std::vector<std::size_t> offsets;
  std::size_t size = 0;
  for (std::size_t i = 0; i < count; ++i) {
    typed::Variable variable;
    variable.name = in.string();
    const auto kind = in.get(), dtype = in.get();
    ops::require(kind <= static_cast<unsigned>(typed::ValueKind::value) && dtype <= 2, "MODEL_FIELD_TYPE");
    variable.type.kind = static_cast<typed::ValueKind>(kind);
    variable.type.dtype = static_cast<typed::DType>(dtype);
    variable.type.semantic_dimension = in.string(); variable.type.price_basis = in.string();
    const auto rank = in.get(); ops::require(rank <= 3, "MODEL_FIELD_RANK");
    ops::Value field;
    field.kind = dtype == 0 ? ops::Kind::number : dtype == 1 ? ops::Kind::mask : ops::Kind::integer;
    field.shape.rank = static_cast<int>(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
      variable.type.axes.push_back(in.string());
      field.shape.dim[axis] = in.get();
      variable.type.shape.push_back(std::to_string(field.shape.dim[axis]));
    }
    variable.type.validate();
    field.set_contiguous_strides();
    offsets.push_back(size);
    const auto raw = planner::checked_mul(field.size(), dtype == 1 ? 1 : 8);
    size = planner::checked_add(size, planner::checked_mul(planner::checked_add(raw, 7) / 8, 8));
    schema.push_back(std::move(variable)); fields.push_back(field);
  }
  ops::require(in.get() == size && size == bytes.size() - in.cursor, "MODEL_PAYLOAD_SIZE");
  // The encoded header is not necessarily aligned. Copy to aligned storage
  // before constructing typed views, including scalar fields.
  std::vector<std::uint64_t> aligned((size + 7) / 8);
  if (size) std::memcpy(aligned.data(), bytes.data() + in.cursor, size);
  for (std::size_t i = 0; i < fields.size(); ++i) {
    auto &field = fields[i];
    field.data = size ? reinterpret_cast<const std::uint8_t *>(aligned.data()) + offsets[i] : nullptr;
    if (!field.shape.rank) {
      if (field.kind == ops::Kind::integer) std::memcpy(&field.integer, field.data, 8);
      else if (field.kind == ops::Kind::mask) field.scalar = *static_cast<const std::uint8_t *>(field.data);
      else std::memcpy(&field.scalar, field.data, 8);
    }
  }
  return create(schema, fields, metadata);
}
} // namespace calmetrics_engine::native
