#pragma once

#include "calmetrics_engine/shared_memory.hpp"
#include "calmetrics_engine/typed_ir.hpp"
#include <map>

namespace calmetrics_engine::native {

// Immutable numeric model state, independent of Python and business persistence.
// A payload is copied once into a sealed native mapping; field views pin it.
class ModelPayload {
public:
  static std::shared_ptr<ModelPayload> create(
      const std::vector<typed::Variable> &schema,
      const std::vector<ops::Value> &fields,
      const std::map<std::string, std::string> &metadata);
  const std::vector<typed::Variable> &schema() const noexcept { return schema_; }
  const std::vector<ops::Value> &fields() const noexcept { return fields_; }
  const std::map<std::string, std::string> &metadata() const noexcept { return metadata_; }
  const std::string &identity() const noexcept { return identity_; }
  const std::shared_ptr<SharedRegion> &owner() const noexcept { return owner_; }
  std::size_t bytes() const noexcept { return owner_->size(); }
  std::vector<std::uint8_t> encode() const;
  static std::shared_ptr<ModelPayload> decode(const std::vector<std::uint8_t> &);

private:
  std::vector<typed::Variable> schema_;
  std::vector<ops::Value> fields_;
  std::map<std::string, std::string> metadata_;
  std::string identity_;
  std::shared_ptr<SharedRegion> owner_;
};

} // namespace calmetrics_engine::native
