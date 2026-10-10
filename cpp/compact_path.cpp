#include "calmetrics_engine/compact_path.hpp"
#include "calmetrics_engine/affine_projection.hpp"
#include "calmetrics_engine/native_runtime.hpp"
#include "operators/axis_primitives.hpp"
#include "operators/numeric_line_sort.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#if defined(EIGEN_VECTORIZE) && !defined(EIGEN_DONT_VECTORIZE)
#define CALMETRICS_COMBINATION_PACKETS 1
#endif
#endif

namespace calmetrics_engine::graph {
namespace {
constexpr std::uint32_t absent = UINT32_MAX;
constexpr std::size_t maximum_assets = 131072;
#ifdef CALMETRICS_COMBINATION_PACKETS
constexpr std::size_t combination_packet_width = 4;
constexpr std::size_t combination_reserve_limit = 16384; // At most 128 KiB.
constexpr std::size_t combination_minimum_updates = 1048576;
#endif

void sort_line(double *keys, std::size_t count, Audit &audit) {
  ops::Audit sort_audit;
  ops::axis_path_detail::sort_numeric_line(keys, count, ops::Isa::automatic, &sort_audit);
  audit.highway_vqsort_calls += sort_audit.highway_vqsort_calls;
  audit.native_library_workspace_reserve_bytes = std::max(
      audit.native_library_workspace_reserve_bytes,
      sort_audit.native_library_workspace_reserve_bytes);
}

std::size_t add(std::size_t a, std::size_t b) {
  ops::require(a <= static_cast<std::size_t>(PTRDIFF_MAX) &&
      b <= static_cast<std::size_t>(PTRDIFF_MAX) - a, "GRAPH_ARENA_OVERFLOW");
  return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
  ops::require(!b || a <= static_cast<std::size_t>(PTRDIFF_MAX) / b, "GRAPH_ARENA_OVERFLOW");
  return a * b;
}
bool operation(const Program &p, std::uint32_t id, ops::Op op, std::size_t arity) {
  return id < p.nodes.size() && p.nodes[id].kind == NodeKind::operation &&
      p.nodes[id].opcode == static_cast<std::uint16_t>(op) && p.nodes[id].parent_count == arity;
}
bool literal(const Program &p, std::uint32_t id, std::size_t &value) {
  if (id >= p.nodes.size()) return false;
  const auto &n = p.nodes[id];
  if (n.kind != NodeKind::constant || !std::isfinite(n.constant) || n.constant < 0 ||
      n.constant > 9007199254740991.0 || n.constant >= static_cast<double>(PTRDIFF_MAX) ||
      std::floor(n.constant) != n.constant) return false;
  value = static_cast<std::size_t>(n.constant); return true;
}
bool exact(const Program &p, std::uint32_t id, std::size_t value) {
  std::size_t actual = 0; return literal(p, id, actual) && actual == value;
}
bool static_input(const Program &p, std::uint32_t id) {
  return id < p.nodes.size() && p.nodes[id].kind == NodeKind::input &&
      !p.input_axes.empty() && p.input_axes[p.nodes[id].input_index] == 1;
}
bool scalar_source(const Program &p, std::uint32_t id) {
  return id < p.nodes.size() && (p.nodes[id].kind == NodeKind::constant ||
      p.nodes[id].kind == NodeKind::parameter);
}
bool slice(const Program &p, std::uint32_t id, std::uint32_t source, std::size_t axis,
           std::size_t first, std::size_t last) {
  if (!operation(p, id, ops::Op::slice_axis, 5)) return false;
  const auto &n = p.nodes[id];
  return n.parents[0] == source && exact(p, n.parents[1], axis) &&
      exact(p, n.parents[2], first) && exact(p, n.parents[3], last) && exact(p, n.parents[4], 1);
}
void checkpoint(std::chrono::steady_clock::time_point deadline, const std::atomic<bool> *cancelled) {
  native::check_deadline(deadline);
  if (cancelled && cancelled->load()) throw native::Timeout("native compact path cancelled");
}
struct Contribution {
  std::uint32_t sum = absent, first = absent, weight_slice = absent,
      returns_slice = absent, product = absent;
  std::size_t asset = 0;
};
struct Group {
  std::uint32_t nav = absent, growth = absent, period = absent, initial = absent,
      weights = absent, broadcast = absent, weighted = absent,
      peak = absent, division = absent, drawdown = absent, maximum = absent,
      first_slice = absent, finite = absent, seed = absent, full = absent,
      quantiles = absent, probability = absent, fan = absent,
      sample_slice = absent, sample = absent, terminal_slice = absent;
  std::size_t steps = 0, samples = 0;
  std::vector<std::uint32_t> stream_nodes;
  std::vector<Contribution> contributions;
  std::vector<std::size_t> fan_roots, sample_roots, post_roots;
  std::vector<std::uint32_t> original_nodes, input_nodes;
  Program post;
};
struct Geometry {
  std::vector<ops::Value> values;
  std::vector<std::uint8_t> payload;
  std::size_t steps = 0, paths = 0, assets = 0, factors = 0,
      batch = 0, doubles = 0, coefficient_reserve = 0,
      innovations = 0, rotation_reserve = 0, scale_cells = 0,
      combination_reserve = 0;
};
struct RawPrefix {
  std::uint32_t raw = absent, rotation = absent, coefficients = absent,
      initial = absent, scaled = absent, scale = absent, scale_view = absent,
      means = absent, means_view = absent, df = absent, chi = absent;
  std::vector<std::uint32_t> nodes, scale_nodes;
};
} // namespace

struct CompactPath {
  std::vector<Group> groups;
  std::uint32_t affine = absent, returns = absent, coefficients = absent,
      source = absent, asset_initial = absent;
  std::vector<std::uint32_t> producer_nodes;
  std::vector<std::uint32_t> shared_views;
  RawPrefix prefix;
};

namespace {
template <class Predicate>
std::uint32_t unique_child(const std::vector<std::vector<std::uint32_t>> &children,
    std::uint32_t parent, Predicate predicate) {
  std::uint32_t found = absent;
  for (auto id : children[parent]) if (predicate(id)) {
    if (found != absent && found != id) return absent;
    found = id;
  }
  return found;
}
bool match_group(const Program &p, std::uint32_t nav,
    const std::vector<std::vector<std::uint32_t>> &children, Group &g) {
  g.nav = nav;
  const auto &n = p.nodes[nav];
  if (!exact(p, n.parents[1], 0) || !scalar_source(p, n.parents[2])) return false;
  g.initial = n.parents[2]; g.growth = n.parents[0];
  if (!operation(p, g.growth, ops::Op::add, 2) || !exact(p, p.nodes[g.growth].parents[0], 1)) return false;
  g.period = p.nodes[g.growth].parents[1];
  if (p.nodes[g.period].kind != NodeKind::input) {
    if (!operation(p, g.period, ops::Op::sum_axis, 2) || !exact(p, p.nodes[g.period].parents[1], 2)) return false;
    g.weighted = p.nodes[g.period].parents[0];
    if (!operation(p, g.weighted, ops::Op::multiply, 2)) return false;
    g.broadcast = p.nodes[g.weighted].parents[0];
    if (!operation(p, g.broadcast, ops::Op::broadcast_like, 2)) return false;
    g.weights = p.nodes[g.broadcast].parents[0];
    if (!static_input(p, g.weights) || p.nodes[g.broadcast].parents[1] != p.nodes[g.weighted].parents[1]) return false;
  } else if (p.input_axes.empty() || p.input_axes[p.nodes[g.period].input_index] != 2) return false;
  g.peak = unique_child(children, nav, [&](auto id) {
    return operation(p, id, ops::Op::cumulative_max_axis, 3) &&
        p.nodes[id].parents[0] == nav && exact(p, p.nodes[id].parents[1], 0) &&
        p.nodes[id].parents[2] == g.initial;
  });
  if (g.peak == absent) return false;
  g.division = unique_child(children, nav, [&](auto id) {
    return operation(p, id, ops::Op::divide, 2) && p.nodes[id].parents[0] == nav &&
        p.nodes[id].parents[1] == g.peak;
  });
  if (g.division == absent) return false;
  g.drawdown = unique_child(children, g.division, [&](auto id) {
    return operation(p, id, ops::Op::subtract, 2) && exact(p, p.nodes[id].parents[0], 1) &&
        p.nodes[id].parents[1] == g.division;
  });
  if (g.drawdown == absent) return false;
  g.maximum = unique_child(children, g.drawdown, [&](auto id) {
    return operation(p, id, ops::Op::max_axis, 2) && p.nodes[id].parents[0] == g.drawdown &&
        exact(p, p.nodes[id].parents[1], 0);
  });
  g.first_slice = unique_child(children, nav, [&](auto id) { return slice(p, id, nav, 0, 0, 1); });
  if (g.maximum == absent || g.first_slice == absent) return false;
  g.finite = unique_child(children, g.first_slice, [&](auto id) {
    return operation(p, id, ops::Op::finite_mask, 1) && p.nodes[id].parents[0] == g.first_slice;
  });
  if (g.finite == absent) return false;
  g.seed = unique_child(children, g.finite, [&](auto id) {
    return operation(p, id, ops::Op::where, 3) && p.nodes[id].parents[0] == g.finite &&
        p.nodes[id].parents[1] == g.initial && p.nodes[id].parents[2] == g.initial;
  });
  if (g.seed == absent) return false;
  g.full = unique_child(children, g.seed, [&](auto id) {
    return operation(p, id, ops::Op::concatenate, 3) && p.nodes[id].parents[0] == g.seed &&
        p.nodes[id].parents[1] == nav && exact(p, p.nodes[id].parents[2], 0);
  });
  if (g.full == absent) return false;
  g.quantiles = unique_child(children, g.full, [&](auto id) {
    return operation(p, id, ops::Op::quantiles_axis, 3) && p.nodes[id].parents[0] == g.full &&
        static_input(p, p.nodes[id].parents[1]) && exact(p, p.nodes[id].parents[2], 1);
  });
  if (g.quantiles == absent) return false;
  g.probability = p.nodes[g.quantiles].parents[1];
  g.fan = unique_child(children, g.quantiles, [&](auto id) {
    return operation(p, id, ops::Op::swap_axes, 3) && p.nodes[id].parents[0] == g.quantiles &&
        exact(p, p.nodes[id].parents[1], 0) && exact(p, p.nodes[id].parents[2], 1);
  });
  g.sample_slice = unique_child(children, g.full, [&](auto id) {
    if (!operation(p, id, ops::Op::slice_axis, 5)) return false;
    const auto &s = p.nodes[id]; std::size_t end = 0;
    return s.parents[0] == g.full && exact(p, s.parents[1], 1) && exact(p, s.parents[2], 0) &&
        literal(p, s.parents[3], end) && exact(p, s.parents[4], 1);
  });
  if (g.fan == absent || g.sample_slice == absent) return false;
  literal(p, p.nodes[g.sample_slice].parents[3], g.samples);
  g.sample = unique_child(children, g.sample_slice, [&](auto id) {
    return operation(p, id, ops::Op::swap_axes, 3) && p.nodes[id].parents[0] == g.sample_slice &&
        exact(p, p.nodes[id].parents[1], 0) && exact(p, p.nodes[id].parents[2], 1);
  });
  g.terminal_slice = unique_child(children, nav, [&](auto id) {
    if (!operation(p, id, ops::Op::slice_axis, 5)) return false;
    const auto &s = p.nodes[id]; std::size_t first = 0, end = 0;
    return s.parents[0] == nav && exact(p, s.parents[1], 0) && literal(p, s.parents[2], first) &&
        literal(p, s.parents[3], end) && end && first == end - 1 && exact(p, s.parents[4], 1) &&
        std::any_of(children[id].begin(), children[id].end(), [&](auto child) {
          return operation(p, child, ops::Op::sum_axis, 2) && p.nodes[child].parents[0] == id &&
              exact(p, p.nodes[child].parents[1], 0);
        });
  });
  if (g.sample == absent || g.terminal_slice == absent) return false;
  literal(p, p.nodes[g.terminal_slice].parents[3], g.steps);
  g.stream_nodes = {g.growth, g.nav, g.peak, g.division, g.drawdown, g.maximum,
      g.first_slice, g.finite, g.seed, g.full, g.quantiles, g.fan,
      g.sample_slice, g.sample, g.terminal_slice};
  if (g.weighted != absent) {
    g.stream_nodes.push_back(g.broadcast); g.stream_nodes.push_back(g.weighted);
    g.stream_nodes.push_back(g.period);
  }
  for (std::size_t r = 0; r < p.roots.size(); ++r) {
    if (p.roots[r] == g.fan) g.fan_roots.push_back(r);
    if (p.roots[r] == g.sample) g.sample_roots.push_back(r);
  }
  return !g.fan_roots.empty() && !g.sample_roots.empty();
}

bool coefficient_source(const Program &p, std::uint32_t id) {
  return static_input(p, id) || (operation(p, id, ops::Op::transpose, 1) &&
      static_input(p, p.nodes[id].parents[0]));
}
bool match_raw_prefix(const Program &p, CompactPath &proof) {
  auto &r = proof.prefix;
  if (!operation(p, proof.source, ops::Op::add, 2)) return false;
  r.means_view = p.nodes[proof.source].parents[0];
  r.scaled = p.nodes[proof.source].parents[1];
  if (!operation(p, r.scaled, ops::Op::multiply, 2)) return false;
  r.rotation = p.nodes[r.scaled].parents[0];
  r.scale_view = p.nodes[r.scaled].parents[1];
  if (!operation(p, r.rotation, ops::Op::affine_contract, 3)) return false;
  const auto &rotation = p.nodes[r.rotation];
  r.raw = rotation.parents[0]; r.coefficients = rotation.parents[1]; r.initial = rotation.parents[2];
  if (p.nodes[r.raw].kind != NodeKind::input ||
      p.input_axes[p.nodes[r.raw].input_index] != 2 || !coefficient_source(p, r.coefficients) ||
      !exact(p, r.initial, 0) || std::signbit(p.nodes[r.initial].constant)) return false;
  r.nodes = {r.rotation, r.scaled, proof.source};
  if (operation(p, r.coefficients, ops::Op::transpose, 1)) r.nodes.push_back(r.coefficients);
  if (operation(p, r.means_view, ops::Op::broadcast_like, 2)) {
    const auto &view = p.nodes[r.means_view]; r.means = view.parents[0];
    if (!static_input(p, r.means) ||
        (view.parents[1] != r.rotation && view.parents[1] != r.scaled)) return false;
    r.nodes.push_back(r.means_view);
  } else {
    r.means = r.means_view; r.means_view = absent;
    if (!(static_input(p, r.means) || scalar_source(p, r.means))) return false;
  }
  if (exact(p, r.scale_view, 1)) { r.scale = r.scale_view; r.scale_view = absent; }
  else {
    if (!operation(p, r.scale_view, ops::Op::broadcast_like, 2) ||
        p.nodes[r.scale_view].parents[1] != r.rotation) return false;
    r.scale = p.nodes[r.scale_view].parents[0];
    if (!operation(p, r.scale, ops::Op::divide, 2)) return false;
    const auto numerator = p.nodes[r.scale].parents[0], denominator = p.nodes[r.scale].parents[1];
    if (!operation(p, numerator, ops::Op::sqrt, 1) ||
        !operation(p, denominator, ops::Op::sqrt, 1)) return false;
    const auto ratio = p.nodes[numerator].parents[0], chi_ratio = p.nodes[denominator].parents[0];
    if (!operation(p, ratio, ops::Op::divide, 2) ||
        !operation(p, chi_ratio, ops::Op::divide, 2)) return false;
    r.df = p.nodes[ratio].parents[1]; r.chi = p.nodes[chi_ratio].parents[0];
    const auto difference = p.nodes[ratio].parents[0];
    if (!(static_input(p, r.df) || scalar_source(p, r.df)) ||
        p.nodes[chi_ratio].parents[1] != r.df ||
        !operation(p, difference, ops::Op::subtract, 2) ||
        p.nodes[difference].parents[0] != r.df || !exact(p, p.nodes[difference].parents[1], 2) ||
        p.nodes[r.chi].kind != NodeKind::input || p.input_axes[p.nodes[r.chi].input_index] != 2) return false;
    r.scale_nodes = {difference, ratio, numerator, chi_ratio, denominator, r.scale};
    std::sort(r.scale_nodes.begin(), r.scale_nodes.end());
    r.nodes.insert(r.nodes.end(), r.scale_nodes.begin(), r.scale_nodes.end());
    r.nodes.push_back(r.scale_view);
  }
  std::sort(r.nodes.begin(), r.nodes.end());
  r.nodes.erase(std::unique(r.nodes.begin(), r.nodes.end()), r.nodes.end());
  proof.producer_nodes.insert(proof.producer_nodes.end(), r.nodes.begin(), r.nodes.end());
  return true;
}
bool match_producer(const Program &p, CompactPath &proof) {
  bool projected = false, direct = false;
  for (const auto &g : proof.groups) {
    direct |= g.weighted == absent; projected |= g.weighted != absent;
    if (g.weighted == absent) continue;
    const auto r = p.nodes[g.weighted].parents[1];
    if (proof.returns != absent && proof.returns != r) return false;
    proof.returns = r;
  }
  if (direct && projected) return false;
  if (!projected) return true;
  if (!operation(p, proof.returns, ops::Op::expm1, 1)) return false;
  proof.affine = p.nodes[proof.returns].parents[0];
  if (!operation(p, proof.affine, ops::Op::affine_contract, 3)) return false;
  const auto &a = p.nodes[proof.affine];
  proof.source = a.parents[0]; proof.coefficients = a.parents[1]; proof.asset_initial = a.parents[2];
  if (!(static_input(p, proof.asset_initial) || scalar_source(p, proof.asset_initial))) return false;
  proof.producer_nodes = {proof.affine, proof.returns};
  if (!coefficient_source(p, proof.coefficients)) return false;
  if (operation(p, proof.coefficients, ops::Op::transpose, 1)) proof.producer_nodes.push_back(proof.coefficients);
  if (p.nodes[proof.source].kind == NodeKind::input) {
    if (p.input_axes[p.nodes[proof.source].input_index] != 2) return false;
  } else if (!match_raw_prefix(p, proof)) return false;
  std::sort(proof.producer_nodes.begin(), proof.producer_nodes.end());
  proof.producer_nodes.erase(std::unique(proof.producer_nodes.begin(), proof.producer_nodes.end()), proof.producer_nodes.end());
  return true;
}

bool match_contributions(const Program &p, CompactPath &proof) {
  if (proof.returns == absent) return true;
  for (std::uint32_t id = 0; id < p.nodes.size(); ++id) {
    if (!operation(p, id, ops::Op::sum, 1)) continue;
    const auto product = p.nodes[id].parents[0];
    if (!operation(p, product, ops::Op::multiply, 2)) continue;
    const auto first = p.nodes[product].parents[0], returns_slice = p.nodes[product].parents[1];
    if (!operation(p, first, ops::Op::first, 1) ||
        !operation(p, returns_slice, ops::Op::slice_axis, 5)) continue;
    const auto weight_slice = p.nodes[first].parents[0];
    if (!operation(p, weight_slice, ops::Op::slice_axis, 5)) continue;
    const auto &r = p.nodes[returns_slice], &w = p.nodes[weight_slice];
    std::size_t asset = 0;
    if (r.parents[0] != proof.returns || !exact(p, r.parents[1], 2) ||
        !literal(p, r.parents[2], asset) || asset == static_cast<std::size_t>(PTRDIFF_MAX) ||
        !slice(p, returns_slice, proof.returns, 2, asset, asset + 1) ||
        !slice(p, weight_slice, w.parents[0], 0, asset, asset + 1)) continue;
    auto found = std::find_if(proof.groups.begin(), proof.groups.end(), [&](const Group &g) {
      return g.weights == w.parents[0];
    });
    if (found == proof.groups.end()) return false;
    found->contributions.push_back({id, first, weight_slice, returns_slice, product, asset});
    for (auto node : {id, first, weight_slice, product}) found->stream_nodes.push_back(node);
    proof.shared_views.push_back(returns_slice);
  }
  std::sort(proof.shared_views.begin(), proof.shared_views.end());
  proof.shared_views.erase(std::unique(proof.shared_views.begin(), proof.shared_views.end()), proof.shared_views.end());
  for (auto &g : proof.groups)
    std::sort(g.contributions.begin(), g.contributions.end(), [](const Contribution &a, const Contribution &b) {
      return a.asset < b.asset;
    });
  return true;
}

// Clone the real canonical post DAG, rather than generating a second set of
// summary formulas. Removed operations remain consumer-free checkpoints.
bool build_posts(const Program &p, CompactPath &proof) {
  const auto count = p.nodes.size();
  std::vector<std::int32_t> owner(count, -1), capture(count, -1), producer(count, 0);
  std::vector<std::uint8_t> stream_member(count, 0);
  for (auto id : proof.producer_nodes) producer[id] = 1;
  for (auto id : proof.shared_views) producer[id] = 2;
  for (std::size_t index = 0; index < proof.groups.size(); ++index) {
    auto &g = proof.groups[index];
    std::sort(g.stream_nodes.begin(), g.stream_nodes.end());
    g.stream_nodes.erase(std::unique(g.stream_nodes.begin(), g.stream_nodes.end()), g.stream_nodes.end());
    for (auto id : g.stream_nodes) {
      if (owner[id] >= 0 && owner[id] != static_cast<std::int32_t>(index)) return false;
      owner[id] = static_cast<std::int32_t>(index);
      stream_member[id] = 1;
    }
    for (auto id : {g.fan, g.sample, g.maximum, g.terminal_slice}) capture[id] = static_cast<std::int32_t>(index);
    for (const auto &c : g.contributions) capture[c.sum] = static_cast<std::int32_t>(index);
  }
  std::vector<std::vector<std::uint8_t>> retained(proof.groups.size(), std::vector<std::uint8_t>(count));
  // A post root must reach exactly one group's capture. All other leaves are
  // static inputs, parameters or literals; cross-group arithmetic is rejected.
  for (std::size_t root = 0; root < p.roots.size(); ++root) {
    const auto original = p.roots[root];
    bool direct = false;
    for (const auto &g : proof.groups) direct |= original == g.fan || original == g.sample;
    if (direct) continue;
    if (p.root_outputs[root].rank != 0) return false;
    std::vector<std::uint8_t> needed(count);
    std::vector<std::uint32_t> pending{original}; std::int32_t group = -1;
    while (!pending.empty()) {
      const auto id = pending.back(); pending.pop_back();
      if (needed[id]) continue;
      needed[id] = 1;
      if (capture[id] >= 0) {
        if (group >= 0 && group != capture[id]) return false;
        group = capture[id]; continue;
      }
      if (stream_member[id] || producer[id]) return false;
      const auto &n = p.nodes[id];
      if (n.kind == NodeKind::input) { if (!static_input(p, id)) return false; continue; }
      if (n.kind == NodeKind::constant || n.kind == NodeKind::parameter) continue;
      if (n.kind != NodeKind::operation) return false;
      for (std::size_t j = 0; j < n.parent_count; ++j) pending.push_back(n.parents[j]);
    }
    if (group < 0) return false;
    auto &g = proof.groups[static_cast<std::size_t>(group)]; g.post_roots.push_back(root);
    for (std::size_t id = 0; id < count; ++id) if (needed[id]) {
      retained[static_cast<std::size_t>(group)][id] = 1;
      if (p.nodes[id].kind == NodeKind::operation && capture[id] < 0) {
        if (owner[id] >= 0 && owner[id] != group) return false;
        owner[id] = group;
      }
    }
  }
  for (const auto &g : proof.groups) if (g.post_roots.empty()) return false;
  // All operation consumers must be proved. A virtual array may not escape to
  // an unclassified operation/root, including an otherwise unused side branch.
  std::int32_t previous = -1;
  for (std::size_t id = 0; id < count; ++id) {
    const auto &n = p.nodes[id];
    if (n.kind != NodeKind::operation) continue;
    if (producer[id]) {
      if (producer[id] == 1 && previous >= 0) return false;
      continue;
    }
    if (owner[id] < 0 || owner[id] < previous) return false;
    previous = owner[id];
    for (std::size_t j = 0; j < n.parent_count; ++j) {
      const auto parent = n.parents[j];
      if (owner[parent] >= 0 && owner[parent] != owner[id]) return false;
      if (producer[parent]) {
        const auto &g = proof.groups[static_cast<std::size_t>(owner[id])];
        const bool return_edge = parent == proof.returns && (id == g.broadcast || id == g.weighted);
        const bool view_edge = producer[parent] == 2 &&
            std::any_of(g.contributions.begin(), g.contributions.end(), [&](const Contribution &c) {
              return c.returns_slice == parent && c.product == id;
            });
        if (!return_edge && !view_edge) return false;
      }
    }
  }
  // Producer edges, including the coefficient transpose, have their own exact
  // grammar; no root or extra producer consumer can observe the private tensor.
  for (auto id : proof.producer_nodes) {
    if (std::find(p.roots.begin(), p.roots.end(), id) != p.roots.end()) return false;
    for (std::size_t child = 0; child < count; ++child) {
      const auto &n = p.nodes[child];
      for (std::size_t j = 0; j < n.parent_count; ++j) if (n.parents[j] == id && producer[child]) {
        const bool prefix_edge = std::binary_search(proof.prefix.nodes.begin(), proof.prefix.nodes.end(), id) &&
            std::binary_search(proof.prefix.nodes.begin(), proof.prefix.nodes.end(), static_cast<std::uint32_t>(child));
        if (!((id == proof.coefficients && child == proof.affine && j == 1) ||
              prefix_edge || (id == proof.source && child == proof.affine && j == 0) ||
              (id == proof.affine && child == proof.returns && j == 0) ||
              (id == proof.returns && producer[child] == 2 && j == 0))) return false;
      }
    }
  }
  for (auto id : proof.shared_views)
    if (std::find(p.roots.begin(), p.roots.end(), id) != p.roots.end()) return false;
  for (std::size_t index = 0; index < proof.groups.size(); ++index) {
    auto &g = proof.groups[index]; auto &post = g.post;
    std::vector<std::uint32_t> remap(count, absent);
    post.output_kind = OutputKind::typed; post.output_dtype = OutputDType::float64;
    post.parameter_count = p.parameter_count; post.numeric_slots = p.numeric_slots;
    post.mask_slots = p.mask_slots; post.integer_slots = p.integer_slots;
    post.scope_work_budget = p.scope_work_budget; post.isolate_errors = false;
    for (std::uint32_t id = 0; id < count; ++id) {
      const bool stream = std::binary_search(g.stream_nodes.begin(), g.stream_nodes.end(), id);
      if (!stream && !retained[index][id]) continue;
      Node n = p.nodes[id];
      if (capture[id] == static_cast<std::int32_t>(index) || n.kind == NodeKind::input) {
        n = {}; n.kind = NodeKind::input;
        if (g.input_nodes.size() > UINT16_MAX) return false;
        n.input_index = static_cast<std::uint16_t>(g.input_nodes.size());
        g.input_nodes.push_back(id);
      } else if (stream) n = {}; // Consumer-free constant error checkpoint.
      else if (n.kind == NodeKind::operation) {
        for (std::size_t j = 0; j < n.parent_count; ++j) {
          if (remap[n.parents[j]] == absent) return false;
          n.parents[j] = remap[n.parents[j]];
        }
      }
      remap[id] = static_cast<std::uint32_t>(post.nodes.size());
      post.nodes.push_back(n); g.original_nodes.push_back(id);
    }
    post.input_count = g.input_nodes.size(); post.input_axes.assign(post.input_count, 1);
    for (auto root : g.post_roots) {
      if (remap[p.roots[root]] == absent) return false;
      post.roots.push_back(remap[p.roots[root]]); post.root_outputs.push_back(p.root_outputs[root]);
    }
    post.finalize();
  }
  return true;
}

bool bind_geometry(const Program &p, const CompactPath &proof,
    const std::vector<ops::Value> &inputs, std::size_t start, std::size_t length,
    const double *parameters, bool numerical, Geometry &geometry) {
  if (!length || std::any_of(proof.groups.begin(), proof.groups.end(), [&](const Group &group) {
        return group.steps != length;
      })) return false;
  ops::require(inputs.size() == p.input_count, "GRAPH_INPUT_COUNT");
  for (const auto &v : inputs)
    if (v.kind != ops::Kind::number || v.shape.rank < 0 || v.shape.rank > 3) return false;
  geometry.values.resize(p.nodes.size()); geometry.payload.assign(p.nodes.size(), 0);
  for (std::size_t id = 0; id < p.nodes.size(); ++id) {
    const auto &n = p.nodes[id]; auto &v = geometry.values[id];
    if (n.kind == NodeKind::input) {
      v = inputs[n.input_index];
      if (p.input_axes[n.input_index] != 1) {
        ops::require(v.shape.rank >= 1, "GRAPH_INPUT_TYPE");
        ops::require(start <= v.shape.dim[0] && length <= v.shape.dim[0] - start, "GRAPH_INTERVAL_BOUNDS");
        v.shape.dim[0] = length;
        if (numerical && v.size() && v.data && start)
          v.data = static_cast<const double *>(v.data) + static_cast<std::ptrdiff_t>(start) * v.stride[0];
      }
      if (numerical) {
        ops::require(!v.shape.rank || !v.size() || v.data, "NULL_INPUT");
        if (!v.shape.rank && v.data) v.scalar = *static_cast<const double *>(v.data);
        geometry.payload[id] = 1;
      } else v.data = nullptr;
      continue;
    }
    if (n.kind == NodeKind::constant || n.kind == NodeKind::parameter) {
      const bool constant = n.kind == NodeKind::constant;
      v = ops::Value::number(constant ? n.constant : numerical ? parameters[n.input_index] : 0.0);
      geometry.payload[id] = constant || numerical; continue;
    }
    ops::Arguments<ops::Value> args(n.parent_count); std::uint32_t available = 0;
    for (std::size_t j = 0; j < n.parent_count; ++j) {
      args[j] = geometry.values[n.parents[j]];
      if (geometry.payload[n.parents[j]]) available |= 1u << j;
    }
    const auto &spec = ops::lookup(n.opcode);
    const auto structure = ops::validate_structure(spec, args.data(), n.parent_count, UINT32_MAX, available);
    if (!structure.geometry_known) return false;
    v = ops::Value::number(0.0); v.kind = structure.output_kind; v.shape = structure.output_shape;
    v.set_contiguous_strides();
    // Only real coefficient transposes receive data. No numeric post node or
    // full-sized scan/asset intermediate is evaluated during physical binding.
    if (numerical && (id == proof.coefficients || id == proof.prefix.coefficients) &&
        operation(p, static_cast<std::uint32_t>(id), ops::Op::transpose, 1)) {
      const auto prepared = ops::prepare(spec, args.data(), n.parent_count);
      ops::require(prepared.borrowed, "INVALID_PHYSICAL_PAIR"); v = prepared.view; geometry.payload[id] = 1;
    }
  }
  geometry.steps = length; geometry.paths = 0; geometry.assets = 0; geometry.factors = 0;
  geometry.innovations = 0; geometry.rotation_reserve = 0; geometry.scale_cells = 0;
  geometry.combination_reserve = 0;
  for (const auto &g : proof.groups) {
    const auto &period = geometry.values[g.period];
    if (length != g.steps || !length || period.shape.rank != 2 || period.shape.dim[0] != length ||
        !period.shape.dim[1] || g.samples > period.shape.dim[1]) return false;
    if (geometry.paths && geometry.paths != period.shape.dim[1]) return false;
    geometry.paths = period.shape.dim[1];
    if (geometry.values[g.initial].shape.rank != 0 || geometry.values[g.probability].shape.rank != 1 ||
        geometry.values[g.fan].shape.rank != 2 || geometry.values[g.sample].shape.rank != 2 ||
        !(geometry.values[g.terminal_slice].shape == ops::matrix_shape(1, geometry.paths)) ||
        !(geometry.values[g.maximum].shape == ops::vector_shape(geometry.paths))) return false;
  }
  if (proof.affine != absent) {
    const auto &x = geometry.values[proof.source], &k = geometry.values[proof.coefficients],
               &initial = geometry.values[proof.asset_initial];
    if (x.shape.rank != 3 || x.shape.dim[0] != length || x.shape.dim[1] != geometry.paths || k.shape.rank != 2) return false;
    geometry.assets = k.shape.dim[1]; geometry.factors = x.shape.dim[2];
    if (geometry.assets > maximum_assets || k.shape.dim[0] != geometry.factors ||
        !(initial.shape.rank == 0 || (initial.shape.rank == 1 && initial.size() == geometry.assets))) return false;
    for (const auto &g : proof.groups) {
      const auto &w = geometry.values[g.weights];
      if (w.shape.rank != 1 || w.size() != geometry.assets || g.contributions.size() > geometry.assets) return false;
      for (const auto &c : g.contributions) if (c.asset >= geometry.assets) return false;
    }
    geometry.coefficient_reserve = affine_producer_coefficient_reserve(
        mul(length, geometry.paths), geometry.factors, geometry.assets);
  } else geometry.coefficient_reserve = 0;
  // Start with states bounded by the source's sequential time/path extent.
  // A proved raw prefix below may share more states within its tensor cost.
  geometry.batch = std::min(proof.groups.size(), std::max<std::size_t>(1, length / 3));
  if (proof.prefix.raw != absent) {
    const auto &r = proof.prefix;
    const auto &raw = geometry.values[r.raw], &rotation = geometry.values[r.coefficients],
        &means = geometry.values[r.means], &scale = geometry.values[r.scale];
    if (raw.shape.rank != 3 || raw.shape.dim[0] != length || raw.shape.dim[1] != geometry.paths ||
        rotation.shape.rank != 2 || rotation.shape.dim[0] != raw.shape.dim[2] ||
        rotation.shape.dim[1] != geometry.factors ||
        !(means.shape.rank == 0 || (r.means_view != absent && means.shape.rank == 1 &&
                                  means.size() == geometry.factors))) return false;
    const auto &shape = geometry.values[proof.source].shape;
    if (!(geometry.values[r.rotation].shape == shape) || !(geometry.values[r.scaled].shape == shape)) return false;
    geometry.innovations = raw.shape.dim[2];
    geometry.rotation_reserve = affine_producer_coefficient_reserve(
        mul(length, geometry.paths), geometry.innovations, geometry.factors);
    if (r.chi != absent) {
      const auto expected = ops::tensor_shape(length, geometry.paths, 1);
      if (!(geometry.values[r.chi].shape == expected) || !(scale.shape == expected) ||
          geometry.values[r.df].shape.rank != 0) return false;
      geometry.scale_cells = mul(length, geometry.paths);
      for (auto id : r.scale_nodes)
        if (geometry.values[id].shape.rank && !(geometry.values[id].shape == expected)) return false;
    } else if (scale.shape.rank != 0) return false;
    if (geometry.batch != proof.groups.size()) {
      // Producing the common prefix once needs either all consumer states or
      // one transformed tensor beside the existing bounded state batch. The
      // extra full-batch state is (groups - batch) * (3 * paths + assets)
      // doubles; admit it only when it fits that tensor's time/path/factor
      // extent. Division keeps the comparison exact without overflowing the
      // left-hand product. Actual workspace extents remain checked below.
      const auto transformed_cells = mul(mul(length, geometry.paths), geometry.factors);
      const auto state_per_group = add(mul(3, geometry.paths), geometry.assets);
      const auto additional_groups = proof.groups.size() - geometry.batch;
      if (additional_groups > transformed_cells / state_per_group) return false;
      geometry.batch = proof.groups.size();
    }
    // Large short-horizon requests trade a deterministic producer replay for
    // fewer live NAV/peak/drawdown states. Keep small requests and long scans
    // shared once; their state saving does not justify repeated projection.
    // This dispatch changes neither raw-prefix admission nor per-group folds.
    constexpr std::size_t large_state_cells = 2 * 1024 * 1024; // 16 MiB.
    const auto live_state_cells = mul(geometry.batch, add(mul(3, geometry.paths), geometry.assets));
    if (length < proof.groups.size() && live_state_cells >= large_state_cells)
      geometry.batch = std::min<std::size_t>(geometry.batch, 4);
  }
#ifdef CALMETRICS_COMBINATION_PACKETS
  if (proof.affine != absent && geometry.assets && geometry.batch >= combination_packet_width &&
      geometry.batch % combination_packet_width == 0) {
    const auto weight_cells = mul(geometry.batch, geometry.assets);
    // The work threshold is only a dispatch comparison, not a new full-tensor
    // allocation extent. Division avoids overflowing a large operation count.
    const auto rows_needed = (combination_minimum_updates - 1) / weight_cells + 1;
    bool dense_unique = true;
    for (const auto &g : proof.groups) {
      dense_unique &= g.contributions.size() == geometry.assets;
      for (std::size_t asset = 0; dense_unique && asset < geometry.assets; ++asset)
        dense_unique &= g.contributions[asset].asset == asset;
    }
    if (dense_unique && geometry.batch < combination_reserve_limit &&
        weight_cells <= combination_reserve_limit - geometry.batch &&
        mul(length, geometry.paths) >= rows_needed)
      geometry.combination_reserve = add(weight_cells, geometry.batch);
  }
#endif
  geometry.doubles = add(mul(mul(geometry.batch, geometry.paths), 3), geometry.paths);
  geometry.doubles = add(geometry.doubles, geometry.assets);
  geometry.doubles = add(geometry.doubles, mul(geometry.batch, geometry.assets));
  geometry.doubles = add(geometry.doubles, geometry.coefficient_reserve);
  if (proof.prefix.raw != absent) {
    geometry.doubles = add(geometry.doubles, geometry.factors);
    geometry.doubles = add(geometry.doubles, geometry.rotation_reserve);
    geometry.doubles = add(geometry.doubles, geometry.scale_cells);
  }
  geometry.doubles = add(geometry.doubles, geometry.combination_reserve);
  return true;
}

std::size_t program_metadata_bytes(const Program &p) {
  auto bytes = add(sizeof(Program), mul(p.nodes.capacity(), sizeof(Node)));
  // Count actual tail ownership, even when a manually constructed program has
  // written an unused slot. Arity alone does not prove zero retained storage.
  for (const auto &node : p.nodes) bytes = add(bytes, node.parents.overflow_bytes());
  bytes = add(bytes, mul(p.roots.capacity(), sizeof(std::uint32_t)));
  bytes = add(bytes, p.input_axes.capacity());
  bytes = add(bytes, mul(p.root_outputs.capacity(), sizeof(RootOutput)));
  if (p.execution_metadata) {
    const auto &m = *p.execution_metadata;
    bytes = add(bytes, sizeof(ExecutionMetadata));
    bytes = add(bytes, mul(m.root_destinations.capacity() + m.deferred_to.capacity() + m.fused_from.capacity(), sizeof(std::int32_t)));
    bytes = add(bytes, mul(m.specs.capacity(), sizeof(const ops::Spec *)));
    bytes = add(bytes, m.summary_consumers.capacity() + m.order_consumers.capacity() + m.position_status_reachable.capacity());
    bytes = add(bytes, mul(m.pointwise_regions.capacity(), sizeof(PointwiseRegion)));
    bytes = add(bytes, mul(m.pointwise_slots.capacity(), sizeof(std::uint32_t)));
    bytes = add(bytes, mul(m.pointwise_root_copies.capacity(), sizeof(std::size_t)));
  }
  return bytes;
}
std::size_t metadata_bytes(const Program &p, const CompactPath &proof) {
  auto bytes = add(sizeof(CompactPath), mul(proof.groups.capacity(), sizeof(Group)));
  bytes = add(bytes, mul(proof.producer_nodes.capacity(), sizeof(std::uint32_t)));
  bytes = add(bytes, mul(proof.shared_views.capacity(), sizeof(std::uint32_t)));
  bytes = add(bytes, mul(proof.prefix.nodes.capacity() + proof.prefix.scale_nodes.capacity(), sizeof(std::uint32_t)));
  if (proof.prefix.raw != absent) bytes = add(bytes, sizeof(ops::Workspace) + sizeof(ops::Prepared) + sizeof(ops::Audit));
  std::size_t child_inputs = 0, child_nodes = 0, child_roots = 0;
  for (const auto &g : proof.groups) {
    bytes = add(bytes, mul(g.stream_nodes.capacity() + g.original_nodes.capacity() + g.input_nodes.capacity(), sizeof(std::uint32_t)));
    bytes = add(bytes, mul(g.contributions.capacity(), sizeof(Contribution)));
    bytes = add(bytes, mul(g.fan_roots.capacity() + g.sample_roots.capacity() + g.post_roots.capacity(), sizeof(std::size_t)));
    bytes = add(bytes, program_metadata_bytes(g.post));
    child_inputs = std::max(child_inputs, g.post.input_count);
    child_nodes = std::max(child_nodes, g.post.nodes.size()); child_roots = std::max(child_roots, g.post.roots.size());
  }
  // Geometry descriptors, original/cloned error maps and one reused post input
  // descriptor vector coexist with the stream workspace and child Scratch.
  bytes = add(bytes, mul(p.nodes.size(), sizeof(ops::Value) + sizeof(std::uint8_t) + sizeof(const char *)));
  bytes = add(bytes, mul(child_inputs, sizeof(ops::Value)));
  bytes = add(bytes, mul(child_nodes, sizeof(const char *)));
  bytes = add(bytes, mul(child_roots, sizeof(ResultSlot) + sizeof(ops::MemoryOrder) +
      sizeof(ops::Shape) + sizeof(std::int16_t)));
  return add(bytes, sizeof(ResultLayout) + 4 * sizeof(std::size_t));
}
void child_inputs(const Group &g, const Geometry &geometry, std::vector<ops::Value> &inputs) {
  inputs.resize(g.input_nodes.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = geometry.values[g.input_nodes[i]];
}
} // namespace

std::shared_ptr<const CompactPath> plan_compact_path(const Program &p) {
  if (p.output_kind != OutputKind::typed || p.isolate_errors || p.input_axes.empty() ||
      !p.apply_scopes.empty() || !p.rolling_scopes.empty() || p.root_outputs.size() != p.roots.size()) return {};
  for (const auto &r : p.root_outputs) if (r.dtype != OutputDType::float64) return {};
  for (const auto &n : p.nodes)
    if (n.kind != NodeKind::input && n.kind != NodeKind::constant &&
        n.kind != NodeKind::parameter && n.kind != NodeKind::operation) return {};
  std::vector<std::vector<std::uint32_t>> children(p.nodes.size());
  for (std::uint32_t id = 0; id < p.nodes.size(); ++id)
    for (std::size_t j = 0; j < p.nodes[id].parent_count; ++j) children[p.nodes[id].parents[j]].push_back(id);
  auto proof = std::make_shared<CompactPath>();
  for (std::uint32_t id = 0; id < p.nodes.size(); ++id) {
    if (!operation(p, id, ops::Op::cumulative_product_axis, 3)) continue;
    Group group;
    if (!match_group(p, id, children, group)) return {};
    proof->groups.push_back(std::move(group));
  }
  if (proof->groups.empty() || !match_producer(p, *proof) ||
      !match_contributions(p, *proof) || !build_posts(p, *proof)) return {};
  return proof;
}

std::optional<CompactPathBudget> compact_path_budget(const Program &p, const CompactPath &proof,
    const std::vector<ops::Value> &inputs, const std::vector<std::size_t> &row_lengths,
    const CompactChildBudget &estimate_child) {
  if (!estimate_child || proof.groups.empty()) return std::nullopt;
  CompactPathBudget budget; budget.metadata_bytes = metadata_bytes(p, proof);
  budget.metadata_bytes = add(budget.metadata_bytes, mul(row_lengths.size(), sizeof(std::size_t)));
  Geometry geometry; std::vector<ops::Value> post_inputs;
  std::size_t max_inputs = 0;
  for (const auto &g : proof.groups) max_inputs = std::max(max_inputs, g.post.input_count);
  post_inputs.reserve(max_inputs);
  for (auto length : row_lengths) {
    if (!bind_geometry(p, proof, inputs, 0, length, nullptr, false, geometry)) return std::nullopt;
    budget.workspace_doubles = std::max(budget.workspace_doubles, geometry.doubles);
    budget.batch_size = geometry.batch;
    for (const auto &g : proof.groups) {
      child_inputs(g, geometry, post_inputs);
      budget.child_scratch_bytes = std::max(budget.child_scratch_bytes, estimate_child(g.post, post_inputs, 0));
    }
    const auto n = static_cast<double>(mul(length, geometry.paths));
    const auto batches = add(proof.groups.size(), geometry.batch - 1) / geometry.batch;
    if (geometry.combination_reserve)
      budget.physical_work += 2.0 * static_cast<double>(proof.groups.size()) *
          static_cast<double>(geometry.assets); // One read/write weight pack per batch.
    budget.physical_work += static_cast<double>(batches) * n * static_cast<double>(geometry.assets) *
        (2.0 * static_cast<double>(geometry.factors) + 1.0);
    if (proof.prefix.raw != absent) {
      budget.physical_work += static_cast<double>(batches) * n * static_cast<double>(geometry.factors) *
          (2.0 * static_cast<double>(geometry.innovations) + 2.0);
      if (proof.prefix.chi != absent) budget.physical_work += n * 3.0 + 3.0;
    }
    for (const auto &g : proof.groups) {
      const auto q = geometry.values[g.probability].size();
      budget.physical_work += n * (static_cast<double>(geometry.assets) * 2.0 + 6.0) +
          static_cast<double>(length + 1) * (static_cast<double>(geometry.paths) *
          std::log2(static_cast<double>(std::max<std::size_t>(geometry.paths, 2))) + static_cast<double>(q) * 4.0) +
          static_cast<double>(g.post.nodes.size()) * static_cast<double>(geometry.paths);
    }
  }
  budget.total_bytes = add(add(mul(budget.workspace_doubles, sizeof(double)), budget.metadata_bytes), budget.child_scratch_bytes);
  return budget;
}

namespace {
#ifdef CALMETRICS_COMBINATION_PACKETS
bool pack_combination_weights(const CompactPath &proof, const Geometry &geometry,
    std::size_t first, std::size_t count, double *packed, Audit &audit) {
  bool supported = true;
  for (std::size_t asset = 0; asset < geometry.assets; ++asset)
    for (std::size_t lane = 0; lane < count; ++lane) {
      const auto value = geometry.values[proof.groups[first + lane].weights].f(asset);
      std::uint64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      const auto magnitude = bits & UINT64_C(0x7fffffffffffffff);
      supported &= magnitude < UINT64_C(0x7ff0000000000000) &&
          (magnitude == 0 || magnitude >= UINT64_C(0x0010000000000000)) &&
          bits != UINT64_C(0x8000000000000000);
      packed[asset * count + lane] = value;
    }
  audit.algorithm_copy_bytes += mul(mul(count, geometry.assets), sizeof(double));
  return supported;
}

void update_combination_packets(std::size_t assets, std::size_t count,
    const double *weights, const double *returns, double *periods,
    double *contributions, Audit &audit) {
  using Packet = Eigen::Array<double, combination_packet_width, 1>;
  using Read = Eigen::Map<const Packet, Eigen::Unaligned>;
  using Write = Eigen::Map<Packet, Eigen::Unaligned>;
  // Each output retains its ascending asset fold; SIMD is solely across
  // independent groups. Fixed-size products materialize before either add.
  for (std::size_t lane = 0; lane < count; lane += combination_packet_width) {
    Packet period = Packet::Zero();
    for (std::size_t asset = 0; asset < assets; ++asset) {
      const auto offset = asset * count + lane;
      const Packet product = Read(weights + offset) * returns[asset];
      Write contribution(contributions + offset);
      period += product;
      contribution += product;
    }
    Write destination(periods + lane);
    destination = period;
  }
  audit.vector_elements += assets * count; // Already bounded by the admitted reserve.
}
#endif

void materialize_scale(const Program &p, const CompactPath &proof, Geometry &geometry,
    double *storage, Audit &audit, std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled) {
  // All scale operations retain their original topological order and canonical
  // domain checks. The three private tensor results have no observer or extra
  // consumer; after each complete pass their dead storage can be reused.
  ops::Workspace work;
  for (auto id : proof.prefix.scale_nodes) {
    checkpoint(deadline, cancelled);
    const auto &node = p.nodes[id];
    ops::Arguments<ops::Value> args(node.parent_count);
    for (std::size_t j = 0; j < node.parent_count; ++j) args[j] = geometry.values[node.parents[j]];
    const auto prepared = ops::prepare(ops::lookup(node.opcode), args.data(), node.parent_count);
    ops::require(!prepared.borrowed && !prepared.scratch_doubles && !prepared.scratch_indices,
        "INVALID_PHYSICAL_PAIR");
    ops::Output result; result.kind = prepared.output_kind; result.shape = prepared.output_shape;
    if (result.shape.rank) result.data = storage;
    ops::Audit scale_audit;
    ops::execute(prepared, result, work, ops::Isa::automatic, scale_audit);
    auto &value = geometry.values[id];
    value.scalar = result.scalar; value.data = result.data; value.set_contiguous_strides();
    geometry.payload[id] = 1;
    audit.vector_elements += scale_audit.vector_elements;
    audit.algorithm_copy_bytes += scale_audit.algorithm_copy_bytes;
  }
  if (proof.prefix.chi != absent) audit.compact_path_raw_scale_elements += geometry.scale_cells;
}
ops::Output root_output(const Program &p, const Group &g, std::size_t root,
    const Geometry &geometry, void *output, const ResultLayout &layout, std::size_t row) {
  (void)g;
  ops::Output destination;
  destination.kind = ops::Kind::number; destination.shape = geometry.values[p.roots[root]].shape;
  destination.order = layout.order(root);
  destination.data = static_cast<std::uint8_t *>(output) + layout.slots[row * p.roots.size() + root].byte_offset;
  return destination;
}
void write_fan(const Program &p, const Group &g, const Geometry &geometry,
    const double *sorted, std::size_t step, void *output, const ResultLayout &layout, std::size_t row) {
  const auto &probabilities = geometry.values[g.probability];
  for (auto root : g.fan_roots) {
    auto destination = root_output(p, g, root, geometry, output, layout, row);
    for (std::size_t q = 0; q < probabilities.size(); ++q)
      destination.set(q * (geometry.steps + 1) + step,
          ops::axis_path_detail::sorted_quantile(sorted, geometry.paths, probabilities.f(q)));
  }
}
void write_samples(const Program &p, const Group &g, const Geometry &geometry,
    const double *states, double seed, std::size_t step,
    void *output, const ResultLayout &layout, std::size_t row) {
  for (auto root : g.sample_roots) {
    auto destination = root_output(p, g, root, geometry, output, layout, row);
    for (std::size_t sample = 0; sample < g.samples; ++sample)
      destination.set(sample * (geometry.steps + 1) + step, step ? states[sample] : seed);
  }
}
bool valid_probabilities(const ops::Value &probabilities) {
  for (std::size_t q = 0; q < probabilities.size(); ++q) {
    const auto probability = probabilities.f(q);
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) return false;
  }
  return true;
}
ops::Value root_view(const Program &p, const Geometry &geometry,
    void *output, const ResultLayout &layout, std::size_t row, std::size_t root) {
  auto value = geometry.values[p.roots[root]];
  value.data = static_cast<std::uint8_t *>(output) + layout.slots[row * p.roots.size() + root].byte_offset;
  value.set_contiguous_strides(layout.order(root));
  return value;
}
void captured_inputs(const Program &p, const Group &g, const Geometry &geometry,
    const double *nav, const double *maximum, const double *contributions,
    std::size_t contribution_stride,
    void *output, const ResultLayout &layout, std::size_t row, std::vector<ops::Value> &inputs) {
  child_inputs(g, geometry, inputs);
  for (std::size_t input = 0; input < inputs.size(); ++input) {
    const auto id = g.input_nodes[input]; auto &value = inputs[input];
    if (id == g.fan) value = root_view(p, geometry, output, layout, row, g.fan_roots.front());
    else if (id == g.sample) value = root_view(p, geometry, output, layout, row, g.sample_roots.front());
    else if (id == g.terminal_slice) {
      value.data = nav; value.shape = ops::matrix_shape(1, geometry.paths); value.set_contiguous_strides();
    } else if (id == g.maximum) {
      value.data = maximum; value.shape = ops::vector_shape(geometry.paths); value.set_contiguous_strides();
    } else {
      const auto c = std::find_if(g.contributions.begin(), g.contributions.end(), [&](const Contribution &entry) {
        return entry.sum == id;
      });
      if (c != g.contributions.end())
        value = ops::Value::number(contributions[c->asset * contribution_stride]);
    }
  }
}
} // namespace

bool execute_compact_path(const Program &p, const CompactPath &proof,
    const std::vector<ops::Value> &inputs, const double *parameters, std::size_t parameter_count,
    const std::int64_t *starts, const std::int64_t *ends, std::size_t rows,
    void *output, const ResultLayout *layout, std::size_t result_row,
    ops::Workspace &workspace, Audit &audit, const CompactChildBudget &estimate_child,
    const CompactPostExecutor &execute_post, std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled) {
  if (!layout || !estimate_child || !execute_post || p.output_kind != OutputKind::typed ||
      p.isolate_errors || !p.apply_scopes.empty() || !p.rolling_scopes.empty()) return false;
  ops::require(inputs.size() == p.input_count, "GRAPH_INPUT_COUNT");
  ops::require(parameter_count == p.parameter_count, "GRAPH_PARAMETER_COUNT");
  ops::require(!parameter_count || parameters, "GRAPH_NULL_PARAMETERS");
  ops::require(!rows || (starts && ends && output), "GRAPH_NULL_BATCH");
  ops::require(layout->columns == p.roots.size() && !layout->row_bytes.empty() &&
      layout->row_statuses.size() == layout->row_bytes.size() &&
      result_row <= layout->row_bytes.size() - 1 && rows <= layout->row_bytes.size() - 1 - result_row &&
      layout->row_bytes[result_row + rows] <= static_cast<std::size_t>(PTRDIFF_MAX) &&
      layout->slots.size() >= mul(add(result_row, rows), p.roots.size()), "RESULT_LAYOUT_BOUNDS");
  ops::require(!rows || reinterpret_cast<std::uintptr_t>(output) % alignof(double) == 0, "RESULT_OUTPUT_ALIGNMENT");
  checkpoint(deadline, cancelled);
  std::vector<std::size_t> lengths; lengths.reserve(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    ops::require(starts[row] >= 0 && ends[row] >= starts[row], "GRAPH_INTERVAL_BOUNDS");
    lengths.push_back(static_cast<std::size_t>(ends[row] - starts[row]));
  }
  const auto budget = compact_path_budget(p, proof, inputs, lengths, estimate_child);
  if (!budget) return false;
  Geometry geometry;
  // Every interval, and every root slot, is proved before any output write.
  for (std::size_t row = 0; row < rows; ++row) {
    if (!bind_geometry(p, proof, inputs, static_cast<std::size_t>(starts[row]), lengths[row], nullptr, false, geometry)) return false;
    for (std::size_t root = 0; root < p.roots.size(); ++root) {
      const auto &shape = geometry.values[p.roots[root]].shape;
      const auto &slot = layout->slots[(result_row + row) * p.roots.size() + root];
      ops::require(p.root_outputs[root].dtype == OutputDType::float64 &&
          shape.rank == p.root_outputs[root].rank && shape.size() <= slot.capacity, "RESULT_CAPACITY_MISMATCH");
      ops::require(slot.byte_offset % alignof(double) == 0, "RESULT_OUTPUT_ALIGNMENT");
      const auto limit = root + 1 < p.roots.size()
          ? layout->slots[(result_row + row) * p.roots.size() + root + 1].byte_offset
          : layout->row_bytes[result_row + row + 1];
      ops::require(slot.byte_offset >= layout->row_bytes[result_row + row] && slot.byte_offset <= limit &&
          mul(slot.capacity, sizeof(double)) <= limit - slot.byte_offset, "RESULT_LAYOUT_BOUNDS");
    }
  }
  // Equal/smaller requests must not inherit a previous larger worker high-water.
  if (workspace.doubles.capacity() != budget->workspace_doubles) std::vector<double>().swap(workspace.doubles);
  if (workspace.indices.capacity()) std::vector<std::size_t>().swap(workspace.indices);
  workspace.ensure(budget->workspace_doubles, 0);
  audit = {}; audit.rows = rows; audit.nodes = p.nodes.size();
  audit.result_shapes.resize(mul(rows, p.roots.size()));
  audit.root_statuses.resize(mul(rows, p.roots.size()));
  std::vector<const char *> original_errors(p.nodes.size(), nullptr), deferred;
  std::vector<ops::Value> post_inputs;
  std::size_t max_inputs = 0, max_nodes = 0;
  for (const auto &g : proof.groups) {
    max_inputs = std::max(max_inputs, g.post.input_count); max_nodes = std::max(max_nodes, g.post.nodes.size());
  }
  post_inputs.reserve(max_inputs); deferred.reserve(max_nodes);
  ops::Value packed_coefficients, packed_rotation;
  bool packed_ready = false, rotation_ready = false;
  for (std::size_t row = 0; row < rows; ++row) {
    checkpoint(deadline, cancelled);
    ops::require(lengths[row] >= p.minimum_observations, "INSUFFICIENT_OBSERVATIONS");
    if (!bind_geometry(p, proof, inputs, static_cast<std::size_t>(starts[row]), lengths[row],
                       parameters, true, geometry)) throw ops::Error("INVALID_PHYSICAL_PAIR");
    std::fill(original_errors.begin(), original_errors.end(), nullptr);
    audit.max_window = std::max(audit.max_window, lengths[row]);
    auto *storage = workspace.doubles.data();
    const auto lane_cells = mul(geometry.batch, geometry.paths);
    auto *nav = storage, *peak = nav + lane_cells, *maximum = peak + lane_cells;
    auto *sorted = maximum + lane_cells, *asset_returns = sorted + geometry.paths;
    auto *contributions = asset_returns + geometry.assets;
    auto *rotation_row = contributions + mul(geometry.batch, geometry.assets);
    auto *scale_storage = rotation_row + (proof.prefix.raw != absent ? geometry.factors : 0);
    auto *coefficient_storage = scale_storage + geometry.scale_cells;
    auto *rotation_storage = coefficient_storage + geometry.coefficient_reserve;
    auto *combination_weights = rotation_storage + geometry.rotation_reserve;
    auto *combination_periods = combination_weights +
        (geometry.combination_reserve ? mul(geometry.batch, geometry.assets) : 0);
    const bool packets = affine_producer_packets_enabled(geometry.coefficient_reserve);
    const ops::Value *coefficients = proof.affine != absent ? &geometry.values[proof.coefficients] : nullptr;
    if (packets && coefficients->stride[1] != 1) {
      if (!packed_ready) {
        auto *packed = coefficient_storage;
        for (std::size_t f = 0; f < geometry.factors; ++f)
          for (std::size_t a = 0; a < geometry.assets; ++a) packed[f * geometry.assets + a] = coefficients->at(f, a);
        packed_coefficients = *coefficients; packed_coefficients.data = packed;
        packed_coefficients.set_contiguous_strides(); packed_ready = true;
        audit.algorithm_copy_bytes += mul(geometry.coefficient_reserve, sizeof(double));
        if (proof.prefix.raw != absent)
          audit.compact_path_raw_pack_bytes += mul(geometry.coefficient_reserve, sizeof(double));
      }
      coefficients = &packed_coefficients;
    }
    const bool rotation_packets = affine_producer_packets_enabled(geometry.rotation_reserve);
    const ops::Value *rotation_coefficients = nullptr;
    if (proof.prefix.raw != absent) {
      rotation_coefficients = &geometry.values[proof.prefix.coefficients];
      if (rotation_packets && rotation_coefficients->stride[1] != 1) {
        if (!rotation_ready) {
          for (std::size_t i = 0; i < geometry.innovations; ++i)
            for (std::size_t f = 0; f < geometry.factors; ++f)
              rotation_storage[i * geometry.factors + f] = rotation_coefficients->at(i, f);
          packed_rotation = *rotation_coefficients; packed_rotation.data = rotation_storage;
          packed_rotation.set_contiguous_strides(); rotation_ready = true;
          audit.algorithm_copy_bytes += mul(geometry.rotation_reserve, sizeof(double));
          audit.compact_path_raw_pack_bytes += mul(geometry.rotation_reserve, sizeof(double));
        }
        rotation_coefficients = &packed_rotation;
      }
      materialize_scale(p, proof, geometry, scale_storage, audit, deadline, cancelled);
    }
    // Prefix errors are checked before any consumer/result write. There is no
    // numerical fallback after the complete owned workspace was admitted.
    for (std::size_t root = 0; root < p.roots.size(); ++root) {
      audit.result_shapes[row * p.roots.size() + root] = geometry.values[p.roots[root]].shape;
      const auto &slot = layout->slots[(result_row + row) * p.roots.size() + root];
      const auto used = add(slot.byte_offset, mul(geometry.values[p.roots[root]].size(), sizeof(double)));
      const auto limit = root + 1 < p.roots.size()
          ? layout->slots[(result_row + row) * p.roots.size() + root + 1].byte_offset
          : layout->row_bytes[result_row + row + 1];
      if (limit > used) std::memset(static_cast<std::uint8_t *>(output) + used, 0, limit - used);
    }
    for (std::size_t first = 0; first < proof.groups.size(); first += geometry.batch) {
      checkpoint(deadline, cancelled);
      const auto count = std::min(geometry.batch, proof.groups.size() - first);
      bool combination_packets = false;
#ifdef CALMETRICS_COMBINATION_PACKETS
      if (geometry.combination_reserve && count % combination_packet_width == 0 &&
          std::fegetround() == FE_TONEAREST)
        combination_packets = pack_combination_weights(
            proof, geometry, first, count, combination_weights, audit);
#endif
      std::fill(contributions, contributions + count * geometry.assets, 0.0);
      for (std::size_t lane = 0; lane < count; ++lane) {
        const auto &g = proof.groups[first + lane]; const auto seed = geometry.values[g.initial].scalar;
        std::fill(nav + lane * geometry.paths, nav + (lane + 1) * geometry.paths, seed);
        std::fill(peak + lane * geometry.paths, peak + (lane + 1) * geometry.paths, seed);
        std::fill(maximum + lane * geometry.paths, maximum + (lane + 1) * geometry.paths,
                  -std::numeric_limits<double>::infinity());
        if (!valid_probabilities(geometry.values[g.probability])) original_errors[g.quantiles] = "INVALID_PARAMETER";
        else if (geometry.values[g.probability].size()) {
          // Even an equal initial column uses the canonical quantile products;
          // replacing it by a literal seed would change some last bits.
          std::fill(sorted, sorted + geometry.paths, seed);
          sort_line(sorted, geometry.paths, audit);
          write_fan(p, g, geometry, sorted, 0, output, *layout, result_row + row);
          ++audit.order_stat_sorts; audit.algorithm_copy_bytes += mul(geometry.paths, sizeof(double));
        }
        write_samples(p, g, geometry, nullptr, seed, 0, output, *layout, result_row + row);
      }
      for (std::size_t step = 0; step < geometry.steps; ++step) {
        checkpoint(deadline, cancelled);
        for (std::size_t path = 0; path < geometry.paths; ++path) {
          if ((path & 1023u) == 0) checkpoint(deadline, cancelled);
          if (proof.prefix.raw != absent) {
            const auto &r = proof.prefix; const auto &raw = geometry.values[r.raw];
            std::ptrdiff_t base = 0;
            if (geometry.innovations) base = static_cast<std::ptrdiff_t>(step) * raw.stride[0] +
                static_cast<std::ptrdiff_t>(path) * raw.stride[1];
            affine_producer_linear_row(geometry.innovations, geometry.factors, *rotation_coefficients,
                geometry.values[r.initial], static_cast<const double *>(raw.data), base, raw.stride[2],
                rotation_row, rotation_packets, audit);
            const auto scale = r.chi == absent ? geometry.values[r.scale].scalar
                : scale_storage[step * geometry.paths + path];
            const auto &means = geometry.values[r.means];
            for (std::size_t f = 0; f < geometry.factors; ++f) {
              const double scaled = rotation_row[f] * scale;
              rotation_row[f] = means.f(f) + scaled;
            }
            ++audit.compact_path_raw_rows;
          }
          if (proof.affine != absent && geometry.assets) {
            const auto &x = geometry.values[proof.source];
            std::ptrdiff_t base = 0;
            if (geometry.factors) base = static_cast<std::ptrdiff_t>(step) * x.stride[0] +
                static_cast<std::ptrdiff_t>(path) * x.stride[1];
            affine_producer_return_row(geometry.factors, geometry.assets, *coefficients,
                geometry.values[proof.asset_initial], proof.prefix.raw != absent ? rotation_row : static_cast<const double *>(x.data),
                proof.prefix.raw != absent ? 0 : base, proof.prefix.raw != absent ? 1 : x.stride[2],
                asset_returns, packets, audit);
          }
#ifdef CALMETRICS_COMBINATION_PACKETS
          if (combination_packets)
            update_combination_packets(geometry.assets, count, combination_weights,
                asset_returns, combination_periods, contributions, audit);
#endif
          for (std::size_t lane = 0; lane < count; ++lane) {
            const auto &g = proof.groups[first + lane]; double period = 0.0;
            if (combination_packets) period = combination_periods[lane];
            else if (proof.affine == absent) period = geometry.values[g.period].at(step, path);
            else {
              const auto &weights = geometry.values[g.weights]; std::size_t next = 0;
              for (std::size_t asset = 0; asset < geometry.assets; ++asset) {
                const double product = weights.f(asset) * asset_returns[asset];
                period += product;
                if (next < g.contributions.size() && g.contributions[next].asset == asset) {
                  // Exactly one T->P left fold for this shared weight-first
                  // product. Duplicate sum consumers read the same final value.
                  contributions[lane * geometry.assets + asset] += product;
                  while (next < g.contributions.size() && g.contributions[next].asset == asset) ++next;
                }
              }
            }
            const auto cell = lane * geometry.paths + path;
            nav[cell] = ops::axis_path_detail::product_step(nav[cell], 1.0 + period);
            peak[cell] = ops::axis_path_detail::strict_max_step(peak[cell], nav[cell]);
            if (peak[cell] == 0.0) original_errors[g.division] = "DIVIDE_BY_ZERO";
            else {
              const double drawdown = 1.0 - nav[cell] / peak[cell];
              if (!std::isfinite(drawdown)) original_errors[g.maximum] = "NON_FINITE_RESULT";
              maximum[cell] = ops::axis_path_detail::strict_max_step(maximum[cell], drawdown);
            }
          }
        }
        for (std::size_t lane = 0; lane < count; ++lane) {
          const auto &g = proof.groups[first + lane]; auto *states = nav + lane * geometry.paths;
          if (!original_errors[g.quantiles] && geometry.values[g.probability].size()) {
            std::copy_n(states, geometry.paths, sorted);
            sort_line(sorted, geometry.paths, audit);
            write_fan(p, g, geometry, sorted, step + 1, output, *layout, result_row + row);
            ++audit.order_stat_sorts; audit.algorithm_copy_bytes += mul(geometry.paths, sizeof(double));
          }
          write_samples(p, g, geometry, states, 0.0, step + 1, output, *layout, result_row + row);
        }
        ++audit.compact_path_steps;
      }
      // Strict post execution replays errors in the original node order. It
      // owns a separate Scratch and cannot release/overwrite these captures.
      for (std::size_t lane = 0; lane < count; ++lane) {
        const auto &g = proof.groups[first + lane];
        captured_inputs(p, g, geometry, nav + lane * geometry.paths, maximum + lane * geometry.paths,
            contributions + (combination_packets ? lane : lane * geometry.assets),
            combination_packets ? count : 1, output, *layout, result_row + row, post_inputs);
        deferred.resize(g.original_nodes.size());
        for (std::size_t node = 0; node < deferred.size(); ++node) deferred[node] = original_errors[g.original_nodes[node]];
        execute_post(g.post, post_inputs, parameters, parameter_count, deferred, output, *layout,
                     result_row + row, g.post_roots, audit, deadline, cancelled);
        for (auto root : g.fan_roots) audit.direct_output_bytes += mul(geometry.values[p.roots[root]].size(), sizeof(double));
        for (auto root : g.sample_roots) audit.direct_output_bytes += mul(geometry.values[p.roots[root]].size(), sizeof(double));
        ++audit.compact_path_groups;
      }
      ++audit.compact_path_batches;
    }
  }
  audit.operator_workspace_capacity_bytes = add(audit.operator_workspace_capacity_bytes, workspace.capacity_bytes());
  audit.pointwise_workspace_capacity_bytes = add(workspace.capacity_bytes(), budget->metadata_bytes);
  return true;
}

} // namespace calmetrics_engine::graph
