#pragma once
// Safetensors -> NPU chunked weight bridge, bfp16ebs8 prepacker, and
// EmbeddingGemma 2 shard planner.
//
// Features:
//   - Parses 8-byte header length + JSON metadata using mbufo::json::parse
//     without slurping multi-GB tensor payloads into RAM.
//   - Reads arbitrary byte chunks (`ReadChunk`) for layer-by-layer streaming.
//   - Encodes BF16/FP32 weight rows into AMD AIE2p `v8bfp16ebs8` (9 bytes per
//     8 elements: [shared_exp_uint8, int8 m0..m7]), matching `aie.utils.bfp.encode`
//     byte-for-byte, plus tiled BF16 transposition/prepacking.
//   - Plans NPU weight shards for EmbeddingGemma 2 (1.49 GB safetensors):
//       * `language_model.embed_tokens.weight` (256 MiB) + vision/audio towers
//         stay `kHostOnly` (CPU gather + optional multimodal side).
//       * 24 text layers + final norm + output projection partition into
//         <= `max_shard_bytes` NPU shards (20 sliding + 4 full attention).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/compute/npu/npu_pooling.hpp"
#include "src/core/json.hpp"

namespace mbufo::core::npu {

struct SafetensorEntry {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::uint64_t data_begin{0};
  std::uint64_t data_end{0};

  [[nodiscard]] std::uint64_t ByteSize() const noexcept {
    return (data_end >= data_begin) ? (data_end - data_begin) : 0u;
  }

  [[nodiscard]] std::size_t ElementCount() const noexcept {
    if (shape.empty()) return 1u;  // scalar tensor
    std::size_t n = 1u;
    for (std::size_t d : shape) n *= d;
    return n;
  }
};

class SafetensorsIndex {
 public:
  [[nodiscard]] static std::optional<SafetensorsIndex> Open(
      const std::string& path, std::string* error = nullptr) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
      if (error != nullptr) *error = "cannot open safetensors: " + path;
      return std::nullopt;
    }
    std::uint8_t len_buf[8]{};
    ifs.read(reinterpret_cast<char*>(len_buf), 8);
    if (ifs.gcount() != 8) {
      if (error != nullptr) *error = "truncated safetensors header length";
      return std::nullopt;
    }
    std::uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) {
      header_len |= static_cast<std::uint64_t>(len_buf[i]) << (8 * i);
    }
    constexpr std::uint64_t kMaxHeaderBytes = 64ull << 20;
    if (header_len == 0 || header_len > kMaxHeaderBytes) {
      if (error != nullptr) *error = "invalid safetensors header length";
      return std::nullopt;
    }
    std::string header_json(header_len, '\0');
    ifs.read(header_json.data(), static_cast<std::streamsize>(header_len));
    if (static_cast<std::uint64_t>(ifs.gcount()) != header_len) {
      if (error != nullptr) *error = "truncated safetensors JSON header";
      return std::nullopt;
    }

    json::Value root;
    try {
      root = json::parse(header_json);
    } catch (const std::exception& e) {
      if (error != nullptr) {
        *error = std::string("safetensors JSON parse failed: ") + e.what();
      }
      return std::nullopt;
    }
    if (!root.is_object()) {
      if (error != nullptr) *error = "safetensors header must be a JSON object";
      return std::nullopt;
    }

    SafetensorsIndex idx;
    idx.path_ = path;
    idx.data_base_offset_ = 8ull + header_len;
    for (const auto& [key, val] : root.members()) {
      if (key == "__metadata__") continue;
      if (!val.is_object()) continue;
      SafetensorEntry e;
      e.name = key;
      e.dtype = val.member_str("dtype");
      if (const json::Value* shp = val.find("shape");
          shp != nullptr && shp->is_array()) {
        for (const auto& dim : shp->items()) {
          e.shape.push_back(dim.as_size());
        }
      }
      const json::Value* offs = val.find("data_offsets");
      if (offs == nullptr || !offs->is_array() || offs->items().size() != 2u) {
        if (error != nullptr) {
          *error = "invalid data_offsets for tensor: " + key;
        }
        return std::nullopt;
      }
      const double b0 = offs->items()[0].as_double(-1.0);
      const double b1 = offs->items()[1].as_double(-1.0);
      if (b0 < 0.0 || b1 < b0) {
        if (error != nullptr) {
          *error = "negative or descending data_offsets for: " + key;
        }
        return std::nullopt;
      }
      e.data_begin = static_cast<std::uint64_t>(b0);
      e.data_end = static_cast<std::uint64_t>(b1);
      idx.by_name_[e.name] = idx.entries_.size();
      idx.entries_.push_back(std::move(e));
    }
    return idx;
  }

  [[nodiscard]] const std::string& path() const noexcept { return path_; }
  [[nodiscard]] std::uint64_t data_base_offset() const noexcept {
    return data_base_offset_;
  }
  [[nodiscard]] const std::vector<SafetensorEntry>& entries() const noexcept {
    return entries_;
  }

  [[nodiscard]] const SafetensorEntry* Find(std::string_view name) const {
    auto it = by_name_.find(std::string(name));
    return (it != by_name_.end()) ? &entries_[it->second] : nullptr;
  }

  [[nodiscard]] std::vector<const SafetensorEntry*> EntriesWithPrefix(
      std::string_view prefix) const {
    std::vector<const SafetensorEntry*> res;
    for (const auto& e : entries_) {
      if (e.name.starts_with(prefix)) res.push_back(&e);
    }
    return res;
  }

  /// Reads `[rel_offset, rel_offset + byte_count)` of `entry` from disk.
  [[nodiscard]] bool ReadChunk(const SafetensorEntry& entry,
                               std::uint64_t rel_offset, std::size_t byte_count,
                               std::vector<std::uint8_t>& out,
                               std::string* error = nullptr) const {
    const std::uint64_t total = entry.ByteSize();
    if (rel_offset > total || byte_count > total - rel_offset) {
      if (error != nullptr) {
        *error = "ReadChunk out of bounds for " + entry.name;
      }
      return false;
    }
    std::ifstream ifs(path_, std::ios::binary);
    if (!ifs) {
      if (error != nullptr) *error = "cannot reopen safetensors: " + path_;
      return false;
    }
    const std::uint64_t abs_off =
        data_base_offset_ + entry.data_begin + rel_offset;
    ifs.seekg(static_cast<std::streamoff>(abs_off), std::ios::beg);
    out.resize(byte_count);
    if (byte_count > 0) {
      ifs.read(reinterpret_cast<char*>(out.data()),
               static_cast<std::streamsize>(byte_count));
      if (static_cast<std::size_t>(ifs.gcount()) != byte_count) {
        if (error != nullptr) {
          *error = "short read on tensor chunk: " + entry.name;
        }
        return false;
      }
    }
    return true;
  }

 private:
  std::string path_;
  std::uint64_t data_base_offset_{0};
  std::vector<SafetensorEntry> entries_;
  std::unordered_map<std::string, std::size_t> by_name_;
};

/// Encodes `fp32` values (`size % 8 == 0`) into AIE2p `v8bfp16ebs8`:
/// 9 bytes per 8 values = `[uint8 shared_exponent, int8 m0..m7]`.
/// When `conv_even == false` (default), matches `aie.utils.bfp.encode` byte-for-byte.
[[nodiscard]] inline bool PackFloatToBfp16Ebs8(
    std::span<const float> vals, std::vector<std::uint8_t>& out,
    bool conv_even = false) {
  if (vals.empty() || (vals.size() % 8u != 0u)) return false;
  for (float v : vals) {
    if (!std::isfinite(v)) return false;
  }
  const std::size_t n_blocks = vals.size() / 8u;
  out.resize(n_blocks * 9u);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    std::uint32_t bits[8];
    bool sign[8];
    std::uint32_t exp[8];
    std::uint32_t mant[8];
    std::uint32_t max_exp = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      const float fv = vals[b * 8u + i];
      std::memcpy(&bits[i], &fv, sizeof(std::uint32_t));
      sign[i] = (bits[i] >> 31u) != 0u;
      exp[i] = (bits[i] >> 23u) & 0xFFu;
      mant[i] = (bits[i] & 0x7FFFFFu) | ((exp[i] != 0u) ? 0x800000u : 0u);
      max_exp = std::max(max_exp, exp[i]);
    }
    out[b * 9u] = static_cast<std::uint8_t>(max_exp);
    for (std::size_t i = 0; i < 8; ++i) {
      const std::uint32_t shift = max_exp - exp[i];
      std::int8_t m = 0;
      if (shift >= 32u) {
        m = sign[i] ? static_cast<std::int8_t>(-1) : static_cast<std::int8_t>(0);
      } else if (conv_even) {
        const std::int64_t s_mant = sign[i] ? -static_cast<std::int64_t>(mant[i])
                                            : static_cast<std::int64_t>(mant[i]);
        const unsigned denom_shift = std::min<unsigned>(17u + shift, 62u);
        const double q =
            std::rint(static_cast<double>(s_mant) / std::ldexp(1.0, denom_shift));
        const long qi = std::clamp<long>(static_cast<long>(q), -128L, 127L);
        m = static_cast<std::int8_t>(qi);
      } else {
        const std::uint32_t m32 =
            sign[i] ? static_cast<std::uint32_t>(-static_cast<std::int32_t>(mant[i]))
                    : mant[i];
        const std::int8_t base_m =
            static_cast<std::int8_t>((m32 >> 17u) & 0xFFu);
        const std::int32_t shifted =
            static_cast<std::int32_t>(base_m) >> std::min<std::uint32_t>(shift, 31u);
        m = static_cast<std::int8_t>(shifted);
      }
      out[b * 9u + 1u + i] = static_cast<std::uint8_t>(m);
    }
  }
  return true;
}

[[nodiscard]] inline bool PackBf16ToBfp16Ebs8(
    std::span<const std::uint16_t> bf16_vals, std::vector<std::uint8_t>& out,
    bool conv_even = false) {
  if (bf16_vals.empty() || (bf16_vals.size() % 8u != 0u)) return false;
  std::vector<float> f32(bf16_vals.size());
  for (std::size_t i = 0; i < bf16_vals.size(); ++i) {
    f32[i] = Bf16BitsToFloat(bf16_vals[i]);
  }
  return PackFloatToBfp16Ebs8(f32, out, conv_even);
}

[[nodiscard]] inline bool UnpackBfp16Ebs8ToFloat(
    std::span<const std::uint8_t> packed, std::vector<float>& out) {
  if (packed.empty() || (packed.size() % 9u != 0u)) return false;
  const std::size_t n_blocks = packed.size() / 9u;
  out.resize(n_blocks * 8u);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const int exp = static_cast<int>(packed[b * 9u]) - 127 - 6;
    const double scale = std::ldexp(1.0, exp);
    for (std::size_t i = 0; i < 8; ++i) {
      const std::int8_t m = static_cast<std::int8_t>(packed[b * 9u + 1u + i]);
      out[b * 8u + i] = static_cast<float>(static_cast<double>(m) * scale);
    }
  }
  return true;
}

/// Rearranges row-major `src[rows, cols]` into tiled blocks of `[tile_r, tile_c]`.
/// When `transpose == true`, logically reads transposed `[cols, rows]`.
[[nodiscard]] inline bool PrepackBf16MatrixTiled(
    std::span<const std::uint16_t> src, std::size_t rows, std::size_t cols,
    std::size_t tile_r, std::size_t tile_c, bool transpose,
    std::vector<std::uint16_t>& dst) {
  if (rows == 0 || cols == 0 || tile_r == 0 || tile_c == 0 ||
      src.size() != rows * cols) {
    return false;
  }
  const std::size_t eff_r = transpose ? cols : rows;
  const std::size_t eff_c = transpose ? rows : cols;
  if ((eff_r % tile_r != 0) || (eff_c % tile_c != 0)) return false;
  dst.resize(src.size());
  std::size_t out_idx = 0;
  for (std::size_t tr = 0; tr < eff_r; tr += tile_r) {
    for (std::size_t tc = 0; tc < eff_c; tc += tile_c) {
      for (std::size_t r = 0; r < tile_r; ++r) {
        for (std::size_t c = 0; c < tile_c; ++c) {
          const std::size_t sr = transpose ? (tc + c) : (tr + r);
          const std::size_t sc = transpose ? (tr + r) : (tc + c);
          dst[out_idx++] = src[sr * cols + sc];
        }
      }
    }
  }
  return true;
}

enum class WeightPlacement : std::uint8_t {
  kHostOnly = 0,
  kNpuBf16,
  kNpuBfp16Ebs8,
};

struct ShardTensorItem {
  std::string name;
  WeightPlacement placement{WeightPlacement::kHostOnly};
  std::size_t raw_bytes{0};
  std::size_t packed_bytes{0};
};

struct NpuShard {
  std::size_t shard_index{0};
  std::size_t start_layer{0};
  std::size_t end_layer{0};  // exclusive for layer shards
  std::size_t packed_bytes{0};
  std::vector<ShardTensorItem> tensors;
};

struct NpuShardPlan {
  std::vector<NpuShard> shards;
  std::size_t host_only_bytes{0};
  std::size_t npu_bf16_bytes{0};
  std::size_t npu_packed_bytes{0};
  std::size_t npu_tensor_count{0};
  std::size_t host_tensor_count{0};
  std::size_t layer_count{0};
};

/// Plans layer-sharded NPU residency for EmbeddingGemma 2 safetensors.
[[nodiscard]] inline NpuShardPlan PlanEmbGemma2NpuShards(
    const SafetensorsIndex& index, bool use_bfp16ebs8 = true,
    std::size_t max_shard_bytes = 64ull << 20) {
  NpuShardPlan plan{};
  NpuShard current{};
  current.shard_index = 0;
  std::size_t max_layer_seen = 0;
  bool any_layer_seen = false;

  auto flush_shard = [&]() {
    if (!current.tensors.empty()) {
      plan.shards.push_back(current);
      current = NpuShard{};
      current.shard_index = plan.shards.size();
    }
  };

  for (const auto& e : index.entries()) {
    const std::size_t raw = static_cast<std::size_t>(e.ByteSize());
    if (e.name.starts_with("vision_tower.") ||
        e.name.starts_with("audio_tower.") ||
        e.name == "embed_vision.embedding_projection.weight" ||
        e.name == "embed_audio.embedding_projection.weight" ||
        e.name == "language_model.embed_tokens.weight") {
      plan.host_only_bytes += raw;
      plan.host_tensor_count++;
      continue;
    }
    if (!e.name.starts_with("language_model.")) {
      plan.host_only_bytes += raw;
      plan.host_tensor_count++;
      continue;
    }

    const bool is_linear_2d =
        e.shape.size() == 2u && (e.ElementCount() % 8u == 0u);
    const WeightPlacement place =
        (use_bfp16ebs8 && is_linear_2d) ? WeightPlacement::kNpuBfp16Ebs8
                                        : WeightPlacement::kNpuBf16;
    const std::size_t packed =
        (place == WeightPlacement::kNpuBfp16Ebs8)
            ? (e.ElementCount() * 9u) / 8u
            : raw;

    std::size_t layer_id = 0;
    if (e.name.starts_with("language_model.layers.")) {
      const std::size_t pos = std::string_view("language_model.layers.").size();
      std::size_t end_pos = e.name.find('.', pos);
      if (end_pos != std::string::npos) {
        layer_id = static_cast<std::size_t>(
            std::stoul(e.name.substr(pos, end_pos - pos)));
        any_layer_seen = true;
        max_layer_seen = std::max(max_layer_seen, layer_id);
      }
    }

    if (max_shard_bytes > 0 && !current.tensors.empty() &&
        current.packed_bytes + packed > max_shard_bytes) {
      flush_shard();
    }
    if (current.tensors.empty()) {
      current.start_layer = layer_id;
    }
    current.end_layer = std::max(current.end_layer, layer_id + 1u);
    current.packed_bytes += packed;
    current.tensors.push_back(ShardTensorItem{
        .name = e.name,
        .placement = place,
        .raw_bytes = raw,
        .packed_bytes = packed,
    });
    if (place == WeightPlacement::kNpuBf16) {
      plan.npu_bf16_bytes += packed;
    }
    plan.npu_packed_bytes += packed;
    plan.npu_tensor_count++;
  }
  flush_shard();
  plan.layer_count = any_layer_seen ? (max_layer_seen + 1u) : 0u;
  return plan;
}

}  // namespace mbufo::core::npu
