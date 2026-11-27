#pragma once
// NPU shared sharded-safetensors weight reader (P1a).
//
// Promotes the previously TU-local `MultiShardSafetensors` (it lived inside
// npu_quant_models.cpp, reachable only by the parity shells) to a shared
// header so every servable NPU runtime (ASR thinker decode, TTS talker,
// Qwen3-4B TE, DiT blocks) streams weights the same way:
//
//   - OpenDir() indexes every .safetensors shard without loading tensors.
//   - ReadF32 / GatherRowsF32 / ReadRowSliceF32 pull only the rows needed,
//     keeping peak RAM < 2 GiB across full-layer runs.
//
// Behavior is byte-identical to the original TU-local version; follow-up
// switches npu_quant_models.cpp over to this header and deletes the copy.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/compute/npu/npu_pooling.hpp"
#include "src/core/compute/npu/npu_safetensors_bridge.hpp"

namespace mbufo::core::npu {

[[nodiscard]] inline float NpuFp16BitsToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
  const std::uint32_t exp = (h >> 10u) & 0x1Fu;
  const std::uint32_t mant = h & 0x03FFu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      float f = static_cast<float>(mant) * (1.0F / 16777216.0F);
      return (sign != 0) ? -f : f;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13u);
  } else {
    bits = sign | ((exp + 112u) << 23u) | (mant << 13u);
  }
  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof(float));
  return out;
}

class NpuShardReader {
 public:
  [[nodiscard]] static bool OpenDir(const std::string& dir,
                                    NpuShardReader* out, std::string* error) {
    out->shards_.clear();
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
      if (error != nullptr) *error = "directory does not exist: " + dir;
      return false;
    }
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
      if (!entry.is_regular_file()) continue;
      const std::string p = entry.path().string();
      if (p.ends_with(".safetensors")) paths.push_back(p);
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
      if (error != nullptr) *error = "no .safetensors found in " + dir;
      return false;
    }
    for (const auto& p : paths) {
      auto idx = SafetensorsIndex::Open(p, error);
      if (!idx.has_value()) return false;
      out->shards_.push_back(std::move(*idx));
    }
    return true;
  }

  [[nodiscard]] bool FindEntry(std::string_view name,
                              const SafetensorsIndex** out_shard,
                              const SafetensorEntry** out_entry) const {
    for (const auto& s : shards_) {
      if (const auto* e = s.Find(name); e != nullptr) {
        *out_shard = &s;
        *out_entry = e;
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool ReadF32(std::string_view name, std::size_t expected_elems,
                            std::vector<float>* out,
                            std::string* error) const {
    const SafetensorsIndex* shard = nullptr;
    const SafetensorEntry* entry = nullptr;
    if (!FindEntry(name, &shard, &entry)) {
      if (error != nullptr) *error = "missing safetensor: " + std::string(name);
      return false;
    }
    if (entry->ElementCount() != expected_elems) {
      if (error != nullptr) {
        *error = "element count mismatch for " + std::string(name) + ": got " +
                 std::to_string(entry->ElementCount()) + ", expected " +
                 std::to_string(expected_elems);
      }
      return false;
    }
    std::vector<std::uint8_t> raw;
    if (!shard->ReadChunk(*entry, 0, entry->ByteSize(), raw, error)) {
      return false;
    }
    out->resize(expected_elems);
    if (entry->dtype == "F32") {
      std::memcpy(out->data(), raw.data(), expected_elems * sizeof(float));
      return true;
    }
    if (entry->dtype == "BF16") {
      const auto* u16 = reinterpret_cast<const std::uint16_t*>(raw.data());
      for (std::size_t i = 0; i < expected_elems; ++i) {
        (*out)[i] = Bf16BitsToFloat(u16[i]);
      }
      return true;
    }
    if (entry->dtype == "F16") {
      const auto* u16 = reinterpret_cast<const std::uint16_t*>(raw.data());
      for (std::size_t i = 0; i < expected_elems; ++i) {
        (*out)[i] = NpuFp16BitsToFloat(u16[i]);
      }
      return true;
    }
    if (error != nullptr) {
      *error =
          "unsupported dtype " + entry->dtype + " in " + std::string(name);
    }
    return false;
  }

  /// Reads a subset of rows `ids` from a 2D embedding table `[vocab, dim]`
  /// without loading the entire table into RAM.
  [[nodiscard]] bool GatherRowsF32(std::string_view name,
                                   std::span<const std::uint32_t> ids,
                                   std::size_t dim, std::vector<float>* out,
                                   std::string* error) const {
    const SafetensorsIndex* shard = nullptr;
    const SafetensorEntry* entry = nullptr;
    if (!FindEntry(name, &shard, &entry)) {
      if (error != nullptr) {
        *error = "missing embedding tensor: " + std::string(name);
      }
      return false;
    }
    const std::size_t elem_bytes = (entry->dtype == "F32") ? 4u : 2u;
    const std::size_t row_bytes = dim * elem_bytes;
    out->resize(ids.size() * dim);
    std::vector<std::uint8_t> buf;
    for (std::size_t t = 0; t < ids.size(); ++t) {
      const std::uint64_t off = static_cast<std::uint64_t>(ids[t]) * row_bytes;
      if (!shard->ReadChunk(*entry, off, row_bytes, buf, error)) return false;
      float* dst = out->data() + t * dim;
      if (entry->dtype == "F32") {
        std::memcpy(dst, buf.data(), dim * sizeof(float));
      } else if (entry->dtype == "BF16") {
        const auto* u16 = reinterpret_cast<const std::uint16_t*>(buf.data());
        for (std::size_t c = 0; c < dim; ++c) dst[c] = Bf16BitsToFloat(u16[c]);
      } else {
        const auto* u16 = reinterpret_cast<const std::uint16_t*>(buf.data());
        for (std::size_t c = 0; c < dim; ++c) {
          dst[c] = NpuFp16BitsToFloat(u16[c]);
        }
      }
    }
    return true;
  }

  /// Reads contiguous rows `[row0, row0 + num_rows)` from `[total_rows, cols]`.
  [[nodiscard]] bool ReadRowSliceF32(std::string_view name, std::size_t row0,
                                     std::size_t num_rows, std::size_t cols,
                                     std::vector<float>* out,
                                     std::string* error) const {
    const SafetensorsIndex* shard = nullptr;
    const SafetensorEntry* entry = nullptr;
    if (!FindEntry(name, &shard, &entry)) {
      if (error != nullptr) *error = "missing tensor: " + std::string(name);
      return false;
    }
    const std::size_t elem_bytes = (entry->dtype == "F32") ? 4u : 2u;
    const std::uint64_t off =
        static_cast<std::uint64_t>(row0) * cols * elem_bytes;
    const std::size_t count = num_rows * cols;
    std::vector<std::uint8_t> raw;
    if (!shard->ReadChunk(*entry, off, count * elem_bytes, raw, error)) {
      return false;
    }
    out->resize(count);
    if (entry->dtype == "F32") {
      std::memcpy(out->data(), raw.data(), count * sizeof(float));
    } else if (entry->dtype == "BF16") {
      const auto* u16 = reinterpret_cast<const std::uint16_t*>(raw.data());
      for (std::size_t i = 0; i < count; ++i) {
        (*out)[i] = Bf16BitsToFloat(u16[i]);
      }
    } else {
      const auto* u16 = reinterpret_cast<const std::uint16_t*>(raw.data());
      for (std::size_t i = 0; i < count; ++i) {
        (*out)[i] = NpuFp16BitsToFloat(u16[i]);
      }
    }
    return true;
  }

 private:
  std::vector<SafetensorsIndex> shards_;
};

}  // namespace mbufo::core::npu
