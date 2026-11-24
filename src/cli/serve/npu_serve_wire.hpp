#pragma once
// M6 status plane: populate + probe the NPU model registry for the four
// in-process NPU-GEMM servables (asr thinker / tts stack / flux / z-image).
// Descriptors only — no weights are loaded here, never boot autoload.
// Header-only, mirrors npu_model_registry.hpp style.

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_model_registry.hpp"

namespace mbufo::server::npu_wire {

using mbufo::core::npu::NpuArtifactKind;
using mbufo::core::npu::NpuModelLane;
using mbufo::core::npu::NpuModelRegistry;
using mbufo::core::npu::NpuModelSpec;

[[nodiscard]] inline const char* NpuLaneName(NpuModelLane lane) noexcept {
  switch (lane) {
    case NpuModelLane::kEmbedding:
      return "embedding";
    case NpuModelLane::kReranker:
      return "reranker";
    case NpuModelLane::kDecide:
      return "decide";
    case NpuModelLane::kAsr:
      return "asr";
    case NpuModelLane::kTts:
      return "tts";
    case NpuModelLane::kImage:
      return "image";
  }
  return "unknown";
}

[[nodiscard]] inline const char* NpuKindName(NpuArtifactKind kind) noexcept {
  switch (kind) {
    case NpuArtifactKind::kIronXclbin:
      return "iron_xclbin";
    case NpuArtifactKind::kFlmSidecar:
      return "flm_sidecar";
    case NpuArtifactKind::kGufoSidecar:
      return "gufo_sidecar";
    case NpuArtifactKind::kNpuInProcess:
      return "npu_in_process";
  }
  return "unknown";
}

struct NpuEntryDef {
  const char* id;        // primary: must match model_mapping.json id
  const char* alias;     // short alias
  NpuModelLane lane;     // kAsr / kTts / kImage
  const char* artifact;  // model root dir (weights present check)
  std::size_t resident_mib;  // M3 peak-RSS estimate (time -v, 2026-11-24)
};

[[nodiscard]] inline std::vector<NpuEntryDef> NpuInProcessEntries() {
  return {
      {"qwen3-asr-1.7b", "asr-thinker", NpuModelLane::kAsr,
       "/mnt/nvme-data/mbufo-models/qwen3-asr-1.7b", 520},
      {"qwen3-tts-12hz-1.7b", "tts-stack", NpuModelLane::kTts,
       "/mnt/nvme-data/mbufo-models/qwen3-tts-1.7b-base", 2200},
      {"flux2-klein-4b", "flux", NpuModelLane::kImage,
       "/mnt/nvme-data/mbufo-models/flux2-klein-4b/transformer", 930},
      {"z-image-turbo", "zimage", NpuModelLane::kImage,
       "/mnt/nvme-data/mbufo-models/z-image-turbo/transformer", 750},
  };
}

/// Register the four entries. Idempotent: existing ids are skipped.
inline void PopulateNpuRegistry(NpuModelRegistry* reg, std::string* error) {
  if (reg == nullptr) {
    if (error != nullptr) *error = "null npu registry";
    return;
  }
  for (const auto& def : NpuInProcessEntries()) {
    if (reg->HasId(def.id)) continue;
    NpuModelSpec spec;
    spec.lane = def.lane;
    spec.kind = NpuArtifactKind::kNpuInProcess;
    spec.artifact = def.artifact;
    spec.endpoint = "";
    spec.default_port = 0;
    spec.estimated_resident_mib = def.resident_mib;
    spec.launch_cmd = "";
    std::string add_err;
    if (!reg->Add(def.id, {def.alias}, spec, /*ready=*/false, &add_err) &&
        error != nullptr) {
      *error = add_err;
    }
  }
}

/// Light probe: weights dir present AND NPU device present. No model load,
// no NPU execution — full verification lives in the npu_*_dit/decode tests.
[[nodiscard]] inline bool ProbeNpuEntry(const NpuModelSpec& spec) noexcept {
  std::error_code ec;
  if (spec.artifact.empty() ||
      !std::filesystem::is_directory(spec.artifact, ec) || ec) {
    return false;
  }
  return std::filesystem::exists("/dev/accel/accel0", ec) && !ec;
}

/// Refresh `ready` for every registered entry. Returns ready count.
inline std::size_t RefreshNpuReady(NpuModelRegistry* reg) {
  if (reg == nullptr) return 0;
  std::size_t n = 0;
  for (const auto& info : reg->List()) {
    auto spec = reg->Resolve(info.id);
    const bool ok = spec != nullptr && ProbeNpuEntry(*spec);
    reg->SetReady(info.id, ok);
    n += ok ? 1 : 0;
  }
  return n;
}

/// dependency-free JSON snapshot for the HTTP layer (hand-rolled: ids
/// and paths contain no quotes/backslashes by construction).
[[nodiscard]] inline std::string NpuStatusJson(
    const NpuModelRegistry* reg) {
  std::ostringstream out;
  out << "{\"npu_models\":[";
  bool first = true;
  if (reg != nullptr) {
    for (const auto& info : reg->List()) {
      if (!first) out << ',';
      first = false;
      out << "{\"id\":\"" << info.id << "\",\"lane\":\""
          << NpuLaneName(info.lane) << "\",\"kind\":\""
          << NpuKindName(info.kind) << "\",\"artifact\":\""
          << info.artifact << "\",\"ready\":"
          << (info.ready ? "true" : "false") << ",\"is_default\":"
          << (info.is_default ? "true" : "false")
          << ",\"estimated_resident_mib\":"
          << info.estimated_resident_mib << ",\"names\":[";
      bool fn = true;
      for (const auto& n : info.names) {
        if (!fn) out << ',';
        fn = false;
        out << '"' << n << '"';
      }
      out << "]}";
    }
  }
  out << "]}";
  return out.str();
}

}  // namespace mbufo::server::npu_wire
