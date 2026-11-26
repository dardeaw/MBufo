#include "src/cli/serve/model_admin_api.hpp"

#include "src/cli/serve/npu_sidecar_proxy.hpp"
#include "src/cli/serve/npu_serve_wire.hpp"
#include "src/cli/serve/admin_ops.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "mbufo_build_version.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/model_admin_ui.hpp"
#include "src/core/json.hpp"

namespace mbufo::server {
namespace {

std::uint64_t NowSteadyMs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

HttpResponse AdminOk(json::Value body) {
  return {.status = 200,
          .reason = "OK",
          .body = body.dump(),
          .headers = {{"Cache-Control", "no-store, no-cache, must-revalidate"}}};
}

HttpResponse AdminErr(int status, const std::string& reason,
                      const std::string& message, const std::string& code) {
  json::Value obj = json::Value::object();
  obj["message"] = message;
  obj["type"] = "admin_error";
  obj["code"] = code;
  json::Value e = json::Value::object();
  e["error"] = std::move(obj);
  return {.status = status, .reason = reason, .body = e.dump()};
}

std::string ToLowerAscii(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

// Strip multi-shard suffix "-00001-of-00004" when deriving a clean model ID.
std::string DeriveId(const std::string& path) {
  std::filesystem::path p(path);
  std::string name = p.stem().string();
  const auto pos = name.find("-00001-of-");
  if (pos != std::string::npos) {
    name = name.substr(0, pos);
  }
  if (name.empty()) name = path;
  return name;
}

bool IsNonFirstShard(std::string_view filename) {
  const auto pos = filename.find("-of-");
  if (pos == std::string_view::npos || pos < 6) return false;
  const auto shard_part = filename.substr(pos - 6, 6);
  if (shard_part[0] != '-') return false;
  for (std::size_t i = 1; i < 6; ++i) {
    if (shard_part[i] < '0' || shard_part[i] > '9') return false;
  }
  return shard_part != "-00001";
}

// Compute total GGUF bytes across all shards when given "-00001-of-NNNNN.gguf".
std::size_t TotalGgufBytes(const std::string& path) {
  std::error_code ec;
  std::filesystem::path p(path);
  const std::string fname = p.filename().string();
  const auto pos = fname.find("-00001-of-");
  if (pos != std::string::npos && p.has_parent_path()) {
    const std::string prefix = fname.substr(0, pos + 1);
    const std::string suffix = fname.substr(pos + 6);
    std::size_t total = 0;
    for (const auto& entry :
         std::filesystem::directory_iterator(p.parent_path(), ec)) {
      if (ec || !entry.is_regular_file(ec)) continue;
      const std::string ename = entry.path().filename().string();
      if (ename.size() == fname.size() && ename.rfind(prefix, 0) == 0 &&
          ename.substr(pos + 6) == suffix) {
        const auto sz = entry.file_size(ec);
        if (!ec) total += static_cast<std::size_t>(sz);
      }
    }
    if (total > 0) return total;
  }
  const auto size = std::filesystem::file_size(p, ec);
  return ec ? 0 : static_cast<std::size_t>(size);
}

// GGUF file size is a lower bound on resident weight bytes; KV cache and
// scratch add on top. A ~6% factor plus a 512 MiB floor gives a realistic
// estimate that avoids over-committing while not rejecting viable loads.
std::size_t EstimateResidentBytes(const std::string& path) {
  const std::size_t size = TotalGgufBytes(path);
  if (size == 0) return 0;  // unknown; caller treats 0 as "cannot gate"
  constexpr std::size_t kMib = 1024ull * 1024ull;
  return static_cast<std::size_t>(size + size / 16 + 512 * kMib);
}

std::size_t ReadProcMemKiB(const char* file_path, std::string_view key) {
  std::ifstream input(file_path);
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind(key, 0) != 0) continue;
    std::istringstream stream(line.substr(key.size()));
    std::size_t kib = 0;
    stream >> kib;
    return kib;
  }
  return 0;
}

struct MemorySnapshot {
  std::size_t host_total_bytes{0};
  std::size_t host_available_bytes{0};
  std::size_t process_rss_bytes{0};
  std::optional<std::size_t> device_free_bytes;

  std::optional<std::size_t> EffectiveFreeBytes() const {
    if (device_free_bytes.has_value() && host_available_bytes > 0) {
      return std::min(*device_free_bytes, host_available_bytes);
    }
    if (device_free_bytes.has_value()) return *device_free_bytes;
    if (host_available_bytes > 0) return host_available_bytes;
    return std::nullopt;
  }

  json::Value ToJson() const {
    json::Value mem = json::Value::object();
    mem["host_total_mib"] = host_total_bytes >> 20;
    mem["host_available_mib"] = host_available_bytes >> 20;
    mem["process_rss_mib"] = process_rss_bytes >> 20;
    if (device_free_bytes.has_value()) {
      mem["device_free_mib"] = *device_free_bytes >> 20;
    } else {
      mem["device_free_mib"] = nullptr;
    }
    if (const auto eff = EffectiveFreeBytes(); eff.has_value()) {
      mem["effective_free_mib"] = *eff >> 20;
    } else {
      mem["effective_free_mib"] = nullptr;
    }
    return mem;
  }
};

MemorySnapshot QueryMemorySnapshot(const MemoryProbe& probe) {
  MemorySnapshot snap;
  snap.host_total_bytes = ReadProcMemKiB("/proc/meminfo", "MemTotal:") * 1024ull;
  snap.host_available_bytes =
      ReadProcMemKiB("/proc/meminfo", "MemAvailable:") * 1024ull;
  snap.process_rss_bytes =
      ReadProcMemKiB("/proc/self/status", "VmRSS:") * 1024ull;
  if (probe) {
    snap.device_free_bytes = probe();
  }
  return snap;
}

enum class GgufRole {
  kMainModel,
  kMmproj,
  kDraftMtp,
  kDraftDFlash,
  kDraftDSpark,
};

GgufRole ClassifyGguf(const std::filesystem::path& p) {
  const std::string lower_name = ToLowerAscii(p.filename().string());
  const std::string lower_parent =
      p.has_parent_path() ? ToLowerAscii(p.parent_path().filename().string())
                          : "";
  if (lower_name.rfind("mmproj", 0) == 0 ||
      lower_name.find("mmproj") != std::string::npos) {
    return GgufRole::kMmproj;
  }
  if (lower_parent == "mtp" || lower_name.rfind("mtp-", 0) == 0 ||
      lower_name.find("-mtp-") != std::string::npos) {
    return GgufRole::kDraftMtp;
  }
  if (lower_name.find("dflash") != std::string::npos) {
    return GgufRole::kDraftDFlash;
  }
  if (lower_name.find("dspark") != std::string::npos) {
    return GgufRole::kDraftDSpark;
  }
  return GgufRole::kMainModel;
}

}  // namespace

ModelCapabilityProfile ClassifyModelCapabilities(std::string_view id_or_name,
                                                 std::string_view family,
                                                 std::string_view path,
                                                 bool has_mmproj,
                                                 std::string_view spec_kind) {
  ModelCapabilityProfile prof;
  const std::string hay = ToLowerAscii(std::string(id_or_name) + " " +
                                       std::string(family) + " " +
                                       std::string(path));
  if (hay.find("rerank") != std::string::npos) {
    prof.task = "reranker";
    prof.badges.push_back("RERANKER");
  } else if (hay.find("embed") != std::string::npos ||
             hay.find("bge-m3") != std::string::npos) {
    prof.task = "embedding";
    prof.badges.push_back("EMBEDDING");
  } else if (hay.find("asr") != std::string::npos ||
             hay.find("whisper") != std::string::npos) {
    prof.task = "audio";
    prof.has_audio = true;
    prof.badges.push_back("AUDIO");
  } else if (hay.find("tts") != std::string::npos) {
    prof.task = "tts";
    prof.has_audio = true;
    prof.badges.push_back("SPEECH");
  } else if (hay.find("d3-") != std::string::npos ||
             hay.find("d3:") != std::string::npos ||
             hay.find("d3_decide") != std::string::npos ||
             hay.find("decision") != std::string::npos ||
             hay.find("systemone") != std::string::npos) {
    prof.task = "decide";
    prof.badges.push_back("DECIDE");
  } else {
    prof.task = "chat";
    prof.badges.push_back("CHAT");
  }

  const bool is_gemma4 = hay.find("gemma-4") != std::string::npos ||
                         hay.find("gemma4") != std::string::npos;
  const bool is_glimmer = hay.find("muse-glimmer") != std::string::npos ||
                          hay.find("glimmer") != std::string::npos;
  const bool is_qwen38 = hay.find("qwen3.8") != std::string::npos;

  if (has_mmproj || is_gemma4 || is_glimmer || is_qwen38 ||
      hay.find("-vl") != std::string::npos ||
      hay.find("vision") != std::string::npos) {
    prof.has_vision = true;
    prof.badges.push_back("VISION");
  }
  if (is_gemma4 || hay.find("audio") != std::string::npos ||
      hay.find("omni") != std::string::npos ||
      hay.find("asr") != std::string::npos ||
      hay.find("whisper") != std::string::npos) {
    prof.has_audio = true;
    if (prof.task != "audio") prof.badges.push_back("AUDIO");
  }
  if (prof.task == "chat") {
    prof.has_tools = true;
    prof.badges.push_back("TOOLS");
  }
  if (!spec_kind.empty() && spec_kind != "off") {
    std::string upper_spec(spec_kind);
    for (char& c : upper_spec) {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    prof.badges.push_back(upper_spec);
  }
  return prof;
}

HttpResponse ModelAdminService::Handle(const HttpRequest& req) {
  if (req.method == "GET" &&
      (req.path == "/" || req.path == "/admin" || req.path == "/admin/" ||
       req.path == "/admin/ui")) {
    return HandleUi(req);
  }
  // Single-modality servers (asr/tts/image): no GPU registry here.
  // Only static/read-only routes stay reachable; the rest 404s
  // instead of dereferencing a null registry.
  if (registry_ == nullptr) {
    const bool safe =
        ((req.method == "GET" &&
          (req.path == "/" || req.path == "/admin" ||
           req.path == "/admin/" || req.path == "/admin/ui" ||
           req.path == "/admin/models" ||
           req.path == "/admin/catalog" ||
           req.path == "/admin/model-mapping" ||
           req.path == "/admin/npu/status")) ||
         (req.method == "POST" &&
          (req.path == "/admin/load" || req.path == "/admin/eject")));
    if (!safe) {
      return AdminErr(404, "Not Found",
                      "admin registry not mounted on this server",
                      "not_found");
    }
  }
  if (req.path == "/admin/models" && req.method == "GET")
    return HandleList(req);
  if (req.path == "/admin/catalog" && req.method == "GET")
    return HandleCatalog(req);
  if (req.path == "/admin/npu/status" && req.method == "GET")
    return HandleNpuStatus(req);
  if (req.path == "/admin/model-mapping" && req.method == "GET")
    return admin_ops::HandleGetModelMapping(req);
  if (req.path == "/admin/model-mapping/sync" && req.method == "POST")
    return admin_ops::HandleSyncModelMapping(req);
  if (req.path == "/admin/hf/files" && req.method == "GET")
    return admin_ops::HandleHfListFiles(req);
  if (req.path == "/admin/download" && req.method == "POST")
    return admin_ops::HandleStartDownload(req);
  if (req.path == "/admin/downloads" && req.method == "GET")
    return admin_ops::HandleListDownloads(req);
  if (req.path == "/admin/download/cancel" && req.method == "POST")
    return admin_ops::HandleCancelDownload(req);
  if (req.path == "/admin/update/check" && req.method == "GET")
    return admin_ops::HandleUpdateCheck(req);
  if (req.path == "/admin/update/apply" && req.method == "POST")
    return admin_ops::HandleUpdateApply(req);
  if (req.path == "/admin/update/status" && req.method == "GET")
    return admin_ops::HandleUpdateStatus(req);
  if (req.path == "/admin/auth/status" && req.method == "GET")
    return admin_ops::HandleAuthStatus(req);
  if (req.path == "/admin/auth/login" && req.method == "POST")
    return admin_ops::HandleAuthLogin(req);
  if (req.path == "/admin/auth/logout" && req.method == "POST")
    return admin_ops::HandleAuthLogout(req);
  if (req.path == "/admin/settings" && req.method == "GET") {
    std::string live_def;
    for (const auto& info : registry_->List()) {
      if (info.is_default) {
        live_def = info.id;
        break;
      }
    }
    return admin_ops::HandleGetSettings(req, live_def);
  }
  if (req.path == "/admin/settings" && req.method == "POST")
    return admin_ops::HandleSaveSettings(req);
  if (req.path == "/admin/server/restart" && req.method == "POST")
    return admin_ops::HandleServerRestart(req);
  if (req.path == "/admin/disk-cache" && req.method == "GET")
    return HandleDiskCacheList(req);
  if (req.path == "/admin/disk-cache/delete" && req.method == "POST")
    return HandleDiskCacheDelete(req);
  if (req.path == "/admin/load" && req.method == "POST")
    return HandleLoad(req);
  if (req.path == "/admin/reload" && req.method == "POST")
    return HandleReload(req);
  if (req.path == "/admin/eject" && req.method == "POST")
    return HandleEject(req);
  if (req.path == "/admin/default" && req.method == "POST")
    return HandleSetDefault(req);
  return AdminErr(404, "Not Found", "no admin route for this path",
                  "not_found");
}

HttpResponse ModelAdminService::HandleNpuStatus(const HttpRequest&) {
  json::Value resp = json::Value::object();
  json::Value arr = json::Value::array();
  if (npu_registry_) {
    for (const auto& info : npu_registry_->List()) {
      json::Value row = json::Value::object();
      row["id"] = info.id;
      json::Value names = json::Value::array();
      for (const auto& n : info.names) names.push_back(n);
      row["names"] = std::move(names);
      row["lane"] = npu_wire::NpuLaneName(info.lane);
      row["kind"] = npu_wire::NpuKindName(info.kind);
      row["artifact"] = info.artifact;
      row["ready"] = info.ready;
      row["is_default"] = info.is_default;
      row["estimated_resident_mib"] =
          static_cast<std::size_t>(info.estimated_resident_mib);
      row["last_used_ms"] = static_cast<std::size_t>(info.last_used_ms);
      arr.push_back(std::move(row));
    }
  }
  resp["npu_models"] = std::move(arr);
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleUi(const HttpRequest&) {
  HttpResponse resp;
  resp.status = 200;
  resp.reason = "OK";
  resp.headers.emplace_back("Content-Type", "text/html; charset=utf-8");
  resp.headers.emplace_back("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
  resp.headers.emplace_back("Pragma", "no-cache");
  resp.body = std::string(kModelAdminUiHtml);
  return resp;
}

HttpResponse ModelAdminService::HandleList(const HttpRequest&) {
  const bool has_gpu = (registry_ != nullptr);
  json::Value resp = json::Value::object();
  resp["object"] = "list";
  const auto lru_order = has_gpu ? registry_->LruCandidates()
                                 : std::vector<std::string>{};
  json::Value data = json::Value::array();
  const auto gpu_list =
      has_gpu ? registry_->List() : decltype(registry_->List()){};
  for (const auto& info : gpu_list) {
    json::Value row = json::Value::object();
    row["id"] = info.id;
    row["path"] = info.path;
    row["model_path"] = info.path;
    json::Value names = json::Value::array();
    for (const auto& name : info.names) names.push_back(name);
    row["names"] = std::move(names);
    row["context_length"] = static_cast<std::size_t>(info.max_context);
    row["ready"] = info.ready;
    row["is_default"] = info.is_default;
    row["last_used_ms"] = static_cast<std::size_t>(info.last_used_ms);
    row["idle_ms"] = static_cast<std::size_t>(info.idle_ms);
    row["supports_images"] = info.supports_images;
    row["speculative"] = info.speculative;
    {
      const std::string spec_s =
          (!info.config.speculative.empty() &&
           info.config.speculative != "off")
              ? info.config.speculative
              : (info.speculative ? std::string("mtp") : std::string{});
      const auto prof = ClassifyModelCapabilities(
          info.id, "", info.path,
          info.supports_images || !info.config.vision_model_path.empty(),
          spec_s);
      row["task"] = prof.task;
      row["has_vision"] = prof.has_vision;
      row["has_audio"] = prof.has_audio;
      row["has_tools"] = prof.has_tools;
      json::Value caps = json::Value::array();
      for (const auto& b : prof.badges) caps.push_back(b);
      row["capabilities"] = std::move(caps);
    }
    {
      json::Value aliases = json::Value::array();
      for (const auto& n : info.names) {
        if (n != info.id) aliases.push_back(n);
      }
      row["aliases"] = std::move(aliases);
    }
    row["kv_cache_quant"] = info.config.kv_cache_quant;
    row["speculative_mode"] =
        info.config.speculative.empty() ? "off" : info.config.speculative;
    row["draft_model_path"] = info.config.draft_model_path;
    row["vision_model_path"] = info.config.vision_model_path;
    row["enable_yarn"] = info.config.enable_yarn;
    row["enable_disk_cache"] = info.config.enable_disk_cache;
    row["npu_quant"] = info.config.npu_quant;
    row["total_sessions"] = info.total_sessions;
    row["active_sessions"] = info.active_sessions;
    row["busy"] = info.active_sessions > 0;
    const std::size_t file_bytes =
        info.path.empty() ? 0 : TotalGgufBytes(info.path);
    const std::size_t est_bytes =
        info.path.empty() ? 0 : EstimateResidentBytes(info.path);
    row["file_size_mib"] = file_bytes >> 20;
    row["estimated_resident_mib"] = est_bytes >> 20;
    row["backend"] = info.backend.empty() ? "gpu" : info.backend;

    int lru_rank = -1;
    for (std::size_t i = 0; i < lru_order.size(); ++i) {
      if (lru_order[i] == info.id) {
        lru_rank = static_cast<int>(i + 1);
        break;
      }
    }
    row["lru_rank"] = lru_rank;

    json::Value sessions = json::Value::array();
    for (std::size_t i = 0; i < info.sessions.size(); ++i) {
      const auto& s = info.sessions[i];
      json::Value slot = json::Value::object();
      slot["id"] = i;
      slot["index"] = i;
      slot["processing"] = s.processing;
      slot["busy"] = s.processing;
      slot["speculative"] = s.speculative;
      slot["request_id"] =
          s.processing ? static_cast<std::int64_t>(s.request_id) : -1;
      slot["generated_tokens"] = s.generated_tokens;
      slot["cached_tokens"] = s.generated_tokens;
      slot["remaining_tokens"] =
          s.processing ? static_cast<std::int64_t>(s.remaining_tokens) : -1;
      sessions.push_back(std::move(slot));
    }
    row["sessions"] = std::move(sessions);
    data.push_back(std::move(row));
  }
  // Running sidecars appear as loaded models so the UI renders the same
  // cards (with Eject) as GPU models.
  for (const auto& st : GlobalSidecarManager().ListRunning()) {
    const SidecarLaunchSpec* spec = nullptr;
    for (const auto& s : KnownSidecars()) {
      if (s.base_url == st.base_url) {
        spec = &s;
        break;
      }
    }
    if (spec == nullptr) continue;
    json::Value row = json::Value::object();
    row["id"] = spec->id;
    row["path"] = spec->base_url;
    row["model_path"] = spec->base_url;
    json::Value names = json::Value::array();
    names.push_back(spec->id);
    row["names"] = std::move(names);
    row["context_length"] = 0;
    row["ready"] = true;
    row["is_default"] = false;
    row["task"] = spec->task;
    row["has_vision"] = false;
    row["has_audio"] = spec->task == "asr" || spec->task == "tts";
    row["has_tools"] = false;
    json::Value caps = json::Value::array();
    caps.push_back(spec->capability);
    row["capabilities"] = std::move(caps);
    row["total_sessions"] = 0;
    row["active_sessions"] = 0;
    row["busy"] = false;
    row["file_size_mib"] = 0;
    row["estimated_resident_mib"] =
        static_cast<std::uint64_t>(spec->estimated_resident_mib);
    row["id_suggestion"] = spec->id;
    row["backend"] = spec->backend;
    row["sidecar"] = true;
    row["default_port"] = spec->port;
    row["endpoint"] = spec->endpoint;
    row["pid"] = static_cast<std::int64_t>(st.pid);
    row["external"] = st.external;
    row["warm"] = st.warm;
    row["warmup_ms"] = st.warmup_ms;
    if (!st.warmup_error.empty()) row["warmup_error"] = st.warmup_error;
    row["ready"] = st.running && (st.warm || st.warmup_skipped);
    row["lru_rank"] = -1;
    if (spec->kind == SidecarLaunchSpec::Kind::kPyImageSidecar) {
      row["file_size_mib"] =
          static_cast<std::uint64_t>(spec->estimated_resident_mib);
      row["image_dtype"] = st.options.dtype;
      row["image_steps"] = static_cast<std::size_t>(st.options.steps);
      row["image_size"] = st.options.default_size;
      row["image_cpu_offload"] = st.options.cpu_offload;
    }
    json::Value sessions = json::Value::array();
    row["sessions"] = std::move(sessions);
    data.push_back(std::move(row));
  }
  // Loaded NPU in-process servables appear as loaded models so the UI
  // renders the same cards (amber border, NPU slots bar, Eject) as GPU
  // models. Task/capability mirror the catalog ExtraDir rows.
  if (npu_registry_ != nullptr) {
    for (const auto& info : npu_registry_->List()) {
      if (!npu_holder_.IsLoaded(info.id)) continue;
      std::string task = "chat";
      std::string cap = "NPU";
      if (info.lane ==
          mbufo::core::npu::NpuModelLane::kAsr) {
        task = "audio";
        cap = "AUDIO";
      } else if (info.lane ==
                 mbufo::core::npu::NpuModelLane::kTts) {
        task = "tts";
        cap = "SPEECH";
      } else if (info.lane ==
                 mbufo::core::npu::NpuModelLane::kImage) {
        task = "image_generation";
        cap = "IMAGE";
      }
      json::Value row = json::Value::object();
      row["id"] = info.id;
      row["path"] = info.artifact;
      row["model_path"] = info.artifact;
      json::Value names = json::Value::array();
      for (const auto& n : info.names) names.push_back(n);
      row["names"] = std::move(names);
      row["context_length"] = 0;
      row["ready"] = true;
      row["is_default"] = info.is_default;
      row["task"] = task;
      row["has_vision"] = false;
      row["has_audio"] =
          task == "audio" || task == "tts";
      row["has_tools"] = false;
      json::Value caps = json::Value::array();
      caps.push_back(cap);
      row["capabilities"] = std::move(caps);
      row["total_sessions"] = 0;
      row["active_sessions"] = 0;
      row["busy"] = false;
      row["file_size_mib"] = 0;
      row["estimated_resident_mib"] =
          static_cast<std::uint64_t>(info.estimated_resident_mib);
      row["id_suggestion"] = info.id;
      row["backend"] = "npu";
      row["sidecar"] = false;
      row["lru_rank"] = -1;
      json::Value sessions = json::Value::array();
      row["sessions"] = std::move(sessions);
      data.push_back(std::move(row));
    }
  }
  resp["build_sha"] = mbufo::build::kGitCommit;
  resp["models"] = data;
  resp["data"] = std::move(data);
  resp["memory"] = QueryMemorySnapshot(probe_).ToJson();

  {
    std::lock_guard lock(loading_state_mutex_);
    if (!loading_id_.empty()) {
      json::Value loading = json::Value::object();
      loading["active"] = true;
      loading["id"] = loading_id_;
      loading["path"] = loading_path_;
      const std::uint64_t now = NowSteadyMs();
      loading["elapsed_ms"] =
          now >= loading_started_ms_ ? (now - loading_started_ms_) : 0;
      resp["loading"] = std::move(loading);
    } else {
      json::Value loading = json::Value::object();
      loading["active"] = false;
      resp["loading"] = std::move(loading);
    }
  }
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleCatalog(const HttpRequest& req) {
  std::vector<std::filesystem::path> roots;
  if (const auto q = req.query_param("dir"); !q.empty()) {
    roots.emplace_back(q);
  } else if (const char* env_dir = std::getenv("MBUFO_MODELS_DIR");
             env_dir != nullptr && *env_dir != '\0') {
    roots.emplace_back(env_dir);
  } else {
    roots.emplace_back("models");
  }

  // Also include the ancestor models root of any currently loaded model
  // (walking up to 3 levels to find a directory containing multiple model folders
  // or named *models*) plus /mnt/nvme-data/mbufo-models if present.
  auto add_root_unique = [&](const std::filesystem::path& cand) {
    std::error_code ec;
    if (!std::filesystem::exists(cand, ec) || !std::filesystem::is_directory(cand, ec))
      return;
    const std::string cs = cand.string();
    for (const auto& r : roots) {
      if (r.string() == cs) return;
    }
    roots.push_back(cand);
  };
  if (std::getenv("MBUFO_MODELS_DIR") == nullptr) {
    add_root_unique("/mnt/nvme-data/mbufo-models");
  }
  const auto loaded_list =
      registry_ ? registry_->List() : decltype(registry_->List()){};
  for (const auto& info : loaded_list) {
    if (info.path.empty()) continue;
    std::filesystem::path p(info.path);
    std::filesystem::path cur = p.parent_path();
    std::filesystem::path best;
    for (int up = 0; up < 3 && cur.has_parent_path(); ++up) {
      cur = cur.parent_path();
      const std::string fname = ToLowerAscii(cur.filename().string());
      if (fname.find("models") != std::string::npos) {
        best = cur;
        break;
      }
      if (best.empty()) best = cur;
    }
    if (!best.empty()) {
      bool covered = false;
      for (const auto& r : roots) {
        if (info.path.rfind(r.string(), 0) == 0 &&
            r.string() != "models") {
          covered = true;
          break;
        }
      }
      if (!covered) add_root_unique(best);
    }
  }

  struct DiscoveredFile {
    std::filesystem::path path;
    std::string family;
    GgufRole role;
    std::size_t size_bytes{0};
  };
  std::vector<DiscoveredFile> files;

  for (const auto& root : roots) {
    std::error_code ec;
    if (!std::filesystem::exists(root, ec) ||
        !std::filesystem::is_directory(root, ec)) {
      continue;
    }
    std::filesystem::recursive_directory_iterator it(
        root,
        std::filesystem::directory_options::skip_permission_denied |
            std::filesystem::directory_options::follow_directory_symlink,
        ec);
    std::filesystem::recursive_directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
      if (it.depth() > 4) {
        it.disable_recursion_pending();
        continue;
      }
      std::error_code entry_ec;
      if (!it->is_regular_file(entry_ec)) continue;
      const auto& p = it->path();
      if (ToLowerAscii(p.extension().string()) != ".gguf") continue;
      const std::string fname = p.filename().string();
      if (IsNonFirstShard(fname)) continue;

      std::string family = "other";
      std::error_code rel_ec;
      const auto rel = std::filesystem::relative(p, root, rel_ec);
      if (!rel_ec && rel.begin() != rel.end()) {
        family = rel.begin()->string();
      }

      DiscoveredFile df;
      df.path = p;
      df.family = std::move(family);
      df.role = ClassifyGguf(p);
      df.size_bytes = TotalGgufBytes(p.string());
      files.push_back(std::move(df));
    }
  }

  std::sort(files.begin(), files.end(),
            [](const DiscoveredFile& a, const DiscoveredFile& b) {
              return a.path.string() < b.path.string();
            });

  json::Value mmproj_list = json::Value::array();
  json::Value mmproj_files = json::Value::array();
  json::Value draft_list = json::Value::array();
  json::Value draft_files = json::Value::array();
  for (const auto& f : files) {
    if (f.role == GgufRole::kMmproj) {
      json::Value item = json::Value::object();
      item["path"] = f.path.string();
      item["name"] = f.path.filename().string();
      item["family"] = f.family;
      item["size_mib"] = f.size_bytes >> 20;
      mmproj_list.push_back(std::move(item));
      mmproj_files.push_back(f.path.string());
    } else if (f.role == GgufRole::kDraftMtp ||
               f.role == GgufRole::kDraftDFlash ||
               f.role == GgufRole::kDraftDSpark) {
      json::Value item = json::Value::object();
      item["path"] = f.path.string();
      item["name"] = f.path.filename().string();
      item["family"] = f.family;
      item["kind"] = f.role == GgufRole::kDraftMtp      ? "mtp"
                     : f.role == GgufRole::kDraftDFlash ? "dflash2"
                                                        : "dspark";
      item["size_mib"] = f.size_bytes >> 20;
      draft_list.push_back(std::move(item));
      draft_files.push_back(f.path.string());
    }
  }

  auto is_flash_next_family = [](const std::string& fam) {
    return fam == "qwen3.8-flash-next" || fam == "qwen3.8-rocmfp4" ||
           fam == "qwen3.8-mxfp4";
  };
  auto role_to_spec = [](GgufRole role) -> std::string {
    switch (role) {
      case GgufRole::kDraftMtp: return "mtp";
      case GgufRole::kDraftDFlash: return "dflash2";
      case GgufRole::kDraftDSpark: return "dspark";
      default: return "";
    }
  };

  json::Value models = json::Value::array();
  for (const auto& f : files) {
    if (f.role != GgufRole::kMainModel) continue;
    const std::string full_path = f.path.string();
    const std::string derived_id = DeriveId(full_path);

    bool is_loaded = false;
    bool is_default = false;
    std::string loaded_id;
    for (const auto& l : loaded_list) {
      if (l.path == full_path || l.id == derived_id) {
        is_loaded = true;
        is_default = l.is_default;
        loaded_id = l.id;
        break;
      }
    }

    const std::size_t est_bytes = EstimateResidentBytes(full_path);
    const std::string id_lower = ToLowerAscii(derived_id);
    const std::string fam_lower = ToLowerAscii(f.family);
    const bool is_qwen38_flash_next =
        is_flash_next_family(fam_lower) ||
        id_lower.find("flash-next") != std::string::npos;
    const bool is_qwen38_27b =
        fam_lower == "qwen3.8-27b" ||
        id_lower.find("qwen3.8-27b") != std::string::npos;
    const bool is_qwen35_35b_or_ornith =
        id_lower.find("qwen3.5-35b") != std::string::npos ||
        id_lower.find("ornith-1.5-35b") != std::string::npos;
    const bool is_gemma4_26b =
        fam_lower == "gemma-4-26b" ||
        (id_lower.find("gemma-4") != std::string::npos &&
         id_lower.find("26b") != std::string::npos);
    const bool is_glimmer =
        fam_lower.find("glimmer") != std::string::npos ||
        id_lower.find("glimmer") != std::string::npos;
    const bool is_deepseek_v4 =
        fam_lower.find("deepseek") != std::string::npos ||
        id_lower.find("deepseek") != std::string::npos;
    const bool is_tiel_coder =
        fam_lower.find("tiel-coder") != std::string::npos ||
        id_lower.find("tiel-coder") != std::string::npos;

    // Supported speculative modes matched strictly against inference_backend.cpp.
    // Note: Qwen3.8-27B (QwenGpuModel) supports DFlash2 only and rejects MTP.
    std::vector<std::string> supported_spec;
    if (is_qwen38_flash_next || is_gemma4_26b || is_tiel_coder) {
      supported_spec.push_back("mtp");
    } else if (is_qwen38_27b || is_glimmer) {
      supported_spec.push_back("dflash2");
    } else if (is_deepseek_v4) {
      supported_spec.push_back("dspark");
    }

    auto family_matches = [&](const std::string& cand_family) {
      const std::string cf = ToLowerAscii(cand_family);
      if (cf == fam_lower) return true;
      if (is_qwen38_flash_next && is_flash_next_family(cf)) return true;
      return false;
    };

    std::vector<std::pair<bool, std::string>> draft_cands;
    for (const auto& cf : files) {
      const std::string mode = role_to_spec(cf.role);
      if (mode.empty()) continue;
      if (std::find(supported_spec.begin(), supported_spec.end(), mode) ==
          supported_spec.end()) {
        continue;
      }
      if (!family_matches(cf.family)) continue;
      draft_cands.emplace_back(ToLowerAscii(cf.family) != fam_lower,
                               cf.path.string());
    }
    std::stable_sort(draft_cands.begin(), draft_cands.end(),
                     [](const auto& a, const auto& b) {
                       return a.first < b.first;
                     });

    std::vector<std::string> compatible_drafts;
    std::string sug_mtp;
    std::string sug_dflash;
    std::string sug_dspark;
    std::string sug_draft;
    std::string sug_spec;
    for (const auto& [_, dpath] : draft_cands) {
      compatible_drafts.push_back(dpath);
      const GgufRole dr = ClassifyGguf(std::filesystem::path(dpath));
      if (dr == GgufRole::kDraftMtp && sug_mtp.empty()) sug_mtp = dpath;
      if (dr == GgufRole::kDraftDFlash && sug_dflash.empty()) sug_dflash = dpath;
      if (dr == GgufRole::kDraftDSpark && sug_dspark.empty()) sug_dspark = dpath;
    }
    if (!compatible_drafts.empty()) {
      sug_draft = compatible_drafts.front();
      sug_spec = role_to_spec(ClassifyGguf(std::filesystem::path(sug_draft)));
    }

    // External --mmproj is only accepted on Qwen3.8-Flash-Next, Qwen3.8-27B,
    // and Qwen3.5/Ornith 35B-A3B. Prefer BF16 (matching mbufo.service).
    const bool arch_supports_mmproj =
        is_qwen38_flash_next || is_qwen38_27b || is_qwen35_35b_or_ornith;
    std::vector<std::pair<int, std::string>> mm_cands;
    if (arch_supports_mmproj) {
      for (const auto& cf : files) {
        if (cf.role != GgufRole::kMmproj) continue;
        if (!family_matches(cf.family)) continue;
        const std::string lower_fn = ToLowerAscii(cf.path.filename().string());
        const int is_not_bf16 = lower_fn.find("bf16") == std::string::npos ? 1 : 0;
        const int not_exact_fam = ToLowerAscii(cf.family) != fam_lower ? 1 : 0;
        mm_cands.emplace_back(is_not_bf16 * 2 + not_exact_fam, cf.path.string());
      }
      std::stable_sort(mm_cands.begin(), mm_cands.end(),
                       [](const auto& a, const auto& b) {
                         return a.first < b.first;
                       });
    }
    std::vector<std::string> compatible_mmprojs;
    for (const auto& [_, mpath] : mm_cands) compatible_mmprojs.push_back(mpath);
    const std::string sug_mm =
        compatible_mmprojs.empty() ? std::string{} : compatible_mmprojs.front();
    const bool supports_mmproj =
        arch_supports_mmproj && !compatible_mmprojs.empty();

    std::string rel_path = full_path;
    for (const auto& r : roots) {
      std::error_code rel_ec;
      const auto rel = std::filesystem::relative(f.path, r, rel_ec);
      if (!rel_ec && !rel.empty() && rel.string().rfind("..", 0) != 0) {
        rel_path = rel.string();
        break;
      }
    }
    json::Value item = json::Value::object();
    item["id"] = derived_id;
    item["id_suggestion"] = derived_id;
    item["name"] = f.path.filename().string();
    const auto prof = ClassifyModelCapabilities(
        derived_id, f.family, full_path, supports_mmproj, sug_spec);
    item["task"] = prof.task;
    item["has_vision"] = prof.has_vision;
    item["has_audio"] = prof.has_audio;
    item["has_tools"] = prof.has_tools;
    {
      json::Value caps = json::Value::array();
      for (const auto& b : prof.badges) caps.push_back(b);
      item["capabilities"] = std::move(caps);
    }
    item["family"] = f.family;
    item["path"] = full_path;
    item["relative_path"] = rel_path;
    // In-process GGUF entries are HIP/GPU resident (see npu_lane.hpp).
    item["backend"] = "gpu";
    item["sidecar"] = false;
    item["actionable"] = true;
    item["npu_capable"] =
        id_lower.find("bge-m3") != std::string::npos ||
        id_lower.find("embeddinggemma") != std::string::npos ||
        id_lower.find("bge-reranker") != std::string::npos;
    item["size_mib"] = f.size_bytes >> 20;
    item["file_size_mib"] = f.size_bytes >> 20;
    item["estimated_resident_mib"] = est_bytes >> 20;
    item["shards"] = 1;
    item["loaded"] = is_loaded;
    item["loaded_id"] = loaded_id;
    item["is_default"] = is_default;

    const bool is_enc_or_special = (prof.task != "chat");
    const bool supports_spec = !is_enc_or_special && !supported_spec.empty();
    const bool supports_yarn = !is_enc_or_special && is_qwen38_flash_next;
    const bool supports_parallel_sessions = (prof.task == "chat");

    item["suggested_mmproj"] = is_enc_or_special ? std::string{} : sug_mm;
    item["suggested_draft"] = is_enc_or_special ? std::string{} : sug_draft;
    item["suggested_mtp"] = is_enc_or_special ? std::string{} : sug_mtp;
    item["suggested_dflash"] = is_enc_or_special ? std::string{} : sug_dflash;
    item["suggested_dspark"] = is_enc_or_special ? std::string{} : sug_dspark;
    item["supports_speculative"] = supports_spec;
    {
      json::Value spec_arr = json::Value::array();
      if (!is_enc_or_special) {
        for (const auto& s : supported_spec) spec_arr.push_back(s);
      }
      item["supported_speculative"] = std::move(spec_arr);
    }
    item["suggested_speculative"] = is_enc_or_special ? std::string{} : sug_spec;
    {
      json::Value cd_arr = json::Value::array();
      if (!is_enc_or_special) {
        for (const auto& p : compatible_drafts) cd_arr.push_back(p);
      }
      item["compatible_drafts"] = std::move(cd_arr);
    }
    item["supports_mmproj"] = (!is_enc_or_special && supports_mmproj);
    {
      json::Value mm_arr = json::Value::array();
      if (!is_enc_or_special) {
        for (const auto& p : compatible_mmprojs) mm_arr.push_back(p);
      }
      item["compatible_mmprojs"] = std::move(mm_arr);
    }
    item["supports_yarn"] = supports_yarn;
    item["supports_parallel_sessions"] = supports_parallel_sessions;
    models.push_back(std::move(item));
  }

  // Non-GGUF image-generation safetensors models present on disk (skipping
  // kernel bundles and duplicate weights already exposed as GGUFs/sidecars).
  {
    struct ExtraDir {
      const char* id;
      const char* subdir;
      const char* family;
      const char* task;
      const char* capability;
      const char* backend;
      bool npu_capable;
      bool has_audio;
      bool has_vision;
      std::size_t resident_mib;
      const char* note;
    };
    const ExtraDir kExtraDirs[] = {
        {"qwen-image-2.1", "qwen-image-2.1", "image", "image_generation",
         "IMAGE", "gpu", false, false, false, 0,
         "GPU safetensors image model (mbufo serve image)"},
        // M6: NPU in-process servables (verified 2026-11-24; M3 MiB).
        {"qwen3-asr-1.7b", "qwen3-asr-1.7b", "qwen3-asr-1.7b", "audio",
         "AUDIO", "npu", true, true, false, 520,
         "NPU in-process thinker (decode cos=1.0, top1 match)"},
        {"qwen3-tts-12hz-1.7b", "qwen3-tts-1.7b-base", "qwen3-tts-1.7b-base",
         "tts", "SPEECH", "npu", true, true, false, 2200,
         "NPU in-process talker+predictor+frame (decode cos=1.0)"},
        {"flux2-klein-4b", "flux2-klein-4b", "flux2-klein-4b",
         "image_generation", "IMAGE", "npu", true, false, false, 930,
         "NPU in-process DiT 4-step (E2E cos(pipe)=0.9485)"},
        {"z-image-turbo", "z-image-turbo", "z-image-turbo",
         "image_generation", "IMAGE", "npu", true, false, false, 750,
         "NPU in-process DiT 8-step (E2E cos(pipe)=0.9918)"},
    };
    for (const auto& ed : kExtraDirs) {
      std::filesystem::path dir;
      for (const auto& r : roots) {
        std::error_code dec;
        auto cand = r / ed.subdir;
        if (std::filesystem::is_directory(cand, dec)) {
          dir = cand;
          break;
        }
      }
      if (dir.empty()) continue;
      std::uintmax_t bytes = 0;
      {
        std::error_code wec;
        std::filesystem::recursive_directory_iterator it(dir, wec);
        std::filesystem::recursive_directory_iterator end;
        for (; !wec && it != end; it.increment(wec)) {
          std::error_code fec;
          if (it->is_regular_file(fec)) bytes += it->file_size(fec);
        }
      }
      json::Value item = json::Value::object();
      item["id"] = ed.id;
      item["id_suggestion"] = ed.id;
      item["name"] = std::string(ed.subdir) + "/";
      item["relative_path"] = std::string(ed.subdir) + "/";
      item["family"] = ed.family;
      item["task"] = ed.task;
      item["has_vision"] = ed.has_vision;
      item["has_audio"] = ed.has_audio;
      item["has_tools"] = false;
      json::Value caps = json::Value::array();
      caps.push_back(ed.capability);
      item["capabilities"] = std::move(caps);
      item["backend"] = ed.backend;
      item["sidecar"] = false;
      item["actionable"] = true;
      const bool npu_loaded =
          std::string(ed.backend) == "npu" && npu_holder_.IsLoaded(ed.id);
      item["note"] = ed.note;
      const std::uint64_t mib = static_cast<std::uint64_t>(bytes >> 20);
      item["size_mib"] = mib;
      item["file_size_mib"] = mib;
      item["estimated_resident_mib"] =
          static_cast<std::uint64_t>(ed.resident_mib);
      item["shards"] = 0;
      item["loaded"] = npu_loaded;
      item["loaded_id"] = "";
      item["is_default"] = false;
      item["running"] = npu_loaded;
      item["npu_capable"] = ed.npu_capable;
      item["path"] = dir.string();
      item["suggested_mmproj"] = "";
      item["suggested_draft"] = "";
      item["suggested_mtp"] = "";
      item["suggested_dflash"] = "";
      item["suggested_dspark"] = "";
      item["supports_speculative"] = false;
      item["supported_speculative"] = json::Value::array();
      item["suggested_speculative"] = "";
      item["compatible_drafts"] = json::Value::array();
      item["supports_mmproj"] = false;
      item["compatible_mmprojs"] = json::Value::array();
      item["supports_yarn"] = false;
      item["supports_parallel_sessions"] = false;
      models.push_back(std::move(item));
    }
  }

  // NPU/GPU sidecar entries for the router admin UI device switch.
  // Single-sourced from KnownSidecars() (the same table SidecarManager
  // spawns from); launched on demand via /admin/load, never at boot.
  {
    auto push_sidecar = [&](const SidecarLaunchSpec& spec) {
      const auto sc_status = GlobalSidecarManager().GetStatus(spec.id);
      std::filesystem::path img_dir;
      std::uint64_t disk_mib = spec.estimated_resident_mib;
      if (spec.kind == SidecarLaunchSpec::Kind::kPyImageSidecar) {
        for (const auto& r : roots) {
          std::error_code dec;
          auto cand = r / spec.flm_model;
          if (std::filesystem::is_directory(cand, dec)) {
            img_dir = cand;
            break;
          }
        }
        if (img_dir.empty() && !sc_status.running) return;
        if (!img_dir.empty()) {
          std::uintmax_t bytes = 0;
          std::error_code wec;
          std::filesystem::recursive_directory_iterator it(img_dir, wec);
          std::filesystem::recursive_directory_iterator end;
          for (; !wec && it != end; it.increment(wec)) {
            std::error_code fec;
            if (it->is_regular_file(fec)) bytes += it->file_size(fec);
          }
          if (bytes > 0) disk_mib = static_cast<std::uint64_t>(bytes >> 20);
        }
      }

      json::Value item = json::Value::object();
      item["id"] = spec.id;
      item["id_suggestion"] = spec.id;
      item["name"] = spec.display_name;
      item["relative_path"] =
          img_dir.empty() ? spec.base_url : (spec.flm_model + "/");
      item["family"] = "sidecar";
      item["task"] = spec.task;
      item["has_vision"] = false;
      item["has_audio"] = spec.capability == "AUDIO";
      item["has_tools"] = false;
      json::Value caps = json::Value::array();
      caps.push_back(spec.capability);
      item["capabilities"] = std::move(caps);
      item["backend"] = spec.backend;
      item["sidecar"] = true;
      item["warm"] = sc_status.warm;
      item["warmup_ms"] = sc_status.warmup_ms;
      item["actionable"] = true;
      item["default_port"] = spec.port;
      item["endpoint"] = spec.endpoint;
      item["launch_cmd"] = SidecarLaunchDisplay(spec);
      item["size_mib"] = disk_mib;
      item["file_size_mib"] = disk_mib;
      item["estimated_resident_mib"] = spec.estimated_resident_mib;
      item["shards"] = 1;
      item["loaded"] = sc_status.running;
      item["loaded_id"] = sc_status.running ? spec.id : std::string{};
      item["is_default"] = false;
      item["running"] = sc_status.running;
      item["path"] = spec.base_url;
      if (spec.kind == SidecarLaunchSpec::Kind::kPyImageSidecar) {
        item["image_dtype"] = sc_status.options.dtype;
        item["image_steps"] = static_cast<std::size_t>(sc_status.options.steps);
        item["image_size"] = sc_status.options.default_size;
        item["image_cpu_offload"] = sc_status.options.cpu_offload;
      }
      item["suggested_mmproj"] = "";
      item["suggested_draft"] = "";
      item["suggested_mtp"] = "";
      item["suggested_dflash"] = "";
      item["suggested_dspark"] = "";
      item["supports_speculative"] = false;
      item["supported_speculative"] = json::Value::array();
      item["suggested_speculative"] = "";
      item["compatible_drafts"] = json::Value::array();
      item["supports_mmproj"] = false;
      item["compatible_mmprojs"] = json::Value::array();
      item["supports_yarn"] = false;
      item["supports_parallel_sessions"] = false;
      models.push_back(std::move(item));
    };
    for (const auto& spec : KnownSidecars()) push_sidecar(spec);
  }

  json::Value sidecars = json::Value::object();
  sidecars["mmproj"] = std::move(mmproj_list);
  sidecars["draft"] = std::move(draft_list);

  json::Value resp = json::Value::object();
  resp["object"] = "catalog";
  resp["models"] = std::move(models);
  {
    json::Value roots_json = json::Value::array();
    for (const auto& r : roots) roots_json.push_back(r.string());
    resp["scan_roots"] = std::move(roots_json);
  }
  resp["mmproj_files"] = std::move(mmproj_files);
  resp["draft_files"] = std::move(draft_files);
  resp["sidecars"] = std::move(sidecars);
  resp["memory"] = QueryMemorySnapshot(probe_).ToJson();
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleLoad(const HttpRequest& req) {
  // M6b L2: in-process NPU servables load weights via the servable host
  // (no GGUF path needed). Mirrors the sidecar branch response shape.
  // NOTE: body is parsed lazily here; the shared parse below reuses it.
  // NOTE: the loader_ gate lives BELOW the sidecar branch: sidecar
  // spawn/stop needs no GPU loader, so neutered single-modality
  // servers can manage sidecars too.
  {
    json::Value pre;
    try {
      pre = json::parse(req.body);
    } catch (const std::exception&) {
    }
    const std::string npu_id =
        pre.is_object()
            ? pre.member_str("id", pre.member_str("model", ""))
            : "";
    if (!npu_id.empty() && npu_registry_ != nullptr) {
      auto npu_spec = npu_registry_->Resolve(npu_id);
      if (npu_spec != nullptr &&
          npu_spec->kind ==
              mbufo::core::npu::NpuArtifactKind::kNpuInProcess) {
        const std::string quant =
            pre.is_object() ? pre.member_str("npu_quant", "q8") : "q8";
        {
          std::lock_guard lock(loading_state_mutex_);
          loading_id_ = npu_id;
          loading_path_ = npu_spec->artifact;
          loading_started_ms_ = NowSteadyMs();
        }
        const auto t_load_start = NowSteadyMs();
        std::string npu_err;
        bool ok = false;
        {
          std::lock_guard serialize(load_mutex_);
          ok = npu_holder_.Load(npu_id, npu_spec->artifact, quant,
                              &npu_err);
        }
        {
          std::lock_guard lock(loading_state_mutex_);
          loading_id_.clear();
          loading_path_.clear();
          loading_started_ms_ = 0;
        }
        if (!ok) {
          return AdminErr(502, "Bad Gateway",
                          npu_err.empty() ? "npu load failed" : npu_err,
                          "npu_load_failed");
        }
        std::string warm_err;
        const bool warmed = npu_holder_.Warm(npu_id, &warm_err);
        const auto warm_info = npu_holder_.GetWarmInfo(npu_id);
        npu_registry_->SetReady(npu_id, true);
        json::Value resp = json::Value::object();
        resp["warm"] = warmed && warm_info.warmed;
        resp["warmup_ms"] = warm_info.ms;
        if (!warm_err.empty()) resp["warmup_error"] = warm_err;
        resp["id"] = npu_id;
        resp["path"] = npu_spec->artifact;
        resp["model_path"] = npu_spec->artifact;
        resp["status"] = "running";
        resp["backend"] = "npu";
        resp["sidecar"] = false;
        resp["estimated_resident_mib"] =
            static_cast<std::uint64_t>(
                npu_spec->estimated_resident_mib);
        resp["load_ms"] = NowSteadyMs() >= t_load_start
                              ? NowSteadyMs() - t_load_start
                              : std::uint64_t{0};
        return AdminOk(std::move(resp));
      }
    }
  }
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return AdminErr(400, "Bad Request", e.what(), "parse_error");
  }
  if (!body.is_object())
    return AdminErr(400, "Bad Request", "request body must be an object",
                    "invalid_request");
  AdminLoadRequest request;
  request.id = body.member_str("id");
  request.path = body.member_str("path");
  if (const auto* aliases = body.find("aliases");
      aliases != nullptr && aliases->is_array()) {
    for (const auto& item : aliases->items()) {
      if (item.is_string() && !item.str().empty())
        request.aliases.push_back(item.str());
    }
  }
  request.max_context = static_cast<std::uint32_t>(
      body.member_size("max_context", body.member_size("context_length", 0)));
  request.session_count =
      body.member_size("session_count", body.member_size("sessions", 1));
  request.kv_cache_quant =
      body.member_str("kv_cache_quant", body.member_str("kv_quant", "q4"));
  request.kv_cache_quant_supplied =
      body.find("kv_cache_quant") != nullptr || body.find("kv_quant") != nullptr;
  // Sidecar default device follows its KnownSidecars() backend (e.g.
  // qwen3-asr/tts are NPU-default); GGUF loads keep the gpu default.
  const std::string want_dev_id =
      body.member_str("id", body.member_str("model", ""));
  const SidecarLaunchSpec* pre_sc =
      !want_dev_id.empty() ? FindSidecarSpec(want_dev_id) : nullptr;
  request.device = body.member_str(
      "device", pre_sc ? std::string(pre_sc->backend) : std::string("gpu"));
  if (request.device != "gpu" && request.device != "npu") {
    return AdminErr(400, "Bad Request",
                    "device must be 'gpu' or 'npu'", "invalid_device");
  }
  request.npu_quant = body.member_str("npu_quant", "");
  request.vision_model_path =
      body.member_str("vision_model_path", body.member_str("mmproj_path", ""));
  request.enable_yarn = body.find("enable_yarn") != nullptr &&
                        body.find("enable_yarn")->as_bool(false);
  request.speculative = body.member_str("speculative", "off");
  request.draft_model_path = body.member_str("draft_model_path");
  request.max_draft_tokens = body.member_size("max_draft_tokens", 0);
  request.enable_disk_cache = body.find("enable_disk_cache") != nullptr &&
                              body.find("enable_disk_cache")->as_bool(false);
  const bool set_default = body.find("set_default") != nullptr &&
                           body.find("set_default")->as_bool(false);

  // allow_evict (default true): under memory pressure, LRU-eject other
  // models to make room. When false, a short-free load fails with 409.
  const bool allow_evict =
      body.find("allow_evict") == nullptr ||
      body.find("allow_evict")->as_bool(true);

  // Sidecars are not GGUFs: spawn (or adopt) the sidecar process instead of
  // loading a model. Same Load button as GPU models; the UI shows RUNNING.
  // Checked before the GGUF path requirement since sidecars carry no file.
  {
    const std::string want_id =
        body.member_str("id", body.member_str("model", ""));
    if (const auto* sc0 = !want_id.empty() ? FindSidecarSpec(want_id) : nullptr;
        sc0 != nullptr) {
      request.id = want_id;
      std::lock_guard serialize(load_mutex_);
      {
        std::lock_guard lock(loading_state_mutex_);
        loading_id_ = request.id;
        loading_path_ = sc0->base_url;
        loading_started_ms_ = NowSteadyMs();
      }
      SidecarStartOptions sc_opts;
      sc_opts.dtype = body.member_str("image_dtype", "bf16");
      sc_opts.steps = static_cast<int>(
          body.member_size("image_steps", sc0->default_steps));
      sc_opts.default_size = body.member_str("image_size", "1024x1024");
      sc_opts.cpu_offload = body.member_str("image_cpu_offload", "none");

      const std::uint64_t t_load_start = NowSteadyMs();
      std::string sc_err;
      const bool was_running = GlobalSidecarManager().IsRunning(request.id);
      const bool ok =
          GlobalSidecarManager().Start(request.id, &sc_err, 180000, sc_opts);
      {
        std::lock_guard lock(loading_state_mutex_);
        loading_id_.clear();
        loading_path_.clear();
        loading_started_ms_ = 0;
      }
      if (!ok) {
        return AdminErr(502, "Bad Gateway",
                        sc_err.empty() ? "sidecar failed to start" : sc_err,
                        "sidecar_start_failed");
      }
      const auto st = GlobalSidecarManager().GetStatus(request.id);
      json::Value resp = json::Value::object();
      resp["id"] = request.id;
      resp["path"] = sc0->base_url;
      resp["model_path"] = sc0->base_url;
      resp["status"] = was_running ? "already_running" : "running";
      resp["backend"] = sc0->backend;
      resp["sidecar"] = true;
      resp["endpoint"] = sc0->endpoint;
      resp["default_port"] = sc0->port;
      resp["estimated_resident_mib"] =
          static_cast<std::uint64_t>(sc0->estimated_resident_mib);
      resp["pid"] = static_cast<std::int64_t>(st.pid);
      resp["external"] = st.external;
      resp["warm"] = st.warm;
      resp["warmup_ms"] = st.warmup_ms;
      if (!st.warmup_error.empty()) resp["warmup_error"] = st.warmup_error;
      if (sc0->kind == SidecarLaunchSpec::Kind::kPyImageSidecar) {
        resp["image_dtype"] = st.options.dtype;
        resp["image_steps"] = static_cast<std::size_t>(st.options.steps);
        resp["image_size"] = st.options.default_size;
        resp["image_cpu_offload"] = st.options.cpu_offload;
      }
      resp["load_ms"] = NowSteadyMs() >= t_load_start
                            ? NowSteadyMs() - t_load_start
                            : std::uint64_t{0};
      return AdminOk(std::move(resp));
    }
  }

  if (!loader_)
    return AdminErr(503, "Service Unavailable",
                    "dynamic model loading is not enabled", "loader_disabled");
  if (request.path.empty())
    return AdminErr(400, "Bad Request", "'path' is required", "missing_path");
  if (request.session_count == 0) request.session_count = 1;
  if (request.id.empty()) request.id = DeriveId(request.path);

  if (registry_->HasName(request.id))
    return AdminErr(409, "Conflict",
                    "model '" + request.id + "' is already loaded",
                    "already_loaded");

  // Serialize loads/ejects so concurrent admin calls cannot both pass the
  // free-memory gate and over-commit the device.
  std::lock_guard serialize(load_mutex_);

  // Memory gate: estimate this model's resident bytes and, when free memory
  // is short, LRU-eject other loaded models until it fits. A probe that
  // reports nothing (no HIP) skips the gate and loads optimistically.
  json::Value evicted = json::Value::array();
  const std::size_t needed = EstimateResidentBytes(request.path);
  if (probe_ && needed > 0) {
    auto free_bytes = probe_();
    if (free_bytes.has_value() && *free_bytes < needed) {
      if (!allow_evict) {
        return AdminErr(409, "Conflict",
                        "insufficient free device memory for '" + request.id +
                            "': need ~" + std::to_string(needed >> 20) +
                            " MiB, have ~" + std::to_string(*free_bytes >> 20) +
                            " MiB; set allow_evict=true or eject a model",
                        "insufficient_memory");
      }
      for (const auto& victim : registry_->LruCandidates()) {
        std::string rm_err;
        if (!registry_->Remove(victim, &rm_err)) continue;
        evicted.push_back(victim);
        free_bytes = probe_();
        if (!free_bytes.has_value() || *free_bytes >= needed) break;
      }
      if (free_bytes.has_value() && *free_bytes < needed) {
        return AdminErr(
            409, "Conflict",
            "insufficient free device memory for '" + request.id +
                "': need ~" + std::to_string(needed >> 20) + " MiB, have ~" +
                std::to_string(*free_bytes >> 20) +
                " MiB after evicting loaded models",
            "insufficient_memory");
      }
    }
  }

  {
    std::lock_guard lock(loading_state_mutex_);
    loading_id_ = request.id;
    loading_path_ = request.path;
    loading_started_ms_ = NowSteadyMs();
  }
  struct ClearLoadingGuard {
    ModelAdminService* self;
    ~ClearLoadingGuard() {
      std::lock_guard lock(self->loading_state_mutex_);
      self->loading_id_.clear();
      self->loading_path_.clear();
      self->loading_started_ms_ = 0;
    }
  } clear_guard{this};

  const std::uint64_t t_load_start = NowSteadyMs();
  std::string error;
  auto backend = loader_(request, &error);
  if (backend == nullptr)
    return AdminErr(500, "Internal Server Error",
                    error.empty() ? "load failed" : error, "load_failed");
  LoadedModelConfig reg_cfg;
  reg_cfg.kv_cache_quant = request.kv_cache_quant;
  reg_cfg.speculative = request.speculative;
  reg_cfg.draft_model_path = request.draft_model_path;
  reg_cfg.vision_model_path = request.vision_model_path;
  reg_cfg.enable_yarn = request.enable_yarn;
  reg_cfg.enable_disk_cache = request.enable_disk_cache;
  reg_cfg.device = request.device;
  reg_cfg.npu_quant = request.npu_quant;
  if (!registry_->Add(request.id, request.aliases, std::move(backend), &error,
                      std::move(reg_cfg)))
    return AdminErr(409, "Conflict", error, "register_failed");
  if (set_default) {
    std::string def_err;
    (void)registry_->SetDefault(request.id, &def_err);
  }
  const std::uint64_t load_ms = NowSteadyMs() - t_load_start;

  json::Value resp = json::Value::object();
  resp["id"] = request.id;
  resp["path"] = request.path;
  resp["backend"] = request.device;
  resp["status"] = "loaded";
  resp["load_ms"] = static_cast<std::size_t>(load_ms);
  resp["estimated_resident_mib"] = needed >> 20;
  if (!evicted.empty()) resp["evicted"] = std::move(evicted);
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleEject(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return AdminErr(400, "Bad Request", e.what(), "parse_error");
  }
  if (!body.is_object())
    return AdminErr(400, "Bad Request", "request body must be an object",
                    "invalid_request");
  const std::string id = body.member_str("id");
  if (id.empty())
    return AdminErr(400, "Bad Request", "'id' is required", "missing_id");
  // M6b L2: in-process NPU servables unload weights via the host.
  // NPU registry wins ties (same id can exist as a sidecar spec).
  if (npu_registry_ != nullptr) {
    auto npu_spec = npu_registry_->Resolve(id);
    if (npu_spec != nullptr &&
        npu_spec->kind ==
            mbufo::core::npu::NpuArtifactKind::kNpuInProcess) {
      std::lock_guard serialize(load_mutex_);
      if (!npu_holder_.IsLoaded(id)) {
        npu_registry_->SetReady(id, false);
        json::Value resp = json::Value::object();
        resp["id"] = id;
        resp["status"] = "ejected";
        resp["note"] = "npu model was already unloaded";
        return AdminOk(std::move(resp));
      }
      std::string npu_err;
      if (!npu_holder_.Unload(id, &npu_err)) {
        return AdminErr(502, "Bad Gateway",
                        npu_err.empty() ? "npu eject failed" : npu_err,
                        "npu_eject_failed");
      }
      npu_registry_->SetReady(id, false);
      json::Value resp = json::Value::object();
      resp["id"] = id;
      resp["status"] = "ejected";
      resp["note"] = "npu weights unloaded";
      return AdminOk(std::move(resp));
    }
  }
  // Sidecars stop their OS process; mirrors the GPU eject flow above.
  if (FindSidecarSpec(id) != nullptr) {
    std::lock_guard serialize(load_mutex_);
    const auto st = GlobalSidecarManager().GetStatus(id);
    const bool was_live = st.running;
    std::string sc_err;
    if (!GlobalSidecarManager().Stop(id, &sc_err)) {
      return AdminErr(502, "Bad Gateway",
                      sc_err.empty() ? "failed to stop sidecar" : sc_err,
                      "sidecar_stop_failed");
    }
    if (!was_live) {
      json::Value already = json::Value::object();
      already["id"] = id;
      already["status"] = "ejected";
      already["note"] = "sidecar was already stopped";
      return AdminOk(std::move(already));
    }
    json::Value resp = json::Value::object();
    resp["id"] = id;
    resp["status"] = "ejected";
    resp["note"] = st.external
                       ? "stopped an adopted external sidecar process"
                       : "sidecar process stopped; port released";
    return AdminOk(std::move(resp));
  }
  if (registry_ == nullptr)
    return AdminErr(404, "Not Found", "model '" + id + "' is not loaded",
                    "not_loaded");
  std::lock_guard serialize(load_mutex_);
  if (!registry_->HasId(id))
    return AdminErr(404, "Not Found", "model '" + id + "' is not loaded",
                    "not_loaded");
  const auto mem_before = QueryMemorySnapshot(probe_);
  std::string error;
  if (!registry_->Remove(id, &error))
    return AdminErr(409, "Conflict", error, "eject_failed");
#if defined(__GLIBC__)
  ::malloc_trim(0);
#endif
  const auto mem_after = QueryMemorySnapshot(probe_);
  Logger::Info("admin",
               "event=model_ejected id=" + id +
                   " host_avail_before_mib=" +
                   std::to_string(mem_before.host_available_bytes >> 20) +
                   " host_avail_after_mib=" +
                   std::to_string(mem_after.host_available_bytes >> 20) +
                   " rss_before_mib=" +
                   std::to_string(mem_before.process_rss_bytes >> 20) +
                   " rss_after_mib=" +
                   std::to_string(mem_after.process_rss_bytes >> 20));
  json::Value resp = json::Value::object();
  resp["id"] = id;
  resp["status"] = "ejected";
  resp["memory"] = mem_after.ToJson();
  resp["note"] =
      "model dropped from registry and GPU/host memory reclaimed (in-flight requests finish cleanly first if busy)";
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleSetDefault(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return AdminErr(400, "Bad Request", e.what(), "parse_error");
  }
  if (!body.is_object())
    return AdminErr(400, "Bad Request", "request body must be an object",
                    "invalid_request");
  const std::string id = body.member_str("id");
  if (id.empty())
    return AdminErr(400, "Bad Request", "'id' is required", "missing_id");
  std::string error;
  if (!registry_->SetDefault(id, &error))
    return AdminErr(404, "Not Found", error, "not_loaded");
  json::Value resp = json::Value::object();
  resp["id"] = id;
  resp["status"] = "default_updated";
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleReload(const HttpRequest& req) {
  if (!loader_)
    return AdminErr(503, "Service Unavailable",
                    "dynamic model loading is not enabled", "loader_disabled");
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return AdminErr(400, "Bad Request", e.what(), "parse_error");
  }
  if (!body.is_object())
    return AdminErr(400, "Bad Request", "request body must be an object",
                    "invalid_request");

  const std::string target_id =
      body.member_str("target_id", body.member_str("id"));
  if (target_id.empty())
    return AdminErr(400, "Bad Request", "'target_id' or 'id' is required",
                    "missing_id");

  // Sidecar reload: stop existing process and restart with updated options
  if (const auto* sc0 = FindSidecarSpec(target_id); sc0 != nullptr) {
    SidecarStartOptions sc_opts;
    sc_opts.dtype = body.member_str("image_dtype", "bf16");
    sc_opts.steps = static_cast<int>(
        body.member_size("image_steps", sc0->default_steps));
    sc_opts.default_size = body.member_str("image_size", "1024x1024");
    sc_opts.cpu_offload = body.member_str("image_cpu_offload", "none");

    std::lock_guard serialize(load_mutex_);
    {
      std::lock_guard lock(loading_state_mutex_);
      loading_id_ = target_id;
      loading_path_ = sc0->base_url;
      loading_started_ms_ = NowSteadyMs();
    }
    struct ClearSidecarGuard {
      ModelAdminService* self;
      ~ClearSidecarGuard() {
        std::lock_guard lock(self->loading_state_mutex_);
        self->loading_id_.clear();
        self->loading_path_.clear();
        self->loading_started_ms_ = 0;
      }
    } sc_guard{this};

    const std::uint64_t t_start = NowSteadyMs();
    std::string sc_err;
    if (GlobalSidecarManager().IsRunning(target_id)) {
      if (!GlobalSidecarManager().Stop(target_id, &sc_err)) {
        return AdminErr(502, "Bad Gateway",
                        sc_err.empty() ? "failed stopping sidecar" : sc_err,
                        "sidecar_stop_failed");
      }
    }
    if (!GlobalSidecarManager().Start(target_id, &sc_err, 180000, sc_opts)) {
      return AdminErr(502, "Bad Gateway",
                      sc_err.empty() ? "failed restarting sidecar" : sc_err,
                      "sidecar_start_failed");
    }
    const auto st = GlobalSidecarManager().GetStatus(target_id);
    json::Value resp = json::Value::object();
    resp["id"] = target_id;
    resp["status"] = "reloaded";
    resp["sidecar"] = true;
    resp["backend"] = sc0->backend;
    resp["pid"] = static_cast<std::int64_t>(st.pid);
    resp["image_dtype"] = st.options.dtype;
    resp["image_steps"] = static_cast<std::size_t>(st.options.steps);
    resp["image_size"] = st.options.default_size;
    resp["image_cpu_offload"] = st.options.cpu_offload;
    resp["load_ms"] = NowSteadyMs() >= t_start ? NowSteadyMs() - t_start : 0;
    return AdminOk(std::move(resp));
  }

  std::lock_guard serialize(load_mutex_);
  std::optional<ModelRegistry::Info> old_info;
  for (const auto& info : registry_->List()) {
    if (info.id == target_id) {
      old_info = info;
      break;
    }
  }
  if (!old_info.has_value())
    return AdminErr(404, "Not Found", "model '" + target_id + "' is not loaded",
                    "not_loaded");

  AdminLoadRequest request;
  request.id = body.member_str("id", target_id);
  if (request.id.empty()) request.id = target_id;
  if (request.id != target_id && registry_->HasName(request.id)) {
    return AdminErr(409, "Conflict",
                    "target id '" + request.id + "' is already in use",
                    "id_conflict");
  }
  request.path = body.member_str("path", old_info->path);
  if (request.path.empty())
    return AdminErr(400, "Bad Request", "'path' is required", "missing_path");

  if (const auto* aliases = body.find("aliases");
      aliases != nullptr && aliases->is_array()) {
    for (const auto& item : aliases->items()) {
      if (item.is_string() && !item.str().empty())
        request.aliases.push_back(item.str());
    }
  } else if (body.find("aliases") == nullptr) {
    for (const auto& n : old_info->names) {
      if (n != old_info->id) request.aliases.push_back(n);
    }
  }

  request.max_context = static_cast<std::uint32_t>(body.member_size(
      "max_context",
      body.member_size("context_length", old_info->max_context)));
  request.session_count = body.member_size(
      "session_count",
      body.member_size("sessions", std::max<std::size_t>(1, old_info->total_sessions)));
  request.kv_cache_quant = body.member_str(
      "kv_cache_quant",
      body.member_str("kv_quant", old_info->config.kv_cache_quant));
  request.kv_cache_quant_supplied = true;
  request.device = body.member_str(
      "device", old_info->backend.empty() ? "gpu" : old_info->backend);
  if (request.device != "gpu" && request.device != "npu") {
    return AdminErr(400, "Bad Request", "device must be 'gpu' or 'npu'",
                    "invalid_device");
  }
  request.npu_quant = body.member_str("npu_quant", old_info->config.npu_quant);
  request.vision_model_path = body.member_str(
      "vision_model_path",
      body.member_str("mmproj_path", old_info->config.vision_model_path));
  request.enable_yarn =
      body.find("enable_yarn") != nullptr
          ? body.find("enable_yarn")->as_bool(false)
          : old_info->config.enable_yarn;
  request.speculative = body.member_str(
      "speculative", old_info->config.speculative.empty()
                         ? "off"
                         : old_info->config.speculative);
  request.draft_model_path =
      body.member_str("draft_model_path", old_info->config.draft_model_path);
  request.max_draft_tokens = body.member_size("max_draft_tokens", 0);
  request.enable_disk_cache =
      body.find("enable_disk_cache") != nullptr
          ? body.find("enable_disk_cache")->as_bool(false)
          : old_info->config.enable_disk_cache;

  const bool was_default = old_info->is_default;
  const bool set_default =
      was_default || (body.find("set_default") != nullptr &&
                      body.find("set_default")->as_bool(false));
  const bool allow_evict = body.find("allow_evict") == nullptr ||
                           body.find("allow_evict")->as_bool(true);

  AdminLoadRequest rollback_req;
  rollback_req.id = old_info->id;
  for (const auto& n : old_info->names) {
    if (n != old_info->id) rollback_req.aliases.push_back(n);
  }
  rollback_req.path = old_info->path;
  rollback_req.max_context = old_info->max_context;
  rollback_req.session_count = std::max<std::size_t>(1, old_info->total_sessions);
  rollback_req.kv_cache_quant = old_info->config.kv_cache_quant;
  rollback_req.kv_cache_quant_supplied = true;
  rollback_req.vision_model_path = old_info->config.vision_model_path;
  rollback_req.enable_yarn = old_info->config.enable_yarn;
  rollback_req.speculative = old_info->config.speculative;
  rollback_req.draft_model_path = old_info->config.draft_model_path;
  rollback_req.enable_disk_cache = old_info->config.enable_disk_cache;
  rollback_req.device = old_info->config.device.empty() ? old_info->backend
                                                      : old_info->config.device;
  rollback_req.npu_quant = old_info->config.npu_quant;

  {
    std::lock_guard lock(loading_state_mutex_);
    loading_id_ = request.id;
    loading_path_ = request.path;
    loading_started_ms_ = NowSteadyMs();
  }
  struct ClearReloadGuard {
    ModelAdminService* self;
    ~ClearReloadGuard() {
      std::lock_guard lock(self->loading_state_mutex_);
      self->loading_id_.clear();
      self->loading_path_.clear();
      self->loading_started_ms_ = 0;
    }
  } clear_guard{this};

  std::string rm_err;
  if (!registry_->Remove(target_id, &rm_err))
    return AdminErr(409, "Conflict", rm_err, "eject_failed");
#if defined(__GLIBC__)
  ::malloc_trim(0);
#endif

  auto try_rollback = [&](const std::string& reason) -> HttpResponse {
    std::string rb_err;
    auto rb_backend = loader_(rollback_req, &rb_err);
    if (rb_backend != nullptr) {
      std::string add_err;
      if (registry_->Add(rollback_req.id, rollback_req.aliases,
                         std::move(rb_backend), &add_err, old_info->config)) {
        if (was_default) {
          std::string def_err;
          (void)registry_->SetDefault(rollback_req.id, &def_err);
        }
        return AdminErr(
            500, "Internal Server Error",
            reason + " (restored previous configuration for '" + rollback_req.id + "')",
            "reload_rolled_back");
      }
    }
    return AdminErr(500, "Internal Server Error",
                    reason + " (rollback also failed: " + rb_err + ")",
                    "reload_failed");
  };

  json::Value evicted = json::Value::array();
  const std::size_t needed = EstimateResidentBytes(request.path);
  if (probe_ && needed > 0) {
    auto free_bytes = probe_();
    if (free_bytes.has_value() && *free_bytes < needed) {
      if (!allow_evict) {
        return try_rollback("insufficient free device memory for '" + request.id +
                            "': need ~" + std::to_string(needed >> 20) +
                            " MiB, have ~" + std::to_string(*free_bytes >> 20) +
                            " MiB");
      }
      for (const auto& victim : registry_->LruCandidates()) {
        std::string v_err;
        if (!registry_->Remove(victim, &v_err)) continue;
        evicted.push_back(victim);
        free_bytes = probe_();
        if (!free_bytes.has_value() || *free_bytes >= needed) break;
      }
      if (free_bytes.has_value() && *free_bytes < needed) {
        return try_rollback("insufficient free device memory for '" +
                            request.id + "' after LRU eviction");
      }
    }
  }

  const std::uint64_t t_load_start = NowSteadyMs();
  std::string load_err;
  auto backend = loader_(request, &load_err);
  if (backend == nullptr) {
    return try_rollback(load_err.empty() ? "reload failed" : load_err);
  }
  LoadedModelConfig reg_cfg;
  reg_cfg.kv_cache_quant = request.kv_cache_quant;
  reg_cfg.speculative = request.speculative;
  reg_cfg.draft_model_path = request.draft_model_path;
  reg_cfg.vision_model_path = request.vision_model_path;
  reg_cfg.enable_yarn = request.enable_yarn;
  reg_cfg.enable_disk_cache = request.enable_disk_cache;
  reg_cfg.device = request.device;
  reg_cfg.npu_quant = request.npu_quant;
  std::string add_err;
  if (!registry_->Add(request.id, request.aliases, std::move(backend),
                      &add_err, std::move(reg_cfg))) {
    return try_rollback(add_err);
  }
  if (set_default) {
    std::string def_err;
    (void)registry_->SetDefault(request.id, &def_err);
  }
  const std::uint64_t load_ms = NowSteadyMs() - t_load_start;

  json::Value resp = json::Value::object();
  resp["id"] = request.id;
  resp["target_id"] = target_id;
  resp["path"] = request.path;
  resp["backend"] = request.device;
  resp["status"] = "reloaded";
  resp["is_default"] = set_default;
  resp["load_ms"] = static_cast<std::size_t>(load_ms);
  resp["estimated_resident_mib"] = needed >> 20;
  if (!evicted.empty()) resp["evicted"] = std::move(evicted);
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleDiskCacheList(const HttpRequest&) {
  bool live = false;
  DiskCacheSnapshot snap;
  if (auto live_snap = registry_->InspectDiskCache()) {
    live = true;
    snap = std::move(*live_snap);
  } else {
    snap = admin_ops::ListDiskCacheOffline(
        admin_ops::GetConfiguredCacheDiskDir(),
        admin_ops::GetConfiguredCacheDiskBytes());
  }

  std::uint64_t total_chunk_refs_bytes = 0;
  std::uint64_t total_private_bytes = 0;
  std::uint64_t total_manifest_bytes = 0;
  std::size_t compat_count = 0;
  std::size_t incompat_count = 0;

  json::Value entries = json::Value::array();
  for (const auto& e : snap.entries) {
    total_chunk_refs_bytes += e.chunk_bytes;
    total_private_bytes += e.private_bytes;
    total_manifest_bytes += e.manifest_bytes;
    if (e.compatible) ++compat_count;
    else ++incompat_count;

    json::Value row = json::Value::object();
    row["checkpoint_id"] = static_cast<std::size_t>(e.checkpoint_id);
    row["lineage_id"] = static_cast<std::size_t>(e.lineage_id);
    row["manifest_file"] = e.manifest_file;
    row["compatibility_hex"] = e.compatibility_hex;
    row["token_count"] = e.token_count;
    row["chunk_count"] = e.chunk_count;
    row["chunk_bytes"] = static_cast<std::size_t>(e.chunk_bytes);
    row["private_bytes"] = static_cast<std::size_t>(e.private_bytes);
    row["manifest_bytes"] = static_cast<std::size_t>(e.manifest_bytes);
    row["total_referenced_bytes"] = static_cast<std::size_t>(
        e.chunk_bytes + e.private_bytes + e.manifest_bytes);
    row["mtime_epoch"] = static_cast<std::size_t>(
        e.mtime_epoch > 0 ? e.mtime_epoch : 0);
    row["purpose"] = e.purpose;
    row["compatible"] = e.compatible;
    row["invalid"] = e.invalid;
    entries.push_back(std::move(row));
  }

  json::Value resp = json::Value::object();
  resp["enabled"] = snap.enabled;
  resp["live"] = live;
  resp["directory"] = snap.directory;
  resp["capacity_bytes"] = static_cast<std::size_t>(snap.capacity_bytes);
  resp["managed_bytes"] = static_cast<std::size_t>(snap.managed_bytes);
  resp["active_compatibility"] = snap.active_compatibility;
  resp["checkpoint_count"] = snap.entries.size();
  resp["compatible_count"] = compat_count;
  resp["incompatible_count"] = incompat_count;
  resp["sum_private_bytes"] = static_cast<std::size_t>(total_private_bytes);
  resp["sum_manifest_bytes"] = static_cast<std::size_t>(total_manifest_bytes);
  resp["sum_chunk_referenced_bytes"] =
      static_cast<std::size_t>(total_chunk_refs_bytes);
  resp["entries"] = std::move(entries);
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleDiskCacheDelete(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return AdminErr(400, "Bad Request", e.what(), "parse_error");
  }
  if (!body.is_object())
    return AdminErr(400, "Bad Request", "request body must be an object",
                    "invalid_request");

  std::vector<std::uint64_t> ids;
  if (const auto* arr = body.find("ids"); arr != nullptr && arr->is_array()) {
    for (const auto& item : arr->items()) {
      if (item.is_number()) {
        ids.push_back(static_cast<std::uint64_t>(item.as_size(0)));
      } else if (item.is_string()) {
        try {
          ids.push_back(std::stoull(item.str()));
        } catch (...) {
        }
      }
    }
  }
  const bool delete_all = body.find("delete_all") != nullptr &&
                          body.find("delete_all")->as_bool(false);
  const bool delete_incompatible =
      body.find("delete_incompatible") != nullptr &&
      body.find("delete_incompatible")->as_bool(false);

  if (!delete_all && !delete_incompatible && ids.empty()) {
    return AdminErr(400, "Bad Request",
                    "Specify ids, delete_incompatible=true, or delete_all=true",
                    "missing_targets");
  }

  DiskCacheDeleteResult res;
  bool live = false;
  if (auto live_res =
          registry_->RetireDiskCache(ids, delete_all, delete_incompatible)) {
    live = true;
    res = *live_res;
  } else {
    res = admin_ops::DeleteDiskCacheOffline(
        admin_ops::GetConfiguredCacheDiskDir(), ids, delete_all,
        delete_incompatible, "");
  }

  json::Value resp = json::Value::object();
  resp["live"] = live;
  resp["deleted"] = res.deleted;
  resp["skipped_pinned"] = res.skipped_pinned;
  resp["skipped_busy"] = res.skipped_busy;
  resp["not_found"] = res.not_found;
  resp["remaining_managed_bytes"] =
      static_cast<std::size_t>(res.remaining_managed_bytes);
  return AdminOk(std::move(resp));
}

}  // namespace mbufo::server
