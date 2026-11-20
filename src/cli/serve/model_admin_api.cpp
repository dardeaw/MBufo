#include "src/cli/serve/model_admin_api.hpp"

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

struct CapabilityProfile {
  std::string task{"decide"};
  bool has_vision{false};
  bool has_audio{false};
  bool has_tools{false};
  std::vector<std::string> badges;
};

CapabilityProfile AnalyzeCapabilities(std::string_view id_or_name,
                                      std::string_view family,
                                      std::string_view path,
                                      bool has_mmproj,
                                      std::string_view spec_kind) {
  CapabilityProfile prof;
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
  } else {
    prof.task = "decide";
    prof.badges.push_back("DECIDE");
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
  if (prof.task == "decide") {
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

}  // namespace

HttpResponse ModelAdminService::Handle(const HttpRequest& req) {
  if (req.method == "GET" &&
      (req.path == "/" || req.path == "/admin" || req.path == "/admin/" ||
       req.path == "/admin/ui")) {
    return HandleUi(req);
  }
  if (req.path == "/admin/models" && req.method == "GET")
    return HandleList(req);
  if (req.path == "/admin/catalog" && req.method == "GET")
    return HandleCatalog(req);
  if (req.path == "/admin/load" && req.method == "POST")
    return HandleLoad(req);
  if (req.path == "/admin/eject" && req.method == "POST")
    return HandleEject(req);
  if (req.path == "/admin/default" && req.method == "POST")
    return HandleSetDefault(req);
  return AdminErr(404, "Not Found", "no admin route for this path",
                  "not_found");
}

HttpResponse ModelAdminService::HandleUi(const HttpRequest&) {
  HttpResponse resp;
  resp.status = 200;
  resp.reason = "OK";
  resp.headers.emplace_back("Content-Type", "text/html; charset=utf-8");
  resp.headers.emplace_back("Cache-Control", "no-store");
  resp.body = std::string(kModelAdminUiHtml);
  return resp;
}

HttpResponse ModelAdminService::HandleList(const HttpRequest&) {
  json::Value resp = json::Value::object();
  resp["object"] = "list";
  const auto lru_order = registry_->LruCandidates();
  json::Value data = json::Value::array();
  for (const auto& info : registry_->List()) {
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
      const std::string spec_s = info.speculative ? "mtp" : "";
      const auto prof = AnalyzeCapabilities(info.id, "", info.path,
                                            info.supports_images, spec_s);
      row["task"] = prof.task;
      row["has_vision"] = prof.has_vision;
      row["has_audio"] = prof.has_audio;
      row["has_tools"] = prof.has_tools;
      json::Value caps = json::Value::array();
      for (const auto& b : prof.badges) caps.push_back(b);
      row["capabilities"] = std::move(caps);
    }
    row["total_sessions"] = info.total_sessions;
    row["active_sessions"] = info.active_sessions;
    row["busy"] = info.active_sessions > 0;
    const std::size_t file_bytes =
        info.path.empty() ? 0 : TotalGgufBytes(info.path);
    const std::size_t est_bytes =
        info.path.empty() ? 0 : EstimateResidentBytes(info.path);
    row["file_size_mib"] = file_bytes >> 20;
    row["estimated_resident_mib"] = est_bytes >> 20;

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

  // Also include the ancestor directory of any currently loaded model if not
  // already under roots.
  const auto loaded_list = registry_->List();
  for (const auto& info : loaded_list) {
    if (info.path.empty()) continue;
    std::filesystem::path p(info.path);
    if (p.has_parent_path() && p.parent_path().has_parent_path()) {
      auto candidate = p.parent_path().parent_path();
      bool covered = false;
      for (const auto& r : roots) {
        if (info.path.rfind(r.string(), 0) == 0) {
          covered = true;
          break;
        }
      }
      if (!covered) roots.push_back(std::move(candidate));
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

  auto find_sidecar = [&](const std::string& family, GgufRole role) -> std::string {
    // In mbufo serve llm, --mmproj vision is supported on Qwen3.8-Flash-Next
    // and Qwen3.8-27B; other families (gemma4, muse-glimmer) reject --mmproj.
    if (role == GgufRole::kMmproj && family.rfind("qwen3.8-", 0) != 0) {
      return {};
    }
    for (const auto& f : files) {
      if (f.role == role && f.family == family) return f.path.string();
    }
    // Qwen3.8 Flash-Next quant folders (qwen3.8-rocmfp4, qwen3.8-mxfp4) share
    // mmproj and MTP with qwen3.8-flash-next.
    if (family.rfind("qwen3.8-", 0) == 0 && family != "qwen3.8-27b") {
      for (const auto& f : files) {
        if (f.role == role && f.family == "qwen3.8-flash-next")
          return f.path.string();
      }
    }
    return {};
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
    const std::string sug_mtp = find_sidecar(f.family, GgufRole::kDraftMtp);
    const std::string sug_dflash = find_sidecar(f.family, GgufRole::kDraftDFlash);
    const std::string sug_dspark = find_sidecar(f.family, GgufRole::kDraftDSpark);
    const std::string sug_draft = !sug_mtp.empty() ? sug_mtp
                                : !sug_dflash.empty() ? sug_dflash
                                                       : sug_dspark;
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
    {
      const std::string sug_mm = find_sidecar(f.family, GgufRole::kMmproj);
      const std::string spec_k = !sug_mtp.empty()      ? "mtp"
                                 : !sug_dflash.empty() ? "dflash2"
                                 : !sug_dspark.empty() ? "dspark"
                                                       : "";
      const auto prof = AnalyzeCapabilities(derived_id, f.family, full_path,
                                            !sug_mm.empty(), spec_k);
      item["task"] = prof.task;
      item["has_vision"] = prof.has_vision;
      item["has_audio"] = prof.has_audio;
      item["has_tools"] = prof.has_tools;
      json::Value caps = json::Value::array();
      for (const auto& b : prof.badges) caps.push_back(b);
      item["capabilities"] = std::move(caps);
    }
    item["family"] = f.family;
    item["path"] = full_path;
    item["relative_path"] = rel_path;
    item["size_mib"] = f.size_bytes >> 20;
    item["file_size_mib"] = f.size_bytes >> 20;
    item["estimated_resident_mib"] = est_bytes >> 20;
    item["shards"] = 1;
    item["loaded"] = is_loaded;
    item["loaded_id"] = loaded_id;
    item["is_default"] = is_default;
    item["suggested_mmproj"] = find_sidecar(f.family, GgufRole::kMmproj);
    item["suggested_draft"] = sug_draft;
    item["suggested_mtp"] = sug_mtp;
    item["suggested_dflash"] = sug_dflash;
    item["suggested_dspark"] = sug_dspark;
    models.push_back(std::move(item));
  }

  json::Value sidecars = json::Value::object();
  sidecars["mmproj"] = std::move(mmproj_list);
  sidecars["draft"] = std::move(draft_list);

  json::Value resp = json::Value::object();
  resp["object"] = "catalog";
  resp["models"] = std::move(models);
  resp["mmproj_files"] = std::move(mmproj_files);
  resp["draft_files"] = std::move(draft_files);
  resp["sidecars"] = std::move(sidecars);
  resp["memory"] = QueryMemorySnapshot(probe_).ToJson();
  return AdminOk(std::move(resp));
}

HttpResponse ModelAdminService::HandleLoad(const HttpRequest& req) {
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
      body.member_str("kv_cache_quant", body.member_str("kv_quant", "q8"));
  request.kv_cache_quant_supplied =
      body.find("kv_cache_quant") != nullptr || body.find("kv_quant") != nullptr;
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
  if (!registry_->Add(request.id, request.aliases, std::move(backend), &error))
    return AdminErr(409, "Conflict", error, "register_failed");
  if (set_default) {
    std::string def_err;
    (void)registry_->SetDefault(request.id, &def_err);
  }
  const std::uint64_t load_ms = NowSteadyMs() - t_load_start;

  json::Value resp = json::Value::object();
  resp["id"] = request.id;
  resp["path"] = request.path;
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
  if (!registry_->HasId(id))
    return AdminErr(404, "Not Found", "model '" + id + "' is not loaded",
                    "not_loaded");
  std::string error;
  if (!registry_->Remove(id, &error))
    return AdminErr(409, "Conflict", error, "eject_failed");
  json::Value resp = json::Value::object();
  resp["id"] = id;
  resp["status"] = "ejected";
  resp["note"] =
      "in-flight requests finish on the released model before its memory frees";
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

}  // namespace mbufo::server
