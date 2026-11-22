// Server-side Admin Operations:
//   1. Local model_mapping.json reader (packaged with repo / release updates)
//   2. Option B HuggingFace repo file browser (quant parsing & exact sizes)
//   3. Background HuggingFace model downloader to MBUFO_MODELS_DIR
//   4. Semantic GitHub release tag check & detached self-update + restart
//   5. Persistent server_config.json (port, startup default model, dual-layer auth)
//   6. Offline disk-cache inspection & safe deletion fallback
#ifndef MBUFO_SERVER_ADMIN_OPS_HPP_
#define MBUFO_SERVER_ADMIN_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"

namespace mbufo::server::admin_ops {

struct DefaultModelConfig {
  bool configured{false};
  std::string id;
  std::vector<std::string> aliases;
  std::string path;
  std::uint32_t max_context{0};
  std::size_t session_count{1};
  std::string kv_cache_quant{"q8"};
  std::string device{"gpu"};
  std::string npu_quant;
  std::string speculative{"off"};
  std::string draft_model_path;
  std::string vision_model_path;
  bool enable_yarn{false};
  bool enable_disk_cache{false};
};

struct ServerConfig {
  int port{8741};
  bool web_auth_enabled{false};
  std::string web_password_salt;
  std::string web_password_hash;
  bool api_auth_enabled{false};
  bool api_auth_share_web_password{false};
  std::string api_key_salt;
  std::string api_key_hash;
  DefaultModelConfig default_model;
};

[[nodiscard]] bool LoadServerConfigFile(const std::filesystem::path& path,
                                        ServerConfig* out,
                                        std::string* error = nullptr);
[[nodiscard]] bool SeedServerConfigIfMissing(const std::filesystem::path& path,
                                             const ServerConfig& seed,
                                             std::string* error = nullptr);

void InitServerRuntime(const std::filesystem::path& config_path,
                       int running_port,
                       const std::filesystem::path& cache_disk_dir,
                       std::size_t cache_disk_bytes,
                       bool cli_api_key_active);

[[nodiscard]] int GetRunningPort();
[[nodiscard]] std::filesystem::path GetConfiguredCacheDiskDir();
[[nodiscard]] std::size_t GetConfiguredCacheDiskBytes();

[[nodiscard]] bool IsWebAuthEnabled();
[[nodiscard]] bool CheckWebAuth(const HttpRequest& req);
[[nodiscard]] bool IsApiAuthEnabled();
[[nodiscard]] bool CheckApiAuth(const HttpRequest& req);

/// GET /admin/model-mapping — read model_mapping.json from local repository.
HttpResponse HandleGetModelMapping(const HttpRequest& req);

/// POST /admin/model-mapping/sync — reload local model_mapping.json from repository.
HttpResponse HandleSyncModelMapping(const HttpRequest& req);

/// GET /admin/hf/files?repo=<org/repo> — query HuggingFace tree API, parse quant/shards/sizes.
HttpResponse HandleHfListFiles(const HttpRequest& req);

/// POST /admin/download — start background `hf download` into MBUFO_MODELS_DIR/<target_subdir>.
HttpResponse HandleStartDownload(const HttpRequest& req);

/// GET /admin/downloads — inspect active/completed server download jobs & NVMe disk space.
HttpResponse HandleListDownloads(const HttpRequest& req);

/// POST /admin/download/cancel — stop a running background download job.
HttpResponse HandleCancelDownload(const HttpRequest& req);

/// GET /admin/update/check — compare baked release tag (mbufo-vX.Y.Z) vs GitHub tags.
HttpResponse HandleUpdateCheck(const HttpRequest& req);

/// POST /admin/update/apply — launch scripts/mbufo_self_update.sh to pull, rebuild, and restart.
HttpResponse HandleUpdateApply(const HttpRequest& req);

/// GET /admin/update/status — inspect persistent update status JSON + update log tail.
HttpResponse HandleUpdateStatus(const HttpRequest& req);

/// Web UI & API dual-layer authentication endpoints
HttpResponse HandleAuthStatus(const HttpRequest& req);
HttpResponse HandleAuthLogin(const HttpRequest& req);
HttpResponse HandleAuthLogout(const HttpRequest& req);

/// Server settings & graceful restart
HttpResponse HandleGetSettings(const HttpRequest& req,
                               const std::string& live_default_id = "");
HttpResponse HandleSaveSettings(const HttpRequest& req);
HttpResponse HandleServerRestart(const HttpRequest& req);

/// Offline KV disk-cache inspection/retirement when no live backend owns the directory.
[[nodiscard]] DiskCacheSnapshot ListDiskCacheOffline(
    const std::filesystem::path& directory, std::uint64_t capacity_bytes);
[[nodiscard]] DiskCacheDeleteResult DeleteDiskCacheOffline(
    const std::filesystem::path& directory,
    const std::vector<std::uint64_t>& ids, bool delete_all,
    bool delete_incompatible, std::string_view active_compatibility);

}  // namespace mbufo::server::admin_ops

#endif  // MBUFO_SERVER_ADMIN_OPS_HPP_
