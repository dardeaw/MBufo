// Server-side Admin Operations:
//   1. Local persistent cache & sync of model_mapping.json
//   2. Option B HuggingFace repo file browser (quant parsing & exact sizes)
//   3. Background HuggingFace model downloader to MBUFO_MODELS_DIR
//   4. Real-time GitHub version check & detached self-update + restart
#ifndef MBUFO_SERVER_ADMIN_OPS_HPP_
#define MBUFO_SERVER_ADMIN_OPS_HPP_

#include "src/cli/serve/http_server.hpp"

namespace mbufo::server::admin_ops {

/// GET /admin/model-mapping — serve from local persistent cache (seeding from repo if absent).
HttpResponse HandleGetModelMapping(const HttpRequest& req);

/// POST /admin/model-mapping/sync — fetch model_mapping.json from GitHub and update local cache.
HttpResponse HandleSyncModelMapping(const HttpRequest& req);

/// GET /admin/hf/files?repo=<org/repo> — query HuggingFace tree API, parse quant/shards/sizes.
HttpResponse HandleHfListFiles(const HttpRequest& req);

/// POST /admin/download — start background `hf download` into MBUFO_MODELS_DIR/<target_subdir>.
HttpResponse HandleStartDownload(const HttpRequest& req);

/// GET /admin/downloads — inspect active/completed server download jobs & NVMe disk space.
HttpResponse HandleListDownloads(const HttpRequest& req);

/// POST /admin/download/cancel — stop a running background download job.
HttpResponse HandleCancelDownload(const HttpRequest& req);

/// GET /admin/update/check — compare baked binary SHA/tag against GitHub dardeaw/MBufo main.
HttpResponse HandleUpdateCheck(const HttpRequest& req);

/// POST /admin/update/apply — launch scripts/mbufo_self_update.sh to pull, rebuild, and restart.
HttpResponse HandleUpdateApply(const HttpRequest& req);

/// GET /admin/update/status — inspect persistent update status JSON + update log tail.
HttpResponse HandleUpdateStatus(const HttpRequest& req);

}  // namespace mbufo::server::admin_ops

#endif  // MBUFO_SERVER_ADMIN_OPS_HPP_
