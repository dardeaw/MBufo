// Admin API for the multi-model registry: list, load and eject models at
// runtime on the same HTTP port that serves inference. The actual weight load
// is delegated to an injected loader callback so this module stays independent
// of InferenceBackend and the (minutes-long) load runs off the request thread.
#ifndef GUFO_SERVER_MODEL_ADMIN_API_HPP_
#define GUFO_SERVER_MODEL_ADMIN_API_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/model_registry.hpp"

namespace gufo::server {

/// Parameters parsed from a POST /admin/load body.
struct AdminLoadRequest {
  std::string id;              // primary served name; empty = derive from path
  std::string path;            // GGUF file path (required)
  std::uint32_t max_context{0};
  std::size_t session_count{1};
  std::string kv_cache_quant{"q8"};
  bool kv_cache_quant_supplied{false};
  std::string vision_model_path;
  bool enable_yarn{false};
};

/// Loads a model file into a fresh backend and returns it ready for the
/// registry. Runs on a worker thread; may block for minutes. Sets *error and
/// returns null on failure. Injected by serve.cpp so the HTTP layer does not
/// depend on InferenceBackend.
using ModelLoader = std::function<std::shared_ptr<TextGenerationBackend>(
    const AdminLoadRequest& request, std::string* error)>;

/// Reports currently-free device memory in bytes, or nullopt when unknown.
/// Injected by serve.cpp (hipMemGetInfo) so this module stays HIP-free. Used
/// to gate a load and drive LRU eviction under memory pressure.
using MemoryProbe = std::function<std::optional<std::size_t>()>;

/// Serializes load/eject so two concurrent admin calls cannot race the
/// registry or over-commit GPU memory.
class ModelAdminService final {
public:
  ModelAdminService(std::shared_ptr<ModelRegistry> registry, ModelLoader loader,
                    MemoryProbe probe = {})
      : registry_(std::move(registry)),
        loader_(std::move(loader)),
        probe_(std::move(probe)) {}

  /// Dispatches an /admin/* request. `req` is the parsed HTTP request.
  HttpResponse Handle(const HttpRequest& req);

private:
  HttpResponse HandleList(const HttpRequest& req);
  HttpResponse HandleLoad(const HttpRequest& req);
  HttpResponse HandleEject(const HttpRequest& req);

  std::shared_ptr<ModelRegistry> registry_;
  ModelLoader loader_;
  MemoryProbe probe_;
  std::mutex load_mutex_;  // one load/eject at a time
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_MODEL_ADMIN_API_HPP_
