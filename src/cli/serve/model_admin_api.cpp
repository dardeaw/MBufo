#include "src/cli/serve/model_admin_api.hpp"

#include <filesystem>
#include <mutex>
#include <string>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::server {
namespace {

// Local copies of the shared Ok/Err helpers (they are static in
// http_server.cpp). Kept minimal for the admin surface.
HttpResponse AdminOk(json::Value body) {
  return {.status = 200, .reason = "OK", .body = body.dump()};
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

std::string DeriveId(const std::string& path) {
  std::filesystem::path p(path);
  std::string name = p.stem().string();
  if (name.empty()) name = path;
  return name;
}

}  // namespace

HttpResponse ModelAdminService::Handle(const HttpRequest& req) {
  if (req.path == "/admin/models" && req.method == "GET")
    return HandleList(req);
  if (req.path == "/admin/load" && req.method == "POST")
    return HandleLoad(req);
  if (req.path == "/admin/eject" && req.method == "POST")
    return HandleEject(req);
  return AdminErr(404, "Not Found", "no admin route for this path",
                  "not_found");
}

HttpResponse ModelAdminService::HandleList(const HttpRequest&) {
  json::Value resp = json::Value::object();
  resp["object"] = "list";
  json::Value data = json::Value::array();
  for (const auto& info : registry_->List()) {
    json::Value row = json::Value::object();
    row["id"] = info.id;
    row["path"] = info.path;
    json::Value names = json::Value::array();
    for (const auto& name : info.names) names.push_back(name);
    row["names"] = std::move(names);
    row["context_length"] = static_cast<std::size_t>(info.max_context);
    row["ready"] = info.ready;
    row["last_used_ms"] = static_cast<std::size_t>(info.last_used_ms);
    data.push_back(std::move(row));
  }
  resp["data"] = std::move(data);
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
  request.max_context =
      static_cast<std::uint32_t>(body.member_size("max_context", 0));
  request.session_count = body.member_size("session_count", 1);
  request.kv_cache_quant = body.member_str("kv_cache_quant", "q8");
  request.vision_model_path = body.member_str("vision_model_path");
  request.enable_yarn = body.find("enable_yarn") != nullptr &&
                        body.find("enable_yarn")->as_bool(false);
  if (request.path.empty())
    return AdminErr(400, "Bad Request", "'path' is required", "missing_path");
  if (request.session_count == 0) request.session_count = 1;
  if (request.id.empty()) request.id = DeriveId(request.path);
  if (registry_->HasName(request.id))
    return AdminErr(409, "Conflict",
                    "model '" + request.id + "' is already loaded",
                    "already_loaded");

  std::string error;
  auto backend = loader_(request, &error);
  if (backend == nullptr)
    return AdminErr(500, "Internal Server Error",
                    error.empty() ? "load failed" : error, "load_failed");
  if (!registry_->Add(request.id, {}, std::move(backend), &error))
    return AdminErr(409, "Conflict", error, "register_failed");

  json::Value resp = json::Value::object();
  resp["id"] = request.id;
  resp["path"] = request.path;
  resp["status"] = "loaded";
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

}  // namespace gufo::server
