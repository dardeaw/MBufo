#include "src/cli/serve/admin_ops.hpp"

#include <curl/curl.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mbufo_build_version.hpp"
#include "src/core/json.hpp"

namespace mbufo::server::admin_ops {
namespace {

namespace fs = std::filesystem;

constexpr const char* kGithubRawMappingUrl =
    "https://raw.githubusercontent.com/dardeaw/MBufo/main/model_mapping.json";
constexpr const char* kGithubCommitsMainUrl =
    "https://api.github.com/repos/dardeaw/MBufo/commits/main";
constexpr const char* kGithubTagsUrl =
    "https://api.github.com/repos/dardeaw/MBufo/tags?per_page=1";
constexpr std::uint64_t kSafetyDiskMarginBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

HttpResponse OkJson(json::Value body) {
  return {.status = 200,
          .reason = "OK",
          .body = body.dump(),
          .headers = {{"Cache-Control", "no-store, no-cache, must-revalidate"},
                      {"Content-Type", "application/json; charset=utf-8"}}};
}

HttpResponse OkRawJson(std::string body) {
  return {.status = 200,
          .reason = "OK",
          .body = std::move(body),
          .headers = {{"Cache-Control", "no-store, no-cache, must-revalidate"},
                      {"Content-Type", "application/json; charset=utf-8"}}};
}

HttpResponse ErrJson(int status, const std::string& reason,
                     const std::string& message, const std::string& code) {
  json::Value obj = json::Value::object();
  obj["message"] = message;
  obj["type"] = "admin_error";
  obj["code"] = code;
  json::Value e = json::Value::object();
  e["error"] = std::move(obj);
  return {.status = status,
          .reason = reason,
          .body = e.dump(),
          .headers = {{"Content-Type", "application/json; charset=utf-8"}}};
}

std::string ToLower(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::uint64_t NowEpochSec() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

std::string NowIsoUtc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm_utc{};
  gmtime_r(&t, &tm_utc);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  return std::string(buf);
}

std::string ReadWholeFile(const fs::path& p) {
  std::ifstream ifs(p, std::ios::binary);
  if (!ifs) return {};
  return std::string((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
}

bool WriteFileAtomic(const fs::path& p, const std::string& content,
                     std::string* error = nullptr) {
  std::error_code ec;
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path(), ec);
    if (ec) {
      if (error != nullptr) *error = "mkdir failed: " + ec.message();
      return false;
    }
  }
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    if (!ofs) {
      if (error != nullptr) *error = "cannot write temp file: " + tmp.string();
      return false;
    }
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!ofs) {
      if (error != nullptr) *error = "write failed: " + tmp.string();
      return false;
    }
  }
  fs::rename(tmp, p, ec);
  if (ec) {
    if (error != nullptr) *error = "rename failed: " + ec.message();
    return false;
  }
  return true;
}

std::string ReadTailLines(const fs::path& p, std::size_t max_lines = 30) {
  const std::string raw = ReadWholeFile(p);
  if (raw.empty()) return {};
  std::vector<std::string> lines;
  std::istringstream iss(raw);
  std::string line;
  while (std::getline(iss, line)) {
    // Strip trailing CR from progress bars cleanly; keep last line segment after \r
    const auto cr = line.rfind('\r');
    if (cr != std::string::npos) {
      line = line.substr(cr + 1);
    }
    if (!line.empty()) lines.push_back(std::move(line));
  }
  if (lines.size() > max_lines) {
    lines.erase(lines.begin(),
                lines.begin() + static_cast<std::ptrdiff_t>(lines.size() - max_lines));
  }
  std::string out;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i > 0) out.push_back('\n');
    out += lines[i];
  }
  return out;
}

fs::path ResolveRepoDir() {
  if (const char* e = std::getenv("MBUFO_REPO_DIR"); e != nullptr && *e != '\0') {
    return fs::path(e);
  }
  if (fs::exists("/home/dardeaw/git/gufo-final2")) {
    return fs::path("/home/dardeaw/git/gufo-final2");
  }
  return fs::current_path();
}

fs::path ResolveModelsRoot() {
  if (const char* e = std::getenv("MBUFO_MODELS_DIR");
      e != nullptr && *e != '\0') {
    return fs::path(e);
  }
  if (fs::exists("/mnt/nvme-data/mbufo-models")) {
    return fs::path("/mnt/nvme-data/mbufo-models");
  }
  return fs::path("models");
}

fs::path ResolveCacheRoot() {
  if (fs::exists("/mnt/nvme-data/mbufo-cache")) {
    return fs::path("/mnt/nvme-data/mbufo-cache");
  }
  if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
    return fs::path(home) / ".cache/mbufo";
  }
  return fs::path("/tmp/mbufo-cache");
}

std::string ResolveHfBinary() {
  if (const char* e = std::getenv("MBUFO_HF_BIN"); e != nullptr && *e != '\0') {
    return e;
  }
  if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
    for (const char* rel : {"/.local/bin/hf", "/venv-npu/bin/hf"}) {
      const fs::path cand = fs::path(home) / (rel + 1);
      if (::access(cand.c_str(), X_OK) == 0) return cand.string();
    }
  }
  for (const char* fixed : {"/home/dardeaw/.local/bin/hf",
                            "/home/dardeaw/venv-npu/bin/hf",
                            "/usr/local/bin/hf", "/usr/bin/hf"}) {
    if (::access(fixed, X_OK) == 0) return fixed;
  }
  return "hf";
}

void QueryDiskSpace(const fs::path& dir, std::uint64_t* out_free,
                    std::uint64_t* out_total) {
  std::error_code ec;
  fs::path probe = dir;
  if (!fs::exists(probe, ec) && probe.has_parent_path()) {
    probe = probe.parent_path();
  }
  const auto sp = fs::space(probe, ec);
  if (!ec) {
    *out_free = static_cast<std::uint64_t>(sp.available);
    *out_total = static_cast<std::uint64_t>(sp.capacity);
  } else {
    *out_free = 0;
    *out_total = 0;
  }
}

std::size_t CurlWriteCallback(void* ptr, std::size_t size, std::size_t nmemb,
                              void* userdata) {
  const std::size_t total = size * nmemb;
  auto* out = static_cast<std::string*>(userdata);
  out->append(static_cast<const char*>(ptr), total);
  return total;
}

bool HttpGet(const std::string& url,
             const std::vector<std::string>& extra_headers, long timeout_sec,
             long* out_http_status, std::string* out_body,
             std::string* out_err) {
  static std::once_flag s_curl_init;
  std::call_once(s_curl_init, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });

  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    if (out_err != nullptr) *out_err = "curl_easy_init failed";
    return false;
  }

  std::string buf;
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "User-Agent: mbufo-admin/1.0");
  headers = curl_slist_append(headers, "Accept: application/json");
  for (const auto& h : extra_headers) {
    headers = curl_slist_append(headers, h.c_str());
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);

  const CURLcode rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (out_http_status != nullptr) *out_http_status = http_code;
  if (rc != CURLE_OK) {
    if (out_err != nullptr) *out_err = curl_easy_strerror(rc);
    return false;
  }
  if (out_body != nullptr) *out_body = std::move(buf);
  return true;
}

bool IsValidRepoSlug(std::string_view repo) {
  const auto slash = repo.find('/');
  if (slash == std::string_view::npos || slash == 0 ||
      slash + 1 >= repo.size()) {
    return false;
  }
  if (repo.find('/', slash + 1) != std::string_view::npos) return false;
  if (repo.find("..") != std::string_view::npos) return false;
  for (char c : repo) {
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '/' ||
                    c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

bool SanitizeSubdir(std::string_view raw, std::string* out) {
  std::string s(raw);
  while (!s.empty() && (s.front() == '/' || s.front() == ' ')) s.erase(0, 1);
  while (!s.empty() && (s.back() == '/' || s.back() == ' ')) s.pop_back();
  if (s.empty() || s.find("..") != std::string::npos ||
      s.find('\\') != std::string::npos) {
    return false;
  }
  for (char& c : s) {
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '.' ||
                    c == '_' || c == '-' || c == '/';
    if (!ok) c = '-';
  }
  *out = std::move(s);
  return true;
}

bool IsValidRepoFilePath(std::string_view fp) {
  if (fp.empty() || fp.front() == '/' || fp.find("..") != std::string::npos ||
      fp.find('\\') != std::string::npos) {
    return false;
  }
  return true;
}

void CloseInheritedFdsAndExec(const std::vector<std::string>& args,
                              const std::string& log_path,
                              const std::string& extra_hf_token,
                              pid_t* out_pid, std::string* error) {
  std::error_code ec;
  fs::create_directories(fs::path(log_path).parent_path(), ec);
  const int log_fd =
      ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (log_fd < 0) {
    if (error != nullptr) {
      *error = "cannot open log " + log_path + ": " + std::strerror(errno);
    }
    *out_pid = -1;
    return;
  }

  std::vector<std::string> args_copy = args;
  std::vector<char*> argv;
  argv.reserve(args_copy.size() + 1);
  for (auto& a : args_copy) argv.push_back(a.data());
  argv.push_back(nullptr);

  long max_fd = ::sysconf(_SC_OPEN_MAX);
  if (max_fd < 256 || max_fd > 65536) max_fd = 4096;

  const pid_t child = ::fork();
  if (child < 0) {
    const int e = errno;
    ::close(log_fd);
    if (error != nullptr) {
      *error = std::string("fork failed: ") + std::strerror(e);
    }
    *out_pid = -1;
    return;
  }
  if (child == 0) {
    ::setsid();
    ::signal(SIGPIPE, SIG_DFL);
    ::signal(SIGTERM, SIG_DFL);
    if (::dup2(log_fd, STDOUT_FILENO) < 0) ::_exit(127);
    if (::dup2(log_fd, STDERR_FILENO) < 0) ::_exit(127);
    for (int fd = 3; fd < static_cast<int>(max_fd); ++fd) {
      if (fd != log_fd) ::close(fd);
    }
    if (log_fd >= 3) ::close(log_fd);
    const int dev_null = ::open("/dev/null", O_RDONLY);
    if (dev_null >= 0) {
      if (dev_null != STDIN_FILENO) {
        ::dup2(dev_null, STDIN_FILENO);
        ::close(dev_null);
      }
    }
    if (!extra_hf_token.empty()) {
      ::setenv("HF_TOKEN", extra_hf_token.c_str(), 1);
    }
    ::execvp(argv[0], argv.data());
    ::_exit(127);
  }
  ::close(log_fd);
  *out_pid = child;
}

// ---- Quantization & shard grouping for Option B HF Browser ----

struct ParsedGgufInfo {
  std::string quant;
  std::string role;   // "main", "mmproj", "draft"
  std::string group_key;
  int shard_idx{1};
  int shard_total{1};
};

std::string DetectQuantFromFilename(const std::string& path_str) {
  static const std::regex kQuantRe(
      "(ROCmFP4-FAST|ROCmFP4|Q4_0_ROCMFP4_FAST|ROCmFP3|MXFP4|BF16|F16|F32|"
      "UD-Q8_K_XL|UD-Q6_K_XL|UD-Q5_K_XL|UD-Q4_K_XL|UD-Q3_K_XL|UD-Q2_K_XL|"
      "UD-IQ4_XS|UD-IQ4_NL|UD-IQ3_XXS|UD-IQ3_S|UD-IQ3_M|UD-IQ2_XXS|UD-IQ2_XS|UD-IQ2_S|UD-IQ2_M|UD-IQ1_M|UD-IQ1_S|"
      "IQ4_NL|IQ4_XS|IQ3_S|IQ3_XS|IQ3_M|IQ3_XXS|IQ2_XXS|IQ2_XS|IQ2_S|IQ2_M|"
      "IQ1_S|IQ1_M|Q8_0|Q6_K_L|Q6_K|Q5_K_M|Q5_K_S|Q5_K_L|Q5_0|Q5_1|"
      "Q4_K_M|Q4_K_S|Q4_K_L|Q4_0|Q4_1|Q3_K_L|Q3_K_M|Q3_K_S|Q2_K_L|Q2_K|"
      "v2q8k|W8A8|W4A16)",
      std::regex::icase);

  // Check basename first, then parent directory name
  const fs::path p(path_str);
  const std::string base = p.filename().string();
  std::smatch m;
  if (std::regex_search(base, m, kQuantRe)) {
    return m.str(1);
  }
  const std::string parent = p.parent_path().filename().string();
  if (!parent.empty() && std::regex_search(parent, m, kQuantRe)) {
    return m.str(1);
  }
  return "GGUF";
}

ParsedGgufInfo ParseGgufPath(const std::string& rel_path) {
  ParsedGgufInfo info;
  const std::string lower = ToLower(rel_path);
  if (lower.find("imatrix") != std::string::npos) {
    info.role = "imatrix";
  } else if (lower.find("mmproj") != std::string::npos) {
    info.role = "mmproj";
  } else if (lower.find("mtp") != std::string::npos ||
             lower.find("dflash") != std::string::npos ||
             lower.find("dspark") != std::string::npos ||
             lower.find("draft") != std::string::npos) {
    info.role = "draft";
  } else {
    info.role = "main";
  }
  info.quant = DetectQuantFromFilename(rel_path);

  static const std::regex kShardRe(
      "^(.*)-([0-9]{5})-of-([0-9]{5})\\.gguf$", std::regex::icase);
  std::smatch sm;
  if (std::regex_match(rel_path, sm, kShardRe)) {
    info.group_key = sm.str(1) + ".gguf";
    info.shard_idx = std::atoi(sm.str(2).c_str());
    info.shard_total = std::atoi(sm.str(3).c_str());
  } else {
    info.group_key = rel_path;
    info.shard_idx = 1;
    info.shard_total = 1;
  }
  return info;
}

bool AllFilesExistUnder(const fs::path& dir,
                        const std::vector<std::string>& files) {
  if (files.empty()) return false;
  std::error_code ec;
  if (!fs::exists(dir, ec)) return false;
  for (const auto& f : files) {
    if (!fs::exists(dir / f, ec) &&
        !fs::exists(dir / fs::path(f).filename(), ec)) {
      return false;
    }
  }
  return true;
}

// ---- Background Download Job Registry ----

struct DownloadJob {
  std::string id;
  std::string repo;
  std::string target_subdir;
  std::string target_dir;
  std::string log_path;
  std::vector<std::string> files;
  bool full_repo{false};
  std::uint64_t expected_bytes{0};
  std::uint64_t started_epoch{0};
  std::uint64_t finished_epoch{0};
  pid_t pid{-1};
  std::string status{"running"};  // running, completed, failed, cancelled
  int exit_code{0};
  std::string error;
};

std::mutex g_dl_mutex;
std::vector<DownloadJob> g_dl_jobs;
std::uint64_t g_dl_seq = 0;

std::uint64_t SumDirectoryBytes(const fs::path& dir) {
  std::error_code ec;
  if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return 0;
  std::uint64_t total = 0;
  fs::recursive_directory_iterator it(
      dir, fs::directory_options::skip_permission_denied, ec);
  fs::recursive_directory_iterator end;
  for (; !ec && it != end; it.increment(ec)) {
    std::error_code fec;
    if (!it->is_regular_file(fec)) continue;
    const std::string fname = it->path().filename().string();
    // Skip lock/metadata tiny files if desired, count weights + .incomplete blobs
    if (fname.size() >= 5 && fname.substr(fname.size() - 5) == ".lock") continue;
    const auto sz = it->file_size(fec);
    if (!fec) total += static_cast<std::uint64_t>(sz);
  }
  return total;
}

void ReapDownloadJobsLocked() {
  for (auto& job : g_dl_jobs) {
    if (job.status != "running" || job.pid <= 0) continue;
    int st = 0;
    const pid_t r = ::waitpid(job.pid, &st, WNOHANG);
    if (r == job.pid) {
      job.finished_epoch = NowEpochSec();
      if (WIFEXITED(st)) {
        job.exit_code = WEXITSTATUS(st);
        job.status = (job.exit_code == 0) ? "completed" : "failed";
        if (job.status == "failed") {
          job.error = "hf exited with code " + std::to_string(job.exit_code);
        }
      } else if (WIFSIGNALED(st)) {
        job.exit_code = 128 + WTERMSIG(st);
        job.status = "cancelled";
      } else {
        job.status = "failed";
      }
      job.pid = -1;
    } else if (r < 0 && errno == ECHILD) {
      job.finished_epoch = NowEpochSec();
      job.status = "completed";
      job.pid = -1;
    }
  }
}

// ---- Version Check Cache ----

struct UpdateCheckCache {
  std::mutex mu;
  std::uint64_t fetched_epoch{0};
  std::string remote_sha;
  std::string remote_short;
  std::string remote_tag;
  std::string remote_message;
  std::string remote_date;
  std::string remote_author;
  std::string check_error;
};

UpdateCheckCache g_upd_cache;

json::Value ReadPersistentUpdateStatus() {
  const fs::path sf = ResolveCacheRoot() / "update_status.json";
  const std::string raw = ReadWholeFile(sf);
  if (raw.empty()) {
    json::Value obj = json::Value::object();
    obj["phase"] = "idle";
    obj["active"] = false;
    return obj;
  }
  try {
    auto v = json::parse(raw);
    if (v.is_object()) return v;
  } catch (...) {
  }
  json::Value obj = json::Value::object();
  obj["phase"] = "idle";
  obj["active"] = false;
  return obj;
}

}  // namespace

// ---- Local model_mapping.json cache & sync ----

HttpResponse HandleGetModelMapping(const HttpRequest&) {
  const fs::path cache_file = ResolveCacheRoot() / "model_mapping.json";
  std::string content = ReadWholeFile(cache_file);
  if (!content.empty()) {
    try {
      auto parsed = json::parse(content);
      if (parsed.is_object()) return OkRawJson(std::move(content));
    } catch (...) {
    }
  }

  const fs::path repo_dir = ResolveRepoDir();
  for (const fs::path& cand :
       {repo_dir / "model_mapping.json", fs::path("model_mapping.json")}) {
    content = ReadWholeFile(cand);
    if (!content.empty()) {
      WriteFileAtomic(cache_file, content);
      return OkRawJson(std::move(content));
    }
  }
  return ErrJson(404, "Not Found",
                 "model_mapping.json not found in cache or repository",
                 "mapping_not_found");
}

HttpResponse HandleSyncModelMapping(const HttpRequest&) {
  long status = 0;
  std::string body;
  std::string err;
  if (!HttpGet(kGithubRawMappingUrl, {}, 15, &status, &body, &err)) {
    return ErrJson(502, "Bad Gateway", "GitHub fetch failed: " + err,
                   "github_fetch_failed");
  }
  if (status != 200) {
    return ErrJson(502, "Bad Gateway",
                   "GitHub returned HTTP " + std::to_string(status),
                   "github_http_error");
  }
  try {
    auto parsed = json::parse(body);
    if (!parsed.is_object() || parsed.find("models") == nullptr ||
        !parsed.find("models")->is_array()) {
      return ErrJson(502, "Bad Gateway",
                     "Invalid model_mapping.json payload from GitHub",
                     "invalid_mapping");
    }
  } catch (const std::exception& e) {
    return ErrJson(502, "Bad Gateway",
                   std::string("JSON parse error: ") + e.what(),
                   "invalid_mapping_json");
  }

  const fs::path cache_file = ResolveCacheRoot() / "model_mapping.json";
  if (!WriteFileAtomic(cache_file, body, &err)) {
    return ErrJson(500, "Internal Server Error",
                   "Failed writing local mapping cache: " + err,
                   "cache_write_failed");
  }
  return OkRawJson(std::move(body));
}

// ---- Option B: Dynamic HF Repo File Browser ----

HttpResponse HandleHfListFiles(const HttpRequest& req) {
  const std::string repo = req.query_param("repo");
  const std::string subdir_q = req.query_param("subdir");
  if (!IsValidRepoSlug(repo)) {
    return ErrJson(400, "Bad Request",
                   "Invalid HuggingFace repo slug (expected org/model)",
                   "invalid_repo");
  }

  std::vector<std::string> hdrs;
  std::string hf_token = req.header("X-HF-Token");
  if (hf_token.empty()) {
    if (const char* env_tok = std::getenv("HF_TOKEN");
        env_tok != nullptr && *env_tok != '\0') {
      hf_token = env_tok;
    }
  }
  if (!hf_token.empty()) {
    hdrs.push_back("Authorization: Bearer " + hf_token);
  }

  const std::string api_url =
      "https://huggingface.co/api/models/" + repo + "/tree/main?recursive=true";
  long http_code = 0;
  std::string body;
  std::string err;
  if (!HttpGet(api_url, hdrs, 20, &http_code, &body, &err)) {
    return ErrJson(502, "Bad Gateway",
                   "HuggingFace API request failed: " + err,
                   "hf_api_failed");
  }
  if (http_code == 401 || http_code == 403) {
    return ErrJson(static_cast<int>(http_code), "Forbidden",
                   "HuggingFace access denied (gated/private repo? Provide HF_TOKEN).",
                   "hf_auth_required");
  }
  if (http_code != 200) {
    return ErrJson(502, "Bad Gateway",
                   "HuggingFace API returned HTTP " + std::to_string(http_code) +
                       " for " + repo,
                   "hf_http_error");
  }

  json::Value tree;
  try {
    tree = json::parse(body);
  } catch (const std::exception& e) {
    return ErrJson(502, "Bad Gateway",
                   std::string("Failed parsing HF tree response: ") + e.what(),
                   "hf_parse_error");
  }
  if (!tree.is_array()) {
    return ErrJson(502, "Bad Gateway", "HF tree response is not an array",
                   "hf_unexpected_response");
  }

  struct GroupEntry {
    std::string id;
    std::string quant;
    std::string role;
    std::string display_name;
    std::vector<std::string> files;
    std::uint64_t size_bytes{0};
    int shards{0};
    bool full_repo{false};
  };

  std::map<std::string, GroupEntry> gguf_groups;
  std::uint64_t safetensors_total_bytes = 0;
  std::uint64_t safetensors_weight_bytes = 0;
  int safetensors_count = 0;
  std::vector<std::pair<std::string, std::uint64_t>> individual_st_files;

  for (const auto& item : tree.items()) {
    if (!item.is_object()) continue;
    if (item.member_str("type") != "file") continue;
    const std::string path_str = item.member_str("path");
    if (path_str.empty()) continue;

    std::uint64_t sz = 0;
    if (const auto* lfs = item.find("lfs"); lfs != nullptr && lfs->is_object()) {
      if (const auto* lsz = lfs->find("size");
          lsz != nullptr && lsz->is_number()) {
        sz = static_cast<std::uint64_t>(lsz->as_double(0.0));
      }
    }
    if (sz == 0) {
      if (const auto* rsz = item.find("size");
          rsz != nullptr && rsz->is_number()) {
        sz = static_cast<std::uint64_t>(rsz->as_double(0.0));
      }
    }

    const std::string low = ToLower(path_str);
    if (low.size() >= 5 && low.substr(low.size() - 5) == ".gguf") {
      const ParsedGgufInfo pi = ParseGgufPath(path_str);
      auto& g = gguf_groups[pi.group_key];
      if (g.files.empty()) {
        g.id = pi.group_key;
        g.quant = pi.quant;
        g.role = pi.role;
        g.display_name = pi.group_key;
      }
      g.files.push_back(path_str);
      g.size_bytes += sz;
      g.shards = std::max(g.shards + 1, pi.shard_total);
    } else if (low.size() >= 12 &&
               low.substr(low.size() - 12) == ".safetensors") {
      ++safetensors_count;
      safetensors_weight_bytes += sz;
      safetensors_total_bytes += sz;
      individual_st_files.emplace_back(path_str, sz);
    } else {
      // Skip large media sample images/videos in README when estimating safetensors repo size
      const bool is_media =
          (low.size() >= 4 && (low.substr(low.size() - 4) == ".png" ||
                               low.substr(low.size() - 4) == ".jpg" ||
                               low.substr(low.size() - 4) == ".mp4" ||
                               low.substr(low.size() - 4) == ".wav")) ||
          (low.size() >= 5 && low.substr(low.size() - 5) == ".webp");
      if (!is_media) safetensors_total_bytes += sz;
    }
  }

  std::string default_subdir = subdir_q;
  if (default_subdir.empty()) {
    const auto slash = repo.find('/');
    default_subdir = ToLower(repo.substr(slash + 1));
  }
  std::string clean_subdir;
  if (!SanitizeSubdir(default_subdir, &clean_subdir)) clean_subdir = "model";
  const fs::path models_root = ResolveModelsRoot();
  const fs::path check_dir = models_root / clean_subdir;

  std::vector<GroupEntry> entries;
  if (safetensors_count > 0) {
    GroupEntry bundle;
    bundle.id = "full-safetensors-repo";
    bundle.quant = "BF16 / Safetensors";
    bundle.role = "safetensors_bundle";
    bundle.display_name =
        "Full Safetensors Model (" + std::to_string(safetensors_count) + " weight files + configs)";
    bundle.files = {"*"};
    bundle.size_bytes = safetensors_total_bytes;
    bundle.shards = safetensors_count;
    bundle.full_repo = true;
    entries.push_back(std::move(bundle));
  }

  for (auto& [k, g] : gguf_groups) {
    g.shards = static_cast<int>(g.files.size());
    entries.push_back(std::move(g));
  }

  std::sort(entries.begin(), entries.end(),
            [](const GroupEntry& a, const GroupEntry& b) {
              auto rank = [](const GroupEntry& e) {
                if (e.full_repo) return 0;
                if (e.role == "main") return 1;
                if (e.role == "mmproj") return 2;
                return 3;
              };
              const int ra = rank(a), rb = rank(b);
              if (ra != rb) return ra < rb;
              return a.size_bytes < b.size_bytes;
            });

  // Also include individual safetensors if no GGUFs exist and <= 20 files
  if (gguf_groups.empty() && individual_st_files.size() > 1 &&
      individual_st_files.size() <= 30) {
    for (const auto& [fp, sz] : individual_st_files) {
      GroupEntry g;
      g.id = fp;
      g.quant = "Safetensors";
      g.role = "shard";
      g.display_name = fp;
      g.files = {fp};
      g.size_bytes = sz;
      g.shards = 1;
      entries.push_back(std::move(g));
    }
  }

  json::Value out_files = json::Value::array();
  for (const auto& g : entries) {
    json::Value item = json::Value::object();
    item["id"] = g.id;
    item["name"] = g.display_name;
    item["quant"] = g.quant;
    item["role"] = g.role;
    item["shards"] = g.shards;
    item["full_repo"] = g.full_repo;
    item["size_bytes"] = static_cast<double>(g.size_bytes);
    std::error_code on_ec;
    const bool on_disk = g.full_repo
                             ? (fs::exists(check_dir / "config.json", on_ec) ||
                                fs::exists(check_dir / "model_index.json", on_ec))
                             : AllFilesExistUnder(check_dir, g.files);
    item["on_disk"] = on_disk;
    json::Value farr = json::Value::array();
    for (const auto& f : g.files) farr.push_back(f);
    item["files"] = std::move(farr);
    out_files.push_back(std::move(item));
  }

  std::uint64_t disk_free = 0, disk_total = 0;
  QueryDiskSpace(models_root, &disk_free, &disk_total);

  json::Value resp = json::Value::object();
  resp["repo"] = repo;
  resp["suggested_subdir"] = clean_subdir;
  resp["models_dir"] = models_root.string();
  resp["disk_free_bytes"] = static_cast<double>(disk_free);
  resp["disk_total_bytes"] = static_cast<double>(disk_total);
  resp["files"] = std::move(out_files);
  return OkJson(std::move(resp));
}

// ---- Server-side `hf download` Manager ----

HttpResponse HandleStartDownload(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return ErrJson(400, "Bad Request", e.what(), "parse_error");
  }
  if (!body.is_object()) {
    return ErrJson(400, "Bad Request", "request body must be an object",
                   "invalid_request");
  }

  const std::string repo = body.member_str("repo");
  if (!IsValidRepoSlug(repo)) {
    return ErrJson(400, "Bad Request", "invalid HuggingFace repo slug",
                   "invalid_repo");
  }

  std::string raw_subdir = body.member_str("target_subdir");
  if (raw_subdir.empty()) {
    raw_subdir = ToLower(repo.substr(repo.find('/') + 1));
  }
  std::string clean_subdir;
  if (!SanitizeSubdir(raw_subdir, &clean_subdir)) {
    return ErrJson(400, "Bad Request", "invalid target_subdir",
                   "invalid_subdir");
  }

  const bool full_repo =
      body.find("full_repo") != nullptr && body.find("full_repo")->as_bool(false);
  std::vector<std::string> files;
  if (const auto* farr = body.find("files"); farr != nullptr && farr->is_array()) {
    for (const auto& v : farr->items()) {
      if (!v.is_string()) continue;
      const std::string fp = v.str();
      if (fp == "*") continue;
      if (!IsValidRepoFilePath(fp)) {
        return ErrJson(400, "Bad Request", "invalid file path: " + fp,
                       "invalid_file");
      }
      files.push_back(fp);
    }
  }
  if (!full_repo && files.empty()) {
    return ErrJson(400, "Bad Request", "no files selected for download",
                   "no_files");
  }

  std::uint64_t expected_bytes = 0;
  if (const auto* tb = body.find("total_bytes");
      tb != nullptr && tb->is_number()) {
    expected_bytes = static_cast<std::uint64_t>(tb->as_double(0.0));
  }

  const fs::path models_root = ResolveModelsRoot();
  const fs::path target_dir = models_root / clean_subdir;
  std::error_code ec;
  fs::create_directories(target_dir, ec);
  if (ec) {
    return ErrJson(500, "Internal Server Error",
                   "cannot create target dir: " + ec.message(),
                   "mkdir_failed");
  }

  std::uint64_t disk_free = 0, disk_total = 0;
  QueryDiskSpace(models_root, &disk_free, &disk_total);
  if (expected_bytes > 0 && disk_free > 0 &&
      expected_bytes + kSafetyDiskMarginBytes > disk_free) {
    return ErrJson(409, "Conflict",
                   "Insufficient disk space on " + models_root.string() +
                       " (need ~" +
                       std::to_string(expected_bytes >> 30) + " GiB + 2 GiB safety, available " +
                       std::to_string(disk_free >> 30) + " GiB)",
                   "insufficient_disk");
  }

  const std::string hf_token = body.member_str("hf_token");
  const std::string hf_bin = ResolveHfBinary();

  std::lock_guard<std::mutex> lock(g_dl_mutex);
  ReapDownloadJobsLocked();

  // Prevent duplicate active download to same repo + target_subdir
  for (const auto& j : g_dl_jobs) {
    if (j.status == "running" && j.repo == repo &&
        j.target_subdir == clean_subdir) {
      return ErrJson(409, "Conflict",
                     "A download for " + repo + " into " + clean_subdir +
                         " is already running (job " + j.id + ")",
                     "already_downloading");
    }
  }

  const std::string job_id =
      "dl-" + std::to_string(NowEpochSec()) + "-" + std::to_string(++g_dl_seq);
  const fs::path log_path =
      ResolveCacheRoot() / "downloads" / (job_id + ".log");

  std::vector<std::string> args = {hf_bin, "download", repo};
  if (full_repo) {
    for (const char* pat : {"*.png", "*.jpg", "*.jpeg", "*.webp", "*.mp4", "*.gif"}) {
      args.emplace_back("--exclude");
      args.emplace_back(pat);
    }
  } else {
    for (const auto& f : files) {
      args.push_back(f);
    }
  }
  args.emplace_back("--local-dir");
  args.push_back(target_dir.string());

  pid_t child = -1;
  std::string err;
  CloseInheritedFdsAndExec(args, log_path.string(), hf_token, &child, &err);
  if (child < 0) {
    return ErrJson(500, "Internal Server Error",
                   "failed launching hf download: " + err,
                   "download_launch_failed");
  }

  DownloadJob job;
  job.id = job_id;
  job.repo = repo;
  job.target_subdir = clean_subdir;
  job.target_dir = target_dir.string();
  job.log_path = log_path.string();
  job.files = std::move(files);
  job.full_repo = full_repo;
  job.expected_bytes = expected_bytes;
  job.started_epoch = NowEpochSec();
  job.pid = child;
  job.status = "running";
  g_dl_jobs.push_back(std::move(job));

  json::Value resp = json::Value::object();
  resp["id"] = job_id;
  resp["repo"] = repo;
  resp["target_dir"] = target_dir.string();
  resp["status"] = "running";
  resp["pid"] = static_cast<long long>(child);
  return OkJson(std::move(resp));
}

HttpResponse HandleListDownloads(const HttpRequest&) {
  std::lock_guard<std::mutex> lock(g_dl_mutex);
  ReapDownloadJobsLocked();

  json::Value arr = json::Value::array();
  for (auto it = g_dl_jobs.rbegin(); it != g_dl_jobs.rend(); ++it) {
    const auto& j = *it;
    const std::uint64_t downloaded_bytes = SumDirectoryBytes(j.target_dir);
    json::Value row = json::Value::object();
    row["id"] = j.id;
    row["repo"] = j.repo;
    row["target_subdir"] = j.target_subdir;
    row["target_dir"] = j.target_dir;
    row["full_repo"] = j.full_repo;
    row["file_count"] = static_cast<std::size_t>(j.files.size());
    row["status"] = j.status;
    row["pid"] = static_cast<long long>(j.pid);
    row["expected_bytes"] = static_cast<double>(j.expected_bytes);
    row["downloaded_bytes"] = static_cast<double>(downloaded_bytes);
    row["started_epoch"] = static_cast<double>(j.started_epoch);
    row["finished_epoch"] = static_cast<double>(j.finished_epoch);
    row["exit_code"] = j.exit_code;
    row["error"] = j.error;
    row["log_tail"] = ReadTailLines(j.log_path, 8);
    arr.push_back(std::move(row));
  }

  const fs::path models_root = ResolveModelsRoot();
  std::uint64_t disk_free = 0, disk_total = 0;
  QueryDiskSpace(models_root, &disk_free, &disk_total);

  json::Value resp = json::Value::object();
  resp["models_dir"] = models_root.string();
  resp["disk_free_bytes"] = static_cast<double>(disk_free);
  resp["disk_total_bytes"] = static_cast<double>(disk_total);
  resp["jobs"] = std::move(arr);
  return OkJson(std::move(resp));
}

HttpResponse HandleCancelDownload(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return ErrJson(400, "Bad Request", e.what(), "parse_error");
  }
  const std::string id = body.is_object() ? body.member_str("id") : "";
  if (id.empty()) {
    return ErrJson(400, "Bad Request", "'id' is required", "missing_id");
  }

  std::lock_guard<std::mutex> lock(g_dl_mutex);
  ReapDownloadJobsLocked();
  for (auto& j : g_dl_jobs) {
    if (j.id == id) {
      if (j.status == "running" && j.pid > 0) {
        // Negative PID sends signal to the child's process group (setsid)
        ::kill(-j.pid, SIGTERM);
        int st = 0;
        for (int i = 0; i < 10; ++i) {
          if (::waitpid(j.pid, &st, WNOHANG) == j.pid) break;
          ::usleep(100000);
        }
        if (::waitpid(j.pid, &st, WNOHANG) == 0) {
          ::kill(-j.pid, SIGKILL);
          ::waitpid(j.pid, &st, 0);
        }
        j.pid = -1;
        j.status = "cancelled";
        j.finished_epoch = NowEpochSec();
      }
      json::Value resp = json::Value::object();
      resp["id"] = j.id;
      resp["status"] = j.status;
      return OkJson(std::move(resp));
    }
  }
  return ErrJson(404, "Not Found", "unknown download job id: " + id,
                 "job_not_found");
}

// ---- Real-Time GitHub Self-Update Endpoints ----

HttpResponse HandleUpdateCheck(const HttpRequest& req) {
  const bool force = (req.query_param("force") == "1" ||
                      req.query_param("force") == "true");
  const std::string running_sha = mbufo::build::kGitCommit;
  const std::string running_short = mbufo::build::kGitShort;
  const std::string running_tag = mbufo::build::kGitTag;
  const std::string build_time = mbufo::build::kBuildTimeUtc;

  std::lock_guard<std::mutex> lock(g_upd_cache.mu);
  const std::uint64_t now = NowEpochSec();
  if (force || g_upd_cache.fetched_epoch == 0 ||
      now - g_upd_cache.fetched_epoch >= 60) {
    std::vector<std::string> hdrs;
    if (const char* ght = std::getenv("GITHUB_TOKEN");
        ght != nullptr && *ght != '\0') {
      hdrs.push_back("Authorization: Bearer " + std::string(ght));
    }

    long code = 0;
    std::string body;
    std::string err;
    g_upd_cache.check_error.clear();

    if (HttpGet(kGithubCommitsMainUrl, hdrs, 10, &code, &body, &err) &&
        code == 200) {
      try {
        auto c = json::parse(body);
        g_upd_cache.remote_sha = c.member_str("sha");
        g_upd_cache.remote_short =
            g_upd_cache.remote_sha.size() >= 8
                ? g_upd_cache.remote_sha.substr(0, 8)
                : g_upd_cache.remote_sha;
        if (const auto* cm = c.find("commit"); cm != nullptr && cm->is_object()) {
          std::string msg = cm->member_str("message");
          const auto nl = msg.find('\n');
          if (nl != std::string::npos) msg = msg.substr(0, nl);
          g_upd_cache.remote_message = std::move(msg);
          if (const auto* auth = cm->find("author");
              auth != nullptr && auth->is_object()) {
            g_upd_cache.remote_date = auth->member_str("date");
            g_upd_cache.remote_author = auth->member_str("name");
          }
        }
      } catch (const std::exception& e) {
        g_upd_cache.check_error =
            std::string("GitHub commit parse error: ") + e.what();
      }
    } else {
      g_upd_cache.check_error =
          err.empty() ? ("GitHub HTTP " + std::to_string(code)) : err;
    }

    // Fetch latest tag if reachable (best effort)
    long tcode = 0;
    std::string tbody;
    std::string terr;
    if (HttpGet(kGithubTagsUrl, hdrs, 8, &tcode, &tbody, &terr) && tcode == 200) {
      try {
        auto tags = json::parse(tbody);
        if (tags.is_array() && !tags.items().empty() &&
            tags.items()[0].is_object()) {
          g_upd_cache.remote_tag = tags.items()[0].member_str("name");
        }
      } catch (...) {
      }
    }
    g_upd_cache.fetched_epoch = now;
  }

  const bool update_available =
      !g_upd_cache.remote_sha.empty() && running_sha != "unknown" &&
      g_upd_cache.remote_sha != running_sha;

  json::Value resp = json::Value::object();
  resp["running_sha"] = running_sha;
  resp["running_short"] = running_short;
  resp["running_tag"] = running_tag;
  resp["build_time_utc"] = build_time;
  resp["remote_sha"] = g_upd_cache.remote_sha;
  resp["remote_short"] = g_upd_cache.remote_short;
  resp["remote_tag"] = g_upd_cache.remote_tag;
  resp["remote_message"] = g_upd_cache.remote_message;
  resp["remote_date"] = g_upd_cache.remote_date;
  resp["remote_author"] = g_upd_cache.remote_author;
  resp["update_available"] = update_available;
  resp["check_error"] = g_upd_cache.check_error;
  resp["checked_at_utc"] = NowIsoUtc();
  resp["repo_dir"] = ResolveRepoDir().string();
  resp["status"] = ReadPersistentUpdateStatus();
  return OkJson(std::move(resp));
}

HttpResponse HandleUpdateApply(const HttpRequest&) {
  const auto cur = ReadPersistentUpdateStatus();
  if (const auto* act = cur.find("active"); act != nullptr && act->as_bool(false)) {
    return ErrJson(409, "Conflict", "A server self-update is already in progress",
                   "update_in_progress");
  }

  const fs::path repo_dir = ResolveRepoDir();
  const fs::path script = repo_dir / "scripts/mbufo_self_update.sh";
  if (!fs::exists(script)) {
    return ErrJson(500, "Internal Server Error",
                   "Missing updater script: " + script.string(),
                   "missing_updater");
  }

  const fs::path cache_dir = ResolveCacheRoot();
  const fs::path status_file = cache_dir / "update_status.json";
  const fs::path log_file = cache_dir / "update.log";
  const fs::path mapping_cache = cache_dir / "model_mapping.json";

  // Pre-seed status file synchronously so immediate UI polls see active=true
  json::Value init_st = json::Value::object();
  init_st["phase"] = "fetching";
  init_st["active"] = true;
  init_st["running_before"] = mbufo::build::kGitCommit;
  init_st["target_sha"] = "";
  init_st["message"] = "Launching self-update worker...";
  init_st["error"] = "";
  init_st["started_at"] = NowIsoUtc();
  init_st["updated_at"] = NowIsoUtc();
  WriteFileAtomic(status_file, init_st.dump());

  const std::string pid_str = std::to_string(static_cast<long long>(::getpid()));
  const std::vector<std::string> args = {
      "/bin/bash",
      script.string(),
      pid_str,
      repo_dir.string(),
      status_file.string(),
      log_file.string(),
      mapping_cache.string(),
  };

  pid_t child = -1;
  std::string err;
  CloseInheritedFdsAndExec(args, log_file.string(), "", &child, &err);
  if (child < 0) {
    init_st["phase"] = "failed";
    init_st["active"] = false;
    init_st["error"] = err;
    WriteFileAtomic(status_file, init_st.dump());
    return ErrJson(500, "Internal Server Error",
                   "Failed launching self-update worker: " + err,
                   "updater_launch_failed");
  }

  json::Value resp = json::Value::object();
  resp["started"] = true;
  resp["worker_pid"] = static_cast<long long>(child);
  resp["running_sha"] = mbufo::build::kGitCommit;
  resp["status_file"] = status_file.string();
  resp["log_file"] = log_file.string();
  return OkJson(std::move(resp));
}

HttpResponse HandleUpdateStatus(const HttpRequest&) {
  const fs::path cache_dir = ResolveCacheRoot();
  json::Value resp = ReadPersistentUpdateStatus();
  resp["running_sha"] = mbufo::build::kGitCommit;
  resp["running_short"] = mbufo::build::kGitShort;
  resp["running_tag"] = mbufo::build::kGitTag;
  resp["build_time_utc"] = mbufo::build::kBuildTimeUtc;
  resp["log_tail"] = ReadTailLines(cache_dir / "update.log", 25);
  return OkJson(std::move(resp));
}

}  // namespace mbufo::server::admin_ops
