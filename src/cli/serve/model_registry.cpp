#include "src/cli/serve/model_registry.hpp"

#include <algorithm>
#include <chrono>

namespace gufo::server {
namespace {

std::uint64_t NowMs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace

bool ModelRegistry::Add(const std::string& primary,
                        std::vector<std::string> extra_names,
                        std::shared_ptr<TextGenerationBackend> backend,
                        std::string* error) {
  if (backend == nullptr) {
    if (error != nullptr) *error = "backend is null";
    return false;
  }
  if (primary.empty()) {
    if (error != nullptr) *error = "model id is empty";
    return false;
  }
  std::vector<std::string> names{primary};
  for (auto& name : extra_names)
    if (!name.empty() && name != primary) names.push_back(std::move(name));

  std::lock_guard lock(mutex_);
  for (const auto& name : names) {
    if (by_name_.count(name) != 0) {
      if (error != nullptr) *error = "model '" + name + "' is already loaded";
      return false;
    }
  }
  auto entry = std::make_shared<Entry>();
  entry->id = primary;
  entry->names = std::move(names);
  entry->backend = std::move(backend);
  entry->last_used.store(NowMs(), std::memory_order_relaxed);
  for (const auto& name : entry->names) by_name_[name] = entry;
  order_.push_back(std::move(entry));
  return true;
}

bool ModelRegistry::Remove(const std::string& id, std::string* error) {
  std::lock_guard lock(mutex_);
  for (std::size_t i = 0; i < order_.size(); ++i) {
    if (order_[i]->id != id) continue;
    for (const auto& name : order_[i]->names) by_name_.erase(name);
    order_.erase(order_.begin() + static_cast<std::ptrdiff_t>(i));
    return true;
  }
  if (error != nullptr) *error = "model '" + id + "' is not loaded";
  return false;
}

std::shared_ptr<TextGenerationBackend> ModelRegistry::Resolve(
    std::string_view name) {
  std::shared_ptr<Entry> entry;
  {
    std::lock_guard lock(mutex_);
    if (name.empty()) {
      for (const auto& candidate : order_)
        if (candidate->backend && candidate->backend->ready()) {
          entry = candidate;
          break;
        }
    } else {
      auto it = by_name_.find(std::string(name));
      if (it != by_name_.end()) entry = it->second.lock();
    }
    if (entry == nullptr) return nullptr;
    entry->last_used.store(NowMs(), std::memory_order_relaxed);
  }
  return entry->backend;
}

std::shared_ptr<TextGenerationBackend> ModelRegistry::Default() {
  return Resolve({});
}

std::vector<ModelRegistry::Info> ModelRegistry::List() const {
  std::lock_guard lock(mutex_);
  std::vector<Info> out;
  out.reserve(order_.size());
  for (const auto& entry : order_) {
    Info info;
    info.id = entry->id;
    info.names = entry->names;
    info.max_context = entry->backend ? entry->backend->max_context() : 0;
    info.last_used_ms = entry->last_used.load(std::memory_order_relaxed);
    info.ready = entry->backend != nullptr && entry->backend->ready();
    if (entry->backend) info.path = entry->backend->model_path();
    out.push_back(std::move(info));
  }
  std::sort(out.begin(), out.end(),
            [](const Info& a, const Info& b) { return a.id < b.id; });
  return out;
}

bool ModelRegistry::HasId(std::string_view name) const {
  std::lock_guard lock(mutex_);
  for (const auto& entry : order_)
    if (entry->id == name) return true;
  return false;
}

bool ModelRegistry::HasName(std::string_view name) const {
  std::lock_guard lock(mutex_);
  return by_name_.count(std::string(name)) != 0;
}

std::size_t ModelRegistry::Size() const {
  std::lock_guard lock(mutex_);
  return order_.size();
}

}  // namespace gufo::server
