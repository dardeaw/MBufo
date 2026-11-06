// EngineModel / EngineSession / ModelRegistry implementation.
#include "src/models/bge_m3/engine.hpp"

#include <chrono>
#include <map>
#include <mutex>

#include "src/core/pooling_norm.hpp"

namespace gufo::models::bge_m3 {

#ifndef ENGINE_ENABLE_HIP
// Out-of-line so the destructor is defined exactly once per build mode;
// the HIP TU (hip/session.hip) provides it when ENGINE_ENABLE_HIP.
EngineModel::~EngineModel() = default;
EngineModel::EngineModel() = default;
#endif

std::unique_ptr<EngineModel> EngineModel::Create(const std::string& model_path,
                                                 std::string* error_msg) {
  auto reader = core::GgufReader::OpenFile(model_path, error_msg);
  if (reader == nullptr) {
    return nullptr;
  }
  auto weights = BgeM3Weights::LoadFromGguf(*reader, error_msg);
  if (!weights.has_value()) {
    return nullptr;
  }
  auto tokenizer = SpmUnigramTokenizer::CreateFromGguf(*reader, error_msg);
  if (tokenizer == nullptr) {
    return nullptr;
  }
  auto model = std::unique_ptr<EngineModel>(new EngineModel());
  model->reader_ = std::move(reader);
  model->weights_ = std::move(*weights);
  model->tokenizer_ = std::move(tokenizer);
  model->f32_ = UpcastAll(model->weights_);
  return model;
}

EngineSession::EngineSession(const EngineModel* model)
    : model_(model),
      scratch_(BgeM3Scratch::Create(model->Config(), kBgeM3Leg1MaxTokens)) {}

EngineSession::~EngineSession() = default;

std::optional<EngineSession::SyncResult> EngineSession::Sync(
    std::span<const std::uint32_t> ids, bool use_hip,
    std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr) *error_msg = message;
    return std::optional<SyncResult>{};
  };
  if (ids.empty()) {
    return refuse("bge-m3 Sync: empty id sequence");
  }
  if (ids.size() > kBgeM3Leg1MaxTokens) {
    return refuse("bge-m3 Sync: " + std::to_string(ids.size()) +
                  " tokens exceeds the leg-1 window");
  }
#ifdef ENGINE_ENABLE_HIP
  if (use_hip) {
    // Device prefill lives in hip/session.hip (HipSync); failure falls
    // through to an explicit refusal, never silently to CPU, so timings and
    // numerics stay attributable to the path that produced them.
    if (auto device = HipSync(*model_, ids, error_msg); device.has_value()) {
      return device;
    }
    return std::nullopt;
  }
#else
  (void)use_hip;
#endif
  const std::size_t dim = model_->Config().hidden_size;
  std::vector<float> hidden(ids.size() * dim);
  const auto start = std::chrono::steady_clock::now();
  if (!CpuForward(model_->f32_, ids, scratch_, hidden)) {
    return refuse("bge-m3 Sync: CPU forward shape/id failure");
  }
  const double prefill_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();
  SyncResult result;
  result.values.assign(dim, 0.0F);
  if (!core::embedding::ApplyPooling(hidden, ids.size(), dim,
                                     core::embedding::PoolingStrategy::kCls,
                                     result.values)) {
    return refuse("bge-m3 Sync: CLS pooling shape failure");
  }
  result.pre_norm = core::embedding::L2Normalize(result.values);
  if (!(result.pre_norm > 0.0)) {
    return refuse("bge-m3 Sync: zero-norm embedding refused");
  }
  result.prefill_ms = prefill_ms;
  return result;
}

ModelRegistry& ModelRegistry::Instance() {
  static ModelRegistry instance;
  return instance;
}

namespace {
// "bge-m3" (the XLM-R encoder kind) self-registers at load time.
const bool kBgeM3Registered = ModelRegistry::Instance().Register(
    "bge-m3",
    [](const std::string& model_path, std::string* error_msg) {
      return EngineModel::Create(model_path, error_msg);
    });
}  // namespace

bool ModelRegistry::Register(const std::string& name, Factory factory) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (factories_.find(name) != factories_.end()) {
    return false;
  }
  factories_.emplace(name, std::move(factory));
  return true;
}

std::unique_ptr<EngineModel> ModelRegistry::Create(
    const std::string& name, const std::string& model_path,
    std::string* error_msg) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = factories_.find(name);
  if (it == factories_.end()) {
    if (error_msg != nullptr) {
      *error_msg = "unknown bge model kind: " + name;
    }
    return nullptr;
  }
  return it->second(model_path, error_msg);
}

}  // namespace gufo::models::bge_m3
