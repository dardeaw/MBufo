// EngineModel / EngineSession / ModelRegistry implementation.
#include "src/models/bge_m3/engine.hpp"

#include <chrono>
#include <map>
#include <mutex>

#include "src/core/compute/npu/npu_scheduler.hpp"
#ifdef MBUFO_ENABLE_NPU
#include "src/core/compute/npu/npu_bge_runtime.hpp"
#include "src/core/compute/npu/npu_graph_models.hpp"
#endif
#include "src/core/pooling_norm.hpp"

#include <cstdlib>

namespace mbufo::models::bge_m3 {

#ifdef MBUFO_ENABLE_NPU
struct EngineModel::NpuState {
  // MBUFO_NPU_BGE_RUNTIME=graph selects the declarative op-graph executor on
  // the shared Q8 engine; default is the dedicated BF16 gemm_1024x1024 runtime.
  bool use_graph{false};
  mbufo::core::npu::NpuBgeRuntime runtime;
  mbufo::core::npu::NpuGraphBgeRuntime graph_runtime;
};
#else
struct EngineModel::NpuState {};
#endif

bool EngineModel::LoadNpuWeights(const std::string& kernel_dir,
                                 std::string* error_msg) {
#ifdef MBUFO_ENABLE_NPU
  std::string dir = kernel_dir;
  if (dir.empty()) {
    if (const char* env = std::getenv("MBUFO_NPU_BGE_KERNELS");
        env != nullptr && env[0] != '\0') {
      dir = env;
    } else {
      dir = "/mnt/nvme-data/mbufo-models/npu-kernels/bge_m64";
    }
  }
  auto state = std::make_shared<NpuState>();
  const char* rt = std::getenv("MBUFO_NPU_BGE_RUNTIME");
  state->use_graph = rt != nullptr && std::string(rt) == "graph";
  const bool init_ok =
      state->use_graph
          ? state->graph_runtime.Init(
                f32_, mbufo::core::npu::graph::LinearBackend::kNpuQ8,
                error_msg)
          : state->runtime.Init(f32_, dir, error_msg);
  if (!init_ok) {
    npu_.reset();
    npu_weights_resident_ = false;
    return false;
  }
  npu_ = std::move(state);
  npu_weights_resident_ = true;
  return true;
#else
  (void)kernel_dir;
  if (error_msg != nullptr) {
    *error_msg = "bge-m3 NPU runtime disabled (MBUFO_ENABLE_NPU=OFF)";
  }
  return false;
#endif
}

bool EngineModel::RunNpuCls(std::span<const std::uint32_t> ids,
                            std::vector<float>* cls_out, double* prefill_ms,
                            std::string* error_msg) const {
#ifdef MBUFO_ENABLE_NPU
  if (!npu_weights_resident_ || npu_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "bge-m3 NPU weights not loaded";
    return false;
  }
  if (npu_->use_graph) {
    return npu_->graph_runtime.ForwardCls(ids, cls_out, prefill_ms, error_msg);
  }
  return npu_->runtime.ForwardCls(ids, f32_, cls_out, prefill_ms, error_msg);
#else
  (void)ids;
  (void)cls_out;
  (void)prefill_ms;
  if (error_msg != nullptr) {
    *error_msg = "bge-m3 NPU runtime disabled (MBUFO_ENABLE_NPU=OFF)";
  }
  return false;
#endif
}

#ifndef ENGINE_ENABLE_HIP
// Out-of-line so the destructor is defined exactly once per build mode;
// the HIP TU (hip/session.hip) provides it when ENGINE_ENABLE_HIP.
EngineModel::~EngineModel() = default;
EngineModel::EngineModel() = default;
#endif

std::unique_ptr<EngineModel> EngineModel::Create(
    const std::string& model_path, std::string* error_msg,
    BgeM3ConfigOptions options) {
  auto reader = core::GgufReader::OpenFile(model_path, error_msg);
  if (reader == nullptr) {
    return nullptr;
  }
  auto weights =
      BgeM3Weights::LoadFromGguf(*reader, error_msg, options);
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
  if (ids.size() > (use_hip ? kBgeM3HipMaxTokens : kBgeM3Leg1MaxTokens)) {
    return refuse("bge-m3 Sync: " + std::to_string(ids.size()) +
                  " tokens exceeds the window");
  }
  const bool npu_ready = npu_scheduler_ != nullptr && !npu_scheduler_->device_lost();
  if (npu::DecideNpuEmbed(npu_config_, ids.size(), model_->npu_weights_resident(),
                          npu_ready) == npu::EmbedNpuRoute::kIron &&
      npu_scheduler_ != nullptr) {
    std::vector<float> cls;
    double prefill_ms = 0.0;
    std::string run_err;
    const auto sched_res = mbufo::core::npu::RunLaneSync(
        *npu_scheduler_,
        mbufo::core::npu::NpuWorkDesc{
            .lane = mbufo::core::npu::NpuWorkLane::kEmbedding,
            .seq_len = ids.size(),
            .client = "bge-m3",
            .coalesce_key = "bge-m3"},
        [&]() {
          mbufo::core::npu::NpuWorkResult wr;
          wr.ok = model_->RunNpuCls(ids, &cls, &prefill_ms, &run_err);
          wr.error = wr.ok ? "ok" : run_err;
          return wr;
        });
    if (!sched_res.ok) {
      return refuse("bge-m3 NPU Sync: " + sched_res.error);
    }
    SyncResult result;
    result.values = std::move(cls);
    result.pre_norm = core::embedding::L2Normalize(result.values);
    if (!(result.pre_norm > 0.0)) {
      return refuse("bge-m3 NPU Sync: zero-norm embedding refused");
    }
    result.prefill_ms = prefill_ms;
    return result;
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

}  // namespace mbufo::models::bge_m3
