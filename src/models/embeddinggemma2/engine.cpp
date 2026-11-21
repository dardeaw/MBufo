// EmbeddingGemma 2: GGUF metadata + tokenizer loading. The device forward is
// in hip/session.hip.
#include "src/models/embeddinggemma2/engine.hpp"

#include "src/core/compute/npu/npu_scheduler.hpp"
#ifdef MBUFO_ENABLE_NPU
#include "src/core/compute/npu/npu_embgemma2_runtime.hpp"
#endif

#include <cstdlib>
#include <utility>

namespace mbufo::models::embeddinggemma2 {

#ifdef MBUFO_ENABLE_NPU
struct EngineModel::NpuState {
  mbufo::core::npu::NpuEmbGemma2Runtime runtime;
};
#else
struct EngineModel::NpuState {};
#endif

bool EngineModel::LoadNpuWeights(const std::string& kernel_dir,
                                 std::string* error_msg) {
#ifdef MBUFO_ENABLE_NPU
  std::string dir = kernel_dir;
  if (dir.empty()) {
    if (const char* env = std::getenv("MBUFO_NPU_EMBGEMMA2_KERNELS");
        env != nullptr && env[0] != '\0') {
      dir = env;
    } else {
      dir = "/mnt/nvme-data/mbufo-models/npu-kernels/embgemma2_s128";
    }
  }
  auto state = std::make_shared<NpuState>();
  if (!state->runtime.Init(*reader_, config_, dir, error_msg)) {
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
    *error_msg = "embeddinggemma2 NPU runtime disabled (MBUFO_ENABLE_NPU=OFF)";
  }
  return false;
#endif
}

bool EngineModel::TryNpuEmbed(std::span<const std::uint32_t> ids,
                              std::optional<EmbedResult>* out,
                              std::string* error_msg) {
  const bool npu_ready =
      npu_scheduler_ != nullptr && !npu_scheduler_->device_lost();
  if (npu::DecideNpuEmbed(npu_config_, ids.size(), npu_weights_resident_,
                          npu_ready) != npu::EmbedNpuRoute::kIron ||
      npu_scheduler_ == nullptr || !npu_weights_resident_ || npu_ == nullptr) {
    return false;
  }
#ifdef MBUFO_ENABLE_NPU
  if (ids.size() > mbufo::core::npu::NpuEmbGemma2Runtime::kFixedSeqLen) {
    return false;
  }
  std::vector<float> values;
  double prefill_ms = 0.0;
  std::string run_err;
  const auto sched_res = mbufo::core::npu::RunLaneSync(
      *npu_scheduler_,
      mbufo::core::npu::NpuWorkDesc{
          .lane = mbufo::core::npu::NpuWorkLane::kEmbedding,
          .seq_len = ids.size(),
          .client = "embeddinggemma2",
          .coalesce_key = "embgemma2-s128"},
      [&]() {
        mbufo::core::npu::NpuWorkResult wr;
        wr.ok = npu_->runtime.Forward(ids, &values, &prefill_ms, &run_err);
        wr.error = wr.ok ? "ok" : run_err;
        return wr;
      });
  if (!sched_res.ok) {
    if (error_msg != nullptr) {
      *error_msg = "embeddinggemma2 NPU Embed: " + sched_res.error;
    }
    *out = std::nullopt;
    return true;
  }
  EmbedResult res;
  res.values = std::move(values);
  res.prefill_ms = prefill_ms;
  *out = std::move(res);
  return true;
#else
  (void)out;
  (void)error_msg;
  return false;
#endif
}

#ifndef ENGINE_ENABLE_HIP
struct EngineModel::HipState {};
EngineModel::EngineModel() = default;
EngineModel::~EngineModel() = default;
std::optional<EmbedResult> EngineModel::Embed(
    std::span<const std::uint32_t> ids, std::string* error_msg) {
  std::optional<EmbedResult> npu_out;
  if (TryNpuEmbed(ids, &npu_out, error_msg)) {
    return npu_out;
  }
  if (error_msg != nullptr)
    *error_msg = "EmbeddingGemma 2 requires the HIP backend";
  return std::nullopt;
}
#endif

std::unique_ptr<EngineModel> EngineModel::Create(const std::string& model_path,
                                                 std::string* error_msg) {
  auto fail = [&](const std::string& message) {
    if (error_msg != nullptr) *error_msg = message;
    return std::unique_ptr<EngineModel>{};
  };
  auto reader = core::GgufReader::OpenFile(model_path, error_msg);
  if (reader == nullptr) return nullptr;
  if (reader->GetMetadataString("general.architecture").value_or("") !=
      "embedding_gemma2") {
    return fail("not an embedding_gemma2 GGUF");
  }
  constexpr const char* p = "embedding_gemma2.";
  auto u32 = [&](const std::string& key) -> std::uint32_t {
    return reader->GetMetadataUint32(p + key).value_or(0);
  };
  auto f32 = [&](const std::string& key, float fallback) -> float {
    return reader->GetMetadataFloat32(p + key).value_or(fallback);
  };
  Config c;
  c.layers = u32("block_count");
  c.hidden = u32("embedding_length");
  c.ffn = u32("feed_forward_length");
  c.heads = u32("attention.head_count");
  c.head_dim_local = u32("attention.head_dim_local");
  c.head_dim_full = u32("attention.head_dim_full");
  c.kv_local = u32("attention.head_count_kv_local");
  c.kv_full = u32("attention.head_count_kv_full");
  c.sliding_window = u32("attention.sliding_window");
  c.ple_dim = u32("per_layer_input_dim");
  c.out_dim = u32("output_embedding_length");
  c.rms_eps = f32("attention.layer_norm_rms_epsilon", 1e-6F);
  c.rope_local = f32("rope.freq_base_local", 10000.0F);
  c.rope_full = f32("rope.freq_base_full", 1000000.0F);
  c.embed_scale = f32("embed_scale", 1.0F);
  const std::uint32_t period = u32("attention.full_layer_period");
  if (c.layers == 0 || c.hidden == 0 || c.heads == 0 || c.out_dim == 0 ||
      c.head_dim_local == 0 || c.head_dim_full == 0 || c.kv_local == 0 ||
      c.kv_full == 0 || period == 0 || c.heads % c.kv_local != 0 ||
      c.heads % c.kv_full != 0) {
    return fail("embedding_gemma2 metadata is incomplete");
  }
  c.layer_is_full.resize(c.layers);
  for (std::uint32_t i = 0; i < c.layers; ++i) {
    c.layer_is_full[i] = ((i + 1) % period == 0) ? 1 : 0;
  }
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (tokenizer == nullptr) return nullptr;
  auto model = std::unique_ptr<EngineModel>(new EngineModel());
  model->reader_ = std::move(reader);
  model->tokenizer_ = std::move(tokenizer);
  model->config_ = std::move(c);
  return model;
}

std::vector<std::uint32_t> EngineModel::Encode(const std::string& text) const {
  tokenization::TokenizerOptions options;
  options.add_bos = true;
  options.add_eos = true;
  options.parse_special_tokens = false;
  return tokenizer_->Encode(text, options);
}

}  // namespace mbufo::models::embeddinggemma2
