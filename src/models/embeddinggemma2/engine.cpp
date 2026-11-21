// EmbeddingGemma 2: GGUF metadata + tokenizer loading. The device forward is
// in hip/session.hip.
#include "src/models/embeddinggemma2/engine.hpp"

#include <utility>

namespace mbufo::models::embeddinggemma2 {

#ifndef ENGINE_ENABLE_HIP
struct EngineModel::HipState {};
EngineModel::EngineModel() = default;
EngineModel::~EngineModel() = default;
std::optional<EmbedResult> EngineModel::Embed(
    std::span<const std::uint32_t> ids, std::string* error_msg) {
  if (npu::DecideNpuEmbed(npu_config_, ids.size()) ==
      npu::EmbedNpuRoute::kIron) {
    // Future NPU IRON prefill dispatch seam (zero-change while kNone).
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
