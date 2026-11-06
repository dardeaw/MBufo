// BgeM3Weights::LoadFromGguf: binds the 5 embedding tensors plus 16 tensors
// per encoder layer from a `bert` GGUF artifact (see weights.hpp).
#include "src/models/bge_m3/weights.hpp"

#include <string>

namespace gufo::models::bge_m3 {
namespace {

BgeM3TensorRef ExtractTensorRef(const core::GgufReader& reader,
                                std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount()};
}

[[nodiscard]] constexpr bool IsStorageType(core::GgmlType type) noexcept {
  // Precision-first leg-1: exact F32/F16/BF16 storage only. Quantized types
  // are refused explicitly by ValidateTensor below; quantization is a later
  // leg, never a silent fallback.
  return type == core::GgmlType::kF32 || type == core::GgmlType::kF16 ||
         type == core::GgmlType::kBF16;
}

bool ValidateTensor(const BgeM3TensorRef& tensor, std::size_t expected_elements,
                    std::string_view name, std::string* error_msg) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing required bge-m3 tensor: " + std::string(name);
    }
    return false;
  }
  if (!IsStorageType(tensor.type)) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported bge-m3 tensor type for " + std::string(name) +
                   ": leg-1 accepts F32/F16/BF16 only";
    }
    return false;
  }
  if (tensor.num_elements != expected_elements) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor shape mismatch for " + std::string(name) +
                   ": expected " + std::to_string(expected_elements) +
                   " elements, found " + std::to_string(tensor.num_elements);
    }
    return false;
  }
  return true;
}

}  // namespace

std::optional<BgeM3Weights> BgeM3Weights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg,
    BgeM3ConfigOptions options) {
  auto config_opt = ExtractBgeM3Config(reader, error_msg, options);
  if (!config_opt.has_value()) {
    return std::nullopt;
  }
  BgeM3Weights weights;
  weights.config = *config_opt;
  const std::size_t hidden = weights.config.hidden_size;
  const std::size_t inter = weights.config.intermediate_size;
  const std::size_t vocab = weights.config.vocab_size;

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.position_embd = ExtractTensorRef(reader, "position_embd.weight");
  weights.token_types = ExtractTensorRef(reader, "token_types.weight");
  weights.token_embd_norm_weight =
      ExtractTensorRef(reader, "token_embd_norm.weight");
  weights.token_embd_norm_bias =
      ExtractTensorRef(reader, "token_embd_norm.bias");
  if (!ValidateTensor(weights.token_embd, vocab * hidden, "token_embd.weight",
                      error_msg) ||
      !ValidateTensor(weights.token_embd_norm_weight, hidden,
                      "token_embd_norm.weight", error_msg) ||
      !ValidateTensor(weights.token_embd_norm_bias, hidden,
                      "token_embd_norm.bias", error_msg)) {
    return std::nullopt;
  }
  // Position table must cover the whole leg-1 window; a longer trained table
  // is fine (AddPositional ignores extra rows). Token-type rows: type_vocab
  // is 1 for bge-m3, but any table with at least the row-0 fallback passes.
  if (weights.position_embd.empty() ||
      (weights.position_embd.num_elements % hidden) != 0 ||
      weights.position_embd.num_elements < kBgeM3Leg1MaxTokens * hidden ||
      !IsStorageType(weights.position_embd.type)) {
    if (error_msg != nullptr) {
      *error_msg =
          "position_embd.weight must cover the leg-1 window as F32/F16/BF16";
    }
    return std::nullopt;
  }
  if (weights.token_types.empty() ||
      (weights.token_types.num_elements % hidden) != 0 ||
      !IsStorageType(weights.token_types.type)) {
    if (error_msg != nullptr) {
      *error_msg = "token_types.weight must be a [n_types, hidden] table";
    }
    return std::nullopt;
  }

  weights.layers.resize(weights.config.num_layers);
  for (std::uint32_t i = 0; i < weights.config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& l = weights.layers[i];
    l.attn_q_weight = ExtractTensorRef(reader, prefix + "attn_q.weight");
    l.attn_q_bias = ExtractTensorRef(reader, prefix + "attn_q.bias");
    l.attn_k_weight = ExtractTensorRef(reader, prefix + "attn_k.weight");
    l.attn_k_bias = ExtractTensorRef(reader, prefix + "attn_k.bias");
    l.attn_v_weight = ExtractTensorRef(reader, prefix + "attn_v.weight");
    l.attn_v_bias = ExtractTensorRef(reader, prefix + "attn_v.bias");
    l.attn_output_weight =
        ExtractTensorRef(reader, prefix + "attn_output.weight");
    l.attn_output_bias = ExtractTensorRef(reader, prefix + "attn_output.bias");
    l.attn_output_norm_weight =
        ExtractTensorRef(reader, prefix + "attn_output_norm.weight");
    l.attn_output_norm_bias =
        ExtractTensorRef(reader, prefix + "attn_output_norm.bias");
    l.ffn_up_weight = ExtractTensorRef(reader, prefix + "ffn_up.weight");
    l.ffn_up_bias = ExtractTensorRef(reader, prefix + "ffn_up.bias");
    l.ffn_down_weight = ExtractTensorRef(reader, prefix + "ffn_down.weight");
    l.ffn_down_bias = ExtractTensorRef(reader, prefix + "ffn_down.bias");
    l.layer_output_norm_weight =
        ExtractTensorRef(reader, prefix + "layer_output_norm.weight");
    l.layer_output_norm_bias =
        ExtractTensorRef(reader, prefix + "layer_output_norm.bias");
    if (!ValidateTensor(l.attn_q_weight, hidden * hidden,
                        prefix + "attn_q.weight", error_msg) ||
        !ValidateTensor(l.attn_q_bias, hidden, prefix + "attn_q.bias",
                        error_msg) ||
        !ValidateTensor(l.attn_k_weight, hidden * hidden,
                        prefix + "attn_k.weight", error_msg) ||
        !ValidateTensor(l.attn_k_bias, hidden, prefix + "attn_k.bias",
                        error_msg) ||
        !ValidateTensor(l.attn_v_weight, hidden * hidden,
                        prefix + "attn_v.weight", error_msg) ||
        !ValidateTensor(l.attn_v_bias, hidden, prefix + "attn_v.bias",
                        error_msg) ||
        !ValidateTensor(l.attn_output_weight, hidden * hidden,
                        prefix + "attn_output.weight", error_msg) ||
        !ValidateTensor(l.attn_output_bias, hidden,
                        prefix + "attn_output.bias", error_msg) ||
        !ValidateTensor(l.attn_output_norm_weight, hidden,
                        prefix + "attn_output_norm.weight", error_msg) ||
        !ValidateTensor(l.attn_output_norm_bias, hidden,
                        prefix + "attn_output_norm.bias", error_msg) ||
        !ValidateTensor(l.ffn_up_weight, inter * hidden,
                        prefix + "ffn_up.weight", error_msg) ||
        !ValidateTensor(l.ffn_up_bias, inter, prefix + "ffn_up.bias",
                        error_msg) ||
        !ValidateTensor(l.ffn_down_weight, hidden * inter,
                        prefix + "ffn_down.weight", error_msg) ||
        !ValidateTensor(l.ffn_down_bias, hidden, prefix + "ffn_down.bias",
                        error_msg) ||
        !ValidateTensor(l.layer_output_norm_weight, hidden,
                        prefix + "layer_output_norm.weight", error_msg) ||
        !ValidateTensor(l.layer_output_norm_bias, hidden,
                        prefix + "layer_output_norm.bias", error_msg)) {
      return std::nullopt;
    }
  }
  return weights;
}

}  // namespace gufo::models::bge_m3
