// LoadRerankHead: binds + upcasts the classification head (see weights.hpp).
#include "src/models/bge_reranker/weights.hpp"

#include <string>
#include <string_view>

namespace gufo::models::bge_reranker {
namespace {

[[nodiscard]] bool IsStorageType(core::GgmlType type) noexcept {
  // Encoder-base precision policy: exact F32/F16/BF16 storage only.
  return type == core::GgmlType::kF32 || type == core::GgmlType::kF16 ||
         type == core::GgmlType::kBF16;
}

// Q4 leg: the classifier matrices accept the shared prefill quant set
// (Q4_K_M stores cls.weight Q4_K); 1D biases stay dense (always F32).
[[nodiscard]] bool IsHeadWeightType(core::GgmlType type) noexcept {
  return IsStorageType(type) || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
         type == core::GgmlType::kQ8_0;
}

[[nodiscard]] bool UpcastToF32(const bge_m3::BgeM3TensorRef& tensor,
                               std::vector<float>& out, std::size_t row_elems,
                               std::string_view name, bool allow_quant,
                               std::string* error_msg) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing required bge-reranker tensor: " + std::string(name);
    }
    return false;
  }
  const bool ok_type =
      allow_quant ? IsHeadWeightType(tensor.type) : IsStorageType(tensor.type);
  if (!ok_type) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported bge-reranker tensor type for " +
                   std::string(name);
    }
    return false;
  }
  return bge_m3::UpcastTensorToF32(tensor, out, row_elems, error_msg);
}

bge_m3::BgeM3TensorRef ExtractTensorRef(const core::GgufReader& reader,
                                        std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount()};
}

}  // namespace

std::optional<BgeRerankerHeadF32> LoadRerankHead(const core::GgufReader& reader,
                                                 std::uint32_t hidden_size,
                                                 std::string* error_msg) {
  auto refuse =
      [&](const std::string& message) -> std::optional<BgeRerankerHeadF32> {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::nullopt;
  };
  if (hidden_size == 0) {
    return refuse("bge-reranker head needs a non-zero hidden size");
  }
  const bge_m3::BgeM3TensorRef dense_w = ExtractTensorRef(reader, "cls.weight");
  const bge_m3::BgeM3TensorRef dense_b = ExtractTensorRef(reader, "cls.bias");
  const bge_m3::BgeM3TensorRef out_w =
      ExtractTensorRef(reader, "cls.output.weight");
  const bge_m3::BgeM3TensorRef out_b =
      ExtractTensorRef(reader, "cls.output.bias");
  const std::size_t hidden = hidden_size;
  if (dense_w.num_elements != hidden * hidden) {
    return refuse("cls.weight shape mismatch: expected [" +
                  std::to_string(hidden) + ", " + std::to_string(hidden) +
                  "] elements");
  }
  if (dense_b.num_elements != hidden) {
    return refuse("cls.bias shape mismatch: expected " +
                  std::to_string(hidden) + " elements");
  }
  if (out_w.num_elements != hidden) {
    return refuse("cls.output.weight shape mismatch: expected " +
                  std::to_string(hidden) + " elements");
  }
  if (out_b.num_elements != 1) {
    return refuse("cls.output.bias shape mismatch: expected 1 element");
  }
  BgeRerankerHeadF32 head;
  head.hidden_size = hidden_size;
  if (!UpcastToF32(dense_w, head.dense_w, hidden, "cls.weight", true,
                   error_msg) ||
      !UpcastToF32(dense_b, head.dense_b, dense_b.num_elements, "cls.bias",
                   false, error_msg) ||
      !UpcastToF32(out_w, head.out_w, hidden, "cls.output.weight", true,
                   error_msg)) {
    return std::nullopt;
  }
  if (!IsStorageType(out_b.type)) {
    return refuse("cls.output.bias needs F32/F16/BF16 storage");
  }
  head.out_b = out_b.GetFloat(0);
  return head;
}

}  // namespace gufo::models::bge_reranker
