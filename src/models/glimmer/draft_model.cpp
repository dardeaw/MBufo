// DFlash draft artifact binding: `dflash` GGUF (depth-pruned dense decoder).
// Q4_K/Q6_K/Q8_0 weights dequantize to F32 once at load (the draft is tiny;
// F32 keeps the draft GEMM path uniform via fn::SmallGemm).
#include "src/models/glimmer/draft_model.hpp"

#include <cstring>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::glimmer::draft {
namespace {

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key) {
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + std::string(key));
      return 0;
    }
    return *value;
  }

  std::uint32_t U32Opt(std::string_view key, std::uint32_t fallback) {
    return gguf.GetMetadataUint32(key).value_or(fallback);
  }
};

}  // namespace

std::optional<DraftConfig> DraftConfig::FromGguf(
    const core::GgufReader& gguf, std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != "dflash") {
    r.Fail("GGUF architecture is not dflash");
    return std::nullopt;
  }
  DraftConfig c;
  c.num_layers = r.U32("dflash.block_count");
  // Geometry from the artifact (fall back to the Glimmer family defaults).
  c.hidden_size = r.U32Opt("dflash.embedding_length", 6656);
  c.intermediate_size = r.U32Opt("dflash.feed_forward_length", 19968);
  c.num_heads = r.U32Opt("dflash.attention.head_count", 32);
  c.num_kv_heads = r.U32Opt("dflash.attention.head_count_kv", 8);
  c.head_dim = r.U32Opt("dflash.attention.key_length", 128);
  if (c.head_dim == 0)
    c.head_dim = r.U32Opt("dflash.attention.value_length", 128);
  c.context_length = r.U32Opt("dflash.context_length", 131072);
  if (const auto theta = gguf.GetMetadataFloat32("dflash.rope.freq_base")) {
    c.rope_theta = *theta;
  }
  if (const auto eps =
          gguf.GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon")) {
    c.rms_eps = *eps;
  }
  c.sliding_window = r.U32Opt("dflash.attention.sliding_window", 2048);
  // target layers: `dflash.target_layers` (also accept `target_layer_ids`).
  c.target_layer_ids.clear();
  {
    const auto* meta = gguf.FindMetadata("dflash.target_layers");
    if (meta == nullptr)
      meta = gguf.FindMetadata("dflash.target_layer_ids");
    if (meta != nullptr) {
      if (const auto* u =
              std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
        for (auto v : *u)
          c.target_layer_ids.push_back(static_cast<std::uint32_t>(v));
      } else if (const auto* s =
                     std::get_if<std::vector<std::int64_t>>(&meta->value)) {
        for (auto v : *s)
          c.target_layer_ids.push_back(static_cast<std::uint32_t>(v));
      }
    }
  }
  const auto* embd = gguf.FindTensor("token_embd.weight");
  if (embd == nullptr || embd->dimensions.size() < 2) {
    r.Fail("dflash artifact is missing token_embd.weight");
    return std::nullopt;
  }
  c.vocab_size = static_cast<std::uint32_t>(embd->dimensions[1]);
  if (c.num_layers == 0 || c.vocab_size == 0 ||
      c.target_layer_ids.size() != c.num_layers) {
    r.Fail("dflash artifact has inconsistent dimensions");
    return std::nullopt;
  }
  if (!r.ok)
    return std::nullopt;
  return c;
}

namespace {

// Dequantize one GGUF tensor to host F32 (F32/BF16 pass through, Q4_K/Q6_K
// and Q8_0 ride the shared core dequantizers).
bool DequantToF32(const core::GgufTensorInfo& info, std::vector<float>& out,
                  std::string* error) {
  namespace quant = gufo::quant;
  const std::uint64_t elems = info.ElementCount();
  if (elems == 0) {
    if (error != nullptr && error->empty())
      *error = "dflash tensor is empty";
    return false;
  }
  out.assign(static_cast<std::size_t>(elems), 0.0f);
  switch (info.type) {
    case core::GgmlType::kF32:
      std::memcpy(out.data(), info.data, out.size() * sizeof(float));
      return true;
    case core::GgmlType::kF16: {
      const auto* s = static_cast<const std::uint16_t*>(info.data);
      for (std::size_t i = 0; i < out.size(); ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(s[i]) << 16;
        // F16 -> F32 via bit manipulation (exponent rebias).
        std::uint32_t sign = (bits >> 16) & 0x8000u;
        std::uint32_t exp = (bits >> 10) & 0x1Fu;
        std::uint32_t mant = bits & 0x3FFu;
        std::uint32_t f;
        if (exp == 0) {
          f = 0;
        } else if (exp == 31) {
          f = (sign << 16) | 0x7F800000u;
        } else {
          f = (sign << 16) | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        std::memcpy(&out[i], &f, sizeof(float));
      }
      return true;
    }
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ6_K: {
      // Contiguous row-major dequant via the shared per-row kernels.
      const std::size_t row_elems = static_cast<std::size_t>(info.dimensions[0]);
      const std::size_t nrows = out.size() / row_elems;
      const std::uint8_t* src = static_cast<const std::uint8_t*>(info.data);
      const std::size_t row_bytes = info.size_bytes / nrows;
      for (std::size_t r = 0; r < nrows; ++r) {
        float* dst = out.data() + r * row_elems;
        switch (info.type) {
          case core::GgmlType::kQ8_0:
            quant::DequantizeQ8_0(src + r * row_bytes, dst, row_elems);
            break;
          case core::GgmlType::kQ4_K:
            quant::DequantizeQ4_K(src + r * row_bytes, dst, row_elems);
            break;
          case core::GgmlType::kQ6_K:
            quant::DequantizeQ6_K(src + r * row_bytes, dst, row_elems);
            break;
          default:
            break;
        }
      }
      return true;
    }
    default:
      break;
  }
  if (error != nullptr && error->empty())
    *error = "dflash tensor has unsupported quant type";
  return false;
}

}  // namespace

std::optional<DraftModelWeights> DraftModelWeights::Bind(
    const core::GgufReader& reader, std::vector<void*>& owned,
    std::string* error_msg) {
  auto config = DraftConfig::FromGguf(reader, error_msg);
  if (!config.has_value())
    return std::nullopt;
  DraftModelWeights w;
  w.config = *config;
  const std::uint64_t hidden = config->hidden_size;

  auto mirror = [&](const std::string& name, DraftTensorRef& ref,
                    std::uint64_t cols, std::uint64_t rows) -> bool {
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (error_msg != nullptr && error_msg->empty())
        *error_msg = "missing draft tensor " + name;
      return false;
    }
    if (info->dimensions.size() < 1 || info->dimensions[0] != cols ||
        (info->dimensions.size() > 1 && info->dimensions[1] != rows)) {
      if (error_msg != nullptr && error_msg->empty())
        *error_msg = "draft tensor has unexpected dimensions: " + name;
      return false;
    }
    std::vector<float> host;
    if (!DequantToF32(*info, host, error_msg))
      return false;
    void* dev = nullptr;
    if (hipMalloc(&dev, host.size() * sizeof(float)) != hipSuccess ||
        dev == nullptr) {
      if (error_msg != nullptr && error_msg->empty())
        *error_msg = "draft weight mirror allocation failed: " + name;
      return false;
    }
    if (hipMemcpy(dev, host.data(), host.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
      (void)hipFree(dev);
      if (error_msg != nullptr && error_msg->empty())
        *error_msg = "draft weight mirror copy failed: " + name;
      return false;
    }
    ref.device = dev;
    ref.cols = cols;
    ref.rows = rows;
    owned.push_back(dev);
    return true;
  };

  if (!mirror("token_embd.weight", w.token_embd, hidden,
              config->vocab_size))
    return std::nullopt;
  // Draft shares the target vocabulary; output head mirrors the embedding
  // unless the artifact carries its own.
  if (reader.FindTensor("output.weight") != nullptr) {
    if (!mirror("output.weight", w.output, hidden, config->vocab_size))
      return std::nullopt;
  } else {
    w.output = w.token_embd;
  }
  if (!mirror("output_norm.weight", w.output_norm, hidden, 1))
    return std::nullopt;

  w.layers.resize(config->num_layers);
  const std::uint64_t qdim =
      static_cast<std::uint64_t>(config->num_heads) * config->head_dim;
  const std::uint64_t kvdim =
      static_cast<std::uint64_t>(config->num_kv_heads) * config->head_dim;
  for (std::uint32_t il = 0; il < config->num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& l = w.layers[il];
    if (!mirror(p + "attn_norm.weight", l.attn_norm, hidden, 1))
      return std::nullopt;
    if (!mirror(p + "attn_q.weight", l.wq, hidden, qdim))
      return std::nullopt;
    if (!mirror(p + "attn_k.weight", l.wk, hidden, kvdim))
      return std::nullopt;
    if (!mirror(p + "attn_v.weight", l.wv, hidden, kvdim))
      return std::nullopt;
    if (!mirror(p + "attn_q_norm.weight", l.attn_q_norm, config->head_dim,
                1))
      return std::nullopt;
    if (!mirror(p + "attn_k_norm.weight", l.attn_k_norm, config->head_dim,
                1))
      return std::nullopt;
    if (!mirror(p + "attn_output.weight", l.wo, qdim, hidden))
      return std::nullopt;
    if (!mirror(p + "ffn_norm.weight", l.ffn_norm, hidden, 1))
      return std::nullopt;
    if (!mirror(p + "ffn_gate.weight", l.ffn_gate, hidden,
                config->intermediate_size))
      return std::nullopt;
    if (!mirror(p + "ffn_up.weight", l.ffn_up, hidden,
                config->intermediate_size))
      return std::nullopt;
    if (!mirror(p + "ffn_down.weight", l.ffn_down, config->intermediate_size,
                hidden))
      return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::glimmer::draft
