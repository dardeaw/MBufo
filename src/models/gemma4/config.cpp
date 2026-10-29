#include "src/models/gemma4/config.hpp"

#include <cmath>
#include <string>

namespace gufo::models::gemma4 {
namespace {

constexpr std::string_view kArchitecture = "gemma4";

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

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return fallback;
  }

  float F32(std::string_view key, float fallback) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value))
      Fail("missing or invalid GGUF float " + std::string(key));
    return value.value_or(fallback);
  }

  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required)
        Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }

  std::vector<bool> BoolArray(std::string_view key, bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<bool> out;
    if (meta == nullptr) {
      if (required)
        Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (auto v : *u) out.push_back(v != 0);
    } else if (const auto* i =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *i) out.push_back(v != 0);
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

}  // namespace

std::uint32_t Config::KvProducingLayer(std::uint32_t layer) const {
  if (layer >= num_layers)
    return num_layers;
  if (shared_kv_layers == 0)
    return layer;  // llama.cpp: n_layer_kv_from_start == num_layers, all own KV.
  if (!LayerIsSwa(layer))
    return layer;
  // SWA layers read the KV cache written by the nearest preceding
  // non-SWA layer (the full-attention producer).
  for (std::uint32_t i = layer; i-- > 0;) {
    if (!LayerIsSwa(i))
      return i;
  }
  return layer;
}

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not gemma4");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "gemma4.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
  };

  c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-6F);
  c.num_heads = r.U32(key("attention.head_count"));
  {
    const auto kv = r.U64Array(key("attention.head_count_kv"));
    if (kv.size() != c.num_layers) {
      r.Fail("attention.head_count_kv must have one entry per layer");
    } else {
      for (auto v : kv) c.num_kv_heads.push_back(static_cast<std::uint32_t>(v));
    }
  }
  c.is_swa = r.BoolArray(key("attention.sliding_window_pattern"));
  if (c.is_swa.size() != c.num_layers) {
    r.Fail("attention.sliding_window_pattern must have one entry per layer");
  }
  c.head_dim_full = r.U32(key("attention.key_length"));
  if (r.U32(key("attention.value_length"), false, c.head_dim_full) !=
      c.head_dim_full) {
    r.Fail("gemma4 requires attention value length to match key length");
  }
  c.head_dim_swa = r.U32(key("attention.key_length_swa"), false,
                         c.head_dim_full);
  c.rotary_full = r.U32(key("rope.dimension_count"), false, c.head_dim_full);
  c.rotary_swa =
      r.U32(key("rope.dimension_count_swa"), false, c.head_dim_swa);
  c.rope_theta_full = r.F32(key("rope.freq_base"), 1000000.0F);
  c.rope_theta_swa = r.F32(key("rope.freq_base_swa"), 10000.0F);

  // Per-layer head dim is derived from the ACTUAL attn_q weight shape
  // (rows / num_heads).  The GGUF sliding_window_pattern polarity is not
  // consistent across Gemma 4 exports, so it is ignored for layer
  // classification; SWA membership = layers whose measured head dim
  // equals key_length_swa.
  c.head_dim_per_layer.assign(c.num_layers, 0);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const auto* t = r.gguf.FindTensor("blk." + std::to_string(il) +
                                      ".attn_q.weight");
    if (t == nullptr || t->dimensions.size() != 2 ||
        t->dimensions[1] == 0 || c.num_heads == 0 ||
        t->dimensions[1] % c.num_heads != 0) {
      r.Fail("gemma4 cannot measure per-layer head dimension from attn_q");
      return std::nullopt;
    }
    c.head_dim_per_layer[il] =
        static_cast<std::uint32_t>(t->dimensions[1] / c.num_heads);
  }
  // is_swa comes from sliding_window_pattern directly (True = sliding
  // window layer).  KV ownership is independent: shared_kv_layers==0 means
  // every layer owns its own KV cache (llama.cpp has_kv semantics).
  for (std::size_t il = 0; il < c.is_swa.size(); ++il) {
    // keep the parsed pattern value; no re-derivation
  }
  std::fprintf(stderr,
               "[gemma4] measured head_dim L0=%u L5=%u swa=%u full=%u "
               "is_swa[0..5]=%d%d%d%d%d %d\n",
               c.head_dim_per_layer[0], c.head_dim_per_layer[5],
               c.head_dim_swa, c.head_dim_full, (int)c.is_swa[0],
               (int)c.is_swa[1], (int)c.is_swa[2], (int)c.is_swa[3],
               (int)c.is_swa[4], (int)c.is_swa[5]);
  c.sliding_window = r.U32(key("attention.sliding_window"));
  c.shared_kv_layers = r.U32(key("attention.shared_kv_layers"), false, 0);
  c.num_experts = r.U32(key("expert_count"));
  c.experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_ff = r.U32(key("feed_forward_length"));
  c.final_logit_softcapping =
      r.F32(key("final_logit_softcapping"), 0.0F);

  // Structural contract for this runtime.
  if (c.num_layers == 0 || c.hidden_size == 0 || c.num_heads == 0) {
    r.Fail("gemma4 artifact is missing required dimensions");
  }
  if (c.head_dim_full % 64 != 0 || c.head_dim_swa % 64 != 0) {
    r.Fail("gemma4 head dimensions must be multiples of 64");
  }
  if (c.experts_used == 0 || c.experts_used > c.num_experts) {
    r.Fail("gemma4 expert selection is out of range");
  }
  if (c.num_experts > 1024) {
    r.Fail("gemma4 expert count exceeds the router kernel limit");
  }
  if (!r.ok)
    return std::nullopt;
  return c;
}

}  // namespace gufo::models::gemma4
