#ifndef GUFO_MODELS_LLAMA_CONFIG_HPP_
#define GUFO_MODELS_LLAMA_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::llama {

/// Architecture parameters of a `llama` GGUF artifact (dense decoder-only
/// transformer: Llama-3.1-8B-Instruct and its quantizations). Every value is
/// read from the file; the runtime is dense/GQA-only (no SWA, no MoE).
struct Config {
  std::uint32_t num_layers{0};        ///< llama.block_count, 32.
  std::uint32_t hidden_size{0};       ///< llama.embedding_length, 4096.
  std::uint32_t intermediate_size{0}; ///< llama.feed_forward_length, 14336.
  std::uint32_t num_heads{0};         ///< llama.attention.head_count, 32.
  std::uint32_t num_kv_heads{0};      ///< llama.attention.head_count_kv, 8.
  std::uint32_t head_dim{0};          ///< llama.attention.key_length, 128.
  std::uint32_t vocab_size{0};        ///< 128256 (from token_embd shape).
  std::uint32_t context_length{0};    ///< llama.context_length, 131072.
  float rope_theta{500000.0F};        ///< llama.rope.freq_base.
  float rms_eps{1e-5F};               ///< llama.attention.layer_norm_rms_epsilon.

  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::llama

#endif  // GUFO_MODELS_LLAMA_CONFIG_HPP_
