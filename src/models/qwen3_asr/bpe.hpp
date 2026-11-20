// Byte-level BPE tokenizer for this model's 151936-token Qwen2 vocabulary.
//
// Copied from src/models/qwen/tokenizer.{hpp,cpp} so that src/models/qwen3_asr
// builds without depending on another model's sources. The encode and decode
// algorithms are unchanged -- the generated token IDs are part of this model's
// quality contract, so this copy must stay behaviourally identical rather than
// being "improved" independently.
//
// The GGUF and binary-vocabulary constructors are omitted: this model loads a
// safetensors checkpoint and only ever builds from an explicit vocabulary, and
// dropping the GGUF path is what removes the last src/core dependency.

#ifndef MBUFO_MODELS_QWEN3_ASR_BPE_HPP_
#define MBUFO_MODELS_QWEN3_ASR_BPE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mbufo::models::qwen3_asr::tokenization {

using TokenId = std::uint32_t;

constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;
constexpr TokenId kDefaultEosTokenId = 151645U;   // <|im_end|>
constexpr TokenId kDefaultEndoftextId = 151643U;  // <|endoftext|>

/// Configuration options for the tokenizer encoding pass.
struct TokenizerOptions {
  bool add_bos{false};
  bool add_eos{false};
  bool parse_special_tokens{true};
};

struct VocabularyLoadOptions {
  bool eager_decoded_tokens{true};
};

/// Deterministic, zero-allocation-on-query Qwen BPE Tokenizer.
class BpeTokenizer {
public:
  ~BpeTokenizer() = default;

  BpeTokenizer(const BpeTokenizer&) = delete;
  BpeTokenizer& operator=(const BpeTokenizer&) = delete;
  BpeTokenizer(BpeTokenizer&& other) noexcept = default;
  BpeTokenizer& operator=(BpeTokenizer&& other) noexcept = default;

  /// Creates a tokenizer directly from token and merge lists (useful for tests
  /// and custom models).
  [[nodiscard]] static std::unique_ptr<BpeTokenizer> CreateFromVocabulary(
      std::span<const std::string> tokens, std::span<const std::string> merges,
      const std::unordered_map<std::string, TokenId>& special_tokens = {},
      std::string* error_msg = nullptr,
      VocabularyLoadOptions load_options = {});

  /// Encodes a UTF-8 text string into token IDs.
  [[nodiscard]] std::vector<TokenId> Encode(
      std::string_view text, const TokenizerOptions& options = {}) const;

  /// Decodes a sequence of token IDs into a UTF-8 string.
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;

  /// Decodes a single token ID into its string representation.
  [[nodiscard]] std::string_view DecodeToken(TokenId token_id) const noexcept;
  /// Decodes a single token by value. This remains exact when eager decoded
  /// token construction was disabled at load time.
  [[nodiscard]] std::string DecodeTokenCopy(TokenId token_id) const;

  [[nodiscard]] std::size_t GetVocabSize() const noexcept {
    return id_to_token_.size();
  }

  [[nodiscard]] TokenId GetEosTokenId() const noexcept { return eos_token_id_; }
  [[nodiscard]] TokenId GetBosTokenId() const noexcept { return bos_token_id_; }
  [[nodiscard]] TokenId GetPadTokenId() const noexcept { return pad_token_id_; }

  [[nodiscard]] std::optional<TokenId> FindSpecialToken(
      std::string_view token_str) const noexcept;
  [[nodiscard]] bool IsSpecialToken(TokenId id) const noexcept;

private:
  struct PairHash {
    std::size_t operator()(
        const std::pair<TokenId, TokenId>& p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32) ^
             static_cast<std::size_t>(p.second);
    }
  };

  BpeTokenizer() = default;

  void InitializeByteTokens(bool eager_decoded_tokens = true);
  std::vector<TokenId> BpeMergeChunk(std::string_view chunk) const;

  std::vector<std::string> id_to_token_;
  std::vector<std::string> id_to_decoded_token_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::pair<TokenId, TokenId>, std::uint32_t, PairHash>
      merge_ranks_;
  std::unordered_map<std::string, TokenId> special_token_to_id_;
  std::unordered_map<TokenId, bool> is_special_token_;

  std::array<TokenId, 256> byte_tokens_{};
  TokenId bos_token_id_{kInvalidTokenId};
  TokenId eos_token_id_{kDefaultEosTokenId};
  TokenId pad_token_id_{kDefaultEndoftextId};
};

}  // namespace mbufo::models::qwen3_asr::tokenization

#endif  // MBUFO_MODELS_QWEN3_ASR_BPE_HPP_
