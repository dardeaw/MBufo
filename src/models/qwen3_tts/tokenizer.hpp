#ifndef MBUFO_MODELS_QWEN3_TTS_TOKENIZER_HPP_
#define MBUFO_MODELS_QWEN3_TTS_TOKENIZER_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mbufo::models::qwen3_tts {

/// Qwen2 byte-level BPE tokenizer used by Qwen3-TTS.
///
/// The CustomVoice checkpoints distribute the tokenizer as vocab.json,
/// merges.txt, and tokenizer_config.json rather than tokenizer.json.
class Tokenizer {
public:
  [[nodiscard]] static bool Load(const std::filesystem::path& model_root,
                                 Tokenizer* output,
                                 std::string* error = nullptr);

  [[nodiscard]] bool Encode(std::string_view text,
                            std::vector<std::uint32_t>* ids,
                            std::string* error = nullptr) const;

  [[nodiscard]] bool EncodeAssistantPrompt(std::string_view text,
                                           std::vector<std::uint32_t>* ids,
                                           std::string* error = nullptr) const;

  [[nodiscard]] bool EncodeInstructionPrompt(
      std::string_view instruction, std::vector<std::uint32_t>* ids,
      std::string* error = nullptr) const;

  [[nodiscard]] bool EncodeReferencePrompt(std::string_view text,
                                           std::vector<std::uint32_t>* ids,
                                           std::string* error = nullptr) const;

  [[nodiscard]] std::size_t vocabulary_size() const noexcept {
    return vocab_.size() + added_tokens_.size();
  }

private:
  struct AddedToken {
    std::string content;
    std::uint32_t id{0};
  };

  [[nodiscard]] bool EncodePlain(std::string_view text,
                                 std::vector<std::uint32_t>* ids,
                                 std::string* error) const;
  [[nodiscard]] bool ApplyBpe(std::string_view piece,
                              std::vector<std::uint32_t>* ids,
                              std::string* error) const;

  std::unordered_map<std::string, std::uint32_t> vocab_;
  std::unordered_map<std::string, std::uint32_t> merge_ranks_;
  std::unordered_map<std::string, std::uint32_t> added_tokens_;
  std::vector<AddedToken> added_alternatives_;
  std::array<std::string, 256> byte_encoder_;
};

}  // namespace mbufo::models::qwen3_tts

#endif  // MBUFO_MODELS_QWEN3_TTS_TOKENIZER_HPP_
