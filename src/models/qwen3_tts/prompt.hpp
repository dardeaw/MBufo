#ifndef MBUFO_MODELS_QWEN3_TTS_PROMPT_HPP_
#define MBUFO_MODELS_QWEN3_TTS_PROMPT_HPP_

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace mbufo::models::qwen3_tts {

struct PromptOutput {
  std::vector<float> embeddings;
  std::vector<float> trailing_text;
  std::vector<float> tts_pad;
  std::size_t tokens{0};
};

/// Projected host-side rows used by CustomVoice, VoiceDesign, and
/// speaker-embedding-only Base prompts.
struct NonIclPromptInput {
  std::size_t hidden_size{0};
  std::span<const float> instruction;
  std::span<const float> role;
  // Codec prefix followed by codec_pad and codec_bos rows.
  std::span<const float> codec;
  std::span<const float> text;
  std::span<const float> tts_bos;
  std::span<const float> tts_eos;
  std::span<const float> tts_pad;
  std::span<const float> codec_pad;
};

/// Projected host-side rows used by a Base in-context voice-clone prompt.
struct IclPromptInput {
  std::size_t hidden_size{0};
  std::span<const float> role;
  // Codec prefix, speaker embedding, codec_pad, and codec_bos rows.
  std::span<const float> codec;
  // Projected reference text followed by projected target text.
  std::span<const float> combined_text;
  // codec_bos followed by one summed 16-codebook embedding per reference frame.
  std::span<const float> reference_codec;
  std::span<const float> tts_bos;
  std::span<const float> tts_eos;
  std::span<const float> tts_pad;
};

[[nodiscard]] bool BuildNonIclPrompt(const NonIclPromptInput& input,
                                     PromptOutput* output,
                                     std::string* error = nullptr);

[[nodiscard]] bool BuildIclPrompt(const IclPromptInput& input,
                                  PromptOutput* output,
                                  std::string* error = nullptr);

}  // namespace mbufo::models::qwen3_tts

#endif  // MBUFO_MODELS_QWEN3_TTS_PROMPT_HPP_
