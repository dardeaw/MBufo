#include "src/models/qwen3_tts/prompt.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mbufo::models::qwen3_tts {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

float RoundBfloat16(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7F800000U) != 0x7F800000U) {
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
  }
  return std::bit_cast<float>(bits & 0xFFFF0000U);
}

void RequireRows(std::span<const float> values, std::size_t hidden_size,
                 const char* name, bool allow_empty = false) {
  if (hidden_size == 0 || values.size() % hidden_size != 0 ||
      (!allow_empty && values.empty())) {
    throw std::invalid_argument(std::string("Qwen3-TTS ") + name +
                                " embedding shape is invalid");
  }
}

void AppendRows(std::vector<float>* destination, std::span<const float> rows) {
  destination->insert(destination->end(), rows.begin(), rows.end());
}

void AppendAddedRow(std::vector<float>* destination,
                    std::span<const float> left, std::span<const float> right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("Qwen3-TTS prompt row shape mismatch");
  }
  destination->reserve(destination->size() + left.size());
  for (std::size_t index = 0; index < left.size(); ++index) {
    destination->push_back(RoundBfloat16(left[index] + right[index]));
  }
}

std::span<const float> Row(std::span<const float> values, std::size_t row,
                           std::size_t hidden_size) {
  return values.subspan(row * hidden_size, hidden_size);
}

void BuildPrefix(std::span<const float> leading_rows,
                 std::span<const float> role, std::span<const float> codec,
                 std::span<const float> tts_bos, std::span<const float> tts_pad,
                 std::size_t hidden_size, PromptOutput* output) {
  AppendRows(&output->embeddings, leading_rows);
  AppendRows(&output->embeddings, role);
  const std::size_t codec_rows = codec.size() / hidden_size;
  if (codec_rows < 2) {
    throw std::invalid_argument("Qwen3-TTS codec prompt is too short");
  }
  for (std::size_t row = 0; row + 2 < codec_rows; ++row) {
    AppendAddedRow(&output->embeddings, tts_pad, Row(codec, row, hidden_size));
  }
  AppendAddedRow(&output->embeddings, tts_bos,
                 Row(codec, codec_rows - 2, hidden_size));
}

void Finish(PromptOutput* output, std::size_t hidden_size) {
  if (output->embeddings.empty() ||
      output->embeddings.size() % hidden_size != 0) {
    throw std::runtime_error("Qwen3-TTS prompt embedding shape mismatch");
  }
  output->tokens = output->embeddings.size() / hidden_size;
}

}  // namespace

bool BuildNonIclPrompt(const NonIclPromptInput& input, PromptOutput* output,
                       std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS prompt output must not be null");
    return false;
  }
  *output = {};
  try {
    RequireRows(input.instruction, input.hidden_size, "instruction", true);
    RequireRows(input.role, input.hidden_size, "role");
    RequireRows(input.codec, input.hidden_size, "codec");
    RequireRows(input.text, input.hidden_size, "text", true);
    RequireRows(input.tts_bos, input.hidden_size, "TTS BOS");
    RequireRows(input.tts_eos, input.hidden_size, "TTS EOS");
    RequireRows(input.tts_pad, input.hidden_size, "TTS pad");
    RequireRows(input.codec_pad, input.hidden_size, "codec pad");
    if (input.tts_bos.size() != input.hidden_size ||
        input.tts_eos.size() != input.hidden_size ||
        input.tts_pad.size() != input.hidden_size ||
        input.codec_pad.size() != input.hidden_size) {
      throw std::invalid_argument(
          "Qwen3-TTS special embedding must contain one row");
    }

    output->tts_pad.assign(input.tts_pad.begin(), input.tts_pad.end());
    output->trailing_text = output->tts_pad;
    BuildPrefix(input.instruction, input.role, input.codec, input.tts_bos,
                input.tts_pad, input.hidden_size, output);
    const std::size_t text_rows = input.text.size() / input.hidden_size;
    for (std::size_t row = 0; row < text_rows; ++row) {
      AppendAddedRow(&output->embeddings,
                     Row(input.text, row, input.hidden_size), input.codec_pad);
    }
    AppendAddedRow(&output->embeddings, input.tts_eos, input.codec_pad);
    const std::size_t codec_rows = input.codec.size() / input.hidden_size;
    AppendAddedRow(&output->embeddings, input.tts_pad,
                   Row(input.codec, codec_rows - 1, input.hidden_size));
    Finish(output, input.hidden_size);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

bool BuildIclPrompt(const IclPromptInput& input, PromptOutput* output,
                    std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS prompt output must not be null");
    return false;
  }
  *output = {};
  try {
    RequireRows(input.role, input.hidden_size, "role");
    RequireRows(input.codec, input.hidden_size, "codec");
    RequireRows(input.combined_text, input.hidden_size, "combined text", true);
    RequireRows(input.reference_codec, input.hidden_size, "reference codec");
    RequireRows(input.tts_bos, input.hidden_size, "TTS BOS");
    RequireRows(input.tts_eos, input.hidden_size, "TTS EOS");
    RequireRows(input.tts_pad, input.hidden_size, "TTS pad");
    if (input.tts_bos.size() != input.hidden_size ||
        input.tts_eos.size() != input.hidden_size ||
        input.tts_pad.size() != input.hidden_size) {
      throw std::invalid_argument(
          "Qwen3-TTS special embedding must contain one row");
    }

    output->tts_pad.assign(input.tts_pad.begin(), input.tts_pad.end());
    BuildPrefix({}, input.role, input.codec, input.tts_bos, input.tts_pad,
                input.hidden_size, output);

    std::vector<float> text(input.combined_text.begin(),
                            input.combined_text.end());
    AppendRows(&text, input.tts_eos);
    const std::size_t text_rows = text.size() / input.hidden_size;
    const std::size_t codec_rows =
        input.reference_codec.size() / input.hidden_size;
    const std::size_t common_rows = std::min(text_rows, codec_rows);
    for (std::size_t row = 0; row < common_rows; ++row) {
      AppendAddedRow(&output->embeddings, Row(text, row, input.hidden_size),
                     Row(input.reference_codec, row, input.hidden_size));
    }
    if (text_rows > codec_rows) {
      const auto remaining =
          std::span<const float>(text).subspan(codec_rows * input.hidden_size);
      output->trailing_text.assign(remaining.begin(), remaining.end());
    } else {
      for (std::size_t row = text_rows; row < codec_rows; ++row) {
        AppendAddedRow(&output->embeddings, input.tts_pad,
                       Row(input.reference_codec, row, input.hidden_size));
      }
      output->trailing_text = output->tts_pad;
    }
    Finish(output, input.hidden_size);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

}  // namespace mbufo::models::qwen3_tts
