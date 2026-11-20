#include "src/models/qwen3_tts/prompt.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace qwen3_tts = mbufo::models::qwen3_tts;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_prompt_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

}  // namespace

int main() {
  constexpr std::size_t hidden = 2;
  const std::vector<float> instruction{1.0F, 2.0F};
  const std::vector<float> role{3.0F, 4.0F};
  const std::vector<float> codec{
      10.0F, 20.0F,  // language
      30.0F, 40.0F,  // codec pad
      50.0F, 60.0F,  // codec BOS
  };
  const std::vector<float> text{5.0F, 6.0F};
  const std::vector<float> tts_bos{100.0F, 200.0F};
  const std::vector<float> tts_eos{300.0F, 400.0F};
  const std::vector<float> tts_pad{7.0F, 8.0F};
  const std::vector<float> codec_pad{9.0F, 10.0F};

  qwen3_tts::PromptOutput non_icl;
  std::string error;
  Check(qwen3_tts::BuildNonIclPrompt(
            {
                .hidden_size = hidden,
                .instruction = instruction,
                .role = role,
                .codec = codec,
                .text = text,
                .tts_bos = tts_bos,
                .tts_eos = tts_eos,
                .tts_pad = tts_pad,
                .codec_pad = codec_pad,
            },
            &non_icl, &error),
        error);
  Check(non_icl.tokens == 7, "non-ICL prompt row count");
  Check(non_icl.trailing_text == tts_pad, "non-ICL trailing text");
  Check(non_icl.embeddings ==
            std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F, 17.0F, 28.0F, 130.0F,
                                240.0F, 14.0F, 16.0F, 308.0F, 410.0F, 57.0F,
                                68.0F}),
        "non-ICL row composition");

  const std::vector<float> combined_text{1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> reference_codec{11.0F, 12.0F, 13.0F, 14.0F};
  qwen3_tts::PromptOutput icl;
  Check(qwen3_tts::BuildIclPrompt(
            {
                .hidden_size = hidden,
                .role = role,
                .codec = codec,
                .combined_text = combined_text,
                .reference_codec = reference_codec,
                .tts_bos = tts_bos,
                .tts_eos = tts_eos,
                .tts_pad = tts_pad,
            },
            &icl, &error),
        error);
  Check(icl.tokens == 5, "ICL prompt row count");
  Check(icl.trailing_text == tts_eos, "ICL preserves unmatched text rows");
  Check(icl.embeddings ==
            std::vector<float>({3.0F, 4.0F, 17.0F, 28.0F, 130.0F, 240.0F, 12.0F,
                                14.0F, 16.0F, 18.0F}),
        "ICL row composition");

  std::cout << "PASS qwen3_tts_prompt_test\n";
  return 0;
}
