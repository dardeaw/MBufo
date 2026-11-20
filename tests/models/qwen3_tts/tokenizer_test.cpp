#include "src/models/qwen3_tts/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kText =
    "The boy who lived. Mr. and Mrs. Dursley, of number four, Privet Drive, "
    "were proud to say that they were perfectly normal, thank you very much. "
    "They were the last people you'd expect to be involved in anything strange "
    "or mysterious, because they just didn't hold with such nonsense.";

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_tokenizer_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path model_root =
      argc > 1 ? argv[1]
               : "/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice";
  if (!std::filesystem::is_regular_file(model_root / "vocab.json")) {
    std::cerr
        << "SKIP qwen3_tts_tokenizer_test: external model is unavailable\n";
    return 77;
  }

  mbufo::models::qwen3_tts::Tokenizer tokenizer;
  std::string error;
  Check(
      mbufo::models::qwen3_tts::Tokenizer::Load(model_root, &tokenizer, &error),
      error);
  std::vector<std::uint32_t> ids;
  Check(tokenizer.EncodeAssistantPrompt(kText, &ids, &error), error);
  const std::vector<std::uint32_t> expected = {
      151644, 77091, 198,   785,   8171,   879,   12163,  13,    4392,
      13,     323,   17618, 13,    422,    1723,  3179,   11,    315,
      1372,   3040,  11,    15438, 295,    16150, 11,     1033,  12409,
      311,    1977,  429,   807,   1033,   13942, 4622,   11,    9702,
      498,    1602,  1753,  13,    2379,   1033,  279,    1537,  1251,
      498,    4172,  1720,  311,   387,    6398,  304,    4113,  14888,
      476,    25382, 11,    1576,  807,    1101,  3207,   944,   3331,
      448,    1741,  40802, 13,    151645, 198,   151644, 77091, 198,
  };
  Check(ids == expected, "assistant prompt differs from official tokenizer");
  std::cout << "PASS qwen3_tts_tokenizer_test tokens=" << ids.size() << '\n';
  return 0;
}
