#include "src/cli/serve/chat_dialect.hpp"

#include <array>

#include "src/models/deepseek_v4_flash/tool_calls.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace mbufo::server {
namespace {

// Mirrors openai_chat.cpp's former kToolMarkers: the JSON fallback has no
// native counterpart, so its grammar leaves text before the marker
// unconstrained and models still write their native call syntax. Admit
// every known opener here (moved verbatim).
constexpr std::array<std::string_view, 7> kJsonOpeners{
    tokenization::QwenToolSyntax::kCallOpen,
    "<｜DSML｜tool_calls｜>",
    models::deepseek_v4_flash::DeepSeekToolSyntax::kCallsOpen,
    "<DSML｜tool_calls｜>",
    "<DSML｜tool_calls>",
    "<tool_calls｜>",
    "<tool_calls>",
};

// Mirrors openai_chat.cpp's former kToolClosers (moved verbatim).
constexpr std::array<std::string_view, 9> kJsonClosers{
    tokenization::QwenToolSyntax::kParameterClose,
    tokenization::QwenToolSyntax::kFunctionClose,
    tokenization::QwenToolSyntax::kCallClose,
    "</invoke>",
    "</function_results>",
    "</tool_calls>",
    models::deepseek_v4_flash::DeepSeekToolSyntax::kParameterCloses[0],
    models::deepseek_v4_flash::DeepSeekToolSyntax::kInvokeCloses[0],
    models::deepseek_v4_flash::DeepSeekToolSyntax::kCallsClose,
};

}  // namespace

DialectMarkers JsonDialect::Markers() const noexcept {
  return {
      .openers = kJsonOpeners,
      .closers = kJsonClosers,
      .reasoning_envelope = {},
  };
}

const IChatDialect& JsonDialectInstance() noexcept {
  static const JsonDialect kInstance;
  return kInstance;
}

// Mirrors openai_chat.cpp's former LlamaToolSyntax::kCallOpen + kLlamaClosers.
constexpr std::string_view kLlamaCallOpen = "{\"name\":";
constexpr std::array<std::string_view, 1> kLlamaOpeners{kLlamaCallOpen};
constexpr std::array<std::string_view, 2> kLlamaDialectClosers{
    "<|eot_id|>",
    "</invoke>",
};

DialectMarkers LlamaDialect::Markers() const noexcept {
  return {
      .openers = kLlamaOpeners,
      .closers = kLlamaDialectClosers,
      .reasoning_envelope = {},
  };
}

const IChatDialect& LlamaDialectInstance() noexcept {
  static const LlamaDialect kInstance;
  return kInstance;
}

}  // namespace mbufo::server
