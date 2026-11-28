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

// Mirrors openai_chat.cpp's former qwen opener + kQwenClosers + native
// function header envelope (ReasoningToolMarkers' qwen branch).
constexpr std::array<std::string_view, 1> kQwenOpeners{
    tokenization::QwenToolSyntax::kCallOpen};
constexpr std::array<std::string_view, 4> kQwenDialectClosers{
    tokenization::QwenToolSyntax::kParameterClose,
    tokenization::QwenToolSyntax::kFunctionClose,
    tokenization::QwenToolSyntax::kCallClose,
    "</invoke>",
};
constexpr std::array<std::string_view, 1> kQwenReasoningEnvelope{
    tokenization::QwenToolSyntax::kNativeCallOpen};

DialectMarkers QwenDialect::Markers() const noexcept {
  return {
      .openers = kQwenOpeners,
      .closers = kQwenDialectClosers,
      .reasoning_envelope = kQwenReasoningEnvelope,
  };
}

const IChatDialect& QwenDialectInstance() noexcept {
  static const QwenDialect kInstance;
  return kInstance;
}

// Mirrors openai_chat.cpp's former deepseek opener pair + envelope pair.
// Closers are an empty set: DeepSeek content is everything outside its
// native call block (no echo to strip).
constexpr std::array<std::string_view, 2> kDeepSeekOpeners{
    models::deepseek_v4_flash::DeepSeekToolSyntax::kCallsOpenWithSeparator,
    models::deepseek_v4_flash::DeepSeekToolSyntax::kCallsOpen};
constexpr std::array<std::string_view, 2> kDeepSeekReasoningEnvelope{
    models::deepseek_v4_flash::DeepSeekToolSyntax::kNativeCallOpenWithSeparator,
    models::deepseek_v4_flash::DeepSeekToolSyntax::kNativeCallOpen};

DialectMarkers DeepSeekDialect::Markers() const noexcept {
  return {
      .openers = kDeepSeekOpeners,
      .closers = {},
      .reasoning_envelope = kDeepSeekReasoningEnvelope,
  };
}

const IChatDialect& DeepSeekDialectInstance() noexcept {
  static const DeepSeekDialect kInstance;
  return kInstance;
}

}  // namespace mbufo::server
