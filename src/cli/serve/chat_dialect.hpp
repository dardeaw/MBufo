#ifndef MBUFO_SERVER_CHAT_DIALECT_HPP_
#define MBUFO_SERVER_CHAT_DIALECT_HPP_

// Phase 2: IChatDialect (per-model chat dialect).
//
// Why: prompt rendering (inference_backend.cpp), tool grammar
// (text_model_runner.cpp ConstrainChatRequest), streaming segmentation
// (openai_chat.cpp ToolMarkers/ToolClosers/ReasoningToolMarkers) and
// completion parsing (Parse*Generation) currently branch on ToolFormat
// enums scattered across three files. A new model easily migrates only
// half of them (e.g. tool names leaking into reasoning_content).
//
// Migration order (one dialect per commit, independently verified):
// Json -> Llama -> Qwen -> DeepSeek -> Gemma4 -> ATEM -> Inkling.
// Each step replaces one `if (format == ...)` branch with a dialect call.

#include <optional>
#include <span>
#include <string_view>

#include "src/core/json_constraint.hpp"

namespace mbufo::server {

// Canonical opener/closer tables for one dialect's tool-call framing.
struct DialectMarkers {
  std::span<const std::string_view> openers;
  std::span<const std::string_view> closers;
  // Native reasoning envelope opener, if the dialect has one (used to
  // detect the implicit reasoning boundary, e.g. Qwen/DeepSeek native
  // function headers). Empty when the dialect has no envelope.
  std::span<const std::string_view> reasoning_envelope;
};

class IChatDialect {
 public:
  virtual ~IChatDialect() = default;

  // Tool-call framing markers for this dialect (replaces ToolMarkers /
  // ToolClosers / ReasoningToolMarkers branches in openai_chat.cpp).
  [[nodiscard]] virtual DialectMarkers Markers() const noexcept = 0;

  // The grammar format this dialect constrains with (replaces
  // ConstrainChatRequest's `if (format == ...)` instruction branches).
  [[nodiscard]] virtual sampling::JsonConstraint::ToolFormat GrammarFormat()
      const noexcept = 0;
};

// JSON fallback dialect: no native counterpart in llama.cpp; admits every
// known opener (models still write their native call syntax under an
// unconstrained grammar prefix).
class JsonDialect final : public IChatDialect {
 public:
  [[nodiscard]] DialectMarkers Markers() const noexcept override;
  [[nodiscard]] sampling::JsonConstraint::ToolFormat GrammarFormat()
      const noexcept override {
    return sampling::JsonConstraint::ToolFormat::kJson;
  }
};

[[nodiscard]] const IChatDialect& JsonDialectInstance() noexcept;

// Llama-3.1 custom tool calls (canonical template): the model emits a
// bare JSON object {"name": "fn", "parameters": {...}} per call.
class LlamaDialect final : public IChatDialect {
 public:
  [[nodiscard]] DialectMarkers Markers() const noexcept override;
  [[nodiscard]] sampling::JsonConstraint::ToolFormat GrammarFormat()
      const noexcept override {
    return sampling::JsonConstraint::ToolFormat::kLlama;
  }
};

[[nodiscard]] const IChatDialect& LlamaDialectInstance() noexcept;

// Qwen family: <tool_call> opener, four closers, and a native function
// header envelope for the implicit reasoning boundary.
class QwenDialect final : public IChatDialect {
 public:
  [[nodiscard]] DialectMarkers Markers() const noexcept override;
  [[nodiscard]] sampling::JsonConstraint::ToolFormat GrammarFormat()
      const noexcept override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
};

[[nodiscard]] const IChatDialect& QwenDialectInstance() noexcept;

// DeepSeek: two-element opener (with/without template separator), no
// closers (content is everything outside the native call block), and a
// two-element native header envelope.
class DeepSeekDialect final : public IChatDialect {
 public:
  [[nodiscard]] DialectMarkers Markers() const noexcept override;
  [[nodiscard]] sampling::JsonConstraint::ToolFormat GrammarFormat()
      const noexcept override {
    return sampling::JsonConstraint::ToolFormat::kDeepSeek;
  }
};

[[nodiscard]] const IChatDialect& DeepSeekDialectInstance() noexcept;

// Gemma 4 native framing (<|tool_call>call:NAME{args}<tool_call|>).
// No reasoning envelope: no thought tags means no implicit boundary.
class GemmaDialect final : public IChatDialect {
 public:
  [[nodiscard]] DialectMarkers Markers() const noexcept override;
  [[nodiscard]] sampling::JsonConstraint::ToolFormat GrammarFormat()
      const noexcept override {
    return sampling::JsonConstraint::ToolFormat::kGemma;
  }
};

[[nodiscard]] const IChatDialect& GemmaDialectInstance() noexcept;

// Muse Glimmer (ATEM) framing (<atem:function_calls> ... </atem:function_calls>).
// No reasoning envelope.
class AtemDialect final : public IChatDialect {
 public:
  [[nodiscard]] DialectMarkers Markers() const noexcept override;
  [[nodiscard]] sampling::JsonConstraint::ToolFormat GrammarFormat()
      const noexcept override {
    return sampling::JsonConstraint::ToolFormat::kAtem;
  }
};

[[nodiscard]] const IChatDialect& AtemDialectInstance() noexcept;

// Inkling: the tool-call message model tag as opener, no closers
// (structure guard steers at message ends), no reasoning envelope.
class InklingDialect final : public IChatDialect {
 public:
  [[nodiscard]] DialectMarkers Markers() const noexcept override;
  [[nodiscard]] sampling::JsonConstraint::ToolFormat GrammarFormat()
      const noexcept override {
    return sampling::JsonConstraint::ToolFormat::kInkling;
  }
};

[[nodiscard]] const IChatDialect& InklingDialectInstance() noexcept;

}  // namespace mbufo::server

#endif  // MBUFO_SERVER_CHAT_DIALECT_HPP_
