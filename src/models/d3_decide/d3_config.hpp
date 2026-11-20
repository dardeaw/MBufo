#pragma once
// D3 decide 模型定義（代碼框 v0，修正：decide = D3，不是 qwen3.8-flash-next）。
//
// 來源（已上網核實）：
//   HF `vllm-sr/d3-*`：zero-shot-classification，tags decision-model + system-one
//     d3-lite  <- Qwen3.5-0.8B
//     d3-nano  <- Qwen3.5-2B
//     d3-mini  <- Qwen3.5-4B
//     d3-flash / d3-turbo / d3-ultra / d3-edge
//   NPU 現貨：HF `FastFlowLM/D3-*-NPU2`（amd-npu + q4nx），flm 側
//     `flm serve --decision d3:lite|nano|mini|turbo|ultra`，`POST /v1/systemone`
// License：Apache-2.0。

#include <cstdint>
#include <string>
#include <string_view>

namespace mbufo::models::d3_decide {

enum class D3Variant : std::uint8_t {
  kLite = 0,  // 0.8B，最輕，NPU 首選
  kNano,      // 2B
  kMini,      // 4B
  kFlash,
  kTurbo,
  kUltra,
  kEdge,
};

[[nodiscard]] inline std::string_view ToFlmTag(D3Variant v) noexcept {
  switch (v) {
    case D3Variant::kLite: return "d3:lite";
    case D3Variant::kNano: return "d3:nano";
    case D3Variant::kMini: return "d3:mini";
    case D3Variant::kFlash: return "d3:flash";
    case D3Variant::kTurbo: return "d3:turbo";
    case D3Variant::kUltra: return "d3:ultra";
    case D3Variant::kEdge: return "d3:edge";
  }
  return "d3:lite";
}

[[nodiscard]] inline std::string_view ToNpuRepo(D3Variant v) noexcept {
  switch (v) {
    case D3Variant::kLite: return "FastFlowLM/D3-Lite-NPU2";
    case D3Variant::kNano: return "FastFlowLM/D3-Nano-NPU2";
    case D3Variant::kMini: return "FastFlowLM/D3-Mini-NPU2";
    case D3Variant::kFlash: return "FastFlowLM/D3-Flash-NPU2";
    case D3Variant::kTurbo: return "FastFlowLM/D3-Turbo-NPU2";
    case D3Variant::kUltra: return "FastFlowLM/D3-Ultra-NPU2";
    case D3Variant::kEdge: return "FastFlowLM/D3-Edge-NPU2";
  }
  return "FastFlowLM/D3-Lite-NPU2";
}

/// sidecar client 介面（樁）：實作 = HTTP POST {decide_url}，body 沿用
/// flm /v1/systemone 格式。接線前不呼叫。
struct DecideRequest {
  std::string text;
  // candidate_labels 等欄位按 /v1/systemone 實測補上（TODO，一小時測試時抓包）。
};

struct DecideResult {
  bool ok{false};
  std::string label;
  float score{0.0f};
  std::string error;
};

}  // namespace mbufo::models::d3_decide
