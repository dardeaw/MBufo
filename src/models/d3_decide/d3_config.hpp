#pragma once
// D3 decide 模型定義（代碼框 v1，修正：decide = D3，不是 qwen3.8-flash-next）。
//
// 來源（已上網核實）：
//   HF `vllm-sr/d3-*`：zero-shot-classification，tags decision-model + system-one
//     d3-lite  <- Qwen3.5-0.8B
//     d3-nano  <- Qwen3.5-2B
//     d3-mini  <- Qwen3.5-4B
//     d3-flash / d3-turbo / d3-ultra / d3-edge
//   NPU 現貨：HF `FastFlowLM/D3-*-NPU2`（amd-npu + q4nx），flm 側
//     `flm serve <chat> --decision d3:lite`，`POST /v1/systemone`
// License：Apache-2.0。
//
// /v1/systemone 線格式（d3_server.py 實測定義）：
//   REQ  {"questions": {id: question}, "state"?: any,
//         "images"?: [base64 dataURL], "videos"?: [...], "model"?: ignored}
//   question = {"type":"choice","criteria":{key: desc|null,...},"instructions"?}
//            | {"type":"noul","criteria"?:{"false":..,"true":..},"instructions"?}
//   RESP {"model": name, "answers": {id: answer}, "usage": {...}}
//   answer(choice) = {"type":"choice","choice": key,
//                     "probabilities": {key: share}}
//   answer(noul)   = {"type":"noul","noul": p_true}
// 錯誤一律 422 {"detail": reason}；questions 缺失/空亦 422。

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/compute/npu/npu_lane.hpp"
#include "src/core/compute/npu/npu_scheduler.hpp"

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

/// 一道題（choice/noul 二選一，與 d3_format.py 同形）。
struct DecideQuestion {
  std::string type;  // "choice" | "noul"
  // choice: key -> 說明（null 表 key 即文字）；noul: {"false":..,"true":..} 可省
  std::map<std::string, std::string> criteria;
  std::optional<std::string> instructions;
};

/// systemone 請求體（images/videos 為 base64 data URL；model 可帶可不帶）。
struct DecideRequest {
  std::map<std::string, DecideQuestion> questions;  // 必填，非空
  std::optional<std::string> state_json;            // state 原樣轉 JSON 字串
  std::vector<std::string> images;
  std::vector<std::string> videos;
};

struct DecideAnswer {
  std::string type;  // "choice" | "noul"
  std::string choice;
  std::map<std::string, float> probabilities;
  float noul{0.0f};
};

struct DecideResult {
  bool ok{false};
  std::string model;
  std::map<std::string, DecideAnswer> answers;
  std::string error;
};

enum class DecideNpuRoute : std::uint8_t { kNone = 0, kIron };

[[nodiscard]] inline DecideNpuRoute DecideNpuDecisionRoute(
    const mbufo::npu::NpuLaneConfig& config, std::size_t question_count,
    bool weights_resident = false, bool device_ready = false) noexcept {
  if (config.decide && weights_resident && device_ready && question_count > 0) {
    return DecideNpuRoute::kIron;
  }
  return DecideNpuRoute::kNone;
}

[[nodiscard]] inline mbufo::core::npu::NpuWorkResult RunDecideSync(
    mbufo::core::npu::NpuScheduler& scheduler, const DecideRequest& req,
    bool weights_resident = false,
    mbufo::core::npu::NpuWork custom_work = {}) {
  if (req.questions.empty()) {
    return mbufo::core::npu::NpuWorkResult{
        .ok = false, .error = "d3-decide empty questions"};
  }
  const bool npu_ready = !scheduler.device_lost();
  if (DecideNpuDecisionRoute(scheduler.config(), req.questions.size(),
                             weights_resident, npu_ready) !=
      DecideNpuRoute::kIron) {
    return mbufo::core::npu::NpuWorkResult{
        .ok = false, .error = "npu-disabled-or-unready"};
  }
  if (!custom_work) {
    custom_work = []() {
      return mbufo::core::npu::NpuWorkResult{
          .ok = false,
          .error = "d3-decide NPU IRON runtime bridge not bound yet"};
    };
  }
  return mbufo::core::npu::RunLaneSync(
      scheduler,
      mbufo::core::npu::NpuWorkDesc{
          .lane = mbufo::core::npu::NpuWorkLane::kDecide,
          .seq_len = req.questions.size(),
          .client = "d3-decide",
          .coalesce_key = "d3-decide"},
      std::move(custom_work));
}

[[nodiscard]] inline mbufo::core::npu::NpuWorkResult RunDecideSync(
    mbufo::core::npu::NpuScheduler* scheduler, const DecideRequest& req,
    bool weights_resident = false,
    mbufo::core::npu::NpuWork custom_work = {}) {
  if (scheduler == nullptr) {
    return mbufo::core::npu::NpuWorkResult{
        .ok = false, .error = "npu-disabled-or-unready"};
  }
  return RunDecideSync(*scheduler, req, weights_resident,
                       std::move(custom_work));
}

}  // namespace mbufo::models::d3_decide
