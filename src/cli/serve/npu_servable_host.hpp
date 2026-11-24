#pragma once
// M6b L2: process-wide NPU servable hosting for the four in-process models.
// Load = engine.Init once + decoder Open(weights) + ready. Unload drops the
// weights. No inference routing here (L3); probe-on-load only proves the
// files + engine + device work (compute itself is covered by npu_*_dit/decode
// tests). Header-only, serve-local.
#include <memory>
#include <mutex>
#include <string>

#include "src/core/compute/npu/flux_dit_npu.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_thinker_decode.hpp"
#include "src/core/compute/npu/qwen3_predictor_decode.hpp"
#include "src/core/compute/npu/qwen3_talker_decode.hpp"
#include "src/core/compute/npu/qwen3_tts_frame.hpp"
#include "src/core/compute/npu/zimage_dit_npu.hpp"

namespace mbufo::server::npu_wire {

class NpuServableHost {
 public:
  /// Open weights for `id` (asr thinker / tts stack / flux / zimage).
  /// Idempotent: already-open is success. Mode "q8" (default) or "q4".
  bool Load(const std::string& id, const std::string& artifact,
            const std::string& quant, std::string* error) {
    namespace npu = mbufo::core::npu;
    npu::NpuQuantMode mode = npu::NpuQuantMode::kQ8Bfp16;
    if (quant == "q4" || quant == "Q4") {
      mode = npu::NpuQuantMode::kQ4Q4nx;
    } else if (!quant.empty() && quant != "q8" && quant != "Q8") {
      if (error != nullptr) *error = "npu_quant must be q8 or q4";
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!engine_ready_) {
      if (!engine_.Init(
              "/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm", error)) {
        return false;
      }
      engine_ready_ = true;
    }
    if (id == "qwen3-asr-1.7b") {
      if (asr_ != nullptr) return true;
      auto d = std::make_unique<npu::ThinkerNpuDecoder>();
      if (!d->Open(artifact, &engine_, mode, {}, error)) return false;
      asr_ = std::move(d);
      return true;
    }
    if (id == "qwen3-tts-12hz-1.7b") {
      if (talker_ != nullptr && predictor_ != nullptr) return true;
      auto t = std::make_unique<npu::TalkerNpuDecoder>();
      if (!t->Open(artifact, &engine_, mode, {}, error)) return false;
      auto p = std::make_unique<npu::PredictorNpuDecoder>();
      if (!p->Open(artifact, &engine_, mode, {}, error)) return false;
      frame_.Bind(t.get(), p.get());
      talker_ = std::move(t);
      predictor_ = std::move(p);
      return true;
    }
    if (id == "flux2-klein-4b") {
      if (flux_ != nullptr) return true;
      auto d = std::make_unique<npu::dit::FluxDitNpu>();
      if (!d->Open(artifact, &engine_, mode, {}, error)) return false;
      flux_ = std::move(d);
      return true;
    }
    if (id == "z-image-turbo") {
      if (zimg_ != nullptr) return true;
      auto d = std::make_unique<npu::dit::ZImageDitNpu>();
      if (!d->Open(artifact, &engine_, mode, {}, error)) return false;
      zimg_ = std::move(d);
      return true;
    }
    if (error != nullptr) *error = "unknown NPU servable id: " + id;
    return false;
  }

  bool Unload(const std::string& id, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id == "qwen3-asr-1.7b") {
      asr_.reset();
      return true;
    }
    if (id == "qwen3-tts-12hz-1.7b") {
      frame_ = mbufo::core::npu::TtsNpuFrameLoop();
      predictor_.reset();
      talker_.reset();
      return true;
    }
    if (id == "flux2-klein-4b") {
      flux_.reset();
      return true;
    }
    if (id == "z-image-turbo") {
      zimg_.reset();
      return true;
    }
    if (error != nullptr) *error = "unknown NPU servable id: " + id;
    return false;
  }

  [[nodiscard]] bool IsLoaded(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id == "qwen3-asr-1.7b") return asr_ != nullptr;
    if (id == "qwen3-tts-12hz-1.7b")
      return talker_ != nullptr && predictor_ != nullptr;
    if (id == "flux2-klein-4b") return flux_ != nullptr;
    if (id == "z-image-turbo") return zimg_ != nullptr;
    return false;
  }

 private:
  mutable std::mutex mutex_;
  mbufo::core::npu::NpuQuantGemmEngine engine_;
  bool engine_ready_ = false;
  std::unique_ptr<mbufo::core::npu::ThinkerNpuDecoder> asr_;
  std::unique_ptr<mbufo::core::npu::TalkerNpuDecoder> talker_;
  std::unique_ptr<mbufo::core::npu::PredictorNpuDecoder> predictor_;
  mbufo::core::npu::TtsNpuFrameLoop frame_;
  std::unique_ptr<mbufo::core::npu::dit::FluxDitNpu> flux_;
  std::unique_ptr<mbufo::core::npu::dit::ZImageDitNpu> zimg_;
};

}  // namespace mbufo::server::npu_wire
