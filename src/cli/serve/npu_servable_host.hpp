#pragma once
// M6b L2: process-wide NPU servable hosting for the four in-process models.
// Load = engine.Init once + decoder Open(weights) + ready. Unload drops the
// weights. No inference routing here (L3); probe-on-load only proves the
// files + engine + device work (compute itself is covered by npu_*_dit/decode
// tests). Header-only, serve-local.
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/cli/serve/npu_warmup.hpp"
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
    mode_ = mode;
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
    warm_.erase(id);
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

  struct WarmInfo {
    bool warmed{false};
    double ms{0.0};
    std::string error;
  };

  /// Engine-level warmup: one M=1 GEMM per quant shape in the loaded mode.
  /// This instantiates the xclbins / HW context and primes DMA, which is the
  /// dominant cold-start cost shared by all four servables. A per-model dummy
  /// decode step is a follow-up (needs each decoder's dims).
  /// Soft-fail: a failed warm never unloads the model.
  bool Warm(const std::string& id, std::string* error) {
    namespace npu = mbufo::core::npu;
    if (!IsLoaded(id)) {
      if (error != nullptr) *error = "servable not loaded: " + id;
      return false;
    }
    const auto out = RunNpuWarmup(id, [&](std::string* e) {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const std::size_t kn : {std::size_t{256}, std::size_t{1024}}) {
        std::vector<float> w(kn * kn, 0.01f), a(kn, 0.5f), c(kn);
        npu::QuantWeightMatrix mat;
        if (!engine_.PackMatrixKN(w.data(), kn, kn, mode_, false, &mat, e)) {
          return false;
        }
        if (!engine_.Gemm(a.data(), 1, mat, c.data(), nullptr, nullptr, e)) {
          return false;
        }
      }
      return true;
    });
    std::lock_guard<std::mutex> lock(mutex_);
    WarmInfo& wi = warm_[id];
    if (!out.ran) {  // disabled via MBUFO_NPU_WARMUP=0
      wi = WarmInfo{};
      return true;
    }
    wi.warmed = out.ok;
    wi.ms = out.ms;
    wi.error = out.error;
    if (!out.ok && error != nullptr) *error = out.error;
    return out.ok;
  }

  [[nodiscard]] WarmInfo GetWarmInfo(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = warm_.find(id);
    return it == warm_.end() ? WarmInfo{} : it->second;
  }

 private:
  mutable std::mutex mutex_;
  mbufo::core::npu::NpuQuantGemmEngine engine_;
  bool engine_ready_ = false;
  mbufo::core::npu::NpuQuantMode mode_ =
      mbufo::core::npu::NpuQuantMode::kQ8Bfp16;
  std::unordered_map<std::string, WarmInfo> warm_;
  std::unique_ptr<mbufo::core::npu::ThinkerNpuDecoder> asr_;
  std::unique_ptr<mbufo::core::npu::TalkerNpuDecoder> talker_;
  std::unique_ptr<mbufo::core::npu::PredictorNpuDecoder> predictor_;
  mbufo::core::npu::TtsNpuFrameLoop frame_;
  std::unique_ptr<mbufo::core::npu::dit::FluxDitNpu> flux_;
  std::unique_ptr<mbufo::core::npu::dit::ZImageDitNpu> zimg_;
};

}  // namespace mbufo::server::npu_wire
