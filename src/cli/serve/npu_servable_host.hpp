#pragma once
// M6b L2: process-wide NPU servable hosting for the four in-process models
// (ASR thinker, TTS talker+predictor, FLUX, Z-Image).
//
// Every model is an INpuServable (npu_servable.hpp): Load = Open (engine init
// once + decoder weights) + Warm, Unload = Close (drop weights) + heap trim,
// and each one is published in NpuServableRegistry so /admin/models reports
// the same lifecycle for all NPU models. No inference routing here (L3).
// Header-only, serve-local.
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "src/cli/serve/npu_warmup.hpp"
#include "src/core/compute/npu/flux_dit_npu.hpp"
#include "src/core/compute/npu/npu_op_graph.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_servable.hpp"
#include "src/core/compute/npu/npu_thinker_decode.hpp"
#include "src/core/compute/npu/qwen3_predictor_decode.hpp"
#include "src/core/compute/npu/qwen3_talker_decode.hpp"
#include "src/core/compute/npu/qwen3_tts_frame.hpp"
#include "src/core/compute/npu/zimage_dit_npu.hpp"

namespace mbufo::server::npu_wire {

class NpuServableHost {
public:
  /// Open weights for `id` (asr thinker / tts stack / flux / zimage) and warm
  /// them through the INpuServable lifecycle. Idempotent: an already-Ready
  /// servable is success. Mode "q8" (default) or "q4". A failed warmup is
  /// best-effort (Ready, warm=false); a failed open returns false.
  bool Load(const std::string& id, const std::string& artifact,
            const std::string& quant, std::string* error) {
    namespace npu = mbufo::core::npu;
    npu::NpuQuantMode mode = npu::NpuQuantMode::kQ8Bfp16;
    if (quant == "q4" || quant == "Q4") {
      mode = npu::NpuQuantMode::kQ4Q4nx;
    } else if (!quant.empty() && quant != "q8" && quant != "Q8") {
      if (error != nullptr)
        *error = "npu_quant must be q8 or q4";
      return false;
    }
    if (!KnownId(id)) {
      if (error != nullptr)
        *error = "unknown NPU servable id: " + id;
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto it = servables_.find(id);
        it != servables_.end() && it->second->Ready()) {
      return true;
    }
    mode_ = mode;
    // One process-wide quant engine (one HRX session, one scratch-BO set)
    // shared with the op-graph models; dropped again when the last servable
    // here is ejected, so it never competes with them for buffer objects.
    if (engine_ == nullptr) {
      engine_ = npu::graph::SharedQuantEngine(error);
      if (engine_ == nullptr)
        return false;
    }
    auto s = std::make_shared<npu::NpuFnServable>(
        id,
        [this, id, mode](const npu::NpuLoadArgs& a, std::string* e) {
          return OpenImpl(id, a.artifact, mode, e);
        },
        [this, id](std::string* e) { return WarmImpl(id, e); },
        [this, id]() { CloseImpl(id); });
    npu::NpuLoadArgs args;
    args.artifact = artifact;
    args.quant = quant;
    args.warmup = true;
    args.warmup_policy = npu::NpuWarmupPolicy::kBestEffort;
    if (!s->Load(args, error)) {
      if (servables_.empty())
        engine_.reset();
      return false;
    }
    const auto st = s->Status();
    Logger::Info("loader",
                 "event=npu_servable model=" + id +
                     " state=" + npu::NpuServableStateName(st.state) +
                     " load_ms=" + std::to_string(st.load_ms) +
                     " warm=" + (st.warmed ? "true" : "false") +
                     " warmup_ms=" + std::to_string(st.warmup_ms) +
                     (st.last_error.empty() ? "" : " error=" + st.last_error));
    servables_[id] = s;
    npu::NpuServableRegistry::Global().Put(id, s);
    return true;
  }

  /// Eject: the servable's Close drops the weights, Unload trims the heap, and
  /// the registry entry is removed so admin stops reporting it.
  bool Unload(const std::string& id, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = servables_.find(id);
    if (it == servables_.end()) {
      if (!KnownId(id)) {
        if (error != nullptr)
          *error = "unknown NPU servable id: " + id;
        return false;
      }
      return true;  // known but not loaded: nothing to release
    }
    it->second->Unload();
    servables_.erase(it);
    if (servables_.empty())
      engine_.reset();  // decoders are closed; release the shared engine
    mbufo::core::npu::NpuServableRegistry::Global().Remove(id);
    return true;
  }

  [[nodiscard]] bool IsLoaded(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = servables_.find(id);
    return it != servables_.end() && it->second->Ready();
  }

  struct WarmInfo {
    bool warmed{false};
    double ms{0.0};
    std::string error;
  };

  /// Re-run the warmup of a loaded servable. Soft-fail: never unloads.
  bool Warm(const std::string& id, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = servables_.find(id);
    if (it == servables_.end()) {
      if (error != nullptr)
        *error = "servable not loaded: " + id;
      return false;
    }
    return it->second->Rewarm(error);
  }

  [[nodiscard]] WarmInfo GetWarmInfo(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = servables_.find(id);
    if (it == servables_.end())
      return WarmInfo{};
    const auto st = it->second->Status();
    return WarmInfo{st.warmed, st.warmup_ms, st.last_error};
  }

private:
  static bool KnownId(const std::string& id) {
    return id == "qwen3-asr-1.7b" || id == "qwen3-tts-12hz-1.7b" ||
           id == "flux2-klein-4b" || id == "z-image-turbo";
  }

  // All *Impl run with mutex_ held (called from Load/Unload/Warm).
  bool OpenImpl(const std::string& id, const std::string& artifact,
                mbufo::core::npu::NpuQuantMode mode, std::string* error) {
    namespace npu = mbufo::core::npu;
    if (id == "qwen3-asr-1.7b") {
      auto d = std::make_unique<npu::ThinkerNpuDecoder>();
      if (!d->Open(artifact, engine_.get(), mode, {}, error))
        return false;
      asr_ = std::move(d);
      return true;
    }
    if (id == "qwen3-tts-12hz-1.7b") {
      auto t = std::make_unique<npu::TalkerNpuDecoder>();
      if (!t->Open(artifact, engine_.get(), mode, {}, error))
        return false;
      auto p = std::make_unique<npu::PredictorNpuDecoder>();
      if (!p->Open(artifact, engine_.get(), mode, {}, error))
        return false;
      frame_.Bind(t.get(), p.get());
      talker_ = std::move(t);
      predictor_ = std::move(p);
      return true;
    }
    if (id == "flux2-klein-4b") {
      auto d = std::make_unique<npu::dit::FluxDitNpu>();
      if (!d->Open(artifact, engine_.get(), mode, {}, error))
        return false;
      flux_ = std::move(d);
      return true;
    }
    auto d = std::make_unique<npu::dit::ZImageDitNpu>();
    if (!d->Open(artifact, engine_.get(), mode, {}, error))
      return false;
    zimg_ = std::move(d);
    return true;
  }

  void CloseImpl(const std::string& id) {
    if (id == "qwen3-asr-1.7b") {
      asr_.reset();
    } else if (id == "qwen3-tts-12hz-1.7b") {
      frame_ = mbufo::core::npu::TtsNpuFrameLoop();
      predictor_.reset();
      talker_.reset();
    } else if (id == "flux2-klein-4b") {
      flux_.reset();
    } else if (id == "z-image-turbo") {
      zimg_.reset();
    }
  }

  /// Engine-level warmup (one M=1 GEMM per quant shape: instantiates xclbins
  /// and the HW context) plus, for ASR and TTS, one dummy prefill + decode
  /// step + LM-head pass on the real decoder; FLUX / Z-Image run one reduced
  /// Euler step (1+1 blocks, 8x8 latent, one text token).
  bool WarmImpl(const std::string& id, std::string* e) {
    namespace npu = mbufo::core::npu;
    for (const std::size_t kn : {std::size_t{256}, std::size_t{1024}}) {
      std::vector<float> w(kn * kn, 0.01f), a(kn, 0.5f), c(kn);
      npu::QuantWeightMatrix mat;
      if (!engine_->PackMatrixKN(w.data(), kn, kn, mode_, false, &mat, e)) {
        return false;
      }
      if (!engine_->Gemm(a.data(), 1, mat, c.data(), nullptr, nullptr, e)) {
        return false;
      }
    }
    // Per-model dummy decode: drives the real decoder kernels (KV cache,
    // attention, LM head). The next real Prefill resets the decoder position.
    if (id == "qwen3-asr-1.7b" && asr_ != nullptr) {
      const std::size_t h = asr_->config().hidden;
      const std::vector<float> row(h, 0.01F);
      std::vector<float> hid, next;
      std::uint32_t top1 = 0;
      return asr_->Prefill(row, 1, &hid, nullptr, e) &&
             asr_->DecodeStep(row, &next, nullptr, e) &&
             asr_->LogitsTop1(next, &top1, nullptr, e);
    }
    if (id == "qwen3-tts-12hz-1.7b" && talker_ != nullptr) {
      const std::size_t h = talker_->config().hidden;
      const std::vector<float> row(h, 0.01F);
      std::vector<float> hid, next;
      std::uint32_t top1 = 0;
      return talker_->Prefill(row, 1, &hid, nullptr, e) &&
             talker_->DecodeStep(row, &next, nullptr, e) &&
             talker_->CodecTop1(next, &top1, nullptr, e);
    }
    if (id == "flux2-klein-4b" && flux_ != nullptr) {
      const std::size_t St = 1;
      const std::vector<float> cond(7680, 0.01F);
      std::vector<float> txt;
      std::vector<float> z(64 * 128, 0.05F);
      return flux_->EncodeContext(cond, St, &txt, nullptr, e) &&
             flux_->FullStep(&z, 64, txt, St, 0.5F, 0.1F, 1, 1, nullptr, e);
    }
    if (id == "z-image-turbo" && zimg_ != nullptr) {
      const std::vector<float> te(2560, 0.01F);
      std::vector<float> cap;
      std::vector<float> z(64 * 64, 0.05F);
      return zimg_->CapEmbed(te, 1, &cap, nullptr, e) &&
             zimg_->FullStep(&z, 64, cap, 1, 0.5F, 0.1F, 1, 1, nullptr, e);
    }
    return true;
  }

  mutable std::mutex mutex_;
  std::shared_ptr<mbufo::core::npu::NpuQuantGemmEngine> engine_;
  mbufo::core::npu::NpuQuantMode mode_ =
      mbufo::core::npu::NpuQuantMode::kQ8Bfp16;
  std::map<std::string, std::shared_ptr<mbufo::core::npu::NpuFnServable>>
      servables_;
  std::unique_ptr<mbufo::core::npu::ThinkerNpuDecoder> asr_;
  std::unique_ptr<mbufo::core::npu::TalkerNpuDecoder> talker_;
  std::unique_ptr<mbufo::core::npu::PredictorNpuDecoder> predictor_;
  mbufo::core::npu::TtsNpuFrameLoop frame_;
  std::unique_ptr<mbufo::core::npu::dit::FluxDitNpu> flux_;
  std::unique_ptr<mbufo::core::npu::dit::ZImageDitNpu> zimg_;
};

}  // namespace mbufo::server::npu_wire