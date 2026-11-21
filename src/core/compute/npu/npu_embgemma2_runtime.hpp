#pragma once
// Real NPU2 runtime for EmbeddingGemma 2 (24 interleaved Attention+FFN layers).
// Compiled only when MBUFO_ENABLE_NPU=ON; never included from HIP translation units.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_device_hrx.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/embeddinggemma2/engine.hpp"

namespace mbufo::core::npu {

class NpuEmbGemma2Runtime {
 public:
  static constexpr std::size_t kFixedSeqLen = 128;

  NpuEmbGemma2Runtime() = default;
  ~NpuEmbGemma2Runtime() = default;
  NpuEmbGemma2Runtime(const NpuEmbGemma2Runtime&) = delete;
  NpuEmbGemma2Runtime& operator=(const NpuEmbGemma2Runtime&) = delete;

  /// Opens HRX session, loads the 10 exported IRON kernels from `kernel_dir`,
  /// and uploads all 24 layers' transposed BF16 weights into resident BOs.
  [[nodiscard]] bool Init(const core::GgufReader& reader,
                          const models::embeddinggemma2::Config& cfg,
                          const std::string& kernel_dir, std::string* error);

  /// Runs 24 interleaved layers (Attention + GeGLU FFN on NPU2 + PLE + scalar)
  /// + final RMSNorm + mean pooling + out_proj + L2 norm for `1 <= ids <= 128`.
  [[nodiscard]] bool Forward(std::span<const std::uint32_t> ids,
                             std::vector<float>* out_embedding,
                             double* prefill_ms, std::string* error);

  [[nodiscard]] bool resident() const noexcept { return resident_; }
  [[nodiscard]] const std::string& kernel_dir() const noexcept {
    return kernel_dir_;
  }

 private:
  struct LayerResident {
    bool is_full{false};
    std::unique_ptr<hrx::ext::bo> in_norm;
    std::unique_ptr<hrx::ext::bo> w_qkv;
    std::unique_ptr<hrx::ext::bo> q_norm;
    std::unique_ptr<hrx::ext::bo> k_norm;
    std::unique_ptr<hrx::ext::bo> w_o;
    std::unique_ptr<hrx::ext::bo> post_attn;
    std::unique_ptr<hrx::ext::bo> pre_ffn;
    std::unique_ptr<hrx::ext::bo> w_gate;
    std::unique_ptr<hrx::ext::bo> w_up;
    std::unique_ptr<hrx::ext::bo> w_down;
    std::unique_ptr<hrx::ext::bo> post_ffn;

    // Host F32 mirrors for PLE block + layer_scalar.
    std::vector<float> ple_proj;  // [512, 512]
    std::vector<float> wpgate;    // [512, 512]
    std::vector<float> wpproj;    // [512, 512]
    std::vector<float> ple_post;  // [512]
    float scalar{1.0F};
  };

  bool resident_{false};
  std::string kernel_dir_;
  models::embeddinggemma2::Config cfg_{};
  std::unique_ptr<HrxSession> session_;

  const std::uint16_t* embed_table_bf16_{nullptr};
  std::size_t vocab_size_{0};
  std::vector<float> ple_norm_f32_;
  std::vector<float> final_norm_f32_;
  std::vector<float> out_proj_f32_;

  std::vector<LayerResident> layers_;

  // Static LUT & dynamic sequence masks.
  std::unique_ptr<hrx::ext::bo> lut_local_;
  std::unique_ptr<hrx::ext::bo> lut_full_;
  std::unique_ptr<hrx::ext::bo> mask_local_;
  std::unique_ptr<hrx::ext::bo> mask_full_;

  // Reusable scratch BOs sized for kFixedSeqLen = 128.
  std::unique_ptr<hrx::ext::bo> h_a_;
  std::unique_ptr<hrx::ext::bo> h_b_;
  std::unique_ptr<hrx::ext::bo> qkv_raw_l_;
  std::unique_ptr<hrx::ext::bo> qkv_raw_f_;
  std::unique_ptr<hrx::ext::bo> q_l_, k_l_, v_l_;
  std::unique_ptr<hrx::ext::bo> q_f_, k_f_, v_f_;
  std::unique_ptr<hrx::ext::bo> ctx_l_, ctx_f_;
  std::unique_ptr<hrx::ext::bo> gate_bo_, up_bo_, act_bo_;
};

}  // namespace mbufo::core::npu
