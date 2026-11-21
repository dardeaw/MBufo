#pragma once
// Real NPU2 runtime for EmbeddingGemma 2 (24 interleaved Attention+FFN layers).
// Supports:
//   - BF16 10-kernel chain (`Init`)
//   - Q8 (`bfp16ebs8` W8A16) and Q4 (`q4nx` W4A16 on-core dequant) via
//     shared `NpuQuantGemmEngine` (`InitQuant`).
// Compiled only when MBUFO_ENABLE_NPU=ON; never included from HIP translation units.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_device_hrx.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
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

  /// Opens HRX session, loads the 10 exported BF16 IRON kernels from `kernel_dir`,
  /// and uploads all 24 layers' transposed BF16 weights into resident BOs.
  [[nodiscard]] bool Init(const core::GgufReader& reader,
                          const models::embeddinggemma2::Config& cfg,
                          const std::string& kernel_dir, std::string* error);

  /// Initializes 24-layer EmbeddingGemma 2 in Q8 (`kQ8Bfp16`) or Q4 (`kQ4Q4nx`)
  /// using the shared 4-kernel `NpuQuantGemmEngine`.
  [[nodiscard]] bool InitQuant(
      const core::GgufReader& reader,
      const models::embeddinggemma2::Config& cfg, NpuQuantMode mode,
      const std::string& quant_kernel_dir =
          "/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm",
      std::string* error = nullptr);

  /// Runs 24 interleaved layers (Attention + GeGLU FFN on NPU2 + PLE + scalar)
  /// + final RMSNorm + mean pooling + out_proj + L2 norm.
  [[nodiscard]] bool Forward(std::span<const std::uint32_t> ids,
                             std::vector<float>* out_embedding,
                             double* prefill_ms, std::string* error);

  [[nodiscard]] bool resident() const noexcept { return resident_; }
  [[nodiscard]] NpuQuantMode quant_mode() const noexcept { return quant_mode_; }
  [[nodiscard]] const std::string& kernel_dir() const noexcept {
    return kernel_dir_;
  }
  [[nodiscard]] std::size_t total_quant_weight_bytes() const noexcept {
    return total_quant_bytes_;
  }

 private:
  [[nodiscard]] bool ForwardQuant(std::span<const std::uint32_t> ids,
                                  std::vector<float>* out_embedding,
                                  double* prefill_ms, std::string* error);

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

    // Host F32 norms + PLE mirrors + layer_scalar.
    std::vector<float> in_norm_f32;
    std::vector<float> q_norm_f32;
    std::vector<float> k_norm_f32;
    std::vector<float> post_attn_f32;
    std::vector<float> pre_ffn_f32;
    std::vector<float> post_ffn_f32;

    std::vector<float> ple_proj;  // [512, 512]
    std::vector<float> wpgate;    // [512, 512]
    std::vector<float> wpproj;    // [512, 512]
    std::vector<float> ple_post;  // [512]
    float scalar{1.0F};

    // Quantized NPU2 weight matrices (used when quant_mode_ != kBf16).
    QuantWeightMatrix q_quant;
    QuantWeightMatrix k_quant;
    QuantWeightMatrix v_quant;
    QuantWeightMatrix o_quant;
    QuantWeightMatrix gate_quant;
    QuantWeightMatrix up_quant;
    QuantWeightMatrix down_quant;
    QuantWeightMatrix ple_proj_quant;
    QuantWeightMatrix wpgate_quant;
    QuantWeightMatrix wpproj_quant;
  };

  bool resident_{false};
  NpuQuantMode quant_mode_{NpuQuantMode::kBf16};
  std::size_t total_quant_bytes_{0};
  std::string kernel_dir_;
  models::embeddinggemma2::Config cfg_{};
  std::unique_ptr<HrxSession> session_;
  std::unique_ptr<NpuQuantGemmEngine> quant_engine_;

  const std::uint16_t* embed_table_bf16_{nullptr};
  std::size_t vocab_size_{0};
  std::vector<float> ple_norm_f32_;
  std::vector<float> final_norm_f32_;
  std::vector<float> out_proj_f32_;

  std::vector<LayerResident> layers_;

  // Static LUT & dynamic sequence masks (BF16 mode).
  std::unique_ptr<hrx::ext::bo> lut_local_;
  std::unique_ptr<hrx::ext::bo> lut_full_;
  std::unique_ptr<hrx::ext::bo> mask_local_;
  std::unique_ptr<hrx::ext::bo> mask_full_;

  // Reusable scratch BOs sized for kFixedSeqLen = 128 (BF16 mode).
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
