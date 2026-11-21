#pragma once
// Universal NPU2 Shared Quantized GEMM Engine (Q8 bfp16ebs8 & Q4 q4nx).
//
// Powers all 5 models on XDNA2 (AIE2P) within 4 shared hwctx slots:
//   1. EmbeddingGemma 2 (24L)
//   2. Qwen3-TTS 12Hz 1.7B (Talker 28L + CodePredictor 5L + SpeechTokenizer 8L)
//   3. Qwen3-ASR 1.7B (AudioEncoder 24L + Thinker 28L)
//   4. FLUX.2-klein-4B (DiT 5+20L + Qwen3-4B TE 36L)
//   5. Z-Image-Turbo (DiT 30+4L + Qwen3-4B TE 36L)
//
// Compiled only when MBUFO_ENABLE_NPU=ON.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_device_hrx.hpp"

namespace mbufo::core::npu {

enum class NpuQuantMode : std::uint8_t {
  kBf16 = 0,
  kQ8Bfp16 = 1,  ///< W8A16 bfp16ebs8 (9 bytes / 8 weights, 8x8x8 BFP16 MAC)
  kQ4Q4nx = 2,   ///< W4A16 q4nx group=32 -> on-core L1 q4nx_dequant -> mm_bfp
};

[[nodiscard]] inline const char* NpuQuantModeName(NpuQuantMode mode) noexcept {
  switch (mode) {
    case NpuQuantMode::kBf16:
      return "bf16";
    case NpuQuantMode::kQ8Bfp16:
      return "q8_bfp16";
    case NpuQuantMode::kQ4Q4nx:
      return "q4_q4nx";
  }
  return "unknown";
}

/// Constants matching e10_quant_gemm.py.
inline constexpr std::size_t kQuantGemmM = 64;
inline constexpr std::size_t kQ8TileBytes = 64 * 64 * 9 / 8;             // 4608
inline constexpr std::size_t kQ4TileBytes = 64 * 64 / 2 + 4 * (64 * 64 / 32);  // 2560
inline constexpr std::size_t kQ8Block256Bytes = 16 * kQ8TileBytes;       // 73,728
inline constexpr std::size_t kQ8Block1024Bytes = 256 * kQ8TileBytes;     // 1,179,648
inline constexpr std::size_t kQ4Block256Bytes = 16 * kQ4TileBytes;       // 40,960
inline constexpr std::size_t kQ4Block1024Bytes = 256 * kQ4TileBytes;     // 655,360

/// Pack one 64x64 tile (given as row-major W_kn[k * ld_n + n]) into Q8 bfp16ebs8 (4608B).
void PackTileQ8Bfp16KN(const float* w_kn, std::size_t ld_n,
                       std::uint8_t* out_4608) noexcept;

/// Pack one 64x64 tile (given as row-major W_kn[k * ld_n + n]) into Q4 q4nx (2560B).
void PackTileQ4Q4nxKN(const float* w_kn, std::size_t ld_n,
                      std::uint8_t* out_2560) noexcept;

/// Reference host dequantizer for one Q4 q4nx 64x64 tile back to float W_kn[64*64].
void DequantTileQ4Q4nxToFloatKN(const std::uint8_t* in_2560,
                                float* out_kn_64x64) noexcept;

struct QuantWeightBlock {
  std::size_t k0{0};
  std::size_t n0{0};
  std::size_t block_dim{256};  ///< 1024 or 256
  std::vector<std::uint8_t> host_bytes;
  std::unique_ptr<hrx::ext::bo> bo;  ///< Optional resident HRX BO
};

struct QuantWeightMatrix {
  NpuQuantMode mode{NpuQuantMode::kQ8Bfp16};
  std::size_t K{0};
  std::size_t N{0};
  std::size_t total_packed_bytes{0};
  std::vector<QuantWeightBlock> blocks;
};

class NpuQuantGemmEngine {
 public:
  static constexpr std::size_t kMaxK1024Slices = 16;  // up to K = 16384
  static constexpr std::size_t kMaxK256Slices = 64;   // up to K = 16384
  static constexpr std::size_t kBatch1024 = 24;
  static constexpr std::size_t kBatch256 = 48;

  NpuQuantGemmEngine() = default;
  ~NpuQuantGemmEngine() = default;
  NpuQuantGemmEngine(const NpuQuantGemmEngine&) = delete;
  NpuQuantGemmEngine& operator=(const NpuQuantGemmEngine&) = delete;

  /// Opens (or shares) an HrxSession and loads the 4 shared quantized GEMM
  /// kernels (`q8_256x256`, `q8_1024x1024`, `q4_256x256`, `q4_1024x1024`).
  [[nodiscard]] bool Init(
      const std::string& kernel_dir =
          "/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm",
      std::string* error = nullptr);

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] HrxSession* session() noexcept { return session_.get(); }

  /// Quantize and pack row-major `W_kn[K, N]` (`C = A[M, K] @ W_kn[K, N]`).
  /// Requires `K % 256 == 0 && N % 256 == 0`.
  /// When `make_bo_resident == true`, allocates persistent HRX BOs per block.
  [[nodiscard]] bool PackMatrixKN(const float* w_kn, std::size_t K,
                                  std::size_t N, NpuQuantMode mode,
                                  bool make_bo_resident,
                                  QuantWeightMatrix* out,
                                  std::string* error = nullptr);

  /// Quantize and pack PyTorch/GGUF row-major `W_nk[N, K]` (`C = A[M, K] @ W_nk^T`).
  /// Requires `K % 256 == 0 && N % 256 == 0`.
  [[nodiscard]] bool PackMatrixNK(const float* w_nk, std::size_t N,
                                  std::size_t K, NpuQuantMode mode,
                                  bool make_bo_resident,
                                  QuantWeightMatrix* out,
                                  std::string* error = nullptr);

  /// Direct BF16 `W_nk[N, K]` packer (avoids allocating a full FP32 matrix copy
  /// when streaming multi-GB safetensors/GGUF weights).
  [[nodiscard]] bool PackMatrixBf16NK(const std::uint16_t* w_bf16_nk,
                                      std::size_t N, std::size_t K,
                                      NpuQuantMode mode, bool make_bo_resident,
                                      QuantWeightMatrix* out,
                                      std::string* error = nullptr);

  /// Execute `C[M, N] = A[M, K] @ W[K, N]` (+ optional `bias[N]`) on NPU2 silicon.
  /// Handles arbitrary `M >= 1` in 64-row chunks and accumulates K-blocks in FP32.
  [[nodiscard]] bool Gemm(const float* A, std::size_t M,
                          const QuantWeightMatrix& W, float* C,
                          const float* bias = nullptr,
                          double* npu_ms_accum = nullptr,
                          std::string* error = nullptr);

 private:
  bool MaterializeBlocks(QuantWeightMatrix* out, bool make_bo_resident,
                         std::string* error);

  bool ready_{false};
  std::string kernel_dir_;
  std::unique_ptr<HrxSession> session_;

  std::array<std::unique_ptr<hrx::ext::bo>, kMaxK1024Slices> a_bo_1024_;
  std::array<std::unique_ptr<hrx::ext::bo>, kMaxK256Slices> a_bo_256_;

  std::array<std::unique_ptr<hrx::ext::bo>, kBatch1024> b_q8_stage_1024_;
  std::array<std::unique_ptr<hrx::ext::bo>, kBatch1024> b_q4_stage_1024_;
  std::array<std::unique_ptr<hrx::ext::bo>, kBatch1024> c_bo_1024_;

  std::array<std::unique_ptr<hrx::ext::bo>, kBatch256> b_q8_stage_256_;
  std::array<std::unique_ptr<hrx::ext::bo>, kBatch256> b_q4_stage_256_;
  std::array<std::unique_ptr<hrx::ext::bo>, kBatch256> c_bo_256_;
};

}  // namespace mbufo::core::npu
