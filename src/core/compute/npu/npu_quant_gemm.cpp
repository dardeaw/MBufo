#include "src/core/compute/npu/npu_quant_gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>

#include "src/core/compute/npu/npu_pooling.hpp"
#include "src/core/quant/parallel.hpp"

namespace mbufo::core::npu {
namespace {

inline void EncodeBfp16Block8(const float vals[8],
                              std::uint8_t out[9]) noexcept {
  std::uint32_t bits[8];
  bool sign[8];
  std::uint32_t exp[8];
  std::uint32_t mant[8];
  std::uint32_t max_exp = 0;
  for (int i = 0; i < 8; ++i) {
    const float fv = std::isfinite(vals[i]) ? vals[i] : 0.0f;
    std::memcpy(&bits[i], &fv, sizeof(std::uint32_t));
    sign[i] = (bits[i] >> 31u) != 0u;
    exp[i] = (bits[i] >> 23u) & 0xFFu;
    mant[i] = (bits[i] & 0x7FFFFFu) | ((exp[i] != 0u) ? 0x800000u : 0u);
    if (exp[i] > max_exp) max_exp = exp[i];
  }
  out[0] = static_cast<std::uint8_t>(max_exp);
  for (int i = 0; i < 8; ++i) {
    const std::uint32_t shift = max_exp - exp[i];
    std::int8_t m = 0;
    if (shift >= 32u) {
      m = sign[i] ? static_cast<std::int8_t>(-1) : static_cast<std::int8_t>(0);
    } else {
      const std::uint32_t m32 =
          sign[i] ? static_cast<std::uint32_t>(-static_cast<std::int32_t>(mant[i]))
                  : mant[i];
      const std::int8_t base_m =
          static_cast<std::int8_t>((m32 >> 17u) & 0xFFu);
      const std::int32_t shifted =
          static_cast<std::int32_t>(base_m) >> std::min<std::uint32_t>(shift, 31u);
      m = static_cast<std::int8_t>(shifted);
    }
    out[1 + i] = static_cast<std::uint8_t>(m);
  }
}

void BuildBlockSchedule(std::size_t K, std::size_t N, NpuQuantMode mode,
                        std::vector<QuantWeightBlock>* blocks,
                        std::size_t* total_bytes) {
  blocks->clear();
  *total_bytes = 0;
  const std::size_t k1024 = (K / 1024u) * 1024u;
  const std::size_t n1024 = (N / 1024u) * 1024u;
  const std::size_t b1024_bytes =
      (mode == NpuQuantMode::kQ8Bfp16) ? kQ8Block1024Bytes : kQ4Block1024Bytes;
  const std::size_t b256_bytes =
      (mode == NpuQuantMode::kQ8Bfp16) ? kQ8Block256Bytes : kQ4Block256Bytes;

  for (std::size_t k0 = 0; k0 < k1024; k0 += 1024u) {
    for (std::size_t n0 = 0; n0 < n1024; n0 += 1024u) {
      QuantWeightBlock blk;
      blk.k0 = k0;
      blk.n0 = n0;
      blk.block_dim = 1024u;
      blk.host_bytes.resize(b1024_bytes);
      *total_bytes += b1024_bytes;
      blocks->push_back(std::move(blk));
    }
    for (std::size_t n0 = n1024; n0 < N; n0 += 256u) {
      for (std::size_t ks = 0; ks < 1024u; ks += 256u) {
        QuantWeightBlock blk;
        blk.k0 = k0 + ks;
        blk.n0 = n0;
        blk.block_dim = 256u;
        blk.host_bytes.resize(b256_bytes);
        *total_bytes += b256_bytes;
        blocks->push_back(std::move(blk));
      }
    }
  }
  for (std::size_t k0 = k1024; k0 < K; k0 += 256u) {
    for (std::size_t n0 = 0; n0 < N; n0 += 256u) {
      QuantWeightBlock blk;
      blk.k0 = k0;
      blk.n0 = n0;
      blk.block_dim = 256u;
      blk.host_bytes.resize(b256_bytes);
      *total_bytes += b256_bytes;
      blocks->push_back(std::move(blk));
    }
  }
}

}  // namespace

void PackTileQ8Bfp16KN(const float* w_kn, std::size_t ld_n,
                       std::uint8_t* out_4608) noexcept {
  for (int n8 = 0; n8 < 8; ++n8) {
    for (int ks = 0; ks < 8; ++ks) {
      for (int t = 0; t < 8; ++t) {
        float vals[8];
        const std::size_t n = static_cast<std::size_t>(n8 * 8 + t);
        for (int s = 0; s < 8; ++s) {
          const std::size_t k = static_cast<std::size_t>(ks * 8 + s);
          vals[s] = w_kn[k * ld_n + n];
        }
        EncodeBfp16Block8(vals, out_4608 + ((n8 * 8 + ks) * 8 + t) * 9);
      }
    }
  }
}

void PackTileQ4Q4nxKN(const float* w_kn, std::size_t ld_n,
                      std::uint8_t* out_2560) noexcept {
  std::uint16_t scales[2][64];
  std::uint16_t mins[2][64];
  std::uint8_t q[64][64];

  for (int kg = 0; kg < 2; ++kg) {
    const int k_base = kg * 32;
    for (int n = 0; n < 64; ++n) {
      float w_min = w_kn[static_cast<std::size_t>(k_base) * ld_n + n];
      float w_max = w_min;
      for (int k = k_base + 1; k < k_base + 32; ++k) {
        const float v = w_kn[static_cast<std::size_t>(k) * ld_n + n];
        if (v < w_min) w_min = v;
        if (v > w_max) w_max = v;
      }
      const std::uint16_t mn_bf = FloatToBf16Rne(w_min);
      const float mn_f = Bf16BitsToFloat(mn_bf);
      const std::uint16_t sc_bf = FloatToBf16Rne((w_max - mn_f) / 15.0f);
      const float sc_f = Bf16BitsToFloat(sc_bf);
      scales[kg][n] = sc_bf;
      mins[kg][n] = mn_bf;

      for (int k = k_base; k < k_base + 32; ++k) {
        const float v = w_kn[static_cast<std::size_t>(k) * ld_n + n];
        int qi = 0;
        if (sc_f > 0.0f) {
          qi = static_cast<int>(std::rint((v - mn_f) / sc_f));
        }
        q[k][n] = static_cast<std::uint8_t>(std::clamp(qi, 0, 15));
      }
    }
  }

  std::memcpy(out_2560, scales, 256);
  std::memcpy(out_2560 + 256, mins, 256);
  std::uint8_t* p = out_2560 + 512;
  for (int nb = 0; nb < 4; ++nb) {
    for (int k = 0; k < 64; ++k) {
      for (int pair = 0; pair < 8; ++pair) {
        const std::uint8_t lo = q[k][nb * 16 + 2 * pair] & 0x0Fu;
        const std::uint8_t hi = q[k][nb * 16 + 2 * pair + 1] & 0x0Fu;
        *p++ = static_cast<std::uint8_t>(lo | (hi << 4u));
      }
    }
  }
}

void DequantTileQ4Q4nxToFloatKN(const std::uint8_t* in_2560,
                                float* out_kn_64x64) noexcept {
  std::uint16_t scales[2][64];
  std::uint16_t mins[2][64];
  std::memcpy(scales, in_2560, 256);
  std::memcpy(mins, in_2560 + 256, 256);
  const std::uint8_t* p = in_2560 + 512;
  for (int nb = 0; nb < 4; ++nb) {
    for (int k = 0; k < 64; ++k) {
      const int kg = k / 32;
      for (int pair = 0; pair < 8; ++pair) {
        const std::uint8_t byte = *p++;
        const int n0 = nb * 16 + 2 * pair;
        const int n1 = n0 + 1;
        const float q0 = static_cast<float>(byte & 0x0Fu);
        const float q1 = static_cast<float>((byte >> 4u) & 0x0Fu);
        out_kn_64x64[k * 64 + n0] =
            Bf16BitsToFloat(mins[kg][n0]) + Bf16BitsToFloat(scales[kg][n0]) * q0;
        out_kn_64x64[k * 64 + n1] =
            Bf16BitsToFloat(mins[kg][n1]) + Bf16BitsToFloat(scales[kg][n1]) * q1;
      }
    }
  }
}

bool NpuQuantGemmEngine::Init(const std::string& kernel_dir,
                              std::string* error) {
  ready_ = false;
  std::string dir = kernel_dir;
  if (dir.empty()) {
    if (const char* env = std::getenv("MBUFO_NPU_QUANT_KERNELS");
        env != nullptr && env[0] != '\0') {
      dir = env;
    } else {
      dir = "/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm";
    }
  }
  kernel_dir_ = dir;
  session_ = HrxSession::Open(0, error);
  if (session_ == nullptr) return false;

  const char* kNames[4] = {"q8_256x256", "q8_1024x1024", "q4_256x256",
                           "q4_1024x1024"};
  for (const char* name : kNames) {
    const std::string xcl = dir + "/" + name + ".xclbin";
    const std::string ins = dir + "/" + name + ".insts.bin";
    if (!session_->LoadIronKernel(name, xcl, ins, error)) return false;
  }

  for (std::size_t i = 0; i < kMaxK1024Slices; ++i) {
    a_bo_1024_[i] = session_->AllocBo(kQuantGemmM * 1024u * sizeof(std::uint16_t), error);
    if (!a_bo_1024_[i]) return false;
  }
  for (std::size_t i = 0; i < kMaxK256Slices; ++i) {
    a_bo_256_[i] = session_->AllocBo(kQuantGemmM * 256u * sizeof(std::uint16_t), error);
    if (!a_bo_256_[i]) return false;
  }
  for (std::size_t i = 0; i < kBatch1024; ++i) {
    b_q8_stage_1024_[i] = session_->AllocBo(kQ8Block1024Bytes, error);
    b_q4_stage_1024_[i] = session_->AllocBo(kQ4Block1024Bytes, error);
    c_bo_1024_[i] = session_->AllocBo(kQuantGemmM * 1024u * sizeof(std::uint16_t), error);
    if (!b_q8_stage_1024_[i] || !b_q4_stage_1024_[i] || !c_bo_1024_[i]) {
      return false;
    }
  }
  for (std::size_t i = 0; i < kBatch256; ++i) {
    b_q8_stage_256_[i] = session_->AllocBo(kQ8Block256Bytes, error);
    b_q4_stage_256_[i] = session_->AllocBo(kQ4Block256Bytes, error);
    c_bo_256_[i] = session_->AllocBo(kQuantGemmM * 256u * sizeof(std::uint16_t), error);
    if (!b_q8_stage_256_[i] || !b_q4_stage_256_[i] || !c_bo_256_[i]) {
      return false;
    }
  }

  ready_ = true;
  return true;
}

bool NpuQuantGemmEngine::MaterializeBlocks(QuantWeightMatrix* out,
                                           bool make_bo_resident,
                                           std::string* error) {
  if (!make_bo_resident) return true;
  if (!ready_ || session_ == nullptr) {
    if (error != nullptr) *error = "NpuQuantGemmEngine not initialized";
    return false;
  }
  for (auto& blk : out->blocks) {
    blk.bo = session_->AllocBo(blk.host_bytes.size(), error);
    if (!blk.bo) return false;
    std::memcpy(blk.bo->map<void*>(), blk.host_bytes.data(),
                blk.host_bytes.size());
    blk.bo->flush();
    std::vector<std::uint8_t>().swap(blk.host_bytes);
  }
  return true;
}

bool NpuQuantGemmEngine::PackMatrixKN(const float* w_kn, std::size_t K,
                                      std::size_t N, NpuQuantMode mode,
                                      bool make_bo_resident,
                                      QuantWeightMatrix* out,
                                      std::string* error) {
  if (out == nullptr || w_kn == nullptr) return false;
  if (K == 0 || N == 0 || (K % 256u != 0u) || (N % 256u != 0u)) {
    if (error != nullptr) {
      *error = "PackMatrixKN requires K and N positive multiples of 256";
    }
    return false;
  }
  if (mode != NpuQuantMode::kQ8Bfp16 && mode != NpuQuantMode::kQ4Q4nx) {
    if (error != nullptr) *error = "PackMatrixKN requires kQ8Bfp16 or kQ4Q4nx";
    return false;
  }

  out->mode = mode;
  out->K = K;
  out->N = N;
  BuildBlockSchedule(K, N, mode, &out->blocks, &out->total_packed_bytes);

  const std::size_t tile_bytes =
      (mode == NpuQuantMode::kQ8Bfp16) ? kQ8TileBytes : kQ4TileBytes;

  auto pack_one = [&](std::size_t bi) {
    auto& blk = out->blocks[bi];
    const std::size_t bdim = blk.block_dim;
    const std::size_t k_tiles = bdim / 64u;
    const std::size_t n_col = bdim / 4u;
    const std::size_t n_tiles_col = n_col / 64u;
    std::uint8_t* dst = blk.host_bytes.data();
    for (std::size_t c = 0; c < 4u; ++c) {
      for (std::size_t nt = 0; nt < n_tiles_col; ++nt) {
        const std::size_t n_off = blk.n0 + c * n_col + nt * 64u;
        for (std::size_t kt = 0; kt < k_tiles; ++kt) {
          const std::size_t k_off = blk.k0 + kt * 64u;
          const float* tile_ptr = w_kn + k_off * N + n_off;
          if (mode == NpuQuantMode::kQ8Bfp16) {
            PackTileQ8Bfp16KN(tile_ptr, N, dst);
          } else {
            PackTileQ4Q4nxKN(tile_ptr, N, dst);
          }
          dst += tile_bytes;
        }
      }
    }
  };

  const std::size_t n_blks = out->blocks.size();
  ::mbufo::quant::ParallelForTasks(n_blks, pack_one);

  return MaterializeBlocks(out, make_bo_resident, error);
}

bool NpuQuantGemmEngine::PackMatrixNK(const float* w_nk, std::size_t N,
                                      std::size_t K, NpuQuantMode mode,
                                      bool make_bo_resident,
                                      QuantWeightMatrix* out,
                                      std::string* error) {
  if (out == nullptr || w_nk == nullptr) return false;
  if (K == 0 || N == 0 || (K % 256u != 0u) || (N % 256u != 0u)) {
    if (error != nullptr) {
      *error = "PackMatrixNK requires K and N positive multiples of 256";
    }
    return false;
  }
  out->mode = mode;
  out->K = K;
  out->N = N;
  BuildBlockSchedule(K, N, mode, &out->blocks, &out->total_packed_bytes);

  const std::size_t tile_bytes =
      (mode == NpuQuantMode::kQ8Bfp16) ? kQ8TileBytes : kQ4TileBytes;

  auto pack_one = [&](std::size_t bi) {
    auto& blk = out->blocks[bi];
    const std::size_t bdim = blk.block_dim;
    const std::size_t k_tiles = bdim / 64u;
    const std::size_t n_col = bdim / 4u;
    const std::size_t n_tiles_col = n_col / 64u;
    std::uint8_t* dst = blk.host_bytes.data();
    float tile_kn[64 * 64];
    for (std::size_t c = 0; c < 4u; ++c) {
      for (std::size_t nt = 0; nt < n_tiles_col; ++nt) {
        const std::size_t n_off = blk.n0 + c * n_col + nt * 64u;
        for (std::size_t kt = 0; kt < k_tiles; ++kt) {
          const std::size_t k_off = blk.k0 + kt * 64u;
          for (std::size_t n = 0; n < 64u; ++n) {
            const float* src_row = w_nk + (n_off + n) * K + k_off;
            for (std::size_t k = 0; k < 64u; ++k) {
              tile_kn[k * 64u + n] = src_row[k];
            }
          }
          if (mode == NpuQuantMode::kQ8Bfp16) {
            PackTileQ8Bfp16KN(tile_kn, 64u, dst);
          } else {
            PackTileQ4Q4nxKN(tile_kn, 64u, dst);
          }
          dst += tile_bytes;
        }
      }
    }
  };

  const std::size_t n_blks = out->blocks.size();
  ::mbufo::quant::ParallelForTasks(n_blks, pack_one);

  return MaterializeBlocks(out, make_bo_resident, error);
}

bool NpuQuantGemmEngine::PackMatrixBf16NK(const std::uint16_t* w_bf16_nk,
                                          std::size_t N, std::size_t K,
                                          NpuQuantMode mode,
                                          bool make_bo_resident,
                                          QuantWeightMatrix* out,
                                          std::string* error) {
  if (out == nullptr || w_bf16_nk == nullptr) return false;
  if (K == 0 || N == 0 || (K % 256u != 0u) || (N % 256u != 0u)) {
    if (error != nullptr) {
      *error = "PackMatrixBf16NK requires K and N positive multiples of 256";
    }
    return false;
  }
  out->mode = mode;
  out->K = K;
  out->N = N;
  BuildBlockSchedule(K, N, mode, &out->blocks, &out->total_packed_bytes);

  const std::size_t tile_bytes =
      (mode == NpuQuantMode::kQ8Bfp16) ? kQ8TileBytes : kQ4TileBytes;

  auto pack_one = [&](std::size_t bi) {
    auto& blk = out->blocks[bi];
    const std::size_t bdim = blk.block_dim;
    const std::size_t k_tiles = bdim / 64u;
    const std::size_t n_col = bdim / 4u;
    const std::size_t n_tiles_col = n_col / 64u;
    std::uint8_t* dst = blk.host_bytes.data();
    float tile_kn[64 * 64];
    for (std::size_t c = 0; c < 4u; ++c) {
      for (std::size_t nt = 0; nt < n_tiles_col; ++nt) {
        const std::size_t n_off = blk.n0 + c * n_col + nt * 64u;
        for (std::size_t kt = 0; kt < k_tiles; ++kt) {
          const std::size_t k_off = blk.k0 + kt * 64u;
          for (std::size_t n = 0; n < 64u; ++n) {
            const std::uint16_t* src_row = w_bf16_nk + (n_off + n) * K + k_off;
            for (std::size_t k = 0; k < 64u; ++k) {
              tile_kn[k * 64u + n] = Bf16BitsToFloat(src_row[k]);
            }
          }
          if (mode == NpuQuantMode::kQ8Bfp16) {
            PackTileQ8Bfp16KN(tile_kn, 64u, dst);
          } else {
            PackTileQ4Q4nxKN(tile_kn, 64u, dst);
          }
          dst += tile_bytes;
        }
      }
    }
  };

  const std::size_t n_blks = out->blocks.size();
  ::mbufo::quant::ParallelForTasks(n_blks, pack_one);

  return MaterializeBlocks(out, make_bo_resident, error);
}

bool NpuQuantGemmEngine::Gemm(const float* A, std::size_t M,
                              const QuantWeightMatrix& W, float* C,
                              const float* bias, double* npu_ms_accum,
                              std::string* error) {
  if (!ready_ || session_ == nullptr) {
    if (error != nullptr) *error = "NpuQuantGemmEngine not initialized";
    return false;
  }
  if (M == 0 || W.K == 0 || W.N == 0 || A == nullptr || C == nullptr) {
    if (error != nullptr) *error = "NpuQuantGemmEngine::Gemm invalid dimensions";
    return false;
  }

  const std::size_t K = W.K;
  const std::size_t N = W.N;
  for (std::size_t r = 0; r < M; ++r) {
    float* c_row = C + r * N;
    if (bias != nullptr) {
      std::memcpy(c_row, bias, N * sizeof(float));
    } else {
      std::memset(c_row, 0, N * sizeof(float));
    }
  }

  bool need_1024[kMaxK1024Slices]{};
  bool need_256[kMaxK256Slices]{};
  std::vector<const QuantWeightBlock*> blks_1024;
  std::vector<const QuantWeightBlock*> blks_256;
  for (const auto& blk : W.blocks) {
    if (blk.block_dim == 1024u) {
      const std::size_t idx = blk.k0 / 1024u;
      if (idx >= kMaxK1024Slices) {
        if (error != nullptr) *error = "K exceeds kMaxK1024Slices";
        return false;
      }
      need_1024[idx] = true;
      blks_1024.push_back(&blk);
    } else {
      const std::size_t idx = blk.k0 / 256u;
      if (idx >= kMaxK256Slices) {
        if (error != nullptr) *error = "K exceeds kMaxK256Slices";
        return false;
      }
      need_256[idx] = true;
      blks_256.push_back(&blk);
    }
  }

  const std::string k_1024_name =
      (W.mode == NpuQuantMode::kQ8Bfp16) ? "q8_1024x1024" : "q4_1024x1024";
  const std::string k_256_name =
      (W.mode == NpuQuantMode::kQ8Bfp16) ? "q8_256x256" : "q4_256x256";

  for (std::size_t m0 = 0; m0 < M; m0 += kQuantGemmM) {
    const std::size_t m_valid = std::min<std::size_t>(kQuantGemmM, M - m0);

    for (std::size_t ks = 0; ks < kMaxK1024Slices; ++ks) {
      if (!need_1024[ks]) continue;
      const std::size_t k0 = ks * 1024u;
      auto* a_dst = a_bo_1024_[ks]->map<std::uint16_t*>();
      std::memset(a_dst, 0, kQuantGemmM * 1024u * sizeof(std::uint16_t));
      for (std::size_t r = 0; r < m_valid; ++r) {
        const float* a_src = A + (m0 + r) * K + k0;
        std::uint16_t* a_row = a_dst + r * 1024u;
        for (std::size_t k = 0; k < 1024u; ++k) {
          a_row[k] = FloatToBf16Rne(a_src[k]);
        }
      }
      a_bo_1024_[ks]->flush();
    }

    for (std::size_t ks = 0; ks < kMaxK256Slices; ++ks) {
      if (!need_256[ks]) continue;
      const std::size_t k0 = ks * 256u;
      auto* a_dst = a_bo_256_[ks]->map<std::uint16_t*>();
      std::memset(a_dst, 0, kQuantGemmM * 256u * sizeof(std::uint16_t));
      for (std::size_t r = 0; r < m_valid; ++r) {
        const float* a_src = A + (m0 + r) * K + k0;
        std::uint16_t* a_row = a_dst + r * 256u;
        for (std::size_t k = 0; k < 256u; ++k) {
          a_row[k] = FloatToBf16Rne(a_src[k]);
        }
      }
      a_bo_256_[ks]->flush();
    }

    // Dispatch 1024x1024 blocks in batches of kBatch1024
    for (std::size_t b0 = 0; b0 < blks_1024.size(); b0 += kBatch1024) {
      const std::size_t b_cnt =
          std::min<std::size_t>(kBatch1024, blks_1024.size() - b0);
      std::vector<std::pair<std::string, std::vector<const hrx::bo*>>> steps;
      steps.reserve(b_cnt);

      for (std::size_t i = 0; i < b_cnt; ++i) {
        const QuantWeightBlock* blk = blks_1024[b0 + i];
        const hrx::bo* b_ptr = nullptr;
        if (blk->bo != nullptr) {
          b_ptr = blk->bo.get();
        } else {
          auto& stg = (W.mode == NpuQuantMode::kQ8Bfp16) ? b_q8_stage_1024_[i]
                                                         : b_q4_stage_1024_[i];
          std::memcpy(stg->map<void*>(), blk->host_bytes.data(),
                      blk->host_bytes.size());
          stg->flush();
          b_ptr = stg.get();
        }
        steps.push_back(
            {k_1024_name,
             {a_bo_1024_[blk->k0 / 1024u].get(), b_ptr, c_bo_1024_[i].get()}});
      }

      const auto t0 = std::chrono::steady_clock::now();
      if (!session_->RunKernelChain(steps, error)) return false;
      const auto t1 = std::chrono::steady_clock::now();
      if (npu_ms_accum != nullptr) {
        *npu_ms_accum +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
      }

      for (std::size_t i = 0; i < b_cnt; ++i) {
        const QuantWeightBlock* blk = blks_1024[b0 + i];
        c_bo_1024_[i]->invalidate();
        const auto* c_src = c_bo_1024_[i]->map<const std::uint16_t*>();
        for (std::size_t r = 0; r < m_valid; ++r) {
          float* c_dst = C + (m0 + r) * N + blk->n0;
          const std::uint16_t* c_row = c_src + r * 1024u;
          for (std::size_t n = 0; n < 1024u; ++n) {
            c_dst[n] += Bf16BitsToFloat(c_row[n]);
          }
        }
      }
    }

    // Dispatch 256x256 blocks in batches of kBatch256
    for (std::size_t b0 = 0; b0 < blks_256.size(); b0 += kBatch256) {
      const std::size_t b_cnt =
          std::min<std::size_t>(kBatch256, blks_256.size() - b0);
      std::vector<std::pair<std::string, std::vector<const hrx::bo*>>> steps;
      steps.reserve(b_cnt);

      for (std::size_t i = 0; i < b_cnt; ++i) {
        const QuantWeightBlock* blk = blks_256[b0 + i];
        const hrx::bo* b_ptr = nullptr;
        if (blk->bo != nullptr) {
          b_ptr = blk->bo.get();
        } else {
          auto& stg = (W.mode == NpuQuantMode::kQ8Bfp16) ? b_q8_stage_256_[i]
                                                         : b_q4_stage_256_[i];
          std::memcpy(stg->map<void*>(), blk->host_bytes.data(),
                      blk->host_bytes.size());
          stg->flush();
          b_ptr = stg.get();
        }
        steps.push_back(
            {k_256_name,
             {a_bo_256_[blk->k0 / 256u].get(), b_ptr, c_bo_256_[i].get()}});
      }

      const auto t0 = std::chrono::steady_clock::now();
      if (!session_->RunKernelChain(steps, error)) return false;
      const auto t1 = std::chrono::steady_clock::now();
      if (npu_ms_accum != nullptr) {
        *npu_ms_accum +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
      }

      for (std::size_t i = 0; i < b_cnt; ++i) {
        const QuantWeightBlock* blk = blks_256[b0 + i];
        c_bo_256_[i]->invalidate();
        const auto* c_src = c_bo_256_[i]->map<const std::uint16_t*>();
        for (std::size_t r = 0; r < m_valid; ++r) {
          float* c_dst = C + (m0 + r) * N + blk->n0;
          const std::uint16_t* c_row = c_src + r * 256u;
          for (std::size_t n = 0; n < 256u; ++n) {
            c_dst[n] += Bf16BitsToFloat(c_row[n]);
          }
        }
      }
    }
  }

  return true;
}

}  // namespace mbufo::core::npu
