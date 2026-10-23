#ifndef GUFO_CORE_QUANT_GGML_DEQUANT_HPP_
#define GUFO_CORE_QUANT_GGML_DEQUANT_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::quant {

// Canonical quantized block layouts. These are the authoritative field
// layouts for the dequant/dot paths. Relocated verbatim from the internal
// definitions in ggml_dequant.cpp so the header is the single source of
// truth; field order and byte sizes match the HIP path
// (src/models/qwen/hip/quant_ops.hpp) and the historical per-file copies
// (models/qwen/state.hpp, tests/). Do NOT redefine block_* locally. Sizes are
// static_asserted; this is the layout contract (see the parity test that
// proves field order against the byte-level spec).
#pragma pack(push, 1)
struct block_q4_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};

struct block_q5_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};

struct block_q6_K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};

struct block_q3_K {
  std::uint8_t hmask[32];
  std::uint8_t qs[64];
  std::uint8_t scales[12];
  std::uint16_t d;
};

struct block_q8_K {
  float d;
  std::int8_t qs[256];
  std::int16_t bsums[16];
};

// Q8_0: fp16 scale + 32 int8 quantized values (QK=32).
struct block_q8_0 {
  std::uint16_t d;
  std::int8_t qs[32];
};

// IQ4_NL: fp16 scale + 32 non-linear 4-bit codebook indices (QK=32). The
// codebook is kValuesIq4Nl; the index nibble order matches Q4_0 (low nibbles
// give the first 16 elements, high nibbles the second 16).
struct block_iq4_nl {
  std::uint16_t d;
  std::uint8_t qs[16];
};

// IQ4_XS: super-block of 256 with eight 6-bit sub-block scales split across
// scales_h (2 high bits each) and scales_l (4 low bits each), plus 128 bytes
// of packed non-linear 4-bit codebook indices.
struct block_iq4_xs {
  std::uint16_t d;
  std::uint16_t scales_h;
  std::uint8_t scales_l[4];
  std::uint8_t qs[128];
};

// IQ3_S: super-block of 256. Each group of 8 elements indexes the 512-entry
// kIq3sGrid table (8 low bits in qs, 1 high bit in qh) and carries a per-group
// sign byte; scales holds two 4-bit sub-block scales per byte.
struct block_iq3_s {
  std::uint16_t d;
  std::uint8_t qs[64];
  std::uint8_t qh[8];
  std::uint8_t signs[32];
  std::uint8_t scales[4];
};

// ROCmFP4 (agentionai/llama.cpp, MIT): AMD E2M1-derived 4-bit codebook
// {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10} with unsigned E4M3 block
// scales. Dual-scale layout: e[0] scales the low-nibble half (elements
// 0..15), e[1] the high-nibble half (16..31).
struct block_q4_0_rocmfp4 {
  std::uint8_t qs[16];
  std::uint8_t e[2];
};

// ROCmFP4 FAST: same packed nibbles (low nibbles give the first 16
// elements, high nibbles the second 16) with one UE4M3 scale per 32
// elements (4.25 bpw).
struct block_q4_0_rocmfp4_fast {
  std::uint8_t qs[16];
  std::uint8_t e;
};

// ROCmFP3: 3-bit codes {0,1,2,4,0,-1,-2,-4} packed eight-per-3-bytes
// (little-endian bit order), dual UE4M3 scales per 16-element half.
struct block_q3_0_rocmfpx {
  std::uint8_t qs[12];
  std::uint8_t e[2];
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl must be 18 bytes");
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be 136 bytes");
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s must be 110 bytes");
static_assert(sizeof(block_q4_0_rocmfp4) == 18,
              "block_q4_0_rocmfp4 must be 18 bytes");
static_assert(sizeof(block_q4_0_rocmfp4_fast) == 17,
              "block_q4_0_rocmfp4_fast must be 17 bytes");
static_assert(sizeof(block_q3_0_rocmfpx) == 14,
              "block_q3_0_rocmfpx must be 14 bytes");

/// Non-linear 4-bit codebook shared by IQ4_NL and IQ4_XS.
inline constexpr std::int8_t kValuesIq4Nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

/// Returns the logical elements represented by one supported quant block, or
/// zero for dense and unsupported storage types.
[[nodiscard]] constexpr std::size_t QuantizedBlockElements(
    core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ4_1:
    case core::GgmlType::kQ5_0:
    case core::GgmlType::kQ5_1:
    case core::GgmlType::kQ8_1:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kQ4_0_ROCMFP4:
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
    case core::GgmlType::kQ3_0_ROCMFPX:
      return 32;
    case core::GgmlType::kQ2_K:
    case core::GgmlType::kIQ2_XXS:
    case core::GgmlType::kIQ4_XS:
    case core::GgmlType::kIQ3_S:
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ8_K:
      return 256;
    default:
      return 0;
  }
}

/// Returns the encoded byte count for one logical quantized row, or zero when
/// the type is not quantized or the element count is not block aligned.
[[nodiscard]] std::size_t QuantizedRowBytes(core::GgmlType type,
                                            std::size_t elements) noexcept;

/// Returns the physical byte count for a tensor with `elements` logical
/// elements, or zero when the format is unsupported/misaligned.
[[nodiscard]] std::size_t EncodedSizeBytes(core::GgmlType type,
                                           std::size_t elements) noexcept;

// Standard 16-bit float helper
float Fp16ToFloat(std::uint16_t h) noexcept;

// Dequantize row of Q4_K to float
void DequantizeQ4_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q5_K to float
void DequantizeQ5_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q6_K to float
void DequantizeQ6_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q3_K to float
void DequantizeQ3_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q8_K to float
void DequantizeQ8_K(const void* src, float* dst, std::size_t k);

// Compute dot product of quantized row with FP32 vector
float DotProductQ4_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
// Compute dot product of Q5_K quantized row with FP32 vector
float DotProductQ5_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ6_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ3_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ8_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);

// Dequantize row of Q8_0 to float
void DequantizeQ8_0(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ4_NL to float
void DequantizeIQ4_NL(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ4_XS to float
void DequantizeIQ4_XS(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ3_S to float
void DequantizeIQ3_S(const void* src, float* dst, std::size_t k);

// Compute dot product of IQ4_NL quantized row with FP32 vector
float DotProductIQ4_NL(const void* row_data, std::span<const float> vec,
                       std::size_t k);
// Compute dot product of IQ4_XS quantized row with FP32 vector
float DotProductIQ4_XS(const void* row_data, std::span<const float> vec,
                       std::size_t k);
// Compute dot product of IQ3_S quantized row with FP32 vector
float DotProductIQ3_S(const void* row_data, std::span<const float> vec,
                      std::size_t k);

/// Returns the 512-entry IQ3_S grid table (each entry packs four uint8
/// magnitudes).
[[nodiscard]] const std::uint32_t* Iq3sGrid() noexcept;

// ROCmFPx family (agentionai/llama.cpp, MIT).

/// Decodes an unsigned E4M3 scale byte: exp==0 -> mant * 2^-10, else
/// (8 + mant) * 2^(exp - 11). Bytes above 0x7e (NaN) decode to 0, matching
/// the reference table guard.
[[nodiscard]] float Ue4m3ScaleToFloat(std::uint8_t e) noexcept;

/// Decodes one 4-bit ROCmFP4 code (0..15) to its codebook value.
[[nodiscard]] std::int8_t RocmFp4Value(std::uint8_t code) noexcept;

/// Decodes one 3-bit ROCmFP3 code (0..7) to its codebook value.
[[nodiscard]] float RocmFp3Value(std::uint8_t code) noexcept;

void DequantizeQ4_0_ROCMFP4(const void* src, float* dst, std::size_t k);
void DequantizeQ4_0_ROCMFP4_FAST(const void* src, float* dst, std::size_t k);
void DequantizeQ3_0_ROCMFPX(const void* src, float* dst, std::size_t k);

float DotProductQ4_0_ROCMFP4(const void* row_data, std::span<const float> vec,
                             std::size_t k);
float DotProductQ4_0_ROCMFP4_FAST(const void* row_data,
                                  std::span<const float> vec, std::size_t k);
float DotProductQ3_0_ROCMFPX(const void* row_data, std::span<const float> vec,
                             std::size_t k);

/// Scale-byte validation mirroring the reference row validators (every UE4M3
/// scale byte must be <= 0x7e; nbytes must be a whole number of blocks).
[[nodiscard]] bool ValidateQ4_0_ROCMFP4(const void* data, std::size_t nbytes);
[[nodiscard]] bool ValidateQ4_0_ROCMFP4_FAST(const void* data,
                                             std::size_t nbytes);
[[nodiscard]] bool ValidateQ3_0_ROCMFPX(const void* data, std::size_t nbytes);

// Compute dot product of Q8_0 quantized row with FP32 vector
float DotProductQ8_0(const void* row_data, std::span<const float> vec,
                     std::size_t k);

}  // namespace gufo::quant

#endif  // GUFO_CORE_QUANT_GGML_DEQUANT_HPP_
