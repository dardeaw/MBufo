#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <utility>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include <cstdlib>
#include <vector>
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

namespace {
// Device row stride for the formats Copy() can emit (raw or converted).
std::size_t DeviceRowBytes(core::GgmlType type, std::uint32_t cols) {
  switch (type) {
    case core::GgmlType::kF32:
      return static_cast<std::size_t>(cols) * 4;
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      return static_cast<std::size_t>(cols) * 2;
    case core::GgmlType::kQ8_0:
      return static_cast<std::size_t>(cols) / 32 * 34;
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      return static_cast<std::size_t>(cols) / 32 * 17;
    default:
      return 0;
  }
}
}  // namespace

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

/// The trunk and the optional MTP block as one traversal's output.
struct Uploaded {
  DeviceTensor token_embd;
  DeviceTensor output;
  DeviceMixer hc_head;
  std::vector<DeviceLayer> layers;
  DeviceLayer mtp;
};

/// Walks the model's tensors in upload order. With `weights` null the walk
/// only measures: it sums the bytes every allocation would request and does
/// no device work, so the weight placement can be decided from the tensors.
struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator* weights;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  std::uint64_t requested{0};

  [[nodiscard]] bool Measuring() const noexcept { return weights == nullptr; }

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t size) {
    requested += size;
    if (Measuring()) {
      return hipSuccess;
    }
    return weights->Allocate(pointer, size);
  }

  // Host-side Q8_0 conversion for formats the device kernels do not read
  // (ROCmFP4 dual-scale attention, Q6_K embeddings): dequantize each row
  // with the core CPU oracle and requantize to Q8_0, then upload the bytes.
  // The Q8_0 step is ~64x finer than the 4-bit source step, so the
  // round-trip error is negligible next to the source quantization itself.
  // Experts never reach here: they upload raw (see layer_supported below).
  DeviceTensor CopyConvertedQ8(const TensorRef& t) {
    DeviceTensor d;
    if (t.experts != 1) {
      Fail("Q8 conversion needs a single-slab tensor: " +
           std::string(t.name));
      return d;
    }
    const std::size_t cols = static_cast<std::size_t>(t.cols);
    const std::size_t rows = static_cast<std::size_t>(t.rows);
    if (cols == 0 || cols % 32 != 0 || rows == 0) {
      Fail("Q8 conversion needs non-empty 32-wide rows: " +
           std::string(t.name));
      return d;
    }
    const std::size_t q8_row = cols / 32 * 34;
    const std::size_t size = q8_row * rows;
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for converted " + std::string(t.name) +
           " (" + std::to_string(size) + " bytes)");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    constexpr std::size_t kChunkRows = 1024;
    std::vector<float> fbuf(cols * kChunkRows);
    std::vector<std::uint8_t> staging(size);
    for (std::size_t r0 = 0; r0 < rows; r0 += kChunkRows) {
      const std::size_t chunk = std::min(kChunkRows, rows - r0);
      const std::uint8_t* src = t.Expert(0) + r0 * t.RowBytes();
      float* frow = fbuf.data();
      for (std::size_t r = 0; r < chunk; ++r) {
        const void* row_src = src + r * t.RowBytes();
        if (t.type == core::GgmlType::kQ4_0_ROCMFP4) {
          gufo::quant::DequantizeQ4_0_ROCMFP4(row_src, frow, cols);
        } else if (t.type == core::GgmlType::kQ4_0_ROCMFP4_FAST) {
          gufo::quant::DequantizeQ4_0_ROCMFP4_FAST(row_src, frow, cols);
        } else if (t.type == core::GgmlType::kQ6_K) {
          gufo::quant::DequantizeQ6_K(row_src, frow, cols);
        } else {
          Fail("unexpected conversion source for " + std::string(t.name));
          return d;
        }
        frow += cols;
      }
      gufo::quant::QuantizeQ8_0(fbuf.data(), staging.data() + r0 * q8_row,
                                chunk * cols);
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (hipMemcpy(ptr, staging.data(), size, hipMemcpyHostToDevice) !=
        hipSuccess) {
      Fail("converted upload failed for " + std::string(t.name));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = core::GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(cols);
    d.rows = static_cast<std::uint32_t>(rows);
    d.experts = 1;
    max_q8_cols = std::max<std::size_t>(max_q8_cols, cols);
    return d;
  }

  // Q8_0-safe dense weights (attention/SSM/indexer/PLE-KV/mixer/
  // shared-expert/embedding/head) may arrive quantized (ROCmFP4 single
  // or dual scale, Q6_K): convert on the host. Everything else uploads
  // raw; F32-only consumers (norms/router/biases/nextn, inject below)
  // never see these types because the bind sets reject them there.
  DeviceTensor CopyQ8(const TensorRef& t) {
    if (!t.empty() && ok &&
        (t.type == core::GgmlType::kQ4_0_ROCMFP4 ||
         t.type == core::GgmlType::kQ4_0_ROCMFP4_FAST ||
         t.type == core::GgmlType::kQ6_K)) {
      return CopyConvertedQ8(t);
    }
    return Copy(t);
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || (!Measuring() && !stager.Finish(error))) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (Allocate(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      if (Measuring()) {
        continue;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    if (Measuring()) {
      return;
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (!Measuring()) {
      allocations.push_back(ptr);
      bytes += size + kTailMargin;
      std::size_t offset = 0;
      for (const TensorRef* t : parts) {
        if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                         static_cast<std::uint8_t*>(ptr) + offset, error)) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += t->SizeBytes();
      }
    }
    if (type == core::GgmlType::kQ8_0) {
      if (Measuring()) {
        return d;
      }
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (Allocate(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  // The mixer combine reads inject as F32 rows: quantized files are
  // dequantized to F32 on the host (exact values, no requant loss).
  DeviceTensor CopyF32(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (t.type == core::GgmlType::kF32) {
      return Copy(t);
    }
    const std::size_t cols = static_cast<std::size_t>(t.cols);
    const std::size_t rows = static_cast<std::size_t>(t.rows);
    const std::size_t size = cols * rows * sizeof(float);
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for F32 " + std::string(t.name));
      return d;
    }
    if (Measuring()) {
      return d;
    }
    std::vector<float> staging(cols * rows);
    for (std::size_t r = 0; r < rows; ++r) {
      const void* row_src =
          t.Expert(0) + r * t.RowBytes();
      float* out = staging.data() + r * cols;
      switch (t.type) {
        case core::GgmlType::kQ8_0:
          gufo::quant::DequantizeQ8_0(row_src, out, cols);
          break;
        case core::GgmlType::kQ4_0_ROCMFP4:
          gufo::quant::DequantizeQ4_0_ROCMFP4(row_src, out, cols);
          break;
        case core::GgmlType::kQ4_0_ROCMFP4_FAST:
          gufo::quant::DequantizeQ4_0_ROCMFP4_FAST(row_src, out, cols);
          break;
        case core::GgmlType::kQ6_K:
          gufo::quant::DequantizeQ6_K(row_src, out, cols);
          break;
        case core::GgmlType::kF16: {
          const auto* h = static_cast<const std::uint16_t*>(row_src);
          for (std::size_t i = 0; i < cols; ++i)
            out[i] = gufo::quant::Fp16ToFloat(h[i]);
          break;
        }
        case core::GgmlType::kBF16: {
          const auto* h = static_cast<const std::uint16_t*>(row_src);
          for (std::size_t i = 0; i < cols; ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(h[i]) << 16;
            std::memcpy(out + i, &bits, sizeof(float));
          }
          break;
        }
        default:
          Fail("cannot dequantize inject " + std::string(t.name));
          return d;
      }
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (hipMemcpy(ptr, staging.data(), size, hipMemcpyHostToDevice) !=
        hipSuccess) {
      Fail("F32 upload failed for " + std::string(t.name));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = core::GgmlType::kF32;
    d.cols = static_cast<std::uint32_t>(cols);
    d.rows = static_cast<std::uint32_t>(rows);
    d.experts = 1;
    return d;
  }

  // Stacks quantized projections (ROCmFP4/Q6_K alpha+beta) as one Q8_0
  // matrix: each part is host-converted, then the bytes are uploaded
  // contiguously. The consumer reads it through the Q8_0 dense path.
  DeviceTensor StackConvertedQ8(
      std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    const core::GgmlType type = (*parts.begin())->type;
    std::size_t rows = 0;
    std::size_t size = 0;
    for (const TensorRef* t : parts) {
      const bool convertible =
          t->type == core::GgmlType::kQ4_0_ROCMFP4 ||
          t->type == core::GgmlType::kQ4_0_ROCMFP4_FAST ||
          t->type == core::GgmlType::kQ6_K;
      if (t->empty() || t->type != type ||
          t->cols != (*parts.begin())->cols || t->cols % 32 != 0 ||
          !convertible) {
        Fail("stacked Q8 conversion needs one shape of quantized rows");
        return d;
      }
      rows += t->rows;
      size += (t->cols / 32 * 34) * t->rows;
    }
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked Q8 tensor allocation failed");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    std::vector<std::uint8_t> staging(size);
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      const std::size_t part = (t->cols / 32 * 34) * t->rows;
      constexpr std::size_t kChunkRows = 1024;
      const std::size_t cols = static_cast<std::size_t>(t->cols);
      const std::size_t nrows = static_cast<std::size_t>(t->rows);
      std::vector<float> fbuf(cols * kChunkRows);
      for (std::size_t r0 = 0; r0 < nrows; r0 += kChunkRows) {
        const std::size_t chunk = std::min(kChunkRows, nrows - r0);
        const std::uint8_t* src = t->Expert(0) + r0 * t->RowBytes();
        float* frow = fbuf.data();
        for (std::size_t r = 0; r < chunk; ++r) {
          const void* row_src = src + r * t->RowBytes();
          if (type == core::GgmlType::kQ4_0_ROCMFP4) {
            gufo::quant::DequantizeQ4_0_ROCMFP4(row_src, frow, cols);
          } else if (type == core::GgmlType::kQ4_0_ROCMFP4_FAST) {
            gufo::quant::DequantizeQ4_0_ROCMFP4_FAST(row_src, frow, cols);
          } else {
            gufo::quant::DequantizeQ6_K(row_src, frow, cols);
          }
          frow += cols;
        }
        gufo::quant::QuantizeQ8_0(fbuf.data(),
                                    staging.data() + offset + r0 * cols / 32 * 34,
                                    chunk * cols);
      }
      offset += part;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (hipMemcpy(ptr, staging.data(), size, hipMemcpyHostToDevice) !=
        hipSuccess) {
      Fail("stacked Q8 upload failed");
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = core::GgmlType::kQ8_0;
    d.cols = (*parts.begin())->cols;
    d.rows = static_cast<std::uint32_t>(rows);
    d.experts = 1;
    max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), CopyQ8(m.down), CopyQ8(m.up), CopyF32(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = CopyQ8(l.ssm_qkv);
      d.ssm_gate = CopyQ8(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      const bool stacked_native =
          l.ssm_alpha.type == l.ssm_beta.type &&
          l.ssm_alpha.cols == l.ssm_beta.cols &&
          (l.ssm_alpha.type == core::GgmlType::kF32 ||
           l.ssm_alpha.type == core::GgmlType::kQ8_0);
      if (stacked_native) {
        d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
      } else {
        d.ssm_alpha_beta = StackConvertedQ8({&l.ssm_alpha, &l.ssm_beta});
      }
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = CopyQ8(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = CopyQ8(l.attn_q);
      d.attn_k = CopyQ8(l.attn_k);
      d.attn_v = CopyQ8(l.attn_v);
    }
    d.attn_out = CopyQ8(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = CopyQ8(l.indexer_q);
    d.indexer_k = CopyQ8(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = CopyQ8(l.ple_key);
    d.ple_value = CopyQ8(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = CopyQ8(l.shexp_gate);
    d.shexp_up = CopyQ8(l.shexp_up);
    d.shexp_down = CopyQ8(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  Uploaded Model(const ModelWeights& w, const MtpWeights* mtp,
                 std::uint32_t trunk_shards) {
    Uploaded u;
    u.token_embd = CopyQ8(w.token_embd);
    u.output =
        w.output.data == w.token_embd.data ? u.token_embd : CopyQ8(w.output);
    u.hc_head = Mixer(w.hc_head);
    u.layers.reserve(w.layers.size());
    for (const auto& l : w.layers) {
      u.layers.push_back(Layer(l));
      if (!ok) {
        return u;
      }
    }
    if (mtp != nullptr) {
      shard_base = trunk_shards;
      u.mtp = Layer(mtp->block);
    }
    return u;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

bool DeviceModel::SetDraftVocab(std::uint32_t prefix_rows,
                                std::vector<std::int32_t> specials,
                                std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  if (output_.empty() || output_.rows == 0 || output_bytes_ % output_.rows != 0)
    return fail("draft vocabulary needs a row-addressable output head");
  if (prefix_rows == 0 || prefix_rows >= output_.rows)
    return fail("draft vocabulary prefix must leave rows out");
  for (const auto id : specials) {
    if (id < static_cast<std::int32_t>(prefix_rows) ||
        static_cast<std::uint32_t>(id) >= output_.rows)
      return fail("draft special tokens must lie above the prefix");
  }
  // The prefix is the head's own leading rows: no copy.
  draft_output_ = output_;
  draft_output_.rows = prefix_rows;
  draft_specials_ = std::move(specials);
  return true;
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // Formats without a device path (ROCmFP4 dual-scale, Q6_K) are
  // converted to Q8_0 inside CopyQ8(); experts stay native-only.
  const auto supported = [&](const TensorRef& t) {
    (void)t;
    return true;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    // Experts upload raw (gate/up stay K-quant, down may also be FAST):
    // anything else was rejected at bind time.
    for (const TensorRef* t :
         {&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps}) {
      if (t->type == core::GgmlType::kQ6_K ||
          t->type == core::GgmlType::kQ4_0_ROCMFP4) {
        if (error_msg != nullptr)
          *error_msg = "unsupported HIP tensor format for experts: " +
                       std::string(t->name);
        return false;
      }
    }
    return true;
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  // Measure first: the placement is decided once, from the bytes this walk
  // will request of the device heap.
  std::uint64_t measured = 0;
  {
    std::vector<Conversion> conversions;
    std::vector<void*> allocations;
    std::size_t bytes = 0;
    std::size_t max_half_cols = 1;
    std::size_t max_q8_cols = 32;
    Uploader measure{*stager, nullptr,       conversions, allocations,
                     bytes,   max_half_cols, max_q8_cols, error_msg};
    (void)measure.Model(w, mtp, shard_count);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  std::vector<Conversion> conversions;
  Uploader up{*stager,   &weights,          conversions,     m->allocations_,
              m->bytes_, m->max_half_cols_, m->max_q8_cols_, error_msg};
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  // Converted heads change row stride: measure the device tensor.
  m->output_bytes_ = uploaded.output.type == w.output.type
                         ? w.output.SizeBytes()
                         : DeviceRowBytes(uploaded.output.type,
                                          uploaded.output.cols) *
                               uploaded.output.rows;
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  // Decode reads each large dense matrix once per round, so its bytes set the
  // round's cost: a 4-bit (Q4_1) copy cuts them by 41% against Q8_0 (a 5-bit
  // Q5_1 copy by 29%). GUFO_DECODE_Q4=0 selects the Q5_1 copy; GUFO_DECODE_Q5=0
  // disables the copies.
  const char* q5_flag = std::getenv("GUFO_DECODE_Q5");
  const bool q5_on = q5_flag == nullptr || q5_flag[0] != '0';
  const char* q4_flag = std::getenv("GUFO_DECODE_Q4");
  const bool q4_on = q5_on && (q4_flag == nullptr || q4_flag[0] != '0');
  if (q4_on || q5_on) {
    const std::size_t blk_bytes = q4_on ? 20 : 24;
    const auto dec_type = q4_on ? core::GgmlType::kQ4_1 : core::GgmlType::kQ5_1;
    constexpr std::size_t kMinElements = std::size_t{2560} * 6144;
    std::size_t copied = 0;
    const auto add = [&](DeviceTensor& t) {
      if (t.type != core::GgmlType::kQ8_0 || t.data == nullptr ||
          t.cols % 32 != 0 || std::size_t{t.rows} * t.cols < kMinElements) {
        return;
      }
      const std::size_t blocks = std::size_t{t.rows} * t.cols / 32;
      void* copy = nullptr;
      if (hipMalloc(&copy, blocks * blk_bytes + kTailMargin) != hipSuccess) {
        return;  // the tensor keeps its Q8_0 decode path
      }
      m->allocations_.push_back(copy);
      m->bytes_ += blocks * blk_bytes + kTailMargin;
      copied += blocks * blk_bytes;
      const bool done = q4_on
                            ? RequantQ8_0ToQ4_1(t.data, copy, blocks, nullptr)
                            : RequantQ8_0ToQ5_1(t.data, copy, blocks, nullptr);
      if (done) {
        t.decode_data = copy;
        t.decode_type = dec_type;
      }
    };
    for (auto& l : m->layers_) {
      add(l.ssm_in);
      add(l.ssm_out);
      add(l.attn_qkv);
      add(l.attn_out);
    }
    if (hipDeviceSynchronize() != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg = "decode weight conversion failed";
      }
      return nullptr;
    }
    std::fprintf(stderr, "decode %s copies: %.2f GiB\n",
                 q4_on ? "Q4_1" : "Q5_1",
                 static_cast<double>(copied) / (1024.0 * 1024.0 * 1024.0));
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
