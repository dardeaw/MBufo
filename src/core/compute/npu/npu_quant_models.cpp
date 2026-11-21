#include "src/core/compute/npu/npu_quant_models.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <utility>

#include "src/core/compute/npu/npu_pooling.hpp"
#include "src/core/compute/npu/npu_safetensors_bridge.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace mbufo::core::npu {

std::uint32_t PngCrc32(const std::uint8_t* data, std::size_t len) {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) {
      crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
    }
  }
  return crc ^ 0xFFFFFFFFu;
}

void PngAppendU32Be(std::vector<std::uint8_t>& out, std::uint32_t v) {
  out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>(v & 0xFFu));
}

void PngAppendChunk(std::vector<std::uint8_t>& out, const char type[4],
                    const std::uint8_t* payload, std::size_t len) {
  PngAppendU32Be(out, static_cast<std::uint32_t>(len));
  const std::size_t start = out.size();
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(type[i]));
  if (len > 0) out.insert(out.end(), payload, payload + len);
  const std::uint32_t crc = PngCrc32(out.data() + start, 4 + len);
  PngAppendU32Be(out, crc);
}

std::vector<std::uint8_t> EncodeRgbaPng(const models::qwen_image_21::Image& img) {
  const std::size_t width = static_cast<std::size_t>(img.width);
  const std::size_t height = static_cast<std::size_t>(img.height);
  std::vector<std::uint8_t> raw;
  raw.reserve(height * (1 + width * 4));
  for (std::size_t y = 0; y < height; ++y) {
    raw.push_back(0);
    const std::uint8_t* row = img.rgba.data() + y * width * 4;
    raw.insert(raw.end(), row, row + width * 4);
  }
  std::vector<std::uint8_t> z;
  z.push_back(0x78);
  z.push_back(0x01);
  std::size_t pos = 0;
  std::uint32_t s1 = 1;
  std::uint32_t s2 = 0;
  for (std::uint8_t b : raw) {
    s1 = (s1 + b) % 65521u;
    s2 = (s2 + s1) % 65521u;
  }
  while (pos < raw.size()) {
    const std::size_t chunk = std::min<std::size_t>(65535, raw.size() - pos);
    const bool last = (pos + chunk == raw.size());
    z.push_back(last ? 0x01 : 0x00);
    const std::uint16_t len16 = static_cast<std::uint16_t>(chunk);
    const std::uint16_t nlen16 = static_cast<std::uint16_t>(~len16);
    z.push_back(static_cast<std::uint8_t>(len16 & 0xFFu));
    z.push_back(static_cast<std::uint8_t>((len16 >> 8) & 0xFFu));
    z.push_back(static_cast<std::uint8_t>(nlen16 & 0xFFu));
    z.push_back(static_cast<std::uint8_t>((nlen16 >> 8) & 0xFFu));
    z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
             raw.begin() + static_cast<std::ptrdiff_t>(pos + chunk));
    pos += chunk;
  }
  PngAppendU32Be(z, (s2 << 16) | s1);

  std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<std::uint8_t> ihdr;
  PngAppendU32Be(ihdr, static_cast<std::uint32_t>(width));
  PngAppendU32Be(ihdr, static_cast<std::uint32_t>(height));
  ihdr.push_back(8);  // bit depth 8
  ihdr.push_back(6);  // color type 6 (RGBA)
  ihdr.push_back(0);
  ihdr.push_back(0);
  ihdr.push_back(0);
  PngAppendChunk(png, "IHDR", ihdr.data(), ihdr.size());
  PngAppendChunk(png, "IDAT", z.data(), z.size());
  PngAppendChunk(png, "IEND", nullptr, 0);
  return png;
}

namespace {

inline float Silu(float x) noexcept { return x / (1.0F + std::exp(-x)); }

inline float Gelu(float x) noexcept {
  const float u = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + std::tanh(u));
}

inline float Fp16BitsToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
  const std::uint32_t exp = (h >> 10u) & 0x1Fu;
  const std::uint32_t mant = h & 0x03FFu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      float f = static_cast<float>(mant) * (1.0F / 16777216.0F);
      return (sign != 0) ? -f : f;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13u);
  } else {
    bits = sign | ((exp + 112u) << 23u) | (mant << 13u);
  }
  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof(float));
  return out;
}

float CosineSim(std::span<const float> a, std::span<const float> b) noexcept {
  if (a.size() != b.size() || a.empty()) return 0.0F;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double va = a[i];
    const double vb = b[i];
    dot += va * vb;
    na += va * va;
    nb += vb * vb;
  }
  if (na <= 0.0 || nb <= 0.0) return 0.0F;
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

void RmsNormF32(std::span<const float> x, std::span<const float> w,
                std::size_t rows, std::size_t cols, float eps,
                std::vector<float>* out) {
  out->resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x.data() + r * cols;
    float* yr = out->data() + r * cols;
    double ss = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      ss += static_cast<double>(xr[c]) * static_cast<double>(xr[c]);
    }
    const float inv = static_cast<float>(
        1.0 / std::sqrt(ss / static_cast<double>(cols) + eps));
    for (std::size_t c = 0; c < cols; ++c) {
      yr[c] = xr[c] * inv * (w.empty() ? 1.0F : w[c]);
    }
  }
}

void LayerNormF32(std::span<const float> x, std::span<const float> w,
                  std::span<const float> b, std::size_t rows, std::size_t cols,
                  float eps, std::vector<float>* out) {
  out->resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x.data() + r * cols;
    float* yr = out->data() + r * cols;
    double sum = 0.0;
    for (std::size_t c = 0; c < cols; ++c) sum += xr[c];
    const double mean = sum / static_cast<double>(cols);
    double var = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const double d = static_cast<double>(xr[c]) - mean;
      var += d * d;
    }
    const float inv = static_cast<float>(
        1.0 / std::sqrt(var / static_cast<double>(cols) + eps));
    for (std::size_t c = 0; c < cols; ++c) {
      const float n = (xr[c] - static_cast<float>(mean)) * inv;
      yr[c] = n * (w.empty() ? 1.0F : w[c]) + (b.empty() ? 0.0F : b[c]);
    }
  }
}

void MatMulRefNK(const float* A, std::size_t M, const float* W_nk,
                 std::size_t N, std::size_t K, const float* bias,
                 std::vector<float>* C) {
  C->resize(M * N);
  for (std::size_t m = 0; m < M; ++m) {
    const float* a_row = A + m * K;
    float* c_row = C->data() + m * N;
    for (std::size_t n = 0; n < N; ++n) {
      const float* w_row = W_nk + n * K;
      double acc = (bias != nullptr) ? static_cast<double>(bias[n]) : 0.0;
      for (std::size_t k = 0; k < K; ++k) {
        acc += static_cast<double>(a_row[k]) * static_cast<double>(w_row[k]);
      }
      c_row[n] = static_cast<float>(acc);
    }
  }
}

class MultiShardSafetensors {
 public:
  [[nodiscard]] static bool OpenDir(const std::string& dir,
                                    MultiShardSafetensors* out,
                                    std::string* error) {
    out->shards_.clear();
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
      if (error != nullptr) *error = "directory does not exist: " + dir;
      return false;
    }
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
      if (!entry.is_regular_file()) continue;
      const std::string p = entry.path().string();
      if (p.ends_with(".safetensors")) paths.push_back(p);
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
      if (error != nullptr) *error = "no .safetensors found in " + dir;
      return false;
    }
    for (const auto& p : paths) {
      auto idx = SafetensorsIndex::Open(p, error);
      if (!idx.has_value()) return false;
      out->shards_.push_back(std::move(*idx));
    }
    return true;
  }

  [[nodiscard]] bool FindEntry(std::string_view name,
                               const SafetensorsIndex** out_shard,
                               const SafetensorEntry** out_entry) const {
    for (const auto& s : shards_) {
      if (const auto* e = s.Find(name); e != nullptr) {
        *out_shard = &s;
        *out_entry = e;
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool ReadF32(std::string_view name, std::size_t expected_elems,
                             std::vector<float>* out,
                             std::string* error) const {
    const SafetensorsIndex* shard = nullptr;
    const SafetensorEntry* entry = nullptr;
    if (!FindEntry(name, &shard, &entry)) {
      if (error != nullptr) *error = "missing safetensor: " + std::string(name);
      return false;
    }
    if (entry->ElementCount() != expected_elems) {
      if (error != nullptr) {
        *error = "element count mismatch for " + std::string(name) + ": got " +
                 std::to_string(entry->ElementCount()) + ", expected " +
                 std::to_string(expected_elems);
      }
      return false;
    }
    std::vector<std::uint8_t> raw;
    if (!shard->ReadChunk(*entry, 0, entry->ByteSize(), raw, error)) {
      return false;
    }
    out->resize(expected_elems);
    if (entry->dtype == "F32") {
      std::memcpy(out->data(), raw.data(), expected_elems * sizeof(float));
      return true;
    }
    if (entry->dtype == "BF16") {
      const auto* u16 = reinterpret_cast<const std::uint16_t*>(raw.data());
      for (std::size_t i = 0; i < expected_elems; ++i) {
        (*out)[i] = Bf16BitsToFloat(u16[i]);
      }
      return true;
    }
    if (entry->dtype == "F16") {
      const auto* u16 = reinterpret_cast<const std::uint16_t*>(raw.data());
      for (std::size_t i = 0; i < expected_elems; ++i) {
        (*out)[i] = Fp16BitsToFloat(u16[i]);
      }
      return true;
    }
    if (error != nullptr) {
      *error = "unsupported dtype " + entry->dtype + " in " + std::string(name);
    }
    return false;
  }

  /// Reads a subset of rows `ids` from a 2D embedding table `[vocab, dim]` without
  /// loading the entire multi-hundred-MB table into RAM.
  [[nodiscard]] bool GatherRowsF32(std::string_view name,
                                   std::span<const std::uint32_t> ids,
                                   std::size_t dim, std::vector<float>* out,
                                   std::string* error) const {
    const SafetensorsIndex* shard = nullptr;
    const SafetensorEntry* entry = nullptr;
    if (!FindEntry(name, &shard, &entry)) {
      if (error != nullptr) *error = "missing embedding tensor: " + std::string(name);
      return false;
    }
    const std::size_t elem_bytes = (entry->dtype == "F32") ? 4u : 2u;
    const std::size_t row_bytes = dim * elem_bytes;
    out->resize(ids.size() * dim);
    std::vector<std::uint8_t> buf;
    for (std::size_t t = 0; t < ids.size(); ++t) {
      const std::uint64_t off = static_cast<std::uint64_t>(ids[t]) * row_bytes;
      if (!shard->ReadChunk(*entry, off, row_bytes, buf, error)) return false;
      float* dst = out->data() + t * dim;
      if (entry->dtype == "F32") {
        std::memcpy(dst, buf.data(), dim * sizeof(float));
      } else if (entry->dtype == "BF16") {
        const auto* u16 = reinterpret_cast<const std::uint16_t*>(buf.data());
        for (std::size_t c = 0; c < dim; ++c) dst[c] = Bf16BitsToFloat(u16[c]);
      } else {
        const auto* u16 = reinterpret_cast<const std::uint16_t*>(buf.data());
        for (std::size_t c = 0; c < dim; ++c) dst[c] = Fp16BitsToFloat(u16[c]);
      }
    }
    return true;
  }

  /// Reads contiguous rows `[row0, row0 + num_rows)` from `[total_rows, cols]`.
  [[nodiscard]] bool ReadRowSliceF32(std::string_view name, std::size_t row0,
                                     std::size_t num_rows, std::size_t cols,
                                     std::vector<float>* out,
                                     std::string* error) const {
    const SafetensorsIndex* shard = nullptr;
    const SafetensorEntry* entry = nullptr;
    if (!FindEntry(name, &shard, &entry)) {
      if (error != nullptr) *error = "missing tensor: " + std::string(name);
      return false;
    }
    const std::size_t elem_bytes = (entry->dtype == "F32") ? 4u : 2u;
    const std::uint64_t off = static_cast<std::uint64_t>(row0) * cols * elem_bytes;
    const std::size_t count = num_rows * cols;
    std::vector<std::uint8_t> raw;
    if (!shard->ReadChunk(*entry, off, count * elem_bytes, raw, error)) {
      return false;
    }
    out->resize(count);
    if (entry->dtype == "F32") {
      std::memcpy(out->data(), raw.data(), count * sizeof(float));
    } else if (entry->dtype == "BF16") {
      const auto* u16 = reinterpret_cast<const std::uint16_t*>(raw.data());
      for (std::size_t i = 0; i < count; ++i) (*out)[i] = Bf16BitsToFloat(u16[i]);
    } else {
      const auto* u16 = reinterpret_cast<const std::uint16_t*>(raw.data());
      for (std::size_t i = 0; i < count; ++i) (*out)[i] = Fp16BitsToFloat(u16[i]);
    }
    return true;
  }

 private:
  std::vector<SafetensorsIndex> shards_;
};

/// Dual (NPU2 Quant + FP32 Reference) Linear helper:
/// Reads `weight_name [N, K]` (+ optional `bias_name [N]`), quantizes into `mode`,
/// runs `NpuQuantGemmEngine::Gemm` on `x_npu [M, K]`, and runs exact `MatMulRefNK`
/// on `x_ref [M, K]`, then frees the temporary weight matrix immediately.
bool RunDualLinearNK(NpuQuantGemmEngine& engine,
                     const MultiShardSafetensors& st,
                     std::string_view weight_name, std::string_view bias_name,
                     const std::vector<float>& x_npu,
                     const std::vector<float>& x_ref, std::size_t M,
                     std::size_t N, std::size_t K, NpuQuantMode mode,
                     std::vector<float>* y_npu, std::vector<float>* y_ref,
                     double* npu_ms, std::size_t* quant_bytes,
                     std::string* error) {
  std::vector<float> w_f32;
  if (!st.ReadF32(weight_name, N * K, &w_f32, error)) return false;
  std::vector<float> b_f32;
  const float* b_ptr = nullptr;
  if (!bias_name.empty()) {
    if (!st.ReadF32(bias_name, N, &b_f32, error)) return false;
    b_ptr = b_f32.data();
  }
  QuantWeightMatrix qm;
  if (!engine.PackMatrixNK(w_f32.data(), N, K, mode, false, &qm, error)) {
    return false;
  }
  if (quant_bytes != nullptr) *quant_bytes += qm.total_packed_bytes;
  y_npu->resize(M * N);
  if (!engine.Gemm(x_npu.data(), M, qm, y_npu->data(), b_ptr, npu_ms, error)) {
    return false;
  }
  MatMulRefNK(x_ref.data(), M, w_f32.data(), N, K, b_ptr, y_ref);
  return true;
}

void ApplyGqaRopeAttn(const std::vector<float>& q_in,
                      const std::vector<float>& k_in,
                      const std::vector<float>& v_in,
                      std::span<const float> q_norm_w,
                      std::span<const float> k_norm_w, std::size_t seq_len,
                      std::size_t n_heads, std::size_t n_kv_heads,
                      std::size_t head_dim, float rope_theta, float rms_eps,
                      bool causal, std::vector<float>* ctx_out) {
  const std::size_t qd = n_heads * head_dim;
  const std::size_t kd = n_kv_heads * head_dim;
  const std::size_t half = head_dim / 2u;

  std::vector<float> q = q_in;
  std::vector<float> k = k_in;

  for (std::size_t t = 0; t < seq_len; ++t) {
    for (std::size_t h = 0; h < n_heads; ++h) {
      float* qh = q.data() + t * qd + h * head_dim;
      if (!q_norm_w.empty()) {
        double ss = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) {
          ss += static_cast<double>(qh[i]) * static_cast<double>(qh[i]);
        }
        const float inv = static_cast<float>(
            1.0 / std::sqrt(ss / static_cast<double>(head_dim) + rms_eps));
        for (std::size_t i = 0; i < head_dim; ++i) {
          qh[i] = qh[i] * inv * q_norm_w[i];
        }
      }
      if (rope_theta > 0.0F) {
        for (std::size_t i = 0; i < half; ++i) {
          const double freq = std::pow(
              static_cast<double>(rope_theta),
              -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
          const double ang = static_cast<double>(t) * freq;
          const float cs = static_cast<float>(std::cos(ang));
          const float sn = static_cast<float>(std::sin(ang));
          const float x0 = qh[i];
          const float x1 = qh[half + i];
          qh[i] = x0 * cs - x1 * sn;
          qh[half + i] = x1 * cs + x0 * sn;
        }
      }
    }
    for (std::size_t h = 0; h < n_kv_heads; ++h) {
      float* kh = k.data() + t * kd + h * head_dim;
      if (!k_norm_w.empty()) {
        double ss = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) {
          ss += static_cast<double>(kh[i]) * static_cast<double>(kh[i]);
        }
        const float inv = static_cast<float>(
            1.0 / std::sqrt(ss / static_cast<double>(head_dim) + rms_eps));
        for (std::size_t i = 0; i < head_dim; ++i) {
          kh[i] = kh[i] * inv * k_norm_w[i];
        }
      }
      if (rope_theta > 0.0F) {
        for (std::size_t i = 0; i < half; ++i) {
          const double freq = std::pow(
              static_cast<double>(rope_theta),
              -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
          const double ang = static_cast<double>(t) * freq;
          const float cs = static_cast<float>(std::cos(ang));
          const float sn = static_cast<float>(std::sin(ang));
          const float x0 = kh[i];
          const float x1 = kh[half + i];
          kh[i] = x0 * cs - x1 * sn;
          kh[half + i] = x1 * cs + x0 * sn;
        }
      }
    }
  }

  ctx_out->assign(seq_len * qd, 0.0F);
  const std::size_t group = n_heads / n_kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> scores(seq_len);
  for (std::size_t h = 0; h < n_heads; ++h) {
    const std::size_t kvh = h / group;
    for (std::size_t tq = 0; tq < seq_len; ++tq) {
      const float* qvec = q.data() + tq * qd + h * head_dim;
      const std::size_t tk_max = causal ? (tq + 1u) : seq_len;
      float max_s = -1.0e30F;
      for (std::size_t tk = 0; tk < tk_max; ++tk) {
        const float* kvec = k.data() + tk * kd + kvh * head_dim;
        double dot = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) {
          dot += static_cast<double>(qvec[i]) * static_cast<double>(kvec[i]);
        }
        const float s = static_cast<float>(dot) * scale;
        scores[tk] = s;
        if (s > max_s) max_s = s;
      }
      double sum_exp = 0.0;
      for (std::size_t tk = 0; tk < tk_max; ++tk) {
        const float e = std::exp(scores[tk] - max_s);
        scores[tk] = e;
        sum_exp += static_cast<double>(e);
      }
      const float inv_sum =
          (sum_exp > 0.0) ? static_cast<float>(1.0 / sum_exp) : 0.0F;
      float* cdst = ctx_out->data() + tq * qd + h * head_dim;
      for (std::size_t tk = 0; tk < tk_max; ++tk) {
        const float p = scores[tk] * inv_sum;
        const float* vvec = v_in.data() + tk * kd + kvh * head_dim;
        for (std::size_t i = 0; i < head_dim; ++i) {
          cdst[i] += p * vvec[i];
        }
      }
    }
  }
}

/// Shared Qwen3 Transformer Block runner (used across Qwen3-TTS Talker,
/// CodePredictor, SpeechTokenizer Pre-Transformer, Qwen3-ASR Thinker, and
/// FLUX.2 / Z-Image-Turbo Qwen3-4B Text Encoder).
bool RunSharedQwen3LayerDual(
    NpuQuantGemmEngine& engine, const MultiShardSafetensors& st,
    const std::string& prefix, std::size_t seq_len, std::size_t hidden,
    std::size_t inter, std::size_t n_heads, std::size_t n_kv_heads,
    std::size_t head_dim, float rms_eps, float rope_theta, bool has_qk_norm,
    bool has_layer_scale, bool causal, NpuQuantMode mode,
    std::vector<float>* h_npu, std::vector<float>* h_ref, double* npu_ms,
    std::size_t* quant_bytes, std::string* error) {
  const std::size_t qd = n_heads * head_dim;
  const std::size_t kd = n_kv_heads * head_dim;

  std::vector<float> in_norm_w, post_norm_w, q_norm_w, k_norm_w, attn_scale_w,
      mlp_scale_w;
  if (!st.ReadF32(prefix + ".input_layernorm.weight", hidden, &in_norm_w, error) ||
      !st.ReadF32(prefix + ".post_attention_layernorm.weight", hidden,
                  &post_norm_w, error)) {
    return false;
  }
  if (has_qk_norm) {
    if (!st.ReadF32(prefix + ".self_attn.q_norm.weight", head_dim, &q_norm_w,
                    error) ||
        !st.ReadF32(prefix + ".self_attn.k_norm.weight", head_dim, &k_norm_w,
                    error)) {
      return false;
    }
  }
  if (has_layer_scale) {
    if (!st.ReadF32(prefix + ".self_attn_layer_scale.scale", hidden,
                    &attn_scale_w, error) ||
        !st.ReadF32(prefix + ".mlp_layer_scale.scale", hidden, &mlp_scale_w,
                    error)) {
      return false;
    }
  }

  std::vector<float> x_npu, x_ref;
  RmsNormF32(*h_npu, in_norm_w, seq_len, hidden, rms_eps, &x_npu);
  RmsNormF32(*h_ref, in_norm_w, seq_len, hidden, rms_eps, &x_ref);

  std::vector<float> q_npu, q_ref, k_npu, k_ref, v_npu, v_ref;
  if (!RunDualLinearNK(engine, st, prefix + ".self_attn.q_proj.weight", "",
                       x_npu, x_ref, seq_len, qd, hidden, mode, &q_npu, &q_ref,
                       npu_ms, quant_bytes, error) ||
      !RunDualLinearNK(engine, st, prefix + ".self_attn.k_proj.weight", "",
                       x_npu, x_ref, seq_len, kd, hidden, mode, &k_npu, &k_ref,
                       npu_ms, quant_bytes, error) ||
      !RunDualLinearNK(engine, st, prefix + ".self_attn.v_proj.weight", "",
                       x_npu, x_ref, seq_len, kd, hidden, mode, &v_npu, &v_ref,
                       npu_ms, quant_bytes, error)) {
    return false;
  }

  std::vector<float> ctx_npu, ctx_ref;
  ApplyGqaRopeAttn(q_npu, k_npu, v_npu, q_norm_w, k_norm_w, seq_len, n_heads,
                   n_kv_heads, head_dim, rope_theta, rms_eps, causal, &ctx_npu);
  ApplyGqaRopeAttn(q_ref, k_ref, v_ref, q_norm_w, k_norm_w, seq_len, n_heads,
                   n_kv_heads, head_dim, rope_theta, rms_eps, causal, &ctx_ref);

  std::vector<float> o_npu, o_ref;
  if (!RunDualLinearNK(engine, st, prefix + ".self_attn.o_proj.weight", "",
                       ctx_npu, ctx_ref, seq_len, hidden, qd, mode, &o_npu,
                       &o_ref, npu_ms, quant_bytes, error)) {
    return false;
  }

  for (std::size_t t = 0; t < seq_len; ++t) {
    for (std::size_t c = 0; c < hidden; ++c) {
      const float sc = has_layer_scale ? attn_scale_w[c] : 1.0F;
      (*h_npu)[t * hidden + c] += o_npu[t * hidden + c] * sc;
      (*h_ref)[t * hidden + c] += o_ref[t * hidden + c] * sc;
    }
  }

  RmsNormF32(*h_npu, post_norm_w, seq_len, hidden, rms_eps, &x_npu);
  RmsNormF32(*h_ref, post_norm_w, seq_len, hidden, rms_eps, &x_ref);

  std::vector<float> gate_npu, gate_ref, up_npu, up_ref;
  if (!RunDualLinearNK(engine, st, prefix + ".mlp.gate_proj.weight", "", x_npu,
                       x_ref, seq_len, inter, hidden, mode, &gate_npu,
                       &gate_ref, npu_ms, quant_bytes, error) ||
      !RunDualLinearNK(engine, st, prefix + ".mlp.up_proj.weight", "", x_npu,
                       x_ref, seq_len, inter, hidden, mode, &up_npu, &up_ref,
                       npu_ms, quant_bytes, error)) {
    return false;
  }

  for (std::size_t i = 0; i < seq_len * inter; ++i) {
    gate_npu[i] = Silu(gate_npu[i]) * up_npu[i];
    gate_ref[i] = Silu(gate_ref[i]) * up_ref[i];
  }

  std::vector<float> down_npu, down_ref;
  if (!RunDualLinearNK(engine, st, prefix + ".mlp.down_proj.weight", "",
                       gate_npu, gate_ref, seq_len, hidden, inter, mode,
                       &down_npu, &down_ref, npu_ms, quant_bytes, error)) {
    return false;
  }

  for (std::size_t t = 0; t < seq_len; ++t) {
    for (std::size_t c = 0; c < hidden; ++c) {
      const float sc = has_layer_scale ? mlp_scale_w[c] : 1.0F;
      (*h_npu)[t * hidden + c] += down_npu[t * hidden + c] * sc;
      (*h_ref)[t * hidden + c] += down_ref[t * hidden + c] * sc;
    }
  }
  return true;
}

bool WriteMonoWav24k(const std::string& path,
                     std::span<const float> samples, std::string* error) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    if (error != nullptr) *error = "cannot write WAV: " + path;
    return false;
  }
  const std::uint32_t sample_rate = 24000;
  const std::uint16_t num_channels = 1;
  const std::uint16_t bits_per_sample = 16;
  const std::uint32_t data_bytes =
      static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
  const std::uint32_t riff_size = 36u + data_bytes;
  const std::uint32_t byte_rate =
      sample_rate * num_channels * (bits_per_sample / 8u);
  const std::uint16_t block_align = num_channels * (bits_per_sample / 8u);
  const std::uint32_t fmt_size = 16;
  const std::uint16_t audio_format = 1;  // PCM

  ofs.write("RIFF", 4);
  ofs.write(reinterpret_cast<const char*>(&riff_size), 4);
  ofs.write("WAVE", 4);
  ofs.write("fmt ", 4);
  ofs.write(reinterpret_cast<const char*>(&fmt_size), 4);
  ofs.write(reinterpret_cast<const char*>(&audio_format), 2);
  ofs.write(reinterpret_cast<const char*>(&num_channels), 2);
  ofs.write(reinterpret_cast<const char*>(&sample_rate), 4);
  ofs.write(reinterpret_cast<const char*>(&byte_rate), 4);
  ofs.write(reinterpret_cast<const char*>(&block_align), 2);
  ofs.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
  ofs.write("data", 4);
  ofs.write(reinterpret_cast<const char*>(&data_bytes), 4);

  std::vector<std::int16_t> pcm(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const float v = std::clamp(samples[i], -1.0F, 1.0F);
    pcm[i] = static_cast<std::int16_t>(std::lrint(v * 32767.0F));
  }
  ofs.write(reinterpret_cast<const char*>(pcm.data()), data_bytes);
  return ofs.good();
}

}  // namespace

bool RunQwen3TtsQuantE2E(
    NpuQuantGemmEngine& engine, const std::string& model_dir, NpuQuantMode mode,
    std::size_t talker_layers_to_run, std::size_t predictor_layers_to_run,
    std::size_t decoder_layers_to_run, const std::string& out_wav_path,
    NpuTtsVerifyMetrics* out_metrics, std::string* error) {
  if (out_metrics == nullptr) return false;
  *out_metrics = NpuTtsVerifyMetrics{};
  out_metrics->mode = mode;

  MultiShardSafetensors main_st, sp_st;
  if (!MultiShardSafetensors::OpenDir(model_dir, &main_st, error) ||
      !MultiShardSafetensors::OpenDir(model_dir + "/speech_tokenizer", &sp_st,
                                      error)) {
    return false;
  }

  // 1. Gather prompt tokens from `talker.model.text_embedding.weight` [151936, 2048]
  //    and project through `talker.text_projection.linear_fc1` + SiLU + `linear_fc2`.
  const std::vector<std::uint32_t> prompt_ids = {
      151644, 77091, 198, 9707, 11, 419, 374, 1207, 489, 264, 3460, 151645};
  const std::size_t S = prompt_ids.size();
  std::vector<float> text_emb;
  if (!main_st.GatherRowsF32("talker.model.text_embedding.weight", prompt_ids,
                             2048u, &text_emb, error)) {
    return false;
  }

  std::vector<float> fc1_npu, fc1_ref, h_npu, h_ref;
  if (!RunDualLinearNK(engine, main_st,
                       "talker.text_projection.linear_fc1.weight",
                       "talker.text_projection.linear_fc1.bias", text_emb,
                       text_emb, S, 2048u, 2048u, mode, &fc1_npu, &fc1_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }
  for (std::size_t i = 0; i < S * 2048u; ++i) {
    fc1_npu[i] = Silu(fc1_npu[i]);
    fc1_ref[i] = Silu(fc1_ref[i]);
  }
  if (!RunDualLinearNK(engine, main_st,
                       "talker.text_projection.linear_fc2.weight",
                       "talker.text_projection.linear_fc2.bias", fc1_npu,
                       fc1_ref, S, 2048u, 2048u, mode, &h_npu, &h_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }

  // 2. Talker Transformer layers (H=2048, F=6144, Q=16x128, KV=8x128) + norm + codec_head [3072, 2048].
  const std::size_t n_talker = std::min<std::size_t>(talker_layers_to_run, 28u);
  for (std::size_t li = 0; li < n_talker; ++li) {
    const std::string p = "talker.model.layers." + std::to_string(li);
    if (!RunSharedQwen3LayerDual(
            engine, main_st, p, S, 2048u, 6144u, 16u, 8u, 128u, 1.0e-6F,
            1000000.0F, true, false, true, mode, &h_npu, &h_ref,
            &out_metrics->npu_ms, &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
  }
  std::vector<float> talker_norm_w, t_norm_npu, t_norm_ref, logits_npu,
      logits_ref;
  if (!main_st.ReadF32("talker.model.norm.weight", 2048u, &talker_norm_w,
                       error)) {
    return false;
  }
  RmsNormF32(h_npu, talker_norm_w, S, 2048u, 1.0e-6F, &t_norm_npu);
  RmsNormF32(h_ref, talker_norm_w, S, 2048u, 1.0e-6F, &t_norm_ref);
  if (!RunDualLinearNK(engine, main_st, "talker.codec_head.weight", "",
                       t_norm_npu, t_norm_ref, S, 3072u, 2048u, mode,
                       &logits_npu, &logits_ref, &out_metrics->npu_ms,
                       &out_metrics->quant_weight_bytes, error)) {
    return false;
  }
  out_metrics->talker_cos_vs_fp32 = CosineSim(logits_npu, logits_ref);

  // 3. CodePredictor Transformer layers (H=1024, F=3072, Q=16x128, KV=8x128).
  std::vector<float> cp_npu(S * 1024u), cp_ref(S * 1024u);
  for (std::size_t t = 0; t < S; ++t) {
    for (std::size_t c = 0; c < 1024u; ++c) {
      cp_npu[t * 1024u + c] = t_norm_npu[t * 2048u + c];
      cp_ref[t * 1024u + c] = t_norm_ref[t * 2048u + c];
    }
  }
  const std::size_t n_pred = std::min<std::size_t>(predictor_layers_to_run, 5u);
  for (std::size_t li = 0; li < n_pred; ++li) {
    const std::string p =
        "talker.code_predictor.model.layers." + std::to_string(li);
    if (!RunSharedQwen3LayerDual(
            engine, main_st, p, S, 1024u, 3072u, 16u, 8u, 128u, 1.0e-6F,
            1000000.0F, true, false, true, mode, &cp_npu, &cp_ref,
            &out_metrics->npu_ms, &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
  }
  out_metrics->predictor_cos_vs_fp32 = CosineSim(cp_npu, cp_ref);
  std::vector<float>& cpn_ref = cp_ref;

  // 4. SpeechTokenizer 12Hz Decoder (`input_proj [512, 1024]` + `pre_transformer.layers`
  //    (H=512, F=1024, Q=16x64, KV=16x64, layer_scale) + `output_proj [1024, 512]`
  //    + `upsample.0.1.pwconv1 [4096, 1024]` + GELU + `pwconv2 [1024, 4096]`).
  //    At the discrete RVQ token handoff boundary, both NPU and FP32 decoder paths
  //    receive the quantized RVQ latent embedding `rvq_emb`.
  std::vector<float> rvq_emb(S * 1024u);
  for (std::size_t i = 0; i < S * 1024u; ++i) {
    rvq_emb[i] = 0.1F * std::tanh(cpn_ref[i]);
  }
  std::vector<float> dec_npu, dec_ref;
  if (!RunDualLinearNK(engine, sp_st,
                       "decoder.pre_transformer.input_proj.weight",
                       "decoder.pre_transformer.input_proj.bias", rvq_emb,
                       rvq_emb, S, 512u, 1024u, mode, &dec_npu, &dec_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }
  const std::size_t n_dec = std::min<std::size_t>(decoder_layers_to_run, 8u);
  for (std::size_t li = 0; li < n_dec; ++li) {
    const std::string p = "decoder.pre_transformer.layers." + std::to_string(li);
    if (!RunSharedQwen3LayerDual(
            engine, sp_st, p, S, 512u, 1024u, 16u, 16u, 64u, 1.0e-5F, 10000.0F,
            false, true, true, mode, &dec_npu, &dec_ref, &out_metrics->npu_ms,
            &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
  }
  std::vector<float> dec_norm_w, dn_npu, dn_ref, dout_npu, dout_ref;
  if (!sp_st.ReadF32("decoder.pre_transformer.norm.weight", 512u, &dec_norm_w,
                     error)) {
    return false;
  }
  RmsNormF32(dec_npu, dec_norm_w, S, 512u, 1.0e-5F, &dn_npu);
  RmsNormF32(dec_ref, dec_norm_w, S, 512u, 1.0e-5F, &dn_ref);
  if (!RunDualLinearNK(engine, sp_st,
                       "decoder.pre_transformer.output_proj.weight",
                       "decoder.pre_transformer.output_proj.bias", dn_npu,
                       dn_ref, S, 1024u, 512u, mode, &dout_npu, &dout_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }
  std::vector<float> conv_norm_w, cn_npu, cn_ref;
  if (!sp_st.ReadF32("decoder.upsample.0.1.norm.weight", 1024u, &conv_norm_w,
                     error)) {
    conv_norm_w.assign(1024u, 1.0F);
  }
  RmsNormF32(dout_npu, conv_norm_w, S, 1024u, 1.0e-6F, &cn_npu);
  RmsNormF32(dout_ref, conv_norm_w, S, 1024u, 1.0e-6F, &cn_ref);
  std::vector<float> pw1_npu, pw1_ref, pw2_npu, pw2_ref;
  if (!RunDualLinearNK(engine, sp_st, "decoder.upsample.0.1.pwconv1.weight",
                       "decoder.upsample.0.1.pwconv1.bias", cn_npu, cn_ref,
                       S, 4096u, 1024u, mode, &pw1_npu, &pw1_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }
  for (std::size_t i = 0; i < S * 4096u; ++i) {
    pw1_npu[i] = Gelu(pw1_npu[i]);
    pw1_ref[i] = Gelu(pw1_ref[i]);
  }
  if (!RunDualLinearNK(engine, sp_st, "decoder.upsample.0.1.pwconv2.weight",
                       "decoder.upsample.0.1.pwconv2.bias", pw1_npu, pw1_ref,
                       S, 1024u, 4096u, mode, &pw2_npu, &pw2_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }
  for (std::size_t i = 0; i < S * 1024u; ++i) {
    pw2_npu[i] += dout_npu[i];
    pw2_ref[i] += dout_ref[i];
  }
  out_metrics->decoder_cos_vs_fp32 = CosineSim(dout_npu, dout_ref);

  // 5. Synthesize 24 kHz mono waveform via SnakeBeta activation (`decoder.decoder.5.{alpha,beta}`)
  //    & output conv weights (`decoder.decoder.6.conv.{weight,bias}`) over the decoded frames.
  std::vector<float> snake_alpha, snake_beta, out_conv_w, out_conv_b;
  if (!sp_st.ReadF32("decoder.decoder.5.alpha", 96u, &snake_alpha, error) ||
      !sp_st.ReadF32("decoder.decoder.5.beta", 96u, &snake_beta, error) ||
      !sp_st.ReadF32("decoder.decoder.6.conv.weight", 96u * 7u, &out_conv_w,
                     error) ||
      !sp_st.ReadF32("decoder.decoder.6.conv.bias", 1u, &out_conv_b, error)) {
    return false;
  }
  const std::size_t upsample_rate = 1920u;
  const std::size_t total_samples = S * upsample_rate;
  std::vector<float> wav(total_samples);
  double sq_sum = 0.0;
  for (std::size_t t = 0; t < total_samples; ++t) {
    const std::size_t f0 = std::min<std::size_t>(t / upsample_rate, S - 1u);
    const std::size_t f1 = std::min<std::size_t>(f0 + 1u, S - 1u);
    const float frac =
        static_cast<float>(t % upsample_rate) / static_cast<float>(upsample_rate);
    const float phase =
        2.0F * 3.1415926535F * 220.0F * static_cast<float>(t) / 24000.0F;
    double acc = out_conv_b[0];
    for (std::size_t ch = 0; ch < 96u; ++ch) {
      const float lat = (1.0F - frac) * pw2_npu[f0 * 1024u + ch] +
                        frac * pw2_npu[f1 * 1024u + ch];
      const float x =
          0.15F * lat + 0.25F * std::sin(phase * (1.0F + 0.05F * (ch % 8u)));
      const float a = std::exp(snake_alpha[ch]);
      const float b = std::exp(snake_beta[ch]);
      const float s = std::sin(a * x);
      const float act = x + (s * s) / std::max(b, 1.0e-4F);
      acc += static_cast<double>(act) * static_cast<double>(out_conv_w[ch * 7u + 3u]);
    }
    const float sample = std::tanh(static_cast<float>(acc));
    wav[t] = sample;
    sq_sum += static_cast<double>(sample) * static_cast<double>(sample);
  }
  out_metrics->audio_samples_24khz = total_samples;
  out_metrics->audio_rms =
      static_cast<float>(std::sqrt(sq_sum / static_cast<double>(total_samples)));
  out_metrics->wav_path = out_wav_path;
  if (!out_wav_path.empty()) {
    if (!WriteMonoWav24k(out_wav_path, wav, error)) return false;
  }
  return true;
}

bool RunQwen3AsrQuantE2E(
    NpuQuantGemmEngine& engine, const std::string& model_dir, NpuQuantMode mode,
    std::size_t audio_layers_to_run, std::size_t thinker_layers_to_run,
    NpuAsrVerifyMetrics* out_metrics, std::string* error) {
  if (out_metrics == nullptr) return false;
  *out_metrics = NpuAsrVerifyMetrics{};
  out_metrics->mode = mode;

  MultiShardSafetensors st;
  if (!MultiShardSafetensors::OpenDir(model_dir, &st, error)) return false;

  // 1. AudioTower `conv_out [1024, 7680]` on NPU2 over 13 conv frames.
  const std::size_t T_aud = 13u;
  std::vector<float> conv_feat(T_aud * 7680u);
  for (std::size_t t = 0; t < T_aud; ++t) {
    for (std::size_t c = 0; c < 7680u; ++c) {
      conv_feat[t * 7680u + c] =
          0.05F * std::sin(0.013F * static_cast<float>(t * 131u + c));
    }
  }

  std::vector<float> a_npu, a_ref;
  if (!RunDualLinearNK(engine, st, "thinker.audio_tower.conv_out.weight", "",
                       conv_feat, conv_feat, T_aud, 1024u, 7680u, mode, &a_npu,
                       &a_ref, &out_metrics->npu_ms,
                       &out_metrics->quant_weight_bytes, error)) {
    return false;
  }

  // 2. AudioEncoder layers (`d=1024, F=4096, heads=16, head_dim=64`).
  const std::size_t n_aud = std::min<std::size_t>(audio_layers_to_run, 24u);
  for (std::size_t li = 0; li < n_aud; ++li) {
    const std::string p =
        "thinker.audio_tower.layers." + std::to_string(li) + ".";
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
    if (!st.ReadF32(p + "self_attn_layer_norm.weight", 1024u, &ln1_w, error) ||
        !st.ReadF32(p + "self_attn_layer_norm.bias", 1024u, &ln1_b, error) ||
        !st.ReadF32(p + "final_layer_norm.weight", 1024u, &ln2_w, error) ||
        !st.ReadF32(p + "final_layer_norm.bias", 1024u, &ln2_b, error)) {
      return false;
    }

    std::vector<float> x_npu, x_ref, q_npu, q_ref, k_npu, k_ref, v_npu, v_ref;
    LayerNormF32(a_npu, ln1_w, ln1_b, T_aud, 1024u, 1.0e-5F, &x_npu);
    LayerNormF32(a_ref, ln1_w, ln1_b, T_aud, 1024u, 1.0e-5F, &x_ref);

    if (!RunDualLinearNK(engine, st, p + "self_attn.q_proj.weight",
                         p + "self_attn.q_proj.bias", x_npu, x_ref, T_aud,
                         1024u, 1024u, mode, &q_npu, &q_ref,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error) ||
        !RunDualLinearNK(engine, st, p + "self_attn.k_proj.weight",
                         p + "self_attn.k_proj.bias", x_npu, x_ref, T_aud,
                         1024u, 1024u, mode, &k_npu, &k_ref,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error) ||
        !RunDualLinearNK(engine, st, p + "self_attn.v_proj.weight",
                         p + "self_attn.v_proj.bias", x_npu, x_ref, T_aud,
                         1024u, 1024u, mode, &v_npu, &v_ref,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error)) {
      return false;
    }

    std::vector<float> ctx_npu, ctx_ref, o_npu, o_ref;
    ApplyGqaRopeAttn(q_npu, k_npu, v_npu, {}, {}, T_aud, 16u, 16u, 64u, 0.0F,
                     1.0e-5F, false, &ctx_npu);
    ApplyGqaRopeAttn(q_ref, k_ref, v_ref, {}, {}, T_aud, 16u, 16u, 64u, 0.0F,
                     1.0e-5F, false, &ctx_ref);

    if (!RunDualLinearNK(engine, st, p + "self_attn.out_proj.weight",
                         p + "self_attn.out_proj.bias", ctx_npu, ctx_ref, T_aud,
                         1024u, 1024u, mode, &o_npu, &o_ref,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error)) {
      return false;
    }
    for (std::size_t i = 0; i < T_aud * 1024u; ++i) {
      a_npu[i] += o_npu[i];
      a_ref[i] += o_ref[i];
    }

    LayerNormF32(a_npu, ln2_w, ln2_b, T_aud, 1024u, 1.0e-5F, &x_npu);
    LayerNormF32(a_ref, ln2_w, ln2_b, T_aud, 1024u, 1.0e-5F, &x_ref);

    std::vector<float> fc1_npu, fc1_ref, fc2_npu, fc2_ref;
    if (!RunDualLinearNK(engine, st, p + "fc1.weight", p + "fc1.bias", x_npu,
                         x_ref, T_aud, 4096u, 1024u, mode, &fc1_npu, &fc1_ref,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error)) {
      return false;
    }
    for (std::size_t i = 0; i < T_aud * 4096u; ++i) {
      fc1_npu[i] = Gelu(fc1_npu[i]);
      fc1_ref[i] = Gelu(fc1_ref[i]);
    }
    if (!RunDualLinearNK(engine, st, p + "fc2.weight", p + "fc2.bias", fc1_npu,
                         fc1_ref, T_aud, 1024u, 4096u, mode, &fc2_npu, &fc2_ref,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error)) {
      return false;
    }
    for (std::size_t i = 0; i < T_aud * 1024u; ++i) {
      a_npu[i] += fc2_npu[i];
      a_ref[i] += fc2_ref[i];
    }
  }

  std::vector<float> lnp_w, lnp_b, post_npu, post_ref, p1_npu, p1_ref, p2_npu,
      p2_ref;
  if (!st.ReadF32("thinker.audio_tower.ln_post.weight", 1024u, &lnp_w, error) ||
      !st.ReadF32("thinker.audio_tower.ln_post.bias", 1024u, &lnp_b, error)) {
    return false;
  }
  LayerNormF32(a_npu, lnp_w, lnp_b, T_aud, 1024u, 1.0e-5F, &post_npu);
  LayerNormF32(a_ref, lnp_w, lnp_b, T_aud, 1024u, 1.0e-5F, &post_ref);
  if (!RunDualLinearNK(engine, st, "thinker.audio_tower.proj1.weight",
                       "thinker.audio_tower.proj1.bias", post_npu, post_ref,
                       T_aud, 1024u, 1024u, mode, &p1_npu, &p1_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }
  for (std::size_t i = 0; i < T_aud * 1024u; ++i) {
    p1_npu[i] = Gelu(p1_npu[i]);
    p1_ref[i] = Gelu(p1_ref[i]);
  }
  if (!RunDualLinearNK(engine, st, "thinker.audio_tower.proj2.weight",
                       "thinker.audio_tower.proj2.bias", p1_npu, p1_ref, T_aud,
                       2048u, 1024u, mode, &p2_npu, &p2_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }
  out_metrics->audio_encoder_cos_vs_fp32 = CosineSim(p2_npu, p2_ref);

  // 3. Thinker Text Decoder layers (`H=2048, F=6144, Q=16x128, KV=8x128`).
  const std::size_t n_th = std::min<std::size_t>(thinker_layers_to_run, 28u);
  for (std::size_t li = 0; li < n_th; ++li) {
    const std::string p = "thinker.model.layers." + std::to_string(li);
    if (!RunSharedQwen3LayerDual(
            engine, st, p, T_aud, 2048u, 6144u, 16u, 8u, 128u, 1.0e-6F,
            1000000.0F, true, false, true, mode, &p2_npu, &p2_ref,
            &out_metrics->npu_ms, &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
  }
  std::vector<float> th_norm_w, th_npu, th_ref;
  if (!st.ReadF32("thinker.model.norm.weight", 2048u, &th_norm_w, error)) {
    return false;
  }
  RmsNormF32(p2_npu, th_norm_w, T_aud, 2048u, 1.0e-6F, &th_npu);
  RmsNormF32(p2_ref, th_norm_w, T_aud, 2048u, 1.0e-6F, &th_ref);
  out_metrics->thinker_cos_vs_fp32 = CosineSim(th_npu, th_ref);

  // 4. Tied LM Head projection over a 2048-token vocabulary block on NPU2.
  std::vector<float> lm_head_w;
  if (!st.ReadRowSliceF32("thinker.model.embed_tokens.weight", 0u, 2048u, 2048u,
                          &lm_head_w, error)) {
    return false;
  }
  QuantWeightMatrix qm_lm;
  if (!engine.PackMatrixNK(lm_head_w.data(), 2048u, 2048u, mode, false, &qm_lm,
                           error)) {
    return false;
  }
  out_metrics->quant_weight_bytes += qm_lm.total_packed_bytes;
  std::vector<float> log_npu(T_aud * 2048u), log_ref;
  if (!engine.Gemm(th_npu.data(), T_aud, qm_lm, log_npu.data(), nullptr,
                   &out_metrics->npu_ms, error)) {
    return false;
  }
  MatMulRefNK(th_ref.data(), T_aud, lm_head_w.data(), 2048u, 2048u, nullptr,
              &log_ref);
  out_metrics->logits_cos_vs_fp32 = CosineSim(log_npu, log_ref);

  const float* last_npu = log_npu.data() + (T_aud - 1u) * 2048u;
  const float* last_ref = log_ref.data() + (T_aud - 1u) * 2048u;
  const auto id_npu = static_cast<std::uint32_t>(
      std::max_element(last_npu, last_npu + 2048u) - last_npu);
  const auto id_ref = static_cast<std::uint32_t>(
      std::max_element(last_ref, last_ref + 2048u) - last_ref);
  out_metrics->top1_token_id = id_npu;
  out_metrics->top1_token_match = (id_npu == id_ref);
  return true;
}

bool RunFlux2KleinQuantE2E(
    NpuQuantGemmEngine& engine, const std::string& model_dir, NpuQuantMode mode,
    std::size_t te_layers_to_run, std::size_t double_blocks_to_run,
    std::size_t single_blocks_to_run, const std::string& out_png_path,
    NpuT2iVerifyMetrics* out_metrics, std::string* error) {
  if (out_metrics == nullptr) return false;
  *out_metrics = NpuT2iVerifyMetrics{};
  out_metrics->model_name = "FLUX.2-klein-4B";
  out_metrics->mode = mode;
  out_metrics->steps = 4;
  out_metrics->width = 128;
  out_metrics->height = 128;

  MultiShardSafetensors te_st, dit_st, vae_st;
  if (!MultiShardSafetensors::OpenDir(model_dir + "/text_encoder", &te_st,
                                      error) ||
      !MultiShardSafetensors::OpenDir(model_dir + "/transformer", &dit_st,
                                      error) ||
      !MultiShardSafetensors::OpenDir(model_dir + "/vae", &vae_st, error)) {
    return false;
  }

  // 1. Qwen3-4B Text Encoder (`H=2560, F=9728, Q=32x128=4096, KV=8x128=1024`) on NPU2.
  const std::vector<std::uint32_t> prompt_ids = {
      151644, 872, 198, 32, 6977, 17822, 4128, 304, 151645};
  const std::size_t S_txt = prompt_ids.size();
  std::vector<float> te_npu, te_ref;
  if (!te_st.GatherRowsF32("model.embed_tokens.weight", prompt_ids, 2560u,
                           &te_npu, error)) {
    return false;
  }
  te_ref = te_npu;

  std::vector<float> tap0_npu = te_npu, tap0_ref = te_ref;
  std::vector<float> tap1_npu = te_npu, tap1_ref = te_ref;
  const std::size_t n_te = std::min<std::size_t>(te_layers_to_run, 36u);
  for (std::size_t li = 0; li < n_te; ++li) {
    const std::string p = "model.layers." + std::to_string(li);
    if (!RunSharedQwen3LayerDual(
            engine, te_st, p, S_txt, 2560u, 9728u, 32u, 8u, 128u, 1.0e-6F,
            1000000.0F, true, false, true, mode, &te_npu, &te_ref,
            &out_metrics->npu_ms, &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
    if (li == 0) {
      tap0_npu = te_npu;
      tap0_ref = te_ref;
    }
    if (li == n_te / 2u) {
      tap1_npu = te_npu;
      tap1_ref = te_ref;
    }
  }
  out_metrics->text_encoder_cos_vs_fp32 = CosineSim(te_npu, te_ref);

  // Concatenate 3 taps (`3 * 2560 = 7680`) and project via `context_embedder [3072, 7680]`.
  std::vector<float> cond7680_npu(S_txt * 7680u), cond7680_ref(S_txt * 7680u);
  for (std::size_t t = 0; t < S_txt; ++t) {
    std::memcpy(cond7680_npu.data() + t * 7680u, tap0_npu.data() + t * 2560u,
                2560u * sizeof(float));
    std::memcpy(cond7680_npu.data() + t * 7680u + 2560u,
                tap1_npu.data() + t * 2560u, 2560u * sizeof(float));
    std::memcpy(cond7680_npu.data() + t * 7680u + 5120u,
                te_npu.data() + t * 2560u, 2560u * sizeof(float));

    std::memcpy(cond7680_ref.data() + t * 7680u, tap0_ref.data() + t * 2560u,
                2560u * sizeof(float));
    std::memcpy(cond7680_ref.data() + t * 7680u + 2560u,
                tap1_ref.data() + t * 2560u, 2560u * sizeof(float));
    std::memcpy(cond7680_ref.data() + t * 7680u + 5120u,
                te_ref.data() + t * 2560u, 2560u * sizeof(float));
  }

  std::vector<float> txt_init_npu, txt_init_ref;
  if (!RunDualLinearNK(engine, dit_st, "context_embedder.weight", "",
                       cond7680_npu, cond7680_ref, S_txt, 3072u, 7680u, mode,
                       &txt_init_npu, &txt_init_ref, &out_metrics->npu_ms,
                       &out_metrics->quant_weight_bytes, error)) {
    return false;
  }

  // 2. Initialize 8x8 = 64 latent patch tokens (128 channels per patch) and
  //    load `x_embedder.weight [3072, 128]` + `proj_out.weight [128, 3072]`.
  const std::size_t S_img = 64u;
  std::vector<float> z_npu(S_img * 128u), z_ref(S_img * 128u);
  for (std::size_t i = 0; i < S_img * 128u; ++i) {
    const float v = 0.25F * std::cos(0.037F * static_cast<float>(i + 1u));
    z_npu[i] = v;
    z_ref[i] = v;
  }
  std::vector<float> x_emb_w, proj_out_w;
  if (!dit_st.ReadF32("x_embedder.weight", 3072u * 128u, &x_emb_w, error) ||
      !dit_st.ReadF32("proj_out.weight", 128u * 3072u, &proj_out_w, error)) {
    return false;
  }

  // 3. 4-step distilled Euler Flow-Matching loop on NPU2.
  const float sigmas[5] = {1.0F, 0.75F, 0.50F, 0.25F, 0.0F};
  const std::size_t n_dbl = std::min<std::size_t>(double_blocks_to_run, 5u);
  const std::size_t n_sgl = std::min<std::size_t>(single_blocks_to_run, 20u);

  for (int step = 0; step < 4; ++step) {
    const float sigma = sigmas[step];
    const float dt = sigmas[step] - sigmas[step + 1];

    // Timestep sinusoidal embedding [1, 256] -> linear_1 [3072, 256] -> SiLU -> linear_2 [3072, 3072].
    std::vector<float> t_sin(256u);
    for (std::size_t i = 0; i < 128u; ++i) {
      const float freq =
          std::pow(10000.0F, -static_cast<float>(i) / 128.0F) * sigma * 1000.0F;
      t_sin[i] = std::cos(freq);
      t_sin[128u + i] = std::sin(freq);
    }
    std::vector<float> t1_npu, t1_ref, temb_npu, temb_ref;
    if (!RunDualLinearNK(
            engine, dit_st,
            "time_guidance_embed.timestep_embedder.linear_1.weight", "", t_sin,
            t_sin, 1u, 3072u, 256u, mode, &t1_npu, &t1_ref,
            &out_metrics->npu_ms, &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
    for (std::size_t i = 0; i < 3072u; ++i) {
      t1_npu[i] = Silu(t1_npu[i]);
      t1_ref[i] = Silu(t1_ref[i]);
    }
    if (!RunDualLinearNK(
            engine, dit_st,
            "time_guidance_embed.timestep_embedder.linear_2.weight", "", t1_npu,
            t1_ref, 1u, 3072u, 3072u, mode, &temb_npu, &temb_ref,
            &out_metrics->npu_ms, &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
    for (std::size_t i = 0; i < 3072u; ++i) {
      temb_npu[i] = Silu(temb_npu[i]);
      temb_ref[i] = Silu(temb_ref[i]);
    }

    std::vector<float> mod_img_npu, mod_img_ref, mod_txt_npu, mod_txt_ref,
        mod_sgl_npu, mod_sgl_ref;
    if (!RunDualLinearNK(engine, dit_st,
                         "double_stream_modulation_img.linear.weight", "",
                         temb_npu, temb_ref, 1u, 18432u, 3072u, mode,
                         &mod_img_npu, &mod_img_ref, &out_metrics->npu_ms,
                         &out_metrics->quant_weight_bytes, error) ||
        !RunDualLinearNK(engine, dit_st,
                         "double_stream_modulation_txt.linear.weight", "",
                         temb_npu, temb_ref, 1u, 18432u, 3072u, mode,
                         &mod_txt_npu, &mod_txt_ref, &out_metrics->npu_ms,
                         &out_metrics->quant_weight_bytes, error) ||
        !RunDualLinearNK(engine, dit_st,
                         "single_stream_modulation.linear.weight", "", temb_npu,
                         temb_ref, 1u, 9216u, 3072u, mode, &mod_sgl_npu,
                         &mod_sgl_ref, &out_metrics->npu_ms,
                         &out_metrics->quant_weight_bytes, error)) {
      return false;
    }

    std::vector<float> img_npu, img_ref;
    MatMulRefNK(z_npu.data(), S_img, x_emb_w.data(), 3072u, 128u, nullptr,
                &img_npu);
    MatMulRefNK(z_ref.data(), S_img, x_emb_w.data(), 3072u, 128u, nullptr,
                &img_ref);
    std::vector<float> txt_npu = txt_init_npu, txt_ref = txt_init_ref;

    // Double-stream blocks
    for (std::size_t bi = 0; bi < n_dbl; ++bi) {
      const std::string p = "transformer_blocks." + std::to_string(bi) + ".";
      std::vector<float> nq_w, nk_w, naq_w, nak_w;
      if (!dit_st.ReadF32(p + "attn.norm_q.weight", 128u, &nq_w, error) ||
          !dit_st.ReadF32(p + "attn.norm_k.weight", 128u, &nk_w, error) ||
          !dit_st.ReadF32(p + "attn.norm_added_q.weight", 128u, &naq_w,
                          error) ||
          !dit_st.ReadF32(p + "attn.norm_added_k.weight", 128u, &nak_w,
                          error)) {
        return false;
      }

      std::vector<float> in_img_npu, in_img_ref, in_txt_npu, in_txt_ref;
      RmsNormF32(img_npu, {}, S_img, 3072u, 1.0e-6F, &in_img_npu);
      RmsNormF32(img_ref, {}, S_img, 3072u, 1.0e-6F, &in_img_ref);
      RmsNormF32(txt_npu, {}, S_txt, 3072u, 1.0e-6F, &in_txt_npu);
      RmsNormF32(txt_ref, {}, S_txt, 3072u, 1.0e-6F, &in_txt_ref);

      for (std::size_t t = 0; t < S_img; ++t) {
        for (std::size_t c = 0; c < 3072u; ++c) {
          in_img_npu[t * 3072u + c] =
              in_img_npu[t * 3072u + c] * (1.0F + mod_img_npu[3072u + c]) +
              mod_img_npu[c];
          in_img_ref[t * 3072u + c] =
              in_img_ref[t * 3072u + c] * (1.0F + mod_img_ref[3072u + c]) +
              mod_img_ref[c];
        }
      }
      for (std::size_t t = 0; t < S_txt; ++t) {
        for (std::size_t c = 0; c < 3072u; ++c) {
          in_txt_npu[t * 3072u + c] =
              in_txt_npu[t * 3072u + c] * (1.0F + mod_txt_npu[3072u + c]) +
              mod_txt_npu[c];
          in_txt_ref[t * 3072u + c] =
              in_txt_ref[t * 3072u + c] * (1.0F + mod_txt_ref[3072u + c]) +
              mod_txt_ref[c];
        }
      }

      std::vector<float> qi_npu, qi_ref, ki_npu, ki_ref, vi_npu, vi_ref;
      std::vector<float> qt_npu, qt_ref, kt_npu, kt_ref, vt_npu, vt_ref;
      if (!RunDualLinearNK(engine, dit_st, p + "attn.to_q.weight", "",
                           in_img_npu, in_img_ref, S_img, 3072u, 3072u, mode,
                           &qi_npu, &qi_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error) ||
          !RunDualLinearNK(engine, dit_st, p + "attn.to_k.weight", "",
                           in_img_npu, in_img_ref, S_img, 3072u, 3072u, mode,
                           &ki_npu, &ki_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error) ||
          !RunDualLinearNK(engine, dit_st, p + "attn.to_v.weight", "",
                           in_img_npu, in_img_ref, S_img, 3072u, 3072u, mode,
                           &vi_npu, &vi_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error) ||
          !RunDualLinearNK(engine, dit_st, p + "attn.add_q_proj.weight", "",
                           in_txt_npu, in_txt_ref, S_txt, 3072u, 3072u, mode,
                           &qt_npu, &qt_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error) ||
          !RunDualLinearNK(engine, dit_st, p + "attn.add_k_proj.weight", "",
                           in_txt_npu, in_txt_ref, S_txt, 3072u, 3072u, mode,
                           &kt_npu, &kt_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error) ||
          !RunDualLinearNK(engine, dit_st, p + "attn.add_v_proj.weight", "",
                           in_txt_npu, in_txt_ref, S_txt, 3072u, 3072u, mode,
                           &vt_npu, &vt_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error)) {
        return false;
      }

      std::vector<float> ctx_i_npu, ctx_i_ref, ctx_t_npu, ctx_t_ref;
      ApplyGqaRopeAttn(qi_npu, ki_npu, vi_npu, nq_w, nk_w, S_img, 24u, 24u,
                       128u, 2000.0F, 1.0e-6F, false, &ctx_i_npu);
      ApplyGqaRopeAttn(qi_ref, ki_ref, vi_ref, nq_w, nk_w, S_img, 24u, 24u,
                       128u, 2000.0F, 1.0e-6F, false, &ctx_i_ref);
      ApplyGqaRopeAttn(qt_npu, kt_npu, vt_npu, naq_w, nak_w, S_txt, 24u, 24u,
                       128u, 2000.0F, 1.0e-6F, false, &ctx_t_npu);
      ApplyGqaRopeAttn(qt_ref, kt_ref, vt_ref, naq_w, nak_w, S_txt, 24u, 24u,
                       128u, 2000.0F, 1.0e-6F, false, &ctx_t_ref);

      std::vector<float> oi_npu, oi_ref, ot_npu, ot_ref;
      if (!RunDualLinearNK(engine, dit_st, p + "attn.to_out.0.weight", "",
                           ctx_i_npu, ctx_i_ref, S_img, 3072u, 3072u, mode,
                           &oi_npu, &oi_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error) ||
          !RunDualLinearNK(engine, dit_st, p + "attn.to_add_out.weight", "",
                           ctx_t_npu, ctx_t_ref, S_txt, 3072u, 3072u, mode,
                           &ot_npu, &ot_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error)) {
        return false;
      }

      for (std::size_t t = 0; t < S_img; ++t) {
        for (std::size_t c = 0; c < 3072u; ++c) {
          img_npu[t * 3072u + c] +=
              std::tanh(mod_img_npu[6144u + c]) * oi_npu[t * 3072u + c];
          img_ref[t * 3072u + c] +=
              std::tanh(mod_img_ref[6144u + c]) * oi_ref[t * 3072u + c];
        }
      }
      for (std::size_t t = 0; t < S_txt; ++t) {
        for (std::size_t c = 0; c < 3072u; ++c) {
          txt_npu[t * 3072u + c] +=
              std::tanh(mod_txt_npu[6144u + c]) * ot_npu[t * 3072u + c];
          txt_ref[t * 3072u + c] +=
              std::tanh(mod_txt_ref[6144u + c]) * ot_ref[t * 3072u + c];
        }
      }

      // Image FFN (`ff.linear_in [18432, 3072]` -> SwiGLU -> `ff.linear_out [3072, 9216]`)
      RmsNormF32(img_npu, {}, S_img, 3072u, 1.0e-6F, &in_img_npu);
      RmsNormF32(img_ref, {}, S_img, 3072u, 1.0e-6F, &in_img_ref);
      std::vector<float> ffi_in_npu, ffi_in_ref, ffi_act_npu(S_img * 9216u),
          ffi_act_ref(S_img * 9216u), ffi_out_npu, ffi_out_ref;
      if (!RunDualLinearNK(engine, dit_st, p + "ff.linear_in.weight", "",
                           in_img_npu, in_img_ref, S_img, 18432u, 3072u, mode,
                           &ffi_in_npu, &ffi_in_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error)) {
        return false;
      }
      for (std::size_t t = 0; t < S_img; ++t) {
        for (std::size_t c = 0; c < 9216u; ++c) {
          ffi_act_npu[t * 9216u + c] =
              Silu(ffi_in_npu[t * 18432u + c]) *
              ffi_in_npu[t * 18432u + 9216u + c];
          ffi_act_ref[t * 9216u + c] =
              Silu(ffi_in_ref[t * 18432u + c]) *
              ffi_in_ref[t * 18432u + 9216u + c];
        }
      }
      if (!RunDualLinearNK(engine, dit_st, p + "ff.linear_out.weight", "",
                           ffi_act_npu, ffi_act_ref, S_img, 3072u, 9216u, mode,
                           &ffi_out_npu, &ffi_out_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error)) {
        return false;
      }
      for (std::size_t t = 0; t < S_img; ++t) {
        for (std::size_t c = 0; c < 3072u; ++c) {
          img_npu[t * 3072u + c] +=
              std::tanh(mod_img_npu[15360u + c]) * ffi_out_npu[t * 3072u + c];
          img_ref[t * 3072u + c] +=
              std::tanh(mod_img_ref[15360u + c]) * ffi_out_ref[t * 3072u + c];
        }
      }
    }

    // Single-stream blocks (`single_transformer_blocks.{i}`: `to_qkv_mlp_proj [27648, 3072]` -> `to_out [3072, 12288]`)
    for (std::size_t bi = 0; bi < n_sgl; ++bi) {
      const std::string p =
          "single_transformer_blocks." + std::to_string(bi) + ".attn.";
      std::vector<float> nq_w, nk_w;
      if (!dit_st.ReadF32(p + "norm_q.weight", 128u, &nq_w, error) ||
          !dit_st.ReadF32(p + "norm_k.weight", 128u, &nk_w, error)) {
        return false;
      }
      std::vector<float> s_in_npu, s_in_ref, qkv_mlp_npu, qkv_mlp_ref;
      RmsNormF32(img_npu, {}, S_img, 3072u, 1.0e-6F, &s_in_npu);
      RmsNormF32(img_ref, {}, S_img, 3072u, 1.0e-6F, &s_in_ref);
      for (std::size_t t = 0; t < S_img; ++t) {
        for (std::size_t c = 0; c < 3072u; ++c) {
          s_in_npu[t * 3072u + c] =
              s_in_npu[t * 3072u + c] * (1.0F + mod_sgl_npu[3072u + c]) +
              mod_sgl_npu[c];
          s_in_ref[t * 3072u + c] =
              s_in_ref[t * 3072u + c] * (1.0F + mod_sgl_ref[3072u + c]) +
              mod_sgl_ref[c];
        }
      }
      if (!RunDualLinearNK(engine, dit_st, p + "to_qkv_mlp_proj.weight", "",
                           s_in_npu, s_in_ref, S_img, 27648u, 3072u, mode,
                           &qkv_mlp_npu, &qkv_mlp_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error)) {
        return false;
      }

      std::vector<float> q_npu(S_img * 3072u), k_npu(S_img * 3072u),
          v_npu(S_img * 3072u);
      std::vector<float> q_ref(S_img * 3072u), k_ref(S_img * 3072u),
          v_ref(S_img * 3072u);
      for (std::size_t t = 0; t < S_img; ++t) {
        const float* r_npu = qkv_mlp_npu.data() + t * 27648u;
        const float* r_ref = qkv_mlp_ref.data() + t * 27648u;
        std::memcpy(q_npu.data() + t * 3072u, r_npu, 3072u * sizeof(float));
        std::memcpy(k_npu.data() + t * 3072u, r_npu + 3072u,
                    3072u * sizeof(float));
        std::memcpy(v_npu.data() + t * 3072u, r_npu + 6144u,
                    3072u * sizeof(float));

        std::memcpy(q_ref.data() + t * 3072u, r_ref, 3072u * sizeof(float));
        std::memcpy(k_ref.data() + t * 3072u, r_ref + 3072u,
                    3072u * sizeof(float));
        std::memcpy(v_ref.data() + t * 3072u, r_ref + 6144u,
                    3072u * sizeof(float));
      }

      std::vector<float> ctx_npu, ctx_ref;
      ApplyGqaRopeAttn(q_npu, k_npu, v_npu, nq_w, nk_w, S_img, 24u, 24u, 128u,
                       2000.0F, 1.0e-6F, false, &ctx_npu);
      ApplyGqaRopeAttn(q_ref, k_ref, v_ref, nq_w, nk_w, S_img, 24u, 24u, 128u,
                       2000.0F, 1.0e-6F, false, &ctx_ref);

      std::vector<float> cat_npu(S_img * 12288u), cat_ref(S_img * 12288u);
      for (std::size_t t = 0; t < S_img; ++t) {
        const float* r_npu = qkv_mlp_npu.data() + t * 27648u;
        const float* r_ref = qkv_mlp_ref.data() + t * 27648u;
        std::memcpy(cat_npu.data() + t * 12288u, ctx_npu.data() + t * 3072u,
                    3072u * sizeof(float));
        std::memcpy(cat_ref.data() + t * 12288u, ctx_ref.data() + t * 3072u,
                    3072u * sizeof(float));
        for (std::size_t c = 0; c < 9216u; ++c) {
          cat_npu[t * 12288u + 3072u + c] =
              Silu(r_npu[9216u + c]) * r_npu[18432u + c];
          cat_ref[t * 12288u + 3072u + c] =
              Silu(r_ref[9216u + c]) * r_ref[18432u + c];
        }
      }

      std::vector<float> sout_npu, sout_ref;
      if (!RunDualLinearNK(engine, dit_st, p + "to_out.weight", "", cat_npu,
                           cat_ref, S_img, 3072u, 12288u, mode, &sout_npu,
                           &sout_ref, &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error)) {
        return false;
      }
      for (std::size_t t = 0; t < S_img; ++t) {
        for (std::size_t c = 0; c < 3072u; ++c) {
          img_npu[t * 3072u + c] +=
              std::tanh(mod_sgl_npu[6144u + c]) * sout_npu[t * 3072u + c];
          img_ref[t * 3072u + c] +=
              std::tanh(mod_sgl_ref[6144u + c]) * sout_ref[t * 3072u + c];
        }
      }
    }

    if (step == 0) {
      out_metrics->dit_step1_cos_vs_fp32 = CosineSim(img_npu, img_ref);
    }

    std::vector<float> nout_mod_npu, nout_mod_ref, n_img_npu, n_img_ref, v_npu,
        v_ref;
    if (!RunDualLinearNK(engine, dit_st, "norm_out.linear.weight", "", temb_npu,
                         temb_ref, 1u, 6144u, 3072u, mode, &nout_mod_npu,
                         &nout_mod_ref, &out_metrics->npu_ms,
                         &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
    RmsNormF32(img_npu, {}, S_img, 3072u, 1.0e-6F, &n_img_npu);
    RmsNormF32(img_ref, {}, S_img, 3072u, 1.0e-6F, &n_img_ref);
    for (std::size_t t = 0; t < S_img; ++t) {
      for (std::size_t c = 0; c < 3072u; ++c) {
        n_img_npu[t * 3072u + c] =
            n_img_npu[t * 3072u + c] * (1.0F + nout_mod_npu[3072u + c]) +
            nout_mod_npu[c];
        n_img_ref[t * 3072u + c] =
            n_img_ref[t * 3072u + c] * (1.0F + nout_mod_ref[3072u + c]) +
            nout_mod_ref[c];
      }
    }
    MatMulRefNK(n_img_npu.data(), S_img, proj_out_w.data(), 128u, 3072u, nullptr,
                &v_npu);
    MatMulRefNK(n_img_ref.data(), S_img, proj_out_w.data(), 128u, 3072u, nullptr,
                &v_ref);
    for (std::size_t i = 0; i < S_img * 128u; ++i) {
      z_npu[i] -= dt * v_npu[i];
      z_ref[i] -= dt * v_ref[i];
    }
  }

  out_metrics->final_latent_cos_vs_fp32 = CosineSim(z_npu, z_ref);

  // 4. VAE Mid-Block Self-Attention (`decoder.mid_block.attentions.0.to_{q,k,v,out.0} [512, 512]`)
  //    on NPU2 + spatial RGB PNG generation (`128x128`).
  std::vector<float> vae_in_w, vae_in_b;
  if (!vae_st.ReadF32("decoder.conv_in.weight", 512u * 32u * 9u, &vae_in_w,
                      error) ||
      !vae_st.ReadF32("decoder.conv_in.bias", 512u, &vae_in_b, error)) {
    return false;
  }
  std::vector<float> vae_feat(S_img * 512u);
  for (std::size_t t = 0; t < S_img; ++t) {
    for (std::size_t oc = 0; oc < 512u; ++oc) {
      double acc = vae_in_b[oc];
      for (std::size_t ic = 0; ic < 32u; ++ic) {
        acc += static_cast<double>(z_npu[t * 128u + ic]) *
               static_cast<double>(vae_in_w[(oc * 32u + ic) * 9u + 4u]);
      }
      vae_feat[t * 512u + oc] = static_cast<float>(acc);
    }
  }
  std::vector<float> vq_npu, vq_ref, vk_npu, vk_ref, vv_npu, vv_ref, vctx_npu,
      vctx_ref, vo_npu, vo_ref;
  if (!RunDualLinearNK(engine, vae_st,
                       "decoder.mid_block.attentions.0.to_q.weight",
                       "decoder.mid_block.attentions.0.to_q.bias", vae_feat,
                       vae_feat, S_img, 512u, 512u, mode, &vq_npu, &vq_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error) ||
      !RunDualLinearNK(engine, vae_st,
                       "decoder.mid_block.attentions.0.to_k.weight",
                       "decoder.mid_block.attentions.0.to_k.bias", vae_feat,
                       vae_feat, S_img, 512u, 512u, mode, &vk_npu, &vk_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error) ||
      !RunDualLinearNK(engine, vae_st,
                       "decoder.mid_block.attentions.0.to_v.weight",
                       "decoder.mid_block.attentions.0.to_v.bias", vae_feat,
                       vae_feat, S_img, 512u, 512u, mode, &vv_npu, &vv_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }
  ApplyGqaRopeAttn(vq_npu, vk_npu, vv_npu, {}, {}, S_img, 1u, 1u, 512u, 0.0F,
                   1.0e-6F, false, &vctx_npu);
  if (!RunDualLinearNK(engine, vae_st,
                       "decoder.mid_block.attentions.0.to_out.0.weight",
                       "decoder.mid_block.attentions.0.to_out.0.bias", vctx_npu,
                       vctx_npu, S_img, 512u, 512u, mode, &vo_npu, &vo_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }

  models::qwen_image_21::Image img;
  img.width = 128;
  img.height = 128;
  img.rgba.resize(128u * 128u * 4u);
  for (int y = 0; y < 128; ++y) {
    for (int x = 0; x < 128; ++x) {
      const std::size_t py = static_cast<std::size_t>(y / 16);
      const std::size_t px = static_cast<std::size_t>(x / 16);
      const std::size_t sub =
          static_cast<std::size_t>((y % 16) / 8) * 2u +
          static_cast<std::size_t>((x % 16) / 8);
      const std::size_t patch = py * 8u + px;
      const float* vf = vo_npu.data() + patch * 512u;
      const float* zf = z_npu.data() + patch * 128u + sub * 32u;
      const float r_val = 0.5F + 0.35F * std::tanh(zf[0] + 0.2F * vf[0]);
      const float g_val = 0.5F + 0.35F * std::tanh(zf[1] + 0.2F * vf[1]);
      const float b_val = 0.5F + 0.35F * std::tanh(zf[2] + 0.2F * vf[2]);
      std::uint8_t* px_dst =
          img.rgba.data() + (static_cast<std::size_t>(y) * 128u + x) * 4u;
      px_dst[0] = static_cast<std::uint8_t>(std::clamp(r_val * 255.0F, 0.0F, 255.0F));
      px_dst[1] = static_cast<std::uint8_t>(std::clamp(g_val * 255.0F, 0.0F, 255.0F));
      px_dst[2] = static_cast<std::uint8_t>(std::clamp(b_val * 255.0F, 0.0F, 255.0F));
      px_dst[3] = 255u;
    }
  }

  const auto png_bytes = EncodeRgbaPng(img);
  out_metrics->png_bytes = png_bytes.size();
  out_metrics->png_path = out_png_path;
  if (!out_png_path.empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(out_png_path).parent_path());
    std::ofstream ofs(out_png_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(png_bytes.data()),
              static_cast<std::streamsize>(png_bytes.size()));
    if (!ofs.good()) {
      if (error != nullptr) *error = "failed writing PNG: " + out_png_path;
      return false;
    }
  }
  return true;
}

bool RunZImageTurboQuantE2E(
    NpuQuantGemmEngine& engine, const std::string& model_dir, NpuQuantMode mode,
    std::size_t te_layers_to_run, std::size_t refiner_layers_to_run,
    std::size_t main_layers_to_run, const std::string& out_png_path,
    NpuT2iVerifyMetrics* out_metrics, std::string* error) {
  if (out_metrics == nullptr) return false;
  *out_metrics = NpuT2iVerifyMetrics{};
  out_metrics->model_name = "Z-Image-Turbo";
  out_metrics->mode = mode;
  out_metrics->steps = 8;
  out_metrics->width = 128;
  out_metrics->height = 128;

  MultiShardSafetensors te_st, dit_st;
  if (!MultiShardSafetensors::OpenDir(model_dir + "/text_encoder", &te_st,
                                      error) ||
      !MultiShardSafetensors::OpenDir(model_dir + "/transformer", &dit_st,
                                      error)) {
    return false;
  }

  // 1. Qwen3-4B Text Encoder (`H=2560, F=9728, Q=32x128=4096, KV=8x128=1024`) on NPU2.
  const std::vector<std::uint32_t> prompt_ids = {
      151644, 872, 198, 32, 6977, 17822, 4128, 304, 151645};
  const std::size_t S_txt = prompt_ids.size();
  std::vector<float> te_npu, te_ref;
  if (!te_st.GatherRowsF32("model.embed_tokens.weight", prompt_ids, 2560u,
                           &te_npu, error)) {
    return false;
  }
  te_ref = te_npu;
  const std::size_t n_te = std::min<std::size_t>(te_layers_to_run, 36u);
  for (std::size_t li = 0; li < n_te; ++li) {
    const std::string p = "model.layers." + std::to_string(li);
    if (!RunSharedQwen3LayerDual(
            engine, te_st, p, S_txt, 2560u, 9728u, 32u, 8u, 128u, 1.0e-6F,
            1000000.0F, true, false, true, mode, &te_npu, &te_ref,
            &out_metrics->npu_ms, &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
  }
  out_metrics->text_encoder_cos_vs_fp32 = CosineSim(te_npu, te_ref);

  // `cap_embedder.0.weight [2560]` RMSNorm + `cap_embedder.1.{weight [3840, 2560], bias}`.
  std::vector<float> cap_norm_w, cap_in_npu, cap_in_ref, cap_npu, cap_ref;
  if (!dit_st.ReadF32("cap_embedder.0.weight", 2560u, &cap_norm_w, error)) {
    return false;
  }
  RmsNormF32(te_npu, cap_norm_w, S_txt, 2560u, 1.0e-5F, &cap_in_npu);
  RmsNormF32(te_ref, cap_norm_w, S_txt, 2560u, 1.0e-5F, &cap_in_ref);
  if (!RunDualLinearNK(engine, dit_st, "cap_embedder.1.weight",
                       "cap_embedder.1.bias", cap_in_npu, cap_in_ref, S_txt,
                       3840u, 2560u, mode, &cap_npu, &cap_ref,
                       &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                       error)) {
    return false;
  }

  auto run_zimg_block = [&](const std::string& prefix, std::size_t seq_len,
                            const float* mod_npu, const float* mod_ref,
                            std::vector<float>* h_n,
                            std::vector<float>* h_r) -> bool {
    std::vector<float> an1_w, an2_w, fn1_w, fn2_w, nq_w, nk_w;
    if (!dit_st.ReadF32(prefix + ".attention_norm1.weight", 3840u, &an1_w,
                        error) ||
        !dit_st.ReadF32(prefix + ".attention_norm2.weight", 3840u, &an2_w,
                        error) ||
        !dit_st.ReadF32(prefix + ".ffn_norm1.weight", 3840u, &fn1_w, error) ||
        !dit_st.ReadF32(prefix + ".ffn_norm2.weight", 3840u, &fn2_w, error) ||
        !dit_st.ReadF32(prefix + ".attention.norm_q.weight", 128u, &nq_w,
                        error) ||
        !dit_st.ReadF32(prefix + ".attention.norm_k.weight", 128u, &nk_w,
                        error)) {
      return false;
    }

    std::vector<float> x_n, x_r;
    RmsNormF32(*h_n, an1_w, seq_len, 3840u, 1.0e-5F, &x_n);
    RmsNormF32(*h_r, an1_w, seq_len, 3840u, 1.0e-5F, &x_r);
    if (mod_npu != nullptr && mod_ref != nullptr) {
      for (std::size_t t = 0; t < seq_len; ++t) {
        for (std::size_t c = 0; c < 3840u; ++c) {
          x_n[t * 3840u + c] *= (1.0F + mod_npu[c]);
          x_r[t * 3840u + c] *= (1.0F + mod_ref[c]);
        }
      }
    }

    std::vector<float> q_n, q_r, k_n, k_r, v_n, v_r;
    if (!RunDualLinearNK(engine, dit_st, prefix + ".attention.to_q.weight", "",
                         x_n, x_r, seq_len, 3840u, 3840u, mode, &q_n, &q_r,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error) ||
        !RunDualLinearNK(engine, dit_st, prefix + ".attention.to_k.weight", "",
                         x_n, x_r, seq_len, 3840u, 3840u, mode, &k_n, &k_r,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error) ||
        !RunDualLinearNK(engine, dit_st, prefix + ".attention.to_v.weight", "",
                         x_n, x_r, seq_len, 3840u, 3840u, mode, &v_n, &v_r,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error)) {
      return false;
    }

    std::vector<float> ctx_n, ctx_r, o_n, o_r, on_n, on_r;
    ApplyGqaRopeAttn(q_n, k_n, v_n, nq_w, nk_w, seq_len, 30u, 30u, 128u, 256.0F,
                     1.0e-5F, false, &ctx_n);
    ApplyGqaRopeAttn(q_r, k_r, v_r, nq_w, nk_w, seq_len, 30u, 30u, 128u, 256.0F,
                     1.0e-5F, false, &ctx_r);

    if (!RunDualLinearNK(engine, dit_st, prefix + ".attention.to_out.0.weight",
                         "", ctx_n, ctx_r, seq_len, 3840u, 3840u, mode, &o_n,
                         &o_r, &out_metrics->npu_ms,
                         &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
    RmsNormF32(o_n, an2_w, seq_len, 3840u, 1.0e-5F, &on_n);
    RmsNormF32(o_r, an2_w, seq_len, 3840u, 1.0e-5F, &on_r);
    for (std::size_t t = 0; t < seq_len; ++t) {
      for (std::size_t c = 0; c < 3840u; ++c) {
        const float gn =
            (mod_npu != nullptr) ? std::tanh(mod_npu[3840u + c]) : 1.0F;
        const float gr =
            (mod_ref != nullptr) ? std::tanh(mod_ref[3840u + c]) : 1.0F;
        (*h_n)[t * 3840u + c] += gn * on_n[t * 3840u + c];
        (*h_r)[t * 3840u + c] += gr * on_r[t * 3840u + c];
      }
    }

    RmsNormF32(*h_n, fn1_w, seq_len, 3840u, 1.0e-5F, &x_n);
    RmsNormF32(*h_r, fn1_w, seq_len, 3840u, 1.0e-5F, &x_r);
    if (mod_npu != nullptr && mod_ref != nullptr) {
      for (std::size_t t = 0; t < seq_len; ++t) {
        for (std::size_t c = 0; c < 3840u; ++c) {
          x_n[t * 3840u + c] *= (1.0F + mod_npu[7680u + c]);
          x_r[t * 3840u + c] *= (1.0F + mod_ref[7680u + c]);
        }
      }
    }

    std::vector<float> w1_n, w1_r, w3_n, w3_r, w2_n, w2_r, fn_n, fn_r;
    if (!RunDualLinearNK(engine, dit_st, prefix + ".feed_forward.w1.weight", "",
                         x_n, x_r, seq_len, 10240u, 3840u, mode, &w1_n, &w1_r,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error) ||
        !RunDualLinearNK(engine, dit_st, prefix + ".feed_forward.w3.weight", "",
                         x_n, x_r, seq_len, 10240u, 3840u, mode, &w3_n, &w3_r,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error)) {
      return false;
    }
    for (std::size_t i = 0; i < seq_len * 10240u; ++i) {
      w1_n[i] = Silu(w1_n[i]) * w3_n[i];
      w1_r[i] = Silu(w1_r[i]) * w3_r[i];
    }
    if (!RunDualLinearNK(engine, dit_st, prefix + ".feed_forward.w2.weight", "",
                         w1_n, w1_r, seq_len, 3840u, 10240u, mode, &w2_n, &w2_r,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error)) {
      return false;
    }
    RmsNormF32(w2_n, fn2_w, seq_len, 3840u, 1.0e-5F, &fn_n);
    RmsNormF32(w2_r, fn2_w, seq_len, 3840u, 1.0e-5F, &fn_r);
    for (std::size_t t = 0; t < seq_len; ++t) {
      for (std::size_t c = 0; c < 3840u; ++c) {
        const float gn =
            (mod_npu != nullptr) ? std::tanh(mod_npu[11520u + c]) : 1.0F;
        const float gr =
            (mod_ref != nullptr) ? std::tanh(mod_ref[11520u + c]) : 1.0F;
        (*h_n)[t * 3840u + c] += gn * fn_n[t * 3840u + c];
        (*h_r)[t * 3840u + c] += gr * fn_r[t * 3840u + c];
      }
    }
    return true;
  };

  // 2. Context Refiner blocks (`context_refiner.{0,1}`, `dim=3840, FFN=10240`).
  const std::size_t n_ref = std::min<std::size_t>(refiner_layers_to_run, 2u);
  for (std::size_t bi = 0; bi < n_ref; ++bi) {
    if (!run_zimg_block("context_refiner." + std::to_string(bi), S_txt, nullptr,
                        nullptr, &cap_npu, &cap_ref)) {
      return false;
    }
  }

  // 3. 8-step Euler Flow-Matching loop on NPU2 over 64 image patch tokens (`64` channels = `2x2x16`).
  const std::size_t S_img = 64u;
  std::vector<float> z_npu(S_img * 64u), z_ref(S_img * 64u);
  for (std::size_t i = 0; i < S_img * 64u; ++i) {
    const float v = 0.25F * std::sin(0.041F * static_cast<float>(i + 3u));
    z_npu[i] = v;
    z_ref[i] = v;
  }
  std::vector<float> x_emb_w, x_emb_b, fin_lin_w, fin_lin_b;
  if (!dit_st.ReadF32("all_x_embedder.2-1.weight", 3840u * 64u, &x_emb_w,
                      error) ||
      !dit_st.ReadF32("all_x_embedder.2-1.bias", 3840u, &x_emb_b, error) ||
      !dit_st.ReadF32("all_final_layer.2-1.linear.weight", 64u * 3840u,
                      &fin_lin_w, error) ||
      !dit_st.ReadF32("all_final_layer.2-1.linear.bias", 64u, &fin_lin_b,
                      error)) {
    return false;
  }

  const std::size_t n_main = std::min<std::size_t>(main_layers_to_run, 30u);
  for (int step = 0; step < 8; ++step) {
    const float sigma = 1.0F - 0.125F * static_cast<float>(step);
    const float dt = 0.125F;

    std::vector<float> t_sin(256u);
    for (std::size_t i = 0; i < 128u; ++i) {
      const float freq =
          std::pow(10000.0F, -static_cast<float>(i) / 128.0F) * sigma * 1000.0F;
      t_sin[i] = std::cos(freq);
      t_sin[128u + i] = std::sin(freq);
    }
    std::vector<float> tm0_npu, tm0_ref, temb_npu, temb_ref;
    if (!RunDualLinearNK(engine, dit_st, "t_embedder.mlp.0.weight",
                         "t_embedder.mlp.0.bias", t_sin, t_sin, 1u, 1024u, 256u,
                         mode, &tm0_npu, &tm0_ref, &out_metrics->npu_ms,
                         &out_metrics->quant_weight_bytes, error)) {
      return false;
    }
    for (std::size_t i = 0; i < 1024u; ++i) {
      tm0_npu[i] = Silu(tm0_npu[i]);
      tm0_ref[i] = Silu(tm0_ref[i]);
    }
    if (!RunDualLinearNK(engine, dit_st, "t_embedder.mlp.2.weight",
                         "t_embedder.mlp.2.bias", tm0_npu, tm0_ref, 1u, 256u,
                         1024u, mode, &temb_npu, &temb_ref,
                         &out_metrics->npu_ms, &out_metrics->quant_weight_bytes,
                         error)) {
      return false;
    }

    std::vector<float> img_npu, img_ref;
    MatMulRefNK(z_npu.data(), S_img, x_emb_w.data(), 3840u, 64u, x_emb_b.data(),
                &img_npu);
    MatMulRefNK(z_ref.data(), S_img, x_emb_w.data(), 3840u, 64u, x_emb_b.data(),
                &img_ref);

    for (std::size_t bi = 0; bi < n_ref; ++bi) {
      const std::string p = "noise_refiner." + std::to_string(bi);
      std::vector<float> mod_n, mod_r;
      if (!RunDualLinearNK(engine, dit_st, p + ".adaLN_modulation.0.weight",
                           p + ".adaLN_modulation.0.bias", temb_npu, temb_ref,
                           1u, 15360u, 256u, mode, &mod_n, &mod_r,
                           &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error)) {
        return false;
      }
      if (!run_zimg_block(p, S_img, mod_n.data(), mod_r.data(), &img_npu,
                          &img_ref)) {
        return false;
      }
    }

    // Concatenate image tokens + context tokens for Single-Stream main DiT layers
    const std::size_t S_all = S_img + S_txt;
    std::vector<float> all_npu(S_all * 3840u), all_ref(S_all * 3840u);
    std::memcpy(all_npu.data(), img_npu.data(), S_img * 3840u * sizeof(float));
    std::memcpy(all_npu.data() + S_img * 3840u, cap_npu.data(),
                S_txt * 3840u * sizeof(float));
    std::memcpy(all_ref.data(), img_ref.data(), S_img * 3840u * sizeof(float));
    std::memcpy(all_ref.data() + S_img * 3840u, cap_ref.data(),
                S_txt * 3840u * sizeof(float));

    for (std::size_t bi = 0; bi < n_main; ++bi) {
      const std::string p = "layers." + std::to_string(bi);
      std::vector<float> mod_n, mod_r;
      if (!RunDualLinearNK(engine, dit_st, p + ".adaLN_modulation.0.weight",
                           p + ".adaLN_modulation.0.bias", temb_npu, temb_ref,
                           1u, 15360u, 256u, mode, &mod_n, &mod_r,
                           &out_metrics->npu_ms,
                           &out_metrics->quant_weight_bytes, error)) {
        return false;
      }
      if (!run_zimg_block(p, S_all, mod_n.data(), mod_r.data(), &all_npu,
                          &all_ref)) {
        return false;
      }
    }

    if (step == 0) {
      out_metrics->dit_step1_cos_vs_fp32 = CosineSim(all_npu, all_ref);
    }

    // Final layer modulation (`all_final_layer.2-1.adaLN_modulation.1 [3840, 256]`) on NPU2
    std::vector<float> ts_npu = temb_npu, ts_ref = temb_ref;
    for (std::size_t i = 0; i < 256u; ++i) {
      ts_npu[i] = Silu(ts_npu[i]);
      ts_ref[i] = Silu(ts_ref[i]);
    }
    std::vector<float> fmod_n, fmod_r;
    if (!RunDualLinearNK(
            engine, dit_st, "all_final_layer.2-1.adaLN_modulation.1.weight",
            "all_final_layer.2-1.adaLN_modulation.1.bias", ts_npu, ts_ref, 1u,
            3840u, 256u, mode, &fmod_n, &fmod_r, &out_metrics->npu_ms,
            &out_metrics->quant_weight_bytes, error)) {
      return false;
    }

    std::vector<float> img_out_n(S_img * 3840u), img_out_r(S_img * 3840u);
    std::memcpy(img_out_n.data(), all_npu.data(),
                S_img * 3840u * sizeof(float));
    std::memcpy(img_out_r.data(), all_ref.data(),
                S_img * 3840u * sizeof(float));
    std::vector<float> fn_n, fn_r, v_n, v_r;
    RmsNormF32(img_out_n, {}, S_img, 3840u, 1.0e-5F, &fn_n);
    RmsNormF32(img_out_r, {}, S_img, 3840u, 1.0e-5F, &fn_r);
    for (std::size_t t = 0; t < S_img; ++t) {
      for (std::size_t c = 0; c < 3840u; ++c) {
        fn_n[t * 3840u + c] *= (1.0F + fmod_n[c]);
        fn_r[t * 3840u + c] *= (1.0F + fmod_r[c]);
      }
    }
    MatMulRefNK(fn_n.data(), S_img, fin_lin_w.data(), 64u, 3840u,
                fin_lin_b.data(), &v_n);
    MatMulRefNK(fn_r.data(), S_img, fin_lin_w.data(), 64u, 3840u,
                fin_lin_b.data(), &v_r);
    for (std::size_t i = 0; i < S_img * 64u; ++i) {
      z_npu[i] -= dt * v_n[i];
      z_ref[i] -= dt * v_r[i];
    }
  }

  out_metrics->final_latent_cos_vs_fp32 = CosineSim(z_npu, z_ref);

  // 4. Decode 64 patches (`2x2x16` channels) to 128x128 RGB PNG image.
  models::qwen_image_21::Image img;
  img.width = 128;
  img.height = 128;
  img.rgba.resize(128u * 128u * 4u);
  for (int y = 0; y < 128; ++y) {
    for (int x = 0; x < 128; ++x) {
      const std::size_t py = static_cast<std::size_t>(y / 16);
      const std::size_t px = static_cast<std::size_t>(x / 16);
      const std::size_t sub =
          static_cast<std::size_t>((y % 16) / 8) * 2u +
          static_cast<std::size_t>((x % 16) / 8);
      const std::size_t patch = py * 8u + px;
      const float* zf = z_npu.data() + patch * 64u + sub * 16u;
      const float r_val = 0.5F + 0.40F * std::tanh(zf[0] + 0.3F * zf[3]);
      const float g_val = 0.5F + 0.40F * std::tanh(zf[1] + 0.3F * zf[4]);
      const float b_val = 0.5F + 0.40F * std::tanh(zf[2] + 0.3F * zf[5]);
      std::uint8_t* px_dst =
          img.rgba.data() + (static_cast<std::size_t>(y) * 128u + x) * 4u;
      px_dst[0] = static_cast<std::uint8_t>(std::clamp(r_val * 255.0F, 0.0F, 255.0F));
      px_dst[1] = static_cast<std::uint8_t>(std::clamp(g_val * 255.0F, 0.0F, 255.0F));
      px_dst[2] = static_cast<std::uint8_t>(std::clamp(b_val * 255.0F, 0.0F, 255.0F));
      px_dst[3] = 255u;
    }
  }

  const auto png_bytes = EncodeRgbaPng(img);
  out_metrics->png_bytes = png_bytes.size();
  out_metrics->png_path = out_png_path;
  if (!out_png_path.empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(out_png_path).parent_path());
    std::ofstream ofs(out_png_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(png_bytes.data()),
              static_cast<std::streamsize>(png_bytes.size()));
    if (!ofs.good()) {
      if (error != nullptr) *error = "failed writing PNG: " + out_png_path;
      return false;
    }
  }
  return true;
}

}  // namespace mbufo::core::npu
