#include "src/models/gptoss/weights.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace gufo::models::gptoss {
namespace {

// GGML block layouts (canonical, must match ggml-quants.c).
struct BlockQ80 {
  std::uint16_t d;
  std::int8_t qs[32];
};

struct BlockQ50 {
  std::uint16_t d;
  std::uint8_t qh[4];
  std::uint8_t qs[16];
};
struct BlockQ4K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(BlockQ80) == 34, "q8_0 block must be 34 bytes");
static_assert(sizeof(BlockQ50) == 22, "q5_0 block must be 22 bytes");
static_assert(sizeof(BlockQ4K) == 144, "q4_K block must be 144 bytes");


float Fp16ToFloat(std::uint16_t h) {
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;
  std::uint32_t bits;
  if (exp == 0) {
    // Subnormal or zero: renormalize via float arithmetic.
    float f = static_cast<float>(mant) / 1024.0F;
    f *= 6.103515625e-05F;  // 2^-14
    return (h & 0x8000) ? -f : f;
  }
  if (exp == 31) {
    bits = ((h & 0x8000) << 16) | 0x7F800000U | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  bits = ((h & 0x8000) << 16) | ((exp + 112) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

void DequantQ50Row(const BlockQ50* x, float* y, std::size_t k) {
  const std::size_t nb = k / 32;
  for (std::size_t i = 0; i < nb; ++i) {
    const float d = Fp16ToFloat(x[i].d);
    std::uint32_t qh = 0;
    std::memcpy(&qh, x[i].qh, sizeof(qh));
    for (std::size_t j = 0; j < 16; ++j) {
      // Bit placement matches llama.cpp inference (qh bit j+12 for the
      // high half; see dequantize_row_q5_0 / CUDA vecdot).
      const std::uint8_t xh_0 =
          static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const std::uint8_t xh_1 =
          static_cast<std::uint8_t>(((qh >> (j + 12))) & 0x10);
      const std::int32_t x0 =
          (static_cast<std::int32_t>(x[i].qs[j] & 0x0F) | xh_0) - 16;
      const std::int32_t x1 =
          (static_cast<std::int32_t>(x[i].qs[j] >> 4) | xh_1) - 16;
      y[i * 32 + j] = static_cast<float>(x0) * d;
      y[i * 32 + j + 16] = static_cast<float>(x1) * d;
    }
  }
}

void GetScaleMinK4(int j, const std::uint8_t* q, std::uint8_t* sc,
                   std::uint8_t* mn) {
  if (j < 4) {
    *sc = q[j] & 63;
    *mn = q[j + 4] & 63;
  } else {
    *sc = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *mn = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

void DequantQ4KRow(const BlockQ4K* x, float* y, std::size_t k) {
  const std::size_t nb = k / 256;
  for (std::size_t i = 0; i < nb; ++i) {
    const std::uint8_t* q = x[i].qs;
    const float d = Fp16ToFloat(x[i].d);
    const float mn = Fp16ToFloat(x[i].dmin);
    float* out = y + i * 256;
    int is = 0;
    for (std::size_t j = 0; j < 256; j += 64, is += 2) {
      std::uint8_t sc, m;
      GetScaleMinK4(is, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = mn * m;
      GetScaleMinK4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = mn * m;
      for (std::size_t l = 0; l < 32; ++l) *out++ = d1 * (q[l] & 0xF) - m1;
      for (std::size_t l = 0; l < 32; ++l) *out++ = d2 * (q[l] >> 4) - m2;
      q += 32;
    }
  }
}

std::uint16_t FloatToFp16(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, 4);
  const std::uint32_t sign = (bits >> 16) & 0x8000;
  const int exp = static_cast<int>((bits >> 23) & 0xFF) - 112;
  const std::uint32_t mant = bits & 0x7FFFFF;
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31)
    return static_cast<std::uint16_t>(sign | 0x7BFF);
  return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
}

void QuantizeQ80Row(const float* x, BlockQ80* y, std::size_t k) {
  for (std::size_t i = 0; i < k / 32; ++i) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      const float v = std::fabs(x[i * 32 + j]);
      if (v > amax) amax = v;
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    y[i].d = FloatToFp16(d);
    for (std::size_t j = 0; j < 32; ++j) {
      y[i].qs[j] =
          static_cast<std::int8_t>(std::round(x[i * 32 + j] * id));
    }
  }
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};
  ModelWeights* out{nullptr};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  const core::GgufTensorInfo* Find(const std::string& name) {
    return reader.FindTensor(name);
  }

  // Binds a tensor that needs no conversion (F32 norms/biases/sinks).
  TensorRef GetRaw(const std::string& name, std::uint64_t cols,
                   std::uint64_t rows = 1, std::uint64_t experts = 1) {
    TensorRef ref;
    const auto* info = Find(name);
    if (info == nullptr) {
      return ref;
    }
    if (info->dimensions.size() < 1 || info->dimensions[0] != cols ||
        (info->dimensions.size() > 1 && info->dimensions[1] != rows) ||
        ((info->dimensions.size() > 2 ? info->dimensions[2] : 1) != experts)) {
      Fail("tensor " + name + " has unexpected dimensions");
      return ref;
    }
    ref.data = info->data;
    ref.size_bytes = info->size_bytes;
    ref.type = info->type;
    ref.cols = cols;
    ref.rows = rows;
    ref.experts = experts;
    return ref;
  }

  std::vector<std::uint8_t>* Arena(std::size_t bytes) {
    return out->NewArena(bytes);
  }

  // Converts one row-major [rows x cols] matrix of `src_type` to Q8_0.
  TensorRef ConvertQ8(const std::string& name, std::uint64_t cols,
                      std::uint64_t rows) {
    TensorRef ref;
    const auto* info = Find(name);
    if (info == nullptr) {
      return ref;
    }
    if (info->dimensions.size() < 2 || info->dimensions[0] != cols ||
        info->dimensions[1] != rows) {
      Fail("tensor " + name + " has unexpected dimensions");
      return ref;
    }
    const std::size_t row_q8 = (cols / 32) * sizeof(BlockQ80);
    auto* arena = Arena(row_q8 * rows);
    // Q8_0 sources are already in the target layout: copy rows verbatim.
    if (info->type == core::GgmlType::kQ8_0) {
      std::memcpy(arena->data(), info->data, arena->size());
      ref.data = arena->data();
      ref.size_bytes = arena->size();
      ref.type = core::GgmlType::kQ8_0;
      ref.cols = cols;
      ref.rows = rows;
      return ref;
    }
    std::vector<float> tmp(cols);
    const auto* src = static_cast<const std::uint8_t*>(info->data);
    auto* dst = reinterpret_cast<BlockQ80*>(arena->data());
    for (std::uint64_t r = 0; r < rows; ++r) {
      switch (info->type) {
        case core::GgmlType::kQ5_0:
          DequantQ50Row(
              reinterpret_cast<const BlockQ50*>(src) + r * (cols / 32),
              tmp.data(), cols);
          break;
        case core::GgmlType::kQ4_K:
          DequantQ4KRow(
              reinterpret_cast<const BlockQ4K*>(src) + r * (cols / 256),
              tmp.data(), cols);
          break;
        default:
          Fail("unsupported source type for Q8 conversion: " + name);
          return ref;
      }
      QuantizeQ80Row(tmp.data(), &dst[r * (cols / 32)], cols);
    }
    ref.data = arena->data();
    ref.size_bytes = arena->size();
    ref.type = core::GgmlType::kQ8_0;
    ref.cols = cols;
    ref.rows = rows;
    return ref;
  }

};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  switch (type) {
    case core::GgmlType::kF32:
      return static_cast<std::size_t>(cols) * 4;
    case core::GgmlType::kF16:
      return static_cast<std::size_t>(cols) * 2;
    case core::GgmlType::kQ8_0:
      return static_cast<std::size_t>(cols / 32) * 34;
    case core::GgmlType::kQ4_0:
      return static_cast<std::size_t>(cols / 32) * 18;
    case core::GgmlType::kMXFP4:
      return static_cast<std::size_t>(cols / 32) * 17;
    default:
      return 0;
  }
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                                std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value())
    return std::nullopt;
  Binder b{reader, error_msg};
  ModelWeights w;
  b.out = &w;
  w.config = *config;
  const std::uint64_t hidden = config->hidden_size;
  const std::uint64_t vocab = config->vocab_size;
  const std::uint64_t qdim =
      static_cast<std::uint64_t>(config->num_heads) * config->head_dim;
  const std::uint64_t kvdim =
      static_cast<std::uint64_t>(config->num_kv_heads) * config->head_dim;
  const std::uint64_t ff = config->expert_ff;
  const std::uint64_t nex = config->num_experts;

  w.token_embd = b.ConvertQ8("token_embd.weight", hidden, vocab);
  w.output_norm = b.GetRaw("output_norm.weight", hidden);
  w.output = b.GetRaw("output.weight", hidden, vocab);
  if (w.output.empty()) {
    w.output = w.token_embd;  // Tied LM head.
  }

  w.layers.resize(config->num_layers);
  for (std::uint32_t il = 0; il < config->num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& l = w.layers[il];
    l.attn_norm = b.GetRaw(p + "attn_norm.weight", hidden);
    l.wq = b.ConvertQ8(p + "attn_q.weight", hidden, qdim);
    l.wk = b.ConvertQ8(p + "attn_k.weight", hidden, kvdim);
    l.wv = b.ConvertQ8(p + "attn_v.weight", hidden, kvdim);
    l.wo = b.ConvertQ8(p + "attn_output.weight", qdim, hidden);
    l.wo_b = b.GetRaw(p + "attn_output.bias", hidden);
    l.attn_sinks =
        b.GetRaw(p + "attn_sinks.weight", config->num_heads);
    l.post_norm = b.GetRaw(p + "post_attention_norm.weight", hidden);
    l.router = b.GetRaw(p + "ffn_gate_inp.weight", hidden, nex);
    l.router_b = b.GetRaw(p + "ffn_gate_inp.bias", nex);
    l.gate_b = b.GetRaw(p + "ffn_gate_exps.bias", ff, nex);
    l.up_b = b.GetRaw(p + "ffn_up_exps.bias", ff, nex);
    l.down_b = b.GetRaw(p + "ffn_down_exps.bias", hidden, nex);

    // Experts stay native MXFP4 (a Q4_0 round-trip crushes the small
    // values in outlier blocks and flips routing); gate+up are memcpy-merged
    // into one slab for the single-pass vec kernel, down is referenced
    // in place (already per-expert contiguous).
    const auto* gate_info = b.Find(p + "ffn_gate_exps.weight");
    const auto* up_info = b.Find(p + "ffn_up_exps.weight");
    const auto* down_info = b.Find(p + "ffn_down_exps.weight");
    if (gate_info == nullptr || up_info == nullptr || down_info == nullptr) {
      b.Fail("missing MoE expert weights in " + p);
      break;
    }
    for (const auto* info :
         {gate_info, up_info, down_info}) {
      if (info->type != core::GgmlType::kMXFP4) {
        b.Fail("expert weights must be MXFP4 in " + p);
        break;
      }
    }
    if (!b.ok) break;
    if (gate_info->dimensions.size() != 3 || up_info->dimensions.size() != 3 ||
        down_info->dimensions.size() != 3 ||
        gate_info->dimensions[0] != hidden || gate_info->dimensions[1] != ff ||
        up_info->dimensions[0] != hidden || up_info->dimensions[1] != ff ||
        down_info->dimensions[0] != ff || down_info->dimensions[1] != hidden ||
        gate_info->dimensions[2] != nex || up_info->dimensions[2] != nex ||
        down_info->dimensions[2] != nex) {
      b.Fail("expert weights have unexpected dimensions in " + p);
      break;
    }
    constexpr std::size_t kMxfp4RowElems = 32;
    constexpr std::size_t kMxfp4BlockBytes = 17;
    const std::size_t mx_row = (hidden / kMxfp4RowElems) * kMxfp4BlockBytes;
    const std::size_t dmx_row = (ff / kMxfp4RowElems) * kMxfp4BlockBytes;
    const std::size_t slab = ff * mx_row;
    // Merged gate_up: [2*ff][hidden] per expert, MXFP4.
    auto* gu_arena = b.Arena(2 * slab * nex);
    for (std::uint64_t e = 0; e < nex && b.ok; ++e) {
      const std::uint8_t* gsrc =
          static_cast<const std::uint8_t*>(gate_info->data) + e * slab;
      const std::uint8_t* usrc =
          static_cast<const std::uint8_t*>(up_info->data) + e * slab;
      std::uint8_t* gdst = gu_arena->data() + e * 2 * slab;
      std::memcpy(gdst, gsrc, slab);
      std::memcpy(gdst + slab, usrc, slab);
    }
    l.gate_up_exps.data = gu_arena->data();
    l.gate_up_exps.size_bytes = gu_arena->size();
    l.gate_up_exps.type = core::GgmlType::kMXFP4;
    l.gate_up_exps.cols = hidden;
    l.gate_up_exps.rows = 2 * ff;
    l.gate_up_exps.experts = nex;
    l.down_exps.data = down_info->data;
    l.down_exps.size_bytes = down_info->size_bytes;
    l.down_exps.type = core::GgmlType::kMXFP4;
    l.down_exps.cols = ff;
    l.down_exps.rows = hidden;
    l.down_exps.experts = nex;
  }

  const auto require = [&](const TensorRef& t, const std::string& name,
                            bool norm = false) {
    if (t.empty()) {
      b.Fail("missing required tensor " + name);
      return;
    }
    if (norm && t.type != core::GgmlType::kF32) {
      b.Fail("norm tensor must be F32: " + name);
    }
  };
  require(w.token_embd, "token_embd.weight");
  require(w.output_norm, "output_norm.weight", true);
  if (w.output.type != core::GgmlType::kQ8_0) {
    b.Fail("output.weight must be Q8_0");
  }
  for (std::uint32_t il = 0; il < config->num_layers && b.ok; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto& l = w.layers[il];
    require(l.attn_norm, p + "attn_norm.weight", true);
    require(l.wq, p + "attn_q.weight");
    require(l.wk, p + "attn_k.weight");
    require(l.wv, p + "attn_v.weight");
    require(l.wo, p + "attn_output.weight");
    require(l.wo_b, p + "attn_output.bias");
    require(l.attn_sinks, p + "attn_sinks.weight");
    require(l.post_norm, p + "post_attention_norm.weight", true);
    require(l.router, p + "ffn_gate_inp.weight");
    if (l.router.type != core::GgmlType::kF32) {
      b.Fail("router must be F32: " + p + "ffn_gate_inp.weight");
    }
    require(l.router_b, p + "ffn_gate_inp.bias");
    require(l.gate_up_exps, p + "ffn_gate_exps.weight");
    require(l.down_exps, p + "ffn_down_exps.weight");
    require(l.gate_b, p + "ffn_gate_exps.bias");
    require(l.up_b, p + "ffn_up_exps.bias");
    require(l.down_b, p + "ffn_down_exps.bias");
  }
  if (!b.ok)
    return std::nullopt;
  if (w.output.empty()) {
    b.Fail("missing output.weight and no tied fallback");
    return std::nullopt;
  }
  if (getenv("GPTOSS_DUMP_CONV") != nullptr) {
    const auto dump = [&](const char* name, const TensorRef& t) {
      std::string path = std::string("/home/dardeaw/conv_dump/") + name;
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f != nullptr) {
        (void)std::fwrite(t.data, 1, t.size_bytes, f);
        (void)std::fclose(f);
      }
    };
    dump("token_embd.q8", w.token_embd);
    dump("l0_wq.q8", w.layers[0].wq);
    dump("l0_wk.q8", w.layers[0].wk);
    dump("l0_wo.q8", w.layers[0].wo);
    dump("l0_gateup.mxfp4", w.layers[0].gate_up_exps);
    dump("l0_down.mxfp4", w.layers[0].down_exps);
  }
  std::fprintf(stderr, "[gptoss] bound %u layers, %zu arenas\n",
               config->num_layers, w.arenas_.size());
  return w;
}

}  // namespace gufo::models::gptoss
