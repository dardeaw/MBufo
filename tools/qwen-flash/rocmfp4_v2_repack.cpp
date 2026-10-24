// Repack the ROCmFP4 v2 GGUF for gufo's kernel set: every Q6_K tensor is
// decoded with gufo's own CPU Q6_K reference and requantized to Q8_0
// (llama.cpp quantize_row_q8_0_ref semantics); all other tensors are copied
// byte-identical. MIT, same family as the port itself.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace {

std::uint64_t Rd64(std::FILE* f) {
  std::uint64_t v = 0;
  if (std::fread(&v, 8, 1, f) != 1) std::exit(1);
  return v;
}
std::uint32_t Rd32(std::FILE* f) {
  std::uint32_t v = 0;
  if (std::fread(&v, 4, 1, f) != 1) std::exit(1);
  return v;
}
std::string RdStr(std::FILE* f) {
  const std::uint64_t n = Rd64(f);
  std::string s(n, '\0');
  if (n != 0 && std::fread(s.data(), 1, n, f) != n) std::exit(1);
  return s;
}
void Wr64(std::FILE* f, std::uint64_t v) { std::fwrite(&v, 8, 1, f); }
void Wr32(std::FILE* f, std::uint32_t v) { std::fwrite(&v, 4, 1, f); }
void WrStr(std::FILE* f, const std::string& s) {
  Wr64(f, s.size());
  std::fwrite(s.data(), 1, s.size(), f);
}

std::uint16_t F32ToF16(float x) {
  std::uint32_t b = 0;
  std::memcpy(&b, &x, 4);
  const std::uint32_t sign = (b >> 16) & 0x8000U;
  std::int32_t exp = static_cast<std::int32_t>((b >> 23) & 0xFFU) - 127 + 15;
  std::uint32_t mant = b & 0x7FFFFFU;
  if (((b >> 23) & 0xFFU) == 0xFFU)
    return static_cast<std::uint16_t>(sign | 0x7C00U | (mant ? 0x200U : 0U));
  if (exp >= 31)
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<std::uint16_t>(sign);
    mant |= 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
    std::uint32_t half = mant >> shift;
    if ((mant >> (shift - 1)) & 1U) half = (half & ~1U) + ((half & 1U) | (((mant & ((1U << (shift - 1)) - 1U)) != 0U) ? 1U : 0U));
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
  const std::uint32_t rem = mant & 0x1FFFU;
  if (rem > 0x1000U || (rem == 0x1000U && (half & 1U))) half += 1U;
  if ((half >> 10) >= 31U) return static_cast<std::uint16_t>(sign | 0x7C00U);
  return static_cast<std::uint16_t>(sign | half);
}

std::size_t TypeBytes(std::uint32_t type, std::uint64_t elements) {
  using gufo::quant::block_iq4_nl;
  using gufo::quant::block_q3_0_rocmfpx;
  using gufo::quant::block_q4_0_rocmfp4;
  using gufo::quant::block_q4_0_rocmfp4_fast;
  std::size_t block = 1, bytes = 4;
  switch (type) {
    case 0: block = 1; bytes = 4; break;
    case 1: case 30: block = 1; bytes = 2; break;
    case 8: block = 32; bytes = 34; break;
    case 12: block = 256; bytes = 144; break;
    case 14: block = 256; bytes = 210; break;
    case 20: block = 32; bytes = sizeof(block_iq4_nl); break;
    case 100: block = 32; bytes = sizeof(block_q4_0_rocmfp4); break;
    case 101: block = 32; bytes = sizeof(block_q4_0_rocmfp4_fast); break;
    case 104: block = 32; bytes = sizeof(block_q3_0_rocmfpx); break;
    default: std::fprintf(stderr, "unsupported ggml type %u\n", type);
             std::exit(1);
  }
  return static_cast<std::size_t>(elements / block * bytes);
}

struct Info {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type;
  std::uint64_t old_offset;
  std::uint64_t new_offset;
  std::size_t old_bytes;
  std::size_t new_bytes;
  std::uint64_t elements;
  bool ple = false;
};

}  // namespace

static bool g_fast_q8 = false;
static bool IsF32Conv(const std::string& n) {
  return n.find("ssm_alpha") != std::string::npos ||
         n.find("ssm_beta") != std::string::npos ||
         n.find("_inject") != std::string::npos;
}

static bool IsF16Conv(const std::string& n) {
  return n.find("ple_conv1d") != std::string::npos;
}

struct ProdRef {
  std::FILE* f = nullptr;
  std::uint64_t db = 0;
  std::uint64_t off = 0;
  std::uint32_t type = 0;
};

static ProdRef OpenProd(const char* path, const std::string& want) {
  ProdRef r;
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return r;
  char m[4];
  std::fread(m, 1, 4, f);
  if (std::memcmp(m, "GGUF", 4) != 0) { std::fclose(f); return r; }
  const std::uint32_t ver = Rd32(f);
  const std::uint64_t nt = Rd64(f), nkv = Rd64(f);
  static const std::uint32_t vs[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  std::uint64_t align = 32;
  for (std::uint64_t i = 0; i < nkv; ++i) {
    const std::string key = RdStr(f);
    const std::uint32_t t = Rd32(f);
    if (t == 8) {
      RdStr(f);
    } else if (t == 9) {
      const std::uint32_t et = Rd32(f);
      const std::uint64_t n = Rd64(f);
      if (et == 8) { for (std::uint64_t j = 0; j < n; ++j) RdStr(f); }
      else std::fseek(f, static_cast<long>(n * vs[et]), SEEK_CUR);
    } else {
      if (t == 4 && key == "general.alignment") {
        std::uint32_t a = 0;
        std::fread(&a, 4, 1, f);
        align = a;
      } else {
        std::fseek(f, static_cast<long>(vs[t]), SEEK_CUR);
      }
    }
  }
  std::uint64_t t_off = 0;
  std::uint32_t t_ty = 0;
  bool found = false;
  for (std::uint64_t i = 0; i < nt; ++i) {
    const std::string nm = RdStr(f);
    const std::uint32_t nd = Rd32(f);
    std::uint64_t elems = 1;
    for (std::uint32_t d = 0; d < nd; ++d) elems *= Rd64(f);
    const std::uint32_t ty = Rd32(f);
    const std::uint64_t off = Rd64(f);
    if (nm == want) {
      t_off = off;
      t_ty = ty;
      found = true;
    }
  }
  if (found) {
    const long tbl_end = std::ftell(f);
    r.db = (static_cast<std::uint64_t>(tbl_end) + align - 1) / align * align;
    (void)ver;
    r.off = t_off;
    r.type = t_ty;
    r.f = f;
    return r;
  }
  std::fclose(f);
  return r;
}


int main(int argc, char** argv) {
  g_fast_q8 = std::getenv("REPACK_FAST_Q8") != nullptr;
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "usage: %s in.gguf out.gguf [prod-shard.gguf]\n", argv[0]);
    return 2;
  }
  ProdRef prod;
  if (argc == 4) {
    prod = OpenProd(argv[3], "per_layer_token_embd.weight");
    if (prod.f != nullptr && prod.type != 20) {
      std::fprintf(stderr, "prod ple type %u != 20\n", prod.type);
      return 1;
    }
    std::printf("prod ple found: %s type=%u off=%llu\n",
                prod.f != nullptr ? "yes" : "no", prod.type,
                static_cast<unsigned long long>(prod.off));
  }
  std::FILE* in = std::fopen(argv[1], "rb");
  if (in == nullptr) { std::perror("open in"); return 1; }

  char magic[4];
  std::fread(magic, 1, 4, in);
  if (std::memcmp(magic, "GGUF", 4) != 0) { std::fprintf(stderr, "not GGUF\n"); return 1; }
  const std::uint32_t version = Rd32(in);
  const std::uint64_t n_tensors = Rd64(in);
  const std::uint64_t n_kv = Rd64(in);

  // Capture the KV section verbatim; only read it to find the alignment.
  const long kv_begin = std::ftell(in);
  std::uint64_t alignment = 32;
  static const std::uint32_t vsize[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  for (std::uint64_t i = 0; i < n_kv; ++i) {
    const std::string key = RdStr(in);
    const std::uint32_t t = Rd32(in);
    if (t == 8) {
      RdStr(in);
    } else if (t == 9) {
      const std::uint32_t et = Rd32(in);
      const std::uint64_t n = Rd64(in);
      if (et == 8) { for (std::uint64_t j = 0; j < n; ++j) RdStr(in); }
      else std::fseek(in, static_cast<long>(n * vsize[et]), SEEK_CUR);
    } else {
      if (t == 4 && key == "general.alignment") {
        std::uint32_t a = 0;
        std::fread(&a, 4, 1, in);
        alignment = a;
      } else {
        std::fseek(in, static_cast<long>(vsize[t]), SEEK_CUR);
      }
    }
  }
  const long kv_end = std::ftell(in);
  std::vector<char> kv(static_cast<std::size_t>(kv_end - kv_begin));
  std::fseek(in, kv_begin, SEEK_SET);
  std::fread(kv.data(), 1, kv.size(), in);
  std::fseek(in, kv_end, SEEK_SET);
  std::printf("alignment=%llu\n", static_cast<unsigned long long>(alignment));

  std::vector<Info> infos;
  infos.reserve(n_tensors);
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    Info t;
    t.name = RdStr(in);
    const std::uint32_t nd = Rd32(in);
    t.dims.resize(nd);
    t.elements = 1;
    for (std::uint32_t d = 0; d < nd; ++d) {
      t.dims[d] = Rd64(in);
      t.elements *= t.dims[d];
    }
    t.type = Rd32(in);
    t.old_offset = Rd64(in);
    t.old_bytes = TypeBytes(t.type, t.elements);
    const bool ple =
        t.name == "per_layer_token_embd.weight" && t.type == 104 &&
        prod.f != nullptr;
    const bool f32c = t.type == 14 && IsF32Conv(t.name);
    const bool f16c = t.type == 1 && IsF16Conv(t.name);
    const bool conv =
        !f32c &&
        (t.type == 14 ||
         (t.type == 101 && (g_fast_q8 || t.name.find("output_hc") != std::string::npos)));
    t.new_bytes = ple ? static_cast<std::size_t>(t.elements / 160 * 90)
                : f32c ? static_cast<std::size_t>(t.elements * 4)
                : f16c ? static_cast<std::size_t>(t.elements * 4)
                : conv ? static_cast<std::size_t>(t.elements / 32 * 34)
                       : t.old_bytes;
    t.ple = ple;
    infos.push_back(std::move(t));
  }
  const long info_end = std::ftell(in);

  std::uint64_t cur = 0;
  std::size_t converted = 0, converted_bytes = 0;
  for (auto& t : infos) {
    t.new_offset = (cur + alignment - 1) / alignment * alignment;
    cur = t.new_offset + t.new_bytes;
    if ((t.type == 14 && !IsF32Conv(t.name)) ||
        (t.type == 101 && (g_fast_q8 || t.name.find("output_hc") != std::string::npos))) {
      ++converted;
      converted_bytes += t.new_bytes - t.old_bytes;
    }
  }
  std::printf("converted %zu Q6_K tensors (+%.2f GiB)\n", converted,
              converted_bytes / 1073741824.0);

  std::FILE* out = std::fopen(argv[2], "wb");
  if (out == nullptr) { std::perror("open out"); return 1; }
  std::fwrite("GGUF", 1, 4, out);
  Wr32(out, version);
  Wr64(out, n_tensors);
  Wr64(out, n_kv);
  std::fwrite(kv.data(), 1, kv.size(), out);
  for (auto& t : infos) {
    WrStr(out, t.name);
    Wr32(out, static_cast<std::uint32_t>(t.dims.size()));
    for (auto d : t.dims) Wr64(out, d);
    Wr32(out, t.ple
              ? 20
              : (t.type == 14 && IsF32Conv(t.name))
              ? 0
              : (t.type == 1 && IsF16Conv(t.name))
              ? 0
              : (t.type == 14 ||
                 (t.type == 101 &&
                  (g_fast_q8 ||
                   t.name.find("output_hc") != std::string::npos)))
              ? 8
              : t.type);
    Wr64(out, t.new_offset);
  }
  const long data_begin_old = (info_end + static_cast<long>(alignment) - 1) /
                              static_cast<long>(alignment) *
                              static_cast<long>(alignment);
  const std::uint64_t data_begin_new =
      (static_cast<std::uint64_t>(std::ftell(out)) + alignment - 1) / alignment *
      alignment;
  std::fseek(out, static_cast<long>(data_begin_new), SEEK_SET);
  std::printf("data begin old=%ld new=%llu\n", data_begin_old,
              static_cast<unsigned long long>(data_begin_new));

  std::vector<char> pad(alignment, 0);
  std::vector<std::uint8_t> raw, enc;
  std::vector<float> dec;
  std::uint64_t written = 0;
  for (std::size_t i = 0; i < infos.size(); ++i) {
    const Info& t = infos[i];
    const std::uint64_t gap = t.new_offset - written;
    if (gap != 0) {
      std::uint64_t left = gap;
      while (left > 0) {
        const std::size_t chunk = left < pad.size() ? static_cast<std::size_t>(left) : pad.size();
        std::fwrite(pad.data(), 1, chunk, out);
        left -= chunk;
      }
    }
    std::fseek(in, static_cast<long>(data_begin_old + t.old_offset), SEEK_SET);
    if (t.ple) {
      std::fseek(prod.f, static_cast<long>(prod.db + prod.off), SEEK_SET);
      std::size_t left = t.new_bytes;
      raw.resize(1u << 20);
      while (left > 0) {
        const std::size_t chunk = left < raw.size() ? left : raw.size();
        if (std::fread(raw.data(), 1, chunk, prod.f) != chunk ||
            std::fwrite(raw.data(), 1, chunk, out) != chunk) {
          std::fprintf(stderr, "short ple copy\n");
          return 1;
        }
        left -= chunk;
      }
    } else if (t.type == 14 && IsF32Conv(t.name)) {
      raw.resize(t.old_bytes);
      if (std::fread(raw.data(), 1, raw.size(), in) != raw.size()) {
        std::fprintf(stderr, "short read %s\n", t.name.c_str());
        return 1;
      }
      dec.resize(t.elements);
      gufo::quant::DequantizeQ6_K(raw.data(), dec.data(), t.elements);
      for (float& v : dec)
        if (!std::isfinite(v))
          v = 0.0F;
      std::fwrite(dec.data(), 1, dec.size() * 4, out);
    } else if (t.type == 1 && IsF16Conv(t.name)) {
      raw.resize(t.old_bytes);
      if (std::fread(raw.data(), 1, raw.size(), in) != raw.size()) {
        std::fprintf(stderr, "short read %s\n", t.name.c_str());
        return 1;
      }
      dec.resize(t.elements);
      for (std::size_t e = 0; e < t.elements; ++e) {
        const std::uint16_t h = static_cast<std::uint16_t>(
            raw[2 * e] | (raw[2 * e + 1] << 8));
        dec[e] = gufo::quant::Fp16ToFloat(h);
        if (!std::isfinite(dec[e])) dec[e] = 0.0F;
      }
      std::fwrite(dec.data(), 1, dec.size() * 4, out);
    } else if (t.type != 14 &&
        !(t.type == 101 && (g_fast_q8 || t.name.find("output_hc") != std::string::npos))) {
      raw.resize(t.old_bytes);
      if (std::fread(raw.data(), 1, raw.size(), in) != raw.size()) {
        std::fprintf(stderr, "short read %s\n", t.name.c_str());
        return 1;
      }
      std::fwrite(raw.data(), 1, raw.size(), out);
    } else {
      // Q6_K -> Q8_0: decode with the gufo reference, requantize per 32.
      raw.resize(t.old_bytes);
      if (std::fread(raw.data(), 1, raw.size(), in) != raw.size()) {
        std::fprintf(stderr, "short read %s\n", t.name.c_str());
        return 1;
      }
      dec.resize(t.elements);
      if (t.type == 14)
        gufo::quant::DequantizeQ6_K(raw.data(), dec.data(), t.elements);
      else
        gufo::quant::DequantizeQ4_0_ROCMFP4_FAST(raw.data(), dec.data(),
                                                 t.elements);
      // The v2 quantizer leaves NaN/inf f16 super-block scales in a few
      // percent of embedding blocks; the source engine treats those rows as
      // zero. Fold them to zero so the requantized Q8_0 stays finite.
      if (t.type == 14) {
        for (float& v : dec)
          if (!std::isfinite(v))
            v = 0.0F;
      }
      enc.resize(t.new_bytes);
      const std::size_t blocks = t.elements / 32;
      auto* q8 = reinterpret_cast<gufo::quant::block_q8_0*>(enc.data());
      for (std::size_t b = 0; b < blocks; ++b) {
        float max_abs = 0.0F;
        for (std::size_t j = 0; j < 32; ++j)
          max_abs = std::fmax(max_abs, std::fabs(dec[b * 32 + j]));
        const std::uint16_t dh = F32ToF16(max_abs / 127.0F);
        q8[b].d = dh;
        const float d = gufo::quant::Fp16ToFloat(dh);
        for (std::size_t j = 0; j < 32; ++j) {
          const float x = dec[b * 32 + j];
          int q = d == 0.0F ? 0 : static_cast<int>(std::lroundf(x / d));
          if (q > 127) q = 127;
          if (q < -127) q = -127;
          q8[b].qs[j] = static_cast<std::int8_t>(q);
        }
      }
      std::fwrite(enc.data(), 1, enc.size(), out);
    }
    written = t.new_offset + t.new_bytes;
    if ((i & 127) == 0) std::printf("%zu/%zu %s\n", i, infos.size(), t.name.c_str());
  }
  std::fclose(out);
  std::fclose(in);
  std::printf("REPACK_DONE total=%llu\n", static_cast<unsigned long long>(written));
  return 0;
}
