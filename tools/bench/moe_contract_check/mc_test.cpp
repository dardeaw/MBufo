// MoE engine contract: unit checks (no GPU) plus a live pass over real engines.
//   mc_test                                  -- contract validation + registry enforcement with fake engines
//   mc_test <engine>=<model.gguf> ...        -- additionally creates each engine through the registry (so the contract is enforced on the
//                                               real model), prints geometry / kernels / implied expert bytes per token, and checks that every
//                                               engine claiming core row-dot kernels uses only formats the row-dot registry serves
//   mc_test +gpu                             -- also runs the kernel conformance of the core attention kernels engines may claim
//                                               (row invariance: row t of an n-row call == that row alone, bit for bit)
// The attention contract (core/engine/attention_contract.hpp) is checked in the same passes: validation, registry enforcement
// with fake engines, and the declaration of every real engine created.
// Built by tools/bench/moe_contract_check/mc_ab.sh against the mbufo link line.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "src/core/compute/attention/blocked_splitk.hpp"
#include "src/core/compute/attention/flash_wmma_large.hpp"
#include "src/core/engine/attention_contract.hpp"
#include "src/core/engine/engine_batch.hpp"
#include "src/core/engine/moe_contract.hpp"
#include "src/core/quant/row_dot_traits.hpp"
#include "src/models/deepseek_v4_flash/engine_adapter.hpp"
#include "src/models/gemma4/engine_adapter.hpp"
#include "src/models/gptoss/engine_adapter.hpp"
#include "src/models/qwen38_flash_next/engine_adapter.hpp"
#include "src/models/qwen/engine_adapter.hpp"

namespace ce = mbufo::core::engine;
namespace q = mbufo::core::quant;

// The engines register themselves from static registrars inside their adapter objects; a static library only contributes an object
// that something references, so take the address of each adapter's Load to pull them in.
// (an extern-linkage array so the references cannot be optimized away)
using namespace mbufo::models;
void* g_engine_link_refs[6] = {
    reinterpret_cast<void*>(static_cast<std::shared_ptr<gemma4::GemmaEngineModel> (*)(const std::string&, std::string*)>(&gemma4::GemmaEngineModel::Load)),
    reinterpret_cast<void*>(static_cast<std::shared_ptr<gptoss::GptossEngineModel> (*)(const std::string&, std::string*)>(&gptoss::GptossEngineModel::Load)),
    reinterpret_cast<void*>(static_cast<std::shared_ptr<qwen38_flash_next::FlashNextEngineModel> (*)(const std::string&, std::string*)>(&qwen38_flash_next::FlashNextEngineModel::Load)),
    reinterpret_cast<void*>(static_cast<std::shared_ptr<deepseek_v4_flash::DeepSeekEngineModel> (*)(const std::string&, const deepseek_v4_flash::ModelOptions&, std::string*)>(&deepseek_v4_flash::DeepSeekEngineModel::Load)),
    reinterpret_cast<void*>(static_cast<std::shared_ptr<qwen::Qwen27BEngineModel> (*)(const std::string&, std::string*)>(&qwen::Qwen27BEngineModel::Load)),
    reinterpret_cast<void*>(static_cast<std::shared_ptr<qwen::QwenMoeEngineModel> (*)(const std::string&, const qwen::Qwen27BModelOptions&, std::string*)>(&qwen::QwenMoeEngineModel::Load))};

static int g_bad = 0;
static void Check(bool ok, const std::string& what) {
  std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_bad;
}

// ---- fake engines for the registry rules ----
class FakeSession final : public ce::EngineSession {
 public:
  bool Sync(std::span<const std::int32_t>, std::string*) override { return true; }
  bool Evaluate(std::int32_t, std::string*) override { return true; }
  bool Forward(std::span<const std::int32_t>, std::uint32_t, std::string*) override { return true; }
  std::span<const float> Logits() const noexcept override { return {}; }
  std::uint32_t Length() const noexcept override { return 0; }
  std::int32_t SampleGreedy() override { return 0; }
};
class FakeModelBase : public ce::EngineModel {
 public:
  std::unique_ptr<ce::EngineSession> CreateSession(std::string*) override { return std::make_unique<FakeSession>(); }
  std::vector<std::int32_t> Tokenize(std::string_view, bool) const override { return {}; }
  std::uint32_t VocabSize() const noexcept override { return 1; }
  std::uint32_t MaxContext() const noexcept override { return 1; }
  std::size_t ResidentBytes() const noexcept override { return 0; }
  std::string ModelName() const override { return "fake"; }
};
static ce::MoeGeometry GoodGeometry() {
  ce::MoeGeometry g;
  g.experts = 256; g.top_k = 8; g.hidden = 2048; g.expert_ff = 512; g.shared_ff = 512;
  for (int i = 0; i < 40; ++i) g.AddLayer(12, 12, 13);  // Q4_K gate/up, Q5_K down
  return g;
}
static ce::MoeKernelInfo CoreKernels() {
  ce::MoeKernelInfo k;
  k.origin = ce::MoeKernelOrigin::kCoreRowDot;
  k.gate_up_kernel = "g"; k.down_kernel = "d";
  return k;
}
class FakeMoe final : public FakeModelBase, public ce::MoeCapable {
 public:
  FakeMoe(ce::MoeGeometry g, ce::MoeKernelInfo k) : g_(std::move(g)), k_(std::move(k)) {}
  ce::MoeGeometry MoeGeometryInfo() const override { return g_; }
  ce::MoeKernelInfo MoeKernelsInfo() const override { return k_; }
 private:
  ce::MoeGeometry g_;
  ce::MoeKernelInfo k_;
};
class FakePlain final : public FakeModelBase {};


// ---- attention contract fakes ----
static ce::AttentionGeometry GoodAttention() {
  ce::AttentionGeometry g;
  ce::AttentionGroup a;
  a.heads = 16; a.kv_heads = 2; a.head_dim = 256;
  for (int i = 0; i < 10; ++i) g.AddLayer(a);
  a.window = 1024; a.heads = 16; a.kv_heads = 8; a.head_dim = 256;
  for (int i = 0; i < 25; ++i) g.AddLayer(a);
  return g;
}
static ce::AttentionKernelInfo CoreAttention() {
  ce::AttentionKernelInfo k;
  k.decode = {ce::AttentionKernelOrigin::kCore, "row-chunk split-K", "", "", true};
  k.prefill = {ce::AttentionKernelOrigin::kCore, "flash wmma large", "", "", true};
  return k;
}
class FakeAttn final : public FakeModelBase, public ce::AttentionCapable {
 public:
  FakeAttn(ce::AttentionGeometry g, ce::AttentionKernelInfo k) : g_(std::move(g)), k_(std::move(k)) {}
  ce::AttentionGeometry AttentionGeometryInfo() const override { return g_; }
  ce::AttentionKernelInfo AttentionKernelsInfo() const override { return k_; }
 private:
  ce::AttentionGeometry g_;
  ce::AttentionKernelInfo k_;
};
static void PrintAttention(const ce::AttentionGeometry& g, const ce::AttentionKernelInfo& k) {
  for (const auto& a : g.groups)
    std::printf("       attention %s: %u layers, %u heads / %u kv heads, d%u, window %u%s, kv %s\n", ce::AttentionKindName(a.kind), a.layers, a.heads,
                a.kv_heads, a.head_dim, a.window, a.sink ? ", sink" : "", ce::AttentionKvFormatName(a.kv_format));
  auto path = [](const char* n, const ce::AttentionPathInfo& p) {
    std::printf("       %s: %s | %s | row-invariant %s\n", n, ce::AttentionKernelOriginName(p.origin), p.kernel.c_str(), p.row_invariant ? "yes" : "not claimed");
    if (!p.private_reason.empty()) std::printf("         private because: %s\n", p.private_reason.c_str());
    if (!p.exceptions.empty()) std::printf("         exceptions: %s\n", p.exceptions.c_str());
  };
  path("decode ", k.decode);
  path("prefill", k.prefill);
}

// ---- kernel conformance: a row's output must not depend on the other rows of the batch ----
#define CKH(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

static void WmmaRowInvariance(bool q8, std::uint32_t heads, std::uint32_t kv, std::uint32_t d, std::uint32_t start, std::uint32_t n, std::uint32_t window) {
  namespace fw = mbufo::core::compute::flash_wmma_large;
  std::mt19937 rng(7 + d + start);
  const std::uint32_t kvlen = start + n, nb = d / 32;
  const std::size_t keys = static_cast<std::size_t>(kvlen) * kv;
  std::vector<__half> hk, hv; std::vector<std::uint8_t> bk, bv;
  void *dk, *dv;
  if (q8) {
    bk.resize(keys * nb * 34); bv.resize(bk.size());
    for (auto* v : {&bk, &bv})
      for (std::size_t b = 0; b < keys * nb; ++b) {
        std::uint8_t* p = v->data() + b * 34;
        for (int i = 0; i < 32; ++i) p[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127));
        const __half sc = __float2half(0.004f + 0.01f * static_cast<float>(rng() % 100) / 100.0f);
        std::memcpy(p + 32, &sc, 2);
      }
    CKH(hipMalloc(&dk, bk.size())); CKH(hipMalloc(&dv, bv.size()));
    CKH(hipMemcpy(dk, bk.data(), bk.size(), hipMemcpyHostToDevice)); CKH(hipMemcpy(dv, bv.data(), bv.size(), hipMemcpyHostToDevice));
  } else {
    hk.resize(keys * d); hv.resize(hk.size());
    for (auto* v : {&hk, &hv}) for (auto& x : *v) x = __float2half(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.0f);
    CKH(hipMalloc(&dk, hk.size() * 2)); CKH(hipMalloc(&dv, hv.size() * 2));
    CKH(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice)); CKH(hipMemcpy(dv, hv.data(), hv.size() * 2, hipMemcpyHostToDevice));
  }
  std::vector<float> hq(static_cast<std::size_t>(n) * heads * d);
  for (auto& x : hq) x = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1500.0f;
  float *dq, *out; CKH(hipMalloc(&dq, hq.size() * 4)); CKH(hipMalloc(&out, hq.size() * 4));
  CKH(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
  auto run = [&](const float* q, float* o, std::uint32_t rows, std::uint32_t st) {
    return q8 ? fw::FlashWmmaLargePrefillQ8(q, static_cast<const std::uint8_t*>(dk), static_cast<const std::uint8_t*>(dv), o, rows, st, heads, kv, d, window, 0.0625f, 0)
              : fw::FlashWmmaLargePrefill(q, static_cast<const __half*>(dk), static_cast<const __half*>(dv), o, rows, st, heads, kv, d, window, 0.0625f, 0);
  };
  bool ok = run(dq, out, n, start);
  CKH(hipDeviceSynchronize());
  std::vector<float> batch(hq.size()); CKH(hipMemcpy(batch.data(), out, batch.size() * 4, hipMemcpyDeviceToHost));
  std::uint32_t diff = 0;
  float* o1; CKH(hipMalloc(&o1, static_cast<std::size_t>(heads) * d * 4));
  for (std::uint32_t t = 0; t < n && ok; ++t) {
    ok &= run(dq + static_cast<std::size_t>(t) * heads * d, o1, 1, start + t);
    CKH(hipDeviceSynchronize());
    std::vector<float> one(static_cast<std::size_t>(heads) * d); CKH(hipMemcpy(one.data(), o1, one.size() * 4, hipMemcpyDeviceToHost));
    diff += std::memcmp(one.data(), batch.data() + static_cast<std::size_t>(t) * heads * d, one.size() * 4) != 0;
  }
  char what[200];
  std::snprintf(what, sizeof what, "flash_wmma_large %s h%u kv%u d%u start %u n %u window %u: row t of the batch == the row alone (%u of %u differ)", q8 ? "blocked-Q8" : "F16", heads, kv, d, start, n, window, diff, n);
  Check(ok, "info (not required of prefill): " + std::string(what));
  CKH(hipFree(dk)); CKH(hipFree(dv)); CKH(hipFree(dq)); CKH(hipFree(out)); CKH(hipFree(o1));
}

static void RowChunkRowInvariance(std::uint32_t start, std::uint32_t base) {
  namespace sk = mbufo::core::compute::blocked_splitk;
  constexpr std::uint32_t kH = 16, kKv = 2, kDim = 256, kBlk = 34, n = 8;
  std::mt19937 rng(99 + start);
  const std::size_t keys = start + n + 8, rb = kKv * kDim / 32;
  std::vector<std::uint8_t> hk(keys * rb * kBlk), hv(hk.size());
  for (auto* v : {&hk, &hv})
    for (std::size_t b = 0; b < keys * rb; ++b) {
      std::uint8_t* p = v->data() + b * kBlk;
      for (int i = 0; i < 32; ++i) p[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127));
      const __half sc = __float2half(0.004f + 0.01f * static_cast<float>(rng() % 100) / 100.0f);
      std::memcpy(p + 32, &sc, 2);
    }
  std::vector<float> hq(static_cast<std::size_t>(n) * kH * kDim);
  for (auto& x : hq) x = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 700.0f;
  std::uint8_t *dk, *dv; float *dq, *out, *part; std::uint32_t* dpos;
  const std::size_t part_bytes = 40u << 20;
  CKH(hipMalloc(&dk, hk.size())); CKH(hipMalloc(&dv, hv.size())); CKH(hipMalloc(&dq, hq.size() * 4)); CKH(hipMalloc(&out, hq.size() * 4));
  CKH(hipMalloc(&part, part_bytes)); CKH(hipMalloc(&dpos, 4));
  CKH(hipMemcpy(dk, hk.data(), hk.size(), hipMemcpyHostToDevice)); CKH(hipMemcpy(dv, hv.data(), hv.size(), hipMemcpyHostToDevice));
  CKH(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
  auto set_pos = [&](std::uint32_t p) { CKH(hipMemcpy(dpos, &p, 4, hipMemcpyHostToDevice)); };
  set_pos(start);
  bool ok = sk::AttentionBlockedSplitKRowChunk<sk::KvCodecQ8, 256>(dq, dk, dv, out, n, dpos, start + n, kH, kKv, kDim, 0, 0.0625f, kBlk, kBlk, part, part_bytes, base);
  CKH(hipDeviceSynchronize());
  std::vector<float> batch(hq.size()); CKH(hipMemcpy(batch.data(), out, batch.size() * 4, hipMemcpyDeviceToHost));
  std::uint32_t diff = 0;
  for (std::uint32_t t = 0; t < n && ok; ++t) {
    set_pos(start + t);
    float* q1 = dq + static_cast<std::size_t>(t) * kH * kDim;
    ok &= sk::AttentionBlockedSplitKRowChunk<sk::KvCodecQ8, 256>(q1, dk, dv, out, 1, dpos, start + t + 1, kH, kKv, kDim, 0, 0.0625f, kBlk, kBlk, part, part_bytes, base);
    CKH(hipDeviceSynchronize());
    std::vector<float> one(static_cast<std::size_t>(kH) * kDim); CKH(hipMemcpy(one.data(), out, one.size() * 4, hipMemcpyDeviceToHost));
    diff += std::memcmp(one.data(), batch.data() + static_cast<std::size_t>(t) * kH * kDim, one.size() * 4) != 0;
  }
  char what[200];
  std::snprintf(what, sizeof what, "row-chunk split-K decode blocked-Q8 h16 kv2 d256 start %u base %u: row t of the batch == the row alone (%u of %u differ)", start, base, diff, n);
  Check(ok && diff == 0, what);
  CKH(hipFree(dk)); CKH(hipFree(dv)); CKH(hipFree(dq)); CKH(hipFree(out)); CKH(hipFree(part)); CKH(hipFree(dpos));
}
static std::string TryCreate(const char* name) {
  std::string err;
  auto m = ce::CreateEngineModel(name, ce::EngineLoadRequest{}, &err);
  return m ? std::string() : err;
}

int main(int argc, char** argv) {
  if (g_engine_link_refs[0] == nullptr) return 3;
  std::printf("contract validation\n");
  {
    auto rep = ce::ValidateMoeContract(GoodGeometry(), CoreKernels());
    Check(rep.ok(), "a coherent qwen35moe-shaped declaration validates");
    Check(ce::ExpectedRoutedBytesPerToken(GoodGeometry()) == 40ull * 8 * (512ull * (1152 + 1152) + 2048ull * 352), "implied routed bytes per token = layers x top_k x (gate+up+down rows)");
    ce::MoeGeometry dense;
    Check(ce::ValidateMoeContract(dense, CoreKernels()).ok() && ce::ExpectedRoutedBytesPerToken(dense) == 0, "a dense variant (moe_layers 0) validates and reads no experts");
    ce::MoeGeometry g = GoodGeometry(); g.top_k = 300;
    Check(!ce::ValidateMoeContract(g, CoreKernels()).ok(), "top_k above the expert count is refused");
    g = GoodGeometry(); g.formats[0].layers -= 1;
    Check(!ce::ValidateMoeContract(g, CoreKernels()).ok(), "formats that do not cover every MoE layer are refused");
    g = GoodGeometry(); g.hidden = 2000;
    Check(!ce::ValidateMoeContract(g, CoreKernels()).ok(), "a hidden width the K-quant block does not divide is refused");
    g = GoodGeometry(); g.formats[0].down_type = 99;
    Check(!ce::ValidateMoeContract(g, CoreKernels()).ok(), "an unknown format code is refused");
    ce::MoeKernelInfo k = CoreKernels(); k.origin = ce::MoeKernelOrigin::kModelPrivate;
    Check(!ce::ValidateMoeContract(GoodGeometry(), k).ok(), "private kernels without a reason are refused");
    k.private_reason = "upstream numerics";
    Check(ce::ValidateMoeContract(GoodGeometry(), k).ok(), "private kernels with a reason validate");
    k = CoreKernels(); k.private_reason = "x";
    Check(!ce::ValidateMoeContract(GoodGeometry(), k).ok(), "core kernels must not claim a private reason");
    k = CoreKernels(); k.down_kernel.clear();
    Check(!ce::ValidateMoeContract(GoodGeometry(), k).ok(), "unnamed kernel families are refused");
  }
  std::printf("registry enforcement\n");
  {
    auto reg = [](const char* name, bool moe, std::function<std::shared_ptr<ce::EngineModel>()> make) {
      ce::RegisterEngineBackend({name, "test", moe, false}, [make](const ce::EngineLoadRequest&, std::string*) { return make(); });
    };
    reg("t-moe-ok", true, [] { return std::make_shared<FakeMoe>(GoodGeometry(), CoreKernels()); });
    reg("t-moe-missing", true, [] { return std::make_shared<FakePlain>(); });
    reg("t-dense-capable", false, [] { return std::make_shared<FakeMoe>(GoodGeometry(), CoreKernels()); });
    reg("t-dense-ok", false, [] { return std::make_shared<FakePlain>(); });
    reg("t-moe-badgeom", true, [] { ce::MoeGeometry g = GoodGeometry(); g.experts = 0; return std::make_shared<FakeMoe>(g, CoreKernels()); });
    reg("t-moe-dense-variant", true, [] { return std::make_shared<FakeMoe>(ce::MoeGeometry{}, CoreKernels()); });
    Check(TryCreate("t-moe-ok").empty(), "an MoE engine whose model implements the contract is created");
    const std::string e1 = TryCreate("t-moe-missing");
    Check(e1.find("does not implement MoeCapable") != std::string::npos, "an MoE engine without MoeCapable is refused: " + e1);
    const std::string e2 = TryCreate("t-dense-capable");
    Check(e2.find("registered as dense") != std::string::npos, "a MoeCapable model registered as dense is refused: " + e2);
    Check(TryCreate("t-dense-ok").empty(), "a plain dense engine is unaffected");
    const std::string e3 = TryCreate("t-moe-badgeom");
    Check(e3.find("violates the MoE contract") != std::string::npos, "an incoherent declaration is refused at creation: " + e3);
    Check(TryCreate("t-moe-dense-variant").empty(), "a dense variant of an MoE engine (moe_layers 0) is created");
  }
  std::printf("attention contract validation\n");
  {
    Check(ce::ValidateAttentionContract(GoodAttention(), CoreAttention()).ok(), "a coherent core GQA declaration validates");
    ce::AttentionGeometry g = GoodAttention(); g.groups.clear();
    Check(!ce::ValidateAttentionContract(g, CoreAttention()).ok(), "an engine with no attention layers is refused (register with attention = false)");
    g = GoodAttention(); g.groups[0].kv_heads = 3;
    Check(!ce::ValidateAttentionContract(g, CoreAttention()).ok(), "heads not a multiple of kv_heads is refused");
    g = GoodAttention(); g.groups[0].head_dim = 100;
    Check(!ce::ValidateAttentionContract(g, CoreAttention()).ok(), "blocked-Q8 KV with head_dim not a multiple of 32 is refused");
    g = GoodAttention(); g.groups[0].head_dim = 100; g.groups[0].kv_format = ce::AttentionKvFormat::kF16;
    Check(ce::ValidateAttentionContract(g, CoreAttention()).ok(), "the same head_dim is accepted when the layer declares F16 KV only");
    ce::AttentionKernelInfo k = CoreAttention(); k.decode.origin = ce::AttentionKernelOrigin::kModelPrivate;
    Check(!ce::ValidateAttentionContract(GoodAttention(), k).ok(), "private decode kernels without a reason are refused");
    k.decode.private_reason = "upstream port";
    Check(ce::ValidateAttentionContract(GoodAttention(), k).ok(), "private decode kernels with a reason validate");
    k = CoreAttention(); k.decode.row_invariant = false;
    Check(!ce::ValidateAttentionContract(GoodAttention(), k).ok(), "a core decode path that is not row invariant is refused");
    k = CoreAttention(); k.prefill.row_invariant = false;
    Check(ce::ValidateAttentionContract(GoodAttention(), k).ok(), "a core prefill path need not be row invariant (informational)");
    k = CoreAttention(); k.decode.kernel.clear();
    Check(!ce::ValidateAttentionContract(GoodAttention(), k).ok(), "an unnamed kernel family is refused");
    ce::AttentionGeometry mla;
    ce::AttentionGroup m; m.kind = ce::AttentionKind::kMla; m.heads = 128; m.head_dim = 192; m.kv_format = ce::AttentionKvFormat::kModelSpecific;
    for (int i = 0; i < 43; ++i) mla.AddLayer(m);
    ce::AttentionKernelInfo priv = CoreAttention(); priv.decode = {ce::AttentionKernelOrigin::kModelPrivate, "mla", "compressed KV cache", "", false}; priv.prefill = priv.decode;
    Check(ce::ValidateAttentionContract(mla, priv).ok(), "an MLA engine with private kernels and a reason validates (the contract does not force GQA)");
    Check(!ce::ValidateAttentionContract(mla, CoreAttention()).ok(), "an MLA engine cannot claim the core softmax kernels");
    ce::AttentionGeometry hybrid = GoodAttention();
    ce::AttentionGroup lin; lin.kind = ce::AttentionKind::kLinearRecurrent; lin.heads = 48; lin.kv_heads = 16; lin.head_dim = 128; lin.kv_format = ce::AttentionKvFormat::kModelSpecific;
    for (int i = 0; i < 30; ++i) hybrid.AddLayer(lin);
    Check(ce::ValidateAttentionContract(hybrid, CoreAttention()).ok(), "a hybrid (linear + softmax) engine can claim core kernels for its softmax layers");
  }
  std::printf("attention registry enforcement\n");
  {
    auto reg = [](const char* name, bool attention, std::function<std::shared_ptr<ce::EngineModel>()> make) {
      ce::RegisterEngineBackend({name, "test", false, attention}, [make](const ce::EngineLoadRequest&, std::string*) { return make(); });
    };
    reg("a-ok", true, [] { return std::make_shared<FakeAttn>(GoodAttention(), CoreAttention()); });
    reg("a-missing", true, [] { return std::make_shared<FakePlain>(); });
    reg("a-none-ok", false, [] { return std::make_shared<FakePlain>(); });
    reg("a-bad", true, [] { ce::AttentionKernelInfo k = CoreAttention(); k.decode.row_invariant = false; return std::make_shared<FakeAttn>(GoodAttention(), k); });
    Check(TryCreate("a-ok").empty(), "an engine whose model declares its attention is created");
    const std::string e1 = TryCreate("a-missing");
    Check(e1.find("does not implement AttentionCapable") != std::string::npos, "an engine without AttentionCapable is refused: " + e1);
    Check(TryCreate("a-none-ok").empty(), "an engine registered with attention = false needs no declaration");
    const std::string e2 = TryCreate("a-bad");
    Check(e2.find("violates the attention contract") != std::string::npos, "an invalid declaration is refused at creation: " + e2);
  }
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "+gpu") != 0) continue;
    std::printf("kernel conformance: decode row invariance is required; prefill row dependence is reported for information\n");
    for (bool q8 : {false, true})
      for (std::uint32_t win : {0u, 128u}) {
        WmmaRowInvariance(q8, 8, 2, 256, 150, 40, win);
        WmmaRowInvariance(q8, 8, 2, 512, 60, 33, win);
        WmmaRowInvariance(q8, 16, 8, 256, 0, 70, win);
      }
    for (std::uint32_t start : {0u, 100u, 300u, 4000u})
      for (std::uint32_t base : {32u, 128u}) RowChunkRowInvariance(start, base);
  }  // every registered engine that claims MoE is either core row-dot with registered formats, or private with a reason
  for (int a = 1; a < argc; ++a) {
    // argument forms: engine=path (must load), !engine=path (must be refused), engine=path+run (also decode a few tokens)
    std::string arg = argv[a];
    const bool expect_refusal = !arg.empty() && arg[0] == '!';
    if (expect_refusal) arg.erase(0, 1);
    const bool smoke = arg.size() > 4 && arg.compare(arg.size() - 4, 4, "+run") == 0;
    if (smoke) arg.erase(arg.size() - 4);
    const auto eq_pos = arg.find('=');
    if (eq_pos == std::string::npos) continue;
    const std::string name = arg.substr(0, eq_pos), path = arg.substr(eq_pos + 1);
    std::printf("engine %s  (%s)%s\n", name.c_str(), path.c_str(), expect_refusal ? "  [must be refused]" : "");
    ce::EngineLoadRequest req;
    req.model_path = path;
    req.max_context = 4096;
    std::string err;
    auto model = ce::CreateEngineModel(name, req, &err);
    if (expect_refusal) {
      Check(model == nullptr, "refused as expected: " + err);
      continue;
    }
    Check(model != nullptr, "created through the registry (contract enforced)" + (model ? std::string() : ": " + err));
    if (!model) continue;
    if (const auto* attn = dynamic_cast<const ce::AttentionCapable*>(model.get())) {
      const ce::AttentionGeometry ag = attn->AttentionGeometryInfo();
      const ce::AttentionKernelInfo ak = attn->AttentionKernelsInfo();
      PrintAttention(ag, ak);
      Check(ce::ValidateAttentionContract(ag, ak).ok(), "attention declaration validates");
    } else {
      bool wants = true;
      for (const auto& b : ce::ListEngineBackends()) if (b.name == name) wants = b.attention;
      Check(!wants, "an engine without AttentionCapable is registered with attention = false");
    }    bool registered_moe = false;
    for (const auto& b : ce::ListEngineBackends()) registered_moe |= (b.name == name && b.moe);
    const auto* moe = dynamic_cast<const ce::MoeCapable*>(model.get());
    if (!registered_moe) {
      Check(moe == nullptr, "a dense engine does not implement MoeCapable");
      continue;
    }
    Check(moe != nullptr, "implements MoeCapable");
    if (moe == nullptr) continue;
    const ce::MoeGeometry g = moe->MoeGeometryInfo();
    const ce::MoeKernelInfo k = moe->MoeKernelsInfo();
    std::printf("       %u MoE layers, %u experts top-%u, hidden %u, expert_ff %u, shared_ff %u\n", g.moe_layers, g.experts, g.top_k, g.hidden, g.expert_ff, g.shared_ff);
    for (const auto& f : g.formats) std::printf("       formats gate %d up %d down %d x %u layers\n", f.gate_type, f.up_type, f.down_type, f.layers);
    std::printf("       kernels: %s | gate/up: %s | down: %s\n", ce::MoeKernelOriginName(k.origin), k.gate_up_kernel.c_str(), k.down_kernel.c_str());
    if (!k.private_reason.empty()) std::printf("       private because: %s\n", k.private_reason.c_str());
    const std::uint64_t bytes = ce::ExpectedRoutedBytesPerToken(g);
    std::printf("       implied routed-expert bytes per token: %.3f GB (at 226 GB/s: %.1f ms)\n", bytes / 1e9, bytes / 226e9 * 1e3);
    Check(ce::ValidateMoeContract(g, k).ok(), "declaration validates");
    if (smoke) {
      auto run = [&](std::vector<std::int32_t>& out) {
        std::string e;
        auto session = model->CreateSession(&e);
        if (!session) { std::printf("       session: %s\n", e.c_str()); return false; }
        auto* spec = dynamic_cast<ce::SpeculativeSession*>(session.get());
        const auto ids = model->Tokenize("The capital of France is", true);
        if (spec == nullptr || !session->Sync(ids, &e)) { std::printf("       sync: %s\n", e.c_str()); return false; }
        for (int i = 0; i < 12; ++i) {
          ce::DecodeOutcome o;
          if (!spec->DecodeStep(0, &o, &e)) { std::printf("       decode: %s\n", e.c_str()); return false; }
          out.push_back(o.token);
        }
        return true;
      };
      std::vector<std::int32_t> t1, t2;
      const bool ok1 = run(t1), ok2 = run(t2);
      std::printf("       decoded:");
      for (auto t : t1) std::printf(" %d", t);
      std::printf("\n");
      bool varied = false;
      for (auto t : t1) varied |= t != t1[0];
      Check(ok1 && ok2 && t1.size() == 12, "creates a session, syncs a prompt and decodes 12 greedy tokens through the registry engine");
      Check(t1 == t2, "two fresh sessions decode identical greedy tokens");
      Check(varied, "the decoded tokens are not degenerate");
    }
    if (k.origin == ce::MoeKernelOrigin::kCoreRowDot) {
      bool all = true;
      for (const auto& f : g.formats)
        for (int t : {f.gate_type, f.up_type, f.down_type}) all &= q::RowFormats::Dispatch(t, [](auto) {});
      Check(all, "every expert format is served by the core row-dot registry (a core kernel change reaches this engine)");
    }
  }
  std::printf(g_bad ? "FAILED (%d)\n" : "all OK\n", g_bad);
  return g_bad ? 1 : 0;
}
