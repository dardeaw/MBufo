#include "src/core/compute/npu/npu_op_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

#include "src/core/compute/gelu/gelu.hpp"
#include "src/core/compute/layernorm/layernorm.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace mbufo::core::npu::graph {
namespace {

using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

float F16ToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15U) & 1U;
  const std::uint32_t exp = (h >> 10U) & 0x1FU;
  const std::uint32_t man = h & 0x3FFU;
  float v;
  if (exp == 0) {
    v = std::ldexp(static_cast<float>(man), -24);
  } else if (exp == 31) {
    v = man == 0 ? INFINITY : NAN;
  } else {
    v = std::ldexp(static_cast<float>(man | 0x400U),
                   static_cast<int>(exp) - 25);
  }
  return sign != 0U ? -v : v;
}

float Bf16ToFloat(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16U;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline float Silu(float x) {
  return x / (1.0F + std::exp(-x));
}

inline float GeluTanh(float x) {
  const float u = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + std::tanh(u));
}

// The engine packer (EncodeBfp16Block8) maps every 8 consecutive K values of
// one output row onto a shared block exponent and keeps 7 magnitude bits via
// an arithmetic shift, i.e. floor(v / Q) with Q = 2^(E_max - 133): a
// systematic -0.5 LSB bias. Pre-rounding each value to the nearest multiple
// of Q makes the packer's floor reproduce the rounded value exactly (the
// product q*Q is exact in fp32), halving the error and removing the bias.
void RoundBfp16Ebs8NK(float* w, std::size_t N, std::size_t K) {
#pragma omp parallel for schedule(static)
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t g = 0; g + 8 <= K; g += 8) {
      float* p = w + n * K + g;
      std::uint32_t max_exp = 0;
      bool finite = true;
      for (int i = 0; i < 8; ++i) {
        if (!std::isfinite(p[i]))
          finite = false;
        std::uint32_t bits;
        std::memcpy(&bits, &p[i], sizeof(bits));
        max_exp = std::max<std::uint32_t>(max_exp, (bits >> 23U) & 0xFFU);
      }
      if (!finite || max_exp < 16U)
        continue;  // zero/denormal: leave
      const float q = std::ldexp(1.0F, static_cast<int>(max_exp) - 133);
      for (int i = 0; i < 8; ++i) {
        const float r = std::nearbyint(p[i] / q);
        p[i] = std::clamp(r, -127.0F, 127.0F) * q;
      }
    }
  }
}

std::string Fail(std::string* error, const std::string& msg) {
  if (error != nullptr)
    *error = msg;
  return msg;
}

}  // namespace

// ------------------------------------------------------------------ Tensor --

Tensor Tensor::ViewF32(const float* p, std::size_t rows, std::size_t cols) {
  return View(TensorKind::kF32, p, rows, cols);
}

Tensor Tensor::Owned(std::vector<float> v, std::size_t rows, std::size_t cols) {
  Tensor t;
  t.kind = TensorKind::kF32;
  t.rows = rows;
  t.cols = cols;
  t.owned = std::move(v);
  t.data = t.owned.data();
  return t;
}

Tensor Tensor::View(TensorKind kind, const void* p, std::size_t rows,
                    std::size_t cols) {
  Tensor t;
  t.kind = kind;
  t.rows = rows;
  t.cols = cols;
  t.data = p;
  return t;
}

void Tensor::ReadRow(std::size_t r, float* dst) const {
  switch (kind) {
    case TensorKind::kF32:
      std::memcpy(dst, static_cast<const float*>(data) + r * cols,
                  cols * sizeof(float));
      break;
    case TensorKind::kBf16: {
      const auto* p = static_cast<const std::uint16_t*>(data) + r * cols;
      for (std::size_t i = 0; i < cols; ++i)
        dst[i] = Bf16ToFloat(p[i]);
      break;
    }
    case TensorKind::kF16: {
      const auto* p = static_cast<const std::uint16_t*>(data) + r * cols;
      for (std::size_t i = 0; i < cols; ++i)
        dst[i] = F16ToFloat(p[i]);
      break;
    }
    case TensorKind::kQ8_0: {
      constexpr std::size_t kBlk = 34;  // fp16 scale + 32 x int8
      const auto* p =
          static_cast<const std::uint8_t*>(data) + r * (cols / 32U) * kBlk;
      mbufo::quant::DequantizeQ8_0(p, dst, cols);
      break;
    }
  }
}

void Tensor::ReadAll(std::vector<float>* out) const {
  out->resize(rows * cols);
#pragma omp parallel for schedule(static) if (rows > 64)
  for (std::size_t r = 0; r < rows; ++r)
    ReadRow(r, out->data() + r * cols);
}

// ---------------------------------------------------------------- builders --

namespace {
std::string L(std::size_t i, const char* s) {
  return "L" + std::to_string(i) + "." + s;
}
Op MakeLinear(std::string in, std::string out, std::string w, std::string b) {
  Op o;
  o.kind = OpKind::kLinear;
  o.in = std::move(in);
  o.out = std::move(out);
  o.w = std::move(w);
  o.b = std::move(b);
  return o;
}
}  // namespace

Graph BuildBertClsGraph(const BertGraphConfig& c) {
  Graph g;
  g.name = "bert_cls";
  Op e;
  e.kind = OpKind::kEmbed;
  e.out = "h0";
  e.w = "tok";
  e.b = "pos";
  e.w2 = "type";
  g.ops.push_back(e);
  Op ln;
  ln.kind = OpKind::kLayerNorm;
  ln.in = "h0";
  ln.out = "h";
  ln.w = "emb_ln.w";
  ln.b = "emb_ln.b";
  ln.eps = c.eps;
  g.ops.push_back(ln);
  for (std::size_t i = 0; i < c.layers; ++i) {
    g.ops.push_back(MakeLinear("h", "q", L(i, "q.w"), L(i, "q.b")));
    g.ops.push_back(MakeLinear("h", "k", L(i, "k.w"), L(i, "k.b")));
    g.ops.push_back(MakeLinear("h", "v", L(i, "v.w"), L(i, "v.b")));
    Op a;
    a.kind = OpKind::kAttention;
    a.in = "q";
    a.in2 = "k";
    a.in3 = "v";
    a.out = "ctx";
    a.heads = a.kv_heads = static_cast<std::uint32_t>(c.heads);
    a.head_dim = static_cast<std::uint32_t>(c.hidden / c.heads);
    a.causal = false;
    g.ops.push_back(a);
    g.ops.push_back(MakeLinear("ctx", "proj", L(i, "o.w"), L(i, "o.b")));
    Op r1;
    r1.kind = OpKind::kResidual;
    r1.in = "h";
    r1.in2 = "proj";
    r1.out = "r1";
    g.ops.push_back(r1);
    Op n1;
    n1.kind = OpKind::kLayerNorm;
    n1.in = "r1";
    n1.out = "a";
    n1.w = L(i, "attn_ln.w");
    n1.b = L(i, "attn_ln.b");
    n1.eps = c.eps;
    g.ops.push_back(n1);
    g.ops.push_back(MakeLinear("a", "up", L(i, "up.w"), L(i, "up.b")));
    Op ge;
    ge.kind = OpKind::kGelu;
    ge.in = "up";
    ge.out = "up";
    g.ops.push_back(ge);
    g.ops.push_back(MakeLinear("up", "dn", L(i, "down.w"), L(i, "down.b")));
    Op r2;
    r2.kind = OpKind::kResidual;
    r2.in = "a";
    r2.in2 = "dn";
    r2.out = "r2";
    g.ops.push_back(r2);
    Op n2;
    n2.kind = OpKind::kLayerNorm;
    n2.in = "r2";
    n2.out = "h";
    n2.w = L(i, "out_ln.w");
    n2.b = L(i, "out_ln.b");
    n2.eps = c.eps;
    g.ops.push_back(n2);
  }
  Op p;
  p.kind = OpKind::kPool;
  p.in = "h";
  p.out = "cls";
  p.pool = PoolMode::kCls;
  g.ops.push_back(p);
  g.output = "cls";
  return g;
}

namespace {
void AppendQwen3Body(Graph* g, const Qwen3GraphConfig& c) {
  Op e;
  e.kind = OpKind::kEmbed;
  e.out = "h";
  e.w = "tok";
  g->ops.push_back(e);
  for (std::size_t i = 0; i < c.layers; ++i) {
    Op n1;
    n1.kind = OpKind::kRmsNorm;
    n1.in = "h";
    n1.out = "x";
    n1.w = L(i, "attn_norm.w");
    n1.eps = c.eps;
    g->ops.push_back(n1);
    g->ops.push_back(MakeLinear("x", "q", L(i, "q.w"), ""));
    g->ops.push_back(MakeLinear("x", "k", L(i, "k.w"), ""));
    g->ops.push_back(MakeLinear("x", "v", L(i, "v.w"), ""));
    Op a;
    a.kind = OpKind::kAttention;
    a.in = "q";
    a.in2 = "k";
    a.in3 = "v";
    a.out = "ctx";
    a.w2 = L(i, "q_norm.w");
    a.w3 = L(i, "k_norm.w");
    a.eps = c.eps;
    a.heads = static_cast<std::uint32_t>(c.heads);
    a.kv_heads = static_cast<std::uint32_t>(c.kv_heads);
    a.head_dim = static_cast<std::uint32_t>(c.head_dim);
    a.causal = true;
    a.rope_theta = c.rope_theta;
    g->ops.push_back(a);
    g->ops.push_back(MakeLinear("ctx", "proj", L(i, "o.w"), ""));
    Op r1;
    r1.kind = OpKind::kResidual;
    r1.in = "h";
    r1.in2 = "proj";
    r1.out = "h";
    g->ops.push_back(r1);
    Op n2;
    n2.kind = OpKind::kRmsNorm;
    n2.in = "h";
    n2.out = "x";
    n2.w = L(i, "ffn_norm.w");
    n2.eps = c.eps;
    g->ops.push_back(n2);
    g->ops.push_back(MakeLinear("x", "gate", L(i, "gate.w"), ""));
    g->ops.push_back(MakeLinear("x", "up", L(i, "up.w"), ""));
    Op sw;
    sw.kind = OpKind::kSwiGlu;
    sw.in = "gate";
    sw.in2 = "up";
    sw.out = "act";
    g->ops.push_back(sw);
    g->ops.push_back(MakeLinear("act", "dn", L(i, "down.w"), ""));
    Op r2;
    r2.kind = OpKind::kResidual;
    r2.in = "h";
    r2.in2 = "dn";
    r2.out = "h";
    g->ops.push_back(r2);
  }
  // Last-token pool BEFORE the final RMSNorm: the norm is per-row, so this is
  // identical to normalizing every row first and avoids n-1 wasted rows.
  Op p;
  p.kind = OpKind::kPool;
  p.in = "h";
  p.out = "last";
  p.pool = PoolMode::kLast;
  g->ops.push_back(p);
  Op fn;
  fn.kind = OpKind::kRmsNorm;
  fn.in = "last";
  fn.out = "last_n";
  fn.w = "out_norm.w";
  fn.eps = c.eps;
  g->ops.push_back(fn);
}
}  // namespace

Graph BuildQwen3EmbeddingGraph(const Qwen3GraphConfig& c) {
  Graph g;
  g.name = "qwen3_embedding";
  AppendQwen3Body(&g, c);
  Op l2;
  l2.kind = OpKind::kL2Norm;
  l2.in = "last_n";
  l2.out = "emb";
  g.ops.push_back(l2);
  g.output = "emb";
  return g;
}

Graph BuildQwen3RerankGraph(const Qwen3GraphConfig& c,
                            std::vector<std::uint32_t> probe_ids) {
  Graph g;
  g.name = "qwen3_rerank";
  AppendQwen3Body(&g, c);
  Op rd;
  rd.kind = OpKind::kRowDot;
  rd.in = "last_n";
  rd.out = "logits";
  rd.w = "tok";  // tied lm-head
  rd.ids = std::move(probe_ids);
  g.ops.push_back(rd);
  g.output = "logits";
  return g;
}

Graph BuildEmbGemma2Graph(const EmbGemma2GraphConfig& c) {
  Graph g;
  g.name = "embgemma2";
  Op e;
  e.kind = OpKind::kEmbed;
  e.out = "emb";
  e.w = "tok";
  e.scale = c.embed_scale;
  g.ops.push_back(e);
  auto norm = [&](const char* in, const char* out, std::string w,
                  float pre_scale = 1.0F) {
    Op n;
    n.kind = OpKind::kRmsNorm;
    n.in = in;
    n.out = out;
    n.w = std::move(w);
    n.eps = c.eps;
    n.scale = pre_scale;
    g.ops.push_back(n);
  };
  auto geglu = [&](const char* gate, const char* up, const char* out) {
    Op ge;
    ge.kind = OpKind::kGeGlu;
    ge.in = gate;
    ge.in2 = up;
    ge.out = out;
    g.ops.push_back(ge);
  };
  auto residual = [&](const char* a, const char* b, const char* out,
                      float scale) {
    Op r;
    r.kind = OpKind::kResidual;
    r.in = a;
    r.in2 = b;
    r.out = out;
    r.scale = scale;
    g.ops.push_back(r);
  };
  const float ple_pre = 1.0F / std::sqrt(static_cast<float>(c.hidden));
  for (std::size_t i = 0; i < c.layers; ++i) {
    const bool full = c.layer_is_full.at(i) != 0;
    const char* cur = i == 0 ? "emb" : "h";
    norm(cur, "x", L(i, "in_norm.w"));
    g.ops.push_back(MakeLinear("x", "q", L(i, "q.w"), ""));
    g.ops.push_back(MakeLinear("x", "k", L(i, "k.w"), ""));
    g.ops.push_back(MakeLinear("x", "v", L(i, "v.w"), ""));
    Op a;
    a.kind = OpKind::kAttention;
    a.in = "q";
    a.in2 = "k";
    a.in3 = "v";
    a.out = "ctx";
    a.w2 = L(i, "q_norm.w");
    a.w3 = L(i, "k_norm.w");
    a.v_norm = true;
    a.eps = c.eps;
    a.heads = static_cast<std::uint32_t>(c.heads);
    a.kv_heads = static_cast<std::uint32_t>(full ? c.kv_full : c.kv_local);
    a.head_dim =
        static_cast<std::uint32_t>(full ? c.head_dim_full : c.head_dim_local);
    a.causal = false;
    a.rope_theta = full ? c.rope_full : c.rope_local;
    a.window = full ? 0U : static_cast<std::uint32_t>(c.sliding_window);
    a.attn_scale = 1.0F;
    g.ops.push_back(a);
    g.ops.push_back(MakeLinear("ctx", "proj", L(i, "o.w"), ""));
    norm("proj", "pn", L(i, "post_attn.w"));
    residual(cur, "pn", "mid", 1.0F);
    norm("mid", "x", L(i, "pre_ffn.w"));
    g.ops.push_back(MakeLinear("x", "gate", L(i, "gate.w"), ""));
    g.ops.push_back(MakeLinear("x", "up", L(i, "up.w"), ""));
    geglu("gate", "up", "act");
    g.ops.push_back(MakeLinear("act", "dn", L(i, "down.w"), ""));
    norm("dn", "dnn", L(i, "post_ffn.w"));
    residual("mid", "dnn", "h", 1.0F);
    // Per-layer input block: gate the layer output with a projection of the
    // (scaled) token embedding, then apply the layer scalar.
    g.ops.push_back(MakeLinear("emb", "pg", L(i, "ple_proj.w"), ""));
    norm("pg", "pl", "ple_norm.w", ple_pre);
    g.ops.push_back(MakeLinear("h", "pgate", L(i, "ple_gate.w"), ""));
    geglu("pgate", "pl", "pact");
    g.ops.push_back(MakeLinear("pact", "py", L(i, "ple_out.w"), ""));
    norm("py", "pyn", L(i, "ple_post.w"));
    residual("h", "pyn", "h", c.layer_scalar.at(i));
  }
  norm("h", "fn", "final_norm.w");
  Op p;
  p.kind = OpKind::kPool;
  p.in = "fn";
  p.out = "mean";
  p.pool = PoolMode::kMean;
  g.ops.push_back(p);
  Op proj = MakeLinear("mean", "proj_out", "out_proj.w", "");
  proj.host_f32 = true;
  g.ops.push_back(proj);
  Op l2;
  l2.kind = OpKind::kL2Norm;
  l2.in = "proj_out";
  l2.out = "emb_out";
  g.ops.push_back(l2);
  g.output = "emb_out";
  return g;
}

// ---------------------------------------------------------------- executor --

bool GraphExecutor::Prepare(Graph graph, const WeightStore* weights,
                            std::shared_ptr<NpuQuantGemmEngine> engine,
                            LinearBackend backend, std::string* error) {
  prepared_ = false;
  packed_.clear();
  hostw_.clear();
  bias_.clear();
  small_.clear();
  bufs_.clear();
  graph_ = std::move(graph);
  weights_ = weights;
  engine_ = std::move(engine);
  backend_ = backend;
  if (weights_ == nullptr) {
    Fail(error, "graph: null weight store");
    return false;
  }
  if (backend_ == LinearBackend::kNpuQ8 &&
      (engine_ == nullptr || !engine_->ready())) {
    Fail(error, "graph: NPU engine not ready");
    return false;
  }
  auto need = [&](const std::string& name) -> const Tensor* {
    const Tensor* t = weights_->Find(name);
    if (t == nullptr)
      Fail(error, "graph: missing weight '" + name + "'");
    return t;
  };
  for (const auto& op : graph_.ops) {
    switch (op.kind) {
      case OpKind::kLinear: {
        if (packed_.count(op.w) != 0 || hostw_.count(op.w) != 0) {
          // shared weight: pack once
        } else {
          const Tensor* w = need(op.w);
          if (w == nullptr)
            return false;
          if (backend_ == LinearBackend::kNpuQ8 && !op.host_f32) {
            if (w->cols % 256U != 0 || w->rows % 256U != 0) {
              Fail(error, "graph: Linear '" + op.w +
                              "' needs K and N multiples of 256");
              return false;
            }
            std::vector<float> tmp;
            w->ReadAll(&tmp);
            if (std::getenv("MBUFO_NPU_GRAPH_FLOOR") == nullptr) {
              RoundBfp16Ebs8NK(tmp.data(), w->rows, w->cols);
            }
            QuantWeightMatrix qm;
            if (!engine_->PackMatrixNK(tmp.data(), w->rows, w->cols,
                                       NpuQuantMode::kQ8Bfp16,
                                       resident_weights_, &qm, error)) {
              return false;
            }
            packed_.emplace(op.w, std::move(qm));
          } else {
            std::vector<float> tmp;
            w->ReadAll(&tmp);
            hostw_.emplace(op.w, std::move(tmp));
          }
        }
        if (!op.b.empty() && bias_.count(op.b) == 0) {
          const Tensor* b = need(op.b);
          if (b == nullptr)
            return false;
          std::vector<float> bv;
          b->ReadAll(&bv);
          bias_.emplace(op.b, std::move(bv));
        }
        break;
      }
      case OpKind::kLayerNorm:
      case OpKind::kRmsNorm:
      case OpKind::kAttention: {
        for (const std::string* nm : {&op.w, &op.w2, &op.w3, &op.b}) {
          if (nm->empty() || small_.count(*nm) != 0)
            continue;
          const Tensor* t = need(*nm);
          if (t == nullptr)
            return false;
          std::vector<float> v;
          t->ReadAll(&v);
          small_.emplace(*nm, std::move(v));
        }
        break;
      }
      case OpKind::kEmbed:
      case OpKind::kRowDot:
        if (need(op.w) == nullptr)
          return false;
        for (const std::string* nm : {&op.b, &op.w2}) {
          if (!nm->empty() && need(*nm) == nullptr)
            return false;
        }
        break;
      default:
        break;
    }
  }
  prepared_ = true;
  return true;
}

bool GraphExecutor::Run(std::span<const std::uint32_t> ids,
                        std::vector<float>* out, RunStats* stats,
                        std::string* error) {
  if (!prepared_) {
    Fail(error, "graph: not prepared");
    return false;
  }
  if (ids.empty()) {
    Fail(error, "graph: empty input");
    return false;
  }
  RunStats local;
  RunStats* st = stats != nullptr ? stats : &local;
  *st = RunStats{};
  const auto t0 = Clock::now();
  for (const auto& op : graph_.ops) {
    if (!RunOp(op, ids, st, error))
      return false;
  }
  st->total_ms = MsSince(t0);
  Buf* o = Get(graph_.output);
  if (o == nullptr) {
    Fail(error, "graph: output buffer '" + graph_.output + "' missing");
    return false;
  }
  *out = o->v;
  return true;
}

bool GraphExecutor::RunOp(const Op& op, std::span<const std::uint32_t> ids,
                          RunStats* st, std::string* error) {
  const auto t0 = Clock::now();
  auto input = [&](const std::string& name) -> Buf* {
    Buf* b = Get(name);
    if (b == nullptr)
      Fail(error, "graph: buffer '" + name + "' not produced");
    return b;
  };
  auto set = [&](const std::string& name, std::vector<float> v,
                 std::size_t rows, std::size_t cols) {
    Buf& b = bufs_[name];
    b.v = std::move(v);
    b.rows = rows;
    b.cols = cols;
  };
  bool ok = true;
  bool is_linear = false;
  bool is_attn = false;

  switch (op.kind) {
    case OpKind::kEmbed: {
      const Tensor* tok = weights_->Find(op.w);
      const Tensor* pos = op.b.empty() ? nullptr : weights_->Find(op.b);
      const Tensor* typ = op.w2.empty() ? nullptr : weights_->Find(op.w2);
      const std::size_t n = ids.size();
      const std::size_t d = tok->cols;
      if (pos != nullptr && pos->rows < n) {
        Fail(error, "graph: sequence longer than position table");
        return false;
      }
      std::vector<float> out(n * d);
      std::vector<float> tmp(d);
      for (std::size_t t = 0; t < n; ++t) {
        if (ids[t] >= tok->rows) {
          Fail(error, "graph: token id out of vocab");
          return false;
        }
        float* row = out.data() + t * d;
        tok->ReadRow(ids[t], row);
        if (pos != nullptr) {
          pos->ReadRow(t, tmp.data());
          for (std::size_t c = 0; c < d; ++c)
            row[c] += tmp[c];
        }
        if (typ != nullptr) {
          typ->ReadRow(0, tmp.data());
          for (std::size_t c = 0; c < d; ++c)
            row[c] += tmp[c];
        }
        if (op.scale != 1.0F) {
          for (std::size_t c = 0; c < d; ++c)
            row[c] *= op.scale;
        }
      }
      set(op.out, std::move(out), n, d);
      break;
    }
    case OpKind::kLayerNorm: {
      Buf* in = input(op.in);
      if (in == nullptr)
        return false;
      const auto& w = small_.at(op.w);
      const auto& b = small_.at(op.b);
      std::vector<float> out(in->rows * in->cols);
      for (std::size_t r = 0; r < in->rows; ++r) {
        if (!core::ln::LayerNorm(
                std::span<const float>(in->v.data() + r * in->cols, in->cols),
                std::span<const float>(w.data(), in->cols),
                std::span<const float>(b.data(), in->cols), op.eps,
                std::span<float>(out.data() + r * in->cols, in->cols))) {
          Fail(error, "graph: LayerNorm failed");
          return false;
        }
      }
      set(op.out, std::move(out), in->rows, in->cols);
      break;
    }
    case OpKind::kRmsNorm: {
      Buf* in = input(op.in);
      if (in == nullptr)
        return false;
      const float* w = op.w.empty() ? nullptr : small_.at(op.w).data();
      const float pre = op.scale;
      std::vector<float> out(in->rows * in->cols);
      for (std::size_t r = 0; r < in->rows; ++r) {
        const float* x = in->v.data() + r * in->cols;
        double ss = 0.0;
        for (std::size_t c = 0; c < in->cols; ++c) {
          const double xv = double(x[c]) * pre;
          ss += xv * xv;
        }
        const float inv = static_cast<float>(
            1.0 / std::sqrt(ss / static_cast<double>(in->cols) + op.eps));
        float* y = out.data() + r * in->cols;
        for (std::size_t c = 0; c < in->cols; ++c) {
          y[c] = x[c] * pre * inv * (w != nullptr ? w[c] : 1.0F);
        }
      }
      set(op.out, std::move(out), in->rows, in->cols);
      break;
    }
    case OpKind::kLinear: {
      is_linear = true;
      Buf* in = input(op.in);
      if (in == nullptr)
        return false;
      const float* bias = op.b.empty() ? nullptr : bias_.at(op.b).data();
      std::size_t N = 0;
      std::vector<float> out;
      if (backend_ == LinearBackend::kNpuQ8 && !op.host_f32) {
        const auto& qm = packed_.at(op.w);
        N = qm.N;
        if (qm.K != in->cols) {
          Fail(error, "graph: Linear '" + op.w + "' K mismatch");
          return false;
        }
        out.resize(in->rows * N);
        if (!engine_->Gemm(in->v.data(), in->rows, qm, out.data(), bias,
                           &st->npu_ms, error)) {
          return false;
        }
      } else {
        const Tensor* wt = weights_->Find(op.w);
        const auto& w = hostw_.at(op.w);
        N = wt->rows;
        const std::size_t K = wt->cols;
        if (K != in->cols) {
          Fail(error, "graph: Linear '" + op.w + "' K mismatch");
          return false;
        }
        out.assign(in->rows * N, 0.0F);
#pragma omp parallel for collapse(2) schedule(static)
        for (std::size_t r = 0; r < in->rows; ++r) {
          for (std::size_t n = 0; n < N; ++n) {
            const float* a = in->v.data() + r * K;
            const float* b = w.data() + n * K;
            double acc = 0.0;
            for (std::size_t k = 0; k < K; ++k)
              acc += double(a[k]) * b[k];
            out[r * N + n] =
                static_cast<float>(acc) + (bias != nullptr ? bias[n] : 0.0F);
          }
        }
      }
      set(op.out, std::move(out), in->rows, N);
      break;
    }
    case OpKind::kAttention: {
      is_attn = true;
      Buf* q = input(op.in);
      Buf* k = input(op.in2);
      Buf* v = input(op.in3);
      if (q == nullptr || k == nullptr || v == nullptr)
        return false;
      const std::size_t n = q->rows;
      const std::size_t H = op.heads, KVH = op.kv_heads, hd = op.head_dim;
      if (H == 0 || KVH == 0 || hd == 0 || H % KVH != 0 || q->cols != H * hd ||
          k->cols != KVH * hd || v->cols != KVH * hd || k->rows != n ||
          v->rows != n || (hd % 2U) != 0) {
        Fail(error, "graph: Attention shape mismatch");
        return false;
      }
      // QK-norm (per-head RMSNorm) + RoPE operate on private copies.
      std::vector<float> qv = q->v;
      std::vector<float> kv = k->v;
      auto head_rms = [&](std::vector<float>& x, std::size_t heads,
                          const std::vector<float>* w) {
        for (std::size_t r = 0; r < n; ++r) {
          for (std::size_t h = 0; h < heads; ++h) {
            float* p = x.data() + r * heads * hd + h * hd;
            double ss = 0.0;
            for (std::size_t d = 0; d < hd; ++d)
              ss += double(p[d]) * p[d];
            const float inv = static_cast<float>(
                1.0 / std::sqrt(ss / static_cast<double>(hd) + op.eps));
            for (std::size_t d = 0; d < hd; ++d) {
              p[d] = p[d] * inv * (w != nullptr ? (*w)[d] : 1.0F);
            }
          }
        }
      };
      if (!op.w2.empty())
        head_rms(qv, H, &small_.at(op.w2));
      if (!op.w3.empty())
        head_rms(kv, KVH, &small_.at(op.w3));
      std::vector<float> vnorm;
      if (op.v_norm) {
        vnorm = v->v;
        head_rms(vnorm, KVH, nullptr);
      }
      const std::vector<float>& vdat = op.v_norm ? vnorm : v->v;
      if (op.rope_theta > 0.0F) {
        const std::size_t h2 = hd / 2;
        std::vector<float> inv_freq(h2);
        for (std::size_t i = 0; i < h2; ++i) {
          inv_freq[i] = static_cast<float>(std::pow(
              static_cast<double>(op.rope_theta),
              -2.0 * static_cast<double>(i) / static_cast<double>(hd)));
        }
        auto rope = [&](std::vector<float>& x, std::size_t heads) {
#pragma omp parallel for schedule(static)
          for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t i = 0; i < h2; ++i) {
              const double ang = static_cast<double>(r) * inv_freq[i];
              const float cs = static_cast<float>(std::cos(ang));
              const float sn = static_cast<float>(std::sin(ang));
              for (std::size_t h = 0; h < heads; ++h) {
                float* p = x.data() + r * heads * hd + h * hd;
                const float a = p[i], b = p[i + h2];
                p[i] = a * cs - b * sn;
                p[i + h2] = b * cs + a * sn;
              }
            }
          }
        };
        rope(qv, H);
        rope(kv, KVH);
      }
      const std::size_t group = H / KVH;
      const float scale = op.attn_scale >= 0.0F
                              ? op.attn_scale
                              : 1.0F / std::sqrt(static_cast<float>(hd));
      std::vector<float> ctx(n * H * hd, 0.0F);
#pragma omp parallel
      {
        std::vector<float> sc(n);
#pragma omp for collapse(2) schedule(dynamic, 4)
        for (std::size_t h = 0; h < H; ++h) {
          for (std::size_t i = 0; i < n; ++i) {
            const std::size_t kvh = h / group;
            const float* qi = qv.data() + i * H * hd + h * hd;
            std::size_t lo = 0;
            std::size_t lim = op.causal ? i + 1 : n;
            if (op.window > 0) {
              lo = i > op.window ? i - op.window : 0;
              lim = std::min<std::size_t>(lim, i + op.window + 1);
            }
            float mx = -INFINITY;
            for (std::size_t j = lo; j < lim; ++j) {
              const float* kj = kv.data() + j * KVH * hd + kvh * hd;
              float dot = 0.0F;
              for (std::size_t d = 0; d < hd; ++d)
                dot += qi[d] * kj[d];
              dot *= scale;
              sc[j] = dot;
              mx = std::max(mx, dot);
            }
            double sum = 0.0;
            for (std::size_t j = lo; j < lim; ++j) {
              sc[j] = std::exp(sc[j] - mx);
              sum += sc[j];
            }
            const float inv = static_cast<float>(1.0 / sum);
            float* ci = ctx.data() + i * H * hd + h * hd;
            for (std::size_t j = lo; j < lim; ++j) {
              const float w = sc[j] * inv;
              const float* vj = vdat.data() + j * KVH * hd + kvh * hd;
              for (std::size_t d = 0; d < hd; ++d)
                ci[d] += w * vj[d];
            }
          }
        }
      }
      set(op.out, std::move(ctx), n, H * hd);
      break;
    }
    case OpKind::kGelu: {
      Buf* in = input(op.in);
      if (in == nullptr)
        return false;
      std::vector<float> out(in->v.size());
      if (!core::gelu::ApplyGelu(std::span<const float>(in->v),
                                 std::span<float>(out),
                                 core::gelu::GeluMode::kExact)) {
        Fail(error, "graph: ApplyGelu failed");
        return false;
      }
      set(op.out, std::move(out), in->rows, in->cols);
      break;
    }
    case OpKind::kSwiGlu: {
      Buf* g = input(op.in);
      Buf* u = input(op.in2);
      if (g == nullptr || u == nullptr)
        return false;
      if (g->v.size() != u->v.size()) {
        Fail(error, "graph: SwiGlu shape mismatch");
        return false;
      }
      std::vector<float> out(g->v.size());
#pragma omp parallel for schedule(static)
      for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = Silu(g->v[i]) * u->v[i];
      }
      set(op.out, std::move(out), g->rows, g->cols);
      break;
    }
    case OpKind::kGeGlu: {
      Buf* g = input(op.in);
      Buf* u = input(op.in2);
      if (g == nullptr || u == nullptr)
        return false;
      if (g->v.size() != u->v.size()) {
        Fail(error, "graph: GeGlu shape mismatch");
        return false;
      }
      std::vector<float> out(g->v.size());
#pragma omp parallel for schedule(static)
      for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = GeluTanh(g->v[i]) * u->v[i];
      }
      set(op.out, std::move(out), g->rows, g->cols);
      break;
    }
    case OpKind::kResidual: {
      Buf* a = input(op.in);
      Buf* b = input(op.in2);
      if (a == nullptr || b == nullptr)
        return false;
      if (a->v.size() != b->v.size()) {
        Fail(error, "graph: Residual shape mismatch");
        return false;
      }
      std::vector<float> out(a->v.size());
      for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = (a->v[i] + b->v[i]) * op.scale;
      }
      set(op.out, std::move(out), a->rows, a->cols);
      break;
    }
    case OpKind::kPool: {
      Buf* in = input(op.in);
      if (in == nullptr)
        return false;
      std::vector<float> out(in->cols, 0.0F);
      if (op.pool == PoolMode::kCls) {
        std::copy(in->v.begin(), in->v.begin() + in->cols, out.begin());
      } else if (op.pool == PoolMode::kLast) {
        std::copy(in->v.begin() + (in->rows - 1) * in->cols, in->v.end(),
                  out.begin());
      } else {
        for (std::size_t r = 0; r < in->rows; ++r) {
          for (std::size_t c = 0; c < in->cols; ++c) {
            out[c] += in->v[r * in->cols + c];
          }
        }
        for (auto& x : out)
          x /= static_cast<float>(in->rows);
      }
      set(op.out, std::move(out), 1, in->cols);
      break;
    }
    case OpKind::kL2Norm: {
      Buf* in = input(op.in);
      if (in == nullptr)
        return false;
      std::vector<float> out = in->v;
      for (std::size_t r = 0; r < in->rows; ++r) {
        double ss = 0.0;
        for (std::size_t c = 0; c < in->cols; ++c) {
          ss += double(out[r * in->cols + c]) * out[r * in->cols + c];
        }
        if (ss <= 0.0) {
          Fail(error, "graph: L2Norm of zero vector");
          return false;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(ss));
        for (std::size_t c = 0; c < in->cols; ++c)
          out[r * in->cols + c] *= inv;
      }
      set(op.out, std::move(out), in->rows, in->cols);
      break;
    }
    case OpKind::kRowDot: {
      Buf* in = input(op.in);
      if (in == nullptr)
        return false;
      const Tensor* w = weights_->Find(op.w);
      if (w->cols != in->cols) {
        Fail(error, "graph: RowDot width mismatch");
        return false;
      }
      const float* x = in->v.data() + (in->rows - 1) * in->cols;
      std::vector<float> out(op.ids.size());
      std::vector<float> row(w->cols);
      for (std::size_t j = 0; j < op.ids.size(); ++j) {
        if (op.ids[j] >= w->rows) {
          Fail(error, "graph: RowDot id out of range");
          return false;
        }
        w->ReadRow(op.ids[j], row.data());
        double acc = 0.0;
        for (std::size_t c = 0; c < w->cols; ++c)
          acc += double(x[c]) * row[c];
        out[j] = static_cast<float>(acc);
      }
      set(op.out, std::move(out), 1, op.ids.size());
      break;
    }
  }
  const double ms = MsSince(t0);
  if (is_linear) {
    st->linear_ms += ms;
  } else if (is_attn) {
    st->attention_ms += ms;
  } else {
    st->other_host_ms += ms;
  }
  return ok;
}

// ----------------------------------------------------------- shared engine --

namespace {
std::mutex g_engine_mu;
std::weak_ptr<NpuQuantGemmEngine> g_engine;
}  // namespace

std::shared_ptr<NpuQuantGemmEngine> SharedQuantEngine(std::string* error) {
  std::lock_guard<std::mutex> lock(g_engine_mu);
  if (auto e = g_engine.lock())
    return e;
  auto e = std::make_shared<NpuQuantGemmEngine>();
  if (!e->Init("", error))
    return nullptr;
  g_engine = e;
  return e;
}

void ReleaseSharedQuantEngine() {
  std::lock_guard<std::mutex> lock(g_engine_mu);
  g_engine.reset();
}

}  // namespace mbufo::core::npu::graph
