#include "src/core/compute/npu/npu_graph_models.hpp"

#include <cmath>
#include <limits>
#include <utility>

#include "src/core/rerank_score.hpp"

namespace mbufo::core::npu {
namespace {

using graph::Tensor;

// Qwen3-Reranker chat template (byte-exact with qwen_reranker/engine.cpp).
constexpr std::string_view kSystemFrameOpen = "<|im_start|>system\n";
constexpr std::string_view kSystemText =
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be "
    "\"yes\" or \"no\".";
constexpr std::string_view kUserFrameOpen =
    "<|im_end|>\n<|im_start|>user\n<Instruct>: ";
constexpr std::string_view kQueryLabel = "\n<Query>: ";
constexpr std::string_view kDocumentLabel = "\n<Document>: ";
constexpr std::string_view kAssistantFrameOpen =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
constexpr std::string_view kDefaultInstruction =
    "Given a web search query, retrieve relevant passages that answer the "
    "query";
constexpr std::size_t kMaxTokens = 4096;

std::string LayerName(std::size_t i, const char* s) {
  return "L" + std::to_string(i) + "." + s;
}

bool SetErr(std::string* e, const std::string& m) {
  if (e != nullptr) *e = m;
  return false;
}

}  // namespace

// ------------------------------------------------------------------- BERT --

bool NpuGraphBgeRuntime::Init(const models::bge_m3::BgeM3F32Weights& w,
                              graph::LinearBackend backend,
                              std::string* error) {
  const auto& cfg = w.config;
  if (cfg.hidden_size != 1024 || cfg.num_layers == 0 ||
      w.layers.size() != cfg.num_layers) {
    return SetErr(error, "NpuGraphBgeRuntime: unsupported geometry");
  }
  const std::size_t H = cfg.hidden_size;
  const std::size_t F = cfg.intermediate_size;
  store_ = graph::WeightStore{};
  store_.Put("tok", Tensor::ViewF32(w.token_embd.data(), cfg.vocab_size, H));
  store_.Put("pos", Tensor::ViewF32(w.position_embd.data(), w.position_rows, H));
  store_.Put("type", Tensor::ViewF32(w.token_types.data(), 1, H));
  store_.Put("emb_ln.w", Tensor::ViewF32(w.emb_norm_w.data(), 1, H));
  store_.Put("emb_ln.b", Tensor::ViewF32(w.emb_norm_b.data(), 1, H));
  for (std::size_t i = 0; i < w.layers.size(); ++i) {
    const auto& l = w.layers[i];
    auto put = [&](const char* n, const std::vector<float>& v, std::size_t r,
                   std::size_t c) {
      store_.Put(LayerName(i, n), Tensor::ViewF32(v.data(), r, c));
    };
    put("q.w", l.q_w, H, H);
    put("q.b", l.q_b, 1, H);
    put("k.w", l.k_w, H, H);
    put("k.b", l.k_b, 1, H);
    put("v.w", l.v_w, H, H);
    put("v.b", l.v_b, 1, H);
    put("o.w", l.out_w, H, H);
    put("o.b", l.out_b, 1, H);
    put("attn_ln.w", l.attn_norm_w, 1, H);
    put("attn_ln.b", l.attn_norm_b, 1, H);
    put("up.w", l.up_w, F, H);
    put("up.b", l.up_b, 1, F);
    put("down.w", l.down_w, H, F);
    put("down.b", l.down_b, 1, H);
    put("out_ln.w", l.out_norm_w, 1, H);
    put("out_ln.b", l.out_norm_b, 1, H);
  }
  graph::BertGraphConfig gc;
  gc.layers = cfg.num_layers;
  gc.hidden = H;
  gc.heads = cfg.num_attention_heads;
  gc.inter = F;
  gc.eps = cfg.layer_norm_eps;
  std::shared_ptr<NpuQuantGemmEngine> engine;
  if (backend == graph::LinearBackend::kNpuQ8) {
    engine = graph::SharedQuantEngine(error);
    if (engine == nullptr) return false;
  }
  return exec_.Prepare(graph::BuildBertClsGraph(gc), &store_, std::move(engine),
                       backend, error);
}

bool NpuGraphBgeRuntime::ForwardCls(std::span<const std::uint32_t> ids,
                                    std::vector<float>* cls_out,
                                    double* prefill_ms, std::string* error) {
  if (ids.empty() || ids.size() > 2048) {
    return SetErr(error, "NpuGraphBgeRuntime: invalid input size");
  }
  if (!exec_.Run(ids, cls_out, &stats_, error)) return false;
  if (prefill_ms != nullptr) *prefill_ms = stats_.total_ms;
  return true;
}

// ------------------------------------------------------------ EmbGemma 2 --

bool NpuGraphEmbGemma2::Init(const core::GgufReader& r,
                             const models::embeddinggemma2::Config& cfg,
                             graph::LinearBackend backend, std::string* error) {
  if (cfg.layers == 0 || cfg.layer_is_full.size() != cfg.layers ||
      cfg.hidden == 0 || cfg.ffn == 0 || cfg.ple_dim == 0 || cfg.out_dim == 0) {
    return SetErr(error, "NpuGraphEmbGemma2: unsupported geometry");
  }
  store_ = graph::WeightStore{};
  // Binds `slice` (row block) of `gguf` as [rows, cols]; the tensor holds
  // `slices * rows * cols` elements.
  auto bind = [&](const std::string& name, const std::string& gguf,
                  std::size_t rows, std::size_t cols, std::size_t slice,
                  std::size_t slices) -> bool {
    const auto* t = r.FindTensor(gguf);
    if (t == nullptr || t->ElementCount() != slices * rows * cols) {
      return SetErr(error, "missing or mis-shaped tensor: " + gguf);
    }
    graph::TensorKind kind;
    std::size_t esz;
    switch (t->type) {
      case core::GgmlType::kF32:
        kind = graph::TensorKind::kF32;
        esz = 4;
        break;
      case core::GgmlType::kF16:
        kind = graph::TensorKind::kF16;
        esz = 2;
        break;
      case core::GgmlType::kBF16:
        kind = graph::TensorKind::kBf16;
        esz = 2;
        break;
      default:
        return SetErr(error, "unsupported tensor type for " + gguf);
    }
    store_.Put(name, Tensor::View(kind,
                                  static_cast<const std::uint8_t*>(t->data) +
                                      slice * rows * cols * esz,
                                  rows, cols));
    return true;
  };
  const std::size_t H = cfg.hidden, F = cfg.ffn, P = cfg.ple_dim;
  if (!bind("tok", "embed_tokens.weight",
            r.FindTensor("embed_tokens.weight") != nullptr
                ? r.FindTensor("embed_tokens.weight")->ElementCount() / H
                : 0,
            H, 0, 1) ||
      !bind("final_norm.w", "norm.weight", 1, H, 0, 1) ||
      !bind("ple_norm.w", "ple.per_layer_projection_norm.weight", 1, P, 0, 1) ||
      !bind("out_proj.w", "embedding_projection.weight", cfg.out_dim, H, 0,
            1)) {
    return false;
  }
  graph::EmbGemma2GraphConfig gc;
  gc.layers = cfg.layers;
  gc.hidden = H;
  gc.ffn = F;
  gc.heads = cfg.heads;
  gc.head_dim_local = cfg.head_dim_local;
  gc.head_dim_full = cfg.head_dim_full;
  gc.kv_local = cfg.kv_local;
  gc.kv_full = cfg.kv_full;
  gc.sliding_window = cfg.sliding_window;
  gc.ple_dim = P;
  gc.eps = cfg.rms_eps;
  gc.rope_local = cfg.rope_local;
  gc.rope_full = cfg.rope_full;
  gc.embed_scale = cfg.embed_scale;
  gc.layer_is_full = cfg.layer_is_full;
  for (std::size_t i = 0; i < cfg.layers; ++i) {
    const bool full = cfg.layer_is_full[i] != 0;
    const std::size_t d = full ? cfg.head_dim_full : cfg.head_dim_local;
    const std::size_t kvh = full ? cfg.kv_full : cfg.kv_local;
    const std::size_t qd = cfg.heads * d, kd = kvh * d;
    const std::string p = "layers." + std::to_string(i) + ".";
    auto n = [&](const char* s) { return LayerName(i, s); };
    if (!bind(n("in_norm.w"), p + "input_layernorm.weight", 1, H, 0, 1) ||
        !bind(n("post_attn.w"), p + "post_attention_layernorm.weight", 1, H, 0,
              1) ||
        !bind(n("pre_ffn.w"), p + "pre_feedforward_layernorm.weight", 1, H, 0,
              1) ||
        !bind(n("post_ffn.w"), p + "post_feedforward_layernorm.weight", 1, H, 0,
              1) ||
        !bind(n("q_norm.w"), p + "self_attn.q_norm.weight", 1, d, 0, 1) ||
        !bind(n("k_norm.w"), p + "self_attn.k_norm.weight", 1, d, 0, 1) ||
        !bind(n("q.w"), p + "self_attn.q_proj.weight", qd, H, 0, 1) ||
        !bind(n("k.w"), p + "self_attn.k_proj.weight", kd, H, 0, 1) ||
        !bind(n("v.w"), p + "self_attn.v_proj.weight", kd, H, 0, 1) ||
        !bind(n("o.w"), p + "self_attn.o_proj.weight", H, qd, 0, 1) ||
        !bind(n("gate.w"), p + "mlp.gate_proj.weight", F, H, 0, 1) ||
        !bind(n("up.w"), p + "mlp.up_proj.weight", F, H, 0, 1) ||
        !bind(n("down.w"), p + "mlp.down_proj.weight", H, F, 0, 1) ||
        !bind(n("ple_proj.w"), "ple.per_layer_model_projection.weight", P, H, i,
              cfg.layers) ||
        !bind(n("ple_gate.w"), p + "ple_block.per_layer_input_gate.weight", P,
              H, 0, 1) ||
        !bind(n("ple_out.w"), p + "ple_block.per_layer_projection.weight", H, P,
              0, 1) ||
        !bind(n("ple_post.w"), p + "ple_block.post_per_layer_input_norm.weight",
              1, H, 0, 1)) {
      return false;
    }
    const auto* sc = r.FindTensor(p + "layer_scalar");
    if (sc == nullptr || sc->type != core::GgmlType::kF32 ||
        sc->ElementCount() != 1) {
      return SetErr(error,
                    "missing layer_scalar at layer " + std::to_string(i));
    }
    gc.layer_scalar.push_back(*static_cast<const float*>(sc->data));
  }
  std::shared_ptr<NpuQuantGemmEngine> engine;
  if (backend == graph::LinearBackend::kNpuQ8) {
    engine = graph::SharedQuantEngine(error);
    if (engine == nullptr)
      return false;
  }
  // ~1 GB of Q8 weights across 24 layers exhausts the NPU BO pool when kept
  // resident (hrx_buffer_allocate failed); stream blocks like the dedicated
  // runtime does. MBUFO_NPU_GRAPH_RESIDENT=1 forces resident BOs.
  exec_.set_resident_weights(std::getenv("MBUFO_NPU_GRAPH_RESIDENT") !=
                             nullptr);
  return exec_.Prepare(graph::BuildEmbGemma2Graph(gc), &store_,
                       std::move(engine), backend, error);
}

bool NpuGraphEmbGemma2::Forward(std::span<const std::uint32_t> ids,
                                std::vector<float>* out, double* ms,
                                std::string* error) {
  if (ids.empty() || ids.size() > kMaxTokens) {
    return SetErr(error, "NpuGraphEmbGemma2: input must be 1.." +
                             std::to_string(kMaxTokens) + " tokens");
  }
  if (!exec_.Run(ids, out, &stats_, error))
    return false;
  if (ms != nullptr)
    *ms = stats_.total_ms;
  return true;
}

// ------------------------------------------------------------------ Qwen3 --

std::unique_ptr<NpuQwen3Encoder> NpuQwen3Encoder::Load(
    const std::string& path, Role role, graph::LinearBackend backend,
    std::string* error) {
  std::unique_ptr<NpuQwen3Encoder> m(new NpuQwen3Encoder());
  m->role_ = role;
  std::string e;
  m->reader_ = core::GgufReader::OpenFile(path, &e);
  if (m->reader_ == nullptr) {
    SetErr(error, "open GGUF failed: " + e);
    return nullptr;
  }
  const auto& r = *m->reader_;
  if (r.GetMetadataString("general.architecture").value_or("") != "qwen3") {
    SetErr(error, "NpuQwen3Encoder: architecture must be qwen3");
    return nullptr;
  }
  auto u32 = [&](const char* k, std::size_t def) -> std::size_t {
    return r.GetMetadataUint32(k).value_or(static_cast<std::uint32_t>(def));
  };
  auto& c = m->cfg_;
  c.layers = u32("qwen3.block_count", 0);
  c.hidden = u32("qwen3.embedding_length", 0);
  c.heads = u32("qwen3.attention.head_count", 0);
  c.kv_heads = u32("qwen3.attention.head_count_kv", 0);
  c.head_dim = u32("qwen3.attention.key_length", c.heads ? c.hidden / c.heads : 0);
  c.inter = u32("qwen3.feed_forward_length", 0);
  c.eps = r.GetMetadataFloat32("qwen3.attention.layer_norm_rms_epsilon")
              .value_or(1e-6F);
  c.rope_theta = r.GetMetadataFloat32("qwen3.rope.freq_base").value_or(1e6F);
  if (c.layers == 0 || c.hidden == 0 || c.heads == 0 || c.kv_heads == 0 ||
      c.head_dim == 0 || c.inter == 0) {
    SetErr(error, "NpuQwen3Encoder: incomplete qwen3 metadata");
    return nullptr;
  }

  auto bind = [&](const std::string& name, const std::string& gguf,
                  std::size_t want_cols) -> bool {
    const auto* t = r.FindTensor(gguf);
    if (t == nullptr) return SetErr(error, "missing tensor " + gguf);
    graph::TensorKind kind;
    switch (t->type) {
      case core::GgmlType::kF32: kind = graph::TensorKind::kF32; break;
      case core::GgmlType::kF16: kind = graph::TensorKind::kF16; break;
      case core::GgmlType::kBF16: kind = graph::TensorKind::kBf16; break;
      case core::GgmlType::kQ8_0: kind = graph::TensorKind::kQ8_0; break;
      default:
        return SetErr(error, "unsupported tensor type for " + gguf);
    }
    const std::size_t cols = t->dimensions.empty() ? 0 : t->dimensions[0];
    const std::size_t rows = t->dimensions.size() > 1 ? t->dimensions[1] : 1;
    if (cols == 0 || (want_cols != 0 && cols != want_cols)) {
      return SetErr(error, "unexpected shape for " + gguf);
    }
    m->store_.Put(name, Tensor::View(kind, t->data, rows, cols));
    return true;
  };
  if (!bind("tok", "token_embd.weight", c.hidden) ||
      !bind("out_norm.w", "output_norm.weight", c.hidden)) {
    return nullptr;
  }
  for (std::size_t i = 0; i < c.layers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    if (!bind(LayerName(i, "attn_norm.w"), p + "attn_norm.weight", c.hidden) ||
        !bind(LayerName(i, "q.w"), p + "attn_q.weight", c.hidden) ||
        !bind(LayerName(i, "k.w"), p + "attn_k.weight", c.hidden) ||
        !bind(LayerName(i, "v.w"), p + "attn_v.weight", c.hidden) ||
        !bind(LayerName(i, "o.w"), p + "attn_output.weight",
              c.heads * c.head_dim) ||
        !bind(LayerName(i, "q_norm.w"), p + "attn_q_norm.weight", c.head_dim) ||
        !bind(LayerName(i, "k_norm.w"), p + "attn_k_norm.weight", c.head_dim) ||
        !bind(LayerName(i, "ffn_norm.w"), p + "ffn_norm.weight", c.hidden) ||
        !bind(LayerName(i, "gate.w"), p + "ffn_gate.weight", c.hidden) ||
        !bind(LayerName(i, "up.w"), p + "ffn_up.weight", c.hidden) ||
        !bind(LayerName(i, "down.w"), p + "ffn_down.weight", c.inter)) {
      return nullptr;
    }
  }

  m->tokenizer_ = tokenization::QwenTokenizer::CreateFromGguf(r, &e);
  if (m->tokenizer_ == nullptr) {
    SetErr(error, "tokenizer load failed: " + e);
    return nullptr;
  }
  graph::Graph g;
  if (role == Role::kEmbedding) {
    g = graph::BuildQwen3EmbeddingGraph(c);
  } else {
    const auto yes = m->tokenizer_->Encode("yes");
    const auto no = m->tokenizer_->Encode("no");
    if (yes.size() != 1 || no.size() != 1) {
      SetErr(error, "yes/no do not encode to one token");
      return nullptr;
    }
    m->yes_id_ = yes.front();
    m->no_id_ = no.front();
    g = graph::BuildQwen3RerankGraph(c, {m->no_id_, m->yes_id_});
  }
  std::shared_ptr<NpuQuantGemmEngine> engine;
  if (backend == graph::LinearBackend::kNpuQ8) {
    engine = graph::SharedQuantEngine(error);
    if (engine == nullptr) return nullptr;
  }
  if (!m->exec_.Prepare(std::move(g), &m->store_, std::move(engine), backend,
                        error)) {
    return nullptr;
  }
  return m;
}

bool NpuQwen3Encoder::EmbedIds(std::span<const std::uint32_t> ids,
                               std::vector<float>* out, double* ms,
                               std::string* error) {
  if (role_ != Role::kEmbedding) {
    return SetErr(error, "NpuQwen3Encoder: not an embedding model");
  }
  if (ids.empty() || ids.size() > kMaxTokens) {
    return SetErr(error, "NpuQwen3Encoder: input must be 1.." +
                             std::to_string(kMaxTokens) + " tokens");
  }
  if (!exec_.Run(ids, out, &stats_, error)) return false;
  if (ms != nullptr) *ms = stats_.total_ms;
  return true;
}

bool NpuQwen3Encoder::Embed(std::string_view text, std::vector<float>* out,
                            std::size_t* tokens, double* ms,
                            std::string* error) {
  auto ids = tokenizer_->Encode(text);
  if (ids.empty()) return SetErr(error, "input tokenizes to zero tokens");
  const auto eos = tokenizer_->GetEosTokenId();
  if (eos == tokenization::kInvalidTokenId) {
    return SetErr(error, "model has no EOS token");
  }
  if (ids.back() != eos) ids.push_back(eos);
  if (tokens != nullptr) *tokens = ids.size();
  return EmbedIds(ids, out, ms, error);
}

bool NpuQwen3Encoder::ScoreIds(std::span<const std::uint32_t> ids,
                               float* score, float* logit_yes, float* logit_no,
                               double* ms, std::string* error) {
  if (role_ != Role::kReranker) {
    return SetErr(error, "NpuQwen3Encoder: not a reranker");
  }
  if (ids.empty() || ids.size() > kMaxTokens) {
    return SetErr(error, "NpuQwen3Encoder: pair must be 1.." +
                             std::to_string(kMaxTokens) + " tokens");
  }
  std::vector<float> lg;
  if (!exec_.Run(ids, &lg, &stats_, error)) return false;
  // probe order = {no, yes}
  const float s = core::rerank::YesNoScore(lg[1], lg[0]);
  if (!std::isfinite(s)) return SetErr(error, "non-finite yes/no readout");
  if (score != nullptr) *score = s;
  if (logit_no != nullptr) *logit_no = lg[0];
  if (logit_yes != nullptr) *logit_yes = lg[1];
  if (ms != nullptr) *ms = stats_.total_ms;
  return true;
}

bool NpuQwen3Encoder::ScorePair(std::string_view query,
                                std::string_view document,
                                std::string_view instruction, float* score,
                                std::size_t* tokens, double* ms,
                                std::string* error) {
  if (query.empty() || document.empty()) {
    return SetErr(error, "rerank pair needs a non-empty query and document");
  }
  std::string prompt;
  prompt += kSystemFrameOpen;
  prompt += kSystemText;
  prompt += kUserFrameOpen;
  prompt += instruction.empty() ? kDefaultInstruction : instruction;
  prompt += kQueryLabel;
  prompt += query;
  prompt += kDocumentLabel;
  prompt += document;
  prompt += kAssistantFrameOpen;
  const auto ids = tokenizer_->Encode(prompt);
  if (tokens != nullptr) *tokens = ids.size();
  return ScoreIds(ids, score, nullptr, nullptr, ms, error);
}

}  // namespace mbufo::core::npu
