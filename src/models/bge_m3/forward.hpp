// bge-m3 CPU full-sequence encoder forward (leg-1 reference).
//
// Exact HF BertModel semantics in F32, single sequence, no padding (leg-1
// inputs are single prompts; there is no pad token in the sequence, so no
// mask is needed or applied):
//   embed  = LayerNorm(word + position + token_type[row 0])
//   layer  = LN2(x + Out(GELU-exact(Up(x)) ... )) with POST-norm BERT order:
//            a = AttnOutputNorm(x + AttnOut(softmax(QK'/sd)V))
//            x = LayerOutputNorm(a + FfnDown(GELU(Up(a))))
//   CLS pooling + L2 live in EngineSession::Sync (core::embedding).
//
// All GEMMs are plain F32 row-major over caller scratch; F16/BF16 storage is
// upcast once per EngineModel (UpcastAll), never per token. OpenMP parallels
// the row loops; determinism holds for a fixed thread count only insofar as
// floating-point summation order is concerned -- see the determinism test,
// which pins same-thread-count bit equality, and the HF cosine check, which
// is the real correctness bar.
#ifndef GUFO_MODELS_BGE_M3_FORWARD_HPP_
#define GUFO_MODELS_BGE_M3_FORWARD_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/models/bge_m3/config.hpp"
#include "src/models/bge_m3/weights.hpp"

namespace gufo::models::bge_m3 {

/// F32 mirror of the weight table (one exact upcast per tensor at model
/// load; CPU and HIP prefill both consume these).
struct BgeM3F32Weights {
  BgeM3Config config;
  std::vector<float> token_embd;  // [vocab, hidden]
  std::vector<float> position_embd;
  std::size_t position_rows{0};  // position_embd.size() / hidden
  std::vector<float> token_types;
  std::vector<float> emb_norm_w;
  std::vector<float> emb_norm_b;
  struct Layer {
    std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b;
    std::vector<float> out_w, out_b;
    std::vector<float> attn_norm_w, attn_norm_b;
    std::vector<float> up_w, up_b, down_w, down_b;
    std::vector<float> out_norm_w, out_norm_b;
  };
  std::vector<Layer> layers;
};

/// Exact single upcast of every tensor (F32 passes through by copy).
[[nodiscard]] BgeM3F32Weights UpcastAll(const BgeM3Weights& weights);

/// Caller-owned scratch arena for one forward of up to `max_seq` tokens.
struct BgeM3Scratch {
  static BgeM3Scratch Create(const BgeM3Config& config, std::size_t max_seq);
  std::size_t max_seq{0};
  std::vector<float> hidden;    // [max_seq, hidden]
  std::vector<float> normed;    // [max_seq, hidden]
  std::vector<float> qkv;       // [max_seq, 3 * hidden]
  std::vector<float> scores;    // [heads, max_seq, max_seq]
  std::vector<float> context;   // [max_seq, hidden]
  std::vector<float> proj;      // [max_seq, hidden]
  std::vector<float> ffn_inter;  // [max_seq, inter]
  std::vector<float> ffn_out;    // [max_seq, hidden]
};

/// Full encoder forward. `ids` must be non-empty with ids.size() <=
// scratch.max_seq. On success writes [n, hidden] last-layer states into
/// `hidden_out` (rows [0, n)) and returns true; shape violations return
/// false. `hidden_out` may alias scratch.hidden.
[[nodiscard]] bool CpuForward(const BgeM3F32Weights& weights,
                              std::span<const std::uint32_t> ids,
                              BgeM3Scratch& scratch,
                              std::span<float> hidden_out) noexcept;

/// Padded multi-sequence encoder forward (the ScoreBatch fast path: one
/// pass over N documents with the projection weights read once).
///
/// `batch` holds B non-empty id sequences; short rows are padded with
/// `pad_id` to Lmax (the longest row) and `hidden_out` receives the
/// [B, Lmax, hidden] last-layer states in row-major order. Attention applies
/// an exact key mask: positions j >= len_b read as -1e30 in every query row
/// (real and pad rows alike, so no all-masked row can form NaN), while real
/// entries add exactly 0.0. Real-query math therefore keeps the single
/// sequence summation order (real keys first, pad keys contribute +0.0) and
/// is bit-identical to CpuForward on the unpadded row; only pad-row states
/// are meaningless (callers must read row b*Lmax + i for i < len_b).
///
/// Returns false on an empty batch, any empty row, `pad_id` outside the
/// vocabulary, Lmax == 0, B * Lmax > scratch.max_seq (buffer capacity -- the
/// caller falls back to the sequential path), or a wrong-size `hidden_out`.
[[nodiscard]] bool CpuForwardBatch(
    const BgeM3F32Weights& weights,
    const std::vector<std::vector<std::uint32_t>>& batch,
    std::uint32_t pad_id, BgeM3Scratch& scratch,
    std::span<float> hidden_out) noexcept;

}  // namespace gufo::models::bge_m3

#endif  // GUFO_MODELS_BGE_M3_FORWARD_HPP_
