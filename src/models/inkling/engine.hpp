#ifndef MBUFO_MODELS_INKLING_ENGINE_HPP_
#define MBUFO_MODELS_INKLING_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/image.hpp"
#include "src/core/sampling.hpp"
#include "src/models/inkling/dspark.hpp"
#include "src/models/inkling/vision.hpp"
#include "src/models/inkling/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace mbufo::models::inkling {

struct ModelOptions {
  std::uint32_t max_context = 32768;
  std::string kv_cache_quant = "f16";
  // True when kv_cache_quant was explicitly supplied (CLI flag / /admin
  // field) rather than a mapped default. Lets the session tell "default
  // q8" apart from an explicit q4-family request that arrived as "q8".
  bool kv_cache_quant_supplied = false;
  std::string vision_model_path;
  std::string dspark_model_path;
};

class Session;

/// HIP runtime for `inkling2` GGUF models (Inkling-2-Mini: 128 experts top-8,
/// loop-core message passing, NVFP4 experts converted to Q8_0 once at load).
/// Forward covers the full v2 schedule (sliding + global layers, static and
/// dynamic sconv, per-token sinks, rel-bias gather, sigmoid top-8 MoE with the
/// fused situ activation, 3-pass loop with message gates) through shared core
/// GEMM/SoftCap/Argmax launchers; only the genuinely model-unique elementwise
/// kernels live in src/models/inkling/kernels/.
class Model final : public std::enable_shared_from_this<Model> {
 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const;
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] const std::string& KvCacheQuant() const noexcept {
    return options_.kv_cache_quant;
  }
  [[nodiscard]] bool KvCacheQuantSupplied() const noexcept {
    return options_.kv_cache_quant_supplied;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const ModelWeights& weights() const noexcept {
    return *weights_;
  }
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const std::string& ModelPath() const noexcept { return model_path_; }
  [[nodiscard]] std::optional<std::string_view> GgufChatTemplate() const {
    if (reader_ == nullptr) return std::nullopt;
    return reader_->GetMetadataString("tokenizer.chat_template");
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] bool HasVision() const noexcept { return vision_ != nullptr; }
  [[nodiscard]] bool HasDspark() const noexcept { return dspark_ != nullptr; }
  [[nodiscard]] std::uint32_t MaxDraftTokens() const noexcept {
    return dspark_ ? dspark_->MaxDraft() : 0;
  }
  [[nodiscard]] bool EncodeImage(const core::Image& image,
                                 std::vector<float>* out,
                                 std::string* error_msg) const {
    if (vision_ == nullptr) {
      if (error_msg != nullptr) *error_msg = "inkling: vision not loaded";
      return false;
    }
    return vision_->Encode(image, out, error_msg);
  }

  struct RegisteredRegion {
    const void* host_data{nullptr};
    void* device_data{nullptr};
    void* host_copy{nullptr};
    std::size_t size{0};
    // True when host_copy is a heap buffer owned elsewhere (converted_f16_):
    // hipHostUnregister on teardown, but never munmap.
    bool heap_backed{false};
  };

 private:
  Model() = default;

  ModelOptions options_;
  std::string model_path_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  // Load-time transposed F32 copies (rel_bias, stored [1024,16]-major
  // in GGUF, bit-exact vs the HF [16,1024] transpose); same heap-region
  // mechanism as converted_f16_.
  std::vector<std::vector<float>> converted_f32_;
  // Load-time converted F16 copies (Q8_0/BF16 dense sources);
  // TensorRef::data points into these buffers, which outlive the reader
  // mapping. (NVFP4 experts go to converted_q80_ instead, below.)
  std::vector<std::vector<std::uint16_t>> converted_f16_;
  // Load-time converted Q8_0 copies (NVFP4 experts, blk.0-19): dequant
  // NVFP4->F32 with per-expert scale, then core stock QuantizeQ8_0. The MoE
  // consume side is type-driven (QuantGemmPrefill/GemvBatched on
  // TensorRef::type), so these ride the same Q8 path as blk.20-23 experts.
  std::vector<std::vector<std::uint8_t>> converted_q80_;
  std::vector<RegisteredRegion> registered_regions_;
  std::unique_ptr<VisionEncoder> vision_;
  std::unique_ptr<DsparkModel> dspark_;
  std::vector<void*> device_allocations_;

  friend class Session;
};

class Session final {
 public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  struct ImageInjection {
    std::uint32_t offset{0};
    std::vector<float> embeds;
  };
  void SetImageInjections(std::vector<ImageInjection> imgs) {
    image_injections_ = std::move(imgs);
  }

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::uint32_t Length() const noexcept;
  [[nodiscard]] std::int32_t SampleGreedy();
  [[nodiscard]] bool SpeculativeRound(std::uint32_t n_draft,
                                      std::vector<std::int32_t>* out_tokens,
                                      std::string* error_msg = nullptr);
  /// Number of tokens the drafter proposed in the last speculative round
  /// (0 when the adaptive controller chose a plain decode step).
  [[nodiscard]] std::uint32_t LastDraftCount() const noexcept {
    return last_draft_n_;
  }
  /// Sampled (temperature / top-k / top-p / penalties) speculative round.
  /// The drafter proposes a deterministic block (q = one-hot), each proposal
  /// d is accepted with probability p_target(d) and a rejection draws the
  /// replacement from the residual (p without d), which is deferred into
  /// `sampler` so the next round starts from it: the emitted stream follows
  /// the target's sampling distribution exactly. `sampler` is the tentative
  /// working copy; it accepts every committed token of the round.
  [[nodiscard]] bool SpeculativeRoundSampled(
      std::uint32_t n_draft, sampling::SamplerState& sampler,
      std::vector<std::int32_t>* out_tokens, std::string* error_msg = nullptr);

 private:
  explicit Session(std::shared_ptr<Model> model);
  [[nodiscard]] bool Init(std::string* error_msg);
  // One prefill/decode chunk; layers 0-5 run once, loop-core layers 6-17 run
  // three passes with message fusion on passes 2/3, layers 18-23 run once.
  [[nodiscard]] bool ForwardChunk(std::span<const std::int32_t> tokens,
                                  std::uint32_t start_pos,
                                  std::string* error_msg,
                                  bool all_logits = false);
  [[nodiscard]] bool ForwardWithAllLogits(
      std::span<const std::int32_t> tokens, std::uint32_t start_pos,
      std::vector<std::int32_t>* out_argmax, std::string* error_msg);
  [[nodiscard]] bool RestoreSconvPrefix(std::uint32_t prefix_idx,
                                        std::string* error_msg);
  bool save_sconv_prefixes_{false};
  // Single physical layer on [n, hidden] hidden (in/out same buffer OK).
  // msg_q/k/v/m are per-token [n, hidden] loop messages for this layer
  // (nullptr on pass 1 / non-loop layers); snap_bank/snap_count track
  // boundary snapshots for res-attention.
  [[nodiscard]] bool ForwardLayer(std::uint32_t layer, float* hidden,
                                std::uint32_t n, std::uint32_t start_pos,
                                const float* msg_q, const float* msg_k,
                                const float* msg_v, const float* msg_m,
                                float* snap_bank, std::uint32_t snap_stride,
                                std::uint32_t* snap_count,
                                std::string* error_msg);
  // MoE block (shared by the FFN path): router + per-expert dense GEMMs +
  // situ activation + weighted scatter-add. Reads ffn_in [n, hidden], writes
  // moe_out [n, hidden].
  [[nodiscard]] bool ForwardMoe(const LayerWeights& lw, const float* ffn_in,
                                float* moe_out, std::uint32_t n,
                                std::uint32_t layer,
                                std::string* error_msg);
  // Gated stream fusion into `out` (loop layers, passes 2/3).
  [[nodiscard]] bool FuseHidden(const float* h, const float* msg,
                                const LoopTensors& lt, float* out,
                                std::uint32_t n, std::string* error_msg);
  // Zero the rolling sconv states and snapshot counters (chunk at position 0).
  void ResetStreamState();

  std::shared_ptr<Model> model_;
  hipStream_t stream_{nullptr};

  static constexpr std::uint32_t kChunk = 512;
  static constexpr std::uint32_t kMaxSnaps = 12;
  static constexpr std::uint32_t kTopK = 8;

  std::vector<float> host_logits_;
  std::vector<float> host_all_logits_;
  // Adaptive draft length (see ChooseDraftLen): decayed accepted-token and
  // rejection counts give the per-position acceptance MLE p = A / (A + R).
  double dspark_acc_sum_{2.0};
  double dspark_rej_sum_{2.0};
  std::uint32_t dspark_plain_streak_{0};
  std::uint32_t last_draft_n_{0};
  [[nodiscard]] std::uint32_t ChooseDraftLen(std::uint32_t max_n);
  void UpdateDraftStats(std::uint32_t n_draft, std::uint32_t accepted);
  std::vector<std::int32_t> history_;

  // Device arena (all sized by kChunk / context at Init).
  std::vector<void*> arena_;
  std::int32_t* d_tokens_{nullptr};
  std::uint32_t* d_pos_{nullptr};
  // (d_store_pos_ declared with the Q8 members above)
  hipGraph_t decode_graph_{nullptr};
  hipGraphExec_t decode_graph_exec_{nullptr};
  bool capture_by_pos_{false};
  float* d_hidden_{nullptr};     // [chunk, h]
  float* d_normed_{nullptr};     // [chunk, h]
  float* d_fuse_a_{nullptr};     // [chunk, h] (fused stream staging)
  float* d_fuse_b_{nullptr};     // [chunk, h]
  float* d_fuse_c_{nullptr};     // [chunk, h]
  float* d_q_{nullptr};          // [chunk, qdim]
  float* d_k_{nullptr};          // [chunk, kvdim]
  float* d_v_{nullptr};          // [chunk, kvdim]
  float* d_rel_{nullptr};        // [chunk, heads*16]
  float* d_sink_{nullptr};       // [chunk, heads]
  float* d_gate_{nullptr};       // [chunk, qdim] (o-gate logits)
  float* d_attn_out_{nullptr};
  float* d_attn_splitk_{nullptr};   // [chunk, qdim]
  float* d_proj_out_{nullptr};   // [chunk, h]
  float* d_conv_delta_{nullptr}; // [chunk, 32] (dynamic sconv weight proj)
  float* d_tau_{nullptr};        // [chunk] host-filled LogScale taus
  float* d_router_logits_{nullptr};  // [chunk, experts]
  std::int32_t* d_exp_ids_{nullptr};     // [chunk * topk]
  float* d_exp_weights_{nullptr};        // [chunk * topk]
  __half* d_x16_{nullptr};       // [chunk, max(h, qdim, fused)]
  __half* d_wdq_{nullptr};       // dequant scratch (max m*k)
  void* d_preq_{nullptr};        // GemvDecode prequant scratch
  std::size_t preq_bytes_{0};
  std::size_t wdq_elems_{0};
  // MoE per-expert staging (max one chunk of rows per expert).
  __half* d_expert_in16_{nullptr};  // [chunk, h]
  float* d_expert_gu_{nullptr};     // [chunk, fused]
  float* d_expert_act_{nullptr};    // [chunk, ff]
  __half* d_expert_act16_{nullptr}; // [chunk, ff]
  float* d_expert_down_{nullptr};   // [chunk, h]
  float* d_moe_out_{nullptr};       // [chunk, h]
  std::uint32_t* d_moe_counts_{nullptr};
  std::int32_t* d_moe_pad_bounds_{nullptr};
  std::int32_t* d_moe_cursors_{nullptr};
  std::int32_t* d_moe_rows_token_{nullptr};
  std::int32_t* d_moe_rows_slot_{nullptr};
  std::int32_t* d_moe_tiles_{nullptr};
  std::uint32_t* d_moe_n_tiles_{nullptr};
  std::size_t moe_tile_cap_{0};
  // Message path staging.
  float* d_pair_{nullptr};   // [chunk, 2h]
  float* d_g64_{nullptr};    // [chunk, 64]
  float* d_fg_{nullptr};     // [chunk, h]
  float* d_mg_{nullptr};     // [chunk, h]
  float* d_msg_{nullptr};    // [chunk, h] loop message: query stream
  float* d_kvmsg_{nullptr};  // [chunk, h] loop message: key stream
  float* d_msgv_{nullptr};   // [chunk, h] loop message: value stream
  float* d_msgm_{nullptr};   // [chunk, h] loop message: mlp stream
  float* d_prevbank_{nullptr};  // [chunk, maxsnaps, h] frozen prev-pass bank
  float* d_snaps_{nullptr};  // [chunk, maxsnaps, h] token-major snapshot bank
  float* d_h5_{nullptr};     // [chunk, h] layer-5 exit (loop reset point)
  float* d_hsave_{nullptr};  // [chunk, h] pass snapshot (prev-pass hidden)
  // Per-pass sconv slot selected by ForwardChunk (0/1/2 for passes 1/2/3).
  std::uint32_t sconv_slot_{0};
  // Per-layer KV planes (F16, or blocked-Q8 rows when kv_q8_) and sconv
  // rolling states (F32).
  std::vector<void*> k_cache_;  // per layer: F16 [rows, kvdim] half, Q8 [rows, row_stride_] bytes
  std::vector<void*> v_cache_;
  std::vector<void*> sconv_k_;  // per layer: [pass_slots, kvdim*3] float
  std::vector<void*> sconv_v_;
  std::vector<void*> sconv_pa_;  // post-attn: [pass_slots, h*3] float
  std::vector<void*> sconv_pm_;  // post-mlp:  [pass_slots, h*3] float
  std::vector<void*> kv_backup_k_;  // loop layers: [chunk, kvdim] half (Q8: [chunk, row_stride_] bytes)
  std::vector<void*> kv_backup_v_;
  // Blocked KV modes: F16 (all-half planes), Q8 (34B blocks both sides)
  // or K8V4 (34B keys, 18B packed-nibble values). The layout is the shared
  // blocked_kv format (see blocked_kv_codec.hpp); keys always stay 8-bit
  // (post-RoPE outliers need the range), values may pack to 4-bit.
  // Attention fused-dequantizes on read through the trait codec.
  enum class KvMode : std::uint8_t { kF16, kQ8, kK8V4 };
  KvMode kv_mode_ = KvMode::kF16;
  std::uint32_t kv_row_stride_k_ = 0;  // per-token K row bytes
  std::uint32_t kv_row_stride_v_ = 0;  // per-token V row bytes
  std::uint32_t* d_store_pos_{nullptr};  // blocked store slot scalar (non-graph)
  // Logits / sampling.
  float* d_logits_{nullptr};
  float* d_all_logits_{nullptr};
  std::int32_t* d_argmax_{nullptr};
  std::int32_t* d_all_argmax_{nullptr};
  void* d_argmax_scratch_{nullptr};
  std::uint32_t vocab_rows_{0};
  std::vector<ImageInjection> image_injections_;
  // DSpark speculative decoding state.
  std::unique_ptr<DsparkSession> dspark_;
  float dspark_adapt_ema_{0.50F};
  std::uint32_t dspark_adapt_cooldown_{0U};
  float* d_dspark_h5_{nullptr};
  float* d_dspark_h23_{nullptr};
  float* d_dspark_h41_{nullptr};
  float* d_dspark_h48_{nullptr};
  std::vector<void*> sconv_bk_k_;
  std::vector<void*> sconv_bk_v_;
  std::vector<void*> sconv_bk_pa_;
  std::vector<void*> sconv_bk_pm_;

  friend class Model;
};

}  // namespace mbufo::models::inkling

#endif  // MBUFO_MODELS_INKLING_ENGINE_HPP_
