// bge-m3 OOP adapter: EngineModel (weights + tokenizer + F32 mirror),
// EngineSession (single-sequence Sync: full prefill + CLS + L2), and the
// model-kind registry.
//
// Embedding semantics, fixed for this model kind:
//   Sync(ids) = full bidirectional prefill over the whole sequence, CLS row
//   (row 0) pooled via the shared core::embedding pipeline, L2-normalized.
//   There is no decode, no KV cache, no speculation, and no lm_head --
//   encoder-only artifacts never generate.
// Registry: model kinds self-register by name; serve and drivers resolve
// "bge-m3" (the XLM-R encoder kind) through ModelRegistry::Create.
#ifndef GUFO_MODELS_BGE_M3_ENGINE_HPP_
#define GUFO_MODELS_BGE_M3_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/bge_m3/config.hpp"
#include "src/models/bge_m3/forward.hpp"
#include "src/models/bge_m3/tokenizer.hpp"
#include "src/models/bge_m3/weights.hpp"

namespace gufo::models::bge_m3 {

#ifdef ENGINE_ENABLE_HIP
/// Raw CLS hidden row for rerank heads (hip/session.hip): the same
/// full-sequence encoder prefill as HipSync, but the row-0 hidden
/// states are returned unpooled and unnormalized for the caller's
/// classification head. Same failure contract as HipSync.
struct ClsHiddenResult {
  std::vector<float> cls;  // row 0, NOT normalized
  double prefill_ms{0.0};
};

/// Padded multi-sequence twin of ClsHiddenResult (hip/session.hip): one
/// encoder forward over the [B, Lmax] padded batch with an exact key mask,
/// returning the B unpooled CLS rows in input order plus the single timed
/// body milliseconds. Covers F16 and K-quant weights through the same
/// ProjGemm dispatch as the single-sequence path (no model-specific kernel).
/// Failure is an explicit nullopt, never a silent sequential fallback --
/// the caller decides between fallback (capacity gate) and refusal.
struct ClsHiddenBatchResult {
  std::vector<std::vector<float>> cls;  // B rows, each hidden_size
  double prefill_ms{0.0};
};
#endif

class EngineModel;

/// One embedding session: caller-owned scratch, synchronous full-sequence
/// Sync. Sessions are not thread-safe; models are freely shared.
class EngineSession {
 public:
  explicit EngineSession(const EngineModel* model);
  ~EngineSession();
  EngineSession(const EngineSession&) = delete;
  EngineSession& operator=(const EngineSession&) = delete;

  struct SyncResult {
    std::vector<float> values;  // L2-normalized, dim == hidden_size
    double pre_norm{0.0};       // norm before normalization (0 == refused)
    double prefill_ms{0.0};
  };

  /// Full prefill + CLS pooling + L2. Returns nullopt (with *error_msg set)
  /// when ids is empty/over-long (leg-1 window) or holds out-of-vocab ids.
  /// `use_hip` requests the device path; without ENGINE_ENABLE_HIP (or when
  /// device init failed) the call runs the CPU reference instead.
  [[nodiscard]] std::optional<SyncResult> Sync(
      std::span<const std::uint32_t> ids, bool use_hip,
      std::string* error_msg = nullptr);

 private:
  const EngineModel* model_;
  BgeM3Scratch scratch_;
};

/// Immutable, shareable bge-m3 model: GGUF storage plus the F32 compute
/// mirror plus the Unigram tokenizer. The GgufReader is owned here so mapped
/// storage outlives every session.
class EngineModel {
 public:
  EngineModel(const EngineModel&) = delete;
  EngineModel& operator=(const EngineModel&) = delete;
  ~EngineModel();

  /// Loads a `bert` GGUF artifact (weights + tokenizer + F32 upcast).
  /// Returns nullptr with *error_msg set on any refusal.
  [[nodiscard]] static std::unique_ptr<EngineModel> Create(
      const std::string& model_path, std::string* error_msg = nullptr,
      BgeM3ConfigOptions options = {});

  [[nodiscard]] const BgeM3Config& Config() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] const SpmUnigramTokenizer& Tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const BgeM3F32Weights& F32() const noexcept { return f32_; }
  [[nodiscard]] const core::GgufReader& GetReader() const noexcept {
    return *reader_;
  }
  [[nodiscard]] const BgeM3Weights& Weights() const noexcept {
    return weights_;
  }

 private:
  EngineModel();
  std::unique_ptr<core::GgufReader> reader_;
  BgeM3Weights weights_;
  BgeM3F32Weights f32_;
  std::unique_ptr<SpmUnigramTokenizer> tokenizer_;
#ifdef ENGINE_ENABLE_HIP
  struct HipState;
  std::unique_ptr<HipState> hip_;
  // Shared encoder prefill for HipSync and HipClsHidden (defined in
  // hip/session.hip).
  [[nodiscard]] static bool PrefillToHidden(EngineModel& model,
                                            std::span<const std::uint32_t> ids,
                                            double* ms_out,
                                            std::string* error_msg);
  [[nodiscard]] static bool RunEncoderBody(
      HipState& st, const BgeM3Config& config, int nn, int hh, int hdd,
      int hdim, int ninter, std::string& body_error);
  // Padded multi-sequence prefill for HipClsHiddenBatch: B rows of Lmax
  // tokens with the per-row lengths on the device (device state owns the
  // lens buffer). Same failure contract as PrefillToHidden.
  [[nodiscard]] static bool PrefillBatchToHidden(
      EngineModel& model,
      const std::vector<std::vector<std::uint32_t>>& batch,
      std::uint32_t pad_id, double* ms_out, std::string* error_msg);
  [[nodiscard]] static bool RunEncoderBatchBody(
      HipState& st, const BgeM3Config& config, int bb, int ll, int hh, int hdd,
      int hdim, int ninter, std::string& body_error);
#endif

  friend class EngineSession;
#ifdef ENGINE_ENABLE_HIP
  // Device prefill (hip/session.hip) touches hip_ directly; friendship
  // keeps the members private to every other TU.
  friend void InitHipState(EngineModel& model);
  friend std::optional<EngineSession::SyncResult> HipSync(
      const EngineModel& model, std::span<const std::uint32_t> ids,
      std::string* error_msg);
  friend std::optional<ClsHiddenResult> HipClsHidden(
      const EngineModel& model, std::span<const std::uint32_t> ids,
      std::string* error_msg);
  friend std::optional<ClsHiddenBatchResult> HipClsHiddenBatch(
      const EngineModel& model,
      const std::vector<std::vector<std::uint32_t>>& batch,
      std::uint32_t pad_id, std::string* error_msg);
#endif
};

/// Model-kind registry: name -> factory. bge_m3 registers "bge-m3" via a
/// function-local static initializer (see engine.cpp).
class ModelRegistry {
 public:
  using Factory = std::function<std::unique_ptr<EngineModel>(
      const std::string& model_path, std::string* error_msg)>;

  static ModelRegistry& Instance();
  /// Returns false when `name` is already registered (first wins).
  bool Register(const std::string& name, Factory factory);
  [[nodiscard]] std::unique_ptr<EngineModel> Create(
      const std::string& name, const std::string& model_path,
      std::string* error_msg = nullptr) const;

 private:
  ModelRegistry() = default;
  mutable std::mutex mutex_;
  std::map<std::string, Factory> factories_;
};

#ifdef ENGINE_ENABLE_HIP
[[nodiscard]] std::optional<ClsHiddenResult> HipClsHidden(
    const EngineModel& model, std::span<const std::uint32_t> ids,
    std::string* error_msg = nullptr);

[[nodiscard]] std::optional<ClsHiddenBatchResult> HipClsHiddenBatch(
    const EngineModel& model,
    const std::vector<std::vector<std::uint32_t>>& batch,
    std::uint32_t pad_id, std::string* error_msg = nullptr);

/// Device prefill twin of the CPU Sync path (hip/session.hip): the same
/// full-sequence encoder math via hipBLASLt F16 projections (F32 accumulate)
/// plus F32 batched attention and elementwise kernels, returning the
/// pooled+L2 vector. Failure is an explicit nullopt, never a silent CPU
/// fallback (see EngineSession::Sync).
[[nodiscard]] std::optional<EngineSession::SyncResult> HipSync(
    const EngineModel& model, std::span<const std::uint32_t> ids,
    std::string* error_msg = nullptr);
#endif

}  // namespace gufo::models::bge_m3

#endif  // GUFO_MODELS_BGE_M3_ENGINE_HPP_
