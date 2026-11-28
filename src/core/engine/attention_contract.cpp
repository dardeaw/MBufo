#include "src/core/engine/attention_contract.hpp"

#include <string>

namespace mbufo::core::engine {

const char* AttentionKindName(AttentionKind kind) noexcept {
  switch (kind) {
    case AttentionKind::kSoftmaxGqa: return "softmax-gqa";
    case AttentionKind::kMla: return "mla";
    case AttentionKind::kLinearRecurrent: return "linear-recurrent";
  }
  return "?";
}

const char* AttentionKernelOriginName(AttentionKernelOrigin origin) noexcept {
  return origin == AttentionKernelOrigin::kCore ? "core" : "model-private";
}

const char* AttentionKvFormatName(AttentionKvFormat format) noexcept {
  switch (format) {
    case AttentionKvFormat::kF16: return "f16";
    case AttentionKvFormat::kBlockedQ8: return "blocked-q8";
    case AttentionKvFormat::kEither: return "f16|blocked-q8";
    case AttentionKvFormat::kModelSpecific: return "model-specific";
  }
  return "?";
}

namespace {

void CheckPath(const char* which, const AttentionPathInfo& p, bool core_ok, bool need_row_invariant, AttentionContractReport* r) {
  const std::string w = which;
  if (p.kernel.empty()) r->problems.push_back(w + " attention kernel is not described");
  if (p.origin == AttentionKernelOrigin::kModelPrivate && p.private_reason.empty())
    r->problems.push_back(w + " attention uses private kernels but gives no private_reason");
  if (p.origin == AttentionKernelOrigin::kCore) {
    if (!core_ok)
      r->problems.push_back(w + " attention claims core kernels but the geometry has non-softmax or core-unsupported layers");
    if (need_row_invariant && !p.row_invariant)
      r->problems.push_back(w + " attention claims core kernels but is not row invariant (core kernels guarantee it)");
  }
}

}  // namespace

AttentionContractReport ValidateAttentionContract(const AttentionGeometry& g, const AttentionKernelInfo& k) {  AttentionContractReport r;
  if (g.groups.empty()) {
    r.problems.push_back("no attention layers declared (engines without attention register with attention = false)");
    return r;
  }
  bool core_ok = false;  // a core claim needs softmax layers the core kernels can serve
  bool has_softmax = false, softmax_specific = false;
  for (const auto& a : g.groups) {
    const std::string tag = std::string(AttentionKindName(a.kind)) + " group";
    if (a.layers == 0) r.problems.push_back(tag + " has zero layers");
    if (a.kind == AttentionKind::kSoftmaxGqa) {
      has_softmax = true;
      if (a.heads == 0 || a.head_dim == 0) r.problems.push_back(tag + " has zero heads or head_dim");
      if (a.kv_heads == 0 || (a.heads != 0 && a.heads % a.kv_heads != 0))
        r.problems.push_back(tag + ": heads must be a positive multiple of kv_heads");
      const bool q8 = a.kv_format == AttentionKvFormat::kBlockedQ8 || a.kv_format == AttentionKvFormat::kEither;
      if (q8 && a.head_dim % 32 != 0)
        r.problems.push_back(tag + ": blocked-Q8 KV needs head_dim % 32 == 0 (declare kF16 or kModelSpecific)");
      if (a.kv_format == AttentionKvFormat::kModelSpecific) softmax_specific = true;
    } else {
      if (a.sink) r.problems.push_back(tag + ": a softmax sink only exists on softmax attention");
      if (a.window != 0) r.problems.push_back(tag + ": a sliding window only exists on softmax attention");
    }
  }
  core_ok = has_softmax && !softmax_specific;
  CheckPath("decode", k.decode, core_ok, /*need_row_invariant=*/true, &r);
  CheckPath("prefill", k.prefill, core_ok, /*need_row_invariant=*/false, &r);
  return r;
}

std::string ValidateChunkAgainstWindow(std::uint32_t prefill_chunk,
                                       std::uint32_t window) {
  if (window == 0) return {};
  if (prefill_chunk == 0)
    return "prefill chunk must be at least one token";
  if (prefill_chunk >= window)
    return "prefill chunk (" + std::to_string(prefill_chunk) +
           ") must be smaller than the sliding window (" +
           std::to_string(window) + "); equal-or-larger chunks silently " +
           "corrupt SWA attention at the chunk boundary";
  return {};
}

}  // namespace mbufo::core::engine