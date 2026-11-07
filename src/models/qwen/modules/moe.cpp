#include "src/models/qwen/modules/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace gufo::models::qwen {
namespace {

[[nodiscard]] float SigmoidF(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}

[[nodiscard]] float SiluF(float x) noexcept {
  return x * SigmoidF(x);
}

// Row-major GEMV row: out = W[row, :] . x.
[[nodiscard]] float DotRow(const QwenTensorRef& w, std::size_t row,
                           std::size_t cols, const float* x) noexcept {
  const std::size_t base = row * cols;
  float acc = 0.0F;
  for (std::size_t i = 0; i < cols; ++i) {
    acc += w.Get(base + i) * x[i];
  }
  return acc;
}

}  // namespace

void MoeForwardCpu(const core::ModelConfig& config,
                   const QwenLayerWeights& layer, std::span<const float> x,
                   std::span<float> out) {
  const std::size_t hidden = config.hidden_size;
  const std::size_t num_experts = config.expert_count;
  const std::size_t top_k = config.expert_used_count;
  const std::size_t expert_ff = config.expert_ffn_size;
  const std::size_t shexp_ff = config.shared_expert_ffn_size;
  if (hidden == 0 || num_experts == 0 || top_k == 0 || expert_ff == 0 ||
      shexp_ff == 0 || x.size() < hidden || out.size() < hidden) {
    return;
  }
  const std::size_t batch = std::min(x.size(), out.size()) / hidden;

  std::vector<float> logits(num_experts);
  std::vector<std::uint32_t> order(num_experts);
  std::vector<std::int32_t> ids(top_k);
  std::vector<float> weights(top_k);
  std::vector<float> gate(expert_ff);
  std::vector<float> up(std::max(expert_ff, shexp_ff));
  std::vector<float> sgate(shexp_ff);
  std::vector<float> sup(shexp_ff);

  for (std::size_t b = 0; b < batch; ++b) {
    const float* xp = x.data() + b * hidden;
    float* op = out.data() + b * hidden;

    // Router: softmax over experts, top-k, renormalize (clamped).
    float max_logit = logits[0] * 0.0F;
    for (std::size_t e = 0; e < num_experts; ++e) {
      logits[e] = DotRow(layer.ffn_gate_inp, e, hidden, xp);
      if (e == 0 || logits[e] > max_logit) {
        max_logit = logits[e];
      }
    }
    float denom = 0.0F;
    for (std::size_t e = 0; e < num_experts; ++e) {
      logits[e] = std::exp(logits[e] - max_logit);
      denom += logits[e];
    }
    for (std::size_t e = 0; e < num_experts; ++e) {
      logits[e] /= denom;
    }
    std::iota(order.begin(), order.end(), 0U);
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return logits[a] > logits[b];
                      });
    float wsum = 0.0F;
    for (std::size_t s = 0; s < top_k; ++s) {
      ids[s] = static_cast<std::int32_t>(order[s]);
      weights[s] = logits[order[s]];
      wsum += weights[s];
    }
    if (wsum < 6.103515625e-5F) {
      wsum = 6.103515625e-5F;
    }
    for (std::size_t s = 0; s < top_k; ++s) {
      weights[s] /= wsum;
    }

    for (std::size_t i = 0; i < hidden; ++i) {
      op[i] = 0.0F;
    }
    // Routed experts: SwiGLU per expert, weighted accumulate.
    for (std::size_t s = 0; s < top_k; ++s) {
      const std::size_t e = static_cast<std::size_t>(ids[s]);
      const std::size_t e_gate_base = e * expert_ff * hidden;
      for (std::size_t f = 0; f < expert_ff; ++f) {
        float g = 0.0F, u = 0.0F;
        const std::size_t row = e_gate_base + f * hidden;
        for (std::size_t i = 0; i < hidden; ++i) {
          g += layer.ffn_gate_exps.Get(row + i) * xp[i];
          u += layer.ffn_up_exps.Get(row + i) * xp[i];
        }
        gate[f] = g;
        up[f] = u;
      }
      const std::size_t e_down_base = e * hidden * expert_ff;
      const float w = weights[s];
      for (std::size_t i = 0; i < hidden; ++i) {
        float acc = 0.0F;
        for (std::size_t f = 0; f < expert_ff; ++f) {
          acc += layer.ffn_down_exps.Get(e_down_base + i * expert_ff + f) *
                 SiluF(gate[f]) * up[f];
        }
        op[i] += w * acc;
      }
    }

    // Shared expert: SwiGLU dense + sigmoid gate scale.
    for (std::size_t f = 0; f < shexp_ff; ++f) {
      sgate[f] = DotRow(layer.ffn_gate_shexp, f, hidden, xp);
      sup[f] = DotRow(layer.ffn_up_shexp, f, hidden, xp);
    }
    const float shared_gate =
        SigmoidF(DotRow(layer.ffn_gate_inp_shexp, 0, hidden, xp));
    // ffn_gate_inp_shexp is a [hidden] vector; DotRow with row 0 covers it.
    float gscale = shared_gate;
    for (std::size_t i = 0; i < hidden; ++i) {
      float acc = 0.0F;
      for (std::size_t f = 0; f < shexp_ff; ++f) {
        acc += layer.ffn_down_shexp.Get(i * shexp_ff + f) * SiluF(sgate[f]) *
               sup[f];
      }
      op[i] += gscale * acc;
    }
  }
}

}  // namespace gufo::models::qwen
