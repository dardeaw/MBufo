// bge-m3 CPU forward property tests (synthetic weights, no external files).
//
// Covers wiring, not numerics: determinism (bit-identical reruns), shape and
// id refusals, CLS-pooling identity (Sync pools row 0), unit L2 norm, and
// input sensitivity. Numeric exactness vs HuggingFace is verified out of tree
// (local torch cosine check); the HIP test pins device-vs-CPU agreement.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "src/core/pooling_norm.hpp"
#include "src/models/bge_m3/engine.hpp"
#include "src/models/bge_m3/forward.hpp"

namespace {

gufo::models::bge_m3::BgeM3F32Weights MakeTinyWeights() {
  using gufo::models::bge_m3::BgeM3F32Weights;
  BgeM3F32Weights weights;
  weights.config.num_layers = 2;
  weights.config.hidden_size = 8;
  weights.config.intermediate_size = 16;
  weights.config.num_attention_heads = 2;
  weights.config.vocab_size = 16;
  weights.config.max_position_embeddings = 8;
  weights.config.layer_norm_eps = 1e-5F;
  const std::size_t h = 8;
  const std::size_t inter = 16;
  auto fill = [](std::vector<float>& v, std::size_t n, float base) {
    v.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      // Deterministic pseudo-random in [-0.5, 0.5).
      const unsigned long long s =
          0x9E3779B97F4A7C15ULL * (i + 1) + 0x12345ULL;
      v[i] = base +
             (static_cast<float>((s >> 17) % 1000) / 1000.0F - 0.5F) * 0.1F;
    }
  };
  fill(weights.token_embd, 16 * h, 0.0F);
  fill(weights.position_embd, 8 * h, 0.0F);
  weights.position_rows = 8;
  fill(weights.token_types, 1 * h, 0.0F);
  fill(weights.emb_norm_w, h, 1.0F);
  fill(weights.emb_norm_b, h, 0.0F);
  weights.layers.resize(2);
  for (auto& layer : weights.layers) {
    fill(layer.q_w, h * h, 0.0F);
    fill(layer.q_b, h, 0.0F);
    fill(layer.k_w, h * h, 0.0F);
    fill(layer.k_b, h, 0.0F);
    fill(layer.v_w, h * h, 0.0F);
    fill(layer.v_b, h, 0.0F);
    fill(layer.out_w, h * h, 0.0F);
    fill(layer.out_b, h, 0.0F);
    fill(layer.attn_norm_w, h, 1.0F);
    fill(layer.attn_norm_b, h, 0.0F);
    fill(layer.up_w, inter * h, 0.0F);
    fill(layer.up_b, inter, 0.0F);
    fill(layer.down_w, h * inter, 0.0F);
    fill(layer.down_b, h, 0.0F);
    fill(layer.out_norm_w, h, 1.0F);
    fill(layer.out_norm_b, h, 0.0F);
  }
  return weights;
}

bool Near(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps;
}

void TestDeterminismAndShape() {
  using gufo::models::bge_m3::BgeM3F32Weights;
  using gufo::models::bge_m3::BgeM3Scratch;
  using gufo::models::bge_m3::CpuForward;
  const BgeM3F32Weights weights = MakeTinyWeights();
  auto scratch = BgeM3Scratch::Create(weights.config, 8);
  const std::vector<std::uint32_t> ids{0, 5, 9, 2};
  std::vector<float> hidden_a(4 * 8, 0.0F);
  std::vector<float> hidden_b(4 * 8, 0.0F);
  assert(CpuForward(weights, ids, scratch, hidden_a));
  assert(CpuForward(weights, ids, scratch, hidden_b));
  assert(hidden_a == hidden_b);  // bit-identical reruns
  // Refusals: empty, over max_seq, out-of-vocab id, wrong-size out.
  std::vector<std::uint32_t> empty;
  assert(!CpuForward(weights, empty, scratch, hidden_a));
  const std::vector<std::uint32_t> too_long(9, 1);
  assert(!CpuForward(weights, too_long, scratch, hidden_a));
  const std::vector<std::uint32_t> bad_id{0, 16, 2};
  assert(!CpuForward(weights, bad_id, scratch, hidden_a));
  std::vector<float> short_out(4 * 7, 0.0F);
  assert(!CpuForward(weights, ids, scratch, short_out));
}

void TestPoolingAndSensitivity() {
  using gufo::core::embedding::ApplyPooling;
  using gufo::core::embedding::L2Normalize;
  using gufo::core::embedding::PoolingStrategy;
  using gufo::models::bge_m3::CpuForward;
  const auto weights = MakeTinyWeights();
  auto scratch = gufo::models::bge_m3::BgeM3Scratch::Create(weights.config, 8);
  const std::vector<std::uint32_t> ids{0, 5, 9, 2};
  const std::vector<std::uint32_t> other{0, 6, 9, 2};
  std::vector<float> ha(4 * 8), hb(4 * 8);
  assert(CpuForward(weights, ids, scratch, ha));
  assert(CpuForward(weights, other, scratch, hb));
  assert(ha != hb);  // input-sensitive
  std::vector<float> pooled(8, 0.0F);
  assert(ApplyPooling(ha, 4, 8, PoolingStrategy::kCls, pooled));
  for (std::size_t d = 0; d < 8; ++d) assert(pooled[d] == ha[d]);  // row 0
  const double pre = L2Normalize(pooled);
  assert(pre > 0.0);
  double norm = 0.0;
  for (const float x : pooled) norm += x * x;
  assert(Near(static_cast<float>(std::sqrt(norm)), 1.0F, 1e-6F));
}

}  // namespace

int main() {
  TestDeterminismAndShape();
  TestPoolingAndSensitivity();
  std::cout << "bge_m3_forward_test: all assertions passed\n";
  return 0;
}
