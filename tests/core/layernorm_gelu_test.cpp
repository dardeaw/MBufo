// CPU coverage for src/core/compute/layernorm/layernorm.hpp and
// src/core/compute/gelu/gelu.hpp: biased-variance LayerNorm with/without
// bias, in-place safety, shape refusal; exact-erf vs tanh GELU pinned against
// closed forms (the bge-m3对照: HF hidden_act "gelu" == exact erf);
// AddPositional row addition. Assert style mirrors tests/core/pooling_norm_test.cpp.
#include "src/core/compute/gelu/gelu.hpp"
#include "src/core/compute/layernorm/layernorm.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <span>
#include <vector>

namespace {

bool Near(float a, float b, float eps = 1e-6f) {
  return std::fabs(a - b) <= eps;
}

void TestLayerNormBasic() {
  // x = [1,2,3,4]: mean 2.5, biased var 1.25.
  const std::vector<float> x{1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> w{1.0F, 1.0F, 1.0F, 1.0F};
  const std::vector<float> no_bias;
  std::vector<float> out(4, 0.0F);
  assert(gufo::core::ln::LayerNorm(x, w, no_bias, 1e-5F, out));
  const double inv = 1.0 / std::sqrt(1.25 + 1e-5);
  assert(Near(out[0], static_cast<float>(-1.5 * inv)));
  assert(Near(out[1], static_cast<float>(-0.5 * inv)));
  assert(Near(out[2], static_cast<float>(0.5 * inv)));
  assert(Near(out[3], static_cast<float>(1.5 * inv)));
}

void TestLayerNormBiasAndScale() {
  const std::vector<float> x{2.0F, 2.0F, 2.0F, 2.0F};  // zero variance
  const std::vector<float> w{0.5F, 2.0F, 1.0F, 1.0F};
  const std::vector<float> b{0.25F, -0.5F, 1.0F, 2.0F};
  std::vector<float> out(4, 0.0F);
  assert(gufo::core::ln::LayerNorm(x, w, b, 1e-5F, out));
  // (x - mean) == 0, so out == bias exactly.
  assert(Near(out[0], 0.25F) && Near(out[1], -0.5F) && Near(out[2], 1.0F) &&
         Near(out[3], 2.0F));
}

void TestLayerNormInPlaceAndRefusal() {
  std::vector<float> x{1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> w{1.0F, 1.0F, 1.0F, 1.0F};
  const std::vector<float> no_bias;
  assert(gufo::core::ln::LayerNorm(x, w, no_bias, 1e-5F, x));  // in-place
  const double inv = 1.0 / std::sqrt(1.25 + 1e-5);
  assert(Near(x[3], static_cast<float>(1.5 * inv)));

  std::vector<float> bad(3, -1.0F);
  const std::vector<float> w_bad{1.0F, 1.0F, 1.0F};
  assert(!gufo::core::ln::LayerNorm(x, w_bad, no_bias, 1e-5F, bad));
  assert(!gufo::core::ln::LayerNorm(x, w, no_bias, 1e-5F,
                                    std::span<float>(bad).first(2)));
  const std::vector<float> b_bad{0.0F};
  assert(!gufo::core::ln::LayerNorm(x, w, b_bad, 1e-5F, x));
  for (const float v : bad) assert(v == -1.0F);
}

void TestGeluModes() {
  using gufo::core::gelu::ApplyGelu;
  using gufo::core::gelu::GeluExact;
  using gufo::core::gelu::GeluMode;
  // Closed forms: exact(1) = Phi(1) = 0.8413447461; tanh-approx(1) =
  // 0.8411919906. The gap proves the two modes are distinct builds.
  assert(Near(GeluExact(1.0F), 0.8413447F, 1e-6F));
  assert(Near(gufo::core::gelu::GeluTanh(1.0F), 0.8411920F, 1e-6F));
  assert(GeluExact(0.0F) == 0.0F);
  assert(Near(GeluExact(-1.0F), -0.1586553F, 1e-6F));
  // Symmetry spot: exact(-x) == -exact-reflected? No: check odd-ish tails.
  assert(GeluExact(5.0F) > 4.999F && GeluExact(5.0F) < 5.0F);
  assert(std::fabs(gufo::core::gelu::GeluTanh(-5.0F)) < 1e-3F);

  const std::vector<float> in{1.0F, -1.0F, 0.0F};
  std::vector<float> out(3, 0.0F);
  assert(ApplyGelu(in, out, GeluMode::kExact));
  assert(Near(out[0], 0.8413447F, 1e-6F));
  assert(ApplyGelu(in, out, GeluMode::kTanh));
  assert(Near(out[0], 0.8411920F, 1e-6F));
  std::vector<float> short_out(2, 0.0F);
  assert(!ApplyGelu(in, short_out, GeluMode::kExact));
}

void TestAddPositional() {
  std::vector<float> hidden{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};  // 3x2
  const std::vector<float> pos{10.0F, 20.0F, 30.0F, 40.0F,
                               50.0F, 60.0F, 70.0F, 80.0F};  // longer table
  assert(gufo::core::ln::AddPositional(hidden, pos, 3, 2));
  assert(Near(hidden[0], 11.0F) && Near(hidden[1], 22.0F));
  assert(Near(hidden[4], 55.0F) && Near(hidden[5], 66.0F));
  assert(!gufo::core::ln::AddPositional(hidden, pos, 5, 2));  // table too short
  assert(!gufo::core::ln::AddPositional(hidden, pos, 3, 3));  // shape mismatch
}

}  // namespace

int main() {
  TestLayerNormBasic();
  TestLayerNormBiasAndScale();
  TestLayerNormInPlaceAndRefusal();
  TestGeluModes();
  TestAddPositional();
  std::cout << "layernorm_gelu_test: all assertions passed\n";
  return 0;
}
