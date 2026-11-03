// CPU coverage for src/core/pooling_norm.hpp: strategy mapping, pooled-row
// selection, L2 normalization, and determinism. Mirrors the assert style of
// the other tests/core suites (no gtest dependency).
#include "src/core/pooling_norm.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using gufo::core::embedding::ApplyPooling;
using gufo::core::embedding::L2Normalize;
using gufo::core::embedding::PoolingStrategy;
using gufo::core::embedding::PoolingStrategyFromGgufType;
using gufo::core::embedding::VectorNorm;

bool Near(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps;
}

void TestGgufMapping() {
  assert(!PoolingStrategyFromGgufType(0).has_value());
  assert(PoolingStrategyFromGgufType(1).value() == PoolingStrategy::kMean);
  assert(PoolingStrategyFromGgufType(2).value() == PoolingStrategy::kCls);
  assert(PoolingStrategyFromGgufType(3).value() ==
         PoolingStrategy::kLastToken);
  assert(!PoolingStrategyFromGgufType(4).has_value());
  assert(!PoolingStrategyFromGgufType(0xFFFFFFFFu).has_value());
}

void TestPoolingRows() {
  // 3 tokens x 4 dims, row-major.
  const std::vector<float> hidden{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> out(4, 0.0f);

  assert(ApplyPooling(hidden, 3, 4, PoolingStrategy::kLastToken, out));
  assert(Near(out[0], 9) && Near(out[1], 10) && Near(out[2], 11) &&
         Near(out[3], 12));

  assert(ApplyPooling(hidden, 3, 4, PoolingStrategy::kCls, out));
  assert(Near(out[0], 1) && Near(out[1], 2) && Near(out[2], 3) &&
         Near(out[3], 4));

  assert(ApplyPooling(hidden, 3, 4, PoolingStrategy::kMean, out));
  assert(Near(out[0], 5) && Near(out[1], 6) && Near(out[2], 7) &&
         Near(out[3], 8));

  // Single-token input: every strategy agrees on the only row.
  const std::vector<float> one{7, 8};
  std::vector<float> one_out(2, 0.0f);
  assert(ApplyPooling(one, 1, 2, PoolingStrategy::kLastToken, one_out));
  assert(ApplyPooling(one, 1, 2, PoolingStrategy::kMean, one_out));
  assert(ApplyPooling(one, 1, 2, PoolingStrategy::kCls, one_out));
  assert(Near(one_out[0], 7) && Near(one_out[1], 8));

  // Shape mismatches refuse without touching scratch.
  std::vector<float> poisoned(4, -1.0f);
  assert(!ApplyPooling(hidden, 3, 4, PoolingStrategy::kMean,
                       std::span<float>(poisoned).first(3)));
  assert(!ApplyPooling(hidden, 2, 4, PoolingStrategy::kMean, poisoned));
  assert(!ApplyPooling(hidden, 0, 4, PoolingStrategy::kMean, poisoned));
  for (const float x : poisoned) assert(Near(x, -1.0f));
}

void TestNorm() {
  std::vector<float> v{3.0f, 4.0f};
  assert(Near(static_cast<float>(VectorNorm(v)), 5.0f));
  const double pre = L2Normalize(v);
  assert(Near(static_cast<float>(pre), 5.0f));
  assert(Near(v[0], 0.6f) && Near(v[1], 0.8f));
  assert(Near(static_cast<float>(VectorNorm(v)), 1.0f, 1e-6f));

  // Zero vector: untouched, reports 0.
  std::vector<float> zero(8, 0.0f);
  assert(L2Normalize(zero) == 0.0);
  for (const float x : zero) assert(x == 0.0f);
}

void TestDeterminism() {
  const std::vector<float> hidden{0.5f, -1.25f, 3.0f,  4.5f,
                                  2.0f, 0.75f,  -0.5f, 1.0f};
  std::vector<float> first(4), second(4);
  assert(ApplyPooling(hidden, 2, 4, PoolingStrategy::kMean, first));
  assert(ApplyPooling(hidden, 2, 4, PoolingStrategy::kMean, second));
  (void)L2Normalize(first);
  (void)L2Normalize(second);
  assert(std::memcmp(first.data(), second.data(),
                     first.size() * sizeof(float)) == 0);
  assert(Near(static_cast<float>(VectorNorm(first)), 1.0f, 1e-6f));
}

}  // namespace

int main() {
  TestGgufMapping();
  TestPoolingRows();
  TestNorm();
  TestDeterminism();
  std::cout << "pooling_norm_test: all passed\n";
  return 0;
}
