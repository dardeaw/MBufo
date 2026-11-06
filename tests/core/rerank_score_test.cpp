// CPU golden tests for the shared rerank scoring math.
#include "src/core/rerank_score.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const char* name) {
  if (!condition) {
    ++failures;
    std::printf("FAIL: %s\n", name);
  } else {
    std::printf("ok: %s\n", name);
  }
}

bool Near(float got, double want, double tol = 1e-5) {
  return std::fabs(static_cast<double>(got) - want) <= tol;
}

}  // namespace

int main() {
  using gufo::core::rerank::RankIndices;
  using gufo::core::rerank::SequenceHeadLogit;
  using gufo::core::rerank::Sigmoid;
  using gufo::core::rerank::YesNoScore;

  Check(Sigmoid(0.0F) == 0.5F, "sigmoid(0) == 0.5");
  Check(Near(Sigmoid(2.0F), 0.8807970780), "sigmoid(2)");
  Check(Near(Sigmoid(-2.0F), 0.1192029220), "sigmoid(-2)");
  Check(Sigmoid(100.0F) > 0.999999F, "sigmoid saturates high");
  Check(Sigmoid(-100.0F) < 0.000001F, "sigmoid saturates low");

  Check(YesNoScore(0.0F, 0.0F) == 0.5F, "yes-no tie == 0.5");
  Check(Near(YesNoScore(2.0F, 0.0F), 0.8807970780), "yes-no(2, 0)");
  Check(Near(YesNoScore(0.0F, 2.0F), 0.1192029220), "yes-no(0, 2)");
  Check(Near(YesNoScore(1000.0F, 1000.0F), 0.5), "yes-no large tie");
  Check(Near(YesNoScore(1001.0F, 1000.0F), 0.7310585786), "yes-no large gap");

  // 2-wide identity head: tanh([1, 0]) . [1, 1] = tanh(1).
  const std::vector<float> cls = {1.0F, 0.0F};
  const std::vector<float> dense = {1.0F, 0.0F, 0.0F, 1.0F};
  const std::vector<float> bias = {0.0F, 0.0F};
  const std::vector<float> out_w = {1.0F, 1.0F};
  const float logit = SequenceHeadLogit(cls, dense, bias, out_w, 0.0F);
  Check(Near(logit, 0.7615941556), "sequence head logit == tanh(1)");
  Check(Near(Sigmoid(logit), 0.6816997426), "sequence head score");
  const float bad =
      SequenceHeadLogit(cls, dense, bias, std::vector<float>{1.0F}, 0.0F);
  Check(bad != bad, "sequence head shape mismatch is NaN");

  const std::vector<float> scores = {0.1F, 0.9F, 0.5F};
  const auto order = RankIndices(scores);
  Check(order.size() == 3 && order[0] == 1 && order[1] == 2 && order[2] == 0,
        "rank order");
  const auto top2 = RankIndices(scores, 2);
  Check(top2.size() == 2 && top2[0] == 1 && top2[1] == 2, "rank top_n");
  const std::vector<float> tied = {0.5F, 0.5F, 0.9F};
  const auto tied_order = RankIndices(tied);
  Check(tied_order.size() == 3 && tied_order[0] == 2 && tied_order[1] == 0 &&
            tied_order[2] == 1,
        "rank ties stable");

  if (failures != 0) {
    std::printf("%d FAILURES\n", failures);
    return 1;
  }
  std::printf("all rerank_score tests passed\n");
  return 0;
}
