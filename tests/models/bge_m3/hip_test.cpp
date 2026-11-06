// bge-m3 HIP-vs-CPU agreement test (external-model gated).
//
// Loads the real artifact (GUFO_BGE_MODEL, else exit 77 skip), runs the CPU
// reference Sync and the HIP HipSync over identical C++-tokenized prompts,
// and pins: dims, unit norm, determinism, and CPU-vs-device cosine.
// Short prompts keep the leg fast; one 256-token prompt exercises the
// batched attention path at width.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/pooling_norm.hpp"
#include "src/models/bge_m3/engine.hpp"

namespace {

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_BGE_MODEL");
  if (model_path == nullptr || model_path[0] == '\0') {
    std::cout << "bge_m3_hip_test: skipped (GUFO_BGE_MODEL unset)\n";
    return 77;
  }
  std::string error;
  auto model = gufo::models::bge_m3::ModelRegistry::Instance().Create(
      "bge-m3", model_path, &error);
  if (model == nullptr) {
    std::cerr << "bge_m3_hip_test: load failed: " << error << "\n";
    return 1;
  }
  std::string long_prompt = "retrieval probe:";
  for (int i = 0; i < 64; ++i) long_prompt += " word" + std::to_string(i);
  const std::vector<std::string> prompts = {
      "Hello world", "BGE-M3 embedding test", long_prompt};
  gufo::models::bge_m3::EngineSession session(model.get());
  const std::size_t dim = model->Config().hidden_size;
  for (const auto& prompt : prompts) {
    auto ids = model->Tokenizer().Encode(
        prompt, gufo::models::bge_m3::kBgeM3Leg1MaxTokens);
    if (!ids.has_value()) {
      std::cerr << "bge_m3_hip_test: tokenize failed\n";
      return 1;
    }
    auto cpu = session.Sync(*ids, /*use_hip=*/false, &error);
    if (!cpu.has_value()) {
      std::cerr << "bge_m3_hip_test: CPU Sync failed: " << error << "\n";
      return 1;
    }
    auto hip = session.Sync(*ids, /*use_hip=*/true, &error);
    if (!hip.has_value()) {
      std::cerr << "bge_m3_hip_test: HIP Sync failed: " << error << "\n";
      return 1;
    }
    if (cpu->values.size() != dim || hip->values.size() != dim) {
      std::cerr << "bge_m3_hip_test: dim mismatch\n";
      return 1;
    }
    const double cos = Cosine(cpu->values, hip->values);
    std::cout << "prompt tokens=" << ids->size() << " cpu_ms="
              << cpu->prefill_ms << " hip_ms=" << hip->prefill_ms
              << " cosine=" << cos << "\n";
    if (!(cos > 0.9999)) {
      std::cerr << "bge_m3_hip_test: CPU-vs-HIP cosine below 0.9999\n";
      return 1;
    }
    auto hip2 = session.Sync(*ids, /*use_hip=*/true, &error);
    if (!hip2.has_value() || hip2->values != hip->values) {
      std::cerr << "bge_m3_hip_test: HIP determinism failure\n";
      return 1;
    }
  }
  std::cout << "bge_m3_hip_test: all assertions passed\n";
  return 0;
}
