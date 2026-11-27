// Hardware regression / accuracy driver for the NPU op-graph executor.
//
//   npu_graph_models_test bge-ref   <bge-m3.gguf> <out.txt>
//       CPU F32 CLS + existing NpuBgeRuntime (gemm_1024x1024 BF16) CLS.
//   npu_graph_models_test bge-graph <bge-m3.gguf> <ref.txt> [--host]
//       Graph (Q8 engine, or --host exact fp32) CLS vs the reference file.
//   npu_graph_models_test qwen-emb  <Qwen3-Embedding.gguf> [out.txt] [--host-only]
//       NPU graph embedding vs the host-fp32 graph (same weights) + latency.
//   npu_graph_models_test qwen-rerank <Qwen3-Reranker.gguf> [--host-only]
//       NPU graph P(yes) vs the host-fp32 graph on fixed pairs + latency.
//   npu_graph_models_test ids-emb|ids-rerank ...  (see usage)
//
// Needs the XDNA2 device (except --host-only); run one NPU process at a time.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_graph_models.hpp"
#include "src/models/bge_m3/engine.hpp"
#include "src/models/bge_m3/forward.hpp"

namespace {

using mbufo::core::npu::NpuGraphBgeRuntime;
using mbufo::core::npu::NpuQwen3Encoder;
namespace graph = mbufo::core::npu::graph;

const std::vector<std::string> kTexts = {
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Neural processing units accelerate matrix multiplication for transformer "
    "inference by keeping quantized weights resident in on-chip memory and "
    "streaming activations through a spatial array of compute tiles. This "
    "design trades peak flexibility for energy efficiency, which matters for "
    "always-on embedding and reranking workloads on laptops and small servers.",
    "台北是台灣的首都，擁有許多夜市與捷運路線，是亞洲重要的科技中心之一。",
};

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

void WriteVecs(const std::string& path,
               const std::map<std::string, std::vector<float>>& m) {
  std::ofstream f(path);
  f.precision(9);
  for (const auto& [k, v] : m) {
    f << k;
    for (float x : v) f << ' ' << x;
    f << '\n';
  }
}
std::map<std::string, std::vector<float>> ReadVecs(const std::string& path) {
  std::map<std::string, std::vector<float>> m;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream is(line);
    std::string k;
    is >> k;
    std::vector<float> v;
    float x;
    while (is >> x) v.push_back(x);
    m[k] = std::move(v);
  }
  return m;
}

int Bge(bool ref_mode, const std::string& gguf, const std::string& file,
        bool host) {
  std::string err;
  auto model = mbufo::models::bge_m3::ModelRegistry::Instance().Create(
      "bge-m3", gguf, &err);
  if (!model) {
    std::cerr << "load failed: " << err << "\n";
    return 2;
  }
  std::vector<std::vector<std::uint32_t>> ids;
  for (const auto& t : kTexts) {
    auto e = model->Tokenizer().Encode(t, 2040);
    if (!e) {
      std::cerr << "tokenize failed\n";
      return 2;
    }
    ids.push_back(*e);
  }
  const auto& cfg = model->Config();
  const std::size_t H = cfg.hidden_size;
  if (ref_mode) {
    std::map<std::string, std::vector<float>> out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      auto scratch =
          mbufo::models::bge_m3::BgeM3Scratch::Create(cfg, ids[i].size());
      std::vector<float> hid(ids[i].size() * H);
      if (!mbufo::models::bge_m3::CpuForward(model->F32(), ids[i], scratch,
                                             hid)) {
        std::cerr << "cpu forward failed\n";
        return 2;
      }
      out["cpu" + std::to_string(i)] =
          std::vector<float>(hid.begin(), hid.begin() + H);
    }
    if (!model->LoadNpuWeights("", &err)) {
      std::cerr << "existing NPU load failed: " << err << "\n";
      return 2;
    }
    for (int rep = 0; rep < 2; ++rep) {
      for (std::size_t i = 0; i < ids.size(); ++i) {
        std::vector<float> cls;
        double ms = 0;
        if (!model->RunNpuCls(ids[i], &cls, &ms, &err)) {
          std::cerr << "existing NPU run failed: " << err << "\n";
          return 2;
        }
        if (rep == 1) {
          std::printf("existing-npu text%zu tokens=%zu ms=%.1f\n", i,
                      ids[i].size(), ms);
          out["old" + std::to_string(i)] = cls;
        }
      }
    }
    WriteVecs(file, out);
    return 0;
  }
  const auto ref = ReadVecs(file);
  NpuGraphBgeRuntime rt;
  const auto t0 = std::chrono::steady_clock::now();
  if (!rt.Init(model->F32(),
               host ? graph::LinearBackend::kHostF32
                    : graph::LinearBackend::kNpuQ8,
               &err)) {
    std::cerr << "graph init failed: " << err << "\n";
    return 2;
  }
  std::printf("graph init ms=%.0f backend=%s\n",
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count(),
              host ? "host-fp32" : "npu-q8");
  double worst_cpu = 1, worst_old = 1;
  for (int rep = 0; rep < 2; ++rep) {
    for (std::size_t i = 0; i < ids.size(); ++i) {
      std::vector<float> cls;
      double ms = 0;
      if (!rt.ForwardCls(ids[i], &cls, &ms, &err)) {
        std::cerr << "graph run failed: " << err << "\n";
        return 2;
      }
      if (rep == 0) continue;  // warm
      const double c_cpu = Cosine(cls, ref.at("cpu" + std::to_string(i)));
      const auto it = ref.find("old" + std::to_string(i));
      const double c_old = it == ref.end() ? 0.0 : Cosine(cls, it->second);
      const double c_old_cpu =
          it == ref.end() ? 0.0
                          : Cosine(it->second, ref.at("cpu" + std::to_string(i)));
      const auto& s = rt.last_stats();
      std::printf(
          "text%zu tokens=%zu ms=%.1f (npu %.1f linear %.1f attn %.1f other "
          "%.1f) cos(graph,cpu)=%.5f cos(graph,existing)=%.5f "
          "cos(existing,cpu)=%.5f\n",
          i, ids[i].size(), ms, s.npu_ms, s.linear_ms, s.attention_ms,
          s.other_host_ms, c_cpu, c_old, c_old_cpu);
      worst_cpu = std::min(worst_cpu, c_cpu);
      if (it != ref.end()) worst_old = std::min(worst_old, c_old);
    }
  }
  std::printf("WORST cos(graph,cpu)=%.5f cos(graph,existing)=%.5f\n",
              worst_cpu, worst_old);
  return 0;
}

// --no-host: NPU only (clean latency numbers; no host-fp32 reference).
int QwenEmbNoHost(const std::string& gguf) {
  std::string err;
  auto npu = NpuQwen3Encoder::Load(gguf, NpuQwen3Encoder::Role::kEmbedding,
                                   graph::LinearBackend::kNpuQ8, &err);
  if (!npu) {
    std::cerr << "npu load failed: " << err << "\n";
    return 2;
  }
  for (int rep = 0; rep < 4; ++rep) {
    for (std::size_t i = 0; i < kTexts.size(); ++i) {
      std::vector<float> n;
      std::size_t toks = 0;
      double ms = 0;
      if (!npu->Embed(kTexts[i], &n, &toks, &ms, &err)) {
        std::cerr << "npu embed failed: " << err << "\n";
        return 2;
      }
      if (rep < 2) continue;
      const auto& s = npu->last_stats();
      std::printf("npu-only text%zu tokens=%zu ms=%.1f (npu %.1f linear %.1f "
                  "attn %.1f other %.1f)\n",
                  i, toks, ms, s.npu_ms, s.linear_ms, s.attention_ms,
                  s.other_host_ms);
    }
  }
  return 0;
}

int QwenEmb(const std::string& gguf, const std::string& out_path,
            bool host_only) {
  std::string err;
  auto host = NpuQwen3Encoder::Load(gguf, NpuQwen3Encoder::Role::kEmbedding,
                                    graph::LinearBackend::kHostF32, &err);
  if (!host) {
    std::cerr << "host load failed: " << err << "\n";
    return 2;
  }
  std::unique_ptr<NpuQwen3Encoder> npu;
  if (!host_only) {
    const auto t0 = std::chrono::steady_clock::now();
    npu = NpuQwen3Encoder::Load(gguf, NpuQwen3Encoder::Role::kEmbedding,
                                graph::LinearBackend::kNpuQ8, &err);
    if (!npu) {
      std::cerr << "npu load failed: " << err << "\n";
      return 2;
    }
    std::printf("npu load ms=%.0f\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count());
  }
  std::map<std::string, std::vector<float>> out;
  double worst = 1;
  for (int rep = 0; rep < 2; ++rep) {
    for (std::size_t i = 0; i < kTexts.size(); ++i) {
      std::vector<float> h, n;
      std::size_t toks = 0;
      double hms = 0, nms = 0;
      if (!host->Embed(kTexts[i], &h, &toks, &hms, &err)) {
        std::cerr << "host embed failed: " << err << "\n";
        return 2;
      }
      out["host" + std::to_string(i)] = h;
      if (!npu) continue;
      if (!npu->Embed(kTexts[i], &n, &toks, &nms, &err)) {
        std::cerr << "npu embed failed: " << err << "\n";
        return 2;
      }
      out["npu" + std::to_string(i)] = n;
      if (rep == 0) continue;
      const auto& s = npu->last_stats();
      const double c = Cosine(h, n);
      worst = std::min(worst, c);
      std::printf(
          "text%zu tokens=%zu npu_ms=%.1f (npu %.1f linear %.1f attn %.1f "
          "other %.1f) host_ms=%.0f cos(npu,host)=%.5f\n",
          i, toks, nms, s.npu_ms, s.linear_ms, s.attention_ms, s.other_host_ms,
          hms, c);
    }
  }
  if (npu) std::printf("WORST cos(npu,host)=%.5f\n", worst);
  if (!out_path.empty()) WriteVecs(out_path, out);
  return 0;
}

const std::vector<std::pair<std::string, std::vector<std::string>>> kRerank = {
    {"What is the capital of France?",
     {"Paris is the capital and most populous city of France.",
      "The mitochondria is the powerhouse of the cell.",
      "Berlin is the capital of Germany, located on the river Spree."}},
    {"how to speed up embedding inference on a laptop NPU",
     {"Keeping quantized weights resident on the NPU and batching rows "
      "reduces latency for embedding models.",
      "Banana bread is made with ripe bananas, flour, sugar and eggs."}},
};

int QwenRerank(const std::string& gguf, bool host_only) {
  std::string err;
  auto host = NpuQwen3Encoder::Load(gguf, NpuQwen3Encoder::Role::kReranker,
                                    graph::LinearBackend::kHostF32, &err);
  if (!host) {
    std::cerr << "host load failed: " << err << "\n";
    return 2;
  }
  std::unique_ptr<NpuQwen3Encoder> npu;
  if (!host_only) {
    npu = NpuQwen3Encoder::Load(gguf, NpuQwen3Encoder::Role::kReranker,
                                graph::LinearBackend::kNpuQ8, &err);
    if (!npu) {
      std::cerr << "npu load failed: " << err << "\n";
      return 2;
    }
  }
  double worst = 0;
  for (int rep = 0; rep < 2; ++rep) {
    for (const auto& [q, docs] : kRerank) {
      for (const auto& d : docs) {
        float hs = 0, ns = 0;
        std::size_t toks = 0;
        double hms = 0, nms = 0;
        if (!host->ScorePair(q, d, "", &hs, &toks, &hms, &err)) {
          std::cerr << "host rerank failed: " << err << "\n";
          return 2;
        }
        if (!npu) {
          if (rep == 1) std::printf("host score=%.5f tokens=%zu\n", hs, toks);
          continue;
        }
        if (!npu->ScorePair(q, d, "", &ns, &toks, &nms, &err)) {
          std::cerr << "npu rerank failed: " << err << "\n";
          return 2;
        }
        if (rep == 0) continue;
        worst = std::max(worst, std::fabs(double(hs) - ns));
        std::printf("tokens=%zu npu_ms=%.1f host_ms=%.0f score npu=%.5f "
                    "host=%.5f | %.40s\n",
                    toks, nms, hms, ns, hs, d.c_str());
      }
    }
  }
  if (npu) std::printf("MAX |score_npu - score_host|=%.5f\n", worst);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: see header comment\n";
    return 2;
  }
  const std::string mode = argv[1];
  const std::string a = argv[2];
  auto has = [&](const char* f) {
    for (int i = 3; i < argc; ++i) {
      if (std::string(argv[i]) == f) return true;
    }
    return false;
  };
  if (mode == "bge-ref" && argc >= 4) return Bge(true, a, argv[3], false);
  if (mode == "bge-graph" && argc >= 4) return Bge(false, a, argv[3], has("--host"));
  if (mode == "qwen-emb" && has("--no-host")) return QwenEmbNoHost(a);
  if (mode == "qwen-emb") {
    return QwenEmb(a, argc >= 4 && argv[3][0] != '-' ? argv[3] : "",
                   has("--host-only"));
  }
  if (mode == "qwen-rerank") return QwenRerank(a, has("--host-only"));
  std::cerr << "unknown mode\n";
  return 2;
}
