#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's full-attention geometry: 24 query heads over 2 KV heads,
// head_dim 256, 4-token selection blocks.
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;
// The per-token kernel accumulates in FP32 over FP16 keys/values; its gated
// output (order one) stays within this of an FP64 evaluation.
constexpr double kFp64Limit = 1e-6;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    CheckHip(hipMemset(allocation, 0, bytes()), "hipMemset");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

void CheckPreparation(std::uint32_t n, std::uint32_t start,
                      std::uint32_t rotary) {
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  const std::size_t count = static_cast<std::size_t>(n) * kQWidth;
  const std::size_t cache_rows = start + n + 2;
  HipBuffer<float> packed(static_cast<std::size_t>(n) * stride);
  HipBuffer<float> q_gamma(kDim), k_gamma(kDim);
  HipBuffer<float> q_ref(count), gate_ref(count), q_out(count), gate_out(count);
  HipBuffer<float> k(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<float> v(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<__half> k_ref(cache_rows * kKvWidth), v_ref(cache_rows * kKvWidth);
  HipBuffer<__half> k_out(cache_rows * kKvWidth), v_out(cache_rows * kKvWidth);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&packed, MakeValues(static_cast<std::size_t>(n) * stride, 17, 4.0F));
  auto qg = MakeValues(kDim, 37, 0.5F);
  auto kg = MakeValues(kDim, 91, 0.5F);
  for (auto& x : qg)
    x += 1.0F;
  for (auto& x : kg)
    x += 1.0F;
  Upload(&q_gamma, qg);
  Upload(&k_gamma, kg);

  struct Capture {
    hipStream_t stream{nullptr};
    hipGraph_t graph{nullptr};
    hipGraphExec_t exec{nullptr};
    Capture() {
      CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "preparation stream");
    }
    ~Capture() {
      if (exec)
        (void)hipGraphExecDestroy(exec);
      if (graph)
        (void)hipGraphDestroy(graph);
      (void)hipStreamDestroy(stream);
    }
  } capture;
  const auto same = [&](auto* expected, auto* actual, std::size_t size,
                        const char* name, std::uint32_t replay) {
    using T = std::remove_pointer_t<decltype(expected)>;
    std::vector<T> a(size), b(size);
    CheckHip(
        hipMemcpy(a.data(), expected, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation reference");
    CheckHip(
        hipMemcpy(b.data(), actual, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation result");
    for (std::size_t i = 0; i < size; ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(T)) == 0)
        continue;
      std::cerr << "preparation n=" << n << " start=" << start
                << " rotary=" << rotary << " replay=" << replay
                << " index=" << i << " expected=" << std::hexfloat
                << static_cast<float>(a[i])
                << " actual=" << static_cast<float>(b[i]) << std::defaultfloat
                << '\n';
      throw std::runtime_error(std::string("attention preparation changed ") +
                               name);
    }
  };
  for (std::uint32_t replay = 0; replay < 2; ++replay) {
    const std::vector<std::uint32_t> position{start + replay};
    Upload(&pos, position);
    q::UnpackQGate(packed.get(), stride, q_ref.get(), gate_ref.get(), k.get(),
                   v.get(), n, kHeads, kDim, kKvWidth, nullptr);
    q::RmsNormRows(q_ref.get(), q_gamma.get(), q_ref.get(), n * kHeads, kDim, 1,
                   1e-6F, nullptr);
    q::RmsNormRows(k.get(), k_gamma.get(), k.get(), n * kKvHeads, kDim, 1,
                   1e-6F, nullptr);
    q::Rope(q_ref.get(), n, kHeads, kDim, rotary, pos.get(), 1e7F, nullptr);
    q::Rope(k.get(), n, kKvHeads, kDim, rotary, pos.get(), 1e7F, nullptr);
    q::StoreKv(k.get(), k_ref.get(), n, kKvWidth, pos.get(), nullptr);
    q::StoreKv(v.get(), v_ref.get(), n, kKvWidth, pos.get(), nullptr);
    CheckHip(hipDeviceSynchronize(), "preparation reference ready");
    if (replay == 0) {
      CheckHip(
          hipStreamBeginCapture(capture.stream, hipStreamCaptureModeGlobal),
          "preparation capture");
      const bool supported = q::PrepareAttention(
          packed.get(), stride, q_gamma.get(), k_gamma.get(), q_out.get(),
          gate_out.get(), k_out.get(), v_out.get(), n, kHeads, kKvHeads, kDim,
          rotary, pos.get(), 1e7F, 1e-6F, capture.stream);
      CheckHip(hipStreamEndCapture(capture.stream, &capture.graph),
               "preparation capture end");
      if (!supported)
        throw std::runtime_error("attention preparation rejected geometry");
      CheckHip(hipGraphInstantiate(&capture.exec, capture.graph, nullptr,
                                   nullptr, 0),
               "preparation instantiate");
    }
    CheckHip(hipGraphLaunch(capture.exec, capture.stream),
             "preparation replay");
    CheckHip(hipStreamSynchronize(capture.stream), "preparation ready");
    same(q_ref.get(), q_out.get(), count, "queries", replay);
    same(gate_ref.get(), gate_out.get(), count, "gates", replay);
    // Include the neighboring cache rows and replay at a new device position.
    const std::size_t first = start == 0 ? 0 : start - 1;
    const std::size_t window = (cache_rows - first) * kKvWidth;
    same(k_ref.get() + first * kKvWidth, k_out.get() + first * kKvWidth, window,
         "key cache", replay);
    same(v_ref.get() + first * kKvWidth, v_out.get() + first * kKvWidth, window,
         "value cache", replay);
  }
  if (n >= 32) {
    // Each small prefill piece must match the full preparation, including
    // normalization rounding and absolute rotary positions.
    for (const std::uint32_t chunk : {1U, 8U, 9U, 16U, 32U}) {
      for (std::uint32_t off = 0; off < n; off += chunk) {
        Upload(&pos, std::vector<std::uint32_t>{start + 1 + off});
        if (!q::PrepareAttention(
                packed.get() + std::size_t(off) * stride, stride, q_gamma.get(),
                k_gamma.get(), q_out.get() + std::size_t(off) * kQWidth,
                gate_out.get() + std::size_t(off) * kQWidth, k_out.get(),
                v_out.get(), std::min(chunk, n - off), kHeads, kKvHeads, kDim,
                rotary, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, true))
          throw std::runtime_error("prefill preparation rejected a short tail");
      }
      CheckHip(hipDeviceSynchronize(), "short prefill preparation");
      same(q_ref.get(), q_out.get(), count, "prefill queries", chunk);
      same(gate_ref.get(), gate_out.get(), count, "prefill gates", chunk);
      same(k_ref.get(), k_out.get(), cache_rows * kKvWidth, "prefill keys",
           chunk);
      same(v_ref.get(), v_out.get(), cache_rows * kKvWidth, "prefill values",
           chunk);
    }
  }
  std::cout << "attention preparation n=" << n << " start=" << start
            << " rotary=" << rotary << ": exact, including graph positions\n";
}

/// FP64 gated attention over the keys each row can see: every key below
/// pos + 1, or with a mask the selected complete blocks plus the incomplete
/// tail block. An independent formula for the per-token kernel.
std::vector<double> Fp64Attention(
    const std::vector<float>& qv, const std::vector<float>& gate,
    const std::vector<__half>& kh, const std::vector<__half>& vh,
    const std::uint32_t* mask, std::uint32_t mask_words, std::uint32_t n_tokens,
    std::uint32_t start_pos, std::uint32_t ratio = kRatio) {
  std::vector<double> out(static_cast<std::size_t>(n_tokens) * kQWidth);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  std::vector<std::uint32_t> keys;
  std::vector<double> scores;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t n_kv = start_pos + t + 1;
    const std::uint32_t tail_start = n_kv / ratio * ratio;
    keys.clear();
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const std::uint32_t b = j / ratio;
      if (mask == nullptr || j >= tail_start ||
          ((mask[static_cast<std::size_t>(t) * mask_words + b / 32] >>
            (b % 32)) &
           1U) != 0) {
        keys.push_back(j);
      }
    }
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::size_t row = (static_cast<std::size_t>(t) * kHeads + h) * kDim;
      const std::size_t kv_head = (h / (kHeads / kKvHeads)) * kDim;
      scores.assign(keys.size(), 0.0);
      double maximum = -INFINITY;
      for (std::size_t k = 0; k < keys.size(); ++k) {
        const std::size_t base = keys[k] * std::size_t{kKvWidth} + kv_head;
        double dot = 0.0;
        for (std::uint32_t d = 0; d < kDim; ++d) {
          dot += static_cast<double>(qv[row + d]) *
                 static_cast<double>(__half2float(kh[base + d]));
        }
        scores[k] = dot * scale;
        maximum = std::max(maximum, scores[k]);
      }
      double sum = 0.0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
      }
      for (std::uint32_t d = 0; d < kDim; ++d) {
        double acc = 0.0;
        for (std::size_t k = 0; k < keys.size(); ++k) {
          acc += scores[k] *
                 static_cast<double>(__half2float(
                     vh[keys[k] * std::size_t{kKvWidth} + kv_head + d]));
        }
        const double g = static_cast<double>(gate[row + d]);
        out[row + d] = acc / sum / (1.0 + std::exp(-g));
      }
    }
  }
  return out;
}

/// Runs the per-token reference and the fused WMMA route over the same
/// cache and reports the worst absolute error of the gated context.
double Compare(std::uint32_t n_tokens, std::uint32_t start_pos, bool masked,
               std::uint32_t seed, bool sparse = false, bool bounded = false) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;

  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  for (std::size_t i = 0; i < kv_count; ++i) {
    kh[i] = __float2half(kf[i]);
    vh[i] = __float2half(vf[i]);
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::uint32_t state = seed ^ 0x77777777U;
  for (std::uint32_t& word : mask) {
    word = NextRandom(&state) & NextRandom(&state);
    if (sparse) {
      // About one block in sixteen: most 16-key tiles are unselected for a
      // whole query block, which is what the tile skipping is for.
      word &= NextRandom(&state) & NextRandom(&state);
    }
  }
  if (bounded) {
    std::fill(mask.begin(), mask.end(), 0);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto complete = (start_pos + t + 1) / kRatio;
      std::uint32_t selected = 0;
      while (selected < std::min(512U, complete)) {
        const auto block = NextRandom(&state) % complete;
        auto& word =
            mask[static_cast<std::size_t>(t) * mask_words + block / 32];
        const auto bit = 1U << (block % 32);
        if ((word & bit) == 0) {
          word |= bit;
          ++selected;
        }
      }
    }
  }

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_ref(q_count);
  HipBuffer<float> d_wmma(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  const std::uint32_t* mask_ptr = masked ? d_mask.get() : nullptr;

  q::Attention(d_q.get(), d_k.get(), d_v.get(), mask_ptr, mask_words,
               d_ref.get(), nullptr, 1, n_tokens, d_pos.get(), kHeads, kKvHeads,
               kDim, kRatio, nullptr);
  q::SigmoidMul(d_ref.get(), d_gate.get(), q_count, nullptr);
  // The split-key form of the reference must match its single-block form
  // (log-sum-exp merge), the way decode runs it.
  {
    constexpr std::uint32_t kSplits = 8;
    HipBuffer<float> d_split(q_count);
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                kSplits * (kDim + 2));
    q::Attention(d_q.get(), d_k.get(), d_v.get(), mask_ptr, mask_words,
                 d_split.get(), d_partials.get(), kSplits, n_tokens,
                 d_pos.get(), kHeads, kKvHeads, kDim, kRatio, nullptr);
    q::SigmoidMul(d_split.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "split attention");
    const auto a = Download(&d_ref, q_count);
    const auto b = Download(&d_split, q_count);
    double worst = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(a[i] - b[i])));
    }
    std::cout << "  split-key reference worst absolute error " << worst << '\n';
    const auto exact =
        Fp64Attention(qv, gate, kh, vh, masked ? mask.data() : nullptr,
                      mask_words, n_tokens, start_pos);
    double single_error = 0.0;
    double split_error = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      single_error = std::max(single_error, std::abs(a[i] - exact[i]));
      split_error = std::max(split_error, std::abs(b[i] - exact[i]));
    }
    std::cout << "  FP64 worst absolute error single " << single_error
              << " split " << split_error << '\n';
    if (std::max(single_error, split_error) > kFp64Limit) {
      throw std::runtime_error("per-token attention exceeds its FP64 limit");
    }
    if (worst > 1e-4) {
      throw std::runtime_error("split-key attention disagrees");
    }
  }
  if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                              mask_ptr, mask_words, d_wmma.get(), n_tokens,
                              start_pos, kHeads, kKvHeads, kDim, kRatio,
                              nullptr)) {
    throw std::runtime_error("WMMA attention rejected the model geometry");
  }
  CheckHip(hipDeviceSynchronize(), "attention synchronization");

  const auto ref = Download(&d_ref, q_count);
  const auto out = Download(&d_wmma, q_count);
  if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                              mask_ptr, mask_words, d_wmma.get(), n_tokens,
                              start_pos, kHeads, kKvHeads, kDim, kRatio,
                              nullptr)) {
    throw std::runtime_error("WMMA attention replay rejected the geometry");
  }
  const auto replay = Download(&d_wmma, q_count);
  if (std::memcmp(out.data(), replay.data(), q_count * sizeof(float)) != 0) {
    throw std::runtime_error("WMMA attention replay changed the output");
  }
  {
    constexpr float poison = -12345.0F;
    Upload(&d_wmma, std::vector<float>(q_count, poison));
    if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                                mask_ptr, mask_words, d_wmma.get(), n_tokens,
                                start_pos, kHeads, kKvHeads, kDim, kRatio,
                                nullptr, true)) {
      throw std::runtime_error("last-tile attention rejected the geometry");
    }
    const auto tail = Download(&d_wmma, q_count);
    const std::uint32_t tile_rows = mask_ptr ? 4 : 16;
    const std::size_t begin =
        static_cast<std::size_t>((n_tokens - 1) / tile_rows * tile_rows) *
        kQWidth;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (i < begin ? tail[i] != poison
                    : std::memcmp(&out[i], &tail[i], sizeof(float)) != 0) {
        throw std::runtime_error("last-tile attention changed a row or guard");
      }
    }
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(out[i])) {
      throw std::runtime_error("WMMA attention output is not finite");
    }
    worst = std::max(worst, std::abs(static_cast<double>(ref[i] - out[i])));
  }
  if (worst > 1e-2) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        double w = 0.0;
        for (std::uint32_t d = 0; d < kDim; ++d) {
          const std::size_t i =
              (static_cast<std::size_t>(t) * kHeads + h) * kDim + d;
          w = std::max(w, std::abs(static_cast<double>(ref[i] - out[i])));
        }
        if (w > 1e-2) {
          std::cout << "  query " << t << " head " << h << " err " << w << '\n';
        }
      }
    }
  }
  return worst;
}

/// Selected blocks for one query row, given its complete-block count.
using BlockPattern =
    std::function<void(std::uint32_t, std::vector<std::uint32_t>*)>;

/// The per-token kernel against FP64 on a structured selection, in its
/// single-block form and split 1, 3 and 8 ways. The masks are built so the
/// sparse path's 1024-block compaction windows and 64-block key tiles meet
/// their edge cases: tiles that straddle windows, a carried partial tile of
/// every size, a single partial tile, and a ninth tile that wraps a split.
void CheckStructured(const char* name, std::uint32_t n_tokens,
                     std::uint32_t start_pos, const BlockPattern& pattern,
                     std::uint32_t seed, std::uint32_t ratio = kRatio) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + ratio - 1) / ratio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  {
    const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
    const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
    for (std::size_t i = 0; i < kv_count; ++i) {
      kh[i] = __float2half(kf[i]);
      vh[i] = __float2half(vf[i]);
    }
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::vector<std::uint32_t> blocks;
  std::size_t selected = 0;
  std::size_t most = 1;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start_pos + t + 1) / ratio;
    blocks.clear();
    pattern(complete, &blocks);
    for (const std::uint32_t b : blocks) {
      if (b >= complete) {
        throw std::runtime_error("structured mask selects past the context");
      }
      mask[static_cast<std::size_t>(t) * mask_words + b / 32] |= 1U << (b % 32);
    }
    selected += blocks.size();
    most = std::max(most, blocks.size());
  }
  const auto exact = Fp64Attention(qv, gate, kh, vh, mask.data(), mask_words,
                                   n_tokens, start_pos, ratio);
  // The list path (ratio 4 only) is compared bitwise with the mask walk. The
  // adversarial patterns select more than the indexer's 512-block budget, so
  // the lists get room for the largest row: that keeps every carry and window
  // edge in play, and the capacity guard is not what is being tested here.
  const std::uint32_t list_capacity = static_cast<std::uint32_t>(most);
  HipBuffer<std::uint32_t> d_lists(static_cast<std::size_t>(n_tokens) *
                                   list_capacity);
  HipBuffer<std::uint32_t> d_counts(n_tokens);

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_out(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  std::cout << "structured " << name << " ratio=" << ratio << " n=" << n_tokens
            << " start=" << start_pos << " blocks/row=" << selected / n_tokens
            << " FP64 worst absolute error";
  double worst = 0.0;
  for (const std::uint32_t splits : {0U, 1U, 3U, 8U}) {
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                std::max(splits, 1U) * (kDim + 2));
    q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                 d_out.get(), splits == 0 ? nullptr : d_partials.get(), splits,
                 n_tokens, d_pos.get(), kHeads, kKvHeads, kDim, ratio, nullptr);
    q::SigmoidMul(d_out.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "structured attention");
    const auto out = Download(&d_out, q_count);
    if (ratio == kRatio) {
      HipBuffer<float> d_list(q_count);
      q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                   d_list.get(), splits == 0 ? nullptr : d_partials.get(),
                   splits, n_tokens, d_pos.get(), kHeads, kKvHeads, kDim, ratio,
                   nullptr, d_lists.get(), d_counts.get(), list_capacity);
      q::SigmoidMul(d_list.get(), d_gate.get(), q_count, nullptr);
      CheckHip(hipDeviceSynchronize(), "structured list attention");
      const auto list = Download(&d_list, q_count);
      if (std::memcmp(out.data(), list.data(), q_count * sizeof(float)) != 0) {
        throw std::runtime_error("structured list attention differs from walk");
      }
    }
    double error = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (!std::isfinite(out[i])) {
        throw std::runtime_error("structured attention output is not finite");
      }
      error = std::max(error, std::abs(out[i] - exact[i]));
    }
    std::cout << (splits == 0 ? " single "
                              : " split" + std::to_string(splits) + " ")
              << error;
    worst = std::max(worst, error);
  }
  std::cout << '\n';
  if (worst > kFp64Limit) {
    throw std::runtime_error("structured attention exceeds its FP64 limit");
  }
}

/// The compacted-selection route against the mask walk. Random masks of at
/// most the 512-block budget per row (a short context selects every block):
/// the outputs must match bit for bit, single-block and split eight ways, and
/// stay within the FP64 limit.
void CheckLists(std::uint32_t n_tokens, std::uint32_t start_pos,
                std::uint32_t seed) {
  constexpr std::uint32_t kBudget = 512;
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  {
    const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
    const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
    for (std::size_t i = 0; i < kv_count; ++i) {
      kh[i] = __float2half(kf[i]);
      vh[i] = __float2half(vf[i]);
    }
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::uint32_t state = seed ^ 0x77777777U;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start_pos + t + 1) / kRatio;
    std::uint32_t* row = mask.data() + static_cast<std::size_t>(t) * mask_words;
    if (complete <= kBudget) {
      // The selector marks every word of a row that fits its budget.
      std::fill(row, row + mask_words, 0xFFFFFFFFU);
      continue;
    }
    // Rows alternate between the full budget and a random smaller count.
    const std::uint32_t want =
        t % 2 == 0 ? kBudget : 1 + NextRandom(&state) % kBudget;
    for (std::uint32_t selected = 0; selected < want;) {
      const auto block = NextRandom(&state) % complete;
      if ((row[block / 32] & (1U << (block % 32))) == 0) {
        row[block / 32] |= 1U << (block % 32);
        ++selected;
      }
    }
  }
  const auto exact = Fp64Attention(qv, gate, kh, vh, mask.data(), mask_words,
                                   n_tokens, start_pos);

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<std::uint32_t> d_lists(static_cast<std::size_t>(n_tokens) *
                                   kBudget);
  HipBuffer<std::uint32_t> d_counts(n_tokens);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  std::cout << "lists n=" << n_tokens << " start=" << start_pos;
  for (const std::uint32_t splits : {0U, 8U}) {
    const std::uint32_t z = std::max(splits, 1U);
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                z * (kDim + 2));
    HipBuffer<float> d_walk(q_count);
    HipBuffer<float> d_list(q_count);
    float* partials = splits == 0 ? nullptr : d_partials.get();
    q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                 d_walk.get(), partials, splits, n_tokens, d_pos.get(), kHeads,
                 kKvHeads, kDim, kRatio, nullptr);
    q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                 d_list.get(), partials, splits, n_tokens, d_pos.get(), kHeads,
                 kKvHeads, kDim, kRatio, nullptr, d_lists.get(), d_counts.get(),
                 kBudget);
    q::SigmoidMul(d_walk.get(), d_gate.get(), q_count, nullptr);
    q::SigmoidMul(d_list.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "list attention");
    const auto walk = Download(&d_walk, q_count);
    const auto list = Download(&d_list, q_count);
    if (std::memcmp(walk.data(), list.data(), q_count * sizeof(float)) != 0) {
      throw std::runtime_error("list attention differs from the mask walk");
    }
    double error = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (!std::isfinite(list[i])) {
        throw std::runtime_error("list attention output is not finite");
      }
      error = std::max(error, std::abs(list[i] - exact[i]));
    }
    std::cout << (splits == 0 ? " single " : " split8 ") << error;
    if (error > kFp64Limit) {
      throw std::runtime_error("list attention exceeds its FP64 limit");
    }
  }
  std::cout << " bitwise-equal\n";
}

void CheckChunks(std::uint32_t n, std::uint32_t split) {
  const std::size_t count = std::size_t(n) * kQWidth;
  HipBuffer<float> queries(count), gates(count), full(count), chunked(count);
  HipBuffer<__half> keys(std::size_t(n) * kKvWidth),
      values(std::size_t(n) * kKvWidth);
  Upload(&queries, MakeValues(count, 412, 4.0F));
  Upload(&gates, MakeValues(count, 721, 3.0F));
  for (auto* destination : {&keys, &values}) {
    const auto f = MakeValues(std::size_t(n) * kKvWidth,
                              destination == &keys ? 891 : 347, 1.0F);
    std::vector<__half> half(f.size());
    std::transform(f.begin(), f.end(), half.begin(),
                   [](float x) { return __float2half(x); });
    Upload(destination, half);
  }
  const auto run = [&](std::uint32_t start, std::uint32_t rows, float* out) {
    const auto offset = std::size_t(start) * kQWidth;
    if (!q::WmmaCausalAttention(queries.get() + offset, gates.get() + offset,
                                keys.get(), values.get(), nullptr, 0,
                                out + offset, rows, start, kHeads, kKvHeads,
                                kDim, kRatio, nullptr)) {
      throw std::runtime_error("chunk attention rejected geometry");
    }
  };
  run(0, n, full.get());
  run(0, split, chunked.get());
  run(split, n - split, chunked.get());
  const auto a = Download(&full, count);
  const auto b = Download(&chunked, count);
  std::size_t differences = 0, first = count;
  float max_error = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (a[i] != b[i]) {
      ++differences;
      first = std::min(first, i);
      max_error = std::max(max_error, std::abs(a[i] - b[i]));
    }
  }
  std::cout << "chunk attention n=" << n << " split=" << split
            << " differing=" << differences << " max=" << max_error
            << " first_row=" << first / kQWidth << '\n';
  if (differences != 0)
    throw std::runtime_error("attention depends on prefill chunk boundary");
}

/// Mixed-batch varlen check: session A contributes one decode row (one
/// query over an 8-key prefix), session B a 4-token prefill chunk over a
/// 16-key prefix with causal self-attention. Each side is verified against
/// its own dense Fp64Attention; the kernel computes both in one launch.
double CompareMixed(std::uint32_t seed) {
  constexpr std::uint32_t kChunk = 4;
  constexpr std::uint32_t kKvA = 35;
  constexpr std::uint32_t kKvB = 52;
  constexpr std::uint32_t kRows = 1 + kChunk;
  const std::size_t q_count = static_cast<std::size_t>(kRows) * kQWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kfA = MakeValues(static_cast<std::size_t>(kKvA) * kKvWidth,
                              seed ^ 0xAAAAU, 1.0F);
  const auto vfA = MakeValues(static_cast<std::size_t>(kKvA) * kKvWidth,
                              seed ^ 0x3333U, 1.0F);
  const auto kfB = MakeValues(static_cast<std::size_t>(kKvB) * kKvWidth,
                              seed ^ 0xBBBBU, 1.0F);
  const auto vfB = MakeValues(static_cast<std::size_t>(kKvB) * kKvWidth,
                              seed ^ 0xCCCCU, 1.0F);
  std::vector<__half> khA(kfA.size()), vhA(vfA.size()), khB(kfB.size()),
      vhB(vfB.size());
  for (std::size_t i = 0; i < kfA.size(); ++i) {
    khA[i] = __float2half(kfA[i]);
    vhA[i] = __float2half(vfA[i]);
  }
  for (std::size_t i = 0; i < kfB.size(); ++i) {
    khB[i] = __float2half(kfB[i]);
    vhB[i] = __float2half(vfB[i]);
  }
  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_kA(kfA.size());
  HipBuffer<__half> d_kB(kfB.size());
  HipBuffer<__half> d_vA(vfA.size());
  HipBuffer<__half> d_vB(vfB.size());
  HipBuffer<float> d_out(q_count);
  HipBuffer<q::MixedBlk> d_blks(2);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_kA, khA);
  Upload(&d_vA, vhA);
  Upload(&d_kB, khB);
  Upload(&d_vB, vhB);
  std::vector<q::MixedBlk> blks(2);
  blks[0] = {0, reinterpret_cast<std::uint64_t>(d_kA.get()),
             reinterpret_cast<std::uint64_t>(d_vA.get()), kKvA,
             kKvA - 1, 1};
  blks[1] = {1, reinterpret_cast<std::uint64_t>(d_kB.get()),
             reinterpret_cast<std::uint64_t>(d_vB.get()), kKvB,
             kKvB - kChunk, kChunk};
  Upload(&d_blks, blks);
  if (!q::WmmaMixedAttention(d_q.get(), d_gate.get(), d_blks.get(), 2,
                             d_out.get(), kHeads, kKvHeads, kDim, kRatio,
                             q::kKvKeyBlockBytes, 0, nullptr)) {
    throw std::runtime_error("mixed attention rejected the geometry");
  }
  CheckHip(hipDeviceSynchronize(), "mixed attention synchronization");
  const auto out = Download(&d_out, q_count);
  // Reference: row 0 is A's decode token (absolute query kKvA - 1 over A's
  // own cache); rows 1..4 are B's chunk (absolute queries kKvB - kChunk + r).
  const auto refA = Fp64Attention(
      std::vector<float>(qv.begin(), qv.begin() + kQWidth),
      std::vector<float>(gate.begin(), gate.begin() + kQWidth), khA, vhA,
      nullptr, 0, 1, kKvA - 1);
  const auto refB = Fp64Attention(
      std::vector<float>(qv.begin() + kQWidth, qv.end()),
      std::vector<float>(gate.begin() + kQWidth, gate.end()), khB, vhB,
      nullptr, 0, kChunk, kKvB - kChunk);
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    const double e = (i < static_cast<std::size_t>(kQWidth))
                         ? refA[i]
                         : refB[i - kQWidth];
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - e));
  }
  std::cout << "  mixed attention worst absolute error " << worst << '\n';
  return worst;
}

double CompareMixedI8(std::uint32_t seed) {
  // Same geometry as CompareMixed, but K arrives as true 34-byte blocked
  // int8 cache (per-32-dim absmax scale + codes) through the int8 WMMA
  // path (i8=true); V stays F16 staging like the production batch mixed
  // route. The reference dequantizes the same blocks, so any excess over
  // the F16 envelope is the int8 path's own arithmetic.
  constexpr std::uint32_t kChunk = 4;
  constexpr std::uint32_t kKvA = 35;
  constexpr std::uint32_t kKvB = 52;
  constexpr std::uint32_t kRows = 1 + kChunk;
  constexpr std::uint32_t kBlkBytes = 34;
  constexpr std::uint32_t kBlksPerRow = kKvWidth / 32;
  const std::size_t q_count = static_cast<std::size_t>(kRows) * kQWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kfA = MakeValues(static_cast<std::size_t>(kKvA) * kKvWidth,
                              seed ^ 0xAAAAU, 1.0F);
  const auto vfA = MakeValues(static_cast<std::size_t>(kKvA) * kKvWidth,
                              seed ^ 0x3333U, 1.0F);
  const auto kfB = MakeValues(static_cast<std::size_t>(kKvB) * kKvWidth,
                              seed ^ 0xBBBBU, 1.0F);
  const auto vfB = MakeValues(static_cast<std::size_t>(kKvB) * kKvWidth,
                              seed ^ 0xCCCCU, 1.0F);
  std::vector<__half> vhA(vfA.size()), vhB(vfB.size());
  std::vector<std::uint8_t> vbA(static_cast<std::size_t>(kKvA) * kBlksPerRow *
                                18),
      vbB(static_cast<std::size_t>(kKvB) * kBlksPerRow * 18);
  const auto pack4 = [&](const std::vector<float>& vf, std::vector<__half>& vh,
                         std::vector<std::uint8_t>& vb, std::uint32_t keys) {
    for (std::uint32_t k = 0; k < keys; ++k) {
      for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
        float m = 0.0F;
        for (std::uint32_t i = 0; i < 32; ++i) {
          m = fmaxf(m, fabsf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                                (b * 32) + i]));
        }
        const float d = m / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        std::uint8_t* blk =
            &vb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 18];
        *reinterpret_cast<__half*>(blk + 16) = __float2half(d);
        for (std::uint32_t i = 0; i < 32; i += 2) {
          const float x0 =
              vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
          const float x1 =
              vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1];
          int q0 = static_cast<int>(static_cast<std::int8_t>(roundf(x0 * id)));
          int q1 = static_cast<int>(static_cast<std::int8_t>(roundf(x1 * id)));
          q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
          q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
          blk[i / 2] = static_cast<std::uint8_t>((q0 & 0xF) | (q1 << 4));
          const auto v0 = static_cast<float>(q0) * d;
          const auto v1 = static_cast<float>(q1) * d;
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
              __float2half(v0);
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1] =
              __float2half(v1);
        }
      }
    }
  };
  pack4(vfA, vhA, vbA, kKvA);
  pack4(vfB, vhB, vbB, kKvB);
  // Pack K into 34-byte blocks; keep the dequantized floats for reference.
  std::vector<std::uint8_t> kbA(static_cast<std::size_t>(kKvA) * kBlksPerRow *
                                kBlkBytes),
      kbB(static_cast<std::size_t>(kKvB) * kBlksPerRow * kBlkBytes);
  std::vector<__half> khA(kfA.size()), khB(kfB.size());
  const auto pack = [&](const std::vector<float>& kf, std::vector<__half>& kh,
                        std::vector<std::uint8_t>& kb, std::uint32_t keys) {
    for (std::uint32_t k = 0; k < keys; ++k) {
      for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
        float m = 0.0F;
        for (std::uint32_t i = 0; i < 32; ++i) {
          m = fmaxf(m, fabsf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                                (b * 32) + i]));
        }
        const float d = m / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        std::uint8_t* blk =
            &kb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * kBlkBytes];
        // Production blocked layout: 32 codes at +0, f16 scale at +32.
        *reinterpret_cast<__half*>(blk + 32) = __float2half(d);
        for (std::uint32_t i = 0; i < 32; ++i) {
          const float x =
              kf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
          const auto c = static_cast<std::int8_t>(roundf(x * id));
          blk[i] = static_cast<std::uint8_t>(c);
          kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
              __float2half(static_cast<float>(c) * d);
        }
      }
    }
  };
  pack(kfA, khA, kbA, kKvA);
  pack(kfB, khB, kbB, kKvB);
  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<std::uint8_t> d_kA(kbA.size());
  HipBuffer<std::uint8_t> d_kB(kbB.size());
  HipBuffer<std::uint8_t> d_vA(vbA.size());
  HipBuffer<std::uint8_t> d_vB(vbB.size());
  HipBuffer<float> d_out(q_count);
  HipBuffer<q::MixedBlk> d_blks(2);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_kA, kbA);
  Upload(&d_kB, kbB);
  Upload(&d_vA, vbA);
  Upload(&d_vB, vbB);
  std::vector<q::MixedBlk> blks(2);
  blks[0] = {0, reinterpret_cast<std::uint64_t>(d_kA.get()),
             reinterpret_cast<std::uint64_t>(d_vA.get()), kKvA,
             kKvA - 1, 1};
  blks[1] = {1, reinterpret_cast<std::uint64_t>(d_kB.get()),
             reinterpret_cast<std::uint64_t>(d_vB.get()), kKvB,
             kKvB - kChunk, kChunk};
  Upload(&d_blks, blks);
  if (!q::WmmaMixedAttention(d_q.get(), d_gate.get(), d_blks.get(), 2,
                             d_out.get(), kHeads, kKvHeads, kDim, kRatio,
                             q::kKvKeyBlockBytes, 18, nullptr, true)) {
    throw std::runtime_error("mixed attention (i8) rejected the geometry");
  }
  CheckHip(hipDeviceSynchronize(), "mixed attention (i8) synchronization");
  const auto out = Download(&d_out, q_count);
  const auto refA = Fp64Attention(
      std::vector<float>(qv.begin(), qv.begin() + kQWidth),
      std::vector<float>(gate.begin(), gate.begin() + kQWidth), khA, vhA,
      nullptr, 0, 1, kKvA - 1);
  const auto refB = Fp64Attention(
      std::vector<float>(qv.begin() + kQWidth, qv.end()),
      std::vector<float>(gate.begin() + kQWidth, gate.end()), khB, vhB,
      nullptr, 0, kChunk, kKvB - kChunk);
  // Host simulation of the kernel's own per-32-dim Q quantization.
  auto qquant = [&](const std::vector<float>& qin) {
    std::vector<float> qout(qin.size());
    for (std::size_t r = 0; r < qin.size() / kDim; ++r) {
      for (std::uint32_t b = 0; b < kDim / 32; ++b) {
        float m = 0.0F;
        for (std::uint32_t i = 0; i < 32; ++i) {
          m = fmaxf(m, fabsf(qin[r * kDim + b * 32 + i]));
        }
        const float d = m / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        for (std::uint32_t i = 0; i < 32; ++i) {
          const float x = qin[r * kDim + b * 32 + i];
          qout[r * kDim + b * 32 + i] =
              static_cast<float>(static_cast<std::int8_t>(roundf(x * id))) *
              d;
        }
      }
    }
    return qout;
  };
  const auto qA = qquant(std::vector<float>(qv.begin(), qv.begin() + kQWidth));
  const auto qB = qquant(std::vector<float>(qv.begin() + kQWidth, qv.end()));
  const auto refAq = Fp64Attention(
      qA, std::vector<float>(gate.begin(), gate.begin() + kQWidth), khA, vhA,
      nullptr, 0, 1, kKvA - 1);
  const auto refBq = Fp64Attention(
      qB, std::vector<float>(gate.begin() + kQWidth, gate.end()), khB, vhB,
      nullptr, 0, kChunk, kKvB - kChunk);
  double worst = 0.0, worst_q = 0.0, worst_qq = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    const bool first = (i < static_cast<std::size_t>(kQWidth));
    const double e = first ? refA[i] : refB[i - kQWidth];
    const double eq = first ? refAq[i] : refBq[i - kQWidth];
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - e));
    worst_q = std::max(worst_q, std::abs(static_cast<double>(out[i]) - eq));
    worst_qq = std::max(worst_qq, std::abs(e - eq));
  }
  std::cout << "  mixed attention (i8 K) worst absolute error " << worst
            << " vs-Qquant-ref " << worst_q << " pure-Qquant " << worst_qq
            << '\n';
  return worst;
}



double CompareMixedI8Deep(std::uint32_t seed) {
  // Production-fidelity: deep context decode + outlier spikes.
  constexpr std::uint32_t kChunk = 4;
  constexpr std::uint32_t kKvA = 512;
  constexpr std::uint32_t kKvB = 517;
  constexpr std::uint32_t kFull = 16;
  constexpr std::uint32_t kRows = 1 + kChunk + kFull;
  constexpr std::uint32_t kBlkBytes = 34;
  constexpr std::uint32_t kBlksPerRow = kKvWidth / 32;
  const std::size_t q_count = static_cast<std::size_t>(kRows) * kQWidth;
  auto qv = MakeValues(q_count, seed, 4.0F);
  auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  auto kfA = MakeValues(static_cast<std::size_t>(kKvA) * kKvWidth,
                        seed ^ 0xAAAAU, 1.0F);
  auto vfA = MakeValues(static_cast<std::size_t>(kKvA) * kKvWidth,
                        seed ^ 0x3333U, 1.0F);
  auto kfB = MakeValues(static_cast<std::size_t>(kKvB) * kKvWidth,
                        seed ^ 0xBBBBU, 1.0F);
  auto vfB = MakeValues(static_cast<std::size_t>(kKvB) * kKvWidth,
                        seed ^ 0xCCCCU, 1.0F);
  // Outlier spikes: every 97th Q element x13, every 131st V element x9.
  for (std::size_t i = 0; i < q_count; i += 97) qv[i] *= 13.0F;
  for (std::size_t i = 0; i < vfA.size(); i += 131) vfA[i] *= 9.0F;
  for (std::size_t i = 0; i < vfB.size(); i += 131) vfB[i] *= 9.0F;
  std::vector<__half> vhA(vfA.size()), vhB(vfB.size());
  std::vector<std::uint8_t> vbA(static_cast<std::size_t>(kKvA) * kBlksPerRow *
                                18),
      vbB(static_cast<std::size_t>(kKvB) * kBlksPerRow * 18);
  const auto pack4 = [&](const std::vector<float>& vf, std::vector<__half>& vh,
                         std::vector<std::uint8_t>& vb, std::uint32_t keys) {
    for (std::uint32_t k = 0; k < keys; ++k) {
      for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
        float m = 0.0F;
        for (std::uint32_t i = 0; i < 32; ++i) {
          m = fmaxf(m, fabsf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                                (b * 32) + i]));
        }
        const float d = m / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        std::uint8_t* blk =
            &vb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 18];
        *reinterpret_cast<__half*>(blk + 16) = __float2half(d);
        for (std::uint32_t i = 0; i < 32; i += 2) {
          const float x0 =
              vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
          const float x1 =
              vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1];
          int q0 = static_cast<int>(static_cast<std::int8_t>(roundf(x0 * id)));
          int q1 = static_cast<int>(static_cast<std::int8_t>(roundf(x1 * id)));
          q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
          q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
          blk[i / 2] = static_cast<std::uint8_t>((q0 & 0xF) | (q1 << 4));
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
              __float2half(static_cast<float>(q0) * d);
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1] =
              __float2half(static_cast<float>(q1) * d);
        }
      }
    }
  };
  pack4(vfA, vhA, vbA, kKvA);
  pack4(vfB, vhB, vbB, kKvB);
  std::vector<std::uint8_t> kbA(static_cast<std::size_t>(kKvA) * kBlksPerRow *
                                kBlkBytes),
      kbB(static_cast<std::size_t>(kKvB) * kBlksPerRow * kBlkBytes);
  std::vector<__half> khA(kfA.size()), khB(kfB.size());
  const auto pack = [&](const std::vector<float>& kf, std::vector<__half>& kh,
                        std::vector<std::uint8_t>& kb, std::uint32_t keys) {
    for (std::uint32_t k = 0; k < keys; ++k) {
      for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
        float m = 0.0F;
        for (std::uint32_t i = 0; i < 32; ++i) {
          m = fmaxf(m, fabsf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                                (b * 32) + i]));
        }
        const float d = m / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        std::uint8_t* blk =
            &kb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * kBlkBytes];
        // Production blocked layout: 32 codes at +0, f16 scale at +32.
        *reinterpret_cast<__half*>(blk + 32) = __float2half(d);
        for (std::uint32_t i = 0; i < 32; ++i) {
          const float x =
              kf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
          const auto c = static_cast<std::int8_t>(roundf(x * id));
          blk[i] = static_cast<std::uint8_t>(c);
          kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
              __float2half(static_cast<float>(c) * d);
        }
      }
    }
  };
  pack(kfA, khA, kbA, kKvA);
  pack(kfB, khB, kbB, kKvB);
  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<std::uint8_t> d_kA(kbA.size());
  HipBuffer<std::uint8_t> d_kB(kbB.size());
  HipBuffer<std::uint8_t> d_vA(vbA.size());
  HipBuffer<std::uint8_t> d_vB(vbB.size());
  HipBuffer<float> d_out(q_count);
  HipBuffer<q::MixedBlk> d_blks(3);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_kA, kbA);
  Upload(&d_kB, kbB);
  Upload(&d_vA, vbA);
  Upload(&d_vB, vbB);
  std::vector<q::MixedBlk> blks(3);
  blks[0] = {0, reinterpret_cast<std::uint64_t>(d_kA.get()),
             reinterpret_cast<std::uint64_t>(d_vA.get()), kKvA,
             kKvA - 1, 1};
  blks[1] = {1, reinterpret_cast<std::uint64_t>(d_kB.get()),
             reinterpret_cast<std::uint64_t>(d_vB.get()), kKvB,
             kKvB - kChunk, kChunk};
  blks[2] = {1 + kChunk, reinterpret_cast<std::uint64_t>(d_kB.get()),
             reinterpret_cast<std::uint64_t>(d_vB.get()), kKvB,
             kKvB - kFull, kFull};
  Upload(&d_blks, blks);
  if (!q::WmmaMixedAttention(d_q.get(), d_gate.get(), d_blks.get(), 3,
                             d_out.get(), kHeads, kKvHeads, kDim, kRatio,
                             q::kKvKeyBlockBytes, 18, nullptr, true)) {
    throw std::runtime_error("mixed attention (i8 deep) rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "mixed attention (i8 deep) sync");
  const auto out = Download(&d_out, q_count);
  // Reference for the full-tile blk (rows 1+kChunk .. q_count).
  const auto refC = Fp64Attention(
      std::vector<float>(qv.begin() + (1 + kChunk) * kQWidth, qv.end()),
      std::vector<float>(gate.begin() + (1 + kChunk) * kQWidth, gate.end()),
      khB, vhB, nullptr, 0, kFull, kKvB - kFull);
  double worstC = 0.0;
  for (std::size_t i = (1 + kChunk) * kQWidth; i < q_count; ++i) {
    worstC = std::max(
        worstC, std::abs(static_cast<double>(out[i]) - refC[i - (1 + kChunk) * kQWidth]));
  }
  std::cout << "  mixed attention (i8 deep full-tile) worst abs err "
            << worstC << '\n';
  const auto refA = Fp64Attention(
      std::vector<float>(qv.begin(), qv.begin() + kQWidth),
      std::vector<float>(gate.begin(), gate.begin() + kQWidth), khA, vhA,
      nullptr, 0, 1, kKvA - 1);
  const auto refB = Fp64Attention(
      std::vector<float>(qv.begin() + kQWidth, qv.end()),
      std::vector<float>(gate.begin() + kQWidth, gate.end()), khB, vhB,
      nullptr, 0, kChunk, kKvB - kChunk);
  double worst = 0.0;
  const std::size_t q_head =
      static_cast<std::size_t>(1 + kChunk) * kQWidth;
  for (std::size_t i = 0; i < q_head; ++i) {
    const double e = (i < static_cast<std::size_t>(kQWidth))
                         ? refA[i]
                         : refB[i - kQWidth];
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - e));
  }
  std::cout << "  mixed attention (i8 deep+outliers) worst abs err " << worst
            << '\n';
  return worst;
}



double CompareMixedI8Early() {
  // Early-prefill geometry: queries start at position 0 over a small live
  // cache (kv 7 and 23). Exercises the causal-tail fixup path from the
  // first tile (first_key + 15 > q_pos_base == 0).
  constexpr std::uint32_t kRowsA = 7;
  constexpr std::uint32_t kRowsB = 16;
  constexpr std::uint32_t kKvA = 7;
  constexpr std::uint32_t kKvB = 23;
  constexpr std::uint32_t kRows = kRowsA + kRowsB;
  constexpr std::uint32_t kBlkBytes = 34;
  constexpr std::uint32_t kBlksPerRow = kKvWidth / 32;
  const std::uint32_t seed = 0xE4711U;
  const std::size_t q_count = static_cast<std::size_t>(kRows) * kQWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kfA = MakeValues(static_cast<std::size_t>(kKvA) * kKvWidth,
                              seed ^ 0xAAAAU, 1.0F);
  const auto vfA = MakeValues(static_cast<std::size_t>(kKvA) * kKvWidth,
                              seed ^ 0x3333U, 1.0F);
  const auto kfB = MakeValues(static_cast<std::size_t>(kKvB) * kKvWidth,
                              seed ^ 0xBBBBU, 1.0F);
  const auto vfB = MakeValues(static_cast<std::size_t>(kKvB) * kKvWidth,
                              seed ^ 0xCCCCU, 1.0F);
  std::vector<__half> vhA(vfA.size()), vhB(vfB.size());
  std::vector<std::uint8_t> vbA(static_cast<std::size_t>(kKvA) * kBlksPerRow *
                                18),
      vbB(static_cast<std::size_t>(kKvB) * kBlksPerRow * 18);
  const auto pack4 = [&](const std::vector<float>& vf, std::vector<__half>& vh,
                         std::vector<std::uint8_t>& vb, std::uint32_t keys) {
    for (std::uint32_t k = 0; k < keys; ++k) {
      for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
        float m = 0.0F;
        for (std::uint32_t i = 0; i < 32; ++i) {
          m = fmaxf(m, fabsf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                                (b * 32) + i]));
        }
        const float d = m / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        std::uint8_t* blk =
            &vb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 18];
        *reinterpret_cast<__half*>(blk + 16) = __float2half(d);
        for (std::uint32_t i = 0; i < 32; i += 2) {
          const float x0 =
              vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
          const float x1 =
              vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1];
          int q0 = static_cast<int>(static_cast<std::int8_t>(roundf(x0 * id)));
          int q1 = static_cast<int>(static_cast<std::int8_t>(roundf(x1 * id)));
          q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
          q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
          blk[i / 2] = static_cast<std::uint8_t>((q0 & 0xF) | (q1 << 4));
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
              __float2half(static_cast<float>(q0) * d);
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1] =
              __float2half(static_cast<float>(q1) * d);
        }
      }
    }
  };
  pack4(vfA, vhA, vbA, kKvA);
  pack4(vfB, vhB, vbB, kKvB);
  std::vector<std::uint8_t> kbA(static_cast<std::size_t>(kKvA) * kBlksPerRow *
                                kBlkBytes),
      kbB(static_cast<std::size_t>(kKvB) * kBlksPerRow * kBlkBytes);
  std::vector<__half> khA(kfA.size()), khB(kfB.size());
  const auto pack = [&](const std::vector<float>& kf, std::vector<__half>& kh,
                        std::vector<std::uint8_t>& kb, std::uint32_t keys) {
    for (std::uint32_t k = 0; k < keys; ++k) {
      for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
        float m = 0.0F;
        for (std::uint32_t i = 0; i < 32; ++i) {
          m = fmaxf(m, fabsf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                                (b * 32) + i]));
        }
        const float d = m / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        std::uint8_t* blk =
            &kb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * kBlkBytes];
        // Production blocked layout: 32 codes at +0, f16 scale at +32.
        *reinterpret_cast<__half*>(blk + 32) = __float2half(d);
        for (std::uint32_t i = 0; i < 32; ++i) {
          const float x =
              kf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
          const auto c = static_cast<std::int8_t>(roundf(x * id));
          blk[i] = static_cast<std::uint8_t>(c);
          kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
              __float2half(static_cast<float>(c) * d);
        }
      }
    }
  };
  pack(kfA, khA, kbA, kKvA);
  pack(kfB, khB, kbB, kKvB);
  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<std::uint8_t> d_kA(kbA.size());
  HipBuffer<std::uint8_t> d_kB(kbB.size());
  HipBuffer<std::uint8_t> d_vA(vbA.size());
  HipBuffer<std::uint8_t> d_vB(vbB.size());
  HipBuffer<float> d_out(q_count);
  HipBuffer<q::MixedBlk> d_blks(2);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_kA, kbA);
  Upload(&d_kB, kbB);
  Upload(&d_vA, vbA);
  Upload(&d_vB, vbB);
  std::vector<q::MixedBlk> blks(2);
  blks[0] = {0, reinterpret_cast<std::uint64_t>(d_kA.get()),
             reinterpret_cast<std::uint64_t>(d_vA.get()), kKvA, 0, kRowsA};
  blks[1] = {kRowsA, reinterpret_cast<std::uint64_t>(d_kB.get()),
             reinterpret_cast<std::uint64_t>(d_vB.get()), kKvB, 0, kRowsB};
  Upload(&d_blks, blks);
  if (!q::WmmaMixedAttention(d_q.get(), d_gate.get(), d_blks.get(), 2,
                             d_out.get(), kHeads, kKvHeads, kDim, kRatio,
                             q::kKvKeyBlockBytes, 18, nullptr, true)) {
    throw std::runtime_error("mixed attention (i8 early) rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "mixed attention (i8 early) sync");
  const auto out = Download(&d_out, q_count);
  const auto refA = Fp64Attention(
      std::vector<float>(qv.begin(), qv.begin() + kRowsA * kQWidth),
      std::vector<float>(gate.begin(), gate.begin() + kRowsA * kQWidth), khA,
      vhA, nullptr, 0, kRowsA, 0);
  const auto refB = Fp64Attention(
      std::vector<float>(qv.begin() + kRowsA * kQWidth, qv.end()),
      std::vector<float>(gate.begin() + kRowsA * kQWidth, gate.end()), khB,
      vhB, nullptr, 0, kRowsB, 0);
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    const double e = (i < static_cast<std::size_t>(kRowsA) * kQWidth)
                         ? refA[i]
                         : refB[i - kRowsA * kQWidth];
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - e));
  }
  std::cout << "  mixed attention (i8 early) worst abs err " << worst
            << '\n';
  return worst;
}



static std::vector<std::uint8_t> LoadBytes(const char* path,
                                           std::size_t expect) {
  FILE* f = std::fopen(path, "rb");
  if (!f)
    throw std::runtime_error(std::string("missing dump ") + path);
  std::vector<std::uint8_t> out(expect);
  if (std::fread(out.data(), 1, expect, f) != expect) {
    (void)std::fclose(f);
    throw std::runtime_error(std::string("short dump ") + path);
  }
  (void)std::fclose(f);
  return out;
}

double CompareMixedReplayIdx(int idx) {
  // Geometry mirrors the dumped batch: 2 blks, qrows=5, kv 35/52.
  constexpr std::uint32_t kKvA = 35;
  constexpr std::uint32_t kKvB = 52;
  constexpr std::uint32_t kRows = 5;
  constexpr std::uint32_t kQDim = kHeads * kDim;
  constexpr std::uint32_t kKvW = kKvHeads * kDim;
  constexpr std::uint32_t kBlksPerRow = kKvW / 32;
  const std::size_t q_count = static_cast<std::size_t>(kRows) * kQDim;
  const auto qb = LoadBytes((std::string("/tmp/mx_") + "q" + std::to_string(idx) + ".bin").c_str(), q_count * 4);
  const auto gb = LoadBytes((std::string("/tmp/mx_") + "gate" + std::to_string(idx) + ".bin").c_str(), q_count * 4);
  std::vector<float> qv(q_count), gate(q_count);
  std::memcpy(qv.data(), qb.data(), qb.size());
  std::memcpy(gate.data(), gb.data(), gb.size());
  const auto kbA = LoadBytes((std::string("/tmp/mx_k") + std::to_string(idx) + "_0.bin").c_str(),
                             static_cast<std::size_t>(kKvA) * kBlksPerRow * 34);
  const auto kbB = LoadBytes((std::string("/tmp/mx_k") + std::to_string(idx) + "_1.bin").c_str(),
                             static_cast<std::size_t>(kKvB) * kBlksPerRow * 34);
  const auto vbA = LoadBytes((std::string("/tmp/mx_v") + std::to_string(idx) + "_0.bin").c_str(),
                             static_cast<std::size_t>(kKvA) * kBlksPerRow * 18);
  const auto vbB = LoadBytes((std::string("/tmp/mx_v") + std::to_string(idx) + "_1.bin").c_str(),
                             static_cast<std::size_t>(kKvB) * kBlksPerRow * 18);
  // Host dequant (mirrors DequantizeKvRowsKernel: codes-first).
  std::vector<__half> khA(kKvA * kKvW), khB(kKvB * kKvW);
  std::vector<__half> vhA(kKvA * kKvW), vhB(kKvB * kKvW);
  const auto deq = [&](const std::vector<std::uint8_t>& kb,
                       const std::vector<std::uint8_t>& vb,
                       std::vector<__half>& kh, std::vector<__half>& vh,
                       std::uint32_t keys) {
    for (std::uint32_t k = 0; k < keys; ++k) {
      for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
        const std::uint8_t* kblk =
            &kb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 34];
        __half khs;
        std::memcpy(&khs, kblk + 32, 2);
        const float kd = __half2float(khs);
        const std::uint8_t* vblk =
            &vb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 18];
        __half vhs;
        std::memcpy(&vhs, vblk + 16, 2);
        const float vd = __half2float(vhs);
        for (std::uint32_t i = 0; i < 32; ++i) {
          const int8_t qc = static_cast<int8_t>(kblk[i]);
          kh[(static_cast<std::size_t>(k) * kKvW) + (b * 32) + i] =
              __float2half(static_cast<float>(qc) * kd);
          const std::uint8_t nbyte = vblk[i / 2];
          const int nib = (i % 2 == 0) ? (nbyte & 0xF) : (nbyte >> 4);
          const int qv4 = nib - (nib & 8 ? 16 : 0);
          vh[(static_cast<std::size_t>(k) * kKvW) + (b * 32) + i] =
              __float2half(static_cast<float>(qv4) * vd);
        }
      }
    }
  };
  deq(kbA, vbA, khA, vhA, kKvA);
  deq(kbB, vbB, khB, vhB, kKvB);

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<std::uint8_t> d_kA(kbA.size());
  HipBuffer<std::uint8_t> d_kB(kbB.size());
  HipBuffer<std::uint8_t> d_vA(vbA.size());
  HipBuffer<std::uint8_t> d_vB(vbB.size());
  HipBuffer<__half> d_skA(khA.size());
  HipBuffer<__half> d_skB(khB.size());
  HipBuffer<__half> d_svA(vhA.size());
  HipBuffer<__half> d_svB(vhB.size());
  HipBuffer<float> d_out(q_count);
  HipBuffer<q::MixedBlk> d_blks(2);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_kA, kbA);
  Upload(&d_kB, kbB);
  Upload(&d_vA, vbA);
  Upload(&d_vB, vbB);
  Upload(&d_skA, khA);
  Upload(&d_skB, khB);
  Upload(&d_svA, vhA);
  Upload(&d_svB, vhB);
  // i8 route over blocked caches.
  std::vector<q::MixedBlk> blkI8(2);
  blkI8[0] = {0, reinterpret_cast<std::uint64_t>(d_kA.get()),
              reinterpret_cast<std::uint64_t>(d_vA.get()), kKvA, kKvA - 1, 1};
  blkI8[1] = {1, reinterpret_cast<std::uint64_t>(d_kB.get()),
              reinterpret_cast<std::uint64_t>(d_vB.get()), kKvB, kKvB - 4, 4};
  Upload(&d_blks, blkI8);
  if (!q::WmmaMixedAttention(d_q.get(), d_gate.get(), d_blks.get(), 2,
                             d_out.get(), kHeads, kKvHeads, kDim, kRatio,
                             q::kKvKeyBlockBytes, 18, nullptr, true)) {
    throw std::runtime_error("replay i8 rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "replay i8 sync");
  const auto outI8 = Download(&d_out, q_count);
  // f16 route over staged (dequantized) caches.
  std::vector<q::MixedBlk> blkF(2);
  blkF[0] = {0, reinterpret_cast<std::uint64_t>(d_skA.get()),
             reinterpret_cast<std::uint64_t>(d_svA.get()), kKvA, kKvA - 1, 1};
  blkF[1] = {1, reinterpret_cast<std::uint64_t>(d_skB.get()),
             reinterpret_cast<std::uint64_t>(d_svB.get()), kKvB, kKvB - 4, 4};
  Upload(&d_blks, blkF);
  if (!q::WmmaMixedAttention(d_q.get(), d_gate.get(), d_blks.get(), 2,
                             d_out.get(), kHeads, kKvHeads, kDim, kRatio,
                             q::kKvKeyBlockBytes, 0, nullptr, false)) {
    throw std::runtime_error("replay f16 rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "replay f16 sync");
  const auto outF = Download(&d_out, q_count);
  // Host scalar reference on dequantized caches.
  const auto refA = Fp64Attention(
      std::vector<float>(qv.begin(), qv.begin() + kQDim),
      std::vector<float>(gate.begin(), gate.begin() + kQDim), khA, vhA,
      nullptr, 0, 1, kKvA - 1);
  const auto refB = Fp64Attention(
      std::vector<float>(qv.begin() + kQDim, qv.end()),
      std::vector<float>(gate.begin() + kQDim, gate.end()), khB, vhB,
      nullptr, 0, 4, kKvB - 4);
  double wI8 = 0.0, wF = 0.0, wIF = 0.0;
  double mI8 = 0.0, mF = 0.0, mIF = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    const double e = (i < static_cast<std::size_t>(kQDim)) ? refA[i]
                                                          : refB[i - kQDim];
    const double dI8 = static_cast<double>(outI8[i]) - e;
    const double dF = static_cast<double>(outF[i]) - e;
    const double dIF = static_cast<double>(outI8[i]) - outF[i];
    wI8 = std::max(wI8, std::abs(dI8));
    wF = std::max(wF, std::abs(dF));
    wIF = std::max(wIF, std::abs(dIF));
    mI8 += dI8;
    mF += dF;
    mIF += dIF;
  }
  mI8 /= q_count;
  mF /= q_count;
  mIF /= q_count;
  std::cout << "  replay[" << idx << "] vs-ref i8 " << wI8 << " f16 " << wF
            << " i8-vs-f16 " << wIF << " mean " << mI8 << " " << mF << " "
            << mIF << '\n';
  return wIF;
}



double CompareMixedStress() {
  // 128-tile sweep, 1-row decode at depth: 3 consecutive launches on the
  // same data must be bit-identical (race detector).
  constexpr std::uint32_t kKv = 2048;
  constexpr std::uint32_t kRows = 1;
  constexpr std::uint32_t kBlksPerRow = kKvWidth / 32;
  const std::uint32_t seed = 0x5EED01;
  const std::size_t q_count = static_cast<std::size_t>(kRows) * kQWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(static_cast<std::size_t>(kKv) * kKvWidth,
                             seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(static_cast<std::size_t>(kKv) * kKvWidth,
                             seed ^ 0x3333U, 1.0F);
  std::vector<__half> vh(vf.size());
  std::vector<std::uint8_t> vb(static_cast<std::size_t>(kKv) * kBlksPerRow *
                               18);
  for (std::uint32_t k = 0; k < kKv; ++k) {
    for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
      float m = 0.0F;
      for (std::uint32_t i = 0; i < 32; ++i)
        m = fmaxf(m, fabsf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                              (b * 32) + i]));
      const float d = m / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      std::uint8_t* blk = &vb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 18];
      *reinterpret_cast<__half*>(blk + 16) = __float2half(d);
      for (std::uint32_t i = 0; i < 32; i += 2) {
        const float x0 = vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
        const float x1 = vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1];
        int q0 = static_cast<int>(static_cast<std::int8_t>(roundf(x0 * id)));
        int q1 = static_cast<int>(static_cast<std::int8_t>(roundf(x1 * id)));
        q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
        q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
        blk[i / 2] = static_cast<std::uint8_t>((q0 & 0xF) | (q1 << 4));
        vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
            __float2half(static_cast<float>(q0) * d);
        vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1] =
            __float2half(static_cast<float>(q1) * d);
      }
    }
  }
  std::vector<std::uint8_t> kb(static_cast<std::size_t>(kKv) * kBlksPerRow * 34);
  std::vector<__half> kh(kf.size());
  for (std::uint32_t k = 0; k < kKv; ++k) {
    for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
      float m = 0.0F;
      for (std::uint32_t i = 0; i < 32; ++i)
        m = fmaxf(m, fabsf(kf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i]));
      const float d = m / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      std::uint8_t* blk = &kb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 34];
      *reinterpret_cast<__half*>(blk + 32) = __float2half(d);
      for (std::uint32_t i = 0; i < 32; ++i) {
        const float x = kf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
        const auto c = static_cast<std::int8_t>(roundf(x * id));
        blk[i] = static_cast<std::uint8_t>(c);
        kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
            __float2half(static_cast<float>(c) * d);
      }
    }
  }
  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<std::uint8_t> d_k(kb.size());
  HipBuffer<std::uint8_t> d_v(vb.size());
  HipBuffer<float> d_out(q_count);
  HipBuffer<q::MixedBlk> d_blks(1);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kb);
  Upload(&d_v, vb);
  std::vector<q::MixedBlk> blks(1);
  blks[0] = {0, reinterpret_cast<std::uint64_t>(d_k.get()),
             reinterpret_cast<std::uint64_t>(d_v.get()), kKv, kKv - 1, 1};
  Upload(&d_blks, blks);
  std::vector<float> outs[3];
  for (int r = 0; r < 3; ++r) {
    if (!q::WmmaMixedAttention(d_q.get(), d_gate.get(), d_blks.get(), 1,
                               d_out.get(), kHeads, kKvHeads, kDim, kRatio,
                               q::kKvKeyBlockBytes, 18, nullptr, true)) {
      throw std::runtime_error("stress rejected geometry");
    }
    CheckHip(hipDeviceSynchronize(), "stress sync");
    outs[r] = Download(&d_out, q_count);
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(outs[0][i]) - outs[1][i]));
    worst = std::max(worst, std::abs(static_cast<double>(outs[0][i]) - outs[2][i]));
  }
  std::cout << "  mixed stress (128 tiles x3) max run-to-run diff "
            << std::scientific << std::setprecision(17) << worst << '\n';
  std::cout << "  dbg le=" << (worst <= 1e-12) << " ne0=" << (worst != 0.0)
            << '\n';
  if (!(worst <= 1e-12)) {
    for (std::size_t i = 0; i < q_count; ++i) {
      const double d1 = std::abs(static_cast<double>(outs[0][i]) - outs[1][i]);
      const double d2 = std::abs(static_cast<double>(outs[0][i]) - outs[2][i]);
      if (d1 > 1e-12 || d2 > 1e-12) {
        std::cout << "  first diff at " << i << " d1=" << d1 << " d2=" << d2
                  << " v0=" << outs[0][i] << " v1=" << outs[1][i]
                  << " v2=" << outs[2][i] << '\n';
        break;
      }
    }
    throw std::runtime_error("mixed stress: nondeterministic kernel!");
  }
  const auto refS = Fp64Attention(
      qv, gate, kh, vh, nullptr, 0, kRows, kKv - kRows);
  double wref = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    wref = std::max(wref,
                    std::abs(static_cast<double>(outs[0][i]) - refS[i]));
  }
  std::cout << "  mixed stress (128 tiles) vs-ref " << wref << '\n';
  return wref;
}



double CompareCausalI8(std::uint32_t seed) {
  // Dense causal int8: blocked K34 + blocked V18 straight into the
  // WMMA cores (no F16 staging), vs Fp64 on dequantized blocks.
  constexpr std::uint32_t kTokens = 5;
  constexpr std::uint32_t kStart = 32;
  constexpr std::uint32_t kKv = kStart + kTokens;
  constexpr std::uint32_t kBlkBytes = 34;
  constexpr std::uint32_t kBlksPerRow = kKvWidth / 32;
  const std::size_t q_count = static_cast<std::size_t>(kTokens) * kQWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(static_cast<std::size_t>(kKv) * kKvWidth,
                             seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(static_cast<std::size_t>(kKv) * kKvWidth,
                             seed ^ 0x3333U, 1.0F);
  // Pack K into 34B codes-first blocks; V into 18B nibble blocks.
  std::vector<std::uint8_t> kb(static_cast<std::size_t>(kKv) * kBlksPerRow *
                               kBlkBytes);
  std::vector<__half> kh(kf.size());
  for (std::uint32_t k = 0; k < kKv; ++k) {
    for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
      float m = 0.0F;
      for (std::uint32_t i = 0; i < 32; ++i)
        m = fmaxf(m, fabsf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                              (b * 32) + i]));
      const float d = m / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      std::uint8_t* blk =
          &kb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * kBlkBytes];
      *reinterpret_cast<__half*>(blk + 32) = __float2half(d);
      for (std::uint32_t i = 0; i < 32; ++i) {
        const float x =
            kf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
        const auto c = static_cast<std::int8_t>(roundf(x * id));
        blk[i] = static_cast<std::uint8_t>(c);
        kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
            __float2half(static_cast<float>(c) * d);
      }
    }
  }
  std::vector<__half> vh(vf.size());
  std::vector<std::uint8_t> vb(static_cast<std::size_t>(kKv) * kBlksPerRow *
                               18);
  for (std::uint32_t k = 0; k < kKv; ++k) {
    for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
      float m = 0.0F;
      for (std::uint32_t i = 0; i < 32; ++i)
        m = fmaxf(m, fabsf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                              (b * 32) + i]));
      const float d = m / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      std::uint8_t* blk = &vb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 18];
      *reinterpret_cast<__half*>(blk + 16) = __float2half(d);
      for (std::uint32_t i = 0; i < 32; i += 2) {
        const float x0 =
            vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
        const float x1 =
            vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1];
        int q0 = static_cast<int>(static_cast<std::int8_t>(roundf(x0 * id)));
        int q1 = static_cast<int>(static_cast<std::int8_t>(roundf(x1 * id)));
        q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
        q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
        blk[i / 2] = static_cast<std::uint8_t>((q0 & 0xF) | (q1 << 4));
        vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
            __float2half(static_cast<float>(q0) * d);
        vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1] =
            __float2half(static_cast<float>(q1) * d);
      }
    }
  }
  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<std::uint8_t> d_k(kb.size());
  HipBuffer<std::uint8_t> d_v(vb.size());
  HipBuffer<float> d_out(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kb);
  Upload(&d_v, vb);
  if (!q::WmmaCausalAttention(
          d_q.get(), d_gate.get(), reinterpret_cast<__half*>(d_k.get()),
          reinterpret_cast<__half*>(d_v.get()), nullptr, 0, d_out.get(),
          kTokens, kStart, kHeads, kKvHeads, kDim, kRatio, nullptr, false,
          q::kKvKeyBlockBytes, q::kKvValue4BitBlockBytes, true)) {
    throw std::runtime_error("causal attention (i8) rejected the geometry");
  }
  CheckHip(hipDeviceSynchronize(), "causal attention (i8) sync");
  const auto out = Download(&d_out, q_count);
  const auto ref = Fp64Attention(qv, gate, kh, vh, nullptr, 0, kTokens,
                                 kStart);
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i)
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - ref[i]));
  std::cout << "  causal attention (i8) worst absolute error " << worst
            << '\n';
  return worst;
}


double CompareCausalI8Sparse(std::uint32_t n_tokens,
                               std::uint32_t start_pos, std::uint32_t seed) {
  // Sparse/packed causal int8: blocked K34 + V18 straight into the WMMA
  // cores with a block-selection mask (the indexer path production takes
  // past top_k), vs Fp64 on the dequantized blocks with the same mask.
  const std::uint32_t kKv = start_pos + n_tokens;
  constexpr std::uint32_t kBlkBytes = 34;
  constexpr std::uint32_t kBlksPerRow = kKvWidth / 32;
  const std::uint32_t max_blocks = (kKv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(static_cast<std::size_t>(kKv) * kKvWidth,
                             seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(static_cast<std::size_t>(kKv) * kKvWidth,
                             seed ^ 0x3333U, 1.0F);
  std::vector<std::uint8_t> kb(static_cast<std::size_t>(kKv) * kBlksPerRow *
                               kBlkBytes);
  std::vector<__half> kh(kf.size());
  for (std::uint32_t k = 0; k < kKv; ++k) {
    for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
      float m = 0.0F;
      for (std::uint32_t i = 0; i < 32; ++i)
        m = fmaxf(m, fabsf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                              (b * 32) + i]));
      const float d = m / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      std::uint8_t* blk =
          &kb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * kBlkBytes];
      *reinterpret_cast<__half*>(blk + 32) = __float2half(d);
      for (std::uint32_t i = 0; i < 32; ++i) {
        const float x =
            kf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
        const auto c = static_cast<std::int8_t>(roundf(x * id));
        blk[i] = static_cast<std::uint8_t>(c);
        kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
            __float2half(static_cast<float>(c) * d);
      }
    }
  }
  std::vector<__half> vh(vf.size());
  std::vector<std::uint8_t> vb(static_cast<std::size_t>(kKv) * kBlksPerRow *
                               18);
  for (std::uint32_t k = 0; k < kKv; ++k) {
    for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
      float m = 0.0F;
      for (std::uint32_t i = 0; i < 32; ++i)
        m = fmaxf(m, fabsf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                              (b * 32) + i]));
      const float d = m / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      std::uint8_t* blk = &vb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * 18];
      *reinterpret_cast<__half*>(blk + 16) = __float2half(d);
      for (std::uint32_t i = 0; i < 32; i += 2) {
        const float x0 =
            vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i];
        const float x1 =
            vf[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1];
        int q0 = static_cast<int>(static_cast<std::int8_t>(roundf(x0 * id)));
        int q1 = static_cast<int>(static_cast<std::int8_t>(roundf(x1 * id)));
        q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
        q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);
        blk[i / 2] = static_cast<std::uint8_t>((q0 & 0xF) | (q1 << 4));
        vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
            __float2half(static_cast<float>(q0) * d);
        vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i + 1] =
            __float2half(static_cast<float>(q1) * d);
      }
    }
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::uint32_t state = seed ^ 0x77777777U;
  for (std::uint32_t& word : mask) {
    word = NextRandom(&state) & NextRandom(&state);
    word &= NextRandom(&state) & NextRandom(&state);
  }
  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<std::uint8_t> d_k(kb.size());
  HipBuffer<std::uint8_t> d_v(vb.size());
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<float> d_out(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kb);
  Upload(&d_v, vb);
  Upload(&d_mask, mask);
  if (!q::WmmaCausalAttention(
          d_q.get(), d_gate.get(), reinterpret_cast<__half*>(d_k.get()),
          reinterpret_cast<__half*>(d_v.get()), d_mask.get(), mask_words,
          d_out.get(), n_tokens, start_pos, kHeads, kKvHeads, kDim, kRatio,
          nullptr, false, q::kKvKeyBlockBytes, q::kKvValue4BitBlockBytes,
          true)) {
    throw std::runtime_error("causal attention (i8 sparse) rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "causal attention (i8 sparse) sync");
  const auto out = Download(&d_out, q_count);
  const auto ref = Fp64Attention(qv, gate, kh, vh, mask.data(), mask_words,
                                 n_tokens, start_pos, kRatio);
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i)
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - ref[i]));
  std::cout << "  causal attention (i8 sparse n=" << n_tokens
            << " start=" << start_pos << ") worst absolute error " << worst
            << '\n';
  return worst;
}

double CompareCausalI8Layout(std::uint32_t n_tokens,
                               std::uint32_t start_pos, std::uint32_t seed,
                               bool k4, bool v4, bool sparse) {
  // Causal int8 direct path over every K/V block layout the cache can hold:
  // K34/K18 x V34/V18, dense or with a block-selection mask. Reference is
  // Fp64 on the dequantized blocks, so this measures kernel numerics only.
  const std::uint32_t kKv = start_pos + n_tokens;
  const std::uint32_t kBlkBytes = k4 ? 18 : 34;
  const std::uint32_t vBlkBytes = v4 ? 18 : 34;
  constexpr std::uint32_t kBlksPerRow = kKvWidth / 32;
  const std::uint32_t max_blocks = (kKv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(static_cast<std::size_t>(kKv) * kKvWidth,
                             seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(static_cast<std::size_t>(kKv) * kKvWidth,
                             seed ^ 0x3333U, 1.0F);
  const float k_range = k4 ? 7.0F : 127.0F;
  const float v_range = v4 ? 7.0F : 127.0F;
  const int k_clamp = k4 ? 7 : 127;
  const int v_clamp = v4 ? 7 : 127;
  std::vector<std::uint8_t> kb(static_cast<std::size_t>(kKv) * kBlksPerRow *
                               kBlkBytes);
  std::vector<__half> kh(kf.size());
  for (std::uint32_t k = 0; k < kKv; ++k) {
    for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
      float m = 0.0F;
      for (std::uint32_t i = 0; i < 32; ++i)
        m = fmaxf(m, fabsf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                              (b * 32) + i]));
      const float d = (m != 0.0F) ? (m / k_range) : 1.0F;
      const float id = 1.0F / d;
      std::uint8_t* blk =
          &kb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * kBlkBytes];
      if (k4) {
        *reinterpret_cast<__half*>(blk + 16) = __float2half(d);
        for (std::uint32_t e = 0; e < 16; ++e) {
          int q0 = static_cast<int>(
              roundf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                        (b * 32) + e * 2] *
                     id));
          int q1 = static_cast<int>(
              roundf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                        (b * 32) + e * 2 + 1] *
                     id));
          q0 = q0 > k_clamp ? k_clamp : (q0 < -k_clamp ? -k_clamp : q0);
          q1 = q1 > k_clamp ? k_clamp : (q1 < -k_clamp ? -k_clamp : q1);
          blk[e] = static_cast<std::uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
          kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + e * 2] =
              __float2half(static_cast<float>(q0) * d);
          kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + e * 2 + 1] =
              __float2half(static_cast<float>(q1) * d);
        }
      } else {
        *reinterpret_cast<__half*>(blk + 32) = __float2half(d);
        for (std::uint32_t i = 0; i < 32; ++i) {
          int q = static_cast<int>(
              roundf(kf[(static_cast<std::size_t>(k) * kKvWidth) +
                        (b * 32) + i] *
                     id));
          q = q > k_clamp ? k_clamp : (q < -k_clamp ? -k_clamp : q);
          blk[i] = static_cast<std::uint8_t>(q);
          kh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
              __float2half(static_cast<float>(q) * d);
        }
      }
    }
  }
  std::vector<__half> vh(vf.size());
  std::vector<std::uint8_t> vb(static_cast<std::size_t>(kKv) * kBlksPerRow *
                               vBlkBytes);
  for (std::uint32_t k = 0; k < kKv; ++k) {
    for (std::uint32_t b = 0; b < kBlksPerRow; ++b) {
      float m = 0.0F;
      for (std::uint32_t i = 0; i < 32; ++i)
        m = fmaxf(m, fabsf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                              (b * 32) + i]));
      const float d = (m != 0.0F) ? (m / v_range) : 1.0F;
      const float id = 1.0F / d;
      std::uint8_t* blk =
          &vb[(static_cast<std::size_t>(k) * kBlksPerRow + b) * vBlkBytes];
      if (v4) {
        *reinterpret_cast<__half*>(blk + 16) = __float2half(d);
        for (std::uint32_t e = 0; e < 16; ++e) {
          int q0 = static_cast<int>(
              roundf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                        (b * 32) + e * 2] *
                     id));
          int q1 = static_cast<int>(
              roundf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                        (b * 32) + e * 2 + 1] *
                     id));
          q0 = q0 > v_clamp ? v_clamp : (q0 < -v_clamp ? -v_clamp : q0);
          q1 = q1 > v_clamp ? v_clamp : (q1 < -v_clamp ? -v_clamp : q1);
          blk[e] = static_cast<std::uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + e * 2] =
              __float2half(static_cast<float>(q0) * d);
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + e * 2 + 1] =
              __float2half(static_cast<float>(q1) * d);
        }
      } else {
        *reinterpret_cast<__half*>(blk + 32) = __float2half(d);
        for (std::uint32_t i = 0; i < 32; ++i) {
          int q = static_cast<int>(
              roundf(vf[(static_cast<std::size_t>(k) * kKvWidth) +
                        (b * 32) + i] *
                     id));
          q = q > v_clamp ? v_clamp : (q < -v_clamp ? -v_clamp : q);
          blk[i] = static_cast<std::uint8_t>(q);
          vh[(static_cast<std::size_t>(k) * kKvWidth) + (b * 32) + i] =
              __float2half(static_cast<float>(q) * d);
        }
      }
    }
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words,
                                  0xFFFFFFFFU);
  if (sparse) {
    std::uint32_t state = seed ^ 0x77777777U;
    for (std::uint32_t& word : mask) {
      word = NextRandom(&state) & NextRandom(&state);
      word &= NextRandom(&state) & NextRandom(&state);
    }
  }
  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<std::uint8_t> d_k(kb.size());
  HipBuffer<std::uint8_t> d_v(vb.size());
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<float> d_out(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kb);
  Upload(&d_v, vb);
  Upload(&d_mask, mask);
  if (!q::WmmaCausalAttention(
          d_q.get(), d_gate.get(), reinterpret_cast<__half*>(d_k.get()),
          reinterpret_cast<__half*>(d_v.get()),
          sparse ? d_mask.get() : nullptr, mask_words, d_out.get(), n_tokens,
          start_pos, kHeads, kKvHeads, kDim, kRatio, nullptr, false,
          k4 ? q::kKvKey4BitBlockBytes : q::kKvKeyBlockBytes,
          v4 ? q::kKvValue4BitBlockBytes : q::kKvKeyBlockBytes, true)) {
    throw std::runtime_error("causal attention (i8 layout) rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "causal attention (i8 layout) sync");
  const auto out = Download(&d_out, q_count);
  const auto ref =
      sparse ? Fp64Attention(qv, gate, kh, vh, mask.data(), mask_words,
                             n_tokens, start_pos, kRatio)
             : Fp64Attention(qv, gate, kh, vh, nullptr, 0, n_tokens, start_pos);
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i)
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - ref[i]));
  std::cout << "  causal attention (i8 K" << (k4 ? 4 : 8) << "V"
            << (v4 ? 4 : 8) << (sparse ? " sparse" : " dense") << " n="
            << n_tokens << " start=" << start_pos
            << ") worst absolute error " << worst << '\n';
  return worst;
}

}  // namespace

int main() {
  try {
    const double mixed = CompareMixed(0xC0FFEEU);
    std::cout << "WMMA mixed attention worst absolute error " << mixed << '\n';
    if (mixed > 2e-2)
      throw std::runtime_error("mixed attention exceeds its FP64 envelope");
    const double mixed_deep = CompareMixedI8Deep(0xC0FFEEU);
    const double mixed_early = CompareMixedI8Early();
    const double causal_i8 = CompareCausalI8(0xC0FFEEU);
    std::cout << "WMMA causal attention (i8) worst absolute error "
              << causal_i8 << '\n';
    if (causal_i8 > 2e-2)
      throw std::runtime_error("causal attention (i8) exceeds its envelope");
    const double causal_i8_sp1 = CompareCausalI8Sparse(8, 9000, 0x5EA5E5U);
    const double causal_i8_sp2 = CompareCausalI8Sparse(5, 32765, 0x5EA5E6U);
    std::cout << "WMMA causal attention (i8 sparse) worst " << causal_i8_sp1
              << " / " << causal_i8_sp2 << '\n';
    if (std::max(causal_i8_sp1, causal_i8_sp2) > 2e-2)
      throw std::runtime_error("causal attention (i8 sparse) exceeds envelope");
    double layoutWorst = 0.0;
    layoutWorst = std::max(
        layoutWorst,
        CompareCausalI8Layout(5, 32, 0xBEEF00U, false, false, false));
    layoutWorst = std::max(
        layoutWorst,
        CompareCausalI8Layout(5, 32, 0xBEEF01U, true, true, false));
    layoutWorst = std::max(
        layoutWorst,
        CompareCausalI8Layout(5, 32, 0xBEEF02U, true, false, false));
    layoutWorst = std::max(
        layoutWorst,
        CompareCausalI8Layout(8, 9000, 0xBEEF03U, true, true, true));
    std::cout << "WMMA causal attention (i8 layouts) worst " << layoutWorst
              << '\n';
    if (layoutWorst > 2e-2)
      throw std::runtime_error("causal attention (i8 layouts) exceeds envelope");
    const double stress = CompareMixedStress();
    std::cout << "WMMA mixed stress 128-tile vs-ref " << stress << '\n';
    if (stress > 2e-2)
      throw std::runtime_error("mixed stress exceeds envelope");
    double replayWorst = 0.0;
    for (int li = 0; li < 12; ++li) {
      const double r = CompareMixedReplayIdx(li);
      replayWorst = std::max(replayWorst, r);
    }
    std::cout << "WMMA mixed replay i8-vs-f16 worst12 " << replayWorst
              << '\n';
    if (replayWorst > 2e-2)
      throw std::runtime_error("replay i8-vs-f16 exceeds envelope");
    std::cout << "WMMA mixed attention (i8 early) worst abs err "
              << mixed_early << '\n';
    if (mixed_early > 2e-2)
      throw std::runtime_error("mixed attention (i8 early) exceeds envelope");
    std::cout << "WMMA mixed attention (i8 deep) worst abs err " << mixed_deep
              << '\n';
    if (mixed_deep > 2e-2)
      throw std::runtime_error("mixed attention (i8 deep) exceeds envelope");
    const double mixed_i8 = CompareMixedI8(0xC0FFEEU);
    std::cout << "WMMA mixed attention (i8 K) worst absolute error "
              << mixed_i8 << '\n';
    if (mixed_i8 > 2e-2)
      throw std::runtime_error(
          "mixed attention (i8) exceeds its FP64 envelope");
    CheckChunks(136, 94);
    CheckChunks(2048, 1025);
    CheckPreparation(1, 0, 64);
    CheckPreparation(8, 4096, 64);
    CheckPreparation(65, 131069, 64);
    CheckPreparation(7, 131069, 256);
    // Structured selections against the sparse path's 1024-block compaction
    // windows and 64-block key tiles. At 131069 the last 512 blocks sit in
    // the final window; 262141 spans all 64 windows of the native context,
    // so 63 blocks per window carry every partial-tile size once.
    constexpr std::uint32_t kWindow = 1024;
    const auto spread = [](std::uint32_t count) {
      return [count](std::uint32_t complete, std::vector<std::uint32_t>* out) {
        for (std::uint32_t j = 0; j < count && j < complete; ++j) {
          out->push_back(static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(j) * complete / count));
        }
      };
    };
    const auto per_window = [](std::uint32_t count) {
      return [count](std::uint32_t complete, std::vector<std::uint32_t>* out) {
        for (std::uint32_t w0 = 0; w0 < complete; w0 += kWindow) {
          for (std::uint32_t j = 0; j < count; ++j) {
            const std::uint32_t b = w0 + j * 16 + (w0 / kWindow) % 16;
            if (b < complete) {
              out->push_back(b);
            }
          }
        }
      };
    };
    const BlockPattern last_budget = [](std::uint32_t complete,
                                        std::vector<std::uint32_t>* out) {
      for (std::uint32_t b = complete - std::min(512U, complete); b < complete;
           ++b) {
        out->push_back(b);
      }
    };
    const BlockPattern one_window = [](std::uint32_t complete,
                                       std::vector<std::uint32_t>* out) {
      for (std::uint32_t j = 0; j < 64 && 7 * kWindow + j * 16 < complete;
           ++j) {
        out->push_back(7 * kWindow + j * 16);
      }
    };
    struct Structured {
      const char* name;
      std::uint32_t start_pos;
      BlockPattern pattern;
    };
    const Structured structured[] = {
        {"last-512", 131069, last_budget},
        {"63-per-window", 262141, per_window(63)},
        {"1-per-window", 262141, per_window(1)},
        {"40-total", 131069, spread(40)},
        {"64-in-one-window", 131069, one_window},
        {"513-total", 131069, spread(513)},
        {"576-total", 131069, spread(576)},
    };
    // Compacted selections: a short context that selects every block, then
    // the budgeted rows at increasing depth, up to the 160K decode target.
    std::uint32_t list_seed = 0x115700U;
    for (const std::uint32_t depth : {1500U, 8192U, 32768U, 163840U}) {
      for (const std::uint32_t n : {1U, 3U, 8U}) {
        CheckLists(n, depth, list_seed++);
      }
    }
    std::uint32_t structured_seed = 0x5EED0000U;
    for (const Structured& c : structured) {
      for (const std::uint32_t n : {1U, 3U}) {
        CheckStructured(c.name, n, c.start_pos, c.pattern, structured_seed++);
      }
    }
    // At ratio 1 a tile is 256 blocks, so partial tiles reach 231 blocks,
    // more than the room kept for a ratio-4 carry. Alternating 999 and 100
    // selected blocks per window attends some partial tiles in their window
    // and carries others.
    const BlockPattern alternating = [](std::uint32_t complete,
                                        std::vector<std::uint32_t>* out) {
      for (std::uint32_t b = 0; b < complete; ++b) {
        if (b % kWindow < ((b / kWindow) % 2 == 0 ? 999U : 100U)) {
          out->push_back(b);
        }
      }
    };
    for (const std::uint32_t n : {1U, 3U}) {
      CheckStructured("ratio-1-alternating", n, 8189, alternating,
                      structured_seed++, 1);
    }
    struct Case {
      std::uint32_t n_tokens;
      std::uint32_t start_pos;
      bool masked;
      bool sparse;
      bool bounded{false};
    };
    const Case cases[] = {
        {4, 4096, false, false},      {3, 9000, true, true},
        {100, 0, false, false},       {64, 37, false, false},
        {100, 0, true, false},        {77, 51, true, false},
        {96, 4096, true, true},       {70, 8000, true, true},
        {5, 32765, true, true, true}, {7, 65533, true, true, true},
        {5, 131069, true, true},      {7, 131069, true, true, true},
        {17, 2047, true, true, true}};
    bool ok = true;
    std::uint32_t seed = 0x1234ABCDU;
    for (const Case& c : cases) {
      const double worst = Compare(c.n_tokens, c.start_pos, c.masked, seed++,
                                   c.sparse, c.bounded);
      std::cout << "WMMA attention n=" << c.n_tokens << " start=" << c.start_pos
                << (c.masked ? (c.sparse ? " sparse" : " masked") : " dense")
                << " worst absolute error " << worst << '\n';
      // The reference accumulates FP32 probabilities; the WMMA route rounds
      // Q and P to FP16, so the contract is a small absolute envelope on
      // values of order one.
      ok = ok && worst < 2e-2;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
