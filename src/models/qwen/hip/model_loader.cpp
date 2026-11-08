#if defined(ENGINE_ENABLE_HIP)
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/hip/blocked_q8.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip {
namespace {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      (void)munmap(region.host_copy, region.size);
    }
    region = {};
  }
}

void CopyMappedWeights(const core::GgufMappedRegion& source, void* copy) {
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  const auto chunks =
      source.size / kChunkBytes + (source.size % kChunkBytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  auto copy_chunk = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * kChunkBytes;
      const auto bytes = std::min(kChunkBytes, source.size - offset);
      auto* input = const_cast<std::uint8_t*>(
                        static_cast<const std::uint8_t*>(source.data)) +
                    offset;
      if (madvise(input, bytes, MADV_POPULATE_READ) != 0) {
        failure.store(errno, std::memory_order_relaxed);
        break;
      }
      std::memcpy(static_cast<std::uint8_t*>(copy) + offset, input, bytes);
      // The immutable copy now owns these resident bytes. Keep the original
      // mapping valid for the reader without retaining its populated PTEs.
      (void)madvise(input, bytes, MADV_DONTNEED);
    }
  };
  {
    std::vector<std::jthread> workers;
    for (std::size_t i = 1; i < std::min<std::size_t>(16, chunks); ++i)
      workers.emplace_back(copy_chunk);
    copy_chunk();
  }
  if (const auto error = failure.load(); error != 0)
    throw std::system_error(error, std::generic_category(),
                            "cannot read mapped Qwen weights");
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* host_copy = mmap(nullptr, source.size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_copy == MAP_FAILED)
    return hipErrorOutOfMemory;
  const auto unmap = [&](void* pointer) { (void)munmap(pointer, source.size); };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  (void)madvise(host_copy, source.size, MADV_HUGEPAGE);
  CopyMappedWeights(source, host_copy);
  const auto register_error = hipHostRegister(
      host_copy, source.size, hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess)
    return register_error;

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  weight_regions.resize(source_regions.size());
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      map_error = MapRegisteredRegion(source, destination);
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible through mapped registration: " +
                     hipGetErrorString(map_error);
      }
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

namespace {

// The routed MoE core instantiates Q4_K/Q5_K (plus Q8_0/Q4_0/Q5_1/FAST) but
// no Q6_K. Four qwen35moe layers carry Q6_K expert-down weights; converting
// them to Q8_0 at load keeps every layer on the identical routed path.
// Host-side data prep only (dequant + requant, no compute kernel).
[[nodiscard]] std::uint16_t F32ToF16Bits(float value) noexcept {
  std::uint32_t bits = 0;
  static_assert(sizeof(float) == sizeof(bits));
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::int32_t exp =
      static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 112;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if ((bits & 0x7FFFFFFFU) == 0) {
    return static_cast<std::uint16_t>(sign);
  }
  if (exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7BFFU);
  }
  if (exp <= 0) {
    return static_cast<std::uint16_t>(sign);
  }
  const std::uint32_t half =
      sign | (static_cast<std::uint32_t>(exp) << 10U) | (mantissa >> 13U);
  const std::uint32_t round_bit = (mantissa >> 12U) & 1U;
  const std::uint32_t sticky = mantissa & 0xFFFU;
  if (round_bit != 0 && (sticky != 0 || (half & 1U) != 0)) {
    return static_cast<std::uint16_t>(half + 1U);
  }
  return static_cast<std::uint16_t>(half);
}

[[nodiscard]] bool ConvertQ6KDownToQ80(models::QwenLayerWeights& layer,
                                       std::vector<void*>& owned,
                                       std::size_t& owned_bytes,
                                       std::vector<float>& host_f32,
                                       std::string* error_msg) {
  auto& tensor = layer.ffn_down_exps;
  if (!layer.is_moe || tensor.empty() ||
      tensor.type != core::GgmlType::kQ6_K) {
    return true;
  }
  const std::size_t num_elements = tensor.num_elements;
  if (num_elements == 0 || num_elements % 32 != 0) {
    if (error_msg != nullptr) {
      *error_msg = "Q6_K expert-down element count is not Q8_0 compatible";
    }
    return false;
  }
  host_f32.resize(num_elements);
  gufo::quant::DequantizeQ6_K(tensor.data, host_f32.data(), num_elements);
  const std::size_t num_blocks = num_elements / 32;
  std::vector<std::uint8_t> q80(num_blocks * 34);
  for (std::size_t b = 0; b < num_blocks; ++b) {
    float amax = 0.0F;
    for (std::size_t i = 0; i < 32; ++i) {
      const float v = host_f32[b * 32 + i];
      const float a = v < 0 ? -v : v;
      if (a > amax) amax = a;
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    auto* block = q80.data() + b * 34;
    std::uint16_t scale_bits = F32ToF16Bits(d);
    std::memcpy(block, &scale_bits, sizeof(scale_bits));
    for (std::size_t i = 0; i < 32; ++i) {
      long q = std::lround(host_f32[b * 32 + i] * id);
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      block[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
    }
  }
  void* device = nullptr;
  if (hipMalloc(&device, q80.size()) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to allocate converted Q8_0 expert-down weights";
    }
    return false;
  }
  if (hipMemcpy(device, q80.data(), q80.size(), hipMemcpyHostToDevice) !=
      hipSuccess) {
    (void)hipFree(device);
    if (error_msg != nullptr) {
      *error_msg = "Failed to upload converted Q8_0 expert-down weights";
    }
    return false;
  }
  tensor.data = device;
  tensor.type = core::GgmlType::kQ8_0;
  tensor.available_bytes = q80.size();
  owned.push_back(device);
  owned_bytes += q80.size();
  return true;
}

}  // namespace

QwenGpuModel::QwenGpuModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions,
    std::shared_ptr<models::qwen::vision::Encoder> vision)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)),
      vision_(std::move(vision)) {}

QwenGpuModel::~QwenGpuModel() {
  for (void* ptr : moe_converted_) {
    (void)hipFree(ptr);
  }
  ReleaseWeightRegions(weight_regions_);
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = vision_ ? vision_->ResidentBytes() : 0;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  if (moe_converted_bytes_ >
      std::numeric_limits<std::size_t>::max() - total) {
    return std::numeric_limits<std::size_t>::max();
  }
  return total + moe_converted_bytes_;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::shared_ptr<models::qwen::vision::Encoder> vision) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }

  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            error_msg)) {
    return nullptr;
  }

  auto weights_opt = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (!tokenizer)
    return nullptr;

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(*reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
               remap(layer.attn_gate) && remap(layer.ssm_out) &&
               remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
               remap(layer.ssm_beta) && remap(layer.ssm_a) &&
               remap(layer.ssm_dt) && remap(layer.ssm_norm) &&
               remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
               remap(layer.ffn_up) && remap(layer.ffn_down) &&
               remap(layer.ffn_gate_exps) && remap(layer.ffn_up_exps) &&
               remap(layer.ffn_down_exps) && remap(layer.ffn_gate_inp) &&
               remap(layer.ffn_gate_shexp) && remap(layer.ffn_up_shexp) &&
               remap(layer.ffn_down_shexp) &&
               remap(layer.ffn_gate_inp_shexp);
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  std::vector<void*> moe_converted;
  std::size_t moe_converted_bytes = 0;
  std::vector<float> moe_host_f32;
  for (auto& layer : weights_opt->layers) {
    if (!ConvertQ6KDownToQ80(layer, moe_converted, moe_converted_bytes,
                              moe_host_f32, error_msg)) {
      for (void* ptr : moe_converted) {
        (void)hipFree(ptr);
      }
      ReleaseWeightRegions(weight_regions);
      return nullptr;
    }
  }

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  auto model = std::make_shared<QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions), std::move(vision));
  model->moe_converted_ = std::move(moe_converted);
  model->moe_converted_bytes_ = moe_converted_bytes;
  return model;
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::Create(
    std::shared_ptr<const QwenGpuModel> model, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  if (model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen GPU model must not be null";
    }
    return nullptr;
  }

  const auto& config = model->GetConfig();
  const std::uint32_t model_context =
      config.context_length > 0 ? config.context_length : max_context;
  if (max_context == 0 || max_context > model_context) {
    if (error_msg != nullptr) {
      *error_msg = "Requested GPU context exceeds the model context length";
    }
    return nullptr;
  }

  // The blocked-Q8 path fails loudly here (never silently) when the model
  // cannot use the shared d == 256-only direct-read kernel.
  if (policy.UsesBlockedAttentionKv()) {
    std::string blocked_error;
    if (!ValidateQwenBlockedQ8Config(config, &blocked_error)) {
      if (error_msg != nullptr) {
        *error_msg = "blocked-Q8 attention rejected: " + blocked_error;
      }
      return nullptr;
    }
  }

  return std::make_unique<QwenGpuExecutor>(std::move(model), max_context,
                                           policy);
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  auto model = QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), error_msg, max_context, policy);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
