#ifndef MBUFO_MODELS_QWEN3_TTS_HIP_ENCODER_SUPPORT_HPP_
#define MBUFO_MODELS_QWEN3_TTS_HIP_ENCODER_SUPPORT_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen3_tts/tensor.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

namespace mbufo::models::qwen3_tts::hip {

void RequireEncoderHip(hipError_t status, std::string_view operation);
void RequireEncoderRocblas(rocblas_status status, std::string_view operation);

template<typename Element>
class EncoderDeviceBuffer {
public:
  EncoderDeviceBuffer() = default;
  explicit EncoderDeviceBuffer(std::size_t count) { Reset(count); }

  ~EncoderDeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  EncoderDeviceBuffer(const EncoderDeviceBuffer&) = delete;
  EncoderDeviceBuffer& operator=(const EncoderDeviceBuffer&) = delete;

  EncoderDeviceBuffer(EncoderDeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  EncoderDeviceBuffer& operator=(EncoderDeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        (void)hipFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  void Reset(std::size_t count) {
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0) {
      return;
    }
    RequireEncoderHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc Qwen3-TTS encoder");
    count_ = count;
  }

  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

class EncoderWeights {
public:
  explicit EncoderWeights(const TensorStore& store) : store_(store) {}

  [[nodiscard]] const float* Load(std::string_view name,
                                  std::span<const std::uint64_t> shape);

  [[nodiscard]] const float* LoadBfloat16CodebookEmbedding(
      std::string_view embedding_sum_name, std::string_view cluster_usage_name,
      std::size_t codebook_size, std::size_t dimension);

private:
  const TensorStore& store_;
  std::vector<std::unique_ptr<EncoderDeviceBuffer<float>>> allocations_;
};

class EncoderF32Gemm {
public:
  EncoderF32Gemm();
  ~EncoderF32Gemm();

  EncoderF32Gemm(const EncoderF32Gemm&) = delete;
  EncoderF32Gemm& operator=(const EncoderF32Gemm&) = delete;

  void Run(const float* weight, const float* input, float* output,
           std::size_t rows, std::size_t output_columns, std::size_t reduction,
           hipStream_t stream);

private:
  rocblas_handle handle_{nullptr};
};

}  // namespace mbufo::models::qwen3_tts::hip
#endif

#endif  // MBUFO_MODELS_QWEN3_TTS_HIP_ENCODER_SUPPORT_HPP_
