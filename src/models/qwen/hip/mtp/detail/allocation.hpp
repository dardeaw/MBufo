#ifndef MBUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
#define MBUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/managed_alloc.hpp"

namespace mbufo::hip::detail {

[[nodiscard]] inline void* AllocateDevice(std::size_t bytes) {
  void* pointer = nullptr;
  const auto error = mbufo::hip::AllocateDevice(&pointer, bytes);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP allocation failed: ") +
                             hipGetErrorString(error));
  }
  return pointer;
}

}  // namespace mbufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // MBUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
