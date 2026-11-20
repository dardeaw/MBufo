#ifndef MBUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_BFLOAT16_HPP_
#define MBUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_BFLOAT16_HPP_

#include <cstdint>
#include <cstring>

namespace mbufo::test {

[[nodiscard]] inline std::uint16_t FloatToBf16Bits(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}

[[nodiscard]] inline float Bf16BitsToFloat(std::uint16_t bits) noexcept {
  const std::uint32_t float_bits = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0F;
  std::memcpy(&value, &float_bits, sizeof(value));
  return value;
}

}  // namespace mbufo::test

#endif  // MBUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_BFLOAT16_HPP_
