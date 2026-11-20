#ifndef MBUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_TENSOR_TYPES_H_
#define MBUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_TENSOR_TYPES_H_

#include <cstdint>

inline constexpr uint32_t DS4_TENSOR_F32 = 0;
inline constexpr uint32_t DS4_TENSOR_F16 = 1;
inline constexpr uint32_t DS4_TENSOR_Q8_0 = 8;
inline constexpr uint32_t DS4_TENSOR_Q2_K = 10;
inline constexpr uint32_t DS4_TENSOR_Q4_K = 12;
inline constexpr uint32_t DS4_TENSOR_IQ2_XXS = 16;
inline constexpr uint32_t DS4_TENSOR_I32 = 26;

#endif  // MBUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_TENSOR_TYPES_H_
