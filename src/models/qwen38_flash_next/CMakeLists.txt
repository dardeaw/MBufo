# Qwen3.8-Flash-Next (qwen4exp) model package. The scalar oracle is built
# only by explicit validation targets, not by production or routine checks.
add_library(mbufo_qwen38_flash_next STATIC
  config.cpp
  continuation_layout.cpp
  ngram.cpp
  rope_scaling.cpp
  weights.cpp)

target_include_directories(mbufo_qwen38_flash_next PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mbufo_qwen38_flash_next PUBLIC mbufo_core Threads::Threads)

add_library(mbufo_qwen38_flash_next_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(mbufo_qwen38_flash_next_reference PUBLIC
  mbufo_qwen38_flash_next)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(QFN_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(QFN_MMQ_DIR "${QFN_ROCM_DIR}/mmq")

  # HIP quantized matrix kernels adapted from llama.cpp, isolated in the
  # qfn_mmq namespace. The adapter retains its C++17 dialect.
  set(QFN_MMQ_SOURCES
    ${QFN_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
    ${QFN_MMQ_DIR}/qfn_mmq.hip.cpp
    ${QFN_MMQ_DIR}/mmid.hip.cpp
    ${QFN_MMQ_DIR}/mmvq.hip.cpp
    ${QFN_MMQ_DIR}/moe_wave64.hip.cpp
    ${QFN_MMQ_DIR}/quantize.hip.cpp)

  add_library(mbufo_qwen38_flash_next_mmq STATIC ${QFN_MMQ_SOURCES})
  set_source_files_properties(${QFN_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(mbufo_qwen38_flash_next_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(mbufo_qwen38_flash_next_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  target_compile_options(mbufo_qwen38_flash_next_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(mbufo_qwen38_flash_next_mmq SYSTEM PUBLIC ${QFN_MMQ_DIR})
  target_link_libraries(mbufo_qwen38_flash_next_mmq PUBLIC hip::host roc::hipblas)

  set(QFN_HIP_SOURCES
    continuation_adapter.cpp
    ${QFN_ROCM_DIR}/kernels.hip.cpp
    ${QFN_ROCM_DIR}/w8a8_wave64.hip
    ${QFN_ROCM_DIR}/blaslt.cpp
    ${QFN_ROCM_DIR}/device_model.cpp
    ${QFN_ROCM_DIR}/executor.cpp
    ${QFN_ROCM_DIR}/batch.cpp
    draft_backend.cpp
    engine.cpp)
  target_sources(mbufo_qwen38_flash_next PRIVATE ${QFN_HIP_SOURCES})
  set_source_files_properties(${QFN_ROCM_DIR}/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(mbufo_qwen38_flash_next PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(mbufo_qwen38_flash_next PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(mbufo_qwen38_flash_next SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(mbufo_qwen38_flash_next PUBLIC
    mbufo_cache_hip
    mbufo_core_blocked_kv
    mbufo_core_gemm
    mbufo_qwen38_flash_next_mmq hip::host roc::hipblas roc::hipblaslt)

  if (MBUFO_BUILD_TOOLS AND NOT BUILD_TESTING)
    add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
      ${PROJECT_SOURCE_DIR}/tests/models/qwen38_flash_next/continuation_probe.cpp)
    target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
      mbufo_qwen38_flash_next)
  endif()
endif()

# L1 registry adapter (EngineModel/SpeculativeSession over flash-next
# Model/Session); HIP-only like the rest of the package.
if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_qwen38_flash_next PRIVATE engine_adapter.cpp)
endif()
