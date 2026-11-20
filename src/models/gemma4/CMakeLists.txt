# Gemma 4 (gemma4) model package: Gemma 4 26B A4B text inference for
# gfx1151. Reuses the Flash-Next mmq GEMM library and the Qwen tokenizer
# (extended with the Gemma 4 sentencepiece-BPE mode).
add_library(mbufo_gemma4 STATIC
  config.cpp
  weights.cpp)

target_include_directories(mbufo_gemma4 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mbufo_gemma4 PUBLIC mbufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_gemma4 PRIVATE
    kernels/kernels.hip.cpp
    kernels/attention_wmma.hip.cpp
    kernels/attention_wmma512.hip.cpp
    engine.cpp
    draft_backend.cpp
    mtp.hip.cpp)
  set_source_files_properties(kernels/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/attention_wmma.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/attention_wmma512.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(mtp.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(mbufo_gemma4 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(mbufo_gemma4 PUBLIC
    mbufo_qwen38_flash_next
    mbufo_core_blocked_kv
    mbufo_core_gemm
    hip::host roc::hipblas)
endif()
# L1 registry adapter (EngineModel/SpeculativeSession over gemma4
# Model/Session); HIP-only like the rest of the package.
if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_gemma4 PRIVATE engine_adapter.cpp)
endif()
