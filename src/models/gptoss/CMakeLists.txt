# GPT-OSS (gpt-oss) MoE-decoder model package: GPT-OSS-20B text inference
# for gfx1151. Reuses the shared core GEMM launchers (narrow-prequant decode,
# DenseF16Gemm prefill, vec MoE), the shared blocked-KV store/dequant and
# the Qwen tokenizer (o200k vocabulary via the generic BPE path).
add_library(mbufo_gptoss STATIC
  config.cpp
  weights.cpp)

target_include_directories(mbufo_gptoss PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mbufo_gptoss PUBLIC mbufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_gptoss PRIVATE
    kernels/kernels.hip.cpp
    engine.cpp
    engine_adapter.cpp)
  set_source_files_properties(kernels/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(mbufo_gptoss PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(mbufo_gptoss PUBLIC
    mbufo_core
    mbufo_core_blocked_kv
    mbufo_core_gemm
    hip::host roc::hipblas)
endif()
