# Inkling-2-Mini (inkling2) text model package: 128 experts top-8, loop-core
# message passing, NVFP4 experts converted to F16 once at load. Reuses shared
# core GEMM (GemvBatched, LaunchDecodeMoeKQuant, RoutedCompact/RoutedF16Gemm,
# MoeEpilogue), blocked-Q8 KV, and QwenTokenizer (pre=inkling2).
# Skeleton revision: engine Forward returns not-implemented; geometry,
# load path, tokenizer and HTTP dispatch are complete.
add_library(mbufo_inkling STATIC
  config.cpp
  weights.cpp)

target_include_directories(mbufo_inkling PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mbufo_inkling PUBLIC mbufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_inkling PRIVATE
    kernels/kernels.hip.cpp
    vision.hip
    dspark.hip
    engine.cpp
    engine_adapter.cpp)
  set_source_files_properties(kernels/kernels.hip.cpp vision.hip dspark.hip PROPERTIES LANGUAGE HIP)
  target_compile_options(mbufo_inkling PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(mbufo_inkling PUBLIC
    mbufo_core_blocked_kv
    mbufo_core_gemm
    hip::host roc::hipblas)
endif()