# Laguna 2.1 (laguna) model package: Laguna-XS-2.1 (33B-A3B) and Laguna-S-2.1 (118B-A8B)
# for gfx1151. Reuses shared core GEMM (GemvBatched, DenseF16Gemm, LaunchDecodeMoeKQuant,
# RoutedCompact/RoutedF16Gemm), blocked-Q8 KV, and QwenTokenizer (pre=laguna).
add_library(mbufo_laguna STATIC
  config.cpp
  weights.cpp)

target_include_directories(mbufo_laguna PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mbufo_laguna PUBLIC mbufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_laguna PRIVATE
    kernels/kernels.hip.cpp
    engine.cpp
    engine_adapter.cpp)
  set_source_files_properties(kernels/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(mbufo_laguna PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(mbufo_laguna PUBLIC
    mbufo_core_blocked_kv
    mbufo_core_gemm
    hip::host roc::hipblas)
endif()
