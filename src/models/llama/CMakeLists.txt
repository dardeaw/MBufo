# Llama (llama) dense-decoder model package: Llama-3.1-8B text inference for
# gfx1151. Reuses the shared core GEMM launchers and the Qwen tokenizer
# (extended with the llama-bpe pre-tokenizer mode).
add_library(mbufo_llama STATIC
  config.cpp
  weights.cpp)

target_include_directories(mbufo_llama PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mbufo_llama PUBLIC mbufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_llama PRIVATE
    kernels/kernels.hip.cpp
    engine.cpp
    engine_adapter.cpp)
  set_source_files_properties(kernels/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(mbufo_llama PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(mbufo_llama PUBLIC
    mbufo_core
    mbufo_core_blocked_kv
    mbufo_core_gemm
    hip::host roc::hipblas)
endif()
