# Llama (llama) dense-decoder model package: Llama-3.1-8B text inference for
# gfx1151. Reuses the shared core GEMM launchers and the Qwen tokenizer
# (extended with the llama-bpe pre-tokenizer mode).
add_library(gufo_llama STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_llama PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_llama PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_llama PRIVATE
    kernels/kernels.hip.cpp
    engine.cpp)
  set_source_files_properties(kernels/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_llama PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_llama PUBLIC
    gufo_core
    gufo_core_blocked_kv
    gufo_core_gemm
    hip::host roc::hipblas)
endif()
