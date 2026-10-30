# Gemma 4 (gemma4) model package: Gemma 4 26B A4B text inference for
# gfx1151. Reuses the Flash-Next mmq GEMM library and the Qwen tokenizer
# (extended with the Gemma 4 sentencepiece-BPE mode).
add_library(gufo_gemma4 STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_gemma4 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_gemma4 PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_gemma4 PRIVATE
    kernels/kernels.hip.cpp
    kernels/attention_wmma.hip.cpp
    engine.cpp)
  set_source_files_properties(kernels/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/attention_wmma.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_gemma4 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_gemma4 PUBLIC
    gufo_qwen38_flash_next
    hip::host roc::hipblas)
endif()
