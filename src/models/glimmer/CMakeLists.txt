# Muse Glimmer (muse-glimmer) dense-decoder model package: Glimmer 30B text
# inference for gfx1151. Reuses the shared core GEMM launchers and the Qwen
# tokenizer (extended with the llama4 pre-tokenizer mode).
add_library(gufo_glimmer STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_glimmer PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_glimmer PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_glimmer PRIVATE
    kernels/kernels.hip.cpp
    kernels/attn_f16_mqb.hip.cpp
    engine.cpp
    draft_model.cpp)
  set_source_files_properties(kernels/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/attn_f16_mqb.hip.cpp PROPERTIES
    LANGUAGE HIP)
  target_compile_options(gufo_glimmer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_glimmer PUBLIC
    gufo_core
    gufo_core_gemm
    gufo_core_blocked_kv
    hip::host roc::hipblas)
endif()
