# bge-m3 tests mirror the model layout: CPU properties always, tokenizer
# HF-parity gated on GUFO_BGE_MODEL (exit 77 skip), HIP agreement gated the
# same way and excluded from fast presets via the external-model label.
function(add_bge_m3_cpu_test target source labels)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "${labels}")
endfunction()

add_bge_m3_cpu_test(bge_m3_forward_test
  tests/models/bge_m3/forward_test.cpp
  "cpu;models;bge-m3;forward")
add_bge_m3_cpu_test(bge_m3_tokenizer_test
  tests/models/bge_m3/tokenizer_test.cpp
  "cpu;models;bge-m3;tokenizer")
target_compile_definitions(bge_m3_tokenizer_test PRIVATE
  GUFO_BGE_TESTDATA_DIR="${PROJECT_SOURCE_DIR}/tests/models/bge_m3/testdata")
set_tests_properties(bge_m3_tokenizer_test PROPERTIES SKIP_RETURN_CODE 77)

if (ENGINE_ENABLE_HIP)
  add_executable(bge_m3_hip_test
    ${PROJECT_SOURCE_DIR}/tests/models/bge_m3/hip_test.cpp)
  target_link_libraries(bge_m3_hip_test PRIVATE
    gufo_core hip::host roc::hipblas roc::rocblas)
  add_test(NAME bge_m3_hip_test COMMAND bge_m3_hip_test)
  set_tests_properties(bge_m3_hip_test PROPERTIES
    LABELS "hip;gpu;bge-m3;external-model"
    SKIP_RETURN_CODE 77)
endif()
