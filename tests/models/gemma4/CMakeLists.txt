# Gemma 4 runner checks (unified mixed-batch DecodeBatch, incremental Sync).
# Built explicitly; not part of the default or CI check sets (needs weights).
if (ENGINE_ENABLE_HIP)
  add_executable(gemma4_decode_batch_test EXCLUDE_FROM_ALL
    decode_batch_test.cpp)
  target_link_libraries(gemma4_decode_batch_test PRIVATE gufo_gemma4
    gufo_qwen38_flash_next gufo_core hip::host)
endif()
