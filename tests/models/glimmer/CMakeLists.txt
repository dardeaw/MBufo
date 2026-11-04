# Glimmer runner checks (unified mixed-batch DecodeBatch, incremental Sync,
# mixed 1+64 rows, f16/q8 short-sentence anchors).
# Built explicitly; not part of the default or CI check sets (needs weights).
if (ENGINE_ENABLE_HIP)
  add_executable(glimmer_decode_batch_test EXCLUDE_FROM_ALL
    decode_batch_test.cpp)
  target_link_libraries(glimmer_decode_batch_test PRIVATE gufo_glimmer
    gufo_core hip::host)
endif()
