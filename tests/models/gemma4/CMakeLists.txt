# Gemma 4 runner checks (unified mixed-batch DecodeBatch, incremental Sync).
# Built explicitly; not part of the default or CI check sets (needs weights).
if (ENGINE_ENABLE_HIP)
  add_executable(gemma4_decode_batch_test EXCLUDE_FROM_ALL
    decode_batch_test.cpp)
  target_link_libraries(gemma4_decode_batch_test PRIVATE gufo_gemma4
    gufo_qwen38_flash_next gufo_core hip::host)
endif()
# Gemma4 registry pilot (direct adapter vs registry, DecodeStep
# fallback, adapter DecodeBatch translation, MTP wiring, q8 smoke;
# needs weights + GPU). Explicit-only, not part of the default or CI check sets.
if (ENGINE_ENABLE_HIP)
  add_executable(gemma4_registry_pilot_test EXCLUDE_FROM_ALL
    registry_pilot_test.cpp)
  target_link_libraries(gemma4_registry_pilot_test PRIVATE gufo_gemma4
    gufo_qwen38_flash_next gufo_core hip::host)
endif()
