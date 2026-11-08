# Llama registry pilot (needs weights + GPU); explicit-only, not part of the
# default or CI check sets.
if (ENGINE_ENABLE_HIP)
  add_executable(llama_registry_pilot_test EXCLUDE_FROM_ALL
    registry_pilot_test.cpp)
  target_link_libraries(llama_registry_pilot_test PRIVATE gufo_llama gufo_core
    hip::host)
endif()
