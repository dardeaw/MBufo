# GPT-OSS registry pilot (needs weights + GPU); explicit-only, not part of
# the default or CI check sets.
if (ENGINE_ENABLE_HIP)
  add_executable(gptoss_registry_pilot_test EXCLUDE_FROM_ALL
    registry_pilot_test.cpp)
  target_link_libraries(gptoss_registry_pilot_test PRIVATE gufo_gptoss gufo_core
    hip::host)
  add_executable(gptoss_bench_test EXCLUDE_FROM_ALL
    bench_test.cpp)
  target_link_libraries(gptoss_bench_test PRIVATE gufo_gptoss gufo_core
    hip::host)
endif()
