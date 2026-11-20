# bge-m3 (XLM-R encoder) leg-1 onboarding: CPU reference always, HIP
# prefill twin under ENGINE_ENABLE_HIP, verification driver under
# BUILD_TESTING. Sources join mbufo_core (serve + tests link one lib).
target_sources(mbufo_core PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/bge_m3/config.cpp
  ${PROJECT_SOURCE_DIR}/src/models/bge_m3/weights.cpp
  ${PROJECT_SOURCE_DIR}/src/models/bge_m3/tokenizer.cpp
  ${PROJECT_SOURCE_DIR}/src/models/bge_m3/forward.cpp
  ${PROJECT_SOURCE_DIR}/src/models/bge_m3/engine.cpp)

if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/models/bge_m3/hip/session.hip)
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/models/bge_m3/hip/session.hip
    PROPERTIES LANGUAGE HIP)
  # session.hip issues hipBLAS SGEMM directly (F32 precision-first prefill).
  target_link_libraries(mbufo_core PRIVATE roc::hipblas)
  # leg-3 flash attention lives in the shared blocked_kv lib (qwen precedent).
  target_link_libraries(mbufo_core PRIVATE mbufo_core_blocked_kv)
endif()

if (BUILD_TESTING)
  # bge_m3_embed: leg verification driver (registry -> tokenize -> Sync).
  add_executable(bge_m3_embed
    ${PROJECT_SOURCE_DIR}/src/models/bge_m3/driver.cpp)
  target_link_libraries(bge_m3_embed PRIVATE mbufo_core)
endif()
