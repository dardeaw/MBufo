# EmbeddingGemma 2 (text path) encoder. Sources join mbufo_core.
target_sources(mbufo_core PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/embeddinggemma2/engine.cpp)

if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/models/embeddinggemma2/hip/session.hip)
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/models/embeddinggemma2/hip/session.hip
    PROPERTIES LANGUAGE HIP)
  target_link_libraries(mbufo_core PRIVATE roc::hipblas)
endif()
