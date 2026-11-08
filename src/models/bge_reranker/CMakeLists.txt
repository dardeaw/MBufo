# bge-reranker (XLM-R sequence-classification) onboarding: CPU reference
# always, HIP prefill via the shared bge_m3 encoder twin, verification
# driver under BUILD_TESTING. Sources join gufo_core (serve + tests link
# one lib).
target_sources(gufo_core PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/bge_reranker/config.cpp
  ${PROJECT_SOURCE_DIR}/src/models/bge_reranker/weights.cpp
  ${PROJECT_SOURCE_DIR}/src/models/bge_reranker/engine.cpp)

if (BUILD_TESTING)
  # bge_rerank_score: leg verification driver (registry -> pair-encode ->
  # Score -> ranking JSON).
  add_executable(bge_rerank_score
    ${PROJECT_SOURCE_DIR}/src/models/bge_reranker/driver.cpp)
  target_link_libraries(bge_rerank_score PRIVATE gufo_core)
endif()
