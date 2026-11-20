# qwen-reranker (Qwen causal yes/no head) onboarding: HIP-only (it drives
# the Qwen dense executor), verification driver under BUILD_TESTING.
# Sources join mbufo_core (serve + tests link one lib).
if (ENGINE_ENABLE_HIP)
  target_sources(mbufo_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/models/qwen_reranker/engine.cpp)

  if (BUILD_TESTING)
    # qwen_rerank_score: leg verification driver (registry -> template
    # render -> Score -> ranking JSON).
    add_executable(qwen_rerank_score
      ${PROJECT_SOURCE_DIR}/src/models/qwen_reranker/driver.cpp)
    target_link_libraries(qwen_rerank_score PRIVATE mbufo_core)
  endif()
endif()
