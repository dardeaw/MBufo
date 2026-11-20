# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

function(add_qwen38_flash_next_test name library labels)
  set(target qwen38_flash_next_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen38_flash_next.${name} COMMAND ${target})
  set_tests_properties(qwen38_flash_next.${name} PROPERTIES
    LABELS "qwen38_flash_next;${labels}")
  add_dependencies(qwen38_flash_next_tests ${target})
endfunction()

add_qwen38_flash_next_test(config mbufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(continuation_layout mbufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(ngram mbufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(mtp_sampling mbufo_core "cpu")
add_qwen38_flash_next_test(rope_scaling mbufo_qwen38_flash_next "cpu")
# The HF-oracle/off-bit-identity checks link the scalar model oracle, so
# (like ngram) this stays out of cmake/Checks.cmake's hosted check-pr set.
add_qwen38_flash_next_test(rope_scaling_oracle
  mbufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(draft_vocab mbufo_core "cpu")
add_qwen38_flash_next_test(lookup_draft mbufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(prompt_lookup mbufo_core "cpu")

add_executable(qwen38_flash_next_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  mbufo_qwen38_flash_next_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops gdn_ops
      attention_ops moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops
      select_ops snapshot_copy verification_sampling)
    add_qwen38_flash_next_test(${name} mbufo_qwen38_flash_next "hip;gpu;gfx1151")
  endforeach()
  # The rope phase and RMS checks compare against the scalar CPU oracle.
  target_link_libraries(qwen38_flash_next_attention_ops_test PRIVATE
    mbufo_qwen38_flash_next_reference)

  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(qwen38_flash_next_session_test PRIVATE mbufo_logging
    mbufo_qwen38_flash_next mbufo_deepseek_v4_flash mbufo_cache_hip hip::host)
  add_executable(qwen38_flash_next_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp
    device_headroom_fault.cpp)
  target_link_libraries(qwen38_flash_next_snapshot_test PRIVATE
    mbufo_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_adapter_test EXCLUDE_FROM_ALL
    continuation_adapter_test.cpp reset_submission_fault.cpp)
  target_link_libraries(qwen38_flash_next_continuation_adapter_test PRIVATE
    mbufo_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
    continuation_probe.cpp)
  target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
    mbufo_qwen38_flash_next)
  add_custom_target(qwen38_flash_next_model_tests
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test
      qwen38_flash_next_continuation_adapter_test)

  add_executable(qwen38_flash_next_dense_gemm_bench EXCLUDE_FROM_ALL
    dense_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_dense_gemm_bench PRIVATE
    mbufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_attn_bench EXCLUDE_FROM_ALL attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_attn_bench PRIVATE
    mbufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_decode_attn_bench EXCLUDE_FROM_ALL
    decode_attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_decode_attn_bench PRIVATE
    mbufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_routed_gemm_bench EXCLUDE_FROM_ALL
    routed_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_routed_gemm_bench PRIVATE
    mbufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp
    mtp_audit.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    mbufo_qwen38_flash_next_reference)
endif()

# Engine-level mixed-batch check (P2): EvaluateBatchWithChunk vs legacy twins.
# Explicit-only (needs weights + GPU); not in the default or CI check sets.
if (ENGINE_ENABLE_HIP)
  add_executable(qwen38_flash_next_engine_mixed_test EXCLUDE_FROM_ALL
    engine_mixed_test.cpp)
  target_link_libraries(qwen38_flash_next_engine_mixed_test PRIVATE
    mbufo_qwen38_flash_next)
endif()

# Flash-next registry pilot (direct adapter vs registry, DecodeStep
# draft-verify path, adapter DecodeBatch/EvaluateBatch translation, snapshot
# round-trip, SyncThrough smoke, MTP wiring, q8 smoke; needs weights + GPU).
# Explicit-only, not part of the default or CI check sets.
if (ENGINE_ENABLE_HIP)
  add_executable(qwen38_flash_next_registry_pilot_test EXCLUDE_FROM_ALL
    registry_pilot_test.cpp)
  target_link_libraries(qwen38_flash_next_registry_pilot_test PRIVATE
    mbufo_qwen38_flash_next mbufo_core hip::host)
endif()
