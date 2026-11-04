# Glimmer runner checks (unified mixed-batch DecodeBatch, incremental Sync,
# mixed 1+64 rows, f16/q8 short-sentence anchors; DFlash end-to-end).
# Built explicitly; not part of the default or CI check sets (needs weights).
if (ENGINE_ENABLE_HIP)
  add_executable(glimmer_decode_batch_test EXCLUDE_FROM_ALL
    decode_batch_test.cpp)
  target_link_libraries(glimmer_decode_batch_test PRIVATE gufo_glimmer
    gufo_core hip::host)
  add_executable(glimmer_dflash_test EXCLUDE_FROM_ALL
    dflash_test.cpp)
  # Whole-archive: the test validates the "glimmer-dflash" registry entry,
  # whose file-scope registrar lives in a TU no symbol references (static
  # libraries would otherwise drop it; serve references the backend for
  # real). HIP builds are Linux-only, so the GNU --whole-archive form is
  # fine here.
  target_link_libraries(glimmer_dflash_test PRIVATE
    -Wl,--whole-archive gufo_glimmer -Wl,--no-whole-archive
    gufo_core hip::host)
endif()
