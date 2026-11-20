#!/bin/bash
# Build and run qt_test (no GPU needed; host decode of the shared quant Traits against the ggml_dequant oracles).
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R; D=${TMPDIR:-/tmp}/mbufobench/qt; mkdir -p $D; cd $D
F="-O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I$R -Wno-unused-value -Wno-ignored-attributes"
/opt/rocm/bin/hipcc $F -x hip $R/tools/bench/quant_traits_check/qt_test.cpp -x none $R/build/release/libmbufo_core.a -o qt_test 2>&1 | grep -E 'error|undefined' -A5 | head -40
LD_LIBRARY_PATH=/opt/rocm/lib ./qt_test
