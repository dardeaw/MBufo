#!/bin/bash
# Build and run attn_test (decode attention at qwen35moe's geometry). Needs the library built first.
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R; D=${TMPDIR:-/tmp}/mbufobench/attn; mkdir -p $D; cd $D
S() { sudo -n "$@"; }
F="$EXTRA_FLAGS -O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I$R -Wno-unused-value -Wno-ignored-attributes"
/opt/rocm/bin/hipcc $F -x hip $R/tools/bench/attn_check/attn_test.cpp -x none $R/build/release/libmbufo_core_blocked_kv.a $R/build/release/libmbufo_core.a -o ${OUTNAME:-attn_test} 2>&1 | grep -E 'error|undefined' -A5 | head -40
ls -la ${OUTNAME:-attn_test}
restore() { sleep 2; S systemctl start mbufo; echo "restored: $(systemctl is-active mbufo)"; }
trap restore EXIT
S systemctl stop mbufo; sleep 5
LD_LIBRARY_PATH=/opt/rocm/lib timeout 900 ./${OUTNAME:-attn_test}
echo "exit=$?"
