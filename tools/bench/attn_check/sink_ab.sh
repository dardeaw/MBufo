#!/bin/bash
# Build and run sink_test (gpt-oss decode attention over blocked Q8). Needs the library built first.
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R; D=${TMPDIR:-/tmp}/mbufobench/sink; mkdir -p $D; cd $D
S() { sudo -n "$@"; }
F="-O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I$R -Wno-unused-value -Wno-ignored-attributes"
/opt/rocm/bin/hipcc $F -x hip $R/tools/bench/attn_check/sink_test.cpp -x none $R/build/release/libmbufo_core_blocked_kv.a $R/build/release/libmbufo_core.a -o sink_test 2>&1 | grep -E 'error|undefined' -A5 | head -40
ls -la sink_test
restore() { sleep 2; S systemctl start mbufo; echo "restored: $(systemctl is-active mbufo)"; }
trap restore EXIT
S systemctl stop mbufo; sleep 5
LD_LIBRARY_PATH=/opt/rocm/lib timeout 900 ./sink_test
echo "exit=$?"
