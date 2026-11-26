#!/bin/bash
# Build and run kv_test (host trait checks + device roundtrip/attention smoke).
# Host part needs no GPU; the device part needs one ROCm GPU on this machine.
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R
B=${MBUFO_BUILD:-$R/build/inkling}
D=${TMPDIR:-/tmp}/mbufobench/kvcodec; mkdir -p $D; cd $D
LIB=$B/libmbufo_core_blocked_kv.a
if [ ! -f "$LIB" ]; then echo "missing $LIB (build mbufo first)"; exit 2; fi
F_HOST="-O2 -std=c++17 -I$R -Wall"
echo "--- host test (plain g++, no ROCm: proves the header is dependency-free) ---"
g++ $F_HOST $R/tools/bench/kv_codec_check/kv_test_host.cpp -o kv_test_host || exit 1
./kv_test_host || exit 1
F="-O2 -std=c++20 -I$R -I/opt/rocm/include"
echo "--- device test (hipcc, needs GPU) ---"
/opt/rocm/bin/hipcc $F -x hip $R/tools/bench/kv_codec_check/kv_test_dev.cpp -x none $LIB -o kv_test_dev 2>&1 | grep -E 'error' -A5 | head -30
LD_LIBRARY_PATH=/opt/rocm/lib ./kv_test_dev || exit 1
echo ALL OK
