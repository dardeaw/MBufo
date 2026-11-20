#!/bin/bash
# LM-head shaped GemvBatched (Q4_0 / Q8_0, m = 262144, k = 2816): current vs the pre-change gemv_wave32_sdot4.hip (git BASE, library flags).
#   lm_ab.sh [base-commit]
BASE=${1:-HEAD}
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R; D=${TMPDIR:-/tmp}/mbufobench/lmh; mkdir -p $D; cd $D
cat > xform.py <<'EOF'
import shlex, sys
toks = shlex.split(sys.argv[1]); out = []; i = 0
while i < len(toks):
    t = toks[i]
    if t == "-MD": i += 1; continue
    if t in ("-MT", "-MF"): i += 2; continue
    if t == "-o": out += ["-o", sys.argv[4]]; i += 2; continue
    if t.endswith(sys.argv[2]): out.append(sys.argv[3]); i += 1; continue
    out.append(t); i += 1
print("cd " + __import__("os").environ["R"] + "/build/release && " + " ".join(shlex.quote(t) for t in out))
EOF
f=gemv_wave32_sdot4
git -C $R show $BASE:src/core/compute/gemm/$f.hip | sed -e 's/namespace mbufo::core::compute::gemm {/namespace mbufo::core::compute::oldw { using namespace mbufo::core::compute::gemm;/' > ${f}_old.hip
cmd=$(cd $R/build/release && ninja -t commands libmbufo_core_gemm.a | grep "compute/gemm/$f.hip" | head -1)
python3 xform.py "$cmd" "compute/gemm/$f.hip" "$D/${f}_old.hip" "$D/${f}_old.o" > build_old.sh
bash build_old.sh 2>&1 | grep -E 'error' -A6 | head -30
F="-O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I$R -Wno-unused-value -Wno-ignored-attributes"
/opt/rocm/bin/hipcc $F -x hip $R/tools/bench/wave32_check/lm_test.cpp -x none ${f}_old.o $R/build/release/libmbufo_core_gemm.a $R/build/release/libmbufo_core.a -o lm_test 2>&1 | grep -E 'error|undefined' -A5 | head -30
ls -la lm_test
S() { sudo -n "$@"; }
restore() { sleep 2; S systemctl start mbufo; echo "restored: $(systemctl is-active mbufo)"; }
trap restore EXIT
S systemctl stop mbufo; sleep 5
LD_LIBRARY_PATH=/opt/rocm/lib timeout 600 ./lm_test
