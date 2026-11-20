#!/bin/bash
# Gemma 4 AttentionCross: current kernels vs the pre-change file (git BASE), both compiled with the library's flags.
#   gemma_ab.sh [base-commit]
BASE=${1:-HEAD}
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R; D=${TMPDIR:-/tmp}/mbufobench/gattn; mkdir -p $D; cd $D
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
f=kernels
git -C $R show $BASE:src/models/gemma4/kernels/kernels.hip.cpp | sed -e 's/^namespace kernels {/namespace oldkernels { using namespace mbufo::models::gemma4::kernels;/' > gemma_old.hip.cpp
cmd=$(cd $R/build/release && ninja -t commands libmbufo_gemma4.a | grep "gemma4/kernels/kernels.hip.cpp" | head -1)
python3 xform.py "$cmd" "gemma4/kernels/kernels.hip.cpp" "$D/gemma_old.hip.cpp" "$D/gemma_old.o" > build_old.sh
bash build_old.sh 2>&1 | grep -E 'error' -A6 | head -30
F="-O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I$R -Wno-unused-value -Wno-ignored-attributes"
/opt/rocm/bin/hipcc $F -x hip $R/tools/bench/gemma_attn_check/gemma_test.cpp -x none gemma_old.o $R/build/release/src/models/gemma4/libmbufo_gemma4.a $R/build/release/libmbufo_core_blocked_kv.a $R/build/release/libmbufo_core_gemm.a $R/build/release/libmbufo_core.a -L/opt/rocm/lib -lhipblas -lhipblaslt -o gemma_test 2>&1 | grep -E 'error|undefined' -A5 | head -30
ls -la gemma_test
S() { sudo -n "$@"; }
restore() { sleep 2; S systemctl start mbufo; echo "restored: $(systemctl is-active mbufo)"; }
trap restore EXIT
S systemctl stop mbufo; sleep 5
LD_LIBRARY_PATH=/opt/rocm/lib timeout 600 ./gemma_test
