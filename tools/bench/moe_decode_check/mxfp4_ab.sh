#!/bin/bash
# Compare the MXFP4 decode MoE kernels with their pre-change implementation (from git), compiled with the exact flags the
# library build uses for moe_fused.hip (floating-point contraction decides some results, so flags must match).
#   mxfp4_ab.sh <base-commit-with-old-moe_fused.hip>
BASE=${1:-3c79ab3a}
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R; D=${TMPDIR:-/tmp}/mbufobench/mdk; mkdir -p $D; cd $D
git -C $R show $BASE:src/core/compute/gemm/moe_fused.hip | sed -e 's/namespace mbufo::core::compute::gemm {/namespace mbufo::core::compute::oldmoe { using namespace mbufo::core::compute::gemm;/' > moe_fused_old.hip
cmd=$(cd $R/build/release && ninja -t commands libmbufo_core_gemm.a | grep 'compute/gemm/moe_fused.hip' | head -1)
python3 - "$cmd" "$D/moe_fused_old.hip" "$D/moe_fused_old.o" > build_old.sh <<'EOF'
import re, shlex, sys
toks = shlex.split(sys.argv[1]); out = []; i = 0
while i < len(toks):
    t = toks[i]
    if t in ("-MD",): i += 1; continue
    if t in ("-MT", "-MF"): i += 2; continue
    if t == "-o": out += ["-o", sys.argv[3]]; i += 2; continue
    if t.endswith("moe_fused.hip"): out.append(sys.argv[2]); i += 1; continue
    out.append(t); i += 1
print("cd " + __import__("os").environ["R"] + "/build/release && " + " ".join(shlex.quote(t) for t in out))
EOF
bash build_old.sh 2>&1 | grep -E 'error' -A6 | head -30
F="-O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I$R -Wno-unused-value -Wno-ignored-attributes"
/opt/rocm/bin/hipcc $F -x hip $R/tools/bench/moe_decode_check/mxfp4_test.cpp -x none moe_fused_old.o $R/build/release/libmbufo_core_gemm.a $R/build/release/libmbufo_core.a -o mxfp4_test 2>&1 | grep -E 'error|undefined' -A5 | head -40
ls -la mxfp4_test
