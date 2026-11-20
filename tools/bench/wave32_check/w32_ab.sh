#!/bin/bash
# Compare gemv_wave32_sdot4.hip / moe_fused.hip kernels with their pre-change implementation (from git), each compiled with the exact
# flags the library build uses for that file (floating-point contraction decides results, so flags must match).
#   w32_ab.sh <base-commit>
BASE=${1:-ba88580c}
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R; D=${TMPDIR:-/tmp}/mbufobench/w32; mkdir -p $D; cd $D
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
for f in gemv_wave32_sdot4 moe_fused; do
  git -C $R show $BASE:src/core/compute/gemm/$f.hip | sed -e 's/namespace mbufo::core::compute::gemm {/namespace mbufo::core::compute::oldw { using namespace mbufo::core::compute::gemm;/' > ${f}_old.hip
  cmd=$(cd $R/build/release && ninja -t commands libmbufo_core_gemm.a | grep "compute/gemm/$f.hip" | head -1)
  python3 xform.py "$cmd" "compute/gemm/$f.hip" "$D/${f}_old.hip" "$D/${f}_old.o" > build_$f.sh
  bash build_$f.sh 2>&1 | grep -E 'error' -A6 | head -30
done
F="-O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I$R -Wno-unused-value -Wno-ignored-attributes"
/opt/rocm/bin/hipcc $F -x hip $R/tools/bench/wave32_check/w32_test.cpp -x none gemv_wave32_sdot4_old.o moe_fused_old.o $R/build/release/libmbufo_core_gemm.a $R/build/release/libmbufo_core.a -o w32_test 2>&1 | grep -E 'error|undefined' -A5 | head -40
ls -la w32_test
