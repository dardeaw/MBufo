#!/bin/bash
# Cross-model regression for shared (core) kernel changes: run the same fixed prompts on a BASE
# binary and a NEW binary for every model that uses the shared kernels, then compare speed and text.
#
#   tools/bench/regress/regress.sh <outdir> <base-gufo> <new-gufo> [--kernels] [--fn] [model-name ...]
#
# Stops gufo.service while it runs and always restores it. Models and flags live in the MODELS table.
# Greedy text can legitimately flip on near-ties when a kernel's summation order changes; identical text
# is the strong signal, "DIFFERENT" needs a look at the similarity and the first differing character.
# --kernels also runs the float64-reference kernel checks under tools/bench/*_check.
set -u
OUT=$1; BASE=$2; NEW=$3; shift 3
RUN_KERNELS=0; RUN_FN=0; ONLY=()
for a in "$@"; do case $a in --kernels) RUN_KERNELS=1;; --fn) RUN_FN=1;; *) ONLY+=("$a");; esac; done
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
G=/mnt/nvme-data/gufo-models
mkdir -p "$OUT"
PORT=8751
S() { sudo -n "$@" 2>/dev/null || (echo are23234 | sudo -S "$@" 2>/dev/null); }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
killsrv() { pkill -TERM -f "[s]erve llm.*-p $PORT" 2>/dev/null; for i in $(seq 1 40); do pgrep -f "[s]erve llm.*-p $PORT" >/dev/null || return; sleep 1; done; pkill -KILL -f "[s]erve llm.*-p $PORT" 2>/dev/null; sleep 2; }
restore() { killsrv; sleep 3; S systemctl start gufo; log "restored: $(systemctl is-active gufo)"; }
trap restore EXIT
export LD_LIBRARY_PATH=/opt/rocm/lib

# name | model | kv | extra env (space separated) | max context to test
MODELS=(
  "llama8b_q8|$(ls $G/llama-3.1-8b/*.gguf | head -1)|q8||6000"
  "llama8b_f16|$(ls $G/llama-3.1-8b/*.gguf | head -1)|f16||6000"
  "gptoss20_q8|$G/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf|q8||6000"
  "gptoss20_f16|$G/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf|f16||6000"
  "glimmerq8w_q8|$G/muse-glimmer-30b/Muse-Glimmer-30B-Q8_0.gguf|q8||2000"
  "glimmerq8w_f16|$G/muse-glimmer-30b/Muse-Glimmer-30B-Q8_0.gguf|f16||2000"
  "gemma26_q8|$G/gemma-4-26b/gemma-4-26B-A4B-it-Q4_0.gguf|q8||6000"
  "qwen27_q8|$G/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf|q8|GUFO_KV_WMMA=1 GUFO_KV_I8=2|6000"
)
[ $RUN_FN = 1 ] && MODELS+=("fn_fast_q8|$G/qwen3.8-rocmfp4/v2q8k/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2q8k.gguf|q8|GUFO_KV_WMMA=1 GUFO_KV_I8=2|6000")

log "stopping gufo.service"; S systemctl stop gufo; sleep 5

if [ $RUN_KERNELS = 1 ]; then
  log "== kernel checks"
  K=$OUT/kernels; mkdir -p $K
  # Each check: prep data (python), build the test with hipcc, run. The build line is in each tool's README docstring.
  I="-I$REPO -DENGINE_ENABLE_HIP=1 --offload-arch=gfx1151 -std=c++20 -O2"
  LIBG=$(find ${NEW%/release/gufo}/ -name libgufo_core_gemm.a 2>/dev/null | head -1)
  python3 $REPO/tools/bench/dequant_rows_check/dq_prep.py $K/dq >/dev/null 2>&1
  hipcc $I -x hip $REPO/tools/bench/dequant_rows_check/dq_test.cpp $REPO/src/core/compute/gemm/dequant_rows.hip -o $K/dq_test 2>/dev/null && $K/dq_test $K/dq > $K/dq.txt 2>&1; python3 $REPO/tools/bench/dequant_rows_check/dq_cmp.py $K/dq 2>&1 | tail -1 | sed 's/^/dequant_rows: /' | tee -a $OUT/run.log
  python3 $REPO/tools/bench/flash_decode_check/fd_prep.py $K/fd >/dev/null 2>&1
  hipcc $I -ffast-math -fno-finite-math-only -x hip $REPO/tools/bench/flash_decode_check/fd_test.cpp $REPO/src/core/compute/attention/flash_sink.hip.cpp -o $K/fd_test 2>/dev/null && $K/fd_test $K/fd 2>&1 | tail -1 | sed 's/^/flash_sink:   /' | tee -a $OUT/run.log
  python3 $REPO/tools/bench/blocked_splitk_check/bs_prep.py $K/bs >/dev/null 2>&1
  hipcc $I -ffast-math -fno-finite-math-only -x hip $REPO/tools/bench/blocked_splitk_check/bs_test.cpp $REPO/src/core/compute/attention/blocked_splitk.hip.cpp -o $K/bs_test 2>/dev/null && $K/bs_test $K/bs 2>&1 | tail -1 | sed 's/^/blocked_splitk: /' | tee -a $OUT/run.log
  if [ -n "$LIBG" ]; then
    python3 $REPO/tools/bench/narrow_quant_check/nq_prep.py $K/nq >/dev/null 2>&1
    hipcc $I -x hip $REPO/tools/bench/narrow_quant_check/nq_test.cpp $REPO/src/core/compute/gemm/gemv_narrow_quant.hip -x none $LIBG -o $K/nq_test 2>/dev/null && $K/nq_test $K/nq 2>&1 | grep -c " OK" | sed 's/^/narrow_quant: cases OK = /' | tee -a $OUT/run.log
  fi
fi

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name model kv envs maxctx <<< "$entry"
  if [ ${#ONLY[@]} -gt 0 ]; then keep=0; for o in "${ONLY[@]}"; do [ "$o" = "$name" ] && keep=1; done; [ $keep = 0 ] && continue; fi
  [ -f "$model" ] || { log "skip $name (missing $model)"; continue; }
  for tag in base new; do
    BIN=$BASE; [ $tag = new ] && BIN=$NEW
    for w in $(seq 1 90); do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "$a" -gt 40000 ] && break; sleep 2; done
    ( for e in $envs; do export "$e"; done
      exec $BIN serve llm --model "$model" --think off --reasoning-budget 0 -c 16384 -j 1 -p $PORT -t 0 \
        --speculative off --kv-quant $kv --served-model-name bench -i 127.0.0.1 ) > "$OUT/${name}_${tag}_srv.log" 2>&1 &
    ok=0
    for i in $(seq 1 200); do sleep 2
      grep -q "event=load_completed" "$OUT/${name}_${tag}_srv.log" && { sleep 2; ok=1; break; }
      pgrep -f "[s]erve llm.*-p $PORT" >/dev/null || break
    done
    if [ $ok = 1 ]; then
      python3 "$HERE/regress_client.py" $PORT $tag $maxctx > "$OUT/${name}_${tag}.jsonl" 2>&1
      log "$name/$tag done"
    else
      echo '{"error": "server failed to load"}' > "$OUT/${name}_${tag}.jsonl"; log "$name/$tag FAILED to load"
    fi
    killsrv; sleep 3
  done
done
log "== comparison"
python3 "$HERE/regress_cmp.py" "$OUT" | tee "$OUT/summary.txt" | tee -a "$OUT/run.log"
log ALL DONE
