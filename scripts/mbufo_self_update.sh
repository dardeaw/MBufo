#!/usr/bin/env bash
set -euo pipefail

SERVER_PID="${1:?missing SERVER_PID}"
REPO_DIR="${2:?missing REPO_DIR}"
STATUS_FILE="${3:?missing STATUS_FILE}"
LOG_FILE="${4:?missing LOG_FILE}"
MAPPING_CACHE="${5:?missing MAPPING_CACHE}"

mkdir -p "$(dirname "$STATUS_FILE")" "$(dirname "$LOG_FILE")" "$(dirname "$MAPPING_CACHE")"
exec >>"$LOG_FILE" 2>&1

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUNNING_BEFORE="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
TARGET_SHA="$RUNNING_BEFORE"

write_status() {
  local phase="$1"
  local msg="$2"
  local err="${3:-}"
  python3 - "$STATUS_FILE" "$phase" "$RUNNING_BEFORE" "$TARGET_SHA" "$msg" "$err" "$STARTED_AT" <<'PY'
import json, os, sys, time
sf, phase, rbefore, tsha, msg, err, started = sys.argv[1:8]
obj = {
    "phase": phase,
    "active": phase in ("fetching", "building", "restarting"),
    "running_before": rbefore,
    "target_sha": tsha,
    "message": msg,
    "error": err,
    "started_at": started,
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
tmp = sf + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)
os.replace(tmp, sf)
PY
}

on_err() {
  local line="$1"
  echo "[self-update] FAILED at line $line"
  if [ -f "$REPO_DIR/build/release/mbufo.prev" ] && [ ! -f "$REPO_DIR/build/release/mbufo" ]; then
    mv -f "$REPO_DIR/build/release/mbufo.prev" "$REPO_DIR/build/release/mbufo" || true
  fi
  write_status "failed" "Update aborted cleanly; running server was left untouched." "Self-update failed at line $line (see $LOG_FILE)"
  exit 1
}
trap 'on_err $LINENO' ERR

echo "========================================"
echo "[self-update] started at $STARTED_AT (pid=$SERVER_PID, repo=$REPO_DIR)"

write_status "fetching" "Fetching latest main branch and release tags from GitHub (fork/main)..."
cd "$REPO_DIR"
git fetch fork main --tags
TARGET_SHA="$(git rev-parse fork/main)"

if ! git merge --ff-only fork/main; then
  echo "[self-update] git merge --ff-only fork/main failed (dirty or diverged local branch)"
  write_status "failed" "Fast-forward merge failed; local repo has uncommitted or diverged changes." "git merge --ff-only fork/main failed"
  exit 1
fi

if [ -f "$REPO_DIR/model_mapping.json" ]; then
  cp -f "$REPO_DIR/model_mapping.json" "$MAPPING_CACHE.tmp"
  mv -f "$MAPPING_CACHE.tmp" "$MAPPING_CACHE"
fi

write_status "building" "Rebuilding release binary (mbufo)..."
mkdir -p "$REPO_DIR/build/release"
if [ -f "$REPO_DIR/build/release/mbufo" ]; then
  cp -f "$REPO_DIR/build/release/mbufo" "$REPO_DIR/build/release/mbufo.bak"
  mv -f "$REPO_DIR/build/release/mbufo" "$REPO_DIR/build/release/mbufo.prev"
fi

if ! cmake --build "$REPO_DIR/build/release" --target mbufo -j 4; then
  echo "[self-update] cmake --build failed; rolling back previous binary"
  if [ -f "$REPO_DIR/build/release/mbufo.prev" ]; then
    mv -f "$REPO_DIR/build/release/mbufo.prev" "$REPO_DIR/build/release/mbufo"
  fi
  rm -f "$REPO_DIR/build/release/mbufo.bak"
  write_status "failed" "Build failed; rolled back previous binary without stopping running server." "cmake --build failed (see $LOG_FILE)"
  exit 1
fi

rm -f "$REPO_DIR/build/release/mbufo.prev" "$REPO_DIR/build/release/mbufo.bak"

write_status "restarting" "Build succeeded. Restarting mbufo service..."
sleep 1

if kill -0 "$SERVER_PID" 2>/dev/null; then
  kill -TERM "$SERVER_PID" || true
fi

HEALTH_PORT=$(python3 - <<'PY' 2>/dev/null || echo 8741
import json, os
cfg = os.environ.get("MBUFO_CONFIG_FILE", "/home/dardeaw/.config/mbufo/server_config.json")
try:
    with open(cfg, "r", encoding="utf-8") as f:
        p = int(json.load(f).get("port", 8741))
    print(p if 1 <= p <= 65535 else 8741)
except Exception:
    print(8741)
PY
)
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${HEALTH_PORT}/health" >/dev/null 2>&1; then
    write_status "completed" "Server updated to ${TARGET_SHA:0:8} and restarted cleanly."
    echo "[self-update] completed at $(date -u +%Y-%m-%dT%H:%M:%SZ) port=${HEALTH_PORT}"
    exit 0
  fi
  sleep 2
done

write_status "failed" "Restart timed out waiting for :${HEALTH_PORT}/health" "Health check timeout after restart"
exit 1
