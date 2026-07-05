#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
LOG_DIR="$ROOT/work_dirs/launch_logs"
mkdir -p "$LOG_DIR"

GPU_ID="${GPU_ID:-0}"
MODE="${MODE:-all}"
SAVE_MAPS="${SAVE_MAPS:-1}"
LIMIT="${LIMIT:-}"
SLEEP_SEC="${SLEEP_SEC:-300}"

train_patterns=(
  "SegFormer_B5_CoSEC_Continue_FromOldDay.py"
  "SegFormer_B5_ACDC_All_FromOldNight.py"
)

training_still_running() {
  local pattern
  for pattern in "${train_patterns[@]}"; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

echo "[$(date '+%F %T')] Waiting for active SegFormer training jobs before ensemble cache."
while training_still_running; do
  echo "[$(date '+%F %T')] Training still running; sleep ${SLEEP_SEC}s."
  sleep "$SLEEP_SEC"
done

echo "[$(date '+%F %T')] Training finished. Start ensemble feature cache: MODE=$MODE GPU_ID=$GPU_ID SAVE_MAPS=$SAVE_MAPS LIMIT=${LIMIT:-none}"
cd "$ROOT"
GPU_ID="$GPU_ID" SAVE_MAPS="$SAVE_MAPS" LIMIT="$LIMIT" \
  bash "$ROOT/tools/launch_ensemble_feature_cache_presets.sh" "$MODE"
echo "[$(date '+%F %T')] Ensemble feature cache finished."
