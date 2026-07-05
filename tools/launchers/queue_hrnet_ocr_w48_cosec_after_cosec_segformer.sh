#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
LOG_DIR="$ROOT/work_dirs/launch_logs"
mkdir -p "$LOG_DIR"

GPU_ID="${GPU_ID:-0}"
SLEEP_SEC="${SLEEP_SEC:-60}"
WAIT_PATTERN="SegFormer_B5_CoSEC_Continue_FromOldDa[y].py"

echo "[$(date '+%F %T')] Waiting for CoSEC SegFormer to finish before HRNet-OCR-W48 CoSEC."
while pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; do
  echo "[$(date '+%F %T')] CoSEC SegFormer still running; sleep ${SLEEP_SEC}s."
  sleep "$SLEEP_SEC"
done

echo "[$(date '+%F %T')] Starting HRNet-OCR-W48 CoSEC on GPU ${GPU_ID}."
GPU_ID="$GPU_ID" "$ROOT/tools/launch_hrnet_ocr_w48_mmseg.sh" cosec
