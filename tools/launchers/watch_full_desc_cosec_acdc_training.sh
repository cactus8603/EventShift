#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
UNIFIED_ROOT="${ROOT}/unified_cosec_acdc/classcover_v1"
SYNC_SCRIPT="${ROOT}/swin_l/tools/sync_full_desc_cosec_acdc_best_checkpoints.sh"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${UNIFIED_ROOT}/training_logs/full_desc_cosec_acdc_watch_${STAMP}.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

active_processes() {
  pgrep -af 'launch_full_desc_cosec_acdc_all_models.sh|train_mask2former_cosec|train_maskdino_cosec|mmsegmentation/tools/train.py|BRENet/tools/train.py' \
    | grep -v 'watch_full_desc_cosec_acdc_training.sh' || true
}

sync_registry() {
  if [[ -x "${SYNC_SCRIPT}" ]]; then
    "${SYNC_SCRIPT}" >> "${LOG_FILE}" 2>&1
  else
    bash "${SYNC_SCRIPT}" >> "${LOG_FILE}" 2>&1
  fi
}

log "WATCH_START interval=${INTERVAL_SECONDS}s"
while :; do
  processes="$(active_processes)"
  if [[ -z "${processes}" ]]; then
    log "NO_ACTIVE_TRAINING final_sync=1"
    sync_registry
    log "WATCH_DONE"
    break
  fi

  log "ACTIVE_TRAINING sync=1"
  printf '%s\n' "${processes}" >> "${LOG_FILE}"
  sync_registry
  sleep "${INTERVAL_SECONDS}"
done
