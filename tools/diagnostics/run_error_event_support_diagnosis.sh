#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run --no-capture-output -n mask2former \
  python tools/diagnose_error_event_support.py \
  --rgb-config configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml \
  --rgb-weights work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_night.pth \
  --event-config configs/Mask2Former_SwinL_CoSEC_EventReliability_Exp2.yaml \
  --event-weights work_dirs/exp2_swinL_seqholdout305_cosec_event_reliability_epoch_from_day65/best_model_cosec_night.pth \
  --dataset cosec_night_val_event \
  --device cuda:0 \
  --out work_dirs/diagnostics/error_event_support_night.json \
  "$@"
