#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CUDA_VISIBLE_DEVICES:=0}"

conda run --no-capture-output -n "${CONDA_ENV:-mask2former}" \
  python tools/diagnose_tta_event_class_routing.py \
  --base-config configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml \
  --base-weights work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7/best_model_cosec_day.pth \
  --event-config configs/Mask2Former_SwinL_CoSEC_FullCoSEC_Exp11B_Night50EventActiveUncertainScore.yaml \
  --event-weights work_dirs/fullcosec-exp11b_night50_event_uncertain_score_bs6/best_model_cosec_night.pth \
  --dataset cosec_night_val_event \
  --flip \
  --out work_dirs/diagnostics/tta_event_class_routing_night_full.json \
  "$@"
