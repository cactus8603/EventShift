#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CUDA_VISIBLE_DEVICES:=1}"

conda run --no-capture-output -n "${CONDA_ENV:-mask2former}" \
  python tools/diagnose_tta_event_class_routing.py \
  --base-config configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml \
  --base-weights work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7/best_model_cosec_day.pth \
  --event-config configs/Mask2Former_SwinL_CoSEC_DayExp1C_StageC_Head.yaml \
  --event-weights work_dirs/day-exp1c_stageC_head_from_stageB_strong_bs8/model_final.pth \
  --dataset cosec_day_val_event \
  --flip \
  --out work_dirs/diagnostics/tta_event_class_routing_day_full.json \
  "$@"
