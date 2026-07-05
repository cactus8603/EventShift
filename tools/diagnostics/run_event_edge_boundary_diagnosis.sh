#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SPLIT="${1:-night}"
shift || true

if [[ "$SPLIT" == "day" ]]; then
  DATASET="cosec_day_val_event"
  WEIGHTS="work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_day.pth"
  OUT="work_dirs/diagnostics/event_edge_boundary_day.json"
elif [[ "$SPLIT" == "night" ]]; then
  DATASET="cosec_night_val_event"
  WEIGHTS="work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_night.pth"
  OUT="work_dirs/diagnostics/event_edge_boundary_night.json"
else
  echo "Usage: $0 [day|night] [extra diagnose_event_edge_boundary.py args...]" >&2
  exit 2
fi

conda run --no-capture-output -n mask2former \
  python tools/diagnose_event_edge_boundary.py \
  --rgb-config configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml \
  --rgb-weights "$WEIGHTS" \
  --dataset "$DATASET" \
  --device cuda:0 \
  --out "$OUT" \
  "$@"
