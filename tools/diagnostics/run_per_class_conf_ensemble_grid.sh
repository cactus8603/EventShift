#!/usr/bin/env bash
set -euo pipefail

ROOT="."
CONDA="${CONDA:-conda}"
ENV_NAME="${ENV_NAME:-mask2former}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/work_dirs/ensemble_feature_cache}"
OUT_ROOT="${OUT_ROOT:-$ROOT/work_dirs/ensemble_per_class_conf}"
WAIT="${WAIT:-0}"
SLEEP_SEC="${SLEEP_SEC:-60}"

models=(
  "swinl_day65_4352"
  "swinl_night50_4284"
  "segformer_b5_cosec_day_best"
  "segformer_b5_cosec_night_best_old"
)

wait_for_caches() {
  local ready=0
  while [[ "$ready" == "0" ]]; do
    ready=1
    for model in "${models[@]}"; do
      if [[ ! -f "$CACHE_ROOT/$model/cosec_day_val_per_class_iou.csv" ]] || \
         [[ ! -f "$CACHE_ROOT/$model/cosec_night_val_per_class_iou.csv" ]] || \
         [[ ! -d "$CACHE_ROOT/$model/maps/cosec_day_val" ]] || \
         [[ ! -d "$CACHE_ROOT/$model/maps/cosec_night_val" ]]; then
        ready=0
      fi
    done
    if [[ "$ready" == "0" ]]; then
      echo "[$(date '+%F %T')] Waiting for ensemble caches..."
      sleep "$SLEEP_SEC"
    fi
  done
}

run_one() {
  local dataset="$1"
  local name="$2"
  local anchor="$3"
  shift 3
  echo "[$(date '+%F %T')] Run ${dataset}/${name}"
  PYTHONNOUSERSITE=1 "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python "$ROOT/tools/ensemble_cache_per_class_confidence.py" \
      --dataset "$dataset" \
      --model-cache "swinl_day65_4352=$CACHE_ROOT/swinl_day65_4352" \
      --model-cache "swinl_night50_4284=$CACHE_ROOT/swinl_night50_4284" \
      --model-cache "segformer_b5_cosec_day_best=$CACHE_ROOT/segformer_b5_cosec_day_best" \
      --model-cache "segformer_b5_cosec_night_best_old=$CACHE_ROOT/segformer_b5_cosec_night_best_old" \
      --out-dir "$OUT_ROOT/${dataset}_${name}" \
      --anchor-model "$anchor" \
      --overwrite \
      "$@"
}

if [[ "$WAIT" == "1" ]]; then
  wait_for_caches
fi

mkdir -p "$OUT_ROOT"

run_one cosec_day_val prior_only swinl_day65_4352 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 0.0 --margin-power 0.0 --anchor-score-ratio 1.0

run_one cosec_day_val prior_conf swinl_day65_4352 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 1.0 --margin-power 0.0 --anchor-score-ratio 1.0

run_one cosec_day_val conservative_anchor swinl_day65_4352 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 1.0 --margin-power 0.5 --anchor-keep-conf 0.75 --anchor-score-ratio 1.05

run_one cosec_day_val specialist_gap swinl_day65_4352 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 0.0 --margin-power 0.0 --anchor-score-bonus 1.05 \
  --candidate-prior-gap 0.03 --anchor-score-ratio 1.0

run_one cosec_day_val specialist_conf_gap swinl_day65_4352 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 0.5 --margin-power 0.0 --anchor-score-bonus 1.10 \
  --candidate-prior-gap 0.05 --anchor-score-ratio 1.05

run_one cosec_night_val prior_only swinl_night50_4284 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 0.0 --margin-power 0.0 --anchor-score-ratio 1.0

run_one cosec_night_val prior_conf swinl_night50_4284 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 1.0 --margin-power 0.0 --anchor-score-ratio 1.0

run_one cosec_night_val conservative_anchor swinl_night50_4284 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 1.0 --margin-power 0.5 --anchor-keep-conf 0.75 --anchor-score-ratio 1.05

run_one cosec_night_val specialist_gap swinl_night50_4284 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 0.0 --margin-power 0.0 --anchor-score-bonus 1.05 \
  --candidate-prior-gap 0.03 --anchor-score-ratio 1.0

run_one cosec_night_val specialist_conf_gap swinl_night50_4284 \
  --iou-weight 0.5 --acc-weight 0.25 --precision-weight 0.25 \
  --conf-power 0.5 --margin-power 0.0 --anchor-score-bonus 1.10 \
  --candidate-prior-gap 0.05 --anchor-score-ratio 1.05

echo "[$(date '+%F %T')] Done. Summaries:"
find "$OUT_ROOT" -maxdepth 2 -name summary.json -print
