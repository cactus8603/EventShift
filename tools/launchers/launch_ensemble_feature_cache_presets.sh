#!/usr/bin/env bash
set -euo pipefail

ROOT="."
PROJECT_ROOT="."
MAMBASEG_ROOT="."
MMSEG_ROOT="third_party/mmsegmentation"
CONDA_EXE="${CONDA_EXE:-conda}"
MASK2FORMER_ENV="${MASK2FORMER_ENV:-mask2former}"
MMSEG_ENV="${MMSEG_ENV:-mmseg}"

MODE="${1:-all}"
GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-}"
SAVE_MAPS="${SAVE_MAPS:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/work_dirs/ensemble_feature_cache}"
SWINL_TTA_SCALE_SET="${SWINL_TTA_SCALE_SET:-s512+s624+s768+s1024}"
SWINL_TTA_FLIP="${SWINL_TTA_FLIP:-1}"

SWINL_DAY_CFG="$ROOT/configs/Mask2Former_SwinL_CoSEC_DayOnly_FromDay65_FreezeBackbone_LR5e-7.yaml"
SWINL_DAY_CKPT="$ROOT/work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7/best_model_cosec_day.pth"
SWINL_NIGHT_CFG="$ROOT/configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml"
SWINL_NIGHT_CKPT="$ROOT/work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_night.pth"
SWINL_ACDC_CFG="$ROOT/configs/Mask2Former_SwinL_ACDC54_7349_ACDCOnly_HeadOnly_LR5e-8.yaml"
SWINL_ACDC_CKPT="$ROOT/work_dirs/acdc54_7349_acdconly_headonly_lr5e-8_bs3/best_model_acdc_night.pth"

SEGFORMER_COSEC_CFG="$ROOT/configs/mmseg/SegFormer_B5_CoSEC_Continue_FromOldDay.py"
SEGFORMER_COSEC_DAY_CKPT="$ROOT/work_dirs/mmseg/segformer_b5_cosec_continue_from_old_day_lr1e-5/best_day_mIoU_iter_6000.pth"
SEGFORMER_COSEC_NIGHT_CKPT="$MAMBASEG_ROOT/log/mmseg/segformer_b5_cosec_daynight_finetune/best_night_mIoU_iter_8000.pth"
SEGFORMER_ACDC_NIGHT_CFG="$ROOT/configs/mmseg/SegFormer_B5_ACDC_Night_FromOldNight.py"
SEGFORMER_ACDC_NIGHT_CKPT="$ROOT/work_dirs/mmseg/segformer_b5_acdc_night_from_old_night_lr1e-5/best_mIoU_iter_5500.pth"
SEGFORMER_ACDC_ALL_CFG="$ROOT/configs/mmseg/SegFormer_B5_ACDC_All_FromOldNight.py"
SEGFORMER_ACDC_ALL_CKPT="$ROOT/work_dirs/mmseg/segformer_b5_acdc_all_from_acdc_night_best_lr1e-5/best_mIoU_iter_4000.pth"

limit_arg=()
if [[ -n "$LIMIT" ]]; then
  limit_arg=(--limit "$LIMIT")
fi

map_arg=()
if [[ "$SAVE_MAPS" == "0" ]]; then
  map_arg=(--no-save-maps)
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

run_swinl() {
  local model_name="$1"
  local cfg="$2"
  local ckpt="$3"
  shift 3
  require_file "$cfg"
  require_file "$ckpt"
  echo "[Swin-L] $model_name"
  (
    cd "$ROOT"
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$PROJECT_ROOT:$ROOT:$ROOT/tools:${PYTHONPATH:-}" \
      "$CONDA_EXE" run --no-capture-output -n "$MASK2FORMER_ENV" \
      python "$ROOT/tools/cache_swinl_ensemble_features.py" \
      --config-file "$cfg" \
      --weights "$ckpt" \
      --model-name "$model_name" \
      --datasets "$@" \
      --out-root "$OUT_ROOT" \
      --device "cuda:$GPU_ID" \
      "${limit_arg[@]}" \
      "${map_arg[@]}" \
      --overwrite
  )
}

run_swinl_tta() {
  local model_name="$1"
  local cfg="$2"
  local ckpt="$3"
  shift 3
  require_file "$cfg"
  require_file "$ckpt"
  local flip_arg=()
  if [[ "$SWINL_TTA_FLIP" == "1" ]]; then
    flip_arg=(--flip)
  fi
  echo "[Swin-L TTA] $model_name scale_set=$SWINL_TTA_SCALE_SET flip=$SWINL_TTA_FLIP"
  (
    cd "$ROOT"
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$PROJECT_ROOT:$ROOT:$ROOT/tools:${PYTHONPATH:-}" \
      "$CONDA_EXE" run --no-capture-output -n "$MASK2FORMER_ENV" \
      python "$ROOT/tools/cache_swinl_ensemble_features.py" \
      --config-file "$cfg" \
      --weights "$ckpt" \
      --model-name "$model_name" \
      --datasets "$@" \
      --out-root "$OUT_ROOT" \
      --device "cuda:$GPU_ID" \
      --scale-set "$SWINL_TTA_SCALE_SET" \
      "${flip_arg[@]}" \
      "${limit_arg[@]}" \
      "${map_arg[@]}" \
      --overwrite
  )
}

run_segformer() {
  local model_name="$1"
  local cfg="$2"
  local ckpt="$3"
  shift 3
  require_file "$cfg"
  require_file "$ckpt"
  echo "[SegFormer] $model_name"
  (
    cd "$ROOT"
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$PROJECT_ROOT:$ROOT:$ROOT/tools:$MAMBASEG_ROOT:$MMSEG_ROOT:${PYTHONPATH:-}" \
      "$CONDA_EXE" run --no-capture-output -n "$MMSEG_ENV" \
      python "$ROOT/tools/cache_mmseg_ensemble_features.py" \
      --config-file "$cfg" \
      --checkpoint "$ckpt" \
      --model-name "$model_name" \
      --record-datasets "$@" \
      --out-root "$OUT_ROOT" \
      --device "cuda:$GPU_ID" \
      "${limit_arg[@]}" \
      "${map_arg[@]}" \
      --overwrite
  )
}

run_swinl_presets() {
  run_swinl swinl_day65_4352 "$SWINL_DAY_CFG" "$SWINL_DAY_CKPT" \
    cosec_day_val cosec_night_val acdc_night_val acdc_all_val
  run_swinl swinl_night50_4284 "$SWINL_NIGHT_CFG" "$SWINL_NIGHT_CKPT" \
    cosec_day_val cosec_night_val acdc_night_val acdc_all_val
  run_swinl swinl_acdc54_754 "$SWINL_ACDC_CFG" "$SWINL_ACDC_CKPT" \
    acdc_night_val acdc_all_val cosec_day_val cosec_night_val
}

run_segformer_presets() {
  run_segformer segformer_b5_cosec_day_best "$SEGFORMER_COSEC_CFG" "$SEGFORMER_COSEC_DAY_CKPT" \
    cosec_day_val cosec_night_val
  run_segformer segformer_b5_cosec_night_best_old "$SEGFORMER_COSEC_CFG" "$SEGFORMER_COSEC_NIGHT_CKPT" \
    cosec_day_val cosec_night_val
  run_segformer segformer_b5_acdc_night52 "$SEGFORMER_ACDC_NIGHT_CFG" "$SEGFORMER_ACDC_NIGHT_CKPT" \
    acdc_night_val
  run_segformer segformer_b5_acdc_all69 "$SEGFORMER_ACDC_ALL_CFG" "$SEGFORMER_ACDC_ALL_CKPT" \
    acdc_all_val acdc_night_val
}

case "$MODE" in
  all)
    run_swinl_presets
    run_segformer_presets
    ;;
  swinl)
    run_swinl_presets
    ;;
  swinl_cosec)
    run_swinl swinl_day65_4352 "$SWINL_DAY_CFG" "$SWINL_DAY_CKPT" \
      cosec_day_val cosec_night_val
    run_swinl swinl_night50_4284 "$SWINL_NIGHT_CFG" "$SWINL_NIGHT_CKPT" \
      cosec_day_val cosec_night_val
    ;;
  swinl_day_cosec)
    run_swinl swinl_day65_4352 "$SWINL_DAY_CFG" "$SWINL_DAY_CKPT" \
      cosec_day_val cosec_night_val
    ;;
  swinl_night_cosec)
    run_swinl swinl_night50_4284 "$SWINL_NIGHT_CFG" "$SWINL_NIGHT_CKPT" \
      cosec_day_val cosec_night_val
    ;;
  swinl_cosec_tta)
    run_swinl_tta swinl_day65_4352_tta5126247681024_flip "$SWINL_DAY_CFG" "$SWINL_DAY_CKPT" \
      cosec_day_val cosec_night_val
    run_swinl_tta swinl_night50_4284_tta5126247681024_flip "$SWINL_NIGHT_CFG" "$SWINL_NIGHT_CKPT" \
      cosec_day_val cosec_night_val
    ;;
  segformer)
    run_segformer_presets
    ;;
  swinl_day)
    run_swinl swinl_day65_4352 "$SWINL_DAY_CFG" "$SWINL_DAY_CKPT" \
      cosec_day_val cosec_night_val acdc_night_val acdc_all_val
    ;;
  swinl_night)
    run_swinl swinl_night50_4284 "$SWINL_NIGHT_CFG" "$SWINL_NIGHT_CKPT" \
      cosec_day_val cosec_night_val acdc_night_val acdc_all_val
    ;;
  swinl_acdc)
    run_swinl swinl_acdc54_754 "$SWINL_ACDC_CFG" "$SWINL_ACDC_CKPT" \
      acdc_night_val acdc_all_val cosec_day_val cosec_night_val
    ;;
  segformer_cosec)
    run_segformer segformer_b5_cosec_day_best "$SEGFORMER_COSEC_CFG" "$SEGFORMER_COSEC_DAY_CKPT" \
      cosec_day_val cosec_night_val
    run_segformer segformer_b5_cosec_night_best_old "$SEGFORMER_COSEC_CFG" "$SEGFORMER_COSEC_NIGHT_CKPT" \
      cosec_day_val cosec_night_val
    ;;
  segformer_acdc)
    run_segformer segformer_b5_acdc_night52 "$SEGFORMER_ACDC_NIGHT_CFG" "$SEGFORMER_ACDC_NIGHT_CKPT" \
      acdc_night_val
    run_segformer segformer_b5_acdc_all69 "$SEGFORMER_ACDC_ALL_CFG" "$SEGFORMER_ACDC_ALL_CKPT" \
      acdc_all_val acdc_night_val
    ;;
  *)
    echo "Usage: $0 {all|swinl|swinl_cosec|swinl_day_cosec|swinl_night_cosec|swinl_cosec_tta|segformer|swinl_day|swinl_night|swinl_acdc|segformer_cosec|segformer_acdc}" >&2
    echo "Optional env: GPU_ID=0 LIMIT=10 SAVE_MAPS=0 OUT_ROOT=$OUT_ROOT" >&2
    exit 2
    ;;
esac
