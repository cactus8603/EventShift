#!/usr/bin/env bash
set -euo pipefail

ROOT="."
cd "$ROOT"

DEVICE="${DEVICE:-cuda:0}"
TEST_ROOT="${TEST_ROOT:-./data/test}"
ENV_NAME="${ENV_NAME:-mask2former}"

TAG="acdc54_754_acdconly"
CONFIG_FILE="work_dirs/acdc54_7349_acdconly_headonly_lr5e-8_bs3/config.yaml"
WEIGHTS="work_dirs/acdc54_7349_acdconly_headonly_lr5e-8_bs3/best_model_acdc_night.pth"

DAY_DIR="work_dirs/submissions/prediction_dirs/swinL_day65_4352_tta5126247681024_day_raw"
NIGHT_DIR="work_dirs/submissions/prediction_dirs/swinL_day65_4352_tta5126247681024_night_raw"
ZIP_DIR="work_dirs/submissions/submission_zips"
COMPOSE_DIR="work_dirs/submissions/composed"
PRED_DIR="work_dirs/submissions/prediction_dirs"
DIAG_DIR="work_dirs/diagnostics/submission_diffs"

REAL_SEQUENCES=(
  REAL_000005
  REAL_000009
  REAL_000010
  REAL_000011
  REAL_000012
  REAL_000019
  REAL_000020
  REAL_000021
  REAL_000023
  REAL_000024
  REAL_000025
  REAL_000026
  REAL_000029
  REAL_000030
  REAL_000031
  REAL_000032
  REAL_000039
  REAL_000040
)

TTA_OPTS=(
  --
  TEST.AUG.ENABLED True
  TEST.AUG.MIN_SIZES "[512,624,768,1024]"
  TEST.AUG.MAX_SIZE 1600
  TEST.AUG.FLIP True
  INPUT.MIN_SIZE_TEST 624
  INPUT.MAX_SIZE_TEST 1200
)

check_inputs() {
  for path in "$CONFIG_FILE" "$WEIGHTS" "$DAY_DIR" "$NIGHT_DIR"; do
    if [ ! -e "$path" ]; then
      echo "Missing required input: $path" >&2
      exit 1
    fi
  done
}

check_cuda() {
  conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this shell; run this script on a GPU-visible shell.")
print(f"CUDA visible devices: {torch.cuda.device_count()}")
PY
}

export_real() {
  local suffix="$1"
  shift
  local real_out="${PRED_DIR}/${TAG}_${suffix}_real_only"

  echo "[export] ${TAG} ${suffix} -> ${real_out}"
  conda run --no-capture-output -n "$ENV_NAME" python tools/export_mask2former_submission.py \
    --config-file "$CONFIG_FILE" \
    --weights "$WEIGHTS" \
    --test-root "$TEST_ROOT" \
    --out-dir "$real_out" \
    --device "$DEVICE" \
    --sequences "${REAL_SEQUENCES[@]}" \
    --overwrite \
    "$@"
}

compose_real_only_zip() {
  local suffix="$1"
  local real_dir="${PRED_DIR}/${TAG}_${suffix}_real_only"
  local out_dir="${COMPOSE_DIR}/sub_swinL_day65_4352_tta5126247681024_daynight_${TAG}_${suffix}_real"
  local zip_path="${ZIP_DIR}/sub_swinL_day65_4352_tta5126247681024_daynight_${TAG}_${suffix}_real.zip"
  local diff_path="${DIAG_DIR}/current_best_vs_${TAG}_${suffix}_real.json"

  echo "[compose] ${zip_path}"
  conda run --no-capture-output -n "$ENV_NAME" python tools/compose_domain_submission.py \
    --day-dir "$DAY_DIR" \
    --night-dir "$NIGHT_DIR" \
    --real-dir "$real_dir" \
    --test-root "$TEST_ROOT" \
    --out-dir "$out_dir" \
    --zip "$zip_path" \
    --overwrite

  mkdir -p "$DIAG_DIR"
  conda run --no-capture-output -n "$ENV_NAME" python tools/compare_submission_zips.py \
    --base "${ZIP_DIR}/current_best.zip" \
    --candidate "$zip_path" \
    --out "$diff_path"
}

check_inputs
check_cuda

export_real "raw"
compose_real_only_zip "raw"

export_real "tta5126247681024" "${TTA_OPTS[@]}"
compose_real_only_zip "tta5126247681024"

echo "[done] Generated ${TAG} REAL-only raw and TTA candidates."
