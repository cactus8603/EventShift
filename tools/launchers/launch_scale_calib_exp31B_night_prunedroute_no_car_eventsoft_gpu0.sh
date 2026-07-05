#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

: "${CONDA_ENV:=mask2former}"
: "${GPU_SCALE_CALIB_EXP31B:=0}"
: "${EVENT_EDGE_CACHE_DIR:=work_dirs/diagnostics/cosec_event_edge_cache_p80_r25_50}"
: "${SOURCE_TRAIN_MATRIX:=work_dirs/diagnostics/scale_calib_exp30B_night_classroute_eventsoft_full/train_matrix.npz}"
: "${SOURCE_EVAL_CACHE_DIR:=work_dirs/diagnostics/scale_calib_exp30B_night_classroute_eventsoft_eval_cache}"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for persistent detached launches." >&2
  exit 1
fi

for dataset_name in cosec_train_event cosec_night_val_event; do
  if [ ! -f "${EVENT_EDGE_CACHE_DIR}/${dataset_name}/manifest.json" ]; then
    echo "Missing event-edge cache manifest: ${EVENT_EDGE_CACHE_DIR}/${dataset_name}/manifest.json" >&2
    echo "Run: bash tools/build_scale_calib_event_edge_cache.sh" >&2
    exit 1
  fi
done

if [ ! -f "${SOURCE_TRAIN_MATRIX}" ]; then
  echo "Missing source train matrix: ${SOURCE_TRAIN_MATRIX}" >&2
  echo "Wait for Exp30B to finish writing its matrix before launching Exp31B." >&2
  exit 1
fi

if [ ! -f "${SOURCE_EVAL_CACHE_DIR}/cosec_night_val_event/manifest.json" ]; then
  echo "Missing source eval cache manifest: ${SOURCE_EVAL_CACHE_DIR}/cosec_night_val_event/manifest.json" >&2
  echo "Wait for Exp30B to finish eval-cache collection before launching Exp31B." >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
session="scale_calib_exp31B_night_no_tsign_no_road_no_car_${stamp}"
log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
log_path="${log_dir}/${session}.log"

config_file="configs/Mask2Former_SwinL_CoSEC_DayOnly_FromDay65_FreezeBackbone_LR5e-7.yaml"
weights="work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7/best_model_cosec_day.pth"
out_dir="work_dirs/diagnostics/scale_calib_exp31B_night_no_tsign_no_road_no_car_eventsoft_full"
eval_cache_dir="work_dirs/diagnostics/scale_calib_exp31B_night_no_tsign_no_road_no_car_eventsoft_eval_cache"

screen -dmS "${session}" bash -lc "
  cd '${ROOT}' &&
  export PYTHONNOUSERSITE=1 &&
  export PYTHONUNBUFFERED=1 &&
  CUDA_VISIBLE_DEVICES='${GPU_SCALE_CALIB_EXP31B}' conda run --no-capture-output -n '${CONDA_ENV}' \
    python -u tools/train_scale_accept_reject_calibrator.py \
    --config-file '${config_file}' \
    --weights '${weights}' \
    --train-dataset cosec_train_event \
    --eval-datasets cosec_night_val_event \
    --train-limit 384 \
    --eval-limit 130 \
    --pixels-per-image 8192 \
    --epochs 12 \
    --lr 5e-4 \
    --init-bias -2.0 \
    --candidate-mode class_route \
    --class-route 'sidewalk->highres768,wall->tta3,vegetation->tta3,sky->highres768' \
    --target-classes sidewalk,wall,vegetation,sky \
    --target-match anchor \
    --reuse-train-matrix \
    --train-matrix-path '${SOURCE_TRAIN_MATRIX}' \
    --lowmargin-q -1 \
    --highentropy-q 20 \
    --uncertainty-mode any \
    --require-pred-disagree \
    --require-scale-disagree \
    --use-semantic-boundary-features \
    --semantic-boundary-radius 3 \
    --require-semantic-boundary \
    --use-event-edge-features \
    --event-edge-cache-dir '${EVENT_EDGE_CACHE_DIR}' \
    --repair-weight 2.0 \
    --damage-weight 4.0 \
    --neutral-weight 0.02 \
    --thresholds 0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3,0.5 \
    --flip \
    --device cuda:0 \
    --eval-cache-dir '${SOURCE_EVAL_CACHE_DIR}' \
    --reuse-eval-cache \
    --out-dir '${out_dir}' \
    > '${log_path}' 2>&1
"

echo "Launched ${session} on GPU ${GPU_SCALE_CALIB_EXP31B}"
echo "log=${log_path}"
echo "out=${out_dir}"
