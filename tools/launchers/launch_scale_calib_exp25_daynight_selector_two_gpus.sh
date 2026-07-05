#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_EXP25_DAY:=0}"
: "${GPU_EXP25_NIGHT:=1}"
: "${EVENT_EDGE_CACHE_DIR:=work_dirs/diagnostics/cosec_event_edge_cache_p80_r25_50}"

cuda_count="$(
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
)"

if [ "${cuda_count}" -lt 2 ]; then
  echo "Need 2 visible CUDA GPUs, but found ${cuda_count}." >&2
  exit 1
fi

for dataset_name in cosec_train_event cosec_day_val_event cosec_night_val_event; do
  if [ ! -f "${EVENT_EDGE_CACHE_DIR}/${dataset_name}/manifest.json" ]; then
    echo "Missing event-edge cache manifest: ${EVENT_EDGE_CACHE_DIR}/${dataset_name}/manifest.json" >&2
    echo "Run: bash tools/build_scale_calib_event_edge_cache.sh" >&2
    exit 1
  fi
done

log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"

config_file="configs/Mask2Former_SwinL_CoSEC_DayOnly_FromDay65_FreezeBackbone_LR5e-7.yaml"
weights="work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7/best_model_cosec_day.pth"
cache_root="work_dirs/diagnostics/scale_calib_exp25_eval_cache_day65_4352_tta4flip"
event_cache_root="work_dirs/diagnostics/scale_calib_exp25_eval_cache_day65_4352_tta4flip_eventdatasets"

CUDA_VISIBLE_DEVICES="${GPU_EXP25_DAY}" \
  nohup conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/train_scale_accept_reject_calibrator.py \
    --config-file "${config_file}" \
    --weights "${weights}" \
    --train-dataset cosec_train \
    --eval-datasets cosec_day_val \
    --train-limit 384 \
    --eval-limit 175 \
    --pixels-per-image 8192 \
    --epochs 4 \
    --lr 3e-4 \
    --candidate-mode highest_conf \
    --target-classes person,motorcycle,"traffic sign",bicycle,rider,pole,sidewalk,building \
    --target-match either \
    --lowmargin-q 20 \
    --highentropy-q 20 \
    --uncertainty-mode any \
    --require-pred-disagree \
    --use-semantic-boundary-features \
    --require-semantic-boundary \
    --semantic-boundary-source any_scale \
    --semantic-boundary-radius 3 \
    --thresholds 0.15,0.3,0.5,0.7,0.9 \
    --repair-weight 2.0 \
    --damage-weight 2.5 \
    --neutral-weight 0.02 \
    --flip \
    --device cuda:0 \
    --eval-cache-dir "${cache_root}" \
    --out-dir work_dirs/diagnostics/scale_calib_exp25A_day_boundary_uncertainty_selector \
    > "${log_dir}/scale-calib-exp25A-day-selector_${stamp}.log" 2>&1 &
pid_day="$!"

CUDA_VISIBLE_DEVICES="${GPU_EXP25_NIGHT}" \
  nohup conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/train_scale_accept_reject_calibrator.py \
    --config-file "${config_file}" \
    --weights "${weights}" \
    --train-dataset cosec_train_event \
    --eval-datasets cosec_night_val_event \
    --train-limit 384 \
    --eval-limit 130 \
    --pixels-per-image 8192 \
    --epochs 4 \
    --lr 3e-4 \
    --candidate-mode highest_conf \
    --target-classes fence,motorcycle,bicycle,"traffic sign",wall,person,building,pole \
    --target-match either \
    --lowmargin-q -1 \
    --highentropy-q 20 \
    --uncertainty-mode any \
    --require-pred-disagree \
    --use-semantic-boundary-features \
    --require-semantic-boundary \
    --semantic-boundary-source any_scale \
    --semantic-boundary-radius 3 \
    --use-event-edge-features \
    --event-edge-cache-dir "${EVENT_EDGE_CACHE_DIR}" \
    --event-edge-threshold 0 \
    --thresholds 0.15,0.3,0.5,0.7,0.9 \
    --repair-weight 2.5 \
    --damage-weight 2.0 \
    --neutral-weight 0.02 \
    --flip \
    --device cuda:0 \
    --eval-cache-dir "${event_cache_root}" \
    --out-dir work_dirs/diagnostics/scale_calib_exp25B_night_boundary_eventsoft_selector \
    > "${log_dir}/scale-calib-exp25B-night-eventsoft-selector_${stamp}.log" 2>&1 &
pid_night="$!"

echo "Launched Exp25A Day boundary/uncertainty selector on GPU ${GPU_EXP25_DAY}: pid=${pid_day}, log=${log_dir}/scale-calib-exp25A-day-selector_${stamp}.log"
echo "Launched Exp25B Night boundary/event-soft selector on GPU ${GPU_EXP25_NIGHT}: pid=${pid_night}, log=${log_dir}/scale-calib-exp25B-night-eventsoft-selector_${stamp}.log"
