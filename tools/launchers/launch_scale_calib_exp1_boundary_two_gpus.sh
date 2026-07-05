#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_SCALE_CALIB_DAY:=0}"
: "${GPU_SCALE_CALIB_NIGHT:=1}"

cuda_count="$(
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
)"

if [ "${cuda_count}" -lt 2 ]; then
  echo "Need 2 visible CUDA GPUs, but found ${cuda_count}." >&2
  exit 1
fi

log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"

config_file="configs/Mask2Former_SwinL_CoSEC_DayOnly_FromDay65_FreezeBackbone_LR5e-7.yaml"
weights="work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7/best_model_cosec_day.pth"
cache_root="work_dirs/diagnostics/scale_calib_exp1_eval_cache_day65_4352_tta4flip"

CUDA_VISIBLE_DEVICES="${GPU_SCALE_CALIB_DAY}" \
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
    --thresholds 0.3,0.5,0.7,0.9 \
    --flip \
    --device cuda:0 \
    --eval-cache-dir "${cache_root}" \
    --reuse-eval-cache \
    --out-dir work_dirs/diagnostics/scale_calib_exp1C_day_boundary_accept_highconf \
    > "${log_dir}/scale-calib-exp1C-day-boundary_${stamp}.log" 2>&1 &
pid_day="$!"

CUDA_VISIBLE_DEVICES="${GPU_SCALE_CALIB_NIGHT}" \
  nohup conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/train_scale_accept_reject_calibrator.py \
    --config-file "${config_file}" \
    --weights "${weights}" \
    --train-dataset cosec_train \
    --eval-datasets cosec_night_val \
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
    --thresholds 0.3,0.5,0.7,0.9 \
    --flip \
    --device cuda:0 \
    --eval-cache-dir "${cache_root}" \
    --reuse-eval-cache \
    --out-dir work_dirs/diagnostics/scale_calib_exp1D_night_boundary_accept_highconf \
    > "${log_dir}/scale-calib-exp1D-night-boundary_${stamp}.log" 2>&1 &
pid_night="$!"

echo "Launched scale-calib-exp1C Day boundary on GPU ${GPU_SCALE_CALIB_DAY}: pid=${pid_day}, log=${log_dir}/scale-calib-exp1C-day-boundary_${stamp}.log"
echo "Launched scale-calib-exp1D Night boundary on GPU ${GPU_SCALE_CALIB_NIGHT}: pid=${pid_night}, log=${log_dir}/scale-calib-exp1D-night-boundary_${stamp}.log"
