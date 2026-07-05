#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

condition="${1:-all}"
fold="${2:-0}"
folds="${FOLDS:-3}"
batch_size="${IMS_PER_BATCH:-3}"
base_lr="${BASE_LR:-0.00000005}"
epochs="${EPOCHS:-3}"
weights="${WEIGHTS:-work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_night.pth}"
output_dir="${OUTPUT_DIR:-work_dirs/acdc_${condition}_kfold${folds}_fold${fold}_headonly_from_night50_lr5e-8_bs${batch_size}}"
device="${DEVICE:-cuda:0}"
export_raw="${EXPORT_RAW:-0}"
export_tta="${EXPORT_TTA:-1}"

export CONDA_ENV="${CONDA_ENV:-mask2former}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export IMS_PER_BATCH="${batch_size}"
export BASE_LR="${base_lr}"
export EPOCHS="${epochs}"
export WEIGHTS="${weights}"
export OUTPUT_DIR="${output_dir}"

bash tools/run_swinl_acdc_kfold_headonly.sh "${condition}" "${fold}"

tag="acdc_${condition}_kfold${folds}_fold${fold}"
best_weights="${output_dir}/best_model_acdc_${condition}_kfold${folds}_fold${fold}_val.pth"
if [ ! -f "${best_weights}" ]; then
  echo "Missing trained checkpoint for export: ${best_weights}" >&2
  exit 1
fi

TAG="${tag}" \
CONFIG_FILE="${output_dir}/config.yaml" \
WEIGHTS="${best_weights}" \
DEVICE="${device}" \
EXPORT_RAW="${export_raw}" \
EXPORT_TTA="${export_tta}" \
ENV_NAME="${CONDA_ENV}" \
  bash tools/export_acdc_kfold_real_candidate.sh
