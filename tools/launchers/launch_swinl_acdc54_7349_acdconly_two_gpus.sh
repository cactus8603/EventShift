#!/usr/bin/env bash
set -euo pipefail

ROOT="."
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_ACDC_LR1:=0}"
: "${GPU_ACDC_LR5:=1}"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for persistent detached launches." >&2
  exit 1
fi

for path in \
  configs/Mask2Former_SwinL_ACDC54_7349_ACDCOnly_HeadOnly_LR1e-7.yaml \
  configs/Mask2Former_SwinL_ACDC54_7349_ACDCOnly_HeadOnly_LR5e-8.yaml \
  work_dirs/real-ssl-acdc54_70_segmentco_eventedge_continue_lr5e-8_bs3/best_model_acdc_night.pth
do
  if [ ! -e "${path}" ]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"

launch_one() {
  local screen_name="$1"
  local gpu_id="$2"
  local config_file="$3"
  local log_file="$4"

  screen -dmS "${screen_name}" bash -lc "
    cd '${ROOT}' &&
    export PYTHONNOUSERSITE=1 &&
    export PYTHONUNBUFFERED=1 &&
    CUDA_VISIBLE_DEVICES='${gpu_id}' conda run --no-capture-output -n '${CONDA_ENV}' \
      python tools/train_mask2former_cosec.py \
      --config-file '${config_file}' \
      --num-gpus 1 \
      > '${log_file}' 2>&1
  "
  echo "Launched ${screen_name} on GPU ${gpu_id}, log=${log_file}"
}

launch_one \
  swinl_acdc54_7349_acdconly_lr1e7 \
  "${GPU_ACDC_LR1}" \
  configs/Mask2Former_SwinL_ACDC54_7349_ACDCOnly_HeadOnly_LR1e-7.yaml \
  "${log_dir}/acdc54_7349_acdconly_lr1e-7_${stamp}.log"

launch_one \
  swinl_acdc54_7349_acdconly_lr5e8 \
  "${GPU_ACDC_LR5}" \
  configs/Mask2Former_SwinL_ACDC54_7349_ACDCOnly_HeadOnly_LR5e-8.yaml \
  "${log_dir}/acdc54_7349_acdconly_lr5e-8_${stamp}.log"
