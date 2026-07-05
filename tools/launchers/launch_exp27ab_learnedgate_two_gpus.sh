#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_EXP27_NIGHT:=1}"
: "${GPU_EXP27_DAY:=0}"

log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"

launch_one() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local log_file="${log_dir}/${name}_${stamp}.log"
  setsid bash -lc "
    cd /work/u1621738/ebmv_eccv/eccv_segment/swin_l
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${gpu}
    conda run --no-capture-output -n ${CONDA_ENV} \
      python tools/train_mask2former_cosec.py \
      --num-gpus 1 \
      --config-file ${config}
  " > "${log_file}" 2>&1 &
  echo "Launched ${name} on GPU ${gpu}: pid=$!, log=${log_file}"
}

launch_one "${GPU_EXP27_NIGHT}" \
  "exp27a_night_event_peft_learnedgate" \
  "configs/Mask2Former_SwinL_CoSEC_Exp27A_NightLearnedGatePEFT.yaml"

launch_one "${GPU_EXP27_DAY}" \
  "exp27b_day_event_peft_learnedgate" \
  "configs/Mask2Former_SwinL_CoSEC_Exp27B_DayLearnedGatePEFT.yaml"
