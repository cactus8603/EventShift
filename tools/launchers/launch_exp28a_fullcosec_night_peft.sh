#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_EXP28:=1}"

log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_dir}/exp28a_fullcosec_night_peft_${stamp}.log"

setsid bash -lc "
  cd /work/u1621738/ebmv_eccv/eccv_segment/swin_l
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_EXP28}
  conda run --no-capture-output -n ${CONDA_ENV} \
    python tools/train_mask2former_cosec.py \
    --num-gpus 1 \
    --config-file configs/Mask2Former_SwinL_CoSEC_Exp28A_FullCoSECNightPEFTReplay.yaml
" > "${log_file}" 2>&1 &

echo "Launched exp28a_fullcosec_night_peft on GPU ${GPU_EXP28}: pid=$!, log=${log_file}"
