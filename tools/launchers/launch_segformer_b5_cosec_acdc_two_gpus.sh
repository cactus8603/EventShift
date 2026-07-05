#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
MMSEG_ROOT="/work/u1621738/ebmv_eccv/mmsegmentation"
MAMBASEG_ROOT="/work/u1621738/ebmv_eccv/MambaSeg"
CONDA="/home/u1621738/miniconda3/bin/conda"
ENV_NAME="mmseg"
LOG_DIR="${ROOT}/work_dirs/launch_logs"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${MAMBASEG_ROOT}:${MMSEG_ROOT}:${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

python "${ROOT}/tools/build_mmseg_acdc_splits.py"

COSEC_CFG="${ROOT}/configs/mmseg/SegFormer_B5_CoSEC_Continue_FromOldDay.py"
ACDC_CFG="${ROOT}/configs/mmseg/SegFormer_B5_ACDC_Night_FromOldNight.py"
STAMP="$(date +%Y%m%d_%H%M%S)"

screen -dmS segformer_cosec_continue bash -lc "
  cd '${MAMBASEG_ROOT}' &&
  export PYTHONPATH='${PYTHONPATH}' &&
  export PYTHONNOUSERSITE=1 &&
  CUDA_VISIBLE_DEVICES=0 '${CONDA}' run --no-capture-output -n '${ENV_NAME}' \
    python '${MMSEG_ROOT}/tools/train.py' '${COSEC_CFG}' \
    2>&1 | tee '${LOG_DIR}/segformer_cosec_continue_${STAMP}.log'
"

screen -dmS segformer_acdc_night bash -lc "
  cd '${MAMBASEG_ROOT}' &&
  export PYTHONPATH='${PYTHONPATH}' &&
  export PYTHONNOUSERSITE=1 &&
  CUDA_VISIBLE_DEVICES=1 '${CONDA}' run --no-capture-output -n '${ENV_NAME}' \
    python '${MMSEG_ROOT}/tools/train.py' '${ACDC_CFG}' \
    2>&1 | tee '${LOG_DIR}/segformer_acdc_night_${STAMP}.log'
"

echo "Launched:"
echo "  segformer_cosec_continue -> ${LOG_DIR}/segformer_cosec_continue_${STAMP}.log"
echo "  segformer_acdc_night     -> ${LOG_DIR}/segformer_acdc_night_${STAMP}.log"
