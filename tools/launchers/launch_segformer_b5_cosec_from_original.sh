#!/usr/bin/env bash
set -euo pipefail

ROOT="."
MMSEG_ROOT="third_party/mmsegmentation"
MAMBASEG_ROOT="."
CONDA="conda"
ENV_NAME="mmseg"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${ROOT}/work_dirs/launch_logs"
PID_DIR="${ROOT}/work_dirs/run_pids"
mkdir -p "${LOG_DIR}"
mkdir -p "${PID_DIR}"

export PYTHONPATH="${MAMBASEG_ROOT}:${MMSEG_ROOT}:${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

CFG="${ROOT}/configs/mmseg/SegFormer_B5_CoSEC_FromOriginalPretrain.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/segformer_cosec_from_original_${STAMP}.log"
PID_FILE="${PID_DIR}/segformer_cosec_from_original.pid"

setsid bash -c "
  echo \$\$ > '${PID_FILE}' &&
  cd '${MAMBASEG_ROOT}' &&
  export PYTHONPATH='${PYTHONPATH}' &&
  export PYTHONNOUSERSITE=1 &&
  CUDA_VISIBLE_DEVICES='${GPU_ID}' '${CONDA}' run --no-capture-output -n '${ENV_NAME}' \
    python '${MMSEG_ROOT}/tools/train.py' '${CFG}'
" > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Launched segformer_cosec_from_original on GPU ${GPU_ID}"
echo "Config: ${CFG}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "Work dir: ${ROOT}/work_dirs/mmseg/segformer_b5_cosec_from_original_pretrain_lr2e-5"
