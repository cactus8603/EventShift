#!/usr/bin/env bash
set -euo pipefail

ROOT="."
SWIN_L_ROOT="${ROOT}/swin_l"
CONDA="${CONDA:-conda}"
M2F_GPU="${M2F_GPU:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SWIN_L_ROOT}/work_dirs/launch_logs/mask2former_full_cosec_day_floor_savefix_${STAMP}"
OUTPUT_DIR="${OUTPUT_DIR:-${SWIN_L_ROOT}/work_dirs/swinL_full_cosec_from_day_best_floor816070_lr5e-7_savefix_${STAMP}}"
MAX_ITER="${MAX_ITER:-6000}"
EVAL_PERIOD="${EVAL_PERIOD:-500}"

mkdir -p "${LOG_DIR}"

log_file="${LOG_DIR}/mask2former_swinl_full_cosec_day_floor_savefix.log"
(
  cd "${SWIN_L_ROOT}"
  exec setsid nohup env \
    SKIP_CODE_BACKUP=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${M2F_GPU}" \
    PYTHONPATH="${SWIN_L_ROOT}:${SWIN_L_ROOT}/tools:${SWIN_L_ROOT}/third_party/Mask2Former:${SWIN_L_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
    "${CONDA}" run --no-capture-output -n mask2former \
    python "${SWIN_L_ROOT}/tools/train_mask2former_cosec.py" \
    --num-gpus 1 \
    --config-file "${SWIN_L_ROOT}/configs/Mask2Former_SwinL_FullCoSEC_FromDayBest_Floor816070.yaml" \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    SOLVER.MAX_ITER "${MAX_ITER}" \
    TEST.EVAL_PERIOD "${EVAL_PERIOD}"
) >"${log_file}" 2>&1 &
pid="$!"
echo "${pid}" >"${LOG_DIR}/mask2former_swinl_full_cosec_day_floor_savefix.pid"
echo "mask2former_swinl_full_cosec_day_floor_savefix pid=${pid} log=${log_file} output_dir=${OUTPUT_DIR}"
