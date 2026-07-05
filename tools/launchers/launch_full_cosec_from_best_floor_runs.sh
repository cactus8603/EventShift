#!/usr/bin/env bash
set -euo pipefail

ROOT="."
SWIN_L_ROOT="${ROOT}/swin_l"
MMSEG_ROOT="third_party/mmsegmentation"
MAMBASEG_ROOT="."
CONDA="${CONDA:-conda}"

M2F_GPU="${M2F_GPU:-0}"
SEGFORMER_GPU="${SEGFORMER_GPU:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SWIN_L_ROOT}/work_dirs/launch_logs/full_cosec_from_best_floor_${STAMP}"

mkdir -p "${LOG_DIR}"

run_bg() {
  local name="$1"
  local workdir="$2"
  shift 2
  local log_file="${LOG_DIR}/${name}.log"
  (
    cd "${workdir}"
    exec setsid nohup "$@"
  ) >"${log_file}" 2>&1 &
  local pid="$!"
  echo "${pid}" >"${LOG_DIR}/${name}.pid"
  echo "${name} pid=${pid} log=${log_file}"
}

run_bg mask2former_swinl_full_cosec_day_floor "${SWIN_L_ROOT}" \
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${M2F_GPU}" \
    PYTHONPATH="${SWIN_L_ROOT}:${SWIN_L_ROOT}/tools:${SWIN_L_ROOT}/third_party/Mask2Former:${SWIN_L_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
    "${CONDA}" run --no-capture-output -n mask2former \
    python "${SWIN_L_ROOT}/tools/train_mask2former_cosec.py" \
    --num-gpus 1 \
    --config-file "${SWIN_L_ROOT}/configs/Mask2Former_SwinL_FullCoSEC_FromDayBest_Floor816070.yaml"

run_bg segformer_b5_full_cosec_night_floor "${MAMBASEG_ROOT}" \
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${SEGFORMER_GPU}" \
    PYTHONPATH="${SWIN_L_ROOT}:${MMSEG_ROOT}:${MAMBASEG_ROOT}:${PYTHONPATH:-}" \
    "${CONDA}" run --no-capture-output -n mmseg \
    python "${MMSEG_ROOT}/tools/train.py" \
    "${SWIN_L_ROOT}/configs/mmseg/SegFormer_B5_FullCoSEC_FromNightBest_Floor546453.py"

echo "log_dir=${LOG_DIR}"
