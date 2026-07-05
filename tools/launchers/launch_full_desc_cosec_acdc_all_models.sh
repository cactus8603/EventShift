#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
MAMBASEG_ROOT="${MAMBASEG_ROOT:-.}"
MMSEG_ROOT="${MMSEG_ROOT:-third_party/mmsegmentation}"
CONDA="${CONDA:-conda}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
MODELS="${MODELS:-mask2former,maskdino,segformer,brenet}"

SWIN_L_ROOT="${ROOT}/swin_l"
MASKDINO_ROOT="${ROOT}/maskdino_swinl"
BRENET_ROOT="${ROOT}/BRENet"
UNIFIED_ROOT="${ROOT}/unified_cosec_acdc/classcover_v1"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${UNIFIED_ROOT}/training_logs/full_desc_cosec_acdc_${STAMP}}"
QUEUE_LOG="${LOG_DIR}/queue.log"
PID_FILE="${PID_FILE:-${UNIFIED_ROOT}/training_logs/full_desc_cosec_acdc_queue.pid}"

mkdir -p "${LOG_DIR}" "$(dirname "${PID_FILE}")"
echo "$$" > "${PID_FILE}"

log_queue() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

run_cmd() {
  local name="$1"
  local cwd="$2"
  shift 2
  local log_file="${LOG_DIR}/${name}.log"

  log_queue "START ${name}"
  log_queue "LOG ${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'cd %q\n' "${cwd}" >> "${log_file}"
    printf '%q ' "$@" >> "${log_file}"
    printf '\n' >> "${log_file}"
    log_queue "DRY_RUN ${name}"
    return 0
  fi

  set +e
  (
    cd "${cwd}"
    "$@"
  ) > "${log_file}" 2>&1
  local status=$?
  set -e

  if [[ ${status} -eq 0 ]]; then
    log_queue "DONE ${name}"
  else
    log_queue "FAILED ${name} status=${status}"
  fi
  return "${status}"
}

should_run_model() {
  local model="$1"
  [[ "${MODELS}" == "all" || ",${MODELS}," == *",${model},"* ]]
}

latest_match() {
  local pattern
  local dirname
  local basename
  local newest
  for pattern in "$@"; do
    dirname="$(dirname "${pattern}")"
    basename="$(basename "${pattern}")"
    newest="$(find "${dirname}" -maxdepth 1 -name "${basename}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    if [[ -n "${newest}" ]]; then
      echo "${newest}"
      return 0
    fi
  done
  return 1
}

run_mask2former() {
  local cfg="${SWIN_L_ROOT}/configs/Mask2Former_SwinL_FullDSEC_CoSEC_ACDC_UnifiedClassCover.yaml"
  local cfg_2step="${SWIN_L_ROOT}/configs/Mask2Former_SwinL_FullDSEC_CoSEC_ACDC_UnifiedClassCover_2Step_LR1e-6.yaml"
  local step1_dir="${SWIN_L_ROOT}/work_dirs/swinL_full_dsec_cosec_acdc_unified_bs1"
  local ckpt

  run_cmd mask2former_swinl_step1 "${SWIN_L_ROOT}" \
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      PYTHONPATH="${SWIN_L_ROOT}:${SWIN_L_ROOT}/tools:${SWIN_L_ROOT}/third_party/Mask2Former:${SWIN_L_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
      "${CONDA}" run --no-capture-output -n mask2former \
      python "${SWIN_L_ROOT}/tools/train_mask2former_cosec.py" \
      --num-gpus 1 --config-file "${cfg}"

  if ckpt="$(latest_match \
      "${step1_dir}/best_model_acdc_all.pth" \
      "${step1_dir}/best_model_acdc.pth" \
      "${step1_dir}/best_model_cosec_night.pth" \
      "${step1_dir}/best_model_cosec_day.pth")"; then
    log_queue "MASK2FORMER_2STEP_LOAD ${ckpt}"
    run_cmd mask2former_swinl_step2 "${SWIN_L_ROOT}" \
      env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        PYTHONPATH="${SWIN_L_ROOT}:${SWIN_L_ROOT}/tools:${SWIN_L_ROOT}/third_party/Mask2Former:${SWIN_L_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
        "${CONDA}" run --no-capture-output -n mask2former \
        python "${SWIN_L_ROOT}/tools/train_mask2former_cosec.py" \
        --num-gpus 1 --config-file "${cfg_2step}" MODEL.WEIGHTS "${ckpt}"
  else
    log_queue "SKIP mask2former_swinl_step2 no step1 best checkpoint"
  fi
}

run_maskdino() {
  local cfg="${MASKDINO_ROOT}/configs/cosec/semantic-segmentation/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1.yaml"
  local cfg_2step="${MASKDINO_ROOT}/configs/cosec/semantic-segmentation/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1_2step_lr1e-6.yaml"
  local step1_dir="${MASKDINO_ROOT}/work_dirs/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1_bs1"
  local ckpt

  run_cmd maskdino_swinl_step1 "${MASKDINO_ROOT}" \
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      PYTHONPATH="${MASKDINO_ROOT}:${SWIN_L_ROOT}/tools:${MAMBASEG_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
      "${CONDA}" run --no-capture-output -n mask2former \
      python "${MASKDINO_ROOT}/tools/train_maskdino_cosec.py" \
      --num-gpus 1 --config-file "${cfg}"

  if ckpt="$(latest_match \
      "${step1_dir}/best_model_acdc_all.pth" \
      "${step1_dir}/best_model_acdc.pth" \
      "${step1_dir}/best_model_cosec_night.pth" \
      "${step1_dir}/best_model_cosec_day.pth")"; then
    log_queue "MASKDINO_2STEP_LOAD ${ckpt}"
    run_cmd maskdino_swinl_step2 "${MASKDINO_ROOT}" \
      env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        PYTHONPATH="${MASKDINO_ROOT}:${SWIN_L_ROOT}/tools:${MAMBASEG_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
        "${CONDA}" run --no-capture-output -n mask2former \
        python "${MASKDINO_ROOT}/tools/train_maskdino_cosec.py" \
        --num-gpus 1 --config-file "${cfg_2step}" MODEL.WEIGHTS "${ckpt}"
  else
    log_queue "SKIP maskdino_swinl_step2 no step1 best checkpoint"
  fi
}

run_segformer() {
  local cfg="${SWIN_L_ROOT}/configs/mmseg/SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified.py"
  local cfg_2step="${SWIN_L_ROOT}/configs/mmseg/SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified_2Step_LR1e-6.py"
  local step1_dir="${SWIN_L_ROOT}/work_dirs/mmseg/segformer_b5_full_dsec_cosec_acdc_unified"
  local ckpt

  run_cmd segformer_b5_step1 "${MAMBASEG_ROOT}" \
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      PYTHONPATH="${SWIN_L_ROOT}:${MMSEG_ROOT}:${MAMBASEG_ROOT}:${PYTHONPATH:-}" \
      "${CONDA}" run --no-capture-output -n mmseg \
      python "${MMSEG_ROOT}/tools/train.py" "${cfg}"

  if ckpt="$(latest_match \
      "${step1_dir}/best_acdc_mIoU*.pth" \
      "${step1_dir}/best_night_mIoU*.pth" \
      "${step1_dir}/best_day_mIoU*.pth" \
      "${step1_dir}/latest.pth")"; then
    log_queue "SEGFORMER_2STEP_LOAD ${ckpt}"
    run_cmd segformer_b5_step2 "${MAMBASEG_ROOT}" \
      env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        PYTHONPATH="${SWIN_L_ROOT}:${MMSEG_ROOT}:${MAMBASEG_ROOT}:${PYTHONPATH:-}" \
        "${CONDA}" run --no-capture-output -n mmseg \
        python "${MMSEG_ROOT}/tools/train.py" "${cfg_2step}" --cfg-options "load_from=${ckpt}"
  else
    log_queue "SKIP segformer_b5_step2 no step1 best checkpoint"
  fi
}

run_brenet() {
  local cfg="${BRENET_ROOT}/projects/brenet_cosec/configs/brenet_b2_full_dsec_cosec_acdc_unified.py"
  local cfg_2step="${BRENET_ROOT}/projects/brenet_cosec/configs/brenet_b2_full_dsec_cosec_acdc_unified_acdc_2step.py"
  local step1_dir="${BRENET_ROOT}/work_dirs/brenet_b2_full_dsec_cosec_acdc_unified_bs4"
  local ckpt

  run_cmd brenet_b2_step1 "${BRENET_ROOT}" \
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      PYTHONPATH="${BRENET_ROOT}:${PYTHONPATH:-}" \
      "${CONDA}" run --no-capture-output -n brenet_mask2former \
      python -u "${BRENET_ROOT}/tools/train.py" "${cfg}" --gpu-ids 0 --seed 0

  if ckpt="$(latest_match \
      "${step1_dir}/best_night_mIoU.pth" \
      "${step1_dir}/best_day_mIoU.pth" \
      "${step1_dir}/latest.pth")"; then
    log_queue "BRENET_2STEP_LOAD ${ckpt}"
    run_cmd brenet_b2_acdc_step2 "${BRENET_ROOT}" \
      env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        PYTHONPATH="${BRENET_ROOT}:${PYTHONPATH:-}" \
        "${CONDA}" run --no-capture-output -n brenet_mask2former \
        python -u "${BRENET_ROOT}/tools/train.py" "${cfg_2step}" --gpu-ids 0 --seed 0 --load-from "${ckpt}"
  else
    log_queue "SKIP brenet_b2_acdc_step2 no step1 best checkpoint"
  fi
}

log_queue "QUEUE_START gpu=${GPU_ID} dry_run=${DRY_RUN} models=${MODELS} logs=${LOG_DIR}"
if should_run_model mask2former; then
  run_mask2former
else
  log_queue "SKIP mask2former not selected"
fi
if should_run_model maskdino; then
  run_maskdino
else
  log_queue "SKIP maskdino not selected"
fi
if should_run_model segformer; then
  run_segformer
else
  log_queue "SKIP segformer not selected"
fi
if should_run_model brenet; then
  run_brenet
else
  log_queue "SKIP brenet not selected"
fi
log_queue "QUEUE_DONE"
