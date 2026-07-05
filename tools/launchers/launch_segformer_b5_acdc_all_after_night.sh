#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
MMSEG_ROOT="/work/u1621738/ebmv_eccv/mmsegmentation"
MAMBASEG_ROOT="/work/u1621738/ebmv_eccv/MambaSeg"
CONDA="/home/u1621738/miniconda3/bin/conda"
ENV_NAME="mmseg"
GPU_ID="${GPU_ID:-1}"
LOG_DIR="${ROOT}/work_dirs/launch_logs"
ACDC_NIGHT_WORK_DIR="${ROOT}/work_dirs/mmseg/segformer_b5_acdc_night_from_old_night_lr1e-5"
ACDC_ALL_CFG="${ROOT}/configs/mmseg/SegFormer_B5_ACDC_All_FromOldNight.py"
ACDC_ALL_WORK_DIR="${ROOT}/work_dirs/mmseg/segformer_b5_acdc_all_from_acdc_night_best_lr1e-5"
NIGHT_PATTERN="SegFormer_B5_ACDC_Night_FromOldNight.py"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${MAMBASEG_ROOT}:${MMSEG_ROOT}:${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

python "${ROOT}/tools/build_mmseg_acdc_splits.py"

echo "Waiting for active ACDC-night SegFormer job to finish..."
while pgrep -f "${NIGHT_PATTERN}" >/dev/null; do
  sleep 60
done

BEST_NIGHT="$(find "${ACDC_NIGHT_WORK_DIR}" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "${BEST_NIGHT}" ]]; then
  echo "No ACDC-night best checkpoint found in ${ACDC_NIGHT_WORK_DIR}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/segformer_acdc_all_after_night_${STAMP}.log"

echo "Launching ACDC all-condition SegFormer from ${BEST_NIGHT}"
echo "Log: ${LOG_FILE}"

cd "${MAMBASEG_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA}" run --no-capture-output -n "${ENV_NAME}" \
  python "${MMSEG_ROOT}/tools/train.py" "${ACDC_ALL_CFG}" \
  --cfg-options \
    "load_from=${BEST_NIGHT}" \
    "work_dir=${ACDC_ALL_WORK_DIR}" \
  2>&1 | tee "${LOG_FILE}"
