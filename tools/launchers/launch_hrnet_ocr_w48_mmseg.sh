#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
GPU_ID="${GPU_ID:-0}"

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
MAMBASEG_ROOT="/work/u1621738/ebmv_eccv/MambaSeg"
MMSEG_ROOT="/work/u1621738/ebmv_eccv/mmsegmentation"
CONDA="/home/u1621738/miniconda3/bin/conda"
LOG_DIR="${ROOT}/work_dirs/launch_logs"
PID_DIR="${ROOT}/work_dirs/run_pids"
WEIGHT="${ROOT}/work_dirs/pretrained/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth"

case "${MODE}" in
    cosec)
        CFG="${ROOT}/configs/mmseg/HRNet_OCR_W48_CoSEC_Finetune.py"
        SESSION="hrnet_ocr_w48_cosec"
        ;;
    acdc-night)
        CFG="${ROOT}/configs/mmseg/HRNet_OCR_W48_ACDC_Night_Finetune.py"
        SESSION="hrnet_ocr_w48_acdc_night"
        ;;
    acdc-all)
        CFG="${ROOT}/configs/mmseg/HRNet_OCR_W48_ACDC_All_Finetune.py"
        SESSION="hrnet_ocr_w48_acdc_all"
        ;;
    *)
        echo "Usage: GPU_ID=0 $0 {cosec|acdc-night|acdc-all}" >&2
        exit 2
        ;;
esac

"${ROOT}/tools/prepare_hrnet_ocr_w48_weights.sh"

if [[ ! -s "${WEIGHT}" ]]; then
    echo "Missing HRNet-OCR-W48 checkpoint: ${WEIGHT}" >&2
    exit 1
fi

if [[ "${MODE}" == acdc-* ]]; then
    env PYTHONNOUSERSITE=1 PYTHONPATH="${MAMBASEG_ROOT}:${MMSEG_ROOT}:${ROOT}" \
        "${CONDA}" run --no-capture-output -n mmseg \
        python "${ROOT}/tools/build_mmseg_acdc_splits.py"
fi

mkdir -p "${LOG_DIR}" "${PID_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${SESSION}_${STAMP}.log"
PID_FILE="${PID_DIR}/${SESSION}.pid"

setsid bash -c "
echo \$\$ > '${PID_FILE}'
cd '${MAMBASEG_ROOT}'
export PYTHONNOUSERSITE=1
export PYTHONPATH='${MAMBASEG_ROOT}:${MMSEG_ROOT}:${ROOT}'
CUDA_VISIBLE_DEVICES='${GPU_ID}' '${CONDA}' run --no-capture-output -n mmseg python '${MMSEG_ROOT}/tools/train.py' '${CFG}'
" > "${LOG}" 2>&1 < /dev/null &

echo "Started ${SESSION} on GPU ${GPU_ID}"
echo "PID file: ${PID_FILE}"
echo "Config: ${CFG}"
echo "Log: ${LOG}"
