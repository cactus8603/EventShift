#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1621738/ebmv_eccv/eccv_segment}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
MAMBASEG_ROOT="${MAMBASEG_ROOT:-/work/u1621738/ebmv_eccv/MambaSeg}"
MMSEG_ROOT="${MMSEG_ROOT:-/work/u1621738/ebmv_eccv/mmsegmentation}"
CONDA="${CONDA:-/home/u1621738/miniconda3/bin/conda}"
TEST_ROOT="${TEST_ROOT:-${MAMBASEG_ROOT}/data/test}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc}"

RUN_TAG="${RUN_TAG:-full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628}"
PRED_ROOT="${PRED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_raw}"
DEVICE="${DEVICE:-cuda:1}"
SEGFORMER_CFG="${SEGFORMER_CFG:-${SWIN_L_ROOT}/configs/mmseg/SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified.py}"

SEGFORMER_NIGHT_WEIGHTS="${SEGFORMER_NIGHT_WEIGHTS:-${CKPT_ROOT}/segformer/step2/best_night_mIoU.pth}"
SEGFORMER_REAL_WEIGHTS="${SEGFORMER_REAL_WEIGHTS:-${CKPT_ROOT}/segformer/step2/best_acdc_mIoU.pth}"

MMSEG_TTA_SCALE_SPECS="${MMSEG_TTA_SCALE_SPECS:-s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600}"
MMSEG_TTA_SCALE_SET="${MMSEG_TTA_SCALE_SET:-s512+s624+s768+s1024}"
MMSEG_TTA_FLIP="${MMSEG_TTA_FLIP:-1}"

NIGHT_SEQUENCES=(
  Night_Campus_010
  Night_City_009
  Night_Park_009
)
REAL_SEQUENCES=(
  REAL_000005
  REAL_000009
  REAL_000010
  REAL_000011
  REAL_000012
  REAL_000019
  REAL_000020
  REAL_000021
  REAL_000023
  REAL_000024
  REAL_000025
  REAL_000026
  REAL_000029
  REAL_000030
  REAL_000031
  REAL_000032
  REAL_000039
  REAL_000040
)

mmseg_tta_args=(
  --scale-specs "${MMSEG_TTA_SCALE_SPECS}"
  --scale-set "${MMSEG_TTA_SCALE_SET}"
)
if [[ "${MMSEG_TTA_FLIP}" == "1" ]]; then
  mmseg_tta_args+=(--flip)
fi

run_segformer_export() {
  local out_name="$1"
  local weights="$2"
  shift 2

  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${SWIN_L_ROOT}:${SWIN_L_ROOT}/tools:${MMSEG_ROOT}:${MAMBASEG_ROOT}:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/export_mmseg_submission.py" \
    --config-file "${SEGFORMER_CFG}" \
    --checkpoint "${weights}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${PRED_ROOT}/${out_name}" \
    --device "${DEVICE}" \
    --sequences "$@" \
    --skip-existing \
    "${mmseg_tta_args[@]}"
}

mkdir -p "${PRED_ROOT}"
echo "RUN_TAG=${RUN_TAG}"
echo "DEVICE=${DEVICE}"
echo "PRED_ROOT=${PRED_ROOT}"
printf 'segformer night=%s\nsegformer real=%s\n' \
  "$(readlink -f "${SEGFORMER_NIGHT_WEIGHTS}")" \
  "$(readlink -f "${SEGFORMER_REAL_WEIGHTS}")"
echo "TTA segformer scales=${MMSEG_TTA_SCALE_SET} flip=${MMSEG_TTA_FLIP}"

run_segformer_export segformer_night "${SEGFORMER_NIGHT_WEIGHTS}" "${NIGHT_SEQUENCES[@]}"
run_segformer_export segformer_real "${SEGFORMER_REAL_WEIGHTS}" "${REAL_SEQUENCES[@]}"
