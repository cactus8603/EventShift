#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1621738/ebmv_eccv/eccv_segment}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
MASKDINO_ROOT="${MASKDINO_ROOT:-${ROOT}/maskdino_swinl}"
MAMBASEG_ROOT="${MAMBASEG_ROOT:-/work/u1621738/ebmv_eccv/MambaSeg}"
CONDA="${CONDA:-/home/u1621738/miniconda3/bin/conda}"
TEST_ROOT="${TEST_ROOT:-${MAMBASEG_ROOT}/data/test}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc}"

RUN_TAG="${RUN_TAG:-full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628}"
PRED_ROOT="${PRED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_raw}"
DEVICE="${DEVICE:-cuda:1}"
MASKDINO_CFG="${MASKDINO_CFG:-${MASKDINO_ROOT}/configs/cosec/semantic-segmentation/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1.yaml}"

MASKDINO_DAY_WEIGHTS="${MASKDINO_DAY_WEIGHTS:-${CKPT_ROOT}/maskdino/step1/best_model_cosec_day.pth}"
MASKDINO_NIGHT_WEIGHTS="${MASKDINO_NIGHT_WEIGHTS:-${CKPT_ROOT}/maskdino/step1/best_model_cosec_night.pth}"
MASKDINO_REAL_WEIGHTS="${MASKDINO_REAL_WEIGHTS:-${CKPT_ROOT}/maskdino/step1/best_model_acdc_all.pth}"

MASKDINO_TTA_MIN_SIZES=(${MASKDINO_TTA_MIN_SIZES:-512 624 768 1024})
MASKDINO_TTA_MAX_SIZE="${MASKDINO_TTA_MAX_SIZE:-1600}"
MASKDINO_TTA_FLIP="${MASKDINO_TTA_FLIP:-true}"

DAY_SEQUENCES=(
  Day_Campus_012
  Day_Park_011
  Day_Suburbs_015
  Day_Suburbs_017
  Day_Village_009
)
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

maskdino_tta_args=(
  --tta
  --aug-min-sizes "${MASKDINO_TTA_MIN_SIZES[@]}"
  --aug-max-size "${MASKDINO_TTA_MAX_SIZE}"
  --aug-flip "${MASKDINO_TTA_FLIP}"
)

run_maskdino_export() {
  local out_name="$1"
  local weights="$2"
  shift 2

  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${MASKDINO_ROOT}:${MASKDINO_ROOT}/tools:${SWIN_L_ROOT}/tools:${MAMBASEG_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n mask2former python "${MASKDINO_ROOT}/tools/export_maskdino_submission.py" \
    --config-file "${MASKDINO_CFG}" \
    --weights "${weights}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${PRED_ROOT}/${out_name}" \
    --device "${DEVICE}" \
    --sequences "$@" \
    --skip-existing \
    "${maskdino_tta_args[@]}"
}

count_pngs() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -type f -name '*.png' | wc -l
  else
    echo 0
  fi
}

run_maskdino_domain() {
  local out_name="$1"
  local weights="$2"
  local expected_count="$3"
  shift 3

  local final_dir="${PRED_ROOT}/${out_name}"
  local tmp_dir="${PRED_ROOT}/.${out_name}.gpu1_tmp"
  local final_count
  final_count="$(count_pngs "${final_dir}")"

  if [[ "${final_count}" -eq "${expected_count}" ]]; then
    echo "Skip ${out_name}: final output already complete (${final_count}/${expected_count})"
    return 0
  fi
  if [[ "${final_count}" -ne 0 ]]; then
    echo "Refuse to pre-export ${out_name}: partial final output exists (${final_count}/${expected_count})" >&2
    return 1
  fi

  run_maskdino_export ".${out_name}.gpu1_tmp" "${weights}" "$@"

  local tmp_count
  tmp_count="$(count_pngs "${tmp_dir}")"
  if [[ "${tmp_count}" -ne "${expected_count}" ]]; then
    echo "Incomplete tmp output for ${out_name}: ${tmp_count}/${expected_count}" >&2
    return 1
  fi
  if [[ -e "${final_dir}" ]]; then
    echo "Refuse to move ${out_name}: final path appeared while pre-export was running" >&2
    return 1
  fi
  mv "${tmp_dir}" "${final_dir}"
  echo "Promoted ${out_name}: ${tmp_count}/${expected_count}"
}

mkdir -p "${PRED_ROOT}"
echo "RUN_TAG=${RUN_TAG}"
echo "DEVICE=${DEVICE}"
echo "PRED_ROOT=${PRED_ROOT}"
printf 'maskdino day=%s\nmaskdino night=%s\nmaskdino real=%s\n' \
  "$(readlink -f "${MASKDINO_DAY_WEIGHTS}")" \
  "$(readlink -f "${MASKDINO_NIGHT_WEIGHTS}")" \
  "$(readlink -f "${MASKDINO_REAL_WEIGHTS}")"
echo "TTA maskdino min=${MASKDINO_TTA_MIN_SIZES[*]} max=${MASKDINO_TTA_MAX_SIZE} flip=${MASKDINO_TTA_FLIP}"

run_maskdino_domain maskdino_day "${MASKDINO_DAY_WEIGHTS}" 574 "${DAY_SEQUENCES[@]}"
run_maskdino_domain maskdino_night "${MASKDINO_NIGHT_WEIGHTS}" 306 "${NIGHT_SEQUENCES[@]}"
run_maskdino_domain maskdino_real "${MASKDINO_REAL_WEIGHTS}" 107 "${REAL_SEQUENCES[@]}"
