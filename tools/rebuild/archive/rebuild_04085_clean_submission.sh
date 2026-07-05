#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
MAMBASEG_ROOT="${MAMBASEG_ROOT:-.}"
MMSEG_ROOT="${MMSEG_ROOT:-third_party/mmsegmentation}"
CONDA="${CONDA:-conda}"
TEST_ROOT="${TEST_ROOT:-${MAMBASEG_ROOT}/data/test}"
DEVICE="${DEVICE:-cuda:0}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_EXPORTS="${RUN_EXPORTS:-0}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_COMPOSE="${RUN_COMPOSE:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
OVERWRITE="${OVERWRITE:-1}"

BASE_RAW_ROOT="${BASE_RAW_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw}"
if [[ "${RUN_EXPORTS}" == "1" ]]; then
  PRED_ROOT="${PRED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/rebuild_04085_clean_raw_${STAMP}}"
else
  PRED_ROOT="${PRED_ROOT:-${BASE_RAW_ROOT}}"
fi

SUB_NAME="${SUB_NAME:-sub_rebuild_04085_clean_m2f_segformer_delta2p5_${STAMP}}"
NIGHT_DIR="${NIGHT_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${SUB_NAME}_night_delta2p5_perclass}"
REAL_DIR="${REAL_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${SUB_NAME}_real_acdc_perclass}"
COMPOSED_DIR="${COMPOSED_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/composed/${SUB_NAME}}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips/${SUB_NAME}.zip}"
REFERENCE_ZIP="${REFERENCE_ZIP:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips/sub_full_desc_clean_m2f_base_segformer_delta2p5_night_acdc_classes_20260628.zip}"
COMPARE_JSON="${COMPARE_JSON:-${SWIN_L_ROOT}/work_dirs/diagnostics/submission_diffs/${SUB_NAME}_vs_04085_reference.json}"

M2F_CFG="${M2F_CFG:-${SWIN_L_ROOT}/configs/Mask2Former_SwinL_FullDSEC_CoSEC_ACDC_UnifiedClassCover.yaml}"
SEGFORMER_CFG="${SEGFORMER_CFG:-${SWIN_L_ROOT}/configs/mmseg/SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified.py}"

M2F_DAY_WEIGHTS="${M2F_DAY_WEIGHTS:-${SWIN_L_ROOT}/work_dirs/swinL_full_dsec_cosec_acdc_unified_2step_lr1e-6_bs1/best_model_cosec_day.pth}"
M2F_NIGHT_WEIGHTS="${M2F_NIGHT_WEIGHTS:-${SWIN_L_ROOT}/work_dirs/swinL_full_dsec_cosec_acdc_unified_bs1/best_model_cosec_night.pth}"
M2F_REAL_WEIGHTS="${M2F_REAL_WEIGHTS:-${SWIN_L_ROOT}/work_dirs/swinL_full_dsec_cosec_acdc_unified_2step_lr1e-6_bs1/best_model_acdc_all.pth}"
SEGFORMER_NIGHT_WEIGHTS="${SEGFORMER_NIGHT_WEIGHTS:-${SWIN_L_ROOT}/work_dirs/mmseg/segformer_b5_full_dsec_cosec_acdc_unified_2step_lr1e-6/best_night_mIoU_iter_500.pth}"
SEGFORMER_REAL_WEIGHTS="${SEGFORMER_REAL_WEIGHTS:-${SWIN_L_ROOT}/work_dirs/mmseg/segformer_b5_full_dsec_cosec_acdc_unified_2step_lr1e-6/best_acdc_mIoU_iter_7500.pth}"

NIGHT_ROUTE_JSON="${NIGHT_ROUTE_JSON:-${SWIN_L_ROOT}/work_dirs/ensemble_class_routes/full_desc_clean_m2f_base_segformer_night_delta2p5.json}"
REAL_ROUTE_JSON="${REAL_ROUTE_JSON:-${SWIN_L_ROOT}/work_dirs/ensemble_class_routes/full_desc_clean_m2f_base_segformer_real_acdc.json}"

M2F_TTA_MIN_SIZES="${M2F_TTA_MIN_SIZES:-[512,624,768,1024]}"
M2F_TTA_MAX_SIZE="${M2F_TTA_MAX_SIZE:-1600}"
M2F_TTA_FLIP="${M2F_TTA_FLIP:-True}"
MMSEG_TTA_SCALE_SPECS="${MMSEG_TTA_SCALE_SPECS:-s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600}"
MMSEG_TTA_SCALE_SET="${MMSEG_TTA_SCALE_SET:-s512+s624+s768+s1024}"

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

write_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  write_args=(--overwrite)
fi

check_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

check_prediction_dir() {
  local path="$1"
  check_path "${path}"
  if ! find "${path}" -path '*/segment_co/*.png' -print -quit | grep -q .; then
    echo "Prediction dir has no segment_co PNGs: ${path}" >&2
    exit 1
  fi
}

run_m2f_export() {
  local out_name="$1"
  local weights="$2"
  shift 2

  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${SWIN_L_ROOT}:${SWIN_L_ROOT}/tools:${SWIN_L_ROOT}/third_party/Mask2Former:${SWIN_L_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n mask2former python "${SWIN_L_ROOT}/tools/export_mask2former_submission.py" \
    --config-file "${M2F_CFG}" \
    --weights "${weights}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${PRED_ROOT}/${out_name}" \
    --device "${DEVICE}" \
    --sequences "$@" \
    "${write_args[@]}" \
    -- \
    TEST.AUG.ENABLED True \
    TEST.AUG.MIN_SIZES "${M2F_TTA_MIN_SIZES}" \
    TEST.AUG.MAX_SIZE "${M2F_TTA_MAX_SIZE}" \
    TEST.AUG.FLIP "${M2F_TTA_FLIP}"
}

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
    --scale-specs "${MMSEG_TTA_SCALE_SPECS}" \
    --scale-set "${MMSEG_TTA_SCALE_SET}" \
    --flip \
    "${write_args[@]}"
}

run_exports() {
  mkdir -p "${PRED_ROOT}"
  check_path "${M2F_CFG}"
  check_path "${SEGFORMER_CFG}"
  check_path "${M2F_DAY_WEIGHTS}"
  check_path "${M2F_NIGHT_WEIGHTS}"
  check_path "${M2F_REAL_WEIGHTS}"
  check_path "${SEGFORMER_NIGHT_WEIGHTS}"
  check_path "${SEGFORMER_REAL_WEIGHTS}"

  run_m2f_export mask2former_day "${M2F_DAY_WEIGHTS}" "${DAY_SEQUENCES[@]}"
  run_m2f_export mask2former_night "${M2F_NIGHT_WEIGHTS}" "${NIGHT_SEQUENCES[@]}"
  run_m2f_export mask2former_real "${M2F_REAL_WEIGHTS}" "${REAL_SEQUENCES[@]}"
  run_segformer_export segformer_night "${SEGFORMER_NIGHT_WEIGHTS}" "${NIGHT_SEQUENCES[@]}"
  run_segformer_export segformer_real "${SEGFORMER_REAL_WEIGHTS}" "${REAL_SEQUENCES[@]}"
}

run_merge() {
  check_prediction_dir "${PRED_ROOT}/mask2former_day"
  check_prediction_dir "${PRED_ROOT}/mask2former_night"
  check_prediction_dir "${PRED_ROOT}/mask2former_real"
  check_prediction_dir "${PRED_ROOT}/segformer_night"
  check_prediction_dir "${PRED_ROOT}/segformer_real"
  check_path "${NIGHT_ROUTE_JSON}"
  check_path "${REAL_ROUTE_JSON}"

  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/merge_prediction_dirs_by_candidate_class.py" \
    --anchor-dir "${PRED_ROOT}/mask2former_night" \
    --candidate "segformer=${PRED_ROOT}/segformer_night" \
    --route-json "${NIGHT_ROUTE_JSON}" \
    --out-dir "${NIGHT_DIR}" \
    "${write_args[@]}"

  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/merge_prediction_dirs_by_candidate_class.py" \
    --anchor-dir "${PRED_ROOT}/mask2former_real" \
    --candidate "segformer=${PRED_ROOT}/segformer_real" \
    --route-json "${REAL_ROUTE_JSON}" \
    --out-dir "${REAL_DIR}" \
    "${write_args[@]}"
}

run_compose() {
  mkdir -p "$(dirname "${SUBMISSION_ZIP}")"
  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${PRED_ROOT}/mask2former_day" \
    --night-dir "${NIGHT_DIR}" \
    --real-dir "${REAL_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${COMPOSED_DIR}" \
    --zip "${SUBMISSION_ZIP}" \
    "${write_args[@]}"
}

run_compare() {
  if [[ ! -f "${REFERENCE_ZIP}" ]]; then
    echo "Skip compare: missing reference zip ${REFERENCE_ZIP}" >&2
    return 0
  fi
  mkdir -p "$(dirname "${COMPARE_JSON}")"
  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
    --base "${REFERENCE_ZIP}" \
    --candidate "${SUBMISSION_ZIP}" \
    --out "${COMPARE_JSON}"
}

main() {
  cd "${SWIN_L_ROOT}"
  echo "Rebuild 0.4085 clean submission"
  echo "RUN_EXPORTS=${RUN_EXPORTS}"
  echo "PRED_ROOT=${PRED_ROOT}"
  echo "SUBMISSION_ZIP=${SUBMISSION_ZIP}"
  echo "day: pure Mask2Former ${M2F_DAY_WEIGHTS}"
  echo "night: Mask2Former anchor + SegFormer classes from ${NIGHT_ROUTE_JSON}"
  echo "real: Mask2Former anchor + SegFormer classes from ${REAL_ROUTE_JSON}"

  if [[ "${RUN_EXPORTS}" == "1" ]]; then
    run_exports
  fi
  if [[ "${RUN_MERGE}" == "1" ]]; then
    run_merge
  fi
  if [[ "${RUN_COMPOSE}" == "1" ]]; then
    run_compose
  fi
  if [[ "${RUN_COMPARE}" == "1" ]]; then
    run_compare
  fi
}

main "$@"
