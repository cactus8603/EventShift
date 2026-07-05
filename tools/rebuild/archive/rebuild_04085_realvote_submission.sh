#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
CONDA="${CONDA:-conda}"
TEST_ROOT="${TEST_ROOT:-./data/test}"

TAG="${TAG:-rebuild_04085_realvote_$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-1}"
RUN_BASE="${RUN_BASE:-1}"
RUN_REQUIRED="${RUN_REQUIRED:-1}"
RUN_COMPOSE="${RUN_COMPOSE:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

PRED_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs"
COMPOSED_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/composed"
ZIP_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/submission_zips"
DIFF_ROOT="${SWIN_L_ROOT}/work_dirs/diagnostics/submission_diffs"

KFOLD_RAW="${KFOLD_RAW:-${PRED_ROOT}/kfold3_swinl_segformer_tta4flip_hardvote_20260627_raw}"
FULLDESC_RAW="${FULLDESC_RAW:-${PRED_ROOT}/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw}"
ANCHOR_SUB="${ANCHOR_SUB:-${COMPOSED_ROOT}/sub_swinL_day65_4352_daytta_night_trainlearnedscale64_acdc54_754_tta_real}"
ACDC_REAL_REQ="${ACDC_REAL_REQ:-${PRED_ROOT}/acdc_all_kfold3_fold0_tta5126247681024_real_only}"
HRNET_DAY="${HRNET_DAY:-${PRED_ROOT}/hrnet_ocr_w48_cosec_day7500_s624_s768_flip}"
HRNET_NIGHT="${HRNET_NIGHT:-${PRED_ROOT}/hrnet_ocr_w48_cosec_night8000_s624_s768_flip}"

BASE_DAY_DIR="${BASE_DAY_DIR:-${PRED_ROOT}/${TAG}_base_day_min4_anchor}"
BASE_NIGHT_DIR="${BASE_NIGHT_DIR:-${PRED_ROOT}/${TAG}_base_night_min4_anchor}"
BASE_COMPOSED_DIR="${BASE_COMPOSED_DIR:-${COMPOSED_ROOT}/${TAG}_base_min4_anchorreal}"
BASE_ZIP="${BASE_ZIP:-${ZIP_ROOT}/${TAG}_base_min4_anchorreal.zip}"

FINAL_DAY_DIR="${FINAL_DAY_DIR:-${PRED_ROOT}/${TAG}_reqswin_day_min3}"
FINAL_NIGHT_DIR="${FINAL_NIGHT_DIR:-${PRED_ROOT}/${TAG}_reqswin_night_min3}"
FINAL_REAL_DIR="${FINAL_REAL_DIR:-${PRED_ROOT}/${TAG}_reqswin_real_min3}"
FINAL_COMPOSED_DIR="${FINAL_COMPOSED_DIR:-${COMPOSED_ROOT}/${TAG}}"
FINAL_ZIP="${FINAL_ZIP:-${ZIP_ROOT}/${TAG}.zip}"

REFERENCE_BASE_ZIP="${REFERENCE_BASE_ZIP:-${ZIP_ROOT}/sub_kfold3_swinl_segformer_hrnet_tta_s624_s768_flip_hardvote_20260628_min4_anchorreal.zip}"
REFERENCE_FINAL_ZIP="${REFERENCE_FINAL_ZIP:-${ZIP_ROOT}/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip}"

DAY_SEQUENCES=(Day_Campus_012 Day_Park_011 Day_Suburbs_015 Day_Suburbs_017 Day_Village_009)
NIGHT_SEQUENCES=(Night_Campus_010 Night_City_009 Night_Park_009)
REAL_SEQUENCES=(
  REAL_000005 REAL_000009 REAL_000010 REAL_000011 REAL_000012 REAL_000019
  REAL_000020 REAL_000021 REAL_000023 REAL_000024 REAL_000025 REAL_000026
  REAL_000029 REAL_000030 REAL_000031 REAL_000032 REAL_000039 REAL_000040
)

write_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  write_args=(--overwrite)
fi

run_py() {
  PYTHONNOUSERSITE=1 "${CONDA}" run --no-capture-output -n mmseg python "$@"
}

run_base() {
  run_py "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${KFOLD_RAW}/swinl_fold0_day" \
      "${KFOLD_RAW}/swinl_fold1_day" \
      "${KFOLD_RAW}/swinl_fold2_day" \
      "${KFOLD_RAW}/segformer_fold0_day" \
      "${KFOLD_RAW}/segformer_fold1_day" \
      "${KFOLD_RAW}/segformer_fold2_day" \
      "${HRNET_DAY}" \
    --anchor-dir "${ANCHOR_SUB}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${BASE_DAY_DIR}" \
    --sequences "${DAY_SEQUENCES[@]}" \
    --num-classes 20 \
    --min-votes 4 \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${KFOLD_RAW}/swinl_fold0_night" \
      "${KFOLD_RAW}/swinl_fold1_night" \
      "${KFOLD_RAW}/swinl_fold2_night" \
      "${KFOLD_RAW}/segformer_fold0_night" \
      "${KFOLD_RAW}/segformer_fold1_night" \
      "${KFOLD_RAW}/segformer_fold2_night" \
      "${HRNET_NIGHT}" \
    --anchor-dir "${ANCHOR_SUB}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${BASE_NIGHT_DIR}" \
    --sequences "${NIGHT_SEQUENCES[@]}" \
    --num-classes 20 \
    --min-votes 4 \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${BASE_DAY_DIR}" \
    --night-dir "${BASE_NIGHT_DIR}" \
    --real-dir "${ANCHOR_SUB}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${BASE_COMPOSED_DIR}" \
    --zip "${BASE_ZIP}" \
    "${write_args[@]}"
}

run_required() {
  run_py "${SWIN_L_ROOT}/tools/compose_required_voter_submission.py" \
    --base-dir "${BASE_COMPOSED_DIR}" \
    --voter-dirs \
      "${BASE_COMPOSED_DIR}" \
      "${FULLDESC_RAW}/segformer_day" \
      "${ANCHOR_SUB}" \
      "${FULLDESC_RAW}/mask2former_day" \
    --required-dir "${ANCHOR_SUB}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${FINAL_DAY_DIR}" \
    --sequences "${DAY_SEQUENCES[@]}" \
    --min-votes 3 \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_required_voter_submission.py" \
    --base-dir "${BASE_COMPOSED_DIR}" \
    --voter-dirs \
      "${BASE_COMPOSED_DIR}" \
      "${FULLDESC_RAW}/segformer_night" \
      "${ANCHOR_SUB}" \
      "${FULLDESC_RAW}/mask2former_night" \
    --required-dir "${ANCHOR_SUB}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${FINAL_NIGHT_DIR}" \
    --sequences "${NIGHT_SEQUENCES[@]}" \
    --min-votes 3 \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_required_voter_submission.py" \
    --base-dir "${BASE_COMPOSED_DIR}" \
    --voter-dirs \
      "${BASE_COMPOSED_DIR}" \
      "${FULLDESC_RAW}/segformer_real" \
      "${ACDC_REAL_REQ}" \
      "${FULLDESC_RAW}/mask2former_real" \
    --required-dir "${ACDC_REAL_REQ}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${FINAL_REAL_DIR}" \
    --sequences "${REAL_SEQUENCES[@]}" \
    --min-votes 3 \
    "${write_args[@]}"
}

run_compose() {
  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${FINAL_DAY_DIR}" \
    --night-dir "${FINAL_NIGHT_DIR}" \
    --real-dir "${FINAL_REAL_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${FINAL_COMPOSED_DIR}" \
    --zip "${FINAL_ZIP}" \
    "${write_args[@]}"
}

run_compare() {
  mkdir -p "${DIFF_ROOT}"
  if [[ -f "${REFERENCE_BASE_ZIP}" && -f "${BASE_ZIP}" ]]; then
    run_py "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
      --base "${REFERENCE_BASE_ZIP}" \
      --candidate "${BASE_ZIP}" \
      --out "${DIFF_ROOT}/${TAG}_base_vs_original.json"
  fi
  if [[ -f "${REFERENCE_FINAL_ZIP}" && -f "${FINAL_ZIP}" ]]; then
    run_py "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
      --base "${REFERENCE_FINAL_ZIP}" \
      --candidate "${FINAL_ZIP}" \
      --out "${DIFF_ROOT}/${TAG}_final_vs_original_04085.json"
  fi
}

main() {
  cd "${SWIN_L_ROOT}"
  echo "TAG=${TAG}"
  echo "BASE_ZIP=${BASE_ZIP}"
  echo "FINAL_ZIP=${FINAL_ZIP}"

  if [[ "${RUN_BASE}" == "1" ]]; then
    run_base
  fi
  if [[ "${RUN_REQUIRED}" == "1" ]]; then
    run_required
  fi
  if [[ "${RUN_COMPOSE}" == "1" ]]; then
    run_compose
  fi
  if [[ "${RUN_COMPARE}" == "1" ]]; then
    run_compare
  fi
}

main "$@"
