#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1621738/ebmv_eccv/eccv_segment}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
CONDA="${CONDA:-/home/u1621738/miniconda3/bin/conda}"
TEST_ROOT="${TEST_ROOT:-/work/u1621738/ebmv_eccv/MambaSeg/data/test}"

TAG="${TAG:-sub_explainable_m2f_anchor_seg_valpositive_$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_COMPOSE="${RUN_COMPOSE:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

PRED_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs"
COMPOSED_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/composed"
ZIP_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/submission_zips"
DIFF_ROOT="${SWIN_L_ROOT}/work_dirs/diagnostics/submission_diffs"

FULLDESC_RAW="${FULLDESC_RAW:-${PRED_ROOT}/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw}"
ROUTE_NIGHT="${ROUTE_NIGHT:-${SWIN_L_ROOT}/work_dirs/ensemble_class_routes/full_desc_clean_m2f_base_segformer_night_wall_building_tta_val.json}"
ROUTE_REAL="${ROUTE_REAL:-${SWIN_L_ROOT}/work_dirs/ensemble_class_routes/full_desc_clean_m2f_base_segformer_acdc_fence_tta_val.json}"

DAY_DIR="${DAY_DIR:-${FULLDESC_RAW}/mask2former_day}"
NIGHT_ANCHOR_DIR="${NIGHT_ANCHOR_DIR:-${FULLDESC_RAW}/mask2former_night}"
NIGHT_CANDIDATE_DIR="${NIGHT_CANDIDATE_DIR:-${FULLDESC_RAW}/segformer_night}"
REAL_ANCHOR_DIR="${REAL_ANCHOR_DIR:-${FULLDESC_RAW}/mask2former_real}"
REAL_CANDIDATE_DIR="${REAL_CANDIDATE_DIR:-${FULLDESC_RAW}/segformer_real}"

NIGHT_OUT="${NIGHT_OUT:-${PRED_ROOT}/${TAG}_night_m2f_anchor_seg_wall_building}"
REAL_OUT="${REAL_OUT:-${PRED_ROOT}/${TAG}_real_m2f_anchor_seg_fence}"
COMPOSED_DIR="${COMPOSED_DIR:-${COMPOSED_ROOT}/${TAG}}"
FINAL_ZIP="${FINAL_ZIP:-${ZIP_ROOT}/${TAG}.zip}"

REFERENCE_04085="${REFERENCE_04085:-${ZIP_ROOT}/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip}"
REFERENCE_VALVERIFIED="${REFERENCE_VALVERIFIED:-${ZIP_ROOT}/sub_full_desc_valverified_m2f_base_segformer_wallbuilding_fence_20260628.zip}"

write_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  write_args=(--overwrite)
fi

run_py() {
  PYTHONNOUSERSITE=1 "${CONDA}" run --no-capture-output -n mmseg python "$@"
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
}

check_inputs() {
  require_path "${DAY_DIR}"
  require_path "${NIGHT_ANCHOR_DIR}"
  require_path "${NIGHT_CANDIDATE_DIR}"
  require_path "${REAL_ANCHOR_DIR}"
  require_path "${REAL_CANDIDATE_DIR}"
  require_path "${ROUTE_NIGHT}"
  require_path "${ROUTE_REAL}"
  require_path "${TEST_ROOT}"
}

run_merge() {
  run_py "${SWIN_L_ROOT}/tools/merge_prediction_dirs_by_candidate_class.py" \
    --anchor-dir "${NIGHT_ANCHOR_DIR}" \
    --route-json "${ROUTE_NIGHT}" \
    --candidate "segformer=${NIGHT_CANDIDATE_DIR}" \
    --sequences-prefix Night_ \
    --out-dir "${NIGHT_OUT}" \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/merge_prediction_dirs_by_candidate_class.py" \
    --anchor-dir "${REAL_ANCHOR_DIR}" \
    --route-json "${ROUTE_REAL}" \
    --candidate "segformer=${REAL_CANDIDATE_DIR}" \
    --sequences-prefix REAL_ \
    --out-dir "${REAL_OUT}" \
    "${write_args[@]}"
}

run_compose() {
  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${DAY_DIR}" \
    --night-dir "${NIGHT_OUT}" \
    --real-dir "${REAL_OUT}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${COMPOSED_DIR}" \
    --zip "${FINAL_ZIP}" \
    "${write_args[@]}"
}

run_compare() {
  mkdir -p "${DIFF_ROOT}"
  if [[ -f "${REFERENCE_04085}" ]]; then
    run_py "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
      --base "${REFERENCE_04085}" \
      --candidate "${FINAL_ZIP}" \
      --out "${DIFF_ROOT}/${TAG}_vs_04085.json"
  fi
  if [[ -f "${REFERENCE_VALVERIFIED}" ]]; then
    run_py "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
      --base "${REFERENCE_VALVERIFIED}" \
      --candidate "${FINAL_ZIP}" \
      --out "${DIFF_ROOT}/${TAG}_vs_existing_valverified.json"
  fi
}

main() {
  cd "${SWIN_L_ROOT}"
  check_inputs

  echo "TAG=${TAG}"
  echo "DAY_DIR=${DAY_DIR}"
  echo "NIGHT_ANCHOR_DIR=${NIGHT_ANCHOR_DIR}"
  echo "NIGHT_CANDIDATE_DIR=${NIGHT_CANDIDATE_DIR}"
  echo "REAL_ANCHOR_DIR=${REAL_ANCHOR_DIR}"
  echo "REAL_CANDIDATE_DIR=${REAL_CANDIDATE_DIR}"
  echo "FINAL_ZIP=${FINAL_ZIP}"

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
