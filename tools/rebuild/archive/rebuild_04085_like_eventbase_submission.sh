#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
CONDA="${CONDA:-conda}"
TEST_ROOT="${TEST_ROOT:-./data/test}"

TAG="${TAG:-eventbase_04085like_$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_COMPOSE="${RUN_COMPOSE:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

PRED_ROOT="${PRED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs}"
COMPOSED_ROOT="${COMPOSED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/composed}"
ZIP_ROOT="${ZIP_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips}"
DIFF_ROOT="${DIFF_ROOT:-${SWIN_L_ROOT}/work_dirs/diagnostics/submission_diffs}"

EVENT_RAW="${EVENT_RAW:-${PRED_ROOT}/event_replacements_tta4flip_20260628_raw}"
FULLDESC_RAW="${FULLDESC_RAW:-${PRED_ROOT}/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw}"

DAY_DIR="${DAY_DIR:-${EVENT_RAW}/mask2former_day_event}"
NIGHT_BASE_DIR="${NIGHT_BASE_DIR:-${EVENT_RAW}/segformer_night_event}"
NIGHT_CANDIDATE_DIR="${NIGHT_CANDIDATE_DIR:-${FULLDESC_RAW}/mask2former_night}"
REAL_ANCHOR_DIR="${REAL_ANCHOR_DIR:-${FULLDESC_RAW}/mask2former_real}"
REAL_CANDIDATE_DIR="${REAL_CANDIDATE_DIR:-${FULLDESC_RAW}/segformer_real}"

NIGHT_ROUTE_JSON="${NIGHT_ROUTE_JSON:-${SWIN_L_ROOT}/work_dirs/ensemble_class_routes/event_segformer_night_base_m2f_delta2p5.json}"
REAL_ROUTE_JSON="${REAL_ROUTE_JSON:-${SWIN_L_ROOT}/work_dirs/ensemble_class_routes/full_desc_clean_m2f_base_segformer_real_acdc.json}"

NIGHT_DIR="${NIGHT_DIR:-${PRED_ROOT}/${TAG}_night_eventseg_base_m2f_delta2p5}"
REAL_DIR="${REAL_DIR:-${PRED_ROOT}/${TAG}_real_acdc_perclass}"
COMPOSED_DIR="${COMPOSED_DIR:-${COMPOSED_ROOT}/sub_${TAG}}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-${ZIP_ROOT}/sub_${TAG}.zip}"

REFERENCE_ZIP="${REFERENCE_ZIP:-${ZIP_ROOT}/sub_full_desc_clean_m2f_base_segformer_delta2p5_night_acdc_classes_20260628.zip}"
COMPARE_JSON="${COMPARE_JSON:-${DIFF_ROOT}/sub_${TAG}_vs_04085_clean_reference.json}"

write_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  write_args=(--overwrite)
fi

check_prediction_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing prediction dir: ${path}" >&2
    exit 1
  fi
  if ! find "${path}" -path '*/segment_co/*.png' -print -quit | grep -q .; then
    echo "Prediction dir has no segment_co PNGs: ${path}" >&2
    exit 1
  fi
}

run_py() {
  PYTHONNOUSERSITE=1 "${CONDA}" run --no-capture-output -n mmseg python "$@"
}

run_merge() {
  check_prediction_dir "${DAY_DIR}"
  check_prediction_dir "${NIGHT_BASE_DIR}"
  check_prediction_dir "${NIGHT_CANDIDATE_DIR}"
  check_prediction_dir "${REAL_ANCHOR_DIR}"
  check_prediction_dir "${REAL_CANDIDATE_DIR}"

  run_py "${SWIN_L_ROOT}/tools/merge_prediction_dirs_by_candidate_class.py" \
    --anchor-dir "${NIGHT_BASE_DIR}" \
    --candidate "mask2former=${NIGHT_CANDIDATE_DIR}" \
    --route-json "${NIGHT_ROUTE_JSON}" \
    --out-dir "${NIGHT_DIR}" \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/merge_prediction_dirs_by_candidate_class.py" \
    --anchor-dir "${REAL_ANCHOR_DIR}" \
    --candidate "segformer=${REAL_CANDIDATE_DIR}" \
    --route-json "${REAL_ROUTE_JSON}" \
    --out-dir "${REAL_DIR}" \
    "${write_args[@]}"
}

run_compose() {
  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${DAY_DIR}" \
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
  mkdir -p "${DIFF_ROOT}"
  run_py "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
    --base "${REFERENCE_ZIP}" \
    --candidate "${SUBMISSION_ZIP}" \
    --out "${COMPARE_JSON}"
}

main() {
  cd "${SWIN_L_ROOT}"
  echo "Build 0.4085-like event-base submission"
  echo "TAG=${TAG}"
  echo "day base: ${DAY_DIR}"
  echo "night base: ${NIGHT_BASE_DIR}"
  echo "night correction: ${NIGHT_CANDIDATE_DIR} via ${NIGHT_ROUTE_JSON}"
  echo "real correction: ${REAL_ANCHOR_DIR} + ${REAL_CANDIDATE_DIR} via ${REAL_ROUTE_JSON}"
  echo "zip: ${SUBMISSION_ZIP}"

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
