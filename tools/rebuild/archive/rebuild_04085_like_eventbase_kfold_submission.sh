#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
CONDA="${CONDA:-conda}"
TEST_ROOT="${TEST_ROOT:-./data/test}"

TAG="${TAG:-eventbase_kfold_04085like_$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-1}"
RUN_BASE="${RUN_BASE:-1}"
RUN_REQUIRED="${RUN_REQUIRED:-1}"
RUN_COMPOSE="${RUN_COMPOSE:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

BASE_MIN_VOTES="${BASE_MIN_VOTES:-5}"
REQUIRED_MIN_VOTES="${REQUIRED_MIN_VOTES:-3}"

PRED_ROOT="${PRED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs}"
COMPOSED_ROOT="${COMPOSED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/composed}"
ZIP_ROOT="${ZIP_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips}"
DIFF_ROOT="${DIFF_ROOT:-${SWIN_L_ROOT}/work_dirs/diagnostics/submission_diffs}"

EVENT_RAW="${EVENT_RAW:-${PRED_ROOT}/event_replacements_tta4flip_20260628_raw}"
KFOLD_RAW="${KFOLD_RAW:-${PRED_ROOT}/kfold3_swinl_segformer_tta4flip_hardvote_20260627_raw}"
KFOLD_MASKDINO_RAW="${KFOLD_MASKDINO_RAW:-${PRED_ROOT}/kfold3_maskdino_tta4flip_20260628_raw}"
FULLDESC_RAW="${FULLDESC_RAW:-${PRED_ROOT}/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw}"

EVENT_DAY_DIR="${EVENT_DAY_DIR:-${EVENT_RAW}/mask2former_day_event}"
EVENT_NIGHT_DIR="${EVENT_NIGHT_DIR:-${EVENT_RAW}/segformer_night_event}"

# Keep the real branch close to the original 0.4085 real-vote recipe.
REAL_BASE_DIR="${REAL_BASE_DIR:-${COMPOSED_ROOT}/sub_swinL_day65_4352_daytta_night_trainlearnedscale64_acdc54_754_tta_real}"
ACDC_REAL_REQ="${ACDC_REAL_REQ:-${PRED_ROOT}/acdc_all_kfold3_fold0_tta5126247681024_real_only}"

BASE_DAY_DIR="${BASE_DAY_DIR:-${PRED_ROOT}/${TAG}_base_day_kfold_min${BASE_MIN_VOTES}_eventanchor}"
BASE_NIGHT_DIR="${BASE_NIGHT_DIR:-${PRED_ROOT}/${TAG}_base_night_kfold_min${BASE_MIN_VOTES}_eventanchor}"
BASE_COMPOSED_DIR="${BASE_COMPOSED_DIR:-${COMPOSED_ROOT}/sub_${TAG}_base_kfold_min${BASE_MIN_VOTES}_eventanchor}"
BASE_ZIP="${BASE_ZIP:-${ZIP_ROOT}/sub_${TAG}_base_kfold_min${BASE_MIN_VOTES}_eventanchor.zip}"

FINAL_DAY_DIR="${FINAL_DAY_DIR:-${PRED_ROOT}/${TAG}_required_event_day_min${REQUIRED_MIN_VOTES}}"
FINAL_NIGHT_DIR="${FINAL_NIGHT_DIR:-${PRED_ROOT}/${TAG}_required_event_night_min${REQUIRED_MIN_VOTES}}"
FINAL_REAL_DIR="${FINAL_REAL_DIR:-${PRED_ROOT}/${TAG}_required_acdc_real_min${REQUIRED_MIN_VOTES}}"
FINAL_COMPOSED_DIR="${FINAL_COMPOSED_DIR:-${COMPOSED_ROOT}/sub_${TAG}}"
FINAL_ZIP="${FINAL_ZIP:-${ZIP_ROOT}/sub_${TAG}.zip}"

REFERENCE_FINAL_ZIP="${REFERENCE_FINAL_ZIP:-${ZIP_ROOT}/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip}"
COMPARE_JSON="${COMPARE_JSON:-${DIFF_ROOT}/sub_${TAG}_vs_original_04085_reqswin.json}"

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

check_inputs() {
  check_prediction_dir "${EVENT_DAY_DIR}"
  check_prediction_dir "${EVENT_NIGHT_DIR}"
  check_prediction_dir "${REAL_BASE_DIR}"
  check_prediction_dir "${ACDC_REAL_REQ}"
  check_prediction_dir "${FULLDESC_RAW}/segformer_day"
  check_prediction_dir "${FULLDESC_RAW}/segformer_night"
  check_prediction_dir "${FULLDESC_RAW}/segformer_real"
  check_prediction_dir "${FULLDESC_RAW}/mask2former_day"
  check_prediction_dir "${FULLDESC_RAW}/mask2former_night"
  check_prediction_dir "${FULLDESC_RAW}/mask2former_real"
}

run_base() {
  check_inputs

  run_py "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${KFOLD_RAW}/swinl_fold0_day" \
      "${KFOLD_RAW}/swinl_fold1_day" \
      "${KFOLD_RAW}/swinl_fold2_day" \
      "${KFOLD_RAW}/segformer_fold0_day" \
      "${KFOLD_RAW}/segformer_fold1_day" \
      "${KFOLD_RAW}/segformer_fold2_day" \
      "${KFOLD_MASKDINO_RAW}/maskdino_fold0_day" \
      "${KFOLD_MASKDINO_RAW}/maskdino_fold1_day" \
      "${KFOLD_MASKDINO_RAW}/maskdino_fold2_day" \
    --anchor-dir "${EVENT_DAY_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${BASE_DAY_DIR}" \
    --sequences "${DAY_SEQUENCES[@]}" \
    --num-classes 20 \
    --min-votes "${BASE_MIN_VOTES}" \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${KFOLD_RAW}/swinl_fold0_night" \
      "${KFOLD_RAW}/swinl_fold1_night" \
      "${KFOLD_RAW}/swinl_fold2_night" \
      "${KFOLD_RAW}/segformer_fold0_night" \
      "${KFOLD_RAW}/segformer_fold1_night" \
      "${KFOLD_RAW}/segformer_fold2_night" \
      "${KFOLD_MASKDINO_RAW}/maskdino_fold0_night" \
      "${KFOLD_MASKDINO_RAW}/maskdino_fold1_night" \
      "${KFOLD_MASKDINO_RAW}/maskdino_fold2_night" \
    --anchor-dir "${EVENT_NIGHT_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${BASE_NIGHT_DIR}" \
    --sequences "${NIGHT_SEQUENCES[@]}" \
    --num-classes 20 \
    --min-votes "${BASE_MIN_VOTES}" \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${BASE_DAY_DIR}" \
    --night-dir "${BASE_NIGHT_DIR}" \
    --real-dir "${REAL_BASE_DIR}" \
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
      "${EVENT_DAY_DIR}" \
      "${FULLDESC_RAW}/mask2former_day" \
    --required-dir "${EVENT_DAY_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${FINAL_DAY_DIR}" \
    --sequences "${DAY_SEQUENCES[@]}" \
    --min-votes "${REQUIRED_MIN_VOTES}" \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_required_voter_submission.py" \
    --base-dir "${BASE_COMPOSED_DIR}" \
    --voter-dirs \
      "${BASE_COMPOSED_DIR}" \
      "${FULLDESC_RAW}/segformer_night" \
      "${EVENT_NIGHT_DIR}" \
      "${FULLDESC_RAW}/mask2former_night" \
    --required-dir "${EVENT_NIGHT_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${FINAL_NIGHT_DIR}" \
    --sequences "${NIGHT_SEQUENCES[@]}" \
    --min-votes "${REQUIRED_MIN_VOTES}" \
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
    --min-votes "${REQUIRED_MIN_VOTES}" \
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
  if [[ ! -f "${REFERENCE_FINAL_ZIP}" ]]; then
    echo "Skip compare: missing reference zip ${REFERENCE_FINAL_ZIP}" >&2
    return 0
  fi
  mkdir -p "${DIFF_ROOT}"
  run_py "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
    --base "${REFERENCE_FINAL_ZIP}" \
    --candidate "${FINAL_ZIP}" \
    --out "${COMPARE_JSON}"
}

main() {
  cd "${SWIN_L_ROOT}"
  echo "Build original-0.4085-like kfold + required-voter submission with event day/night anchor"
  echo "TAG=${TAG}"
  echo "BASE_MIN_VOTES=${BASE_MIN_VOTES}"
  echo "REQUIRED_MIN_VOTES=${REQUIRED_MIN_VOTES}"
  echo "event day anchor: ${EVENT_DAY_DIR}"
  echo "event night anchor: ${EVENT_NIGHT_DIR}"
  echo "final zip: ${FINAL_ZIP}"

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
