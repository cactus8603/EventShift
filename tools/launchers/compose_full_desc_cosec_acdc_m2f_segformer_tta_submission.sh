#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
MAMBASEG_ROOT="${MAMBASEG_ROOT:-.}"
CONDA="${CONDA:-conda}"
TEST_ROOT="${TEST_ROOT:-${MAMBASEG_ROOT}/data/test}"

RUN_TAG="${RUN_TAG:-full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628}"
PRED_ROOT="${PRED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_raw}"
OUT_TAG="${OUT_TAG:-full_desc_cosec_acdc_m2f_segformer_tta_domainweights_20260628}"
DAY_VOTE_DIR="${DAY_VOTE_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${OUT_TAG}_day_min2_m2f_anchor}"
NIGHT_VOTE_DIR="${NIGHT_VOTE_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${OUT_TAG}_night_min2_m2f_anchor}"
REAL_VOTE_DIR="${REAL_VOTE_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${OUT_TAG}_real_min2_m2f_anchor}"
SUB_NAME="${SUB_NAME:-sub_${OUT_TAG}_min2_m2fanchor}"
COMPOSED_DIR="${COMPOSED_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/composed/${SUB_NAME}}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips/${SUB_NAME}.zip}"
ANCHOR_ZIP="${ANCHOR_ZIP:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips/sub_swinL_day65_4352_daytta_night_trainlearnedscale64_acdc54_754_tta_real.zip}"
COMPARE_JSON="${COMPARE_JSON:-${SWIN_L_ROOT}/work_dirs/diagnostics/submission_diffs/${SUB_NAME}_vs_anchor4050.json}"

MIN_VOTES="${MIN_VOTES:-2}"
OVERWRITE_VOTE="${OVERWRITE_VOTE:-1}"
OVERWRITE_COMPOSE="${OVERWRITE_COMPOSE:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

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

vote_write_args=()
if [[ "${OVERWRITE_VOTE}" == "1" ]]; then
  vote_write_args=(--overwrite)
fi

compose_write_args=()
if [[ "${OVERWRITE_COMPOSE}" == "1" ]]; then
  compose_write_args=(--overwrite)
fi

check_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo "Missing prediction dir: ${dir}" >&2
    exit 1
  fi
}

run_vote() {
  local bucket="$1"
  local out_dir="$2"
  shift 2
  local sequences=("$@")

  check_dir "${PRED_ROOT}/mask2former_${bucket}"
  check_dir "${PRED_ROOT}/segformer_${bucket}"

  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${PRED_ROOT}/mask2former_${bucket}" \
      "${PRED_ROOT}/segformer_${bucket}" \
    --anchor-dir "${PRED_ROOT}/mask2former_${bucket}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${out_dir}" \
    --sequences "${sequences[@]}" \
    --min-votes "${MIN_VOTES}" \
    "${vote_write_args[@]}"
}

mkdir -p "$(dirname "${SUBMISSION_ZIP}")" "$(dirname "${COMPARE_JSON}")"
echo "PRED_ROOT=${PRED_ROOT}"
echo "SUBMISSION_ZIP=${SUBMISSION_ZIP}"
echo "MIN_VOTES=${MIN_VOTES}"
echo "day candidates: mask2former_day + segformer_day"
echo "night candidates: mask2former_night + segformer_night"
echo "real candidates: mask2former_real + segformer_real"

run_vote day "${DAY_VOTE_DIR}" "${DAY_SEQUENCES[@]}"
run_vote night "${NIGHT_VOTE_DIR}" "${NIGHT_SEQUENCES[@]}"
run_vote real "${REAL_VOTE_DIR}" "${REAL_SEQUENCES[@]}"

PYTHONNOUSERSITE=1 \
"${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
  --day-dir "${DAY_VOTE_DIR}" \
  --night-dir "${NIGHT_VOTE_DIR}" \
  --real-dir "${REAL_VOTE_DIR}" \
  --test-root "${TEST_ROOT}" \
  --out-dir "${COMPOSED_DIR}" \
  --zip "${SUBMISSION_ZIP}" \
  "${compose_write_args[@]}"

if [[ "${RUN_COMPARE}" == "1" ]]; then
  if [[ -f "${ANCHOR_ZIP}" ]]; then
    PYTHONNOUSERSITE=1 \
    "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
      --base "${ANCHOR_ZIP}" \
      --candidate "${SUBMISSION_ZIP}" \
      --out "${COMPARE_JSON}"
  else
    echo "Skip compare: missing anchor zip ${ANCHOR_ZIP}" >&2
  fi
fi
