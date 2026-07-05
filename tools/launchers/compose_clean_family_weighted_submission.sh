#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
CONDA="${CONDA:-conda}"
TEST_ROOT="${TEST_ROOT:-./data/test}"

RUN_TAG="${RUN_TAG:-clean_fulldesc_kfold3_weighted_tta_20260628}"
FULLDESC_RAW="${FULLDESC_RAW:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw}"
FULLDESC_MASKDINO_RAW="${FULLDESC_MASKDINO_RAW:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/full_desc_maskdino_tta4flip_20260628_raw}"
KFOLD_RAW="${KFOLD_RAW:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/kfold3_swinl_segformer_tta4flip_hardvote_20260627_raw}"
KFOLD_MASKDINO_RAW="${KFOLD_MASKDINO_RAW:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/kfold3_maskdino_tta4flip_20260628_raw}"
EVENT_REPLACEMENT_RAW="${EVENT_REPLACEMENT_RAW:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/event_replacements_tta4flip_20260628_raw}"
USE_EVENT_FULL_REPLACEMENTS="${USE_EVENT_FULL_REPLACEMENTS:-1}"

default_m2f_day_dir="${FULLDESC_RAW}/mask2former_day"
default_seg_night_dir="${FULLDESC_RAW}/segformer_night"
if [[ "${USE_EVENT_FULL_REPLACEMENTS}" == "1" ]]; then
  default_m2f_day_dir="${EVENT_REPLACEMENT_RAW}/mask2former_day_event"
  default_seg_night_dir="${EVENT_REPLACEMENT_RAW}/segformer_night_event"
fi

FULL_M2F_DAY_DIR="${FULL_M2F_DAY_DIR:-${default_m2f_day_dir}}"
FULL_M2F_NIGHT_DIR="${FULL_M2F_NIGHT_DIR:-${FULLDESC_RAW}/mask2former_night}"
FULL_M2F_REAL_DIR="${FULL_M2F_REAL_DIR:-${FULLDESC_RAW}/mask2former_real}"
FULL_SEG_DAY_DIR="${FULL_SEG_DAY_DIR:-${FULLDESC_RAW}/segformer_day}"
FULL_SEG_NIGHT_DIR="${FULL_SEG_NIGHT_DIR:-${default_seg_night_dir}}"
FULL_SEG_REAL_DIR="${FULL_SEG_REAL_DIR:-${FULLDESC_RAW}/segformer_real}"
FULL_DINO_DAY_DIR="${FULL_DINO_DAY_DIR:-${FULLDESC_MASKDINO_RAW}/maskdino_day}"
FULL_DINO_NIGHT_DIR="${FULL_DINO_NIGHT_DIR:-${FULLDESC_MASKDINO_RAW}/maskdino_night}"
FULL_DINO_REAL_DIR="${FULL_DINO_REAL_DIR:-${FULLDESC_MASKDINO_RAW}/maskdino_real}"

DAY_OUT="${DAY_OUT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_day}"
NIGHT_OUT="${NIGHT_OUT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_night}"
REAL_OUT="${REAL_OUT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_real}"
COMPOSED_DIR="${COMPOSED_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/composed/sub_${RUN_TAG}}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips/sub_${RUN_TAG}.zip}"

ALLOW_MISSING="${ALLOW_MISSING:-0}"
OVERWRITE="${OVERWRITE:-1}"
INCLUDE_FULLDESC_MASKDINO="${INCLUDE_FULLDESC_MASKDINO:-1}"
INCLUDE_KFOLD_MASKDINO="${INCLUDE_KFOLD_MASKDINO:-1}"
INCLUDE_REAL_KFOLD_MASKDINO="${INCLUDE_REAL_KFOLD_MASKDINO:-1}"

FD_M2F_W="${FD_M2F_W:-1.25}"
FD_SEG_W="${FD_SEG_W:-1.00}"
FD_DINO_W="${FD_DINO_W:-0.70}"
KFOLD_SWINL_FOLD_W="${KFOLD_SWINL_FOLD_W:-0.38}"
KFOLD_SEG_FOLD_W="${KFOLD_SEG_FOLD_W:-0.32}"
KFOLD_DINO_FOLD_W="${KFOLD_DINO_FOLD_W:-0.24}"

REAL_FD_M2F_W="${REAL_FD_M2F_W:-1.25}"
REAL_FD_SEG_W="${REAL_FD_SEG_W:-1.00}"
REAL_FD_DINO_W="${REAL_FD_DINO_W:-0.70}"
REAL_KFOLD_DINO_FOLD_W="${REAL_KFOLD_DINO_FOLD_W:-0.24}"

DAY_SEQUENCES=(Day_Campus_012 Day_Park_011 Day_Suburbs_015 Day_Suburbs_017 Day_Village_009)
NIGHT_SEQUENCES=(Night_Campus_010 Night_City_009 Night_Park_009)
REAL_SEQUENCES=(
  REAL_000005 REAL_000009 REAL_000010 REAL_000011 REAL_000012 REAL_000019
  REAL_000020 REAL_000021 REAL_000023 REAL_000024 REAL_000025 REAL_000026
  REAL_000029 REAL_000030 REAL_000031 REAL_000032 REAL_000039 REAL_000040
)

compose_weighted() {
  local out_dir="$1"
  shift
  local -a sequences=()
  while [[ "$#" -gt 0 && "$1" != "--" ]]; do
    sequences+=("$1")
    shift
  done
  shift
  local -a names=()
  local -a dirs=()
  local -a weights=()
  while [[ "$#" -gt 0 ]]; do
    names+=("$1")
    dirs+=("$2")
    weights+=("$3")
    shift 3
  done

  local -a args=()
  local skipped=0
  for idx in "${!dirs[@]}"; do
    if [[ -d "${dirs[$idx]}" ]]; then
      args+=(--name "${names[$idx]}" --input-dir "${dirs[$idx]}" --weight "${weights[$idx]}")
    elif [[ "${ALLOW_MISSING}" == "1" ]]; then
      echo "Skip missing prediction dir: ${names[$idx]} -> ${dirs[$idx]}" >&2
      skipped=$((skipped + 1))
    else
      echo "Missing prediction dir: ${names[$idx]} -> ${dirs[$idx]}" >&2
      echo "Set ALLOW_MISSING=1 to skip it, or export the missing model predictions first." >&2
      exit 1
    fi
  done
  if [[ "${#args[@]}" -lt 12 ]]; then
    echo "Not enough active inputs for weighted vote. skipped=${skipped}" >&2
    exit 1
  fi

  local -a overwrite_args=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi

  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_weighted_vote_submission.py" \
    "${args[@]}" \
    --tie-dir "${dirs[0]}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${out_dir}" \
    --sequences "${sequences[@]}" \
    "${overwrite_args[@]}"
}

day_inputs=(
  fd_m2f_day "${FULL_M2F_DAY_DIR}" "${FD_M2F_W}"
  fd_seg_day "${FULL_SEG_DAY_DIR}" "${FD_SEG_W}"
)
night_inputs=(
  fd_m2f_night "${FULL_M2F_NIGHT_DIR}" "${FD_M2F_W}"
  fd_seg_night "${FULL_SEG_NIGHT_DIR}" "${FD_SEG_W}"
)
real_inputs=(
  fd_m2f_real "${FULL_M2F_REAL_DIR}" "${REAL_FD_M2F_W}"
  fd_seg_real "${FULL_SEG_REAL_DIR}" "${REAL_FD_SEG_W}"
)

if [[ "${INCLUDE_FULLDESC_MASKDINO}" == "1" ]]; then
  day_inputs+=(fd_dino_day "${FULL_DINO_DAY_DIR}" "${FD_DINO_W}")
  night_inputs+=(fd_dino_night "${FULL_DINO_NIGHT_DIR}" "${FD_DINO_W}")
  real_inputs+=(fd_dino_real "${FULL_DINO_REAL_DIR}" "${REAL_FD_DINO_W}")
fi

for fold in 0 1 2; do
  day_inputs+=("kfold_swinl${fold}_day" "${KFOLD_RAW}/swinl_fold${fold}_day" "${KFOLD_SWINL_FOLD_W}")
  day_inputs+=("kfold_seg${fold}_day" "${KFOLD_RAW}/segformer_fold${fold}_day" "${KFOLD_SEG_FOLD_W}")
  night_inputs+=("kfold_swinl${fold}_night" "${KFOLD_RAW}/swinl_fold${fold}_night" "${KFOLD_SWINL_FOLD_W}")
  night_inputs+=("kfold_seg${fold}_night" "${KFOLD_RAW}/segformer_fold${fold}_night" "${KFOLD_SEG_FOLD_W}")
  if [[ "${INCLUDE_KFOLD_MASKDINO}" == "1" ]]; then
    day_inputs+=("kfold_dino${fold}_day" "${KFOLD_MASKDINO_RAW}/maskdino_fold${fold}_day" "${KFOLD_DINO_FOLD_W}")
    night_inputs+=("kfold_dino${fold}_night" "${KFOLD_MASKDINO_RAW}/maskdino_fold${fold}_night" "${KFOLD_DINO_FOLD_W}")
  fi
  if [[ "${INCLUDE_REAL_KFOLD_MASKDINO}" == "1" ]]; then
    real_inputs+=("kfold_dino${fold}_real" "${KFOLD_MASKDINO_RAW}/maskdino_fold${fold}_real" "${REAL_KFOLD_DINO_FOLD_W}")
  fi
done

compose_weighted "${DAY_OUT}" "${DAY_SEQUENCES[@]}" -- "${day_inputs[@]}"
compose_weighted "${NIGHT_OUT}" "${NIGHT_SEQUENCES[@]}" -- "${night_inputs[@]}"
compose_weighted "${REAL_OUT}" "${REAL_SEQUENCES[@]}" -- "${real_inputs[@]}"

compose_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  compose_args=(--overwrite)
fi

PYTHONNOUSERSITE=1 \
"${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
  --day-dir "${DAY_OUT}" \
  --night-dir "${NIGHT_OUT}" \
  --real-dir "${REAL_OUT}" \
  --test-root "${TEST_ROOT}" \
  --out-dir "${COMPOSED_DIR}" \
  --zip "${SUBMISSION_ZIP}" \
  "${compose_args[@]}"

echo "Wrote clean weighted-family submission: ${SUBMISSION_ZIP}"
