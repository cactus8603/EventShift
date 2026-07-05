#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1621738/ebmv_eccv/eccv_segment}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
MASKDINO_ROOT="${MASKDINO_ROOT:-${ROOT}/maskdino_swinl}"
MAMBASEG_ROOT="${MAMBASEG_ROOT:-/work/u1621738/ebmv_eccv/MambaSeg}"
MMSEG_ROOT="${MMSEG_ROOT:-/work/u1621738/ebmv_eccv/mmsegmentation}"
CONDA="${CONDA:-/home/u1621738/miniconda3/bin/conda}"
TEST_ROOT="${TEST_ROOT:-${MAMBASEG_ROOT}/data/test}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc}"
SYNC_CHECKPOINTS="${SYNC_CHECKPOINTS:-1}"

RUN_TAG="${RUN_TAG:-full_desc_cosec_acdc_three_model_tta_$(date +%Y%m%d_%H%M%S)}"
PRED_ROOT="${PRED_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_raw}"
DAY_VOTE_DIR="${DAY_VOTE_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_day_min2_m2f_anchor}"
NIGHT_VOTE_DIR="${NIGHT_VOTE_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_night_min2_m2f_anchor}"
REAL_VOTE_DIR="${REAL_VOTE_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/${RUN_TAG}_real_min2_m2f_anchor}"
SUB_NAME="${SUB_NAME:-sub_${RUN_TAG}_min2_m2fanchor_domainweights}"
COMPOSED_DIR="${COMPOSED_DIR:-${SWIN_L_ROOT}/work_dirs/submissions/composed/${SUB_NAME}}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips/${SUB_NAME}.zip}"
ANCHOR_ZIP="${ANCHOR_ZIP:-${SWIN_L_ROOT}/work_dirs/submissions/submission_zips/sub_swinL_day65_4352_daytta_night_trainlearnedscale64_acdc54_754_tta_real.zip}"
COMPARE_JSON="${COMPARE_JSON:-${SWIN_L_ROOT}/work_dirs/diagnostics/submission_diffs/${SUB_NAME}_vs_anchor4050.json}"

DEVICE="${DEVICE:-cuda:0}"
OVERWRITE_EXPORTS="${OVERWRITE_EXPORTS:-0}"
OVERWRITE_VOTE="${OVERWRITE_VOTE:-1}"
OVERWRITE_COMPOSE="${OVERWRITE_COMPOSE:-1}"
RUN_EXPORTS="${RUN_EXPORTS:-1}"
RUN_VOTE="${RUN_VOTE:-1}"
RUN_COMPOSE="${RUN_COMPOSE:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"
MIN_VOTES="${MIN_VOTES:-2}"

M2F_TTA_MIN_SIZES="${M2F_TTA_MIN_SIZES:-[512,624,768,1024]}"
M2F_TTA_MAX_SIZE="${M2F_TTA_MAX_SIZE:-1600}"
M2F_TTA_FLIP="${M2F_TTA_FLIP:-True}"
MMSEG_TTA_SCALE_SPECS="${MMSEG_TTA_SCALE_SPECS:-s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600}"
MMSEG_TTA_SCALE_SET="${MMSEG_TTA_SCALE_SET:-s512+s624+s768+s1024}"
MMSEG_TTA_FLIP="${MMSEG_TTA_FLIP:-1}"
MASKDINO_TTA_MIN_SIZES=(${MASKDINO_TTA_MIN_SIZES:-512 624 768 1024})
MASKDINO_TTA_MAX_SIZE="${MASKDINO_TTA_MAX_SIZE:-1600}"
MASKDINO_TTA_FLIP="${MASKDINO_TTA_FLIP:-true}"

M2F_CFG="${M2F_CFG:-${SWIN_L_ROOT}/configs/Mask2Former_SwinL_FullDSEC_CoSEC_ACDC_UnifiedClassCover.yaml}"
SEGFORMER_CFG="${SEGFORMER_CFG:-${SWIN_L_ROOT}/configs/mmseg/SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified.py}"
MASKDINO_CFG="${MASKDINO_CFG:-${MASKDINO_ROOT}/configs/cosec/semantic-segmentation/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1.yaml}"

# Domain-specific weights:
# - CoSEC day uses day-best
# - CoSEC night uses night-best
# - REAL uses ACDC-best
M2F_DAY_WEIGHTS="${M2F_DAY_WEIGHTS:-${CKPT_ROOT}/mask2former/step2/best_model_cosec_day.pth}"
M2F_NIGHT_WEIGHTS="${M2F_NIGHT_WEIGHTS:-${CKPT_ROOT}/mask2former/step1/best_model_cosec_night.pth}"
M2F_REAL_WEIGHTS="${M2F_REAL_WEIGHTS:-${CKPT_ROOT}/mask2former/step2/best_model_acdc_all.pth}"

SEGFORMER_DAY_WEIGHTS="${SEGFORMER_DAY_WEIGHTS:-${CKPT_ROOT}/segformer/step1/best_day_mIoU.pth}"
SEGFORMER_NIGHT_WEIGHTS="${SEGFORMER_NIGHT_WEIGHTS:-${CKPT_ROOT}/segformer/step2/best_night_mIoU.pth}"
SEGFORMER_REAL_WEIGHTS="${SEGFORMER_REAL_WEIGHTS:-${CKPT_ROOT}/segformer/step2/best_acdc_mIoU.pth}"

MASKDINO_DAY_WEIGHTS="${MASKDINO_DAY_WEIGHTS:-${CKPT_ROOT}/maskdino/step1/best_model_cosec_day.pth}"
MASKDINO_NIGHT_WEIGHTS="${MASKDINO_NIGHT_WEIGHTS:-${CKPT_ROOT}/maskdino/step1/best_model_cosec_night.pth}"
MASKDINO_REAL_WEIGHTS="${MASKDINO_REAL_WEIGHTS:-${CKPT_ROOT}/maskdino/step1/best_model_acdc_all.pth}"

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

limit_args=()
if [[ -n "${SMOKE_LIMIT}" ]]; then
  limit_args=(--limit "${SMOKE_LIMIT}")
fi

export_write_args=(--skip-existing)
if [[ "${OVERWRITE_EXPORTS}" == "1" ]]; then
  export_write_args=(--overwrite)
fi

vote_write_args=()
if [[ "${OVERWRITE_VOTE}" == "1" ]]; then
  vote_write_args=(--overwrite)
fi

compose_write_args=()
if [[ "${OVERWRITE_COMPOSE}" == "1" ]]; then
  compose_write_args=(--overwrite)
fi

m2f_tta_opts=(
  --
  TEST.AUG.ENABLED True
  TEST.AUG.MIN_SIZES "${M2F_TTA_MIN_SIZES}"
  TEST.AUG.MAX_SIZE "${M2F_TTA_MAX_SIZE}"
  TEST.AUG.FLIP "${M2F_TTA_FLIP}"
)

mmseg_tta_args=(
  --scale-specs "${MMSEG_TTA_SCALE_SPECS}"
  --scale-set "${MMSEG_TTA_SCALE_SET}"
)
if [[ "${MMSEG_TTA_FLIP}" == "1" ]]; then
  mmseg_tta_args+=(--flip)
fi

maskdino_tta_args=(
  --tta
  --aug-min-sizes "${MASKDINO_TTA_MIN_SIZES[@]}"
  --aug-max-size "${MASKDINO_TTA_MAX_SIZE}"
  --aug-flip "${MASKDINO_TTA_FLIP}"
)

check_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

check_inputs() {
  local required=(
    "${TEST_ROOT}"
    "${M2F_CFG}"
    "${SEGFORMER_CFG}"
    "${MASKDINO_CFG}"
    "${SWIN_L_ROOT}/tools/export_mask2former_submission.py"
    "${SWIN_L_ROOT}/tools/export_mmseg_submission.py"
    "${MASKDINO_ROOT}/tools/export_maskdino_submission.py"
    "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py"
    "${SWIN_L_ROOT}/tools/compose_domain_submission.py"
    "${M2F_DAY_WEIGHTS}"
    "${M2F_NIGHT_WEIGHTS}"
    "${M2F_REAL_WEIGHTS}"
    "${SEGFORMER_DAY_WEIGHTS}"
    "${SEGFORMER_NIGHT_WEIGHTS}"
    "${SEGFORMER_REAL_WEIGHTS}"
    "${MASKDINO_DAY_WEIGHTS}"
    "${MASKDINO_NIGHT_WEIGHTS}"
    "${MASKDINO_REAL_WEIGHTS}"
  )
  local path
  for path in "${required[@]}"; do
    check_path "${path}"
  done
}

log_setup() {
  mkdir -p "${PRED_ROOT}" "$(dirname "${SUBMISSION_ZIP}")" "$(dirname "${COMPARE_JSON}")"
  echo "RUN_TAG=${RUN_TAG}"
  echo "DEVICE=${DEVICE}"
  echo "PRED_ROOT=${PRED_ROOT}"
  echo "SUBMISSION_ZIP=${SUBMISSION_ZIP}"
  echo "domain weights:"
  printf '  mask2former day=%s\n  mask2former night=%s\n  mask2former real=%s\n' \
    "$(readlink -f "${M2F_DAY_WEIGHTS}")" "$(readlink -f "${M2F_NIGHT_WEIGHTS}")" "$(readlink -f "${M2F_REAL_WEIGHTS}")"
  printf '  segformer day=%s\n  segformer night=%s\n  segformer real=%s\n' \
    "$(readlink -f "${SEGFORMER_DAY_WEIGHTS}")" "$(readlink -f "${SEGFORMER_NIGHT_WEIGHTS}")" "$(readlink -f "${SEGFORMER_REAL_WEIGHTS}")"
  printf '  maskdino day=%s\n  maskdino night=%s\n  maskdino real=%s\n' \
    "$(readlink -f "${MASKDINO_DAY_WEIGHTS}")" "$(readlink -f "${MASKDINO_NIGHT_WEIGHTS}")" "$(readlink -f "${MASKDINO_REAL_WEIGHTS}")"
  echo "TTA mask2former min=${M2F_TTA_MIN_SIZES} max=${M2F_TTA_MAX_SIZE} flip=${M2F_TTA_FLIP}"
  echo "TTA segformer scales=${MMSEG_TTA_SCALE_SET} flip=${MMSEG_TTA_FLIP}"
  echo "TTA maskdino min=${MASKDINO_TTA_MIN_SIZES[*]} max=${MASKDINO_TTA_MAX_SIZE} flip=${MASKDINO_TTA_FLIP}"
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
    "${export_write_args[@]}" \
    "${limit_args[@]}" \
    "${m2f_tta_opts[@]}"
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
    "${export_write_args[@]}" \
    "${limit_args[@]}" \
    "${mmseg_tta_args[@]}"
}

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
    "${export_write_args[@]}" \
    "${limit_args[@]}" \
    "${maskdino_tta_args[@]}"
}

run_exports() {
  run_m2f_export mask2former_day "${M2F_DAY_WEIGHTS}" "${DAY_SEQUENCES[@]}"
  run_m2f_export mask2former_night "${M2F_NIGHT_WEIGHTS}" "${NIGHT_SEQUENCES[@]}"
  run_m2f_export mask2former_real "${M2F_REAL_WEIGHTS}" "${REAL_SEQUENCES[@]}"

  run_segformer_export segformer_day "${SEGFORMER_DAY_WEIGHTS}" "${DAY_SEQUENCES[@]}"
  run_segformer_export segformer_night "${SEGFORMER_NIGHT_WEIGHTS}" "${NIGHT_SEQUENCES[@]}"
  run_segformer_export segformer_real "${SEGFORMER_REAL_WEIGHTS}" "${REAL_SEQUENCES[@]}"

  run_maskdino_export maskdino_day "${MASKDINO_DAY_WEIGHTS}" "${DAY_SEQUENCES[@]}"
  run_maskdino_export maskdino_night "${MASKDINO_NIGHT_WEIGHTS}" "${NIGHT_SEQUENCES[@]}"
  run_maskdino_export maskdino_real "${MASKDINO_REAL_WEIGHTS}" "${REAL_SEQUENCES[@]}"
}

run_domain_vote() {
  local out_dir="$1"
  local anchor_dir="$2"
  shift 2
  local sequences=("$@")

  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${PRED_ROOT}/mask2former_${out_dir}" \
      "${PRED_ROOT}/segformer_${out_dir}" \
      "${PRED_ROOT}/maskdino_${out_dir}" \
    --anchor-dir "${anchor_dir}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${!out_dir^^_VOTE_DIR}" \
    --sequences "${sequences[@]}" \
    --min-votes "${MIN_VOTES}" \
    "${vote_write_args[@]}"
}

run_vote() {
  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${PRED_ROOT}/mask2former_day" \
      "${PRED_ROOT}/segformer_day" \
      "${PRED_ROOT}/maskdino_day" \
    --anchor-dir "${PRED_ROOT}/mask2former_day" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${DAY_VOTE_DIR}" \
    --sequences "${DAY_SEQUENCES[@]}" \
    --min-votes "${MIN_VOTES}" \
    "${vote_write_args[@]}"

  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${PRED_ROOT}/mask2former_night" \
      "${PRED_ROOT}/segformer_night" \
      "${PRED_ROOT}/maskdino_night" \
    --anchor-dir "${PRED_ROOT}/mask2former_night" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${NIGHT_VOTE_DIR}" \
    --sequences "${NIGHT_SEQUENCES[@]}" \
    --min-votes "${MIN_VOTES}" \
    "${vote_write_args[@]}"

  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_hardvote_submission.py" \
    --candidate-dirs \
      "${PRED_ROOT}/mask2former_real" \
      "${PRED_ROOT}/segformer_real" \
      "${PRED_ROOT}/maskdino_real" \
    --anchor-dir "${PRED_ROOT}/mask2former_real" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${REAL_VOTE_DIR}" \
    --sequences "${REAL_SEQUENCES[@]}" \
    --min-votes "${MIN_VOTES}" \
    "${vote_write_args[@]}"
}

run_compose() {
  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${DAY_VOTE_DIR}" \
    --night-dir "${NIGHT_VOTE_DIR}" \
    --real-dir "${REAL_VOTE_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${COMPOSED_DIR}" \
    --zip "${SUBMISSION_ZIP}" \
    "${compose_write_args[@]}"
}

run_compare() {
  if [[ ! -f "${ANCHOR_ZIP}" ]]; then
    echo "Skip compare: missing anchor zip ${ANCHOR_ZIP}" >&2
    return 0
  fi
  PYTHONNOUSERSITE=1 \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
    --base "${ANCHOR_ZIP}" \
    --candidate "${SUBMISSION_ZIP}" \
    --out "${COMPARE_JSON}"
}

main() {
  cd "${SWIN_L_ROOT}"
  if [[ "${SYNC_CHECKPOINTS}" == "1" ]]; then
    bash "${SWIN_L_ROOT}/tools/sync_full_desc_cosec_acdc_best_checkpoints.sh"
  fi
  check_inputs
  log_setup
  if [[ "${RUN_EXPORTS}" == "1" ]]; then
    run_exports
  fi
  if [[ "${RUN_VOTE}" == "1" ]]; then
    run_vote
  fi
  if [[ "${RUN_COMPOSE}" == "1" ]]; then
    run_compose
  fi
  if [[ "${RUN_COMPARE}" == "1" ]]; then
    run_compare
  fi
}

main "$@"
