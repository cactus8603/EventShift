#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1621738/ebmv_eccv/eccv_segment}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
MASKDINO_ROOT="${MASKDINO_ROOT:-${ROOT}/maskdino_swinl}"
MAMBASEG_ROOT="${MAMBASEG_ROOT:-/work/u1621738/ebmv_eccv/MambaSeg}"
CONDA="${CONDA:-/home/u1621738/miniconda3/bin/conda}"
TEST_ROOT="${TEST_ROOT:-${MAMBASEG_ROOT}/data/test}"
DEVICE="${DEVICE:-cuda:0}"

FULL_CKPT_ROOT="${FULL_CKPT_ROOT:-${ROOT}/unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc}"
FULL_MASKDINO_RAW="${FULL_MASKDINO_RAW:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/full_desc_maskdino_tta4flip_20260628_raw}"
KFOLD_MASKDINO_RAW="${KFOLD_MASKDINO_RAW:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/kfold3_maskdino_tta4flip_20260628_raw}"

RUN_FULLDESC_MASKDINO="${RUN_FULLDESC_MASKDINO:-1}"
RUN_KFOLD_MASKDINO="${RUN_KFOLD_MASKDINO:-1}"
KFOLD_MASKDINO_FOLDS="${KFOLD_MASKDINO_FOLDS:-0 1 2}"
CHECK_EXISTING_SWING_SEG="${CHECK_EXISTING_SWING_SEG:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

MASKDINO_TTA_MIN_SIZES=(${MASKDINO_TTA_MIN_SIZES:-512 624 768 1024})
MASKDINO_TTA_MAX_SIZE="${MASKDINO_TTA_MAX_SIZE:-1600}"
MASKDINO_TTA_FLIP="${MASKDINO_TTA_FLIP:-true}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"

FULL_MASKDINO_CFG="${FULL_MASKDINO_CFG:-${MASKDINO_ROOT}/configs/cosec/semantic-segmentation/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1.yaml}"
FULL_MASKDINO_DAY_WEIGHTS="${FULL_MASKDINO_DAY_WEIGHTS:-${FULL_CKPT_ROOT}/maskdino/step1/best_model_cosec_day.pth}"
FULL_MASKDINO_NIGHT_WEIGHTS="${FULL_MASKDINO_NIGHT_WEIGHTS:-${FULL_CKPT_ROOT}/maskdino/step1/best_model_cosec_night.pth}"
FULL_MASKDINO_REAL_WEIGHTS="${FULL_MASKDINO_REAL_WEIGHTS:-${FULL_CKPT_ROOT}/maskdino/step1/best_model_acdc_all.pth}"

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

count_pngs() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -type f -name '*.png' | wc -l
  else
    echo 0
  fi
}

expected_count() {
  local -a seqs=("$@")
  local total=0
  local seq
  for seq in "${seqs[@]}"; do
    total=$((total + $(find "${TEST_ROOT}/${seq}/img_co_left" -maxdepth 1 -type f -name '*.png' | wc -l)))
  done
  echo "${total}"
}

run_maskdino_export() {
  local cfg="$1"
  local weights="$2"
  local out_dir="$3"
  shift 3

  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${MASKDINO_ROOT}:${MASKDINO_ROOT}/tools:${SWIN_L_ROOT}/tools:${MAMBASEG_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n mask2former python "${MASKDINO_ROOT}/tools/export_maskdino_submission.py" \
    --config-file "${cfg}" \
    --weights "${weights}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE}" \
    --sequences "$@" \
    --skip-existing \
    "${limit_args[@]}" \
    "${maskdino_tta_args[@]}"
}

export_maskdino_domain() {
  local label="$1"
  local cfg="$2"
  local weights="$3"
  local out_dir="$4"
  local expected="$5"
  shift 5

  local current
  current="$(count_pngs "${out_dir}")"
  if [[ "${current}" -eq "${expected}" ]]; then
    echo "[skip] ${label}: complete (${current}/${expected})"
    return 0
  fi

  echo "[export] ${label}: ${current}/${expected} -> ${out_dir}"
  mkdir -p "$(dirname "${out_dir}")"
  run_maskdino_export "${cfg}" "${weights}" "${out_dir}" "$@"

  current="$(count_pngs "${out_dir}")"
  if [[ "${current}" -ne "${expected}" ]]; then
    echo "Incomplete output for ${label}: ${current}/${expected}" >&2
    exit 1
  fi
  echo "[done] ${label}: ${current}/${expected}"
}

check_existing_dir() {
  local label="$1"
  local dir="$2"
  local expected="$3"
  local current
  current="$(count_pngs "${dir}")"
  if [[ "${current}" -eq "${expected}" ]]; then
    echo "[ok] ${label}: ${current}/${expected}"
  else
    echo "[warn] ${label}: ${current}/${expected} (${dir})" >&2
  fi
}

check_inputs() {
  check_path "${TEST_ROOT}"
  check_path "${MASKDINO_ROOT}/tools/export_maskdino_submission.py"
  check_path "${FULL_MASKDINO_CFG}"
  check_path "${FULL_MASKDINO_DAY_WEIGHTS}"
  check_path "${FULL_MASKDINO_NIGHT_WEIGHTS}"
  check_path "${FULL_MASKDINO_REAL_WEIGHTS}"
  local fold
  for fold in 0 1 2; do
    check_path "${MASKDINO_ROOT}/work_dirs/maskdino_swinl_cosec_dsec_acdc_safe_kfold3_fold${fold}_bs1/config.yaml"
    check_path "${MASKDINO_ROOT}/work_dirs/maskdino_swinl_cosec_dsec_acdc_safe_kfold3_fold${fold}_bs1/best_model_cosec_day.pth"
    check_path "${MASKDINO_ROOT}/work_dirs/maskdino_swinl_cosec_dsec_acdc_safe_kfold3_fold${fold}_bs1/best_model_cosec_night.pth"
    check_path "${MASKDINO_ROOT}/work_dirs/maskdino_swinl_cosec_dsec_acdc_safe_kfold3_fold${fold}_bs1/best_model_acdc_all.pth"
  done
}

main() {
  check_inputs

  local day_count night_count real_count
  day_count="$(expected_count "${DAY_SEQUENCES[@]}")"
  night_count="$(expected_count "${NIGHT_SEQUENCES[@]}")"
  real_count="$(expected_count "${REAL_SEQUENCES[@]}")"

  echo "DEVICE=${DEVICE}"
  echo "TEST_ROOT=${TEST_ROOT}"
  echo "TTA maskdino min=${MASKDINO_TTA_MIN_SIZES[*]} max=${MASKDINO_TTA_MAX_SIZE} flip=${MASKDINO_TTA_FLIP}"
  echo "expected counts: day=${day_count} night=${night_count} real=${real_count}"

  if [[ "${CHECK_EXISTING_SWING_SEG}" == "1" ]]; then
    local full_raw="${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw"
    local kfold_raw="${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/kfold3_swinl_segformer_tta4flip_hardvote_20260627_raw"
    check_existing_dir "full mask2former day" "${full_raw}/mask2former_day" "${day_count}"
    check_existing_dir "full mask2former night" "${full_raw}/mask2former_night" "${night_count}"
    check_existing_dir "full mask2former real" "${full_raw}/mask2former_real" "${real_count}"
    check_existing_dir "full segformer day" "${full_raw}/segformer_day" "${day_count}"
    check_existing_dir "full segformer night" "${full_raw}/segformer_night" "${night_count}"
    check_existing_dir "full segformer real" "${full_raw}/segformer_real" "${real_count}"
    local fold
    for fold in 0 1 2; do
      check_existing_dir "kfold swinl fold${fold} day" "${kfold_raw}/swinl_fold${fold}_day" "${day_count}"
      check_existing_dir "kfold swinl fold${fold} night" "${kfold_raw}/swinl_fold${fold}_night" "${night_count}"
      check_existing_dir "kfold segformer fold${fold} day" "${kfold_raw}/segformer_fold${fold}_day" "${day_count}"
      check_existing_dir "kfold segformer fold${fold} night" "${kfold_raw}/segformer_fold${fold}_night" "${night_count}"
    done
  fi

  if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "CHECK_ONLY=1; export not started."
    echo "FULL_MASKDINO_RAW=${FULL_MASKDINO_RAW}"
    echo "KFOLD_MASKDINO_RAW=${KFOLD_MASKDINO_RAW}"
    return 0
  fi

  if [[ "${RUN_FULLDESC_MASKDINO}" == "1" ]]; then
    export_maskdino_domain "full maskdino day" "${FULL_MASKDINO_CFG}" "${FULL_MASKDINO_DAY_WEIGHTS}" "${FULL_MASKDINO_RAW}/maskdino_day" "${day_count}" "${DAY_SEQUENCES[@]}"
    export_maskdino_domain "full maskdino night" "${FULL_MASKDINO_CFG}" "${FULL_MASKDINO_NIGHT_WEIGHTS}" "${FULL_MASKDINO_RAW}/maskdino_night" "${night_count}" "${NIGHT_SEQUENCES[@]}"
    export_maskdino_domain "full maskdino real" "${FULL_MASKDINO_CFG}" "${FULL_MASKDINO_REAL_WEIGHTS}" "${FULL_MASKDINO_RAW}/maskdino_real" "${real_count}" "${REAL_SEQUENCES[@]}"
  fi

  if [[ "${RUN_KFOLD_MASKDINO}" == "1" ]]; then
    local fold fold_dir cfg
    local -a kfold_folds
    read -r -a kfold_folds <<< "${KFOLD_MASKDINO_FOLDS}"
    for fold in "${kfold_folds[@]}"; do
      fold_dir="${MASKDINO_ROOT}/work_dirs/maskdino_swinl_cosec_dsec_acdc_safe_kfold3_fold${fold}_bs1"
      cfg="${fold_dir}/config.yaml"
      export_maskdino_domain "kfold maskdino fold${fold} day" "${cfg}" "${fold_dir}/best_model_cosec_day.pth" "${KFOLD_MASKDINO_RAW}/maskdino_fold${fold}_day" "${day_count}" "${DAY_SEQUENCES[@]}"
      export_maskdino_domain "kfold maskdino fold${fold} night" "${cfg}" "${fold_dir}/best_model_cosec_night.pth" "${KFOLD_MASKDINO_RAW}/maskdino_fold${fold}_night" "${night_count}" "${NIGHT_SEQUENCES[@]}"
      export_maskdino_domain "kfold maskdino fold${fold} real" "${cfg}" "${fold_dir}/best_model_acdc_all.pth" "${KFOLD_MASKDINO_RAW}/maskdino_fold${fold}_real" "${real_count}" "${REAL_SEQUENCES[@]}"
    done
  fi

  echo "FULL_MASKDINO_RAW=${FULL_MASKDINO_RAW}"
  echo "KFOLD_MASKDINO_RAW=${KFOLD_MASKDINO_RAW}"
}

main "$@"
