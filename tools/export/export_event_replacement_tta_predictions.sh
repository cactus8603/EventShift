#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1621738/ebmv_eccv/eccv_segment}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
MMSEG_ROOT="${MMSEG_ROOT:-/work/u1621738/ebmv_eccv/mmsegmentation}"
MAMBASEG_ROOT="${MAMBASEG_ROOT:-/work/u1621738/ebmv_eccv/MambaSeg}"
CONDA="${CONDA:-/home/u1621738/miniconda3/bin/conda}"
TEST_ROOT="${TEST_ROOT:-${MAMBASEG_ROOT}/data/test}"
DEVICE="${DEVICE:-cuda:0}"

OUT_ROOT="${OUT_ROOT:-${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs/event_replacements_tta4flip_20260628_raw}"

RUN_M2F_DAY="${RUN_M2F_DAY:-1}"
RUN_SEG_NIGHT="${RUN_SEG_NIGHT:-1}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"

M2F_DAY_CFG="${M2F_DAY_CFG:-${SWIN_L_ROOT}/work_dirs/swinL_full_cosec_from_day_best_floor816070_lr5e-7_savefix_20260628_225858/config.yaml}"
M2F_DAY_WEIGHTS="${M2F_DAY_WEIGHTS:-${SWIN_L_ROOT}/work_dirs/swinL_full_cosec_from_day_best_floor816070_lr5e-7_savefix_20260628_225858/best_model_cosec_day.pth}"

SEG_NIGHT_CFG="${SEG_NIGHT_CFG:-${SWIN_L_ROOT}/work_dirs/mmseg/segformer_b5_full_cosec_from_night_best_floor546453_lr1e-6/SegFormer_B5_FullCoSEC_FromNightBest_Floor546453.py}"
SEG_NIGHT_WEIGHTS="${SEG_NIGHT_WEIGHTS:-${SWIN_L_ROOT}/work_dirs/mmseg/segformer_b5_full_cosec_from_night_best_floor546453_lr1e-6/best_night_mIoU_iter_4500.pth}"

M2F_TTA_MIN_SIZES="${M2F_TTA_MIN_SIZES:-[512,624,768,1024]}"
M2F_TTA_MAX_SIZE="${M2F_TTA_MAX_SIZE:-1600}"
M2F_TTA_FLIP="${M2F_TTA_FLIP:-True}"
MMSEG_TTA_SCALE_SPECS="${MMSEG_TTA_SCALE_SPECS:-s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600}"
MMSEG_TTA_SCALE_SET="${MMSEG_TTA_SCALE_SET:-s512+s624+s768+s1024}"
MMSEG_TTA_FLIP="${MMSEG_TTA_FLIP:-1}"

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

limit_args=()
if [[ -n "${SMOKE_LIMIT}" ]]; then
  limit_args=(--limit "${SMOKE_LIMIT}")
fi

m2f_tta_opts=(
  --
  TEST.AUG.ENABLED True
  TEST.AUG.MIN_SIZES "${M2F_TTA_MIN_SIZES}"
  TEST.AUG.MAX_SIZE "${M2F_TTA_MAX_SIZE}"
  TEST.AUG.FLIP "${M2F_TTA_FLIP}"
  INPUT.MIN_SIZE_TEST 624
  INPUT.MAX_SIZE_TEST 1200
)

mmseg_tta_args=(
  --scale-specs "${MMSEG_TTA_SCALE_SPECS}"
  --scale-set "${MMSEG_TTA_SCALE_SET}"
)
if [[ "${MMSEG_TTA_FLIP}" == "1" ]]; then
  mmseg_tta_args+=(--flip)
fi

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

run_m2f_day() {
  local out_dir="${OUT_ROOT}/mask2former_day_event"
  local expected="$1"
  local current
  current="$(count_pngs "${out_dir}")"
  if [[ "${current}" -eq "${expected}" ]]; then
    echo "[skip] mask2former_day_event: ${current}/${expected}"
    return 0
  fi

  echo "[export] mask2former_day_event: ${current}/${expected} -> ${out_dir}"
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${SWIN_L_ROOT}:${SWIN_L_ROOT}/tools:${SWIN_L_ROOT}/third_party/Mask2Former:${SWIN_L_ROOT}/third_party/detectron2:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n mask2former python "${SWIN_L_ROOT}/tools/export_mask2former_submission.py" \
    --config-file "${M2F_DAY_CFG}" \
    --weights "${M2F_DAY_WEIGHTS}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE}" \
    --sequences "${DAY_SEQUENCES[@]}" \
    --skip-existing \
    "${limit_args[@]}" \
    "${m2f_tta_opts[@]}"

  current="$(count_pngs "${out_dir}")"
  if [[ "${current}" -ne "${expected}" ]]; then
    echo "Incomplete mask2former_day_event: ${current}/${expected}" >&2
    exit 1
  fi
}

run_seg_night() {
  local out_dir="${OUT_ROOT}/segformer_night_event"
  local expected="$1"
  local current
  current="$(count_pngs "${out_dir}")"
  if [[ "${current}" -eq "${expected}" ]]; then
    echo "[skip] segformer_night_event: ${current}/${expected}"
    return 0
  fi

  echo "[export] segformer_night_event: ${current}/${expected} -> ${out_dir}"
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${SWIN_L_ROOT}:${SWIN_L_ROOT}/tools:${MMSEG_ROOT}:${MAMBASEG_ROOT}:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n mmseg python "${SWIN_L_ROOT}/tools/export_mmseg_submission.py" \
    --config-file "${SEG_NIGHT_CFG}" \
    --checkpoint "${SEG_NIGHT_WEIGHTS}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE}" \
    --sequences "${NIGHT_SEQUENCES[@]}" \
    --skip-existing \
    "${limit_args[@]}" \
    "${mmseg_tta_args[@]}"

  current="$(count_pngs "${out_dir}")"
  if [[ "${current}" -ne "${expected}" ]]; then
    echo "Incomplete segformer_night_event: ${current}/${expected}" >&2
    exit 1
  fi
}

main() {
  check_path "${TEST_ROOT}"
  check_path "${M2F_DAY_CFG}"
  check_path "${M2F_DAY_WEIGHTS}"
  check_path "${SEG_NIGHT_CFG}"
  check_path "${SEG_NIGHT_WEIGHTS}"
  check_path "${SWIN_L_ROOT}/tools/export_mask2former_submission.py"
  check_path "${SWIN_L_ROOT}/tools/export_mmseg_submission.py"
  mkdir -p "${OUT_ROOT}"

  local day_count night_count
  day_count="$(expected_count "${DAY_SEQUENCES[@]}")"
  night_count="$(expected_count "${NIGHT_SEQUENCES[@]}")"

  echo "DEVICE=${DEVICE}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "expected counts: day=${day_count} night=${night_count}"
  echo "m2f_day_weights=$(readlink -f "${M2F_DAY_WEIGHTS}")"
  echo "seg_night_weights=$(readlink -f "${SEG_NIGHT_WEIGHTS}")"

  if [[ "${RUN_M2F_DAY}" == "1" ]]; then
    run_m2f_day "${day_count}"
  fi
  if [[ "${RUN_SEG_NIGHT}" == "1" ]]; then
    run_seg_night "${night_count}"
  fi
}

main "$@"
