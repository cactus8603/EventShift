#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/work/u1621738/ebmv_eccv/eccv_segment}"
SWIN_L_ROOT="${SWIN_L_ROOT:-${ROOT}/swin_l}"
CONDA="${CONDA:-/home/u1621738/miniconda3/bin/conda}"
TEST_ROOT="${TEST_ROOT:-/work/u1621738/ebmv_eccv/MambaSeg/data/test}"
DEVICE="${DEVICE:-cuda:0}"

TAG="${TAG:-ckpt_rebuild_recommended_b85_eventseg_plus_realgate60a5000_$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-1}"

# 1 = re-export event Day/Night predictions from checkpoints.
# 0 = use EVENT_OUT_ROOT as an already-exported raw prediction root.
RUN_EVENT_EXPORT="${RUN_EVENT_EXPORT:-1}"

# Rebuild the 0.4085 realvote anchor from its constituent raw prediction dirs.
# This does not by itself re-export all old kfold/full-desc/HRNet raw predictions;
# use the corresponding export scripts first if those caches must be regenerated too.
RUN_REBUILD_04085="${RUN_REBUILD_04085:-1}"

RUN_COMPARE="${RUN_COMPARE:-1}"

PRED_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/prediction_dirs"
COMPOSED_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/composed"
ZIP_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/submission_zips"
REPORT_ROOT="${SWIN_L_ROOT}/work_dirs/submissions/reports"
DIFF_ROOT="${SWIN_L_ROOT}/work_dirs/diagnostics/submission_diffs"

EVENT_OUT_ROOT="${EVENT_OUT_ROOT:-${PRED_ROOT}/${TAG}_event_from_ckpt_raw}"
EVENT_DAY_DIR="${EVENT_OUT_ROOT}/mask2former_day_event"
EVENT_NIGHT_DIR="${EVENT_OUT_ROOT}/segformer_night_event"

FULLDESC_RAW="${FULLDESC_RAW:-${PRED_ROOT}/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw}"
MAXUP_SUPPORTED_DIR="${MAXUP_SUPPORTED_DIR:-${PRED_ROOT}/sub_04085realvote_maxup_maskdino2supported_20260629}"

ANCHOR_TAG="${TAG}_04085realvote_anchor"
ANCHOR_COMPOSED_DIR="${ANCHOR_COMPOSED_DIR:-${COMPOSED_ROOT}/${ANCHOR_TAG}}"
ANCHOR_ZIP="${ANCHOR_ZIP:-${ZIP_ROOT}/${ANCHOR_TAG}.zip}"

REFERENCE_04110="${REFERENCE_04110:-${ZIP_ROOT}/sub_pipeline_b85_eventseg_plus_realgate60a5000_20260629.zip}"
REFERENCE_04085="${REFERENCE_04085:-${ZIP_ROOT}/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip}"

NIGHT_RSWBVEG_DIR="${PRED_ROOT}/${TAG}_nightrswbveg_b5comp60a5000"
NIGHT_RSWBVEG_COMPOSED="${COMPOSED_ROOT}/${TAG}_eventday_nightrswbveg_keepreal"
NIGHT_RSWBVEG_ZIP="${ZIP_ROOT}/${TAG}_eventday_nightrswbveg_keepreal.zip"

NIGHT_VALPAIR_DIR="${COMPOSED_ROOT}/${TAG}_night_valpair_p50_b5comp60a5000"
NIGHT_VALPAIR_ZIP="${ZIP_ROOT}/${TAG}_night_valpair_p50_b5comp60a5000.zip"
NIGHT_VALPAIR_SUMMARY="${REPORT_ROOT}/${TAG}_night_valpair_p50_b5comp60a5000_summary.json"

PIPELINE_BASE_DIR="${COMPOSED_ROOT}/${TAG}_pipeline_base_eventday_valpairnight_keepreal"
PIPELINE_BASE_ZIP="${ZIP_ROOT}/${TAG}_pipeline_base_eventday_valpairnight_keepreal.zip"

EVENTSEG_NIGHT_DIR="${COMPOSED_ROOT}/${TAG}_eventseg_night_p70_b5comp60a5000"
EVENTSEG_NIGHT_ZIP="${ZIP_ROOT}/${TAG}_eventseg_night_p70_b5comp60a5000.zip"
EVENTSEG_NIGHT_SUMMARY="${REPORT_ROOT}/${TAG}_eventseg_night_p70_b5comp60a5000_summary.json"

EVENTSEG_SUB_DIR="${COMPOSED_ROOT}/${TAG}_eventday_eventsegnight_p70_keepreal"
EVENTSEG_SUB_ZIP="${ZIP_ROOT}/${TAG}_eventday_eventsegnight_p70_keepreal.zip"

B85_KEEPREAL_DIR="${COMPOSED_ROOT}/${TAG}_b85_eventseg_keepreal"
B85_KEEPREAL_ZIP="${ZIP_ROOT}/${TAG}_b85_eventseg_keepreal.zip"
B85_KEEPREAL_SUMMARY="${REPORT_ROOT}/${TAG}_b85_eventseg_keepreal_summary.json"

REAL_GATE_DIR="${COMPOSED_ROOT}/${TAG}_realgate60a5000"

FINAL_DIR="${COMPOSED_ROOT}/${TAG}"
FINAL_ZIP="${ZIP_ROOT}/${TAG}.zip"

NIGHT_P50_ALLOW_PAIRS="${NIGHT_P50_ALLOW_PAIRS:-${SWIN_L_ROOT}/work_dirs/diagnostics/score_free_repair_gate/full_desc_night_segformer_vs_m2f_pairs_b5comp60a5000_p50_n100_c500_20260629.txt}"
EVENT_NIGHT_P70_ALLOW_PAIRS="${EVENT_NIGHT_P70_ALLOW_PAIRS:-${SWIN_L_ROOT}/work_dirs/diagnostics/score_free_repair_gate/event_segformer_night_fullcosec_vs_m2f_pairs_b5comp60a5000_p70_n100_c500_20260629.txt}"

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

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

check_prediction_count() {
  local label="$1"
  local dir="$2"
  local expected="$3"
  local count
  count="$(find "${dir}" -type f -name '*.png' 2>/dev/null | wc -l)"
  if [[ "${count}" -ne "${expected}" ]]; then
    echo "Incomplete ${label}: ${count}/${expected} (${dir})" >&2
    exit 1
  fi
  echo "[ok] ${label}: ${count}/${expected}"
}

run_event_export() {
  RUN_M2F_DAY=1 RUN_SEG_NIGHT=1 DEVICE="${DEVICE}" OUT_ROOT="${EVENT_OUT_ROOT}" \
    bash "${SWIN_L_ROOT}/tools/export_event_replacement_tta_predictions.sh"
}

rebuild_04085_anchor() {
  TAG="${ANCHOR_TAG}" \
  FINAL_COMPOSED_DIR="${ANCHOR_COMPOSED_DIR}" \
  FINAL_ZIP="${ANCHOR_ZIP}" \
  OVERWRITE="${OVERWRITE}" \
  RUN_BASE=1 RUN_REQUIRED=1 RUN_COMPOSE=1 RUN_COMPARE=1 \
    bash "${SWIN_L_ROOT}/tools/rebuild_04085_realvote_submission.sh"
}

merge_nightrswbveg() {
  run_py "${SWIN_L_ROOT}/tools/merge_class_boundary_gated_from_prediction_dir.py" \
    --anchor-dir "${ANCHOR_COMPOSED_DIR}" \
    --candidate-dir "${FULLDESC_RAW}/segformer_night" \
    --out-dir "${NIGHT_RSWBVEG_DIR}" \
    --class-id 0 --class-id 1 --class-id 2 --class-id 3 --class-id 8 \
    --prefix Night_ \
    --boundary-radius 5 \
    --boundary-source either \
    --gate-mode component \
    --component-min-boundary-rate 0.60 \
    --component-max-area 5000 \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${EVENT_DAY_DIR}" \
    --night-dir "${NIGHT_RSWBVEG_DIR}" \
    --real-dir "${ANCHOR_COMPOSED_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${NIGHT_RSWBVEG_COMPOSED}" \
    --zip "${NIGHT_RSWBVEG_ZIP}" \
    "${write_args[@]}"
}

build_pipeline_base() {
  local allow_pairs
  allow_pairs="$(cat "${NIGHT_P50_ALLOW_PAIRS}")"

  run_py "${SWIN_L_ROOT}/tools/filter_submission_delta_by_transition.py" \
    --base "${ANCHOR_ZIP}" \
    --candidate "${NIGHT_RSWBVEG_ZIP}" \
    --out-dir "${NIGHT_VALPAIR_DIR}" \
    --zip "${NIGHT_VALPAIR_ZIP}" \
    --summary "${NIGHT_VALPAIR_SUMMARY}" \
    --domains Night \
    --allow-pairs "${allow_pairs}" \
    --component-min-boundary5-rate 0.60 \
    --component-max-area 5000 \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${EVENT_DAY_DIR}" \
    --night-dir "${NIGHT_VALPAIR_DIR}" \
    --real-dir "${ANCHOR_COMPOSED_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${PIPELINE_BASE_DIR}" \
    --zip "${PIPELINE_BASE_ZIP}" \
    "${write_args[@]}"
}

build_eventseg_candidate() {
  local allow_pairs
  allow_pairs="$(cat "${EVENT_NIGHT_P70_ALLOW_PAIRS}")"

  run_py "${SWIN_L_ROOT}/tools/filter_submission_delta_by_transition.py" \
    --base "${FULLDESC_RAW}/mask2former_night" \
    --candidate "${EVENT_NIGHT_DIR}" \
    --out-dir "${EVENTSEG_NIGHT_DIR}" \
    --zip "${EVENTSEG_NIGHT_ZIP}" \
    --summary "${EVENTSEG_NIGHT_SUMMARY}" \
    --domains Night \
    --allow-pairs "${allow_pairs}" \
    --component-min-boundary5-rate 0.60 \
    --component-max-area 5000 \
    "${write_args[@]}"

  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${EVENT_DAY_DIR}" \
    --night-dir "${EVENTSEG_NIGHT_DIR}" \
    --real-dir "${ANCHOR_COMPOSED_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${EVENTSEG_SUB_DIR}" \
    --zip "${EVENTSEG_SUB_ZIP}" \
    "${write_args[@]}"
}

build_b85_keepreal() {
  local allow_pairs
  allow_pairs="$(cat "${EVENT_NIGHT_P70_ALLOW_PAIRS}")"

  run_py "${SWIN_L_ROOT}/tools/filter_submission_delta_by_transition.py" \
    --base "${PIPELINE_BASE_ZIP}" \
    --candidate "${EVENTSEG_SUB_ZIP}" \
    --out-dir "${B85_KEEPREAL_DIR}" \
    --zip "${B85_KEEPREAL_ZIP}" \
    --summary "${B85_KEEPREAL_SUMMARY}" \
    --domains Night \
    --allow-pairs "${allow_pairs}" \
    --component-min-boundary5-rate 0.85 \
    --component-max-area 1000 \
    "${write_args[@]}"
}

build_real_gate() {
  run_py "${SWIN_L_ROOT}/tools/merge_class_boundary_gated_from_prediction_dir.py" \
    --anchor-dir "${ANCHOR_COMPOSED_DIR}" \
    --candidate-dir "${MAXUP_SUPPORTED_DIR}" \
    --out-dir "${REAL_GATE_DIR}" \
    --class-id 4 \
    --prefix REAL_ \
    --boundary-radius 5 \
    --boundary-source either \
    --gate-mode component \
    --component-min-boundary-rate 0.60 \
    --component-max-area 5000 \
    "${write_args[@]}"
}

compose_final() {
  run_py "${SWIN_L_ROOT}/tools/compose_domain_submission.py" \
    --day-dir "${B85_KEEPREAL_DIR}" \
    --night-dir "${B85_KEEPREAL_DIR}" \
    --real-dir "${REAL_GATE_DIR}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${FINAL_DIR}" \
    --zip "${FINAL_ZIP}" \
    "${write_args[@]}"
}

compare_final() {
  mkdir -p "${DIFF_ROOT}"
  if [[ -f "${REFERENCE_04110}" ]]; then
    run_py "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
      --base "${REFERENCE_04110}" \
      --candidate "${FINAL_ZIP}" \
      --out "${DIFF_ROOT}/${TAG}_vs_reference_04110.json"
  fi
  if [[ -f "${REFERENCE_04085}" ]]; then
    run_py "${SWIN_L_ROOT}/tools/compare_submission_zips.py" \
      --base "${REFERENCE_04085}" \
      --candidate "${FINAL_ZIP}" \
      --out "${DIFF_ROOT}/${TAG}_vs_reference_04085.json"
  fi
  python -m zipfile -t "${FINAL_ZIP}"
  sha256sum "${FINAL_ZIP}"
}

check_inputs() {
  require_path "${TEST_ROOT}"
  require_path "${SWIN_L_ROOT}/tools/export_event_replacement_tta_predictions.sh"
  require_path "${SWIN_L_ROOT}/tools/rebuild_04085_realvote_submission.sh"
  require_path "${SWIN_L_ROOT}/tools/merge_class_boundary_gated_from_prediction_dir.py"
  require_path "${SWIN_L_ROOT}/tools/filter_submission_delta_by_transition.py"
  require_path "${SWIN_L_ROOT}/tools/compose_domain_submission.py"
  require_path "${NIGHT_P50_ALLOW_PAIRS}"
  require_path "${EVENT_NIGHT_P70_ALLOW_PAIRS}"
  require_path "${FULLDESC_RAW}/segformer_night"
  require_path "${FULLDESC_RAW}/mask2former_night"
  require_path "${MAXUP_SUPPORTED_DIR}"
}

main() {
  cd "${SWIN_L_ROOT}"
  check_inputs

  echo "TAG=${TAG}"
  echo "DEVICE=${DEVICE}"
  echo "EVENT_OUT_ROOT=${EVENT_OUT_ROOT}"
  echo "ANCHOR_ZIP=${ANCHOR_ZIP}"
  echo "FINAL_ZIP=${FINAL_ZIP}"

  if [[ "${RUN_EVENT_EXPORT}" == "1" ]]; then
    run_event_export
  else
    echo "[skip] event checkpoint export; using EVENT_OUT_ROOT=${EVENT_OUT_ROOT}"
  fi
  check_prediction_count "event Mask2Former day" "${EVENT_DAY_DIR}" 574
  check_prediction_count "event SegFormer night" "${EVENT_NIGHT_DIR}" 306

  if [[ "${RUN_REBUILD_04085}" == "1" ]]; then
    rebuild_04085_anchor
  else
    echo "[skip] 0.4085 anchor rebuild; using ANCHOR_COMPOSED_DIR=${ANCHOR_COMPOSED_DIR}"
    require_path "${ANCHOR_COMPOSED_DIR}"
    require_path "${ANCHOR_ZIP}"
  fi

  merge_nightrswbveg
  build_pipeline_base
  build_eventseg_candidate
  build_b85_keepreal
  build_real_gate
  compose_final

  if [[ "${RUN_COMPARE}" == "1" ]]; then
    compare_final
  fi
}

main "$@"
