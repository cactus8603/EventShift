#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${EVENTSHIFT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

usage() {
  cat <<USAGE
Usage: bash scripts/rebuild_04111.sh --test-root /path/to/test [options]

Rebuild the 0.4111 b75 submission from bundled configs, checkpoints, and artifacts.

Required:
  --test-root PATH          CoSEC test root containing sequence/img_co_left folders.

Common options:
  --out-root PATH           Output root. Defaults to outputs/rebuild_04111_b75_from_checkpoints_<timestamp>.
  --conda PATH              Conda executable. Defaults to CONDA or conda.
  --m2f-env NAME            Conda env for Mask2Former exports. Defaults to M2F_ENV or mask2former.
  --mmseg-env NAME          Conda env for MMSeg/SegFormer export. Defaults to MMSEG_ENV or mmseg.
  --device DEVICE           Torch device. Defaults to DEVICE or cuda:0.
  --smoke-limit N           Export only the first N frames per model and stop before composition.
  --skip-inference          Reuse existing raw masks under --out-root.
  --run-inference           Run model inference. This is the default.
  --deterministic           Enable deterministic torch settings. This is the default.
  --non-deterministic       Disable deterministic torch settings.
  -h, --help                Show this help.

Environment-variable fallbacks remain supported for the same names:
  TEST_ROOT, OUT_ROOT, CONDA, M2F_ENV, MMSEG_ENV, DEVICE, SMOKE_LIMIT,
  RUN_INFERENCE, EVENTSHIFT_DETERMINISTIC.
USAGE
}

need_value() {
  local flag="$1"
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "${flag} requires a value." >&2
    exit 2
  fi
}

CONDA="${CONDA:-conda}"
M2F_ENV="${M2F_ENV:-mask2former}"
MMSEG_ENV="${MMSEG_ENV:-mmseg}"
DEVICE="${DEVICE:-cuda:0}"
TEST_ROOT="${TEST_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
EVENTSHIFT_DETERMINISTIC="${EVENTSHIFT_DETERMINISTIC:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-root)
      need_value "$@"
      TEST_ROOT="$2"
      shift 2
      ;;
    --out-root)
      need_value "$@"
      OUT_ROOT="$2"
      shift 2
      ;;
    --conda)
      need_value "$@"
      CONDA="$2"
      shift 2
      ;;
    --m2f-env)
      need_value "$@"
      M2F_ENV="$2"
      shift 2
      ;;
    --mmseg-env)
      need_value "$@"
      MMSEG_ENV="$2"
      shift 2
      ;;
    --device)
      need_value "$@"
      DEVICE="$2"
      shift 2
      ;;
    --smoke-limit)
      need_value "$@"
      SMOKE_LIMIT="$2"
      shift 2
      ;;
    --skip-inference)
      RUN_INFERENCE=0
      shift
      ;;
    --run-inference)
      RUN_INFERENCE=1
      shift
      ;;
    --deterministic)
      EVENTSHIFT_DETERMINISTIC=1
      shift
      ;;
    --non-deterministic)
      EVENTSHIFT_DETERMINISTIC=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "Unexpected positional arguments: $*" >&2
  usage >&2
  exit 2
fi

export EVENTSHIFT_DETERMINISTIC
if [[ "${EVENTSHIFT_DETERMINISTIC}" != "0" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

if [[ -z "${TEST_ROOT}" ]]; then
  echo "--test-root is required. Example:" >&2
  echo "  bash scripts/rebuild_04111.sh --test-root /path/to/test" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${BUNDLE_DIR}/outputs/rebuild_04111_b75_from_checkpoints_${STAMP}}"
RAW_DIR="${OUT_ROOT}/prediction_dirs/from_checkpoints_raw"
COMPOSED_DIR="${OUT_ROOT}/composed"
ZIP_DIR="${OUT_ROOT}/submission_zips"
REPORT_DIR="${OUT_ROOT}/reports"
EXTRACT_DIR="${OUT_ROOT}/extracted_artifacts"

EXPORT_DIR="${BUNDLE_DIR}/tools/export"
POSTPROCESS_DIR="${BUNDLE_DIR}/tools/postprocess"
CONFIG_DIR="${BUNDLE_DIR}/configs"
CKPT_DIR="${BUNDLE_DIR}/checkpoints"
ARTIFACT_ZIP_DIR="${BUNDLE_DIR}/artifacts/submission_zips"

M2F_EXPORT="${EXPORT_DIR}/export_mask2former_submission.py"
MMSEG_EXPORT="${EXPORT_DIR}/export_mmseg_submission.py"
FILTER_SCRIPT="${POSTPROCESS_DIR}/filter_submission_delta_by_transition.py"
COMPOSE_SCRIPT="${POSTPROCESS_DIR}/compose_domain_submission.py"

M2F_EVENT_DAY_CFG="${CONFIG_DIR}/mask2former/Mask2Former_SwinL_FullCoSEC_FromDayBest_Floor816070_LR5e-7.yaml"
M2F_EVENT_DAY_WEIGHTS="${CKPT_DIR}/m2f_event_full_cosec_from_day_best_floor816070_lr5e-7.pth"
M2F_FULL_NIGHT_CFG="${CONFIG_DIR}/mask2former/Mask2Former_SwinL_FullDSEC_CoSEC_ACDC_UnifiedClassCover.yaml"
M2F_FULL_NIGHT_WEIGHTS="${CKPT_DIR}/m2f_full_desc_selected_cosec_night.pth"
SEG_EVENT_NIGHT_CFG="${CONFIG_DIR}/segformer/SegFormer_B5_FullCoSEC_FromNightBest_Floor546453.py"
SEG_EVENT_NIGHT_WEIGHTS="${CKPT_DIR}/segformer_b5_event_full_cosec_from_night_best_floor546453_lr1e-6_iter4500.pth"

ANCHOR_04075_ZIP="${ARTIFACT_ZIP_DIR}/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip"
NIGHT_VALPAIR_ZIP="${ARTIFACT_ZIP_DIR}/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629.zip"
REALGATE_ZIP="${ARTIFACT_ZIP_DIR}/sub_04085realvote_eventday_valpairnight_p50n100c500_realgate60a5000_20260629.zip"
EXPECTED_FINAL_ZIP="${ARTIFACT_ZIP_DIR}/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip"

EXPECTED_FINAL_SHA="4c369c3d3ce554618366a0db66189f5b92cf7ffe64ebc28ac251374d56bda46b"

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

EVENTSEG_P70_ALLOW_PAIRS="building->road,wall->road,wall->fence,fence->road,fence->wall,pole->rider,traffic light->vegetation,vegetation->road,vegetation->sidewalk,vegetation->building,vegetation->wall,terrain->sidewalk,sky->building,sky->vegetation"

M2F_TTA_MIN_SIZES="${M2F_TTA_MIN_SIZES:-[512,624,768,1024]}"
M2F_TTA_MAX_SIZE="${M2F_TTA_MAX_SIZE:-1600}"
M2F_TTA_FLIP="${M2F_TTA_FLIP:-True}"
MMSEG_TTA_SCALE_SPECS="${MMSEG_TTA_SCALE_SPECS:-s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600}"
MMSEG_TTA_SCALE_SET="${MMSEG_TTA_SCALE_SET:-s512+s624+s768+s1024}"
MMSEG_TTA_FLIP="${MMSEG_TTA_FLIP:-1}"

limit_args=()
if [[ -n "${SMOKE_LIMIT}" ]]; then
  limit_args=(--limit "${SMOKE_LIMIT}")
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
  local total=0
  local seq
  for seq in "$@"; do
    total=$((total + $(find "${TEST_ROOT}/${seq}/img_co_left" -maxdepth 1 -type f -name '*.png' | wc -l)))
  done
  echo "${total}"
}

run_bundle_python() {
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${BUNDLE_DIR}:${BUNDLE_DIR}/tools:${BUNDLE_DIR}/third_party/Mask2Former:${BUNDLE_DIR}/third_party/detectron2:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n "${M2F_ENV}" python "$@"
}

run_m2f_export() {
  local name="$1"
  local config="$2"
  local weights="$3"
  local expected="$4"
  shift 4
  local out_dir="${RAW_DIR}/${name}"
  local current
  current="$(count_pngs "${out_dir}")"
  if [[ "${current}" -eq "${expected}" ]]; then
    echo "[skip] ${name}: ${current}/${expected}"
    return 0
  fi
  echo "[export] ${name}: ${current}/${expected} -> ${out_dir}"
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${BUNDLE_DIR}:${BUNDLE_DIR}/third_party/Mask2Former:${BUNDLE_DIR}/third_party/detectron2:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n "${M2F_ENV}" python "${M2F_EXPORT}" \
    --config-file "${config}" \
    --weights "${weights}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE}" \
    --progress-desc "${name}" \
    --sequences "$@" \
    --skip-existing \
    "${limit_args[@]}" \
    -- \
    TEST.AUG.ENABLED True \
    TEST.AUG.MIN_SIZES "${M2F_TTA_MIN_SIZES}" \
    TEST.AUG.MAX_SIZE "${M2F_TTA_MAX_SIZE}" \
    TEST.AUG.FLIP "${M2F_TTA_FLIP}" \
    INPUT.MIN_SIZE_TEST 624 \
    INPUT.MAX_SIZE_TEST 1200
}

run_mmseg_export() {
  local name="$1"
  local config="$2"
  local weights="$3"
  local expected="$4"
  shift 4
  local out_dir="${RAW_DIR}/${name}"
  local current
  current="$(count_pngs "${out_dir}")"
  if [[ "${current}" -eq "${expected}" ]]; then
    echo "[skip] ${name}: ${current}/${expected}"
    return 0
  fi
  echo "[export] ${name}: ${current}/${expected} -> ${out_dir}"
  local flip_args=()
  if [[ "${MMSEG_TTA_FLIP}" == "1" ]]; then
    flip_args=(--flip)
  fi
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${BUNDLE_DIR}:${BUNDLE_DIR}/tools:${BUNDLE_DIR}/third_party/mmsegmentation:${PYTHONPATH:-}" \
  "${CONDA}" run --no-capture-output -n "${MMSEG_ENV}" python "${MMSEG_EXPORT}" \
    --config-file "${config}" \
    --checkpoint "${weights}" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE}" \
    --progress-desc "${name}" \
    --sequences "$@" \
    --skip-existing \
    "${limit_args[@]}" \
    --scale-specs "${MMSEG_TTA_SCALE_SPECS}" \
    --scale-set "${MMSEG_TTA_SCALE_SET}" \
    "${flip_args[@]}"
}

extract_zip() {
  local zip_path="$1"
  local out_dir="$2"
  if [[ -d "${out_dir}" ]] && [[ "$(count_pngs "${out_dir}")" -gt 0 ]]; then
    return 0
  fi
  mkdir -p "${out_dir}"
  python - "${zip_path}" "${out_dir}" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
print(f"extracted {zip_path} -> {out_dir}")
PY
}

verify_sha() {
  local label="$1"
  local zip_path="$2"
  local expected_sha="$3"
  local actual_sha
  actual_sha="$(sha256sum "${zip_path}" | awk '{print $1}')"
  echo "[sha] ${label}: ${actual_sha}"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "WARNING: ${label} SHA mismatch. Expected ${expected_sha}" >&2
    return 1
  fi
  return 0
}

main() {
  mkdir -p "${RAW_DIR}" "${COMPOSED_DIR}" "${ZIP_DIR}" "${REPORT_DIR}" "${EXTRACT_DIR}"

  check_path "${TEST_ROOT}"
  check_path "${CONDA}"
  check_path "${M2F_EXPORT}"
  check_path "${MMSEG_EXPORT}"
  check_path "${FILTER_SCRIPT}"
  check_path "${COMPOSE_SCRIPT}"
  check_path "${M2F_EVENT_DAY_CFG}"
  check_path "${M2F_EVENT_DAY_WEIGHTS}"
  check_path "${M2F_FULL_NIGHT_CFG}"
  check_path "${M2F_FULL_NIGHT_WEIGHTS}"
  check_path "${SEG_EVENT_NIGHT_CFG}"
  check_path "${SEG_EVENT_NIGHT_WEIGHTS}"
  check_path "${ANCHOR_04075_ZIP}"
  check_path "${NIGHT_VALPAIR_ZIP}"
  check_path "${REALGATE_ZIP}"
  check_path "${EXPECTED_FINAL_ZIP}"

  local day_count night_count
  day_count="$(expected_count "${DAY_SEQUENCES[@]}")"
  night_count="$(expected_count "${NIGHT_SEQUENCES[@]}")"

  echo "BUNDLE_DIR=${BUNDLE_DIR}"
  echo "TEST_ROOT=${TEST_ROOT}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "DEVICE=${DEVICE}"
  echo "expected counts: day=${day_count} night=${night_count}"

  if [[ "${RUN_INFERENCE}" == "1" ]]; then
    run_m2f_export "mask2former_day_event" "${M2F_EVENT_DAY_CFG}" "${M2F_EVENT_DAY_WEIGHTS}" "${day_count}" "${DAY_SEQUENCES[@]}"
    run_m2f_export "mask2former_night_full_desc" "${M2F_FULL_NIGHT_CFG}" "${M2F_FULL_NIGHT_WEIGHTS}" "${night_count}" "${NIGHT_SEQUENCES[@]}"
    run_mmseg_export "segformer_night_event" "${SEG_EVENT_NIGHT_CFG}" "${SEG_EVENT_NIGHT_WEIGHTS}" "${night_count}" "${NIGHT_SEQUENCES[@]}"
  else
    echo "[skip] RUN_INFERENCE=0; using existing masks under ${RAW_DIR}"
  fi

  if [[ -n "${SMOKE_LIMIT}" ]]; then
    echo "SMOKE_LIMIT=${SMOKE_LIMIT}; stopping after export smoke test."
    exit 0
  fi

  extract_zip "${ANCHOR_04075_ZIP}" "${EXTRACT_DIR}/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628"
  extract_zip "${NIGHT_VALPAIR_ZIP}" "${EXTRACT_DIR}/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629"
  extract_zip "${REALGATE_ZIP}" "${EXTRACT_DIR}/sub_04085realvote_eventday_valpairnight_p50n100c500_realgate60a5000_20260629"

  run_bundle_python "${FILTER_SCRIPT}" \
    --base "${RAW_DIR}/mask2former_night_full_desc" \
    --candidate "${RAW_DIR}/segformer_night_event" \
    --out-dir "${COMPOSED_DIR}/full_desc_m2fnight_eventseg_valrepairpairs_p70n100c500_b5comp60a5000_20260629" \
    --zip "${ZIP_DIR}/full_desc_m2fnight_eventseg_valrepairpairs_p70n100c500_b5comp60a5000_20260629.zip" \
    --summary "${REPORT_DIR}/full_desc_m2fnight_eventseg_valrepairpairs_p70n100c500_b5comp60a5000_summary_20260629.json" \
    --domains Night \
    --allow-pairs "${EVENTSEG_P70_ALLOW_PAIRS}" \
    --component-min-boundary5-rate 0.60 \
    --component-max-area 5000 \
    --overwrite

  run_bundle_python "${COMPOSE_SCRIPT}" \
    --day-dir "${RAW_DIR}/mask2former_day_event" \
    --night-dir "${EXTRACT_DIR}/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629" \
    --real-dir "${EXTRACT_DIR}/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${COMPOSED_DIR}/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629" \
    --zip "${ZIP_DIR}/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip" \
    --overwrite

  run_bundle_python "${COMPOSE_SCRIPT}" \
    --day-dir "${RAW_DIR}/mask2former_day_event" \
    --night-dir "${COMPOSED_DIR}/full_desc_m2fnight_eventseg_valrepairpairs_p70n100c500_b5comp60a5000_20260629" \
    --real-dir "${EXTRACT_DIR}/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${COMPOSED_DIR}/sub_eventday_eventsegnight_p70n100c500_b5comp60a5000_keepreal_20260629" \
    --zip "${ZIP_DIR}/sub_eventday_eventsegnight_p70n100c500_b5comp60a5000_keepreal_20260629.zip" \
    --overwrite

  run_bundle_python "${FILTER_SCRIPT}" \
    --base "${ZIP_DIR}/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip" \
    --candidate "${ZIP_DIR}/sub_eventday_eventsegnight_p70n100c500_b5comp60a5000_keepreal_20260629.zip" \
    --out-dir "${COMPOSED_DIR}/sub_pipeline_main_plus_eventsegnight_p70_b75a2000_keepreal_20260629" \
    --zip "${ZIP_DIR}/sub_pipeline_main_plus_eventsegnight_p70_b75a2000_keepreal_20260629.zip" \
    --summary "${REPORT_DIR}/sub_pipeline_main_plus_eventsegnight_p70_b75a2000_keepreal_summary_20260629.json" \
    --domains Night \
    --allow-pairs "${EVENTSEG_P70_ALLOW_PAIRS}" \
    --component-min-boundary5-rate 0.75 \
    --component-max-area 2000 \
    --overwrite

  run_bundle_python "${COMPOSE_SCRIPT}" \
    --day-dir "${COMPOSED_DIR}/sub_pipeline_main_plus_eventsegnight_p70_b75a2000_keepreal_20260629" \
    --night-dir "${COMPOSED_DIR}/sub_pipeline_main_plus_eventsegnight_p70_b75a2000_keepreal_20260629" \
    --real-dir "${EXTRACT_DIR}/sub_04085realvote_eventday_valpairnight_p50n100c500_realgate60a5000_20260629" \
    --test-root "${TEST_ROOT}" \
    --out-dir "${COMPOSED_DIR}/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629" \
    --zip "${ZIP_DIR}/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip" \
    --overwrite

  python -m zipfile -t "${ZIP_DIR}/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip"
  verify_sha "final_04111" "${ZIP_DIR}/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip" "${EXPECTED_FINAL_SHA}"
  if cmp -s "${ZIP_DIR}/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip" "${EXPECTED_FINAL_ZIP}"; then
    echo "byte_compare=identical_to_bundle_authoritative_final_zip"
  else
    echo "WARNING: final zip differs byte-for-byte from bundle authoritative final zip." >&2
    exit 1
  fi
  echo "OK: rebuilt 0.4111 b75 anchor from bundle checkpoints and bundle artifacts."
  echo "final_zip=${ZIP_DIR}/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip"
}

main "$@"
