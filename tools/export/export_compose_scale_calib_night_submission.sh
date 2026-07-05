#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

: "${CONDA_ENV:=mask2former}"
: "${DEVICE:=cuda:0}"
: "${TAG:=exp30B}"
: "${CALIBRATOR:=work_dirs/diagnostics/scale_calib_exp30B_night_classroute_eventsoft_full/accept_reject_calibrator.pth}"
: "${DAY_DIR:=work_dirs/submissions/prediction_dirs/swinL_day65_4352_tta5126247681024_day_raw}"
: "${REAL_DIR:=work_dirs/submissions/prediction_dirs/acdc54_754_acdconly_tta5126247681024_real_only}"
: "${OUT_ROOT:=work_dirs/submissions/prediction_dirs}"
: "${ZIP_ROOT:=work_dirs/submissions/submission_zips}"

if [ "${CANDIDATE_ALL:-0}" = "1" ]; then
  threshold_slug="candidate_all"
  threshold_args=(--candidate-all)
else
  if [ -z "${THRESHOLD:-}" ]; then
    echo "Set THRESHOLD, for example: THRESHOLD=0.02 TAG=exp30B bash $0" >&2
    exit 1
  fi
  threshold_slug="t${THRESHOLD//./p}"
  threshold_args=(--threshold "${THRESHOLD}")
fi

if [ ! -f "${CALIBRATOR}" ]; then
  echo "Missing calibrator: ${CALIBRATOR}" >&2
  exit 1
fi

night_dir="${OUT_ROOT}/swinL_day65_4352_${TAG}_${threshold_slug}_night"
compose_dir="${OUT_ROOT}/swinL_day65_4352_daytta_${TAG}_${threshold_slug}_night_acdc54_754_tta_real"
zip_path="${ZIP_ROOT}/sub_swinL_day65_4352_daytta_${TAG}_${threshold_slug}_night_acdc54_754_tta_real.zip"

conda run --no-capture-output -n "${CONDA_ENV}" \
  python tools/export_scale_accept_reject_calibrator.py \
    --calibrator "${CALIBRATOR}" \
    --test-root data/test \
    --sequences Night_Campus_010 Night_City_009 Night_Park_009 \
    "${threshold_args[@]}" \
    --out-dir "${night_dir}" \
    --overwrite \
    --device "${DEVICE}"

python tools/compose_domain_submission.py \
  --day-dir "${DAY_DIR}" \
  --night-dir "${night_dir}" \
  --real-dir "${REAL_DIR}" \
  --test-root data/test \
  --out-dir "${compose_dir}" \
  --zip "${zip_path}" \
  --overwrite

echo "night_dir=${night_dir}"
echo "compose_dir=${compose_dir}"
echo "zip=${zip_path}"
