#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

: "${ENV_NAME:=mask2former}"
: "${OUT_TAG:?Set OUT_TAG, e.g. acdc_all_kfold3_tta_vote_anchor}"
: "${INPUT_TAGS:?Set INPUT_TAGS as space-separated prediction tags without trailing _real_only.}"

TEST_ROOT="${TEST_ROOT:-/work/u1621738/ebmv_eccv/MambaSeg/data/test}"
PRED_DIR="work_dirs/submissions/prediction_dirs"
COMPOSE_DIR="work_dirs/submissions/composed"
ZIP_DIR="work_dirs/submissions/submission_zips"
DIAG_DIR="work_dirs/diagnostics/submission_diffs"

DAY_DIR="${DAY_DIR:-work_dirs/submissions/prediction_dirs/swinL_day65_4352_tta5126247681024_day_raw}"
NIGHT_DIR="${NIGHT_DIR:-work_dirs/submissions/prediction_dirs/swinL_day65_4352_trainlearned_scale_night64_raw}"
TIE_DIR="${TIE_DIR:-work_dirs/submissions/prediction_dirs/acdc54_754_acdconly_tta5126247681024_real_only}"
ANCHOR_ZIP="${ANCHOR_ZIP:-work_dirs/submissions/submission_zips/sub_swinL_day65_4352_daytta_night_trainlearnedscale64_acdc54_754_tta_real.zip}"

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

input_args=()
for tag in ${INPUT_TAGS}; do
  dir="${PRED_DIR}/${tag}_real_only"
  if [ ! -d "${dir}" ]; then
    echo "Missing input prediction dir: ${dir}" >&2
    exit 1
  fi
  input_args+=(--input-dir "${dir}")
done

for path in "${DAY_DIR}" "${NIGHT_DIR}" "${TIE_DIR}" "${ANCHOR_ZIP}"; do
  if [ ! -e "${path}" ]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

real_out="${PRED_DIR}/${OUT_TAG}_real_only"
compose_out="${COMPOSE_DIR}/sub_swinL_day65_4352_daytta_night_trainlearnedscale64_${OUT_TAG}_real"
zip_path="${ZIP_DIR}/sub_swinL_day65_4352_daytta_night_trainlearnedscale64_${OUT_TAG}_real.zip"
diff_path="${DIAG_DIR}/anchor4050_vs_${OUT_TAG}_real.json"

conda run --no-capture-output -n "${ENV_NAME}" python tools/majority_vote_prediction_dirs.py \
  "${input_args[@]}" \
  --tie-dir "${TIE_DIR}" \
  --out-dir "${real_out}" \
  --sequences "${REAL_SEQUENCES[@]}" \
  --overwrite

conda run --no-capture-output -n "${ENV_NAME}" python tools/compose_domain_submission.py \
  --day-dir "${DAY_DIR}" \
  --night-dir "${NIGHT_DIR}" \
  --real-dir "${real_out}" \
  --test-root "${TEST_ROOT}" \
  --out-dir "${compose_out}" \
  --zip "${zip_path}" \
  --overwrite

mkdir -p "${DIAG_DIR}"
conda run --no-capture-output -n "${ENV_NAME}" python tools/compare_submission_zips.py \
  --base "${ANCHOR_ZIP}" \
  --candidate "${zip_path}" \
  --out "${diff_path}"

echo "[done] ${zip_path}"
echo "[diff] ${diff_path}"
