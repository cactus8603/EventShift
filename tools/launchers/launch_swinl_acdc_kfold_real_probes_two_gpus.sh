#!/usr/bin/env bash
set -euo pipefail

ROOT="."
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
: "${CONDA_ENV:=mask2former}"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for persistent detached launches." >&2
  exit 1
fi

for path in \
  configs/Mask2Former_SwinL_ACDC_KFold3_HeadOnly.yaml \
  tools/run_swinl_acdc_kfold_headonly.sh \
  work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_night.pth
do
  if [ ! -e "${path}" ]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

stamp="$(date +%Y%m%d_%H%M%S)"
log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"

run_probe() {
  local name="$1"
  local gpu="$2"
  local condition="$3"
  local fold="$4"
  local log_path="${log_dir}/${name}_${stamp}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" screen -dmS "${name}" bash -lc \
    "cd '${ROOT}' && export CONDA_ENV='${CONDA_ENV}' && export PYTHONNOUSERSITE=1 && export PYTHONUNBUFFERED=1 && bash tools/run_swinl_acdc_kfold_headonly.sh ${condition} ${fold} > '${log_path}' 2>&1"
  echo "${name}: GPU ${gpu}, ${condition} fold${fold}, log ${log_path}"
}

run_probe "swinl_acdc_night_kfold0" "${GPU_NIGHT:-0}" "night" "${FOLD_NIGHT:-0}"
run_probe "swinl_acdc_all_kfold0" "${GPU_ALL:-1}" "all" "${FOLD_ALL:-0}"
