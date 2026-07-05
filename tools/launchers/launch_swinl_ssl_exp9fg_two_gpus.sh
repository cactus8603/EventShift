#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_EXP9F:=0}"
: "${GPU_EXP9G:=1}"

cuda_count="$(
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
)"

if [ "${cuda_count}" -lt 2 ]; then
  echo "Need 2 visible CUDA GPUs, but found ${cuda_count}." >&2
  exit 1
fi

log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="${GPU_EXP9F}" CONDA_ENV="${CONDA_ENV}" \
  nohup bash tools/run_swinl_ssl_currentbest_tta_day_gap_focus_headonly.sh \
  > "${log_dir}/ssl-exp9f_${stamp}.log" 2>&1 &
pid_exp9f="$!"

CUDA_VISIBLE_DEVICES="${GPU_EXP9G}" CONDA_ENV="${CONDA_ENV}" \
  nohup bash tools/run_swinl_ssl_currentbest_tta_night_gap_focus_headonly.sh \
  > "${log_dir}/ssl-exp9g_${stamp}.log" 2>&1 &
pid_exp9g="$!"

echo "Launched ssl-exp9f on GPU ${GPU_EXP9F}: pid=${pid_exp9f}, log=${log_dir}/ssl-exp9f_${stamp}.log"
echo "Launched ssl-exp9g on GPU ${GPU_EXP9G}: pid=${pid_exp9g}, log=${log_dir}/ssl-exp9g_${stamp}.log"
