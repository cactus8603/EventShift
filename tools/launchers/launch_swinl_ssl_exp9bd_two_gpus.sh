#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_EXP9B:=0}"
: "${GPU_EXP9D:=1}"

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

CUDA_VISIBLE_DEVICES="${GPU_EXP9B}" CONDA_ENV="${CONDA_ENV}" \
  nohup bash tools/run_swinl_ssl_currentbest_tta_segformer_agree_headonly.sh \
  > "${log_dir}/ssl-exp9b_${stamp}.log" 2>&1 &
pid_exp9b="$!"

CUDA_VISIBLE_DEVICES="${GPU_EXP9D}" CONDA_ENV="${CONDA_ENV}" \
  nohup bash tools/run_swinl_ssl_currentbest_tta_segformer_agree_rare_boundary_headonly.sh \
  > "${log_dir}/ssl-exp9d_${stamp}.log" 2>&1 &
pid_exp9d="$!"

echo "Launched ssl-exp9b on GPU ${GPU_EXP9B}: pid=${pid_exp9b}, log=${log_dir}/ssl-exp9b_${stamp}.log"
echo "Launched ssl-exp9d on GPU ${GPU_EXP9D}: pid=${pid_exp9d}, log=${log_dir}/ssl-exp9d_${stamp}.log"
