#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CUDA_VISIBLE_DEVICES:=0}"

conda run --no-capture-output -n "${CONDA_ENV:-mask2former}" \
  python tools/train_mask2former_cosec.py \
  --config-file configs/Mask2Former_SwinL_CoSEC_SSL_CurrentBestTTA_All_HeadOnly.yaml \
  --num-gpus 1 \
  "$@"
