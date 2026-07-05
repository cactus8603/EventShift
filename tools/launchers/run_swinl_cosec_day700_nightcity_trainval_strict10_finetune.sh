#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
conda run --no-capture-output -n "${CONDA_ENV:-mask2former}" \
  python tools/train_mask2former_cosec.py \
  --config-file configs/Mask2Former_SwinL_CoSEC_Day700_NightCity_TrainValStrict10_Finetune.yaml \
  --num-gpus 1 \
  "$@"
