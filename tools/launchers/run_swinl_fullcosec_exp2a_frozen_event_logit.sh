#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="${1:-configs/Mask2Former_SwinL_CoSEC_FullCoSEC_Exp2A_FrozenRGB_EventLogitCorrection.yaml}"
shift || true

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run --no-capture-output -n "${CONDA_ENV:-mask2former}" \
  python tools/train_mask2former_cosec.py \
  --config-file "${CONFIG_FILE}" \
  --num-gpus 1 \
  "$@"
