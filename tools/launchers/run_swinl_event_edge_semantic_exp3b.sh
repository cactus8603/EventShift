#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python tools/train_mask2former_cosec.py \
  --config-file configs/Mask2Former_SwinL_CoSEC_EventEdgeSemantic_Exp3b.yaml \
  --num-gpus 1 \
  "$@"
