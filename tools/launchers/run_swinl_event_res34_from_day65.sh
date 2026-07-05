#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run --no-capture-output -n mask2former \
  python tools/train_mask2former_cosec.py \
  --config-file configs/Mask2Former_SwinL_CoSEC_EventRes34_Epoch.yaml \
  --num-gpus 1 \
  "$@"
